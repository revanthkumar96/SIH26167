"""Turn benchmark **train** splits into the Stage B adaptation corpus.

Stage A trained on BigEarthNet alone and the result was a model that answers in
BigEarthNet's idiom -- captions grew from 66.8 to 79.9 words against a 47.4-word
reference and CIDEr-D fell. That is not a training bug; it is what training on
one corpus and scoring on another produces. Stage B fixes it by adding the train
splits of the benchmarks themselves, which this module builds.

Three properties matter more than throughput here:

*Parity.* Prompts come from ``build_prompt`` -- the same function the eval
harness and the serving path call -- so a record's human turn is byte-identical
to the prompt the model meets at inference. A converter that writes its own
phrasing trains the model on a prompt shape that never occurs again.

*Safety.* Every record is checked against the benchmark test fingerprints before
it is written. The check is on by default and fatal by default, because a corpus
that leaks test images produces a better-looking number that means nothing, and
there is no later stage that would catch it.

*Balance.* RSVQA's train split runs to hundreds of thousands of templated yes/no
rows; VRSBench's captions number in the thousands. Concatenated raw, the corpus
is RSVQA with a rounding error of everything else, and the caption regression
Stage B exists to repair goes unrepaired. Per-source caps are therefore part of
the conversion, not an afterthought for the training script.
"""

from __future__ import annotations

import json
import os
import random
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from satquery.data.bigearthnet import Record, to_milli_box
from satquery.data.contamination import (
    ContaminationError,
    Fingerprints,
    image_key,
    question_key,
    read_corpus,
    record_overlap,
)
from satquery.data.evidence import Measurements, choose_evidence
from satquery.eval.prompts import build_prompt
from satquery.schema import Sample, Task

#: How a source with more rows than its cap is reduced. Seeded, so a rebuilt
#: corpus is the same corpus.
DEFAULT_SEED = 1234

#: What each train split may contribute, keyed by the config `name` in
#: configs/train/. Sized against what each source actually holds, and against
#: what Stage A measured rather than by even division:
#:
#:   caption    20,264 available, capped at 12,000. The regression Stage B
#:              exists to repair, so it gets the largest share of a source that
#:              is small to begin with.
#:   referring  36,287 available, capped at 10,000. Grounding scored 0.000 both
#:              before and after Stage A and the parser fix proved that is real
#:              localisation failure, so it needs volume, not a token presence.
#:   vqa        85,813 available, capped at 12,000. Already the strongest column
#:              (+0.1825); more of it buys the least.
#:   rsvqa_lr   hundreds of thousands available, capped at 10,000. Templated
#:              yes/no and counting. Uncapped it is the corpus, and the caption
#:              regression goes unrepaired -- which is the whole point of Stage B.
#:   cdvqa      65,967 available, capped at 8,000, and only if the guard clears
#:              it: train and test may share SECOND tiles.
#:
#: Roughly 52,000 records, about two hours per epoch at Stage A's measured 7.6
#: samples/s. Override per run; do not remove.
DEFAULT_CAPS: dict[str, int] = {
    "vrsbench_train_caption": 12_000,
    "vrsbench_train_vqa": 12_000,
    "vrsbench_train_referring": 10_000,
    "rsvqa_lr_train": 10_000,
    "cdvqa_train": 8_000,
    # The rehearsal slice, included via --include rather than converted from a
    # config. Sized as rehearsal, not training: Stage B resumes from Stage A's
    # adapter, so cross-modal ability is already in the weights and this is here
    # to stop it being trained away.
    "bigearthnet": 20_000,
    # Bench-config names too, so pointing this at a bench-named config for a
    # dry run does not silently produce an uncapped source.
    "vrsbench_caption": 12_000,
    "vrsbench_vqa": 12_000,
    "vrsbench_referring": 10_000,
    "rsvqa_lr": 10_000,
    "rsvqa_hr": 10_000,
    "cdvqa": 8_000,
}

#: Tasks dropped from an included rehearsal slice by default.
#:
#: The slice is there to preserve cross-modal ability and evidence-preamble
#: familiarity, and BigEarthNet's binary/mcq records carry both. Its captioning
#: records carry neither and actively fight the benchmark: median 96 words
#: against VRSBench's 52 and a 47.4-word reference. Stage B included them and
#: produced 112-word captions that overran the 128-token budget mid-sentence,
#: taking CIDEr-D from a 0.128 base to 0.006. Dropping them keeps what the slice
#: is for and removes what it costs.
DEFAULT_SLICE_DROP_TASKS = frozenset({"captioning"})

#: A measurement hook: given a sample, return whatever the specialists can
#: actually measure on it. Benchmark images are RGB, so NDWI and NDBI -- which
#: need NIR and SWIR -- are unavailable and the honest return is an empty
#: ``Measurements``. Once the land-cover CNN is trained it can fill
#: ``landcover_classes`` here, and these preambles become real.
Measurer = Callable[[Sample], Measurements]


class ConversionError(RuntimeError):
    """A source could not be converted at all."""


def no_measurements(_: Sample) -> Measurements:
    """The default hook. Nothing measurable on a three-channel JPEG."""
    return Measurements()


def format_box_answer(bbox: Sequence[float]) -> str:
    """Gold answer for a grounding record, on the grid the prompt asks for.

    ``eval/prompts.py`` demands "integer coordinates normalised to a 0-1000
    grid". Writing unit floats here would train the model to emit exactly the
    form the scorer rejects, and every grounding item would then read as a
    localisation failure rather than a formatting one.
    """
    x1, y1, x2, y2 = to_milli_box(bbox)
    return f"[{x1}, {y1}, {x2}, {y2}]"


#: How a SAR rendering is named by scripts/prepare_bigearthnet.py. Cross-modal
#: capability cannot be read off the `task` field: the prepared BigEarthNet
#: corpus stores the *annotation type* there ("binary", "mcq"), and the bench
#: adapter is what maps that to Task.CROSSMODAL_VQA at load time. The presence
#: of a SAR image is the thing that is actually true about the record.
_SAR_MARKER = "_sar"


def is_crossmodal(record: Mapping[str, Any]) -> bool:
    """Whether a prepared record shows the model an optical-SAR pair."""
    if str(record.get("task", "")) == str(Task.CROSSMODAL_VQA):
        return True
    return any(_SAR_MARKER in str(image).lower() for image in record.get("images") or ())


@dataclass
class SourceStats:
    """What one benchmark split contributed, and what it lost on the way."""

    name: str
    loaded: int = 0
    written: int = 0
    dropped_empty: int = 0
    dropped_missing_image: int = 0
    dropped_contaminated: int = 0
    dropped_duplicate: int = 0
    dropped_capped: int = 0
    with_evidence: int = 0
    crossmodal: int = 0
    shared_questions: int = 0
    dropped_task: int = 0
    dropped_too_long: int = 0

    def line(self) -> str:
        return (
            f"{self.name:<20} loaded={self.loaded:<8,} written={self.written:<8,} "
            f"empty={self.dropped_empty:<6,} no_image={self.dropped_missing_image:<6,} "
            f"contaminated={self.dropped_contaminated:<6,} "
            f"dup={self.dropped_duplicate:<6,} capped={self.dropped_capped:<7,} "
            f"task={self.dropped_task:<6,} long={self.dropped_too_long:<6,} "
            f"evidence={self.with_evidence:,}"
        )


@dataclass
class ConversionReport:
    """The corpus that was built, per source and in total."""

    sources: list[SourceStats] = field(default_factory=list)
    tasks: dict[str, int] = field(default_factory=dict)

    @property
    def written(self) -> int:
        return sum(s.written for s in self.sources)

    @property
    def contaminated(self) -> int:
        return sum(s.dropped_contaminated for s in self.sources)

    #: First path segment of every image reference written, with counts. One
    #: corpus must have one root: a mixture whose sources disagree resolves some
    #: fraction of its images one directory too deep, and that surfaces as a
    #: file-not-found thousands of training steps in rather than at build time.
    image_roots: dict[str, int] = field(default_factory=dict)

    @property
    def crossmodal(self) -> int:
        return sum(s.crossmodal for s in self.sources)

    def render(self) -> str:
        lines = [s.line() for s in self.sources]
        lines.append("-" * 100)
        mix = ", ".join(f"{k}={v:,}" for k, v in sorted(self.tasks.items()))
        lines.append(f"total written={self.written:,}  tasks: {mix}")
        # Reported separately because the `task` field cannot carry it: the
        # prepared BigEarthNet corpus stores its annotation type there.
        evidence = sum(s.with_evidence for s in self.sources)
        roots = ", ".join(
            f"{k}/={v:,}" for k, v in sorted(self.image_roots.items())
        )
        lines.append(f"image roots: {roots}")
        lines.append(
            f"optical-SAR records={self.crossmodal:,}  "
            f"evidence preambles={evidence:,} "
            f"({evidence / self.written:.1%})"
            if self.written
            else "empty corpus"
        )
        if self.contaminated:
            lines.append(
                f"NOTE: {self.contaminated:,} records dropped for overlapping the "
                "benchmark test splits"
            )
        return "\n".join(lines)


def sample_to_record(
    sample: Sample,
    source: str,
    image_root: Path | None = None,
    measure: Measurer = no_measurements,
    seed: int = DEFAULT_SEED,
    rate: float | None = None,
) -> Record | None:
    """One benchmark ``Sample`` as one training ``Record``.

    Returns ``None`` for a sample with nothing to learn from -- a blank answer,
    a grounding row with no box -- rather than emitting a record whose target is
    the empty string. Stage A's own diagnosis showed how expensive silence is to
    train on.
    """
    if sample.task is Task.GROUNDING:
        if sample.bbox is None:
            return None
        answer = format_box_answer(sample.bbox)
    else:
        answer = (sample.answer or "").strip()
        if not answer:
            return None

    prompt = build_prompt(sample)

    images: list[str] = []
    for image in sample.images:
        path = Path(image.path)
        if image_root is not None:
            # Resolved on both sides: a config carries a relative root
            # ("data/VRSBench_train") while the mixture's root is absolute, and
            # comparing them raw raises ValueError. The old fallback then wrote
            # the unrooted path, which looks correct in the file and resolves one
            # directory too deep at training time.
            try:
                rel = os.path.relpath(
                    os.path.abspath(path), os.path.abspath(image_root)
                )
            except (ValueError, OSError):
                rel = ".."
            # relpath walks upwards rather than failing, and "../../elsewhere"
            # is not a path the training box can resolve. Such an image does not
            # belong to this root; it is left absolute so mixture_warnings()
            # reports it instead of the corpus quietly carrying a broken link.
            if not rel.startswith(".."):
                images.append(rel.replace("\\", "/"))
                continue
        images.append(str(path).replace("\\", "/"))
    if not images:
        return None

    kwargs: dict[str, Any] = {}
    if rate is not None:
        kwargs["rate"] = rate
    choice = choose_evidence(
        sample.sample_id, prompt, answer, measure(sample), seed=seed, **kwargs
    )

    # Adapters already namespace their ids with the config name. Prefixing again
    # yields "vrsbench_train_vqa-vrsbench_train_vqa-28109", which is only ugly
    # until someone greps a corpus for a record they saw in a report.
    sample_id = (
        sample.sample_id
        if sample.sample_id.startswith(f"{source}-")
        else f"{source}-{sample.sample_id}"
    )

    return Record(
        sample_id=sample_id,
        patch_id=Path(sample.images[0].path).stem,
        task=str(sample.task),
        prompt=prompt,
        answer=answer,
        images=tuple(images),
        preamble=choice.preamble,
        evidence_kind=str(choice.kind.value),
    )


def _is_contaminated(sample: Sample, marks: Fingerprints) -> bool:
    """Whether this training sample touches anything the benchmark scores.

    Image identity alone is disqualifying. A shared scene under a different
    question is still a scene the model has been shown, and the benchmark can no
    longer distinguish generalisation from recall.
    """
    if not marks.images:
        return False
    if any(image_key(image.path) in marks.images for image in sample.images):
        return True
    question = question_key(sample.question)
    return bool(question) and any(
        (image_key(image.path), question) in marks.pairs for image in sample.images
    )


def convert_source(
    config: Any,
    marks: Fingerprints | None = None,
    image_root: Path | None = None,
    cap: int | None = None,
    measure: Measurer = no_measurements,
    seed: int = DEFAULT_SEED,
    rate: float | None = None,
    require_images: bool = True,
    on_contamination: str = "raise",
) -> tuple[list[Record], SourceStats]:
    """Convert one benchmark train split.

    ``on_contamination`` is ``"raise"`` by default. ``"drop"`` is the deliberate
    escape hatch for releases that genuinely reuse tiles between splits: the
    overlapping rows are excluded and counted, which is a defensible corpus,
    whereas keeping them is not. There is no option that keeps them.
    """
    from satquery.eval.datasets import load_benchmark

    if on_contamination not in {"raise", "drop"}:
        raise ValueError("on_contamination must be 'raise' or 'drop'")

    stats = SourceStats(name=config.name)
    samples = load_benchmark(config).load()
    stats.loaded = len(samples)

    records: list[Record] = []
    seen: set[tuple[str, str]] = set()

    for sample in samples:
        if marks is not None and _is_contaminated(sample, marks):
            stats.dropped_contaminated += 1
            if on_contamination == "raise":
                raise ContaminationError(
                    f"{config.name}: training sample {sample.sample_id} reuses an "
                    f"image or question from a benchmark test split "
                    f"({Path(sample.images[0].path).name}). This split is not safe "
                    "to train on as configured -- point 'annotations' at the train "
                    "file, or pass on_contamination='drop' to exclude the overlap "
                    "deliberately."
                )
            continue

        if require_images and not all(Path(i.path).exists() for i in sample.images):
            stats.dropped_missing_image += 1
            continue

        record = sample_to_record(
            sample, config.name, image_root, measure=measure, seed=seed, rate=rate
        )
        if record is None:
            stats.dropped_empty += 1
            continue

        # Templated corpora repeat the identical question on the identical image.
        # Kept once: duplicates spend the token budget teaching nothing new and
        # quietly reweight the mixture toward whichever template recurs most.
        key = (record.images[0], question_key(record.prompt) + "|" + record.answer)
        if key in seen:
            stats.dropped_duplicate += 1
            continue
        seen.add(key)
        records.append(record)

    if cap is not None and len(records) > cap:
        rng = random.Random(seed)
        chosen = rng.sample(range(len(records)), cap)
        stats.dropped_capped = len(records) - cap
        records = [records[i] for i in sorted(chosen)]

    stats.written = len(records)
    stats.with_evidence = sum(1 for r in records if r.evidence_kind != "none")
    stats.crossmodal = sum(
        1 for r in records if r.task == str(Task.CROSSMODAL_VQA)
    )
    return records, stats


def slice_image_prefix(corpus: str | Path, image_root: Path | None) -> str:
    """Where a prepared slice's own image paths sit relative to the mixture.

    A prepared corpus records its images relative to itself -- BigEarthNet's say
    ``images/<patch>_optical.png`` -- while converted benchmark records are
    written relative to the data root. Mixed without adjustment the corpus
    carries two different roots and a third of it cannot be found at training
    time, which surfaces as a file-not-found thousands of steps in.
    """
    if image_root is None:
        return ""
    try:
        rel = os.path.relpath(
            os.path.abspath(Path(corpus).parent), os.path.abspath(image_root)
        )
    except (ValueError, OSError):
        return ""
    # relpath happily walks upwards; a slice outside the root has no prefix that
    # would make its paths resolvable from inside it.
    return "" if rel.startswith("..") else rel.replace("\\", "/")


def load_slice(
    path: str | Path,
    name: str,
    marks: Fingerprints | None = None,
    cap: int | None = None,
    seed: int = DEFAULT_SEED,
    on_contamination: str = "raise",
    image_prefix: str = "",
    drop_tasks: frozenset[str] = frozenset(),
    max_answer_words: int | None = None,
) -> tuple[list[dict[str, Any]], SourceStats]:
    """Take a slice of an already-prepared corpus into the mixture.

    This is how BigEarthNet gets into Stage B. It is not converted -- it was
    written by ``scripts/prepare_bigearthnet.py`` and is already in the on-disk
    record shape, preamble applied. Re-rendering it would be wrong twice over:
    the preamble is baked into the human turn and would be applied again, and
    the measurements behind it are not recoverable from the written record.

    It is checked for contamination like anything else, and this is the check
    nobody has run yet. ``bigearthnet_bench`` is rendered from the same LMDB by
    the same script to the same ``{patch}_optical.png`` filenames, which makes
    it the split a BigEarthNet training corpus is most likely to overlap.
    """
    if on_contamination not in {"raise", "drop"}:
        raise ValueError("on_contamination must be 'raise' or 'drop'")

    stats = SourceStats(name=name)
    rows: list[dict[str, Any]] = []
    seen: set[tuple[str, str]] = set()

    for record in read_corpus(path):
        stats.loaded += 1
        record_id = str(record.get("id", f"{name}-{stats.loaded}"))

        if marks is not None:
            found = record_overlap(record, marks, record_id=record_id)
            if found is not None and not found.fatal:
                # Question-only reuse over a different image. Counted so the
                # report can say so, but the record is kept: every corpus here
                # is templated, so this fires in the thousands on splits that
                # share no imagery at all.
                stats.shared_questions += 1
                found = None
            if found is not None:
                stats.dropped_contaminated += 1
                if on_contamination == "raise":
                    raise ContaminationError(
                        f"{name}: record {record_id} overlaps a benchmark test "
                        f"split ({found.detail}). This corpus is not safe to "
                        f"train on -- rebuild the slice, or pass "
                        f"on_contamination='drop' to exclude the overlap "
                        f"deliberately."
                    )
                continue

        turns = record.get("conversations") or []
        if len(turns) != 2 or not str(turns[1].get("value", "")).strip():
            stats.dropped_empty += 1
            continue

        # A rehearsal slice is included for one capability, and it brings its
        # whole task mixture with it. BigEarthNet's captions run to a median of
        # 96 words against VRSBench's 52, so including them to preserve
        # cross-modal ability also teaches the model to write captions twice the
        # length the benchmark rewards -- measured, not hypothetical: Stage B
        # produced 112-word captions that overran the token budget and scored
        # CIDEr-D 0.006 against a base of 0.128.
        task = str(record.get("task", ""))
        if task in drop_tasks:
            stats.dropped_task += 1
            continue
        if max_answer_words is not None:
            if len(str(turns[1].get("value", "")).split()) > max_answer_words:
                stats.dropped_too_long += 1
                continue

        images = list(record.get("images") or ())
        if not images:
            stats.dropped_empty += 1
            continue

        key = (
            image_key(images[0]),
            question_key(turns[0].get("value")) + "|" + str(turns[1].get("value")),
        )
        if key in seen:
            stats.dropped_duplicate += 1
            continue
        seen.add(key)
        if image_prefix and image_prefix != ".":
            record = dict(record)
            record["images"] = [
                f"{image_prefix}/{image}".replace(chr(92), "/") for image in images
            ]
        rows.append(record)

    if cap is not None and len(rows) > cap:
        rng = random.Random(seed)
        chosen = rng.sample(range(len(rows)), cap)
        stats.dropped_capped = len(rows) - cap
        rows = [rows[i] for i in sorted(chosen)]

    stats.written = len(rows)
    stats.with_evidence = sum(
        1 for r in rows if str(r.get("evidence_kind", "none")) != "none"
    )
    stats.crossmodal = sum(1 for r in rows if is_crossmodal(r))
    return rows, stats


def build_corpus(
    configs: Sequence[Any],
    out: str | Path,
    marks: Fingerprints | None = None,
    image_root: Path | None = None,
    caps: dict[str, int] | None = None,
    measure: Measurer = no_measurements,
    seed: int = DEFAULT_SEED,
    rate: float | None = None,
    require_images: bool = True,
    on_contamination: str = "raise",
    shuffle: bool = True,
    include: Sequence[tuple[str, str | Path]] = (),
    slice_drop_tasks: frozenset[str] = DEFAULT_SLICE_DROP_TASKS,
    slice_max_answer_words: int | None = None,
) -> ConversionReport:
    """Build one shuffled JSONL corpus from train splits and prepared slices.

    ``include`` carries ``(name, path)`` pairs for corpora that are already in
    record shape. Stage B needs at least one: no benchmark train split contains
    an optical-SAR pair, so a corpus built from the three benchmarks alone
    trains a model with no cross-modal exposure at all, and the +0.1750 Stage A
    measured on that criterion is trained away. The same slice is the only
    source of evidence preambles, which no RGB benchmark image can support.

    Shuffled on write because the sources are concatenated: left in order, the
    model sees twelve thousand captions, then twelve thousand yes/no answers,
    and a run that stops early -- which is how every time-budgeted run ends --
    has trained on a corpus nobody chose.
    """
    caps = DEFAULT_CAPS if caps is None else caps
    report = ConversionReport()
    everything: list[dict[str, Any]] = []

    for config in configs:
        records, stats = convert_source(
            config,
            marks=marks,
            image_root=image_root,
            cap=caps.get(config.name),
            measure=measure,
            seed=seed,
            rate=rate,
            require_images=require_images,
            on_contamination=on_contamination,
        )
        report.sources.append(stats)
        everything.extend(json.loads(r.as_jsonl()) for r in records)

    for name, path in include:
        rows, stats = load_slice(
            path,
            name,
            marks=marks,
            cap=caps.get(name),
            seed=seed,
            on_contamination=on_contamination,
            image_prefix=slice_image_prefix(path, image_root),
            drop_tasks=slice_drop_tasks,
            max_answer_words=slice_max_answer_words,
        )
        report.sources.append(stats)
        everything.extend(rows)

    if shuffle:
        random.Random(seed).shuffle(everything)

    for record in everything:
        task = str(record.get("task", "?"))
        report.tasks[task] = report.tasks.get(task, 0) + 1
        for image in record.get("images") or ():
            head = str(image).replace("\\", "/").split("/")[0]
            report.image_roots[head] = report.image_roots.get(head, 0) + 1

    target = Path(out)
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("w", encoding="utf-8") as handle:
        for record in everything:
            print(json.dumps(record, ensure_ascii=False), file=handle)

    return report


#: Below this share of records carrying an evidence preamble, the adapted model
#: meets a prompt shape at inference it barely saw in training. Stage A ran at
#: 38%; scripts/train_lora.py warns under 20% for the same reason.
MIN_EVIDENCE_SHARE = 0.20


def mixture_warnings(report: ConversionReport) -> list[str]:
    """Composition faults that produce a healthy-looking run and a worse model.

    Neither of these raises, because both are legitimate for a deliberate
    ablation. Both are said loudly, because both are far more likely to be an
    accident -- and neither would show up in the loss curve, the sweep, or any
    other signal before the delta comes back smaller than the one before it.
    """
    warnings: list[str] = []
    total = report.written
    if not total:
        return ["the corpus is empty"]

    if not sum(s.crossmodal for s in report.sources):
        warnings.append(
            "no optical-SAR records in the mixture. No benchmark train split "
            "contains a cross-modal pair, so this corpus cannot maintain the "
            "capability Stage A measured at +0.1750 -- include the BigEarthNet "
            "slice with --include bigearthnet=<path>"
        )

    # An absolute path in a corpus is a path from the machine that built it.
    # Training happens somewhere else.
    absolute = [p for p in report.image_roots if Path(p).is_absolute() or ":" in p]
    if absolute:
        warnings.append(
            f"{len(absolute)} image path root(s) are absolute ({absolute[:2]}). "
            f"They point at the machine that built the corpus, not the one that "
            f"will train on it"
        )

    evidence = sum(s.with_evidence for s in report.sources)
    share = evidence / total
    if share < MIN_EVIDENCE_SHARE:
        warnings.append(
            f"only {share:.1%} of records carry an evidence preamble "
            f"({evidence:,} of {total:,}). The controller prepends one at "
            f"inference on every query, so the adapted model would meet a "
            f"prompt shape it barely saw in training. No RGB benchmark image "
            f"can supply one, so the share is set by the included slice: "
            f"re-prepare it with a higher --preamble-rate rather than by "
            f"making it bigger, which would just rerun Stage A"
        )
    return warnings


def mixture_of(path: str | Path) -> dict[str, int]:
    """Task histogram of a written corpus, for a quick sanity read."""
    counts: dict[str, int] = {}
    with Path(path).open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                task = json.loads(line).get("task", "?")
                counts[task] = counts.get(task, 0) + 1
    return counts
