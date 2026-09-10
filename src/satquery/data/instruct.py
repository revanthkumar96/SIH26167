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
import random
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from satquery.data.bigearthnet import Record, to_milli_box
from satquery.data.contamination import (
    ContaminationError,
    Fingerprints,
    image_key,
    question_key,
)
from satquery.data.evidence import Measurements, choose_evidence
from satquery.eval.prompts import build_prompt
from satquery.schema import Sample, Task

#: How a source with more rows than its cap is reduced. Seeded, so a rebuilt
#: corpus is the same corpus.
DEFAULT_SEED = 1234

#: What each benchmark train split may contribute. Chosen so no single source
#: exceeds roughly a third of the mixture and captioning is not drowned by the
#: templated VQA corpora. Override per run; do not remove.
DEFAULT_CAPS: dict[str, int] = {
    "vrsbench_caption": 12_000,
    "vrsbench_vqa": 12_000,
    "vrsbench_referring": 8_000,
    "rsvqa_lr": 12_000,
    "rsvqa_hr": 12_000,
    "cdvqa": 8_000,
}

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

    def line(self) -> str:
        return (
            f"{self.name:<20} loaded={self.loaded:<8,} written={self.written:<8,} "
            f"empty={self.dropped_empty:<6,} no_image={self.dropped_missing_image:<6,} "
            f"contaminated={self.dropped_contaminated:<6,} "
            f"dup={self.dropped_duplicate:<6,} capped={self.dropped_capped:<7,} "
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

    def render(self) -> str:
        lines = [s.line() for s in self.sources]
        lines.append("-" * 100)
        mix = ", ".join(f"{k}={v:,}" for k, v in sorted(self.tasks.items()))
        lines.append(f"total written={self.written:,}  tasks: {mix}")
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
            try:
                images.append(str(path.relative_to(image_root)).replace("\\", "/"))
                continue
            except ValueError:
                pass
        images.append(str(path).replace("\\", "/"))
    if not images:
        return None

    kwargs: dict[str, Any] = {}
    if rate is not None:
        kwargs["rate"] = rate
    choice = choose_evidence(
        sample.sample_id, prompt, answer, measure(sample), seed=seed, **kwargs
    )

    return Record(
        sample_id=f"{source}-{sample.sample_id}",
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
    return records, stats


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
) -> ConversionReport:
    """Convert every configured train split into one shuffled JSONL corpus.

    Shuffled on write because the sources are concatenated: left in order, the
    model sees twelve thousand captions, then twelve thousand yes/no answers,
    and a run that stops early -- which is how every time-budgeted run ends --
    has trained on a corpus nobody chose.
    """
    caps = DEFAULT_CAPS if caps is None else caps
    report = ConversionReport()
    everything: list[Record] = []

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
        everything.extend(records)

    if shuffle:
        random.Random(seed).shuffle(everything)

    for record in everything:
        report.tasks[record.task] = report.tasks.get(record.task, 0) + 1

    target = Path(out)
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("w", encoding="utf-8") as handle:
        for record in everything:
            handle.write(record.as_jsonl() + "\n")

    return report


def mixture_of(path: str | Path) -> dict[str, int]:
    """Task histogram of a written corpus, for a quick sanity read."""
    counts: dict[str, int] = {}
    with Path(path).open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                task = json.loads(line).get("task", "?")
                counts[task] = counts.get(task, 0) + 1
    return counts
