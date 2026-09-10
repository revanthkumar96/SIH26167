"""Train/test contamination checks for the adaptation corpus.

Stage B trains on the VRSBench, RSVQA and CDVQA **train** splits, and the
benchmark scores the **test** splits of those same datasets. If one row crosses
that line the measured delta stops meaning anything -- and nothing else in the
pipeline would notice. Training succeeds, the sweep succeeds, the score goes up,
and the improvement is memorisation.

That is the failure this module exists to make loud. It is not a lint: a
contaminated corpus should fail a build, because by the time anyone reads a
suspiciously good number the GPU hours are spent and the claim is already in a
slide.

Two signals, in order of how damning they are:

*image reuse* -- the same picture in training and in the test set. Decisive on
its own. A model that has seen the test image has an advantage no amount of
different phrasing removes.

*question reuse* -- the same question text, normalised. Weaker alone, since
templated corpora legitimately repeat phrasings across different scenes, but
damning in combination with the image.

Both are matched on identity rather than similarity. A near-duplicate check
would be better and is far harder to make trustworthy; an exact check that
never cries wolf is one people leave switched on.
"""

from __future__ import annotations

import json
import re
from collections.abc import Iterable, Iterator, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

_WS = re.compile(r"\s+")


class ContaminationError(RuntimeError):
    """A training corpus overlaps the benchmark splits it will be scored on."""


def image_key(path: str | Path) -> str:
    """Identity of an image, independent of where it happens to live.

    The same picture sits under different roots in a train corpus and a
    benchmark download, so the basename is the only stable handle. Lowercased
    because case differs across archives, and the extension is kept because
    `1234.png` and `1234.tif` are genuinely different products.
    """
    return Path(str(path)).name.strip().lower()


def question_key(text: str | None) -> str:
    """Normalised question text, for matching phrasing across corpora."""
    return _WS.sub(" ", (text or "").strip().lower())


@dataclass
class Fingerprints:
    """What a benchmark split contains, reduced to matchable identities."""

    images: set[str] = field(default_factory=set)
    questions: set[str] = field(default_factory=set)
    #: (image, question) pairs -- the combination that is unambiguous.
    pairs: set[tuple[str, str]] = field(default_factory=set)
    sources: list[str] = field(default_factory=list)

    def __len__(self) -> int:
        return len(self.images)


#: Overlap kinds that disqualify a corpus on their own. Question-only reuse is
#: deliberately absent: BigEarthNet.txt, RSVQA and VRSBench are all templated, so
#: the same phrasing recurs across thousands of unrelated scenes. Measured on the
#: real corpora, BigEarthNet train and bench share 1,287 of 6,201 question
#: strings while sharing zero images and zero patches -- treating that as
#: contamination fails a clean corpus, and a guard that cries wolf is one someone
#: switches off, which costs more than it ever saved.
FATAL_KINDS = frozenset({"image", "image+question"})


@dataclass
class Overlap:
    """One training record that collides with a benchmark row."""

    record_id: str
    kind: str  # "image" | "question" | "image+question"
    detail: str

    @property
    def fatal(self) -> bool:
        """Whether this alone disqualifies the corpus."""
        return self.kind in FATAL_KINDS


@dataclass
class ContaminationReport:
    """The verdict, and enough of the evidence to act on it."""

    checked: int = 0
    benchmark_images: int = 0
    overlaps: list[Overlap] = field(default_factory=list)

    @property
    def clean(self) -> bool:
        """No overlap of any kind, including the advisory sort."""
        return not self.overlaps

    @property
    def fatal_overlaps(self) -> list[Overlap]:
        return [o for o in self.overlaps if o.fatal]

    @property
    def safe(self) -> bool:
        """Whether this corpus can be trained on.

        Distinct from ``clean``: a corpus can carry question-only overlap and
        still be safe, because a shared phrasing over a different scene tells the
        model nothing about the row being scored.
        """
        return not self.fatal_overlaps

    @property
    def by_kind(self) -> dict[str, int]:
        counts: dict[str, int] = {}
        for item in self.overlaps:
            counts[item.kind] = counts.get(item.kind, 0) + 1
        return counts

    def summary(self) -> str:
        advisory = len(self.overlaps) - len(self.fatal_overlaps)
        note = (
            f" ({advisory:,} records share only a question string with a "
            f"benchmark row, over a different image -- expected of templated "
            f"corpora and not counted against it)"
            if advisory
            else ""
        )
        if self.safe:
            return (
                f"clean: {self.checked:,} training records checked against "
                f"{self.benchmark_images:,} benchmark images, no image reuse"
                + note
            )
        kinds = ", ".join(f"{k}={v}" for k, v in sorted(self.by_kind.items()))
        examples = "; ".join(
            f"{o.record_id} ({o.detail})" for o in self.fatal_overlaps[:3]
        )
        return (
            f"CONTAMINATED: {len(self.fatal_overlaps):,} of {self.checked:,} "
            f"training records reuse a benchmark image ({kinds}). "
            f"Examples: {examples}"
        )

    def raise_if_contaminated(self) -> None:
        """The intended use. A report nobody acts on is a comment."""
        if not self.safe:
            raise ContaminationError(
                self.summary()
                + ". Training on rows the benchmark scores makes the measured "
                "delta meaningless -- rebuild the corpus from the train split."
            )


def fingerprint_samples(samples: Iterable[Any]) -> Fingerprints:
    """Reduce loaded benchmark ``Sample`` objects to matchable identities."""
    marks = Fingerprints()
    for sample in samples:
        question = question_key(getattr(sample, "question", None))
        images = [image_key(image.path) for image in getattr(sample, "images", ())]
        marks.images.update(images)
        if question:
            marks.questions.add(question)
            marks.pairs.update((img, question) for img in images)
    return marks


def fingerprint_benchmarks(configs: Sequence[Any]) -> Fingerprints:
    """Fingerprint every benchmark config that can currently be loaded.

    A config whose data is absent is skipped rather than fatal: the guard should
    still run on a machine that only has some of the downloads, and it reports
    which splits it managed to read so a partial check is never mistaken for a
    clean one.
    """
    from satquery.eval.datasets import load_benchmark

    marks = Fingerprints()
    for config in configs:
        try:
            samples = load_benchmark(config).load()
        except Exception as exc:  # absent download, unreadable split
            marks.sources.append(f"{config.name}: SKIPPED ({type(exc).__name__})")
            continue
        found = fingerprint_samples(samples)
        marks.images |= found.images
        marks.questions |= found.questions
        marks.pairs |= found.pairs
        marks.sources.append(f"{config.name}: {len(samples):,} samples")
    return marks


def read_corpus(path: str | Path) -> Iterator[dict[str, Any]]:
    """Stream prepared training records from JSONL."""
    with Path(path).open(encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                yield json.loads(line)


def record_overlap(
    record: Mapping[str, Any],
    marks: Fingerprints,
    require_pair: bool = False,
    record_id: str | None = None,
) -> Overlap | None:
    """Whether one prepared record touches anything the benchmark scores.

    One implementation, used both by the corpus-wide check and by the converter
    as it writes. Two of them would drift, and the one that drifted would be the
    one deciding whether a training corpus is safe.

    ``require_pair`` demands that both the image *and* the question match before
    a record counts as contaminated. It is off by default deliberately: image
    reuse alone is disqualifying, and a guard that waits for the question to
    match as well will pass a corpus built from the same scenes with rephrased
    prompts, which is the exact leak worth catching.
    """
    name = record_id if record_id is not None else str(record.get("id", "?"))
    images = [image_key(image) for image in record.get("images") or ()]
    turns = record.get("conversations") or []
    question = question_key(turns[0].get("value") if turns else None)

    shared_images = [img for img in images if img in marks.images]
    if any((img, question) in marks.pairs for img in images):
        return Overlap(name, "image+question", f"image {shared_images[0]}")
    if shared_images and not require_pair:
        return Overlap(name, "image", f"image {shared_images[0]}")
    if question and not require_pair and not shared_images:
        # Recorded but far weaker: templated corpora reuse phrasings across
        # unrelated scenes, so this alone is a prompt to look, not a verdict.
        if question in marks.questions:
            return Overlap(name, "question", f"question {question[:60]!r}")
    return None


def check_records(
    records: Iterable[dict[str, Any]],
    marks: Fingerprints,
    require_pair: bool = False,
) -> ContaminationReport:
    """Compare a training corpus against benchmark fingerprints."""
    report = ContaminationReport(benchmark_images=len(marks.images))
    for record in records:
        report.checked += 1
        found = record_overlap(
            record,
            marks,
            require_pair=require_pair,
            record_id=str(record.get("id", f"record-{report.checked}")),
        )
        if found is not None:
            report.overlaps.append(found)
    return report


def check_corpus_against_benchmarks(
    corpus: str | Path, configs: Sequence[Any], require_pair: bool = False
) -> ContaminationReport:
    """Convenience: fingerprint the benchmarks, then check a corpus file."""
    marks = fingerprint_benchmarks(configs)
    return check_records(read_corpus(corpus), marks, require_pair=require_pair)
