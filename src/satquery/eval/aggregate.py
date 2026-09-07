"""Score normalisation and aggregation across heterogeneous metrics.

The judging criteria require scores to be normalised *before* they are
aggregated, and the reason is visible in our own numbers rather than
theoretical: ``cider_d`` is scaled by 10 in the standard formulation, so a plain
mean over headline metrics lets captioning outweigh every other criterion by an
order of magnitude. A model that captions well and cannot ground at all would
outrank one that is competent at both.

Two decisions worth stating, because both are places a defensible aggregate can
quietly stop being one.

**Fixed theoretical bounds, not the range observed across the sweep.**
Min-maxing over the models present would make the better of any two 1.0 and the
worse 0.0 whatever their absolute quality, and the same model would score
differently depending on who it was compared against. The final evaluation is
against a hidden reference we never see, so a score has to mean something on its
own. Every bound here is the metric's own range: accuracies, F-measures and IoU
are [0, 1] by construction, and CIDEr-D is [0, 10] because its per-n-gram term is
a cosine similarity times a gaussian -- both at most 1 -- multiplied by the
conventional factor of 10.

**Criteria are averaged, not benchmarks.** VRSBench contributes three splits and
CDVQA one, so a flat mean over benchmarks would weight single-image performance
three times as heavily as change understanding. The problem statement scores
criteria, so benchmarks are averaged within their criterion first.

Un-normalisable metrics are reported as such rather than coerced. A number
outside its declared range means the metric changed and the bound did not, and
silently clamping it would hide exactly the drift worth catching.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

from satquery.eval.report import HEADLINE_METRIC

#: Declared range per metric, used to map a raw score onto [0, 1].
#:
#: These are the metrics' own bounds, not observed extremes. ``cider_d`` is the
#: only one that is not already [0, 1]: ``caption.py`` multiplies by 10.0 after
#: dividing a cosine similarity by the n-gram count, so 10.0 is the value a
#: perfect candidate against its own reference attains.
METRIC_RANGES: Mapping[str, tuple[float, float]] = {
    "oa": (0.0, 1.0),
    "aa": (0.0, 1.0),
    "miou": (0.0, 1.0),
    "acc@0.25": (0.0, 1.0),
    "acc@0.5": (0.0, 1.0),
    "bleu1": (0.0, 1.0),
    "bleu2": (0.0, 1.0),
    "bleu3": (0.0, 1.0),
    "bleu4": (0.0, 1.0),
    "rouge_l": (0.0, 1.0),
    "meteor": (0.0, 1.0),
    "cider_d": (0.0, 10.0),
    "parse_rate": (0.0, 1.0),
}

#: Which scored criterion a task belongs to. Change understanding is measured by
#: two tasks and single-image work by three benchmarks; grouping here is what
#: stops the count of splits from becoming a weighting.
TASK_CRITERION: Mapping[str, str] = {
    "vqa": "single_image_vqa",
    "caption": "captioning",
    "grounding": "grounding",
    "change_vqa": "change_understanding",
    "change_caption": "change_understanding",
    "crossmodal_vqa": "crossmodal",
}

#: Human-readable criterion names, and whether the problem statement makes the
#: criterion mandatory. Reported alongside the score so a missing mandatory
#: criterion is visible rather than merely absent from a table.
CRITERIA: Mapping[str, tuple[str, bool]] = {
    "single_image_vqa": ("Single-image VQA", True),
    "captioning": ("Captioning", False),
    "grounding": ("Grounding", False),
    "change_understanding": ("Change understanding", True),
    "crossmodal": ("Optical-SAR joint", True),
}


class NormalisationError(ValueError):
    """A metric value falls outside the range declared for it."""


def normalise_metric(metric: str, value: float) -> float:
    """Map one raw metric onto [0, 1] using its declared range.

    Raises rather than clamping when the value is out of range: that means the
    metric's definition moved and ``METRIC_RANGES`` did not, which is a bug to
    fix rather than a number to round off.
    """
    if metric not in METRIC_RANGES:
        raise NormalisationError(
            f"no declared range for metric '{metric}'. Add it to METRIC_RANGES "
            f"so its contribution to the aggregate is explicit."
        )
    low, high = METRIC_RANGES[metric]
    # A small tolerance, because floating-point accumulation in CIDEr-D can land
    # a hair outside a bound that the mathematics guarantees.
    if not (low - 1e-9) <= value <= (high + 1e-9):
        raise NormalisationError(
            f"metric '{metric}' = {value:.6f} lies outside its declared range "
            f"[{low}, {high}]"
        )
    span = high - low
    if span <= 0:
        raise NormalisationError(f"metric '{metric}' has an empty range")
    return min(max((value - low) / span, 0.0), 1.0)


@dataclass(frozen=True, slots=True)
class BenchmarkScore:
    """One benchmark's headline metric, raw and normalised."""

    benchmark: str
    task: str
    criterion: str
    metric: str
    raw: float
    normalised: float
    num_samples: int = 0

    def as_dict(self) -> dict[str, Any]:
        return {
            "benchmark": self.benchmark,
            "task": self.task,
            "criterion": self.criterion,
            "metric": self.metric,
            "raw": round(self.raw, 6),
            "normalised": round(self.normalised, 6),
            "num_samples": self.num_samples,
        }


@dataclass
class AggregateScore:
    """A model's normalised score, per criterion and overall."""

    model: str
    benchmarks: list[BenchmarkScore] = field(default_factory=list)
    criteria: dict[str, float] = field(default_factory=dict)
    overall: float | None = None
    missing_criteria: list[str] = field(default_factory=list)
    skipped: list[dict[str, str]] = field(default_factory=list)

    def as_dict(self) -> dict[str, Any]:
        return {
            "model": self.model,
            "overall": None if self.overall is None else round(self.overall, 6),
            "criteria": {k: round(v, 6) for k, v in sorted(self.criteria.items())},
            "benchmarks": [b.as_dict() for b in self.benchmarks],
            "missing_criteria": self.missing_criteria,
            "skipped": self.skipped,
        }


def _headline(task: str, metrics: Mapping[str, float]) -> tuple[str, float] | None:
    """The metric a task is ranked by, if the run produced it."""
    metric = HEADLINE_METRIC.get(task, "oa")
    value = metrics.get(metric)
    if value is None:
        return None
    return metric, float(value)


def score_model(model: str, cells: Iterable[Mapping[str, Any]]) -> AggregateScore:
    """Normalise and aggregate one model's benchmark cells.

    ``cells`` are mappings carrying at least ``benchmark``, ``task`` and
    ``metrics`` -- the shape ``MatrixCell.as_dict()`` and ``EvalResult.summary()``
    both already produce, so neither has to be adapted to be scored.

    A cell that errored, produced no headline metric, or belongs to no known
    criterion is recorded in ``skipped`` with the reason. Dropping it silently
    would let a failed benchmark quietly raise the mean of the ones that ran.
    """
    result = AggregateScore(model=model)
    by_criterion: dict[str, list[float]] = {}

    for cell in cells:
        benchmark = str(cell.get("benchmark", ""))
        task = str(cell.get("task", ""))
        metrics = cell.get("metrics") or {}

        criterion = TASK_CRITERION.get(task)
        if criterion is None:
            result.skipped.append(
                {
                    "benchmark": benchmark,
                    "reason": f"task '{task}' maps to no criterion",
                }
            )
            continue

        headline = _headline(task, metrics)
        if headline is None:
            result.skipped.append(
                {
                    "benchmark": benchmark,
                    "reason": f"no '{HEADLINE_METRIC.get(task, 'oa')}' in metrics",
                }
            )
            continue

        metric, raw = headline
        try:
            normalised = normalise_metric(metric, raw)
        except NormalisationError as exc:
            result.skipped.append({"benchmark": benchmark, "reason": str(exc)})
            continue

        result.benchmarks.append(
            BenchmarkScore(
                benchmark=benchmark,
                task=task,
                criterion=criterion,
                metric=metric,
                raw=raw,
                normalised=normalised,
                num_samples=int(cell.get("num_samples", 0) or 0),
            )
        )
        by_criterion.setdefault(criterion, []).append(normalised)

    result.criteria = {
        criterion: sum(values) / len(values)
        for criterion, values in sorted(by_criterion.items())
    }
    # Mandatory criteria are listed first: an unscored mandatory capability is
    # the one gap that costs more than a low score on it would have.
    result.missing_criteria = [
        name
        for name, (_, mandatory) in sorted(
            CRITERIA.items(), key=lambda item: not item[1][1]
        )
        if name not in result.criteria
    ]
    if result.criteria:
        result.overall = sum(result.criteria.values()) / len(result.criteria)
    return result


def score_matrix(cells: Sequence[Mapping[str, Any]]) -> list[AggregateScore]:
    """Aggregate every model in a sweep, preserving the order they appear in."""
    order: list[str] = []
    grouped: dict[str, list[Mapping[str, Any]]] = {}
    for cell in cells:
        model = str(cell.get("model_id") or cell.get("model") or "")
        if model not in grouped:
            grouped[model] = []
            order.append(model)
        grouped[model].append(cell)
    return [score_model(model, grouped[model]) for model in order]


def cells_from_results(rows: Iterable[Mapping[str, str]]) -> list[dict[str, Any]]:
    """Pivot the long-format ``results.csv`` back into one cell per run.

    The CSV stores a row per metric so that adding a metric never changes the
    schema, which means aggregating from it needs the inverse. Keyed by model
    *and* timestamp: two runs of the same model at different sample limits are
    different measurements, and silently merging them would average a 200-sample
    probe into a 1,000-sample headline.
    """
    grouped: dict[tuple[str, str, str], dict[str, Any]] = {}
    for row in rows:
        key = (row["timestamp"], row["model"], row["benchmark"])
        cell = grouped.setdefault(
            key,
            {
                "model_id": row["model"],
                "model": row["model"],
                "backend": row["backend"],
                "benchmark": row["benchmark"],
                "task": row["task"],
                "timestamp": row["timestamp"],
                "num_samples": int(row["num_samples"]),
                "prompt_version": row.get("prompt_version", ""),
                "git_sha": row.get("git_sha", ""),
                "metrics": {},
            },
        )
        try:
            cell["metrics"][row["metric"]] = float(row["value"])
        except (TypeError, ValueError):
            continue
    return list(grouped.values())


def delta_table(scores: Sequence[AggregateScore]) -> str:
    """Markdown comparison of normalised criterion scores, first row as baseline.

    The base-versus-adapted delta is the artefact the whole programme exists to
    produce, so it gets a rendering of its own rather than being left for a
    reader to subtract by eye.
    """
    if not scores:
        return "_no scores_"

    criteria = [name for name in CRITERIA if any(name in s.criteria for s in scores)]
    labels = [CRITERIA[name][0] for name in criteria]

    header = "| model | " + " | ".join(labels) + " | overall |"
    divider = "|---|" + "---|" * (len(criteria) + 1)

    def cell(value: float | None) -> str:
        return "-" if value is None else f"{value:.4f}"

    rows = []
    baseline = scores[0]
    for index, score in enumerate(scores):
        values = [cell(score.criteria.get(name)) for name in criteria]
        row = f"| {score.model} | " + " | ".join(values) + f" | {cell(score.overall)} |"
        rows.append(row)

        if index == 0 or score.overall is None or baseline.overall is None:
            continue
        deltas = []
        for name in criteria:
            here, there = score.criteria.get(name), baseline.criteria.get(name)
            deltas.append(
                "-" if here is None or there is None else f"{here - there:+.4f}"
            )
        deltas.append(f"{score.overall - baseline.overall:+.4f}")
        rows.append("| ^ delta | " + " | ".join(deltas) + " |")

    note = (
        "\nScores are min-max normalised onto [0, 1] against each metric's own "
        "declared range before aggregation, then averaged within a criterion and "
        "across criteria. Deltas are against the first row."
    )
    return "\n".join([header, divider, *rows]) + note
