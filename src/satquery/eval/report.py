"""Results logging and the bake-off comparison table.

Every run appends one row to a single CSV. The submission's results table is
generated from that file, never retyped -- a retyped number is a number nobody can
reproduce.
"""

from __future__ import annotations

import csv
from collections.abc import Iterable, Sequence
from pathlib import Path

from satquery.eval.runner import EvalResult

RESULTS_COLUMNS = (
    "timestamp",
    "benchmark",
    "task",
    "backend",
    "model",
    "num_samples",
    "metric",
    "value",
    "duration_s",
    "prompt_version",
    "config_hash",
    "git_sha",
)

#: The metric each benchmark is ranked by in the bake-off summary.
HEADLINE_METRIC = {
    "vqa": "oa",
    "change_vqa": "oa",
    "crossmodal_vqa": "oa",
    "caption": "cider_d",
    "change_caption": "cider_d",
    "grounding": "acc@0.5",
}


def append_results(results: Iterable[EvalResult], path: Path) -> None:
    """Append one row per metric, long-format so new metrics never break the schema."""
    path.parent.mkdir(parents=True, exist_ok=True)
    exists = path.exists()
    with path.open("a", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=RESULTS_COLUMNS)
        if not exists:
            writer.writeheader()
        for result in results:
            for metric, value in sorted(result.metrics.items()):
                writer.writerow(
                    {
                        "timestamp": result.timestamp,
                        "benchmark": result.benchmark,
                        "task": result.task,
                        "backend": result.backend,
                        "model": result.model,
                        "num_samples": result.num_samples,
                        "metric": metric,
                        "value": f"{value:.6f}",
                        "duration_s": f"{result.duration_s:.2f}",
                        "prompt_version": result.prompt_version,
                        "config_hash": result.config_hash,
                        "git_sha": result.git_sha,
                    }
                )


def format_result(result: EvalResult) -> str:
    """Single-run summary for the console."""
    headline = HEADLINE_METRIC.get(result.task)
    lines = [
        f"{result.benchmark}  [{result.task}]  {result.model} via {result.backend}",
        f"  samples: {result.num_samples}   time: {result.duration_s:.1f}s "
        f"({result.num_samples / max(result.duration_s, 1e-6):.1f}/s)",
    ]
    for metric, value in sorted(result.metrics.items()):
        if metric.startswith("n/") or metric == "n":
            continue
        marker = " <-" if metric == headline else ""
        lines.append(f"  {metric:<20} {value:.4f}{marker}")
    return "\n".join(lines)


def comparison_table(results: Sequence[EvalResult]) -> str:
    """Markdown table of headline metrics, models as rows, benchmarks as columns."""
    if not results:
        return "_no results_"

    benchmarks: list[str] = []
    for result in results:
        if result.benchmark not in benchmarks:
            benchmarks.append(result.benchmark)

    models: list[str] = []
    for result in results:
        if result.model not in models:
            models.append(result.model)

    cells: dict[tuple[str, str], str] = {}
    for result in results:
        metric = HEADLINE_METRIC.get(result.task, "oa")
        value = result.metrics.get(metric)
        cells[(result.model, result.benchmark)] = (
            f"{value:.4f}" if value is not None else "-"
        )

    header = "| model | " + " | ".join(benchmarks) + " |"
    divider = "|---|" + "---|" * len(benchmarks)
    rows = [
        "| "
        + model
        + " | "
        + " | ".join(cells.get((model, b), "-") for b in benchmarks)
        + " |"
        for model in models
    ]
    note = "\nHeadline metric per task: " + ", ".join(
        f"{task}={metric}" for task, metric in sorted(HEADLINE_METRIC.items())
    )
    return "\n".join([header, divider, *rows]) + "\n" + note
