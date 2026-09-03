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


def read_results(path: Path) -> list[dict[str, str]]:
    """Load the long-format results CSV."""
    if not path.is_file():
        return []
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def results_dashboard(path: Path) -> dict:
    """Pivot ``results.csv`` into a bake-off comparison for the UI."""
    rows = read_results(path)
    if not rows:
        return {
            "total_rows": 0,
            "benchmarks": [],
            "models": [],
            "runs": [],
            "table_markdown": "_no results yet_",
        }

    benchmarks: list[str] = []
    for row in rows:
        if row["benchmark"] not in benchmarks:
            benchmarks.append(row["benchmark"])

    model_keys: list[tuple[str, str]] = []
    for row in rows:
        key = (row["backend"], row["model"])
        if key not in model_keys:
            model_keys.append(key)

    headline_cells: dict[tuple[str, str, str], float] = {}
    for row in rows:
        headline = HEADLINE_METRIC.get(row["task"], "oa")
        if row["metric"] != headline:
            continue
        headline_cells[(row["backend"], row["model"], row["benchmark"])] = float(
            row["value"]
        )

    models = [
        {
            "backend": backend,
            "model": model,
            "scores": {
                bench: headline_cells.get((backend, model, bench))
                for bench in benchmarks
            },
        }
        for backend, model in model_keys
    ]

    runs: dict[tuple[str, str, str], dict] = {}
    for row in rows:
        key = (row["timestamp"], row["backend"], row["model"])
        if key not in runs:
            runs[key] = {
                "timestamp": row["timestamp"],
                "backend": row["backend"],
                "model": row["model"],
                "benchmarks": {},
            }
        headline = HEADLINE_METRIC.get(row["task"], "oa")
        if row["metric"] == headline:
            runs[key]["benchmarks"][row["benchmark"]] = {
                "task": row["task"],
                "metric": row["metric"],
                "value": float(row["value"]),
                "num_samples": int(row["num_samples"]),
                "duration_s": float(row["duration_s"]),
            }

    return {
        "total_rows": len(rows),
        "benchmarks": benchmarks,
        "models": models,
        "runs": sorted(runs.values(), key=lambda item: item["timestamp"], reverse=True),
        "table_markdown": _markdown_from_rows(rows, benchmarks, model_keys),
    }


def _markdown_from_rows(
    rows: list[dict[str, str]],
    benchmarks: list[str],
    model_keys: list[tuple[str, str]],
) -> str:
    cells: dict[tuple[str, str, str], float] = {}
    for row in rows:
        headline = HEADLINE_METRIC.get(row["task"], "oa")
        if row["metric"] == headline:
            cells[(row["backend"], row["model"], row["benchmark"])] = float(
                row["value"]
            )

    header = "| model | " + " | ".join(benchmarks) + " |"
    divider = "|---|" + "---|" * len(benchmarks)
    body = [
        "| "
        + f"{model} ({backend})"
        + " | "
        + " | ".join(
            f"{cells.get((backend, model, bench), 0):.4f}"
            if (backend, model, bench) in cells
            else "-"
            for bench in benchmarks
        )
        + " |"
        for backend, model in model_keys
    ]
    note = "\nHeadline metric per task: " + ", ".join(
        f"{task}={metric}" for task, metric in sorted(HEADLINE_METRIC.items())
    )
    return "\n".join([header, divider, *body]) + note


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
