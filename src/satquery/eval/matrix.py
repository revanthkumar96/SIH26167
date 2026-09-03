"""Multi-model benchmark matrix: pull, select, run, compare.

Loops **model-outer, dataset-inner** so each model is constructed once and reused
across every benchmark. Loading a vision-language model is the expensive part, and
rebuilding it per dataset would dominate the run.
  pattern: VLMEvalKit@main run.py:624 (model loop) then :658 (dataset loop)

Each ``(model, benchmark)`` cell writes its own result file and is skipped when that
file already exists, so an interrupted sweep resumes instead of restarting.
  pattern: VLMEvalKit@main run.py:645-672

Sample selection is a seeded subset shared by every model in the sweep -- without
that the comparison is between different question sets and means nothing.
"""

from __future__ import annotations

import json
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from satquery.eval.backends import BackendConfig, build_backend
from satquery.eval.datasets import BenchmarkConfig, load_benchmark
from satquery.eval.report import HEADLINE_METRIC, append_results
from satquery.eval.runner import EvalResult, run_benchmark
from satquery.models import catalog_entry

ProgressCallback = Callable[["MatrixProgress"], None]


@dataclass
class MatrixCell:
    """One model evaluated on one benchmark."""

    model_id: str
    model: str
    backend: str
    benchmark: str
    task: str = ""
    metrics: dict[str, float] = field(default_factory=dict)
    num_samples: int = 0
    duration_s: float = 0.0
    status: str = "pending"  # pending | running | done | cached | error
    error: str = ""

    @property
    def headline(self) -> float | None:
        return self.metrics.get(HEADLINE_METRIC.get(self.task, "oa"))

    def as_dict(self) -> dict[str, Any]:
        return {
            "model_id": self.model_id,
            "model": self.model,
            "backend": self.backend,
            "benchmark": self.benchmark,
            "task": self.task,
            "metrics": self.metrics,
            "num_samples": self.num_samples,
            "duration_s": round(self.duration_s, 2),
            "status": self.status,
            "error": self.error,
            "headline": self.headline,
            "headline_metric": HEADLINE_METRIC.get(self.task, "oa"),
        }


@dataclass
class MatrixProgress:
    """Sweep state, safe to serialise to a client after every cell."""

    state: str = "idle"  # idle | loading | running | ready | error
    detail: str = ""
    current_model: str = ""
    current_benchmark: str = ""
    cells_done: int = 0
    cells_total: int = 0
    cells: list[MatrixCell] = field(default_factory=list)

    @property
    def percent(self) -> float | None:
        if self.cells_total <= 0:
            return None
        return round(100.0 * self.cells_done / self.cells_total, 1)

    def as_dict(self) -> dict[str, Any]:
        return {
            "state": self.state,
            "detail": self.detail,
            "current_model": self.current_model,
            "current_benchmark": self.current_benchmark,
            "cells_done": self.cells_done,
            "cells_total": self.cells_total,
            "percent": self.percent,
            "cells": [c.as_dict() for c in self.cells],
        }


def model_slug(model: str) -> str:
    return model.replace("/", "__")


def cell_dir(out_root: Path, model: str, benchmark: str) -> Path:
    return out_root / model_slug(model) / benchmark


def load_cached(path: Path) -> dict[str, Any] | None:
    """A previously written metrics file, or ``None``."""
    metrics_file = path / "metrics.json"
    if not metrics_file.is_file():
        return None
    try:
        return json.loads(metrics_file.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None


def _backend_config(entry: dict[str, Any], overrides: dict[str, Any]) -> BackendConfig:
    return BackendConfig(
        model=str(entry["model"]),
        dtype=str(overrides.get("dtype", "auto")),
        max_pixels=overrides.get("max_pixels", BackendConfig.max_pixels),
        batch_size=int(overrides.get("batch_size", 8)),
    )


def run_matrix(
    model_ids: Sequence[str],
    configs: Sequence[BenchmarkConfig],
    out_root: Path,
    results_csv: Path | None = None,
    limit: int | None = 200,
    seed: int = 1234,
    reuse_cached: bool = True,
    on_update: ProgressCallback | None = None,
    overrides: dict[str, Any] | None = None,
) -> MatrixProgress:
    """Evaluate every model against every benchmark.

    Never raises: a model that fails to load marks its whole row as errored and
    the sweep continues to the next one, so one bad checkpoint does not lose the
    results already earned.
    """
    overrides = overrides or {}
    progress = MatrixProgress(state="loading")

    # Freeze the sample selection before anything runs: every model must see the
    # identical subset for the comparison to mean anything.
    prepared: list[BenchmarkConfig] = []
    for config in configs:
        if limit is not None:
            config.limit = limit
        config.seed = seed
        prepared.append(config)

    entries: list[tuple[str, dict[str, Any]]] = []
    for model_id in model_ids:
        entry = catalog_entry(model_id)
        if entry is None:
            progress.state = "error"
            progress.detail = f"unknown model '{model_id}'"
            if on_update is not None:
                on_update(progress)
            return progress
        entries.append((model_id, entry))

    progress.cells_total = len(entries) * len(prepared)
    for model_id, entry in entries:
        for config in prepared:
            progress.cells.append(
                MatrixCell(
                    model_id=model_id,
                    model=str(entry["model"]),
                    backend=str(entry["backend"]),
                    benchmark=config.name,
                    task=str(config.task),
                )
            )

    def emit() -> None:
        if on_update is not None:
            on_update(progress)

    emit()
    results: list[EvalResult] = []

    for model_id, entry in entries:
        row = [c for c in progress.cells if c.model_id == model_id]
        progress.current_model = str(entry["label"])
        progress.state = "loading"
        progress.detail = f"loading {entry['label']}"
        emit()

        backend = None
        try:
            backend = build_backend(
                str(entry["backend"]), _backend_config(entry, overrides)
            )
        except Exception as exc:
            message = f"{type(exc).__name__}: {exc}"
            for cell in row:
                cell.status = "error"
                cell.error = message
                progress.cells_done += 1
            progress.detail = f"{entry['label']} failed to load: {message}"
            emit()
            continue

        try:
            for config, cell in zip(prepared, row, strict=True):
                progress.current_benchmark = config.name
                target = cell_dir(out_root, cell.model, config.name)

                cached = load_cached(target) if reuse_cached else None
                if cached is not None:
                    cell.metrics = dict(cached.get("metrics", {}))
                    cell.num_samples = int(cached.get("num_samples", 0))
                    cell.duration_s = float(cached.get("duration_s", 0.0))
                    cell.status = "cached"
                    progress.cells_done += 1
                    progress.detail = f"{config.name}: reused cached result"
                    emit()
                    continue

                cell.status = "running"
                progress.state = "running"
                progress.detail = f"{entry['label']} on {config.name}"
                emit()

                try:
                    result = run_benchmark(
                        load_benchmark(config),
                        backend,
                        output_dir=target,
                        progress_every=0,
                    )
                except Exception as exc:
                    cell.status = "error"
                    cell.error = f"{type(exc).__name__}: {exc}"
                    progress.cells_done += 1
                    emit()
                    continue

                cell.metrics = dict(result.metrics)
                cell.num_samples = result.num_samples
                cell.duration_s = result.duration_s
                cell.status = "done"
                results.append(result)
                progress.cells_done += 1
                emit()
        finally:
            backend.close()

    if results and results_csv is not None:
        append_results(results, results_csv)

    progress.state = "ready"
    progress.current_model = ""
    progress.current_benchmark = ""
    errors = sum(1 for c in progress.cells if c.status == "error")
    progress.detail = f"{progress.cells_done} cell(s) complete" + (
        f", {errors} failed" if errors else ""
    )
    emit()
    return progress


def comparison_matrix(cells: Sequence[MatrixCell]) -> dict[str, Any]:
    """Models as rows, benchmarks as columns, headline metric per cell."""
    benchmarks: list[str] = []
    models: list[str] = []
    for cell in cells:
        if cell.benchmark not in benchmarks:
            benchmarks.append(cell.benchmark)
        if cell.model_id not in models:
            models.append(cell.model_id)

    lookup = {(c.model_id, c.benchmark): c for c in cells}
    rows = []
    for model_id in models:
        row: dict[str, Any] = {"model_id": model_id, "values": []}
        for benchmark in benchmarks:
            cell = lookup.get((model_id, benchmark))
            row["values"].append(
                {
                    "benchmark": benchmark,
                    "headline": cell.headline if cell else None,
                    "metric": HEADLINE_METRIC.get(cell.task, "oa") if cell else "",
                    "status": cell.status if cell else "pending",
                    "samples": cell.num_samples if cell else 0,
                }
            )
        rows.append(row)

    return {"benchmarks": benchmarks, "rows": rows}
