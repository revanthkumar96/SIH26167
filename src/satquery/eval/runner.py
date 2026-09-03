"""Benchmark runner: dataset -> backend -> predictions -> metrics.

Every run writes three artefacts: ``predictions.jsonl`` (raw output kept alongside
the parsed answer, so a scoring dispute can always be traced back), ``metrics.json``,
and one appended row in the shared ``results.csv``.
"""

from __future__ import annotations

import hashlib
import json
import subprocess
import time
from collections.abc import Sequence
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from satquery.eval.backends.base import VLMBackend
from satquery.eval.datasets.base import BenchmarkDataset
from satquery.eval.metrics import caption_metrics, grounding_metrics, vqa_metrics
from satquery.eval.normalize import parse_bbox
from satquery.eval.prompts import PROMPT_VERSION, build_prompt, max_new_tokens
from satquery.schema import GenerationRequest, Prediction, Sample, Task

_VQA_TASKS = frozenset({Task.VQA, Task.CHANGE_VQA, Task.CROSSMODAL_VQA})
_CAPTION_TASKS = frozenset({Task.CAPTION, Task.CHANGE_CAPTION})


@dataclass
class EvalResult:
    """One benchmark x one model."""

    benchmark: str
    task: str
    backend: str
    model: str
    num_samples: int
    metrics: dict[str, float]
    duration_s: float
    prompt_version: str = PROMPT_VERSION
    git_sha: str = ""
    config_hash: str = ""
    timestamp: str = ""
    predictions: list[Prediction] = field(default_factory=list, repr=False)

    def summary(self) -> dict[str, Any]:
        payload = asdict(self)
        payload.pop("predictions", None)
        return payload


def git_sha() -> str:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            capture_output=True,
            text=True,
            timeout=5,
            check=False,
        )
    except (OSError, subprocess.SubprocessError):
        return ""
    return result.stdout.strip() if result.returncode == 0 else ""


def config_hash(payload: dict[str, Any]) -> str:
    blob = json.dumps(payload, sort_keys=True, default=str).encode("utf-8")
    return hashlib.sha256(blob).hexdigest()[:12]


def build_requests(samples: Sequence[Sample]) -> list[GenerationRequest]:
    return [
        GenerationRequest(
            sample_id=sample.sample_id,
            prompt=build_prompt(sample),
            images=tuple(image.path for image in sample.images),
            max_new_tokens=max_new_tokens(sample.task),
        )
        for sample in samples
    ]


def score(
    samples: Sequence[Sample], predictions: Sequence[Prediction]
) -> dict[str, float]:
    """Dispatch to the metric family for this task."""
    if not samples:
        return {}
    task = samples[0].task

    if task in _VQA_TASKS:
        return vqa_metrics(
            [p.answer or p.raw_text for p in predictions],
            [s.answer or "" for s in samples],
            [s.qtype for s in samples],
        )

    if task in _CAPTION_TASKS:
        metrics = caption_metrics(
            [p.raw_text for p in predictions],
            [list(s.gold_texts) for s in samples],
        )
        metrics["n"] = float(len(samples))
        return metrics

    if task is Task.GROUNDING:
        return grounding_metrics(
            [p.bbox for p in predictions], [s.bbox for s in samples]
        )

    raise ValueError(f"no metric family registered for task {task}")


def _to_prediction(sample: Sample, raw: str, box_scale: str) -> Prediction:
    if sample.task is Task.GROUNDING:
        box = parse_bbox(raw, scale=box_scale)  # type: ignore[arg-type]
        return Prediction(
            sample_id=sample.sample_id,
            raw_text=raw,
            bbox=box,
            parse_ok=box is not None,
        )
    return Prediction(sample_id=sample.sample_id, raw_text=raw, answer=raw)


def run_benchmark(
    dataset: BenchmarkDataset,
    backend: VLMBackend,
    output_dir: Path | None = None,
    batch_size: int = 32,
    progress_every: int = 10,
) -> EvalResult:
    """Evaluate one benchmark split against one backend."""
    samples = dataset.load_subset()
    if not samples:
        raise ValueError(f"{dataset.config.name}: no samples loaded")

    requests = build_requests(samples)
    started = time.perf_counter()
    raw_outputs: list[str] = []

    total_batches = (len(requests) + batch_size - 1) // batch_size
    for index, start in enumerate(range(0, len(requests), batch_size), start=1):
        raw_outputs.extend(backend.generate(requests[start : start + batch_size]))
        if progress_every and index % progress_every == 0:
            elapsed = time.perf_counter() - started
            done = min(start + batch_size, len(requests))
            rate = done / elapsed if elapsed else 0.0
            print(
                f"  [{dataset.config.name}] batch {index}/{total_batches} "
                f"({done}/{len(requests)}) {rate:.1f} samples/s",
                flush=True,
            )

    duration = time.perf_counter() - started
    box_scale = dataset.config.extra.get("prediction_box_scale", "auto")
    predictions = [
        _to_prediction(sample, raw, str(box_scale))
        for sample, raw in zip(samples, raw_outputs, strict=True)
    ]

    result = EvalResult(
        benchmark=dataset.config.name,
        task=str(dataset.config.task),
        backend=backend.name,
        model=backend.config.model,
        num_samples=len(samples),
        metrics=score(samples, predictions),
        duration_s=duration,
        git_sha=git_sha(),
        config_hash=config_hash(
            {
                "benchmark": dataset.config.name,
                "adapter": dataset.config.adapter,
                "limit": dataset.config.limit,
                "seed": dataset.config.seed,
                "prompt_version": PROMPT_VERSION,
            }
        ),
        timestamp=datetime.now(UTC).isoformat(timespec="seconds"),
        predictions=predictions,
    )

    if output_dir is not None:
        write_artifacts(result, samples, output_dir)
    return result


def write_artifacts(
    result: EvalResult, samples: Sequence[Sample], output_dir: Path
) -> None:
    """Persist predictions and metrics next to each other."""
    output_dir.mkdir(parents=True, exist_ok=True)

    predictions_path = output_dir / "predictions.jsonl"
    with predictions_path.open("w", encoding="utf-8") as handle:
        for sample, prediction in zip(samples, result.predictions, strict=True):
            handle.write(
                json.dumps(
                    {
                        "sample_id": sample.sample_id,
                        "question": sample.question,
                        "reference": sample.answer,
                        "references": list(sample.references),
                        "reference_bbox": sample.bbox,
                        "qtype": sample.qtype,
                        "raw_text": prediction.raw_text,
                        "answer": prediction.answer,
                        "bbox": prediction.bbox,
                        "parse_ok": prediction.parse_ok,
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )

    (output_dir / "metrics.json").write_text(
        json.dumps(result.summary(), indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
