"""Core types shared by the eval harness, the agent controller and the API.

These are the frozen contracts described in ``docs/ARCHITECTURE.md``. Nothing here
imports torch, so the module stays importable in CI and on CPU-only machines.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from enum import StrEnum
from pathlib import Path
from typing import Any

BBox = tuple[float, float, float, float]
"""Axis-aligned box as ``(x1, y1, x2, y2)`` normalised to ``[0, 1]``."""


class Task(StrEnum):
    """A unit of work the controller can route to exactly one tool chain."""

    VQA = "vqa"
    CAPTION = "caption"
    GROUNDING = "grounding"
    CHANGE_VQA = "change_vqa"
    CHANGE_CAPTION = "change_caption"
    CROSSMODAL_VQA = "crossmodal_vqa"


class Modality(StrEnum):
    """Sensor family of a single input image."""

    OPTICAL = "optical"
    SAR = "sar"
    RGB = "rgb"


class ImageRole(StrEnum):
    """What an image means within its input configuration."""

    SINGLE = "single"
    BEFORE = "before"
    AFTER = "after"
    OPTICAL = "optical"
    SAR = "sar"


class InputConfig(StrEnum):
    """Derived from the uploaded set, never declared by the user."""

    SINGLE = "single"
    BITEMPORAL_PAIR = "bitemporal_pair"
    CROSSMODAL_PAIR = "crossmodal_pair"


#: Which input configuration each task consumes.
TASK_INPUT_CONFIG: Mapping[Task, InputConfig] = {
    Task.VQA: InputConfig.SINGLE,
    Task.CAPTION: InputConfig.SINGLE,
    Task.GROUNDING: InputConfig.SINGLE,
    Task.CHANGE_VQA: InputConfig.BITEMPORAL_PAIR,
    Task.CHANGE_CAPTION: InputConfig.BITEMPORAL_PAIR,
    Task.CROSSMODAL_VQA: InputConfig.CROSSMODAL_PAIR,
}


@dataclass(frozen=True, slots=True)
class ImageRef:
    """A pointer to one image plus the metadata the controller checks."""

    path: Path
    modality: Modality = Modality.RGB
    role: ImageRole = ImageRole.SINGLE

    def __post_init__(self) -> None:
        if not isinstance(self.path, Path):
            object.__setattr__(self, "path", Path(self.path))


@dataclass(frozen=True, slots=True)
class Sample:
    """One evaluated example, normalised across every benchmark.

    ``references`` carries the multiple ground-truth captions that captioning
    metrics need; single-answer tasks leave it empty and use ``answer``.
    """

    sample_id: str
    task: Task
    images: tuple[ImageRef, ...]
    question: str | None = None
    answer: str | None = None
    references: tuple[str, ...] = ()
    bbox: BBox | None = None
    qtype: str | None = None
    meta: Mapping[str, Any] = field(default_factory=dict)

    @property
    def input_config(self) -> InputConfig:
        return TASK_INPUT_CONFIG[self.task]

    @property
    def gold_texts(self) -> tuple[str, ...]:
        """References if present, else the single answer, else empty."""
        if self.references:
            return self.references
        return (self.answer,) if self.answer is not None else ()


@dataclass(frozen=True, slots=True)
class Prediction:
    """A model output, kept alongside the raw text so parsing stays auditable."""

    sample_id: str
    raw_text: str
    answer: str | None = None
    bbox: BBox | None = None
    latency_s: float = 0.0
    parse_ok: bool = True


@dataclass(frozen=True, slots=True)
class GenerationRequest:
    """What a backend needs to produce one prediction."""

    sample_id: str
    prompt: str
    images: tuple[Path, ...]
    max_new_tokens: int = 64


@dataclass(frozen=True, slots=True)
class ToolSpec:
    """A declarative registry entry.

    ``allowed_params`` is enforced at call time: the problem statement permits the
    agent to configure only sanctioned parameters, so out-of-range values are
    rejected rather than trusted.
    """

    name: str
    version: str
    accepts: InputConfig
    tasks: tuple[Task, ...]
    allowed_params: Mapping[str, Any] = field(default_factory=dict)
    outputs: tuple[str, ...] = ()

    def validate_params(self, params: Mapping[str, Any]) -> list[str]:
        """Return a list of human-readable violations; empty means valid."""
        errors: list[str] = []
        for key, value in params.items():
            if key not in self.allowed_params:
                errors.append(f"{self.name}: parameter '{key}' is not permitted")
                continue
            spec = self.allowed_params[key]
            if isinstance(spec, tuple) and len(spec) == 2:
                low, high = spec
                if not (low <= value <= high):
                    errors.append(
                        f"{self.name}: '{key}'={value} outside [{low}, {high}]"
                    )
            elif isinstance(spec, (set, frozenset)) and value not in spec:
                errors.append(f"{self.name}: '{key}'={value} not in {sorted(spec)}")
        return errors


@dataclass(frozen=True, slots=True)
class TraceStep:
    """One observable execution step. This is scored, so it is a product surface."""

    step: int
    tool: str
    version: str
    params: Mapping[str, Any] = field(default_factory=dict)
    outputs: Mapping[str, Any] = field(default_factory=dict)
    adapter: str | None = None
    confidence: float | None = None
    duration_ms: int = 0


@dataclass(frozen=True, slots=True)
class InputCheck:
    """Result of validating the uploaded image set."""

    config: InputConfig
    images: Sequence[Mapping[str, Any]] = ()
    coregistered: bool = False
    checks_passed: tuple[str, ...] = ()
    warnings: tuple[str, ...] = ()


@dataclass(frozen=True, slots=True)
class ExecutionTrace:
    """The auditable summary returned with every answer."""

    run_id: str
    query: str
    input_check: InputCheck
    routed_task: Task
    steps: tuple[TraceStep, ...] = ()
    answer: str = ""
    evidence: tuple[Mapping[str, Any], ...] = ()
    confidence: float | None = None
    duration_ms: int = 0
