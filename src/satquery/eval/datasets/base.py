"""Benchmark adapter framework.

Public benchmark releases move their JSON keys between versions, so adapters resolve
every field through a candidate-key list that a YAML config can override. Run
``satquery bench validate`` against a real download once: it reports which keys were
found and which could not be mapped, and the fix is a config edit rather than a code
change.
"""

from __future__ import annotations

import json
import random
from abc import ABC, abstractmethod
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, ClassVar

import yaml

from satquery.schema import Sample, Task

_ADAPTERS: dict[str, type[BenchmarkDataset]] = {}


def register(name: str):
    """Class decorator adding an adapter to the registry."""

    def decorator(cls: type[BenchmarkDataset]) -> type[BenchmarkDataset]:
        _ADAPTERS[name] = cls
        return cls

    return decorator


def get_adapter(name: str) -> type[BenchmarkDataset]:
    if name not in _ADAPTERS:
        known = ", ".join(sorted(_ADAPTERS)) or "<none>"
        raise KeyError(f"unknown adapter '{name}'. Registered: {known}")
    return _ADAPTERS[name]


def available_adapters() -> list[str]:
    return sorted(_ADAPTERS)


@dataclass
class BenchmarkConfig:
    """Everything needed to turn a download on disk into ``Sample`` objects."""

    name: str
    adapter: str
    task: Task
    root: Path
    annotations: str = ""
    image_dir: str = ""
    image_suffix: str = ""
    fields: Mapping[str, str] = field(default_factory=dict)
    box_format: str = "xyxy"
    box_scale: str = "auto"
    limit: int | None = None
    seed: int = 1234
    extra: Mapping[str, Any] = field(default_factory=dict)

    @classmethod
    def from_yaml(cls, path: str | Path) -> BenchmarkConfig:
        data = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
        return cls.from_dict(data)

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> BenchmarkConfig:
        payload = dict(data)
        known = {f.name for f in cls.__dataclass_fields__.values()}
        extra = {k: v for k, v in payload.items() if k not in known}
        payload = {k: v for k, v in payload.items() if k in known}
        payload["root"] = Path(payload.get("root", "data"))
        payload["task"] = Task(payload["task"])
        payload.setdefault("extra", {})
        payload["extra"] = {**payload["extra"], **extra}
        return cls(**payload)

    @property
    def annotation_path(self) -> Path:
        return self.root / self.annotations

    @property
    def image_root(self) -> Path:
        return self.root / self.image_dir


class BenchmarkDataset(ABC):
    """Loads one benchmark split into the unified ``Sample`` schema."""

    def __init__(self, config: BenchmarkConfig) -> None:
        self.config = config

    #: Field name -> candidate keys, tried in order. Overridden per adapter.
    default_fields: ClassVar[Mapping[str, tuple[str, ...]]] = {}

    @abstractmethod
    def load(self) -> list[Sample]:
        """Return every sample in the split, before subsampling."""

    def load_subset(self) -> list[Sample]:
        """Load, then take a seeded random subset when ``limit`` is set.

        Seeded and sorted so every candidate model sees an identical subset --
        without that the bake-off comparison is meaningless.
        """
        samples = self.load()
        limit = self.config.limit
        if limit is None or limit >= len(samples):
            return samples
        rng = random.Random(self.config.seed)
        chosen = rng.sample(range(len(samples)), limit)
        return [samples[i] for i in sorted(chosen)]

    # -- helpers shared by adapters ---------------------------------------

    @property
    def image_root(self) -> Path:
        return self.config.image_root

    def field_keys(self, logical: str) -> tuple[str, ...]:
        """Candidate keys for a logical field, config override winning."""
        override = self.config.fields.get(logical)
        if override:
            return (override,)
        return self.default_fields.get(logical, (logical,))

    def pick(self, record: Mapping[str, Any], logical: str, default: Any = None) -> Any:
        for key in self.field_keys(logical):
            if key in record and record[key] is not None:
                return record[key]
        return default

    def require(self, record: Mapping[str, Any], logical: str) -> Any:
        value = self.pick(record, logical)
        if value is None:
            tried = ", ".join(self.field_keys(logical))
            raise KeyError(
                f"{self.config.name}: could not resolve field '{logical}' "
                f"(tried: {tried}). Available keys: {sorted(record)}. "
                f"Set fields.{logical} in the benchmark config."
            )
        return value

    def resolve_image(self, image_id: Any) -> Path:
        """Map an annotation's image id to a path on disk."""
        name = str(image_id)
        if self.config.image_suffix and not name.endswith(self.config.image_suffix):
            name = f"{name}{self.config.image_suffix}"
        return self.image_root / name

    def read_json(self, path: Path | None = None) -> Any:
        target = path or self.config.annotation_path
        if not target.exists():
            raise FileNotFoundError(
                f"{self.config.name}: annotations not found at {target}. "
                f"Check 'root' and 'annotations' in the benchmark config."
            )
        return json.loads(target.read_text(encoding="utf-8"))

    # -- introspection for `bench validate` --------------------------------

    def describe(self, probe: int = 200) -> dict[str, Any]:
        """Report what the adapter can see, without failing on a bad mapping."""
        report: dict[str, Any] = {
            "name": self.config.name,
            "adapter": self.config.adapter,
            "task": str(self.config.task),
            "annotations": str(self.config.annotation_path),
            "annotations_exist": self.config.annotation_path.exists(),
            "image_root": str(self.image_root),
            "image_root_exists": self.image_root.exists(),
        }
        try:
            samples = self.load()
        except Exception as exc:  # reported in the payload, never raised
            report["error"] = f"{type(exc).__name__}: {exc}"
            return report

        report["num_samples"] = len(samples)
        checked = samples[:probe]
        missing = [
            str(s.images[0].path) for s in checked if not s.images[0].path.exists()
        ]
        report["images_probed"] = len(checked)
        report["images_missing"] = len(missing)
        report["missing_examples"] = missing[:5]
        qtypes = sorted({s.qtype for s in samples if s.qtype})
        report["question_types"] = qtypes[:20]
        if samples:
            first = samples[0]
            report["example"] = {
                "sample_id": first.sample_id,
                "question": first.question,
                "answer": first.answer,
                "bbox": first.bbox,
                "images": [str(i.path.name) for i in first.images],
            }
        return report


def sniff_keys(records: Iterable[Mapping[str, Any]], limit: int = 50) -> list[str]:
    """Union of keys across the first ``limit`` records, for diagnostics."""
    keys: set[str] = set()
    for index, record in enumerate(records):
        if index >= limit:
            break
        keys.update(record)
    return sorted(keys)


def as_records(
    payload: Any, container_keys: Sequence[str] = ()
) -> list[dict[str, Any]]:
    """Coerce a loaded JSON payload into a list of records.

    Handles the three shapes these benchmarks ship in: a bare list, a dict wrapping
    a list under a known key, and a dict keyed by sample id.
    """
    if isinstance(payload, list):
        return [r for r in payload if isinstance(r, dict)]
    if isinstance(payload, dict):
        for key in container_keys:
            if isinstance(payload.get(key), list):
                return [r for r in payload[key] if isinstance(r, dict)]
        for value in payload.values():
            if isinstance(value, list) and value and isinstance(value[0], dict):
                return list(value)
        if all(isinstance(v, dict) for v in payload.values()):
            return [{"_key": k, **v} for k, v in payload.items()]
    raise TypeError(f"unsupported annotation payload of type {type(payload).__name__}")
