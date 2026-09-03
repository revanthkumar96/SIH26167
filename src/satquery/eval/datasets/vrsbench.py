"""VRSBench adapters: VQA, captioning and referring (grounding).

VRSBench ships one JSON per task, each a list of records keyed by image. Release
versions differ in whether the answer key is ``ground_truth``, ``answer`` or
``caption``, so every field is resolved through candidate keys.
"""

from __future__ import annotations

import re
from collections.abc import Mapping
from typing import Any, ClassVar

from satquery.eval.datasets.base import (
    BenchmarkDataset,
    as_records,
    register,
)
from satquery.eval.normalize import normalize_box
from satquery.schema import BBox, ImageRef, Modality, Sample, Task

_NUMBERS = re.compile(r"-?\d+(?:\.\d+)?")
_CONTAINERS = ("annotations", "data", "questions", "items")


def coerce_box(value: Any) -> tuple[float, float, float, float] | None:
    """Accept a box as a list, a tuple, or a delimited string."""
    if value is None:
        return None
    if isinstance(value, (list, tuple)) and len(value) >= 4:
        try:
            return tuple(float(v) for v in value[:4])  # type: ignore[return-value]
        except (TypeError, ValueError):
            return None
    if isinstance(value, str):
        found = _NUMBERS.findall(value)
        if len(found) >= 4:
            return tuple(float(v) for v in found[:4])  # type: ignore[return-value]
    return None


class _VRSBenchBase(BenchmarkDataset):
    """Shared record loading for the three VRSBench splits."""

    def records(self) -> list[dict[str, Any]]:
        return as_records(self.read_json(), _CONTAINERS)

    def image_for(self, record: Mapping[str, Any]) -> ImageRef:
        image_id = self.require(record, "image_id")
        return ImageRef(
            path=self.resolve_image(image_id),
            modality=Modality.RGB,
        )


@register("vrsbench_vqa")
class VRSBenchVQA(_VRSBenchBase):
    default_fields: ClassVar[dict[str, tuple[str, ...]]] = {
        "image_id": ("image_id", "image", "img_id", "filename", "image_name"),
        "question": ("question", "text", "query"),
        "answer": ("ground_truth", "answer", "gt", "label"),
        "qtype": ("type", "question_type", "category"),
    }

    def load(self) -> list[Sample]:
        samples: list[Sample] = []
        for index, record in enumerate(self.records()):
            samples.append(
                Sample(
                    sample_id=f"{self.config.name}-{index}",
                    task=Task.VQA,
                    images=(self.image_for(record),),
                    question=str(self.require(record, "question")),
                    answer=str(self.require(record, "answer")),
                    qtype=(
                        str(self.pick(record, "qtype"))
                        if self.pick(record, "qtype")
                        else None
                    ),
                )
            )
        return samples


@register("vrsbench_caption")
class VRSBenchCaption(_VRSBenchBase):
    default_fields: ClassVar[dict[str, tuple[str, ...]]] = {
        "image_id": ("image_id", "image", "img_id", "filename", "image_name"),
        "answer": ("caption", "ground_truth", "captions", "text", "description"),
    }

    def load(self) -> list[Sample]:
        samples: list[Sample] = []
        for index, record in enumerate(self.records()):
            raw = self.require(record, "answer")
            references = (
                tuple(str(r) for r in raw)
                if isinstance(raw, (list, tuple))
                else (str(raw),)
            )
            samples.append(
                Sample(
                    sample_id=f"{self.config.name}-{index}",
                    task=Task.CAPTION,
                    images=(self.image_for(record),),
                    answer=references[0],
                    references=references,
                )
            )
        return samples


@register("vrsbench_referring")
class VRSBenchReferring(_VRSBenchBase):
    """Text-guided region grounding.

    Ground-truth boxes are in pixel coordinates in most releases, so
    ``box_scale: pixel`` plus ``extra.image_size`` -- or ``box_scale: auto`` -- is
    usually what the config wants.
    """

    default_fields: ClassVar[dict[str, tuple[str, ...]]] = {
        "image_id": ("image_id", "image", "img_id", "filename", "image_name"),
        "question": ("question", "expression", "referring", "text", "query"),
        "bbox": ("ground_truth", "bbox", "box", "gt_box", "answer"),
    }

    def _image_size(self) -> tuple[int, int] | None:
        size = self.config.extra.get("image_size")
        if isinstance(size, (list, tuple)) and len(size) == 2:
            return (int(size[0]), int(size[1]))
        if isinstance(size, int):
            return (size, size)
        return None

    def load(self) -> list[Sample]:
        samples: list[Sample] = []
        image_size = self._image_size()
        for index, record in enumerate(self.records()):
            raw_box = coerce_box(self.require(record, "bbox"))
            box: BBox | None = None
            if raw_box is not None:
                box = normalize_box(
                    raw_box,
                    image_size=image_size,
                    box_format=self.config.box_format,  # type: ignore[arg-type]
                    scale=self.config.box_scale,  # type: ignore[arg-type]
                )
            samples.append(
                Sample(
                    sample_id=f"{self.config.name}-{index}",
                    task=Task.GROUNDING,
                    images=(self.image_for(record),),
                    question=str(self.require(record, "question")),
                    bbox=box,
                )
            )
        return samples
