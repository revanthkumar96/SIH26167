"""CDVQA adapter -- change-based VQA over bi-temporal pairs.

CDVQA is built on the SECOND change-detection dataset, whose two dates live in
parallel directories (``im1/`` and ``im2/``) sharing a filename. Records therefore
usually carry one image name rather than two paths, though some releases name both
explicitly; both shapes are handled.

This is the only benchmark in the bake-off that exercises multi-image input, which
makes it the sharpest test of whether a candidate base model can reason over pairs
at all.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any, ClassVar

from satquery.eval.datasets.base import BenchmarkDataset, as_records, register
from satquery.schema import ImageRef, ImageRole, Modality, Sample, Task

_CONTAINERS = ("annotations", "data", "questions", "items")


@register("cdvqa")
class CDVQA(BenchmarkDataset):
    default_fields: ClassVar[dict[str, tuple[str, ...]]] = {
        "image_id": ("img_name", "image", "image_id", "filename", "name"),
        "image_before": ("img1", "image1", "im1", "before"),
        "image_after": ("img2", "image2", "im2", "after"),
        "question": ("question", "text", "query"),
        "answer": ("answer", "ground_truth", "gt", "label"),
        "qtype": ("type", "question_type", "category"),
    }

    def _pair_dirs(self) -> tuple[str, str]:
        before = str(self.config.extra.get("before_dir", "im1"))
        after = str(self.config.extra.get("after_dir", "im2"))
        return before, after

    def _images(self, record: Mapping[str, Any]) -> tuple[ImageRef, ImageRef]:
        before_dir, after_dir = self._pair_dirs()
        explicit_before = self.pick(record, "image_before")
        explicit_after = self.pick(record, "image_after")

        if explicit_before and explicit_after:
            before_path = self.image_root / str(explicit_before)
            after_path = self.image_root / str(explicit_after)
        else:
            name = str(self.require(record, "image_id"))
            if self.config.image_suffix and not name.endswith(self.config.image_suffix):
                name = f"{name}{self.config.image_suffix}"
            before_path = self.image_root / before_dir / name
            after_path = self.image_root / after_dir / name

        return (
            ImageRef(before_path, Modality.OPTICAL, ImageRole.BEFORE),
            ImageRef(after_path, Modality.OPTICAL, ImageRole.AFTER),
        )

    def load(self) -> list[Sample]:
        samples: list[Sample] = []
        for index, record in enumerate(as_records(self.read_json(), _CONTAINERS)):
            samples.append(
                Sample(
                    sample_id=f"{self.config.name}-{index}",
                    task=Task.CHANGE_VQA,
                    images=self._images(record),
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
