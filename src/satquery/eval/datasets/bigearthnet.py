"""BigEarthNet.txt bench split -- the cross-modal benchmark.

Optical-SAR joint analysis is one of the three mandatory capabilities and, until
this adapter, the only one with no score at all. ``eval/prompts.py`` records why:
no public benchmark covers it. RSVQA and VRSBench are single-image and CDVQA is
bi-temporal, so the cross-modal path could be demonstrated but never measured.

BigEarthNet.txt's ``bench`` split closes that. 15,029 annotations over 1,082
co-registered Sentinel-1 + Sentinel-2 pairs, manually verified by the publishers
rather than generated -- which is what makes it usable as a reference rather than
as more training data.

It reads the *prepared* JSONL rather than the published parquet, because the
pairs have to be rendered to PNG before a model can be shown them and
``scripts/prepare_bigearthnet.py`` already does that with the serving path's own
normalisation. Preparing the bench split with the same renderer the training
split uses also means a score here is not measuring a rendering difference.

**Caveat to state whenever this number is reported:** it is Sentinel at 10 m over
Europe, while the hidden evaluation set is Cartosat and RISAT over India. It
measures cross-modal *reasoning*, not performance on the evaluation domain.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, ClassVar

from satquery.eval.datasets.base import BenchmarkDataset, register
from satquery.schema import ImageRef, ImageRole, Modality, Sample, Task

#: Annotation types that are answerable as short-form VQA. Captioning and
#: bounding-box rows live in the same split but are scored by different metric
#: families, so mixing them into one config would average incomparable numbers.
DEFAULT_TYPES = ("binary", "mcq")


class BenchSplitError(ValueError):
    """The prepared file is not usable as a benchmark."""


@register("bigearthnet_bench")
class BigEarthNetBench(BenchmarkDataset):
    """Cross-modal VQA over co-registered optical and SAR pairs."""

    default_fields: ClassVar[dict[str, tuple[str, ...]]] = {}

    def _jsonl_path(self) -> Path:
        path = self.config.annotation_path
        if not path.is_file():
            raise FileNotFoundError(
                f"{self.config.name}: no prepared bench split at {path}. Build it "
                f"with 'python scripts/prepare_bigearthnet.py --split bench "
                f"--lmdb <store> --out {path.parent}'."
            )
        return path

    def _images(self, record: dict[str, Any], index: int) -> tuple[ImageRef, ImageRef]:
        """The optical and SAR renderings of one patch, in that order.

        Order is the contract, not a convenience: the cross-modal prompt tells
        the model that image 1 is optical and image 2 is SAR, so swapping them
        would invert every statement it makes about which sensor supports what.
        """
        images = list(record.get("images") or [])
        if len(images) != 2:
            raise BenchSplitError(
                f"{self.config.name}: record {index} has {len(images)} image(s); "
                f"a cross-modal pair needs exactly two"
            )
        optical, sar = (self.image_root.parent / name for name in images)
        # The same roles assign_roles() derives for a cross-modal pair at
        # inference, so a benchmark sample and an uploaded pair are the same
        # shape to everything downstream.
        return (
            ImageRef(optical, Modality.OPTICAL, ImageRole.OPTICAL),
            ImageRef(sar, Modality.SAR, ImageRole.SAR),
        )

    def load(self) -> list[Sample]:
        keep = {
            str(t).strip().lower()
            for t in self.config.extra.get("types", DEFAULT_TYPES)
        }

        samples: list[Sample] = []
        with self._jsonl_path().open(encoding="utf-8") as handle:
            for index, line in enumerate(handle):
                line = line.strip()
                if not line:
                    continue
                record = json.loads(line)

                task_type = str(record.get("task", "")).strip().lower()
                if keep and task_type not in keep:
                    continue

                # Evidence baked into an evaluation question would measure the
                # preamble rather than the model. Preparation refuses to do it
                # on this split; a file that predates that refusal is rejected
                # loudly rather than quietly inflating the score.
                if str(record.get("evidence_kind", "none")) != "none":
                    raise BenchSplitError(
                        f"{self.config.name}: record {index} carries a synthesised "
                        f"evidence preamble. Re-prepare the bench split -- "
                        f"evaluation questions must be bare."
                    )

                turns = record.get("conversations") or []
                if len(turns) < 2:
                    continue

                samples.append(
                    Sample(
                        sample_id=str(record.get("id", f"{self.config.name}-{index}")),
                        task=Task.CROSSMODAL_VQA,
                        images=self._images(record, index),
                        question=str(turns[0].get("value", "")).strip(),
                        answer=str(turns[1].get("value", "")).strip(),
                        # The annotation type doubles as the question type, which
                        # is what AA is averaged over. Binary and mcq are heavily
                        # imbalanced in this split, so OA alone would flatter.
                        qtype=task_type or None,
                    )
                )
        return samples
