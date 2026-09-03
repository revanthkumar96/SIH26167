"""Grounding metrics: IoU, Acc@IoU thresholds and mIoU.

A prediction that could not be parsed into a box counts as IoU 0 rather than being
excluded, so the score reflects the model's real end-to-end usability.
"""

from __future__ import annotations

from collections.abc import Sequence

from satquery.schema import BBox

DEFAULT_THRESHOLDS = (0.25, 0.5)


def iou(box_a: BBox | None, box_b: BBox | None) -> float:
    """Intersection over union of two unit-normalised xyxy boxes."""
    if box_a is None or box_b is None:
        return 0.0

    ax1, ay1, ax2, ay2 = box_a
    bx1, by1, bx2, by2 = box_b

    inter_w = min(ax2, bx2) - max(ax1, bx1)
    inter_h = min(ay2, by2) - max(ay1, by1)
    if inter_w <= 0 or inter_h <= 0:
        return 0.0

    intersection = inter_w * inter_h
    union = (ax2 - ax1) * (ay2 - ay1) + (bx2 - bx1) * (by2 - by1) - intersection
    return intersection / union if union > 0 else 0.0


def grounding_metrics(
    predictions: Sequence[BBox | None],
    references: Sequence[BBox | None],
    thresholds: Sequence[float] = DEFAULT_THRESHOLDS,
) -> dict[str, float]:
    """mIoU plus Acc@IoU at each threshold, and the parse rate."""
    if len(predictions) != len(references):
        raise ValueError("predictions and references must be the same length")
    if not predictions:
        return {"miou": 0.0, "n": 0.0, "parse_rate": 0.0}

    scores = [iou(pred, ref) for pred, ref in zip(predictions, references, strict=True)]
    parsed = sum(1 for pred in predictions if pred is not None)

    results: dict[str, float] = {
        "miou": sum(scores) / len(scores),
        "n": float(len(scores)),
        "parse_rate": parsed / len(predictions),
    }
    for threshold in thresholds:
        hits = sum(1 for score in scores if score >= threshold)
        results[f"acc@{threshold:g}"] = hits / len(scores)
    return results
