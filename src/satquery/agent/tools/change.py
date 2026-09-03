"""Bi-temporal change mask.

A deterministic difference-and-threshold baseline that works today with no trained
weights. The Siamese CNN from the ML plan drops in behind this same ``ToolSpec``
later: same name, same outputs, bumped version. Nothing upstream changes.

Being deterministic is not only expedient. The mask is a measurement the user can
inspect, and its statistics are injected into the change-VQA prompt, so the model
answers from something checkable rather than from two pictures alone.
"""

from __future__ import annotations

import numpy as np

from satquery.agent.context import RunContext
from satquery.agent.registry import Tool, ToolResult
from satquery.agent.tools._imaging import (
    otsu_threshold,
    quadrant_summary,
    resize_to,
    save_overlay,
    to_gray,
)
from satquery.schema import ImageRole, InputConfig, Task, ToolSpec

#: Below this changed fraction the scene is described as effectively unchanged.
_QUIET_FRACTION = 0.02


class ChangeMaskTool(Tool):
    """Absolute-difference change detection between two co-registered dates."""

    spec = ToolSpec(
        name="change_mask",
        version="1.0.0",
        accepts=InputConfig.BITEMPORAL_PAIR,
        tasks=(Task.CHANGE_VQA, Task.CHANGE_CAPTION),
        allowed_params={"threshold": (0.02, 0.9), "min_area_frac": (0.0, 0.2)},
        outputs=(
            "changed_area_frac",
            "change_location",
            "direction",
            "mask_uri",
        ),
        summary=(
            "Absolute-difference change detection between two co-registered dates, "
            "thresholded by Otsu unless told otherwise."
        ),
        kind="measurement",
        category="change",
        cost="fast",
        requires="Two images of the same ground, ideally same CRS and extent.",
        emits_evidence=True,
        param_docs={
            "threshold": (
                "Difference above which a pixel counts as changed. Left unset, "
                "Otsu picks it from the histogram."
            ),
            "min_area_frac": (
                "Discard the mask entirely below this changed fraction, to avoid "
                "reporting speckle as change."
            ),
        },
    )

    def run(
        self,
        ctx: RunContext,
        threshold: float | None = None,
        min_area_frac: float = 0.0,
    ) -> ToolResult:
        before_image = ctx.require_role(ImageRole.BEFORE)
        after_image = ctx.require_role(ImageRole.AFTER)

        before = to_gray(before_image.path)
        after = to_gray(after_image.path)
        if before.shape != after.shape:
            # Inputs already passed the compatibility check, which warns rather than
            # blocks on a size mismatch, so align here instead of failing the run.
            after = resize_to(after, before.shape)

        difference = np.abs(after - before)
        cut = otsu_threshold(difference) if threshold is None else threshold
        mask = difference > cut

        changed = float(mask.mean())
        if changed < min_area_frac:
            mask = np.zeros_like(mask)
            changed = 0.0

        # Sign of the mean shift tells brightening from darkening, which maps onto
        # "built-up increased" vs "vegetation or water gained" often enough to be
        # worth handing to the model as a hint rather than a conclusion.
        delta = float((after - before)[mask].mean()) if mask.any() else 0.0
        if changed < _QUIET_FRACTION:
            direction = "no significant change"
        elif delta > 0.02:
            direction = "brightened"
        elif delta < -0.02:
            direction = "darkened"
        else:
            direction = "mixed"

        filename = "change_mask.png"
        save_overlay(
            after_image.path, mask, ctx.artifact_path(filename), colour=(255, 72, 72)
        )
        uri = ctx.artifact_uri(filename)

        # Confidence tracks separability: a decisive Otsu split is trustworthy, a
        # marginal one is not.
        separation = float(np.abs(difference.mean() - cut)) if difference.size else 0.0
        confidence = round(min(0.5 + separation * 2.0, 0.9), 3)

        return ToolResult(
            outputs={
                "threshold_method": "otsu" if threshold is None else "explicit",
                "threshold": round(float(cut), 4),
                "changed_area_frac": round(changed, 4),
                "change_location": quadrant_summary(mask),
                "direction": direction,
                "mean_intensity_delta": round(delta, 4),
                "mask_uri": uri,
            },
            confidence=confidence,
            evidence=({"type": "mask", "uri": uri, "label": "detected change"},),
        )
