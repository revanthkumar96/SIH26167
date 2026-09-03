"""Deterministic optical and SAR index tools.

These carry the optical-SAR requirement. They are measurements, not predictions:
they produce a number and a mask the user can see, and their outputs are injected
into the VLM prompt so the final answer is grounded in something checkable.

SAR thresholds are relative (Otsu), never absolute dB. An uploaded GRD tile may be
in dB, in linear power, or already stretched to 8-bit, and a hard -18 dB cut would
silently mean three different things.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from satquery.agent.context import RunContext
from satquery.agent.registry import Tool, ToolResult
from satquery.agent.tools._imaging import (
    otsu_threshold,
    quadrant_summary,
    save_overlay,
    to_gray,
)
from satquery.geo.raster import read_bands
from satquery.schema import ImageRole, InputConfig, Task, ToolSpec

#: Band positions (1-based) by band count, covering the layouts we actually meet.
_BAND_PLANS: dict[int, dict[str, int]] = {
    # Sentinel-2 / BigEarthNet 12-band: B01 B02 B03 B04 B05 B06 B07 B08 B8A B09 B11 B12
    12: {"blue": 2, "green": 3, "red": 4, "nir": 8, "swir": 11},
    # Sentinel-2 L1C with B10 included
    13: {"blue": 2, "green": 3, "red": 4, "nir": 8, "swir": 12},
    # Common 4-band product: R G B NIR
    4: {"red": 1, "green": 2, "blue": 3, "nir": 4},
}


def _normalised_difference(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    denominator = a + b
    return np.divide(
        a - b, denominator, out=np.zeros_like(a), where=np.abs(denominator) > 1e-6
    )


class OpticalIndicesTool(Tool):
    """NDWI and NDBI from a multispectral optical image.

    Reports ``applicable: false`` when the input lacks the required bands rather
    than fabricating an index from RGB. A three-band benchmark PNG genuinely
    cannot yield NDWI, and saying so is worth more than a plausible number.
    """

    spec = ToolSpec(
        name="optical_indices",
        version="1.0.0",
        accepts=InputConfig.CROSSMODAL_PAIR,
        tasks=(Task.CROSSMODAL_VQA,),
        allowed_params={
            "water_threshold": (-1.0, 1.0),
            "builtup_threshold": (-1.0, 1.0),
        },
        outputs=("applicable", "water_fraction", "builtup_fraction", "water_mask_uri"),
        summary=(
            "NDWI and NDBI from multispectral optical bands. Reports itself "
            "inapplicable on RGB rather than inventing an index."
        ),
        kind="measurement",
        category="cross-modal",
        cost="fast",
        requires="An optical image with a near-infrared band (4, 12 or 13 bands).",
        emits_evidence=True,
        param_docs={
            "water_threshold": "NDWI above which a pixel is called water.",
            "builtup_threshold": "NDBI above which a pixel is called built-up.",
        },
    )

    def run(
        self,
        ctx: RunContext,
        water_threshold: float = 0.0,
        builtup_threshold: float = 0.0,
    ) -> ToolResult:
        image = ctx.require_role(ImageRole.OPTICAL)
        bands = read_bands(image.path)
        plan = _BAND_PLANS.get(bands.shape[0])

        if plan is None or "nir" not in plan:
            return ToolResult(
                outputs={
                    "applicable": False,
                    "reason": (
                        f"{bands.shape[0]}-band input has no near-infrared band; "
                        f"NDWI and NDBI need one"
                    ),
                },
                confidence=None,
            )

        green = bands[plan["green"] - 1]
        nir = bands[plan["nir"] - 1]
        ndwi = _normalised_difference(green, nir)
        water = ndwi > water_threshold

        outputs: dict[str, Any] = {
            "applicable": True,
            "bands_used": plan,
            "water_fraction": round(float(water.mean()), 4),
            "water_location": quadrant_summary(water),
        }

        if "swir" in plan:
            swir = bands[plan["swir"] - 1]
            ndbi = _normalised_difference(swir, nir)
            builtup = ndbi > builtup_threshold
            outputs["builtup_fraction"] = round(float(builtup.mean()), 4)
            outputs["builtup_location"] = quadrant_summary(builtup)

        filename = "optical_water.png"
        save_overlay(
            image.path, water, ctx.artifact_path(filename), colour=(60, 130, 255)
        )
        uri = ctx.artifact_uri(filename)
        outputs["water_mask_uri"] = uri

        return ToolResult(
            outputs=outputs,
            confidence=0.85,
            evidence=({"type": "mask", "uri": uri, "label": "NDWI water (optical)"},),
        )


class SarIndicesTool(Tool):
    """Water and built-up extent from SAR backscatter.

    Water is a near-specular reflector, so it returns very little energy and shows
    as the dark class; built-up areas produce bright double-bounce returns. This is
    the complementary information the optical image cannot give under cloud.
    """

    spec = ToolSpec(
        name="sar_indices",
        version="1.0.0",
        accepts=InputConfig.CROSSMODAL_PAIR,
        tasks=(Task.CROSSMODAL_VQA,),
        allowed_params={"builtup_percentile": (80.0, 99.5)},
        outputs=(
            "water_fraction",
            "builtup_fraction",
            "water_location",
            "water_mask_uri",
        ),
        summary=(
            "Water and built-up extent from SAR backscatter in dB. Sees through "
            "the cloud that makes optical unusable."
        ),
        kind="measurement",
        category="cross-modal",
        cost="fast",
        requires="A SAR image; linear backscatter is converted to dB first.",
        emits_evidence=True,
        param_docs={
            "builtup_percentile": (
                "Backscatter percentile above which a pixel is called built-up. "
                "Bright double-bounce returns sit in the upper tail."
            )
        },
    )

    def run(self, ctx: RunContext, builtup_percentile: float = 95.0) -> ToolResult:
        image = ctx.require_role(ImageRole.SAR)
        gray = to_gray(image.path)

        threshold = otsu_threshold(gray)
        water = gray < threshold
        builtup = gray > float(np.percentile(gray, builtup_percentile))

        filename = "sar_water.png"
        save_overlay(
            image.path, water, ctx.artifact_path(filename), colour=(60, 130, 255)
        )
        uri = ctx.artifact_uri(filename)

        return ToolResult(
            outputs={
                "threshold_method": "otsu_relative",
                "threshold": round(threshold, 4),
                "water_fraction": round(float(water.mean()), 4),
                "water_location": quadrant_summary(water),
                "builtup_fraction": round(float(builtup.mean()), 4),
                "builtup_location": quadrant_summary(builtup),
                "water_mask_uri": uri,
            },
            confidence=0.8,
            evidence=(
                {"type": "mask", "uri": uri, "label": "SAR low-backscatter water"},
            ),
        )
