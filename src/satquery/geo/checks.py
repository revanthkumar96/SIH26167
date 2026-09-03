"""Input compatibility checking and configuration inference.

The problem statement makes this a scored duty of the controller: "check the number,
modality, format, metadata, and compatibility of the input images". So the result is
a structured ``InputCheck`` that goes straight into the execution trace and onto the
screen, not an internal assertion.

Nothing here guesses silently: every conclusion lands in ``checks_passed`` or
``warnings`` so a user can see why the system routed the way it did.
"""

from __future__ import annotations

from collections.abc import Sequence

from satquery.geo.raster import RasterInfo, filename_hint
from satquery.schema import ImageRole, InputConfig, Modality

#: Extent IoU above which two rasters are treated as covering the same ground.
COREGISTRATION_IOU = 0.90
#: Relative GSD difference tolerated before the pair is flagged as mismatched.
GSD_TOLERANCE = 0.05


def infer_modality(info: RasterInfo) -> Modality:
    """Classify a raster as SAR, optical/multispectral, or plain RGB.

    Filename hints win over band counts: a user who names a file ``*_VV.tif`` is
    telling us something the header often does not.
    """
    hint = filename_hint(info.path)
    if hint == "sar":
        return Modality.SAR
    if hint == "optical":
        return Modality.OPTICAL

    if info.band_count <= 2:
        # One or two bands with a floating dtype is the classic GRD backscatter
        # signature (VV, or VV+VH, in dB or linear power).
        if info.dtype.startswith("float") or info.band_count == 2:
            return Modality.SAR
        return Modality.OPTICAL if info.georeferenced else Modality.RGB
    if info.band_count >= 4:
        return Modality.OPTICAL
    return Modality.OPTICAL if info.georeferenced else Modality.RGB


def _extent_iou(a: RasterInfo, b: RasterInfo) -> float | None:
    if not (a.bounds and b.bounds):
        return None
    ax1, ay1, ax2, ay2 = a.bounds
    bx1, by1, bx2, by2 = b.bounds
    inter_w = min(ax2, bx2) - max(ax1, bx1)
    inter_h = min(ay2, by2) - max(ay1, by1)
    if inter_w <= 0 or inter_h <= 0:
        return 0.0
    intersection = inter_w * inter_h
    union = (ax2 - ax1) * (ay2 - ay1) + (bx2 - bx1) * (by2 - by1) - intersection
    return intersection / union if union > 0 else 0.0


def check_pair(a: RasterInfo, b: RasterInfo) -> tuple[bool, list[str], list[str]]:
    """Compatibility of two images. Returns (coregistered, passed, warnings)."""
    passed: list[str] = []
    warnings: list[str] = []

    if a.size == b.size:
        passed.append("size_match")
    else:
        warnings.append(
            f"pixel dimensions differ: {a.width}x{a.height} vs {b.width}x{b.height}"
        )

    if a.georeferenced and b.georeferenced:
        if a.crs == b.crs:
            passed.append("crs_match")
        else:
            warnings.append(f"CRS differs: {a.crs} vs {b.crs}")

        iou = _extent_iou(a, b)
        if iou is None:
            warnings.append("extent could not be compared")
        elif iou >= COREGISTRATION_IOU:
            passed.append("extent_overlap")
        elif iou > 0:
            warnings.append(f"extents only partially overlap (IoU {iou:.2f})")
        else:
            warnings.append("extents do not overlap: images cover different ground")

        if a.gsd_m and b.gsd_m:
            relative = abs(a.gsd_m - b.gsd_m) / max(a.gsd_m, b.gsd_m)
            if relative <= GSD_TOLERANCE:
                passed.append("gsd_match")
            else:
                warnings.append(
                    f"ground sample distance differs: {a.gsd_m:.2f} m vs {b.gsd_m:.2f} m"
                )
    else:
        warnings.append(
            "one or both inputs are not georeferenced; "
            "co-registration assumed from identical pixel dimensions"
        )

    coregistered = "crs_match" in passed and "extent_overlap" in passed
    if not coregistered and "size_match" in passed and not a.georeferenced:
        # Benchmark PNG pairs carry no geotransform but are co-registered by
        # construction. Accept them, and say so rather than claiming a check passed.
        coregistered = True

    return coregistered, passed, warnings


def infer_input_config(
    infos: Sequence[RasterInfo],
) -> tuple[InputConfig, list[Modality]]:
    """Derive the input configuration from the uploaded set.

    Never declared by the user: two images of different modality are a cross-modal
    pair, two of the same modality are bi-temporal.
    """
    if not infos:
        raise ValueError("at least one image is required")

    modalities = [infer_modality(info) for info in infos]

    if len(infos) == 1:
        return InputConfig.SINGLE, modalities
    if len(infos) > 2:
        raise ValueError(
            f"{len(infos)} images supplied; the supported configurations are one "
            f"image, a bi-temporal pair, or a co-registered optical-SAR pair"
        )

    has_sar = Modality.SAR in modalities
    has_optical = any(m is not Modality.SAR for m in modalities)
    if has_sar and has_optical:
        return InputConfig.CROSSMODAL_PAIR, modalities
    return InputConfig.BITEMPORAL_PAIR, modalities


def assign_roles(
    config: InputConfig, modalities: Sequence[Modality]
) -> list[ImageRole]:
    """Give each image its role, so tools and prompts can reference them by meaning."""
    if config is InputConfig.SINGLE:
        return [ImageRole.SINGLE]
    if config is InputConfig.CROSSMODAL_PAIR:
        return [
            ImageRole.SAR if m is Modality.SAR else ImageRole.OPTICAL
            for m in modalities
        ]
    # Bi-temporal: upload order is acquisition order.
    return [ImageRole.BEFORE, ImageRole.AFTER]
