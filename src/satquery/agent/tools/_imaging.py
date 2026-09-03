"""Numpy helpers shared by the deterministic specialist tools."""

from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image

from satquery.geo.raster import read_bands, to_rgb8


def to_gray(path: str | Path) -> np.ndarray:
    """Percentile-stretched single-channel view of any raster, in ``[0, 1]``.

    Stretching before comparison matters: SAR arrives in dB or linear power and
    optical in uint16, so absolute thresholds are meaningless across inputs.
    """
    rgb = to_rgb8(read_bands(path)).astype(np.float32)
    gray = rgb @ np.array([0.299, 0.587, 0.114], dtype=np.float32)
    return gray / 255.0


def otsu_threshold(values: np.ndarray, bins: int = 256) -> float:
    """Otsu's between-class variance threshold over values in ``[0, 1]``."""
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        return 0.5
    hist, edges = np.histogram(finite, bins=bins, range=(0.0, 1.0))
    hist = hist.astype(np.float64)
    total = hist.sum()
    if total == 0:
        return 0.5

    weight_bg = np.cumsum(hist)
    weight_fg = total - weight_bg
    centres = (edges[:-1] + edges[1:]) / 2.0
    cumulative = np.cumsum(hist * centres)
    mean_total = cumulative[-1]

    valid = (weight_bg > 0) & (weight_fg > 0)
    if not valid.any():
        return 0.5

    mean_bg = np.divide(
        cumulative, weight_bg, out=np.zeros_like(cumulative), where=valid
    )
    mean_fg = np.divide(
        mean_total - cumulative, weight_fg, out=np.zeros_like(cumulative), where=valid
    )
    between = weight_bg * weight_fg * (mean_bg - mean_fg) ** 2
    between[~valid] = -1.0
    return float(centres[int(np.argmax(between))])


def resize_to(mask: np.ndarray, shape: tuple[int, int]) -> np.ndarray:
    """Nearest-neighbour resize of a boolean mask to ``(height, width)``."""
    if mask.shape == shape:
        return mask
    rows = np.clip(
        (np.arange(shape[0]) * mask.shape[0] // shape[0]), 0, mask.shape[0] - 1
    )
    cols = np.clip(
        (np.arange(shape[1]) * mask.shape[1] // shape[1]), 0, mask.shape[1] - 1
    )
    return mask[np.ix_(rows, cols)]


def save_overlay(
    base_path: str | Path,
    mask: np.ndarray,
    out_path: str | Path,
    colour: tuple[int, int, int] = (255, 64, 64),
    alpha: float = 0.45,
    max_side: int = 1024,
) -> Path:
    """Render a boolean mask over its source image as visual evidence."""
    rgb = to_rgb8(read_bands(base_path))

    aligned = resize_to(mask.astype(bool), rgb.shape[:2])
    tint = np.array(colour, dtype=np.float32)
    blended = rgb.astype(np.float32)
    blended[aligned] = (1 - alpha) * blended[aligned] + alpha * tint

    image = Image.fromarray(blended.clip(0, 255).astype(np.uint8), mode="RGB")
    if max(image.size) > max_side:
        scale = max_side / max(image.size)
        image = image.resize(
            (max(int(image.width * scale), 1), max(int(image.height * scale), 1)),
            Image.BICUBIC,
        )

    destination = Path(out_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    image.save(destination, format="PNG")
    return destination


def quadrant_summary(mask: np.ndarray) -> str:
    """Where in the frame a mask concentrates, as a compass phrase.

    Gives the VLM a spatial anchor it can quote, instead of leaving it to invent
    a location from the picture alone.
    """
    if not mask.any():
        return "nowhere"
    height, width = mask.shape
    mid_y, mid_x = height // 2, width // 2
    quadrants = {
        "north-west": mask[:mid_y, :mid_x].sum(),
        "north-east": mask[:mid_y, mid_x:].sum(),
        "south-west": mask[mid_y:, :mid_x].sum(),
        "south-east": mask[mid_y:, mid_x:].sum(),
    }
    total = float(mask.sum())
    ranked = sorted(quadrants.items(), key=lambda kv: kv[1], reverse=True)
    top, top_count = ranked[0]
    if top_count / total < 0.35:
        return "spread across the scene"
    second, second_count = ranked[1]
    if second_count / total > 0.28:
        return f"mainly {top} and {second}"
    return f"mainly {top}"
