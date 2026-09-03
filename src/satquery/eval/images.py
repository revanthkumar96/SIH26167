"""Image loading for benchmark inputs.

Remote-sensing TIFFs are frequently uint16 or multiband, which PIL renders as a
black frame if handed straight to a model. Anything that is not already 8-bit RGB
gets a percentile stretch so the model sees what a human would.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image

Image.MAX_IMAGE_PIXELS = None

_STRETCH_LOW = 2.0
_STRETCH_HIGH = 98.0


def _stretch(array: np.ndarray) -> np.ndarray:
    """Percentile-stretch a band to uint8, robust to outliers and constant bands."""
    finite = array[np.isfinite(array)]
    if finite.size == 0:
        return np.zeros(array.shape, dtype=np.uint8)
    low, high = np.percentile(finite, [_STRETCH_LOW, _STRETCH_HIGH])
    if high <= low:
        low, high = float(finite.min()), float(finite.max())
    if high <= low:
        return np.zeros(array.shape, dtype=np.uint8)
    clipped = np.clip((array - low) / (high - low), 0.0, 1.0)
    return (clipped * 255.0).astype(np.uint8)


def _from_rasterio(path: Path) -> Image.Image | None:
    """Read a multiband or non-8-bit raster if rasterio is installed."""
    try:
        import rasterio
    except ImportError:
        return None
    with rasterio.open(path) as src:
        count = min(src.count, 3)
        bands = [src.read(i + 1).astype(np.float32) for i in range(count)]
    while len(bands) < 3:
        bands.append(bands[-1])
    stacked = np.dstack([_stretch(b) for b in bands[:3]])
    return Image.fromarray(stacked, mode="RGB")


def load_image(path: str | Path, max_side: int | None = None) -> Image.Image:
    """Load any benchmark image as 8-bit RGB, optionally downscaled.

    ``max_side`` caps the longest edge. Vision tokens dominate VLM inference cost,
    so capping resolution is the main throughput lever in the bake-off.
    """
    target = Path(path)
    if not target.exists():
        raise FileNotFoundError(f"image not found: {target}")

    try:
        image = Image.open(target)
        image.load()
        if image.mode not in {"RGB", "L"}:
            raise ValueError(f"unsupported mode {image.mode}")
        if image.mode == "L":
            image = image.convert("RGB")
    except Exception:  # fall back to the geospatial reader
        fallback = _from_rasterio(target)
        if fallback is None:
            raise
        image = fallback

    if image.mode != "RGB":
        image = image.convert("RGB")

    if max_side and max(image.size) > max_side:
        scale = max_side / max(image.size)
        new_size = (max(int(image.width * scale), 1), max(int(image.height * scale), 1))
        image = image.resize(new_size, Image.BICUBIC)

    return image


def image_size(path: str | Path) -> tuple[int, int]:
    """Width and height without decoding the full raster."""
    with Image.open(path) as image:
        return image.size
