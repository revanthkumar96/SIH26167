"""Raster inspection and band access.

Reads GeoTIFF through rasterio when it is installed and falls back to PIL for the
PNG/JPEG inputs the public benchmarks ship. Everything downstream consumes
``RasterInfo`` rather than touching a file handle, so the controller's input checks
work identically for both paths.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np

_SAR_HINTS = ("sar", "_vv", "_vh", "vv_", "vh_", "grd", "risat", "sentinel-1", "s1_")
_OPTICAL_HINTS = ("optical", "msi", "cartosat", "sentinel-2", "s2_", "pan", "mss")

#: Metres per degree of latitude, for estimating GSD from a geographic CRS.
_M_PER_DEG = 111_320.0


@dataclass(frozen=True, slots=True)
class RasterInfo:
    """Everything the controller needs to validate an input image."""

    path: Path
    width: int
    height: int
    band_count: int
    dtype: str
    driver: str
    crs: str | None = None
    bounds: tuple[float, float, float, float] | None = None
    transform: tuple[float, ...] | None = None
    gsd_m: float | None = None

    @property
    def georeferenced(self) -> bool:
        return self.crs is not None and self.bounds is not None

    @property
    def size(self) -> tuple[int, int]:
        return (self.width, self.height)

    def as_dict(self) -> dict[str, object]:
        """Trace-friendly payload; this appears verbatim in the execution trace."""
        return {
            "name": self.path.name,
            "format": self.driver,
            "size": [self.width, self.height],
            "bands": self.band_count,
            "dtype": self.dtype,
            "crs": self.crs,
            "bounds": list(self.bounds) if self.bounds else None,
            "gsd_m": round(self.gsd_m, 4) if self.gsd_m else None,
            "georeferenced": self.georeferenced,
        }


def _gsd_from_transform(
    transform: tuple[float, ...], crs: str | None, bounds: tuple[float, ...] | None
) -> float | None:
    """Ground sample distance in metres, converting from degrees when needed."""
    pixel_x = abs(transform[0])
    if pixel_x <= 0:
        return None
    if crs and "4326" in crs:
        latitude = (bounds[1] + bounds[3]) / 2.0 if bounds else 0.0
        return pixel_x * _M_PER_DEG * max(math.cos(math.radians(latitude)), 0.01)
    return pixel_x


def _read_info_rasterio(path: Path) -> RasterInfo | None:
    try:
        import rasterio
    except ImportError:
        return None

    with rasterio.open(path) as src:
        crs = str(src.crs) if src.crs else None
        bounds = tuple(src.bounds) if src.crs else None
        transform = tuple(src.transform)[:6]
        return RasterInfo(
            path=path,
            width=src.width,
            height=src.height,
            band_count=src.count,
            dtype=str(src.dtypes[0]) if src.dtypes else "unknown",
            driver=src.driver or "GTiff",
            crs=crs,
            bounds=bounds,  # type: ignore[arg-type]
            transform=transform,
            gsd_m=_gsd_from_transform(transform, crs, bounds) if src.crs else None,
        )


def _read_info_pil(path: Path) -> RasterInfo:
    from PIL import Image

    with Image.open(path) as image:
        bands = len(image.getbands())
        return RasterInfo(
            path=path,
            width=image.width,
            height=image.height,
            band_count=bands,
            dtype="uint8" if image.mode in {"RGB", "L", "RGBA"} else image.mode,
            driver=(image.format or path.suffix.lstrip(".")).upper(),
        )


def read_info(path: str | Path) -> RasterInfo:
    """Inspect a raster without loading its pixels.

    GeoTIFF goes through rasterio; PNG/JPEG and rasterio-less environments fall
    back to PIL, which yields a non-georeferenced ``RasterInfo``.
    """
    target = Path(path)
    if not target.exists():
        raise FileNotFoundError(f"raster not found: {target}")

    if target.suffix.lower() in {".tif", ".tiff", ".jp2"}:
        info = _read_info_rasterio(target)
        if info is not None:
            return info
    return _read_info_pil(target)


def read_bands(path: str | Path, indexes: list[int] | None = None) -> np.ndarray:
    """Read bands as a float32 array shaped ``(bands, height, width)``.

    ``indexes`` is 1-based to match rasterio's convention.
    """
    target = Path(path)
    try:
        import rasterio
    except ImportError:
        rasterio = None  # type: ignore[assignment]

    if rasterio is not None and target.suffix.lower() in {".tif", ".tiff", ".jp2"}:
        with rasterio.open(target) as src:
            data = src.read(indexes) if indexes else src.read()
        return np.asarray(data, dtype=np.float32)

    from PIL import Image

    with Image.open(target) as image:
        array = np.asarray(image.convert("RGB"), dtype=np.float32)
    stacked = np.transpose(array, (2, 0, 1))
    if indexes:
        stacked = stacked[[i - 1 for i in indexes]]
    return stacked


def stretch_to_uint8(
    band: np.ndarray, low: float = 2.0, high: float = 98.0
) -> np.ndarray:
    """Percentile stretch, the only way uint16 or dB rasters render usefully."""
    finite = band[np.isfinite(band)]
    if finite.size == 0:
        return np.zeros(band.shape, dtype=np.uint8)
    lo, hi = np.percentile(finite, [low, high])
    if hi <= lo:
        lo, hi = float(finite.min()), float(finite.max())
    if hi <= lo:
        return np.zeros(band.shape, dtype=np.uint8)
    return (np.clip((band - lo) / (hi - lo), 0.0, 1.0) * 255.0).astype(np.uint8)


def render_preview(
    path: str | Path, out_path: str | Path, max_side: int = 1024
) -> Path:
    """Write a browser-displayable PNG for any input raster."""
    from PIL import Image

    bands = read_bands(path)
    if bands.shape[0] >= 3:
        rgb = np.dstack([stretch_to_uint8(b) for b in bands[:3]])
    else:
        grey = stretch_to_uint8(bands[0])
        rgb = np.dstack([grey, grey, grey])

    image = Image.fromarray(rgb, mode="RGB")
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


def filename_hint(path: Path) -> str | None:
    """Modality suggested by the filename, which users encode more often than not."""
    name = path.name.lower()
    if any(hint in name for hint in _SAR_HINTS):
        return "sar"
    if any(hint in name for hint in _OPTICAL_HINTS):
        return "optical"
    return None
