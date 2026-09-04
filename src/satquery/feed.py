"""Live satellite feed: search recent acquisitions, then pull them for analysis.

Backed by Microsoft Planetary Computer's open STAC API with anonymous SAS signing,
so no credentials are needed. Browsing uses the catalogue's own
``rendered_preview`` asset and loading uses windowed COG reads.

We deliberately do **not** run a tile server. Planetary Computer already publishes
``rendered_preview`` and ``tilejson`` per item, so standing up TiTiler alongside it
would be reinvention.
  verdict: ADOPT (avoids REINVENTING) -- see docs/RESEARCH.md row 9

Every scene in a request is warped onto one explicitly defined UTM grid, so an
optical/SAR or before/after pair comes back genuinely co-registered -- identical CRS,
extent and pixel size -- rather than merely overlapping.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import UTC, date, datetime, timedelta
from pathlib import Path
from typing import Any

STAC_SEARCH = "https://planetarycomputer.microsoft.com/api/stac/v1/search"
SAS_SIGN = "https://planetarycomputer.microsoft.com/api/sas/v1/sign"

#: Sentinel-2 L2A in BigEarthNet band order, which the optical index tool maps
#: directly onto. B10 is absent from L2A, giving exactly twelve bands.
S2_BANDS = (
    "B01",
    "B02",
    "B03",
    "B04",
    "B05",
    "B06",
    "B07",
    "B08",
    "B8A",
    "B09",
    "B11",
    "B12",
)
#: Compact optical stack: red, green, blue, near-infrared.
S2_BANDS_RGBN = ("B04", "B03", "B02", "B08")
#: Sentinel-1 RTC polarisations.
S1_BANDS = ("vv", "vh")

DEFAULT_SIZE = 1024
DEFAULT_RES = 10.0


@dataclass(frozen=True)
class Collection:
    """A searchable imagery source."""

    id: str
    label: str
    modality: str
    description: str


COLLECTIONS: tuple[Collection, ...] = (
    Collection(
        "sentinel-2-l2a",
        "Sentinel-2 L2A",
        "optical",
        "Multispectral optical, 10 m, surface reflectance.",
    ),
    Collection(
        "sentinel-1-rtc",
        "Sentinel-1 RTC",
        "sar",
        "Radiometrically terrain-corrected SAR backscatter, VV and VH.",
    ),
)


@dataclass
class FeedProgress:
    """Progress of a scene pull, safe to serialise to a client."""

    state: str = "idle"  # idle | searching | fetching | ready | error
    detail: str = ""
    bands_done: int = 0
    bands_total: int = 0
    files: list[str] = field(default_factory=list)

    @property
    def percent(self) -> float | None:
        if self.bands_total <= 0:
            return None
        return round(100.0 * self.bands_done / self.bands_total, 1)

    def as_dict(self) -> dict[str, Any]:
        return {
            "state": self.state,
            "detail": self.detail,
            "bands_done": self.bands_done,
            "bands_total": self.bands_total,
            "percent": self.percent,
            "files": list(self.files),
        }


class FeedUnavailableError(RuntimeError):
    """Raised when the geospatial stack needed to pull imagery is missing."""


def _require_rasterio():
    try:
        import rasterio
    except ImportError as exc:  # pragma: no cover - environment dependent
        raise FeedUnavailableError(
            "reading remote imagery needs rasterio; install with "
            "pip install -e '.[geo]'"
        ) from exc
    return rasterio


def describe_collections() -> list[dict[str, Any]]:
    return [
        {
            "id": c.id,
            "label": c.label,
            "modality": c.modality,
            "description": c.description,
        }
        for c in COLLECTIONS
    ]


def _query(
    collection: str,
    bbox: tuple[float, float, float, float],
    start: str,
    end: str,
    max_cloud: float | None,
    limit: int,
) -> list[dict[str, Any]]:
    import requests

    body: dict[str, Any] = {
        "collections": [collection],
        "bbox": list(bbox),
        "datetime": f"{start}/{end}",
        "limit": max(1, min(limit, 50)),
        "sortby": [{"field": "properties.datetime", "direction": "desc"}],
    }
    if max_cloud is not None and collection.startswith("sentinel-2"):
        body["query"] = {"eo:cloud_cover": {"lt": max_cloud}}

    response = requests.post(STAC_SEARCH, json=body, timeout=90)
    response.raise_for_status()

    scenes: list[dict[str, Any]] = []
    for feature in response.json().get("features", []):
        properties = feature.get("properties", {})
        assets = feature.get("assets", {})
        scenes.append(
            {
                "id": feature["id"],
                "collection": feature.get("collection", collection),
                "datetime": properties.get("datetime", ""),
                "cloud_cover": properties.get("eo:cloud_cover"),
                "platform": properties.get("platform"),
                "orbit_state": properties.get("sat:orbit_state"),
                "instrument_mode": properties.get("sar:instrument_mode"),
                "preview": (assets.get("rendered_preview") or {}).get("href"),
                "tilejson": (assets.get("tilejson") or {}).get("href"),
                "bbox": feature.get("bbox"),
            }
        )
    return scenes


def search(
    collection: str,
    bbox: tuple[float, float, float, float],
    start: str | None = None,
    end: str | None = None,
    max_cloud: float | None = 20.0,
    limit: int = 12,
) -> list[dict[str, Any]]:
    """Recent acquisitions over a bounding box, newest first."""
    if end is None:
        end = date.today().isoformat()
    if start is None:
        start = (datetime.now(UTC) - timedelta(days=60)).date().isoformat()
    return _query(collection, bbox, start, end, max_cloud, limit)


def search_with_diagnosis(
    collection: str,
    bbox: tuple[float, float, float, float],
    start: str | None = None,
    end: str | None = None,
    max_cloud: float | None = 20.0,
    limit: int = 12,
) -> dict[str, Any]:
    """Search, and explain an empty result instead of showing a blank grid.

    A cloud threshold that filters everything out is the normal case over a
    monsoon region -- Mumbai in August runs 90-98% cloud -- and an unexplained
    empty grid reads as a broken feed. When the filter is what emptied the
    result, the unfiltered count is reported alongside a hint.

    That case is also the problem statement's own argument for SAR, so the hint
    says so: radar is unaffected by the cloud that makes optical useless.
    """
    if end is None:
        end = date.today().isoformat()
    if start is None:
        start = (datetime.now(UTC) - timedelta(days=60)).date().isoformat()

    scenes = _query(collection, bbox, start, end, max_cloud, limit)
    result: dict[str, Any] = {
        "collection": collection,
        "start": start,
        "end": end,
        "max_cloud": max_cloud,
        "scenes": scenes,
        "hint": "",
    }
    if scenes or max_cloud is None or not collection.startswith("sentinel-2"):
        return result

    unfiltered = _query(collection, bbox, start, end, None, limit)
    result["filtered_out"] = len(unfiltered)
    if unfiltered:
        cloudiest = min(
            (s["cloud_cover"] for s in unfiltered if s["cloud_cover"] is not None),
            default=None,
        )
        clearest = f"{cloudiest:.0f}%" if cloudiest is not None else "unknown"
        result["hint"] = (
            f"{len(unfiltered)} acquisition(s) exist here, but none below "
            f"{max_cloud:.0f}% cloud -- the clearest is {clearest}. Raise the "
            f"threshold, widen the dates, or use Sentinel-1: radar is unaffected "
            f"by the cloud that makes optical unusable."
        )
    else:
        result["hint"] = (
            "No acquisitions at all over this footprint in this window. Widen "
            "the date range or move the area of interest."
        )
    return result


def _sign(href: str) -> str:
    import requests

    response = requests.get(SAS_SIGN, params={"href": href}, timeout=60)
    response.raise_for_status()
    return response.json()["href"]


def _fetch_item(item_id: str, collection: str) -> dict[str, Any]:
    """Re-read one STAC item so asset hrefs are fresh when we sign them."""
    import requests

    response = requests.post(
        STAC_SEARCH,
        json={"collections": [collection], "ids": [item_id], "limit": 1},
        timeout=90,
    )
    response.raise_for_status()
    features = response.json().get("features", [])
    if not features:
        raise LookupError(f"scene '{item_id}' not found in '{collection}'")
    return features[0]


def utm_epsg(longitude: float, latitude: float) -> int:
    """EPSG code of the UTM zone containing a point."""
    zone = int((longitude + 180.0) // 6.0) + 1
    return (32600 if latitude >= 0 else 32700) + zone


def target_grid(
    bbox: tuple[float, float, float, float],
    size: int = DEFAULT_SIZE,
    resolution: float = DEFAULT_RES,
):
    """A fixed UTM grid centred on the bbox, shared by every scene in a pull."""
    _require_rasterio()
    from rasterio.crs import CRS
    from rasterio.transform import from_origin
    from rasterio.warp import transform_bounds

    left, bottom, right, top = bbox
    epsg = utm_epsg((left + right) / 2.0, (bottom + top) / 2.0)
    destination = CRS.from_epsg(epsg)

    x0, y0, x1, y1 = transform_bounds(CRS.from_epsg(4326), destination, *bbox)
    centre_x, centre_y = (x0 + x1) / 2.0, (y0 + y1) / 2.0
    half = size * resolution / 2.0
    # Snap the origin to the resolution so pixel edges line up exactly.
    origin_x = round((centre_x - half) / resolution) * resolution
    origin_y = round((centre_y + half) / resolution) * resolution
    return destination, from_origin(origin_x, origin_y, resolution, resolution)


def _read_on_grid(url: str, crs, transform, size: int):
    rasterio = _require_rasterio()
    from rasterio.vrt import WarpedVRT

    with (
        rasterio.open(url) as source,
        WarpedVRT(
            source, crs=crs, transform=transform, width=size, height=size, resampling=1
        ) as vrt,
    ):
        return vrt.read(1)


def _write_stack(path: Path, arrays, crs, transform, dtype: str, names) -> Path:
    rasterio = _require_rasterio()

    path.parent.mkdir(parents=True, exist_ok=True)
    with rasterio.open(
        path,
        "w",
        driver="GTiff",
        height=arrays[0].shape[0],
        width=arrays[0].shape[1],
        count=len(arrays),
        dtype=dtype,
        crs=crs,
        transform=transform,
        compress="deflate",
        tiled=True,
        blockxsize=512,
        blockysize=512,
    ) as destination:
        for index, array in enumerate(arrays, start=1):
            destination.write(array.astype(dtype), index)
            destination.set_band_description(index, names[index - 1])
    return path


def pull_scenes(
    scenes: list[dict[str, str]],
    bbox: tuple[float, float, float, float],
    out_dir: Path,
    size: int = DEFAULT_SIZE,
    compact_optical: bool = False,
    on_update=None,
) -> FeedProgress:
    """Download one or two scenes onto a shared grid as GeoTIFFs.

    ``scenes`` is a list of ``{"id": ..., "collection": ...}``. Passing an optical
    and a SAR scene yields a co-registered cross-modal pair; passing two optical
    scenes from different dates yields a bi-temporal pair.

    Never raises: the returned progress carries the outcome.
    """
    progress = FeedProgress(state="searching", detail="resolving scenes")

    def emit() -> None:
        if on_update is not None:
            on_update(progress)

    emit()

    try:
        crs, transform = target_grid(bbox, size=size)
        items = [_fetch_item(s["id"], s["collection"]) for s in scenes]

        plans: list[tuple[dict[str, Any], tuple[str, ...], str, str]] = []
        for item in items:
            collection = item.get("collection", "")
            if collection.startswith("sentinel-1"):
                plans.append((item, S1_BANDS, "float32", "sar"))
            else:
                bands = S2_BANDS_RGBN if compact_optical else S2_BANDS
                plans.append((item, bands, "uint16", "optical"))

        progress.bands_total = sum(len(bands) for _, bands, _, _ in plans)
        progress.state = "fetching"
        emit()

        written: list[Path] = []
        for item, bands, dtype, kind in plans:
            arrays = []
            for band in bands:
                asset = item["assets"].get(band)
                if asset is None:
                    raise LookupError(f"{item['id']} has no asset '{band}'")
                progress.detail = f"{item['id'][:28]} · {band}"
                emit()
                arrays.append(_read_on_grid(_sign(asset["href"]), crs, transform, size))
                progress.bands_done += 1
                emit()

            stamp = str(item.get("properties", {}).get("datetime", ""))[:10]
            name = f"{kind}_{stamp.replace('-', '')}_{item['id'][:18]}.tif"
            written.append(
                _write_stack(out_dir / name, arrays, crs, transform, dtype, list(bands))
            )

        progress.files = [str(p) for p in written]
        progress.state = "ready"
        progress.detail = f"{len(written)} scene(s) on a shared UTM grid"
        emit()

    except Exception as exc:
        progress.state = "error"
        progress.detail = f"{type(exc).__name__}: {exc}"
        emit()

    return progress
