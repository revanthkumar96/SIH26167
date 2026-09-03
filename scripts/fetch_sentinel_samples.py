"""Fetch real Sentinel-1/2 test scenes from Microsoft Planetary Computer.

Open STAC with anonymous SAS signing -- no credentials needed. Produces two
AOIs that exercise the mandatory input configurations with genuine data:

  Mumbai / Thane creek   optical (S2, 12 band) + SAR (S1 RTC, VV/VH), same day
  Ujani reservoir        bi-temporal S2, dry season vs post-monsoon

Every scene is warped onto one explicitly defined UTM grid per AOI, so the
pairs come out genuinely co-registered -- identical CRS, extent and pixel size
-- rather than merely overlapping.

Usage::

    pip install -e ".[geo]" requests
    python scripts/fetch_sentinel_samples.py runs/samples
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import rasterio
import requests
from rasterio.crs import CRS
from rasterio.transform import from_origin
from rasterio.vrt import WarpedVRT
from rasterio.warp import transform_bounds

os.environ.setdefault("GDAL_DISABLE_READDIR_ON_OPEN", "EMPTY_DIR")
os.environ.setdefault("CPL_VSIL_CURL_ALLOWED_EXTENSIONS", ".tif,.TIF,.tiff,.TIFF")
os.environ.setdefault("GDAL_HTTP_MAX_RETRY", "4")
os.environ.setdefault("GDAL_HTTP_RETRY_DELAY", "2")

STAC = "https://planetarycomputer.microsoft.com/api/stac/v1/search"
SIGN = "https://planetarycomputer.microsoft.com/api/sas/v1/sign"

# BigEarthNet band order: our optical index tool maps 12 bands to this exactly.
S2_BANDS_12 = [
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
]
# Compact stack for the bi-temporal pair: red, green, blue, nir.
S2_BANDS_4 = ["B04", "B03", "B02", "B08"]

RES = 10.0
SIZE = 1024


def search(collection, bbox, start, end, query=None, limit=10):
    body = {
        "collections": [collection],
        "bbox": bbox,
        "datetime": f"{start}/{end}",
        "limit": limit,
    }
    if query:
        body["query"] = query
    r = requests.post(STAC, json=body, timeout=90)
    r.raise_for_status()
    return r.json().get("features", [])


def sign(href: str) -> str:
    r = requests.get(SIGN, params={"href": href}, timeout=60)
    r.raise_for_status()
    return r.json()["href"]


def target_grid(bbox, epsg):
    """A fixed UTM grid centred on the AOI. Shared by every scene we write."""
    dst = CRS.from_epsg(epsg)
    left, bottom, right, top = transform_bounds(CRS.from_epsg(4326), dst, *bbox)
    cx, cy = (left + right) / 2.0, (bottom + top) / 2.0
    half = SIZE * RES / 2.0
    # Snap the origin to the resolution so pixel edges line up exactly.
    ox = round((cx - half) / RES) * RES
    oy = round((cy + half) / RES) * RES
    return dst, from_origin(ox, oy, RES, RES)


def read_on_grid(url, dst_crs, dst_transform, band=1):
    """Read one band of a remote COG straight onto the shared target grid."""
    with (
        rasterio.open(url) as src,
        WarpedVRT(
            src,
            crs=dst_crs,
            transform=dst_transform,
            width=SIZE,
            height=SIZE,
            resampling=1,
        ) as vrt,
    ):
        return vrt.read(band)


def write_stack(path, arrays, dst_crs, dst_transform, dtype, descriptions):
    path.parent.mkdir(parents=True, exist_ok=True)
    with rasterio.open(
        path,
        "w",
        driver="GTiff",
        height=SIZE,
        width=SIZE,
        count=len(arrays),
        dtype=dtype,
        crs=dst_crs,
        transform=dst_transform,
        compress="deflate",
        tiled=True,
        blockxsize=512,
        blockysize=512,
    ) as dst:
        for i, arr in enumerate(arrays, start=1):
            dst.write(arr.astype(dtype), i)
            dst.set_band_description(i, descriptions[i - 1])
    mb = path.stat().st_size / 1024 / 1024
    print(f"    wrote {path.name}  ({len(arrays)} bands, {mb:.1f} MB)")


def fetch_s2(item, bands, out, dst_crs, dst_transform):
    if out.exists():
        print(f"    {out.name} already present, skipping")
        return
    arrays = []
    for name in bands:
        url = sign(item["assets"][name]["href"])
        arrays.append(read_on_grid(url, dst_crs, dst_transform))
        print(f"      {name}", end="", flush=True)
    print()
    write_stack(out, arrays, dst_crs, dst_transform, "uint16", bands)


def fetch_s1(item, out, dst_crs, dst_transform):
    if out.exists():
        print(f"    {out.name} already present, skipping")
        return
    arrays = []
    for name in ("vv", "vh"):
        url = sign(item["assets"][name]["href"])
        arrays.append(read_on_grid(url, dst_crs, dst_transform))
        print(f"      {name.upper()}", end="", flush=True)
    print()
    write_stack(out, arrays, dst_crs, dst_transform, "float32", ["VV", "VH"])


def main() -> int:
    out = Path(sys.argv[1])

    # ---- AOI A: Mumbai / Thane creek -- sea, creek, dense built-up ----------
    print("AOI A  Mumbai / Thane creek (cross-modal optical + SAR)")
    bbox_a = [72.90, 19.02, 73.02, 19.12]
    crs_a, tf_a = target_grid(bbox_a, 32643)

    s2 = search(
        "sentinel-2-l2a",
        bbox_a,
        "2024-02-01",
        "2024-02-12",
        {"eo:cloud_cover": {"lt": 5}},
    )
    s1 = search("sentinel-1-rtc", bbox_a, "2024-02-01", "2024-02-12")
    if not s2 or not s1:
        print("  no scenes found", file=sys.stderr)
        return 1

    s2_item = min(s2, key=lambda f: f["properties"].get("eo:cloud_cover", 100))
    s1_item = s1[0]
    print(
        f"  S2 {s2_item['id']}  cloud "
        f"{s2_item['properties'].get('eo:cloud_cover'):.1f}%"
    )
    print(f"  S1 {s1_item['id']}")
    fetch_s2(s2_item, S2_BANDS_12, out / "mumbai_optical_S2_20240206.tif", crs_a, tf_a)
    fetch_s1(s1_item, out / "mumbai_sar_S1_VV_20240206.tif", crs_a, tf_a)

    # ---- AOI B: Ujani reservoir -- dry season vs post-monsoon --------------
    print("\nAOI B  Ujani reservoir, Maharashtra (bi-temporal water change)")
    bbox_b = [75.05, 18.03, 75.20, 18.13]
    crs_b, tf_b = target_grid(bbox_b, 32643)

    dry = search(
        "sentinel-2-l2a",
        bbox_b,
        "2024-05-15",
        "2024-06-10",
        {"eo:cloud_cover": {"lt": 12}},
    )
    wet = search(
        "sentinel-2-l2a",
        bbox_b,
        "2024-10-15",
        "2024-12-05",
        {"eo:cloud_cover": {"lt": 12}},
    )
    if not dry or not wet:
        print("  no scenes found for one of the seasons", file=sys.stderr)
        return 1

    dry_item = min(dry, key=lambda f: f["properties"].get("eo:cloud_cover", 100))
    wet_item = min(wet, key=lambda f: f["properties"].get("eo:cloud_cover", 100))
    for label, item in (("before/dry", dry_item), ("after/wet", wet_item)):
        print(
            f"  {label:<10} {item['properties']['datetime'][:10]}  cloud "
            f"{item['properties'].get('eo:cloud_cover'):.1f}%"
        )

    d1 = dry_item["properties"]["datetime"][:10].replace("-", "")
    d2 = wet_item["properties"]["datetime"][:10].replace("-", "")
    fetch_s2(dry_item, S2_BANDS_4, out / f"ujani_before_{d1}.tif", crs_b, tf_b)
    fetch_s2(wet_item, S2_BANDS_4, out / f"ujani_after_{d2}.tif", crs_b, tf_b)

    print(
        "\nAll scenes on their AOI's shared UTM grid: identical CRS, extent "
        "and pixel size."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
