"""Fetch real Sentinel-1/2 test scenes from Microsoft Planetary Computer.

Open STAC with anonymous SAS signing -- no credentials needed. Produces bundled
demo scenes that exercise every mandatory input configuration:

  Mumbai / Thane creek     optical (S2, 12 band) + SAR (S1 RTC, VV/VH), same day
  Ujani reservoir          bi-temporal S2, dry season vs post-monsoon
  Sundarbans delta         single 12-band S2 + bi-temporal shoreline pair
  Jaisalmer, Rajasthan     single 12-band S2 (arid, cloud-free)
  Sriharikota              single 12-band S2 + bi-temporal coastline pair
  Delhi NCR                bi-temporal urban growth pair
  Chennai coast            optical + SAR cross-modal pair

Every scene is warped onto one explicitly defined UTM grid per AOI, so pairs
come out genuinely co-registered -- identical CRS, extent and pixel size.

Filenames are stable (no acquisition dates) so ``SAMPLE_SETS`` in app.py never
drifts when scenes are re-fetched.

Usage::

    pip install -e ".[geo]" requests
    python scripts/fetch_sentinel_samples.py runs/samples
"""

from __future__ import annotations

import os
import shutil
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
# Compact stack for bi-temporal pairs: red, green, blue, nir.
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


def best_s2(features):
    return min(features, key=lambda f: f["properties"].get("eo:cloud_cover", 100))


def print_scene(label, item):
    print(
        f"  {label:<12} {item['properties']['datetime'][:10]}  cloud "
        f"{item['properties'].get('eo:cloud_cover', 0):.1f}%  {item['id']}"
    )


def promote_legacy_ujani(out: Path) -> None:
    """Copy dated Ujani filenames to stable names when a re-fetch is not needed."""
    for stable, pattern in (
        ("ujani_before.tif", "ujani_before_*.tif"),
        ("ujani_after.tif", "ujani_after_*.tif"),
    ):
        target = out / stable
        if target.exists():
            continue
        matches = sorted(out.glob(pattern))
        if matches:
            shutil.copy2(matches[0], target)
            print(f"    promoted {matches[0].name} -> {stable}")


def fetch_crossmodal(
    out: Path,
    label: str,
    bbox: list[float],
    epsg: int,
    optical_name: str,
    sar_name: str,
    start: str,
    end: str,
    cloud_lt: float = 8,
) -> bool:
    print(f"\n{label} (cross-modal optical + SAR)")
    crs, transform = target_grid(bbox, epsg)
    s2 = search(
        "sentinel-2-l2a",
        bbox,
        start,
        end,
        {"eo:cloud_cover": {"lt": cloud_lt}},
    )
    s1 = search("sentinel-1-rtc", bbox, start, end)
    if not s2 or not s1:
        print("  no scenes found", file=sys.stderr)
        return False
    s2_item = best_s2(s2)
    s1_item = s1[0]
    print_scene("S2", s2_item)
    print_scene("S1", s1_item)
    fetch_s2(s2_item, S2_BANDS_12, out / optical_name, crs, transform)
    fetch_s1(s1_item, out / sar_name, crs, transform)
    return True


def fetch_single(
    out: Path,
    label: str,
    bbox: list[float],
    epsg: int,
    optical_name: str,
    start: str,
    end: str,
    cloud_lt: float = 15,
) -> bool:
    print(f"\n{label} (single 12-band S2)")
    crs, transform = target_grid(bbox, epsg)
    s2 = search(
        "sentinel-2-l2a",
        bbox,
        start,
        end,
        {"eo:cloud_cover": {"lt": cloud_lt}},
    )
    if not s2:
        print("  no scenes found", file=sys.stderr)
        return False
    item = best_s2(s2)
    print_scene("S2", item)
    fetch_s2(item, S2_BANDS_12, out / optical_name, crs, transform)
    return True


def fetch_bitemporal(
    out: Path,
    label: str,
    bbox: list[float],
    epsg: int,
    before_name: str,
    after_name: str,
    before_range: tuple[str, str],
    after_range: tuple[str, str],
    cloud_lt: float = 15,
) -> bool:
    print(f"\n{label} (bi-temporal S2)")
    crs, transform = target_grid(bbox, epsg)
    before = search(
        "sentinel-2-l2a",
        bbox,
        before_range[0],
        before_range[1],
        {"eo:cloud_cover": {"lt": cloud_lt}},
    )
    after = search(
        "sentinel-2-l2a",
        bbox,
        after_range[0],
        after_range[1],
        {"eo:cloud_cover": {"lt": cloud_lt}},
    )
    if not before or not after:
        print("  no scenes found for one of the seasons", file=sys.stderr)
        return False
    before_item = best_s2(before)
    after_item = best_s2(after)
    print_scene("before", before_item)
    print_scene("after", after_item)
    fetch_s2(before_item, S2_BANDS_4, out / before_name, crs, transform)
    fetch_s2(after_item, S2_BANDS_4, out / after_name, crs, transform)
    return True


def main() -> int:
    out = Path(sys.argv[1])
    out.mkdir(parents=True, exist_ok=True)
    promote_legacy_ujani(out)

    failures: list[str] = []

    # ---- Mumbai / Thane creek ------------------------------------------------
    if not fetch_crossmodal(
        out,
        "AOI A  Mumbai / Thane creek",
        [72.90, 19.02, 73.02, 19.12],
        32643,
        "mumbai_optical_S2_20240206.tif",
        "mumbai_sar_S1_VV_20240206.tif",
        "2024-02-01",
        "2024-02-12",
        cloud_lt=5,
    ):
        failures.append("mumbai")

    # ---- Ujani reservoir ---------------------------------------------------
    if not fetch_bitemporal(
        out,
        "AOI B  Ujani reservoir, Maharashtra",
        [75.05, 18.03, 75.20, 18.13],
        32643,
        "ujani_before.tif",
        "ujani_after.tif",
        ("2024-05-15", "2024-06-10"),
        ("2024-10-15", "2024-12-05"),
        cloud_lt=12,
    ):
        failures.append("ujani")

    # ---- Sundarbans delta --------------------------------------------------
    if not fetch_single(
        out,
        "AOI C  Sundarbans delta",
        [88.75, 21.85, 88.90, 21.97],
        32645,
        "sundarbans_optical_S2.tif",
        "2024-01-10",
        "2024-02-28",
        cloud_lt=20,
    ):
        failures.append("sundarbans_single")

    if not fetch_bitemporal(
        out,
        "AOI C  Sundarbans shoreline",
        [88.75, 21.85, 88.90, 21.97],
        32645,
        "sundarbans_before.tif",
        "sundarbans_after.tif",
        ("2024-02-01", "2024-03-15"),
        ("2024-10-01", "2024-11-30"),
        cloud_lt=20,
    ):
        failures.append("sundarbans_bitemporal")

    # ---- Jaisalmer ---------------------------------------------------------
    if not fetch_single(
        out,
        "AOI D  Jaisalmer, Rajasthan",
        [70.85, 26.86, 71.00, 26.98],
        32642,
        "jaisalmer_optical_S2.tif",
        "2024-01-01",
        "2024-03-31",
        cloud_lt=5,
    ):
        failures.append("jaisalmer")

    # ---- Sriharikota -------------------------------------------------------
    if not fetch_single(
        out,
        "AOI E  Sriharikota (SDSC SHAR)",
        [80.18, 13.66, 80.30, 13.78],
        32644,
        "sriharikota_optical_S2.tif",
        "2024-01-15",
        "2024-03-15",
        cloud_lt=10,
    ):
        failures.append("sriharikota_single")

    if not fetch_bitemporal(
        out,
        "AOI E  Sriharikota coastline",
        [80.18, 13.66, 80.30, 13.78],
        32644,
        "sriharikota_before.tif",
        "sriharikota_after.tif",
        ("2024-02-01", "2024-03-31"),
        ("2024-09-01", "2024-11-30"),
        cloud_lt=12,
    ):
        failures.append("sriharikota_bitemporal")

    # ---- Delhi NCR urban growth --------------------------------------------
    if not fetch_bitemporal(
        out,
        "AOI F  Delhi NCR",
        [77.05, 28.50, 77.25, 28.70],
        32643,
        "delhi_before.tif",
        "delhi_after.tif",
        ("2023-01-01", "2023-03-31"),
        ("2024-01-01", "2024-03-31"),
        cloud_lt=15,
    ):
        failures.append("delhi")

    # ---- Chennai coast -----------------------------------------------------
    if not fetch_crossmodal(
        out,
        "AOI G  Chennai coast",
        [80.15, 12.95, 80.30, 13.10],
        32644,
        "chennai_optical_S2.tif",
        "chennai_sar_S1.tif",
        "2024-01-15",
        "2024-02-28",
        cloud_lt=10,
    ):
        failures.append("chennai")

    present = sorted(p.name for p in out.glob("*.tif"))
    print(f"\n{len(present)} GeoTIFF(s) in {out}:")
    for name in present:
        print(f"  {name}")

    if failures:
        print(
            f"\nFailed AOIs: {', '.join(failures)} (other scenes were still written).",
            file=sys.stderr,
        )
        return 1

    print(
        "\nAll scenes on their AOI's shared UTM grid: identical CRS, extent "
        "and pixel size."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
