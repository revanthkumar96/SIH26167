"""Input inspection and compatibility checks.

Uses PNG fixtures so the suite runs without rasterio or GDAL installed.
"""

from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from satquery.geo.checks import (
    assign_roles,
    check_pair,
    infer_input_config,
    infer_modality,
)
from satquery.geo.raster import (
    RasterInfo,
    looks_like_linear_backscatter,
    preview_bands,
    read_bands,
    read_info,
    render_preview,
    stretch_to_uint8,
    to_rgb8,
)
from satquery.schema import ImageRole, InputConfig, Modality


def _png(path, size=(64, 48), colour=(10, 120, 40)):
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", size, colour).save(path)
    return path


def _info(name, bands=3, dtype="uint8", crs=None, bounds=None, gsd=None, size=(64, 48)):
    return RasterInfo(
        path=Path(name),
        width=size[0],
        height=size[1],
        band_count=bands,
        dtype=dtype,
        driver="PNG",
        crs=crs,
        bounds=bounds,
        gsd_m=gsd,
    )


def test_read_info_png(tmp_path):
    info = read_info(_png(tmp_path / "scene.png"))
    assert info.size == (64, 48)
    assert info.band_count == 3
    assert not info.georeferenced


def test_read_info_missing_file(tmp_path):
    with pytest.raises(FileNotFoundError):
        read_info(tmp_path / "nope.png")


def test_read_bands_shape(tmp_path):
    bands = read_bands(_png(tmp_path / "scene.png"))
    assert bands.shape == (3, 48, 64)


def test_render_preview_writes_png(tmp_path):
    out = render_preview(_png(tmp_path / "scene.png"), tmp_path / "prev" / "p.png")
    assert out.exists()
    with Image.open(out) as image:
        assert image.mode == "RGB"


def test_render_preview_downscales(tmp_path):
    source = _png(tmp_path / "big.png", size=(2048, 1024))
    out = render_preview(source, tmp_path / "small.png", max_side=256)
    with Image.open(out) as image:
        assert max(image.size) == 256


def test_render_preview_handles_single_band(tmp_path):
    """A one-band SAR raster must still render, as greyscale rather than crashing."""
    path = tmp_path / "sar_vv.png"
    Image.fromarray(np.full((32, 32), 90, dtype=np.uint8), mode="L").save(path)
    assert render_preview(path, tmp_path / "out.png").exists()


def test_preview_bands_true_colour_for_multispectral():
    """A 12-band stack starts B01 B02 B03; the first three are not true colour."""
    assert preview_bands(12) == (4, 3, 2)
    assert preview_bands(13) == (4, 3, 2)
    assert preview_bands(4) == (1, 2, 3)
    assert preview_bands(3) == (1, 2, 3)
    assert preview_bands(2) is None
    assert preview_bands(1) is None


def _synthetic_sar(seed=1, shape=(1, 128, 128)):
    """Distributed scatterers plus bright urban double-bounce, as real SAR has.

    A plain exponential is not skewed enough to stand in for a real tile: the
    Sentinel-1 RTC scene this was calibrated against had a p99/median of 26.
    """
    rng = np.random.default_rng(seed)
    sar = rng.exponential(0.12, shape).astype(np.float32)
    bright = rng.random(shape) < 0.02
    sar[bright] *= rng.uniform(50, 300, int(bright.sum())).astype(np.float32)
    return sar


def test_linear_backscatter_detected_from_skew():
    """Sentinel-1 RTC gamma0 is linear power and heavily skewed."""
    assert looks_like_linear_backscatter(_synthetic_sar(shape=(2, 64, 64)))


def test_elevation_raster_is_not_mistaken_for_backscatter():
    """A float DEM is positive and single-band but nothing like as skewed."""
    rng = np.random.default_rng(3)
    dem = rng.normal(500.0, 300.0, (1, 64, 64)).clip(1.0).astype(np.float32)
    assert not looks_like_linear_backscatter(dem)


def test_decibel_data_is_not_converted_twice():
    rng = np.random.default_rng(0)
    db = (rng.normal(-12.0, 4.0, (2, 64, 64))).astype(np.float32)
    assert not looks_like_linear_backscatter(db)


def test_optical_is_not_mistaken_for_backscatter():
    rng = np.random.default_rng(0)
    optical = rng.integers(200, 4000, (12, 64, 64)).astype(np.uint16)
    assert not looks_like_linear_backscatter(optical)


def test_linear_sar_renders_with_usable_contrast():
    """Stretching linear power directly leaves a near-black frame.

    This is what made SAR water detection report 94% of a real Sentinel-1 tile
    as water: almost every pixel landed in the darkest bin, so the threshold
    between "water" and "land" had nothing to separate.
    """
    sar = _synthetic_sar()

    naive = stretch_to_uint8(sar[0])
    rendered = to_rgb8(sar)[:, :, 0]

    assert (naive < 26).mean() > 0.4
    assert (rendered < 26).mean() < 0.15
    assert rendered.mean() > 2 * naive.mean()


def test_modality_from_filename_beats_band_count(tmp_path):
    """A user naming a file *_VV.tif is telling us more than the header does."""
    sar = _info(tmp_path / "scene_VV.tif", bands=3)
    assert infer_modality(sar) is Modality.SAR


def test_modality_two_band_float_is_sar(tmp_path):
    assert infer_modality(_info(tmp_path / "a.tif", bands=2, dtype="float32")) is (
        Modality.SAR
    )


def test_modality_multispectral_is_optical(tmp_path):
    assert infer_modality(_info(tmp_path / "a.tif", bands=12)) is Modality.OPTICAL


def test_single_image_config(tmp_path):
    config, modalities = infer_input_config([_info(tmp_path / "a.png")])
    assert config is InputConfig.SINGLE
    assert assign_roles(config, modalities) == [ImageRole.SINGLE]


def test_crossmodal_pair_detected_from_mixed_modalities(tmp_path):
    infos = [_info(tmp_path / "optical.tif", bands=4), _info(tmp_path / "sar_vv.tif")]
    config, modalities = infer_input_config(infos)
    assert config is InputConfig.CROSSMODAL_PAIR
    assert assign_roles(config, modalities) == [ImageRole.OPTICAL, ImageRole.SAR]


def test_bitemporal_pair_when_modalities_match(tmp_path):
    infos = [_info(tmp_path / "t1.png"), _info(tmp_path / "t2.png")]
    config, modalities = infer_input_config(infos)
    assert config is InputConfig.BITEMPORAL_PAIR
    assert assign_roles(config, modalities) == [ImageRole.BEFORE, ImageRole.AFTER]


def test_more_than_two_images_is_rejected_with_a_useful_message(tmp_path):
    infos = [_info(tmp_path / f"{i}.png") for i in range(3)]
    with pytest.raises(ValueError, match="bi-temporal pair"):
        infer_input_config(infos)


def test_no_images_is_rejected():
    with pytest.raises(ValueError, match="at least one image"):
        infer_input_config([])


def test_check_pair_georeferenced_match(tmp_path):
    a = _info(tmp_path / "a.tif", crs="EPSG:32643", bounds=(0, 0, 100, 100), gsd=0.65)
    b = _info(tmp_path / "b.tif", crs="EPSG:32643", bounds=(0, 0, 100, 100), gsd=0.65)
    coregistered, passed, warnings = check_pair(a, b)
    assert coregistered
    assert {"crs_match", "extent_overlap", "size_match", "gsd_match"} <= set(passed)
    assert warnings == []


def test_check_pair_flags_crs_mismatch(tmp_path):
    a = _info(tmp_path / "a.tif", crs="EPSG:32643", bounds=(0, 0, 100, 100))
    b = _info(tmp_path / "b.tif", crs="EPSG:4326", bounds=(0, 0, 100, 100))
    coregistered, passed, warnings = check_pair(a, b)
    assert not coregistered
    assert "crs_match" not in passed
    assert any("CRS differs" in w for w in warnings)


def test_check_pair_flags_disjoint_extents(tmp_path):
    a = _info(tmp_path / "a.tif", crs="EPSG:32643", bounds=(0, 0, 10, 10))
    b = _info(tmp_path / "b.tif", crs="EPSG:32643", bounds=(50, 50, 60, 60))
    coregistered, _, warnings = check_pair(a, b)
    assert not coregistered
    assert any("do not overlap" in w for w in warnings)


def test_check_pair_flags_size_mismatch(tmp_path):
    a = _info(tmp_path / "a.png", size=(64, 48))
    b = _info(tmp_path / "b.png", size=(128, 96))
    _, passed, warnings = check_pair(a, b)
    assert "size_match" not in passed
    assert any("pixel dimensions differ" in w for w in warnings)


def test_benchmark_png_pair_accepted_but_warned(tmp_path):
    """Benchmark pairs carry no geotransform; accept them and say why."""
    a = _info(tmp_path / "a.png")
    b = _info(tmp_path / "b.png")
    coregistered, passed, warnings = check_pair(a, b)
    assert coregistered
    assert "size_match" in passed
    assert any("not georeferenced" in w for w in warnings)
