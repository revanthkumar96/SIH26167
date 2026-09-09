"""Robustness against the hidden evaluation imagery.

The evaluation set is Cartosat-2S plus RISAT over India, and it breaks three
assumptions the code was written under: that optical input is a 12-band
Sentinel-2 stack, that SWIR exists, and that ground sample distance is around
10 m. None of those hold.

The behaviour these tests pin down is already correct. That is exactly why they
are worth writing: the roadmap lists this path as *assumed* rather than proven,
and an assumption nobody exercises is one a later refactor can quietly break.
The failure mode is not an exception -- it is `optical_indices` inventing a
number from bands that cannot produce one, or a 1-band raster falling through a
branch written for three.

Fixtures are synthesised rather than downloaded: we are told to plan as though
Cartosat imagery will never be obtained, so the shape of the input is modelled
from its published specification and nothing depends on getting the real thing.
"""

from __future__ import annotations

import numpy as np
import pytest

rasterio = pytest.importorskip("rasterio")

from rasterio.transform import Affine  # noqa: E402

from satquery.agent.controller import Controller  # noqa: E402
from satquery.eval.backends import build_backend  # noqa: E402
from satquery.geo.raster import (  # noqa: E402
    looks_like_linear_backscatter,
    preview_bands,
    read_bands,
    read_info,
    to_rgb8,
)
from satquery.schema import InputConfig, Task  # noqa: E402

#: Cartosat-2S panchromatic is ~0.65 m; its multispectral bands are ~1.6 m.
PAN_GSD = 0.65
MX_GSD = 1.6


def geotiff(path, data, gsd, crs="EPSG:32643"):
    """Write a georeferenced raster in a UTM zone over India."""
    bands, height, width = data.shape
    with rasterio.open(
        path,
        "w",
        driver="GTiff",
        height=height,
        width=width,
        count=bands,
        dtype=data.dtype.name,
        crs=crs,
        # Built directly rather than via from_origin(), which multiplies two
        # Affines with `*` and trips affine's PendingDeprecationWarning -- and
        # this suite treats warnings as errors on purpose.
        transform=Affine(gsd, 0.0, 700000.0, 0.0, -gsd, 2000000.0),
    ) as dst:
        dst.write(data)
    return path


def cartosat_pan(tmp_path, name="pan.tif"):
    """Single-band panchromatic, sub-metre. The shape with no colour at all."""
    rng = np.random.default_rng(0)
    data = rng.integers(0, 1023, (1, 64, 64), dtype=np.uint16)
    return geotiff(tmp_path / name, data, PAN_GSD)


def cartosat_mx(tmp_path, name="mx.tif"):
    """Four-band multispectral: blue, green, red, NIR. No SWIR anywhere."""
    rng = np.random.default_rng(1)
    data = rng.integers(0, 4000, (4, 64, 64), dtype=np.uint16)
    return geotiff(tmp_path / name, data, MX_GSD)


def _backscatter_scene(rng, shape=(2, 64, 64)):
    """Linear backscatter over a mixed scene: water, land, built-up.

    A single exponential is not good enough here. Real gamma0 spans roughly
    0.001 to 300 because a scene mixes near-specular water, moderate vegetated
    land and very bright urban double-bounce, and it is that dynamic range --
    not the shape of one distribution -- that
    ``looks_like_linear_backscatter`` keys on. Drawn from one exponential the
    fixture has a p99/p50 near 6.6 and the guard correctly declines to call it
    linear, which says the fixture is wrong rather than the guard.
    """
    classes = rng.choice(3, size=shape, p=[0.25, 0.6, 0.15])
    scale = np.select([classes == 0, classes == 1, classes == 2], [0.004, 0.10, 2.5])
    return rng.exponential(scale).astype(np.float32)


def risat_sar(tmp_path, name="risat.tif", linear=False):
    """Dual-polarisation SAR, in dB by default.

    The two forms are the same scene, so a test that asserts dB is left alone
    and one that asserts linear is detected are talking about one physical
    input rather than two unrelated arrays.
    """
    rng = np.random.default_rng(2)
    power = _backscatter_scene(rng)
    data = power if linear else (10.0 * np.log10(np.clip(power, 1e-6, None)))
    return geotiff(tmp_path / name, data.astype(np.float32), MX_GSD)


# -- the raster layer must not crash on one band -------------------------


def test_panchromatic_reads_without_colour(tmp_path):
    """One band has no true-colour rendering, and saying so beats guessing."""
    path = cartosat_pan(tmp_path)
    info = read_info(path)

    assert info.band_count == 1
    assert info.georeferenced
    # Sub-metre, not the ~10 m the Sentinel-shaped code was written around.
    assert info.gsd_m == pytest.approx(PAN_GSD)
    assert preview_bands(info.band_count) is None


def test_panchromatic_still_renders_a_preview(tmp_path):
    """`None` from preview_bands must not mean "no image for the user"."""
    bands = read_bands(cartosat_pan(tmp_path))
    assert bands.shape[0] == 1
    assert to_rgb8(bands).shape == (64, 64, 3)


def test_multispectral_reads_as_four_bands(tmp_path):
    info = read_info(cartosat_mx(tmp_path))
    assert info.band_count == 4
    assert info.gsd_m == pytest.approx(MX_GSD)
    assert preview_bands(info.band_count) == (1, 2, 3)


# -- indices degrade rather than fabricate -------------------------------


def run_crossmodal(tmp_path, optical, sar, query="Where is water and built-up area?"):
    with build_backend("echo") as backend:
        controller = Controller(backend, workroot=tmp_path / "runs")
        return controller.run(query, [optical, sar])


def step_named(trace, name):
    return next(step for step in trace.steps if step.tool == name)


def test_panchromatic_reports_indices_inapplicable_and_the_run_continues(tmp_path):
    """No NIR means no NDWI. The tool must say so, not return a plausible number."""
    trace = run_crossmodal(tmp_path, cartosat_pan(tmp_path), risat_sar(tmp_path))

    optical = step_named(trace, "optical_indices")
    assert optical.outputs["applicable"] is False
    assert "near-infrared" in optical.outputs["reason"]
    assert "water_fraction" not in optical.outputs

    # The run reaches the model regardless: a missing index degrades the
    # evidence, it does not abort the query.
    assert trace.routed_task is Task.CROSSMODAL_VQA
    assert step_named(trace, "vlm_crossmodal_vqa")


def test_multispectral_computes_ndwi_but_not_ndbi(tmp_path):
    """The precise asymmetry the hidden set creates: NIR yes, SWIR no."""
    trace = run_crossmodal(tmp_path, cartosat_mx(tmp_path), risat_sar(tmp_path))

    optical = step_named(trace, "optical_indices")
    assert optical.outputs["applicable"] is True
    assert 0.0 <= optical.outputs["water_fraction"] <= 1.0
    assert "builtup_fraction" not in optical.outputs, (
        "NDBI needs SWIR, which 4-band MX does not carry"
    )


def test_builtup_falls_back_to_sar_and_reaches_the_model(tmp_path):
    """This is the cross-modal complementarity the problem statement tests.

    With no SWIR the only built-up signal is radar, so the check is not merely
    that SAR measured something -- it is that the number arrives in the prompt.
    A measurement computed and never injected is the failure that looks like
    model weakness.
    """
    from satquery.agent.tools.vlm import format_evidence

    trace = run_crossmodal(tmp_path, cartosat_mx(tmp_path), risat_sar(tmp_path))
    sar = step_named(trace, "sar_indices")

    # Where the bright returns are, not what fraction they cover: the fraction
    # above a percentile is ~5% on every scene and carries no information.
    assert sar.outputs["builtup_location"]
    preamble = format_evidence(
        {"sar_builtup_location": sar.outputs["builtup_location"]}
    )
    assert "location of brightest SAR returns" in preamble


def test_single_panchromatic_image_routes_and_answers(tmp_path):
    """A lone PAN scene is a valid query, not an unsupported input."""
    with build_backend("echo") as backend:
        trace = Controller(backend, workroot=tmp_path / "runs").run(
            "What land cover is visible?", [cartosat_pan(tmp_path)]
        )
    assert trace.input_check.config is InputConfig.SINGLE
    assert trace.routed_task is Task.VQA
    assert trace.answer


# -- the decibel trap ----------------------------------------------------


def test_decibel_sar_is_not_converted_again(tmp_path):
    """RISAT in dB must not be pushed through a linear-to-dB pass.

    The scar this guards: SAR linear power read as dB reported 94% water on a
    scene where optical NDWI said 31.5%.
    """
    bands = read_bands(risat_sar(tmp_path, linear=False))
    assert looks_like_linear_backscatter(bands) is False


def test_linear_sar_is_still_detected(tmp_path):
    """The guard has to keep firing on the input it was written for."""
    bands = read_bands(risat_sar(tmp_path, name="linear.tif", linear=True))
    assert looks_like_linear_backscatter(bands) is True


# -- geometry ------------------------------------------------------------


def test_resolution_mismatch_is_warned_not_hidden(tmp_path):
    """PAN at 0.65 m against SAR at 1.6 m is not co-registered by our tolerance."""
    trace = run_crossmodal(tmp_path, cartosat_pan(tmp_path), risat_sar(tmp_path))
    warnings = " ".join(trace.input_check.warnings)
    assert "ground sample distance" in warnings
    assert trace.input_check.coregistered is False


def test_matched_resolution_pair_is_accepted(tmp_path):
    """MX and SAR at the same GSD over the same footprint must pass cleanly."""
    trace = run_crossmodal(tmp_path, cartosat_mx(tmp_path), risat_sar(tmp_path))
    assert trace.input_check.coregistered is True
    assert not [w for w in trace.input_check.warnings if "co-registered" in w]
