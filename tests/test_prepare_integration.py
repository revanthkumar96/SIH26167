"""The preparation script against a real LMDB, end to end.

``render_patches`` had no coverage at all, because it needs ``lmdb`` and that is
behind the ``data`` extra. Two bugs reached the repo through that gap and were
only found by running the script: a numpy truthiness error on the parquet label
column, and a ``Measurements`` field that had been renamed at every call site the
unit tests touched but not in the renderer.

Both were invisible to a green suite, so the renderer gets a test that builds a
store in the published layout and runs the pipeline over it.
"""

from __future__ import annotations

import importlib.util
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("lmdb")
pytest.importorskip("pandas")
pytest.importorskip("safetensors")

if importlib.util.find_spec("rasterio") is None:  # rendering reads back a PNG
    pytest.skip("rasterio is needed to read rendered patches", allow_module_level=True)

REPO = Path(__file__).resolve().parents[1]

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
#: Native ground resolutions differ per band, which is the thing stack_bands has
#: to resample away. A store where every band is already 120x120 would not
#: exercise that at all.
NATIVE = {
    "B01": 20,
    "B09": 20,
    "B05": 60,
    "B06": 60,
    "B07": 60,
    "B8A": 60,
    "B11": 60,
    "B12": 60,
    "B02": 120,
    "B03": 120,
    "B04": 120,
    "B08": 120,
}


def build_store(root: Path, patches: int = 6) -> Path:
    """A BigEarthNet v2.0 LMDB plus the two parquets, in the published layout."""
    import lmdb
    import pandas as pd
    from safetensors.numpy import save as st_save

    root.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(7)
    ids = [f"S2A_MSIL2A_2017_N{i:03d}_30_40" for i in range(patches)]
    s1_of = {p: p.replace("S2A_MSIL2A", "S1A_IW_GRDH") for p in ids}

    env = lmdb.open(str(root / "BENv2.lmdb"), map_size=1 << 30)
    with env.begin(write=True) as txn:
        for i, patch in enumerate(ids):
            s2 = {}
            for band in S2_BANDS:
                n = NATIVE[band]
                ramp = np.linspace(200, 3000, n, dtype=np.float32)[None, :].repeat(n, 0)
                s2[band] = (ramp + rng.normal(0, 150, (n, n))).astype(np.float32)
            if i % 3 == 0:  # green >> nir, so NDWI reads as water
                s2["B03"] = s2["B03"] * 3.0
            txn.put(patch.encode(), st_save(s2))
            # Sentinel-1 in BigEarthNet is already in dB: negative, published
            # means -12.64 (VV) and -19.35 (VH).
            txn.put(
                s1_of[patch].encode(),
                st_save(
                    {
                        "VV": rng.normal(-12.64, 3, (120, 120)).astype(np.float32),
                        "VH": rng.normal(-19.35, 3, (120, 120)).astype(np.float32),
                    }
                ),
            )
    env.close()

    labels = [["Arable land", "Pastures"], ["Inland waters"], ["Urban fabric"]]
    pd.DataFrame(
        [
            {
                "patch_id": p,
                "labels": labels[i % len(labels)],
                "split": "train",
                "s1_name": s1_of[p],
                # One patch under cloud, one under snow: both must be dropped.
                "contains_cloud_or_shadow": i == 1,
                "contains_seasonal_snow": i == 2,
            }
            for i, p in enumerate(ids)
        ]
    ).to_parquet(root / "metadata.parquet")

    rows = []
    for i, p in enumerate(ids):
        country = ["Lithuania", "Austria", "Portugal"][i % 3]
        rows += [
            {
                "ID": f"{p}-b",
                "s1_name": s1_of[p],
                "patch_id": p,
                "input": "Is there water in this image?",
                "output": "yes" if i % 3 == 0 else "no",
                "type": "binary",
                "category": "water",
                "split": "train",
                "country": country,
                "season": "summer",
            },
            {
                "ID": f"{p}-c",
                "s1_name": s1_of[p],
                "patch_id": p,
                "input": "Describe the image.",
                "output": (
                    "This satellite image, captured in Austria during summer, "
                    'depicts farmland within the "cold, no dry season, warm '
                    'summer" climate zone. Arable land is prominent.'
                ),
                "type": "captioning",
                "category": "land cover",
                "split": "train",
                "country": "Austria",
                "season": "summer",
            },
            {
                "ID": f"{p}-m",
                "s1_name": s1_of[p],
                "patch_id": p,
                "input": "Identify the country: a) Lithuania b) Belgium",
                "output": "a",
                "type": "mcq",
                "category": "country",
                "split": "train",
                "country": country,
                "season": "summer",
            },
            {
                "ID": f"{p}-x",
                "s1_name": s1_of[p],
                "patch_id": p,
                "input": "<ref>the field</ref> <point>(0.82, 0.28)</point>",
                "output": "[0.64 0.0, 1.0 0.71]",
                "type": "bounding box",
                "category": "object",
                "split": "train",
                "country": country,
                "season": "summer",
            },
        ]
    pd.DataFrame(rows).to_parquet(root / "BigEarthNet.txt.parquet")
    return root


@pytest.fixture(scope="module")
def prepared(tmp_path_factory):
    """Run the actual script, as an operator would, and return its output."""
    base = tmp_path_factory.mktemp("ben")
    store = build_store(base / "store")
    out = base / "prepared"

    result = subprocess.run(
        [
            sys.executable,
            str(REPO / "scripts" / "prepare_bigearthnet.py"),
            "--lmdb",
            str(store / "BENv2.lmdb"),
            "--parquet",
            str(store / "BigEarthNet.txt.parquet"),
            "--metadata",
            str(store / "metadata.parquet"),
            "--out",
            str(out),
            "--per-type",
            "20",
            "--preamble-rate",
            "1.0",
        ],
        capture_output=True,
        text=True,
        cwd=REPO,
        timeout=600,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    return out, result.stdout


def records(prepared):
    out, _ = prepared
    return [
        json.loads(line)
        for line in (out / "train.jsonl").read_text(encoding="utf-8").splitlines()
    ]


def test_the_script_runs_and_writes_a_corpus(prepared):
    out, _ = prepared
    assert (out / "train.jsonl").is_file()
    assert (out / "manifest.json").is_file()
    assert records(prepared)


def test_both_renderings_exist_for_every_patch(prepared):
    """One optical and one SAR PNG each -- the cross-modal pair the model sees."""
    out, _ = prepared
    manifest = json.loads((out / "manifest.json").read_text(encoding="utf-8"))
    images = sorted((out / "images").glob("*.png"))
    assert len(images) == manifest["patches"] * 2
    assert any(p.name.endswith("_optical.png") for p in images)
    assert any(p.name.endswith("_sar.png") for p in images)


def test_renderings_are_readable_rgb_at_patch_size(prepared):
    from PIL import Image

    out, _ = prepared
    for path in sorted((out / "images").glob("*.png"))[:4]:
        with Image.open(path) as image:
            assert image.mode == "RGB"
            # 120x120 at 10 m, after the 20 m and 60 m bands are resampled up.
            assert image.size == (120, 120)


def test_cloud_and_snow_patches_are_dropped(prepared):
    """Two of six patches are flagged, and neither may appear in the corpus."""
    out, _ = prepared
    manifest = json.loads((out / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["quality_filter_applied"] is True
    assert manifest["patches_excluded_cloud_or_snow"] == 2
    assert manifest["rows_after_quality_filter"] < manifest["rows_before_clean"]


def test_unanswerable_categories_never_reach_the_corpus(prepared):
    """A country MCQ teaches geographic hallucination on unseen regions."""
    for record in records(prepared):
        turn = record["conversations"][0]["value"]
        assert "Identify the country" not in turn


def test_captions_lose_their_acquisition_metadata(prepared):
    for record in records(prepared):
        if record["task"] != "captioning":
            continue
        answer = record["conversations"][1]["value"]
        assert "captured in" not in answer.lower()
        assert "climate zone" not in answer.lower()
        assert "Austria" not in answer
        assert "Arable land" in answer  # the part worth keeping survives


def test_boxes_are_converted_to_the_milli_grid(prepared):
    seen = False
    for record in records(prepared):
        if record["task"] != "bounding box":
            continue
        seen = True
        assert json.loads(record["conversations"][1]["value"]) == [640, 0, 1000, 710]
        # <ref> stripped and <point> rewritten to the 0-1000 convention.
        turn = record["conversations"][0]["value"]
        assert "<ref>" not in turn and "<point>" not in turn
        assert "(820, 280)" in turn
    assert seen, "no grounding rows survived preparation"


def test_preambles_are_rendered_by_format_evidence(prepared):
    """The parity contract, checked on what the script actually wrote."""
    from satquery.agent.tools.vlm import format_evidence

    carried = [r for r in records(prepared) if r["evidence_kind"] != "none"]
    assert carried, "no record carried a preamble at --preamble-rate 1.0"

    header = format_evidence({"sar_water_fraction": 0.5}).splitlines()[0]
    for record in carried:
        turn = record["conversations"][0]["value"]
        assert turn.startswith(header)
        # Exactly the two-newline join the controller uses.
        assert "\n\n" in turn


def test_the_sar_constant_is_not_in_the_corpus(prepared):
    """SAR contributes a location; its bright-tail fraction is ~5% on every scene."""
    for record in records(prepared):
        turn = record["conversations"][0]["value"]
        assert "built-up fraction from SAR" not in turn
        if record["evidence_kind"] != "none":
            assert "location of brightest SAR returns" in turn


def test_manifest_records_the_mixture(prepared):
    out, _ = prepared
    manifest = json.loads((out / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["seed"] == 1234
    assert manifest["prompt_version"]
    assert set(manifest["dropped_categories"]) == {"country", "season", "climate zone"}
    assert sum(manifest["evidence_mix"].values()) == manifest["records"]
