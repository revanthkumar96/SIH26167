"""BigEarthNet.txt preparation: cleaning, geometry conversion, sampling.

The cleaning rules carry most of the risk. A wrong box conversion scores zero
silently, and an uncleaned caption teaches the model to name a country it cannot
see -- neither failure raises anything at training time.
"""

from __future__ import annotations

import numpy as np
import pytest

# The preparation path is gated behind the 'data' extra; without it there is
# nothing to test rather than something to fail.
pd = pytest.importorskip("pandas")

from satquery.data.bigearthnet import (  # noqa: E402
    PreparationError,
    build_record,
    is_unanswerable,
    parse_box,
    prepare,
    redact_caption,
    rewrite_prompt,
    sar_composite,
    stack_bands,
    stratified_sample,
    to_milli_box,
    write_jsonl,
)

# Verbatim rows from the published corpus, so the parsers are tested against the
# real formats rather than against what we assume they look like.
REAL_CAPTION = (
    "This satellite image, captured in Austria during summer, depicts a diverse "
    'landscape dominated by agricultural and forested areas within the "cold, no '
    'dry season, warm summer" climate zone. Arable land (~708,000 sqm) is the '
    "most prominent feature."
)
REAL_CAPTION_ALT = (
    "This satellite image, captured during the summer season in Austria, "
    'showcases a predominantly agricultural landscape within the "temperate, no '
    'dry season, warm summer" climate zone. The largest area is arable land.'
)


def test_unanswerable_categories_are_recognised():
    for category in ("country", "season", "climate zone", "Climate Zone"):
        assert is_unanswerable(category)
    for category in ("presence", "area", "count", "adjacency", "point", "reference"):
        assert not is_unanswerable(category)


def test_redaction_removes_country_season_and_climate_zone():
    cleaned = redact_caption(REAL_CAPTION, "Austria")
    assert cleaned is not None
    for leaked in ("Austria", "summer", "climate zone", "captured"):
        assert leaked.lower() not in cleaned.lower()
    # The land-cover content is the part worth keeping.
    assert "Arable land" in cleaned
    assert "708,000 sqm" in cleaned


def test_redaction_handles_the_other_caption_phrasing():
    cleaned = redact_caption(REAL_CAPTION_ALT, "Austria")
    assert cleaned is not None
    assert "Austria" not in cleaned
    assert "climate zone" not in cleaned
    assert "arable land" in cleaned.lower()


def test_caption_still_naming_the_country_is_rejected():
    """A caption we cannot clean confidently is dropped, not patched."""
    stubborn = "A view of rural Austria with mixed forest and pasture."
    assert redact_caption(stubborn, "Austria") is None


def test_box_parsing_and_conversion_to_our_grid():
    assert parse_box("[0.64 0.0, 1.0 0.71]") == (0.64, 0.0, 1.0, 0.71)
    assert to_milli_box((0.64, 0.0, 1.0, 0.71)) == [640, 0, 1000, 710]


def test_box_corners_are_ordered_and_clamped():
    assert parse_box("[0.9 0.8, 0.2 0.1]") == (0.2, 0.1, 0.9, 0.8)
    assert to_milli_box((-0.5, 0.0, 1.7, 1.0)) == [0, 0, 1000, 1000]


def test_box_parsing_rejects_text_without_a_box():
    with pytest.raises(PreparationError):
        parse_box("somewhere in the north-east")


def test_prompt_markup_is_rewritten_to_our_convention():
    prompt = rewrite_prompt(
        "Provide a bounding box for the land cover class instance at "
        "<point>(0.82, 0.28)</point> in the satellite image."
    )
    assert "<point>" not in prompt and "</point>" not in prompt
    assert "(820, 280)" in prompt

    referring = rewrite_prompt(
        "Identify the location of the <ref>largest connected region of pastures</ref>."
    )
    assert "<ref>" not in referring
    assert "largest connected region of pastures" in referring


def _row(**kw):
    base = {
        "ID": 1,
        "patch_id": "P1",
        "type": "binary",
        "category": "presence",
        "input": "Is any portion of the image covered by mixed forest?",
        "output": "yes",
        "country": "Austria",
    }
    base.update(kw)
    return base


def test_build_record_skips_visually_unanswerable_rows():
    row = _row(type="mcq", category="country", input="Which country?", output="c")
    with pytest.raises(PreparationError):
        build_record(row, ("a.png",))


def test_build_record_converts_a_grounding_answer():
    row = _row(type="bounding box", category="reference", output="[0.0 0.33, 0.28 0.8]")
    record = build_record(row, ("s2.png",))
    assert record.answer == "[0, 330, 280, 800]"
    assert record.task == "bounding box"


def test_prepare_drops_unanswerable_rows_and_keeps_the_rest():
    frame = pd.DataFrame(
        [
            _row(ID=1, category="presence"),
            _row(ID=2, category="country", type="mcq", output="c"),
            _row(ID=3, category="season", type="mcq", output="a"),
            _row(ID=4, category="adjacency"),
        ]
    )
    records = list(prepare(frame, {"P1": ("s2.png", "s1.png")}))
    assert [r.sample_id for r in records] == ["1", "4"]
    assert all(r.images == ("s2.png", "s1.png") for r in records)


def test_prepare_skips_patches_with_no_imagery():
    frame = pd.DataFrame([_row(patch_id="missing")])
    assert list(prepare(frame, {"P1": ("s2.png",)})) == []


def test_stratified_sample_spreads_across_countries():
    """The pilot drew everything from one country; this is the guard against that."""
    rows = []
    for country in ("Finland", "Portugal", "Austria"):
        for i in range(50):
            rows.append(_row(ID=f"{country}{i}", country=country, season="Summer"))
    frame = pd.DataFrame(rows)

    sampled = stratified_sample(frame, per_type=30, seed=7)

    assert set(sampled["country"]) == {"Finland", "Portugal", "Austria"}
    # No single country may dominate the draw.
    counts = sampled["country"].value_counts()
    assert counts.max() - counts.min() <= 1


def test_stack_bands_resamples_to_a_common_grid():
    payload = {
        "B04": np.ones((60, 60), dtype=np.float32),
        "B03": np.ones((120, 120), dtype=np.float32) * 2,
        "B02": np.ones((20, 20), dtype=np.float32) * 3,
    }
    stack = stack_bands(payload, ("B04", "B03", "B02"), size=120)
    assert stack.shape == (3, 120, 120)
    assert stack[1].mean() == pytest.approx(2.0)


def test_stack_bands_reports_a_missing_band():
    with pytest.raises(PreparationError, match="B08"):
        stack_bands({"B04": np.zeros((2, 2), dtype=np.float32)}, ("B04", "B08"), size=2)


def test_sar_composite_uses_the_db_difference():
    """BigEarthNet S1 is already in dB, so the ratio plane is a difference."""
    vv = np.full((4, 4), -12.0, dtype=np.float32)
    vh = np.full((4, 4), -19.0, dtype=np.float32)
    stack = sar_composite(vv, vh)
    assert stack.shape == (3, 4, 4)
    assert stack[2].mean() == pytest.approx(7.0)


def test_write_jsonl_round_trips(tmp_path):
    import json

    frame = pd.DataFrame([_row(ID=1), _row(ID=2, category="area")])
    out = tmp_path / "train.jsonl"
    count = write_jsonl(prepare(frame, {"P1": ("s2.png",)}), out)

    assert count == 2
    lines = out.read_text(encoding="utf-8").strip().splitlines()
    first = json.loads(lines[0])
    assert first["images"] == ["s2.png"]
    assert first["conversations"][0]["from"] == "human"
    assert first["conversations"][1]["value"] == "yes"
