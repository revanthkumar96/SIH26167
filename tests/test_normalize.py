"""Parsing is scored as much as the model is, so it gets real tests."""

import pytest

from satquery.eval.normalize import (
    answers_match,
    normalize_answer,
    normalize_box,
    parse_bbox,
)


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("Yes.", "yes"),
        ("  YEP  ", "yes"),
        ("No, there is not.", "no"),
        ("The residential area", "residential area"),
        ("three", "3"),
        ("Two buildings\nand more text", "2 buildings"),
        ("", ""),
    ],
)
def test_normalize_answer(raw, expected):
    assert normalize_answer(raw) == expected


def test_answers_match_exact_and_contained():
    assert answers_match("yes", "Yes")
    assert answers_match("It is a forest", "forest")
    assert not answers_match("forest", "water")


def test_answers_match_rejects_long_rambles():
    rambling = " ".join(["padding"] * 20) + " water"
    assert not answers_match(rambling, "water")


def test_answers_match_empty_reference_is_false():
    assert not answers_match("anything", "")


@pytest.mark.parametrize(
    "text",
    [
        "<|box_start|>(100,200),(300,400)<|box_end|>",
        '{"bbox_2d": [100, 200, 300, 400]}',
        "[100, 200, 300, 400]",
        "(100,200),(300,400)",
        "The region is at [100, 200, 300, 400].",
    ],
)
def test_parse_bbox_formats_agree(text):
    assert parse_bbox(text, scale="milli") == pytest.approx((0.1, 0.2, 0.3, 0.4))


def test_parse_bbox_geochat_angle_form():
    assert parse_bbox("{<10><20><30><40>}", scale="percent") == pytest.approx(
        (0.1, 0.2, 0.3, 0.4)
    )


def test_parse_bbox_auto_scale_infers_magnitude():
    assert parse_bbox("[0.1, 0.2, 0.3, 0.4]") == pytest.approx((0.1, 0.2, 0.3, 0.4))
    assert parse_bbox("[10, 20, 30, 40]") == pytest.approx((0.1, 0.2, 0.3, 0.4))


def test_parse_bbox_swaps_inverted_corners():
    assert parse_bbox("[300, 400, 100, 200]", scale="milli") == pytest.approx(
        (0.1, 0.2, 0.3, 0.4)
    )


@pytest.mark.parametrize("text", ["", "no box here", "only 1, 2 numbers"])
def test_parse_bbox_returns_none_when_unparseable(text):
    assert parse_bbox(text) is None


def test_parse_bbox_rejects_degenerate_box():
    assert parse_bbox("[100, 200, 100, 200]", scale="milli") is None


def test_parse_bbox_pixel_scale_needs_image_size():
    with pytest.raises(ValueError, match="image_size"):
        parse_bbox("[10, 20, 30, 40]", scale="pixel")


def test_normalize_box_xywh_to_unit():
    box = normalize_box(
        (100, 100, 200, 200), image_size=(1000, 1000), box_format="xywh", scale="pixel"
    )
    assert box == pytest.approx((0.1, 0.1, 0.3, 0.3))
