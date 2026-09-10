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


# -- box formats seen from real models -----------------------------------
#
# Pinned after a live run against Qwen3-VL, which emits a fenced JSON array
# rather than the bare `[x1, y1, x2, y2]` the prompt asks for. A grounding
# answer that cannot be parsed scores zero and reads as model failure, so every
# shape a model has actually produced gets a test.


@pytest.mark.parametrize(
    ("label", "text"),
    [
        (
            "qwen fenced json",
            '```json\n[{"bbox_2d": [801, 208, 875, 305], "label": "ship"}]\n```',
        ),
        ("json array", '[{"bbox_2d": [801, 208, 875, 305]}]'),
        ("bare bbox key", '{"bbox": [801, 208, 875, 305]}'),
        ("bare bracket", "[801, 208, 875, 305]"),
        ("keyed xyxy", '{"x1": 801, "y1": 208, "x2": 875, "y2": 305}'),
        ("keyed edges", '{"left": 801, "top": 208, "right": 875, "bottom": 305}'),
        ("keyed minmax", '{"xmin": 801, "ymin": 208, "xmax": 875, "ymax": 305}'),
        ("angle form", "{<801><208><875><305>}"),
        ("box tags", "<box>801, 208, 875, 305</box>"),
        ("qwen special tokens", "<|box_start|>(801,208),(875,305)<|box_end|>"),
        ("ref plus box", "<ref>ship</ref><box>(801,208),(875,305)</box>"),
        ("prose with numbers", "The ship is at approximately 801, 208 to 875, 305."),
        ("reversed corners", "[875, 305, 801, 208]"),
    ],
)
def test_every_observed_box_format_parses(label, text):
    assert parse_bbox(text, scale="milli") == pytest.approx(
        (0.801, 0.208, 0.875, 0.305)
    ), label


def test_identifier_digits_are_not_read_as_coordinates():
    """The numeric fallback must not scrape digits out of key names.

    `{"x1": 801, ...}` contains the numbers 1, 801, 1, 208 before it contains
    the box. A fallback that takes the first four builds a box out of the
    *labels*, which is worse than returning nothing: it scores as a confident
    miss rather than a parse failure.
    """
    box = parse_bbox('{"x1": 801, "y1": 208, "x2": 875, "y2": 305}', scale="milli")
    assert box == pytest.approx((0.801, 0.208, 0.875, 0.305))


def test_the_first_box_wins_when_several_are_offered():
    """Grounding asks for one region; a list means the model hedged."""
    text = '[{"bbox_2d":[801,208,875,305]},{"bbox_2d":[10,20,30,40]}]'
    assert parse_bbox(text, scale="milli") == pytest.approx(
        (0.801, 0.208, 0.875, 0.305)
    )


@pytest.mark.parametrize(
    ("label", "text"),
    [
        ("empty", ""),
        ("refusal", "I cannot determine the location from this image."),
        ("prose, no numbers", "The ship is near the top right of the image."),
        ("too few numbers", "[801, 208]"),
        ("zero area", "[801, 208, 801, 208]"),
    ],
)
def test_unparseable_output_returns_none_rather_than_a_guess(label, text):
    """A miss must be reported as a miss.

    grounding_metrics scores None as IoU 0, which is the honest answer. Inventing
    a box from whatever numbers are lying about would inflate the score.
    """
    assert parse_bbox(text, scale="milli") is None, label
