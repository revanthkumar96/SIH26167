"""Answer normalisation and bounding-box parsing.

Scoring depends as much on parsing model output as on the model itself, so this
module is versioned, unit-tested, and shared by every benchmark. A parse failure is
reported as a failure and scored wrong -- it is never silently dropped, because
dropping unparseable outputs inflates scores.
"""

from __future__ import annotations

import re
from typing import Literal

from satquery.schema import BBox

BoxScale = Literal["auto", "unit", "percent", "milli", "pixel"]

_ARTICLES = frozenset({"a", "an", "the"})

_YES = frozenset({"yes", "yeah", "yep", "true", "correct", "present", "y"})
_NO = frozenset({"no", "nope", "false", "incorrect", "absent", "none", "n"})

_NUMBER_WORDS = {
    "zero": "0",
    "one": "1",
    "two": "2",
    "three": "3",
    "four": "4",
    "five": "5",
    "six": "6",
    "seven": "7",
    "eight": "8",
    "nine": "9",
    "ten": "10",
    "eleven": "11",
    "twelve": "12",
    "thirteen": "13",
    "fourteen": "14",
    "fifteen": "15",
    "sixteen": "16",
    "seventeen": "17",
    "eighteen": "18",
    "nineteen": "19",
    "twenty": "20",
}

_PUNCT = re.compile(r"[^\w\s./%-]")
_WS = re.compile(r"\s+")
_NUM = r"-?\d+(?:\.\d+)?"

# Qwen2-VL / Qwen2.5-VL special-token form: <|box_start|>(x1,y1),(x2,y2)<|box_end|>
_BOX_TOKENS = re.compile(
    rf"<\|box_start\|>\s*\(\s*({_NUM})\s*,\s*({_NUM})\s*\)\s*,?\s*"
    rf"\(\s*({_NUM})\s*,\s*({_NUM})\s*\)\s*<\|box_end\|>"
)
# GeoChat / LLaVA-RS form: {<x1><y1><x2><y2>}
_BOX_ANGLE = re.compile(
    rf"\{{\s*<\s*({_NUM})\s*>\s*<\s*({_NUM})\s*>\s*"
    rf"<\s*({_NUM})\s*>\s*<\s*({_NUM})\s*>\s*\}}"
)
# JSON form emitted by Qwen2.5-VL grounding prompts: "bbox_2d": [x1, y1, x2, y2]
_BOX_JSON = re.compile(
    rf'"bbox(?:_2d)?"\s*:\s*\[\s*({_NUM})\s*,\s*({_NUM})\s*,'
    rf"\s*({_NUM})\s*,\s*({_NUM})\s*\]"
)
# Bare bracket form: [x1, y1, x2, y2]
_BOX_BRACKET = re.compile(
    rf"\[\s*({_NUM})\s*,\s*({_NUM})\s*,\s*({_NUM})\s*,\s*({_NUM})\s*\]"
)
# Paired-parenthesis form without special tokens: (x1,y1),(x2,y2)
_BOX_PAIRS = re.compile(
    rf"\(\s*({_NUM})\s*,\s*({_NUM})\s*\)\s*,\s*\(\s*({_NUM})\s*,\s*({_NUM})\s*\)"
)
_ANY_NUMBERS = re.compile(_NUM)

_BOX_PATTERNS = (_BOX_TOKENS, _BOX_ANGLE, _BOX_JSON, _BOX_BRACKET, _BOX_PAIRS)


def normalize_answer(text: str) -> str:
    """Normalise a free-text answer for exact-match scoring.

    Lowercases, keeps only the first line, strips articles and punctuation, maps
    yes/no variants to a canonical form, and converts number words to digits.
    """
    if not text:
        return ""
    first = text.strip().splitlines()[0] if text.strip() else ""
    lowered = _PUNCT.sub(" ", first.lower())
    # Trailing sentence punctuation survives _PUNCT because "." and "-" are kept
    # for decimals and hyphenated words; strip it at token edges only.
    tokens = [t.strip("./-%") for t in _WS.sub(" ", lowered).strip().split(" ")]
    tokens = [t for t in tokens if t]
    tokens = [t for t in tokens if t not in _ARTICLES]
    tokens = [_NUMBER_WORDS.get(t, t) for t in tokens]
    if not tokens:
        return ""
    if tokens[0] in _YES:
        return "yes"
    if tokens[0] in _NO:
        return "no"
    return " ".join(tokens)


def answers_match(prediction: str, reference: str) -> bool:
    """Exact match after normalisation, with a containment fallback.

    The fallback catches models that answer a closed-vocabulary question in a
    sentence ("There is water present." vs "yes") without opening the door to
    scoring an unrelated long answer as correct.
    """
    pred, ref = normalize_answer(prediction), normalize_answer(reference)
    if not ref:
        return False
    if pred == ref:
        return True
    return len(pred.split()) <= 12 and f" {ref} " in f" {pred} "


def _rescale(
    values: tuple[float, float, float, float],
    scale: BoxScale,
    image_size: tuple[int, int] | None,
) -> tuple[float, ...]:
    """Map raw coordinates into the unit square."""
    if scale == "unit":
        return values
    if scale == "percent":
        return tuple(v / 100.0 for v in values)
    if scale == "milli":
        return tuple(v / 1000.0 for v in values)
    if scale == "pixel":
        if not image_size:
            raise ValueError("scale='pixel' requires image_size")
        width, height = image_size
        return (
            values[0] / width,
            values[1] / height,
            values[2] / width,
            values[3] / height,
        )

    # auto: infer from magnitude. Ambiguous by nature -- prefer an explicit scale
    # in the benchmark config when the model's convention is known.
    peak = max(values)
    if peak <= 1.0:
        return values
    if peak <= 100.0:
        return tuple(v / 100.0 for v in values)
    if peak <= 1000.0:
        return tuple(v / 1000.0 for v in values)
    if image_size:
        return _rescale(values, "pixel", image_size)
    return tuple(v / peak for v in values)


def parse_bbox(
    text: str,
    scale: BoxScale = "auto",
    image_size: tuple[int, int] | None = None,
) -> BBox | None:
    """Extract the first bounding box from model output as xyxy in ``[0, 1]``.

    Returns ``None`` when nothing parseable is present, which the caller scores as
    a miss.
    """
    if not text:
        return None

    raw: tuple[float, float, float, float] | None = None
    for pattern in _BOX_PATTERNS:
        match = pattern.search(text)
        if match:
            raw = tuple(float(g) for g in match.groups())  # type: ignore[assignment]
            break

    if raw is None:
        numbers = _ANY_NUMBERS.findall(text)
        if len(numbers) < 4:
            return None
        raw = tuple(float(n) for n in numbers[:4])  # type: ignore[assignment]

    scaled = _rescale(raw, scale, image_size)
    x1, y1, x2, y2 = (min(max(v, 0.0), 1.0) for v in scaled)
    if x2 < x1:
        x1, x2 = x2, x1
    if y2 < y1:
        y1, y2 = y2, y1
    if x2 <= x1 or y2 <= y1:
        return None
    return (x1, y1, x2, y2)


def xywh_to_xyxy(box: tuple[float, float, float, float]) -> BBox:
    """Convert ``(x, y, w, h)`` to ``(x1, y1, x2, y2)``."""
    x, y, w, h = box
    return (x, y, x + w, y + h)


def normalize_box(
    box: tuple[float, float, float, float],
    image_size: tuple[int, int] | None = None,
    box_format: Literal["xyxy", "xywh"] = "xyxy",
    scale: BoxScale = "auto",
) -> BBox | None:
    """Normalise a ground-truth box from a dataset into unit xyxy."""
    coords = xywh_to_xyxy(box) if box_format == "xywh" else box
    x1, y1, x2, y2 = _rescale(coords, scale, image_size)
    if x2 <= x1 or y2 <= y1:
        return None
    return (
        min(max(x1, 0.0), 1.0),
        min(max(y1, 0.0), 1.0),
        min(max(x2, 0.0), 1.0),
        min(max(y2, 0.0), 1.0),
    )
