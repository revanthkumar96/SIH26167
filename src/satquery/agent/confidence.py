"""Confidence estimation and specialist/VLM agreement.

Three sources, as set out in the architecture: specialist heads report their own
calibrated value, the VLM gets a lexical estimate, and agreement between the two
moves the final number in either direction.

A bare 1.0 is never returned. An honest, moving confidence is worth more to a user
than a confident wrong answer, and disagreement is surfaced as a warning rather than
quietly averaged away.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence

CONFIDENCE_CEILING = 0.92
CONFIDENCE_FLOOR = 0.05

#: Words that mark the model declining to commit.
_HEDGES = frozenset(
    {
        "may",
        "might",
        "possibly",
        "perhaps",
        "unclear",
        "unsure",
        "uncertain",
        "cannot",
        "can't",
        "unknown",
        "appears",
        "seems",
        "likely",
        "probably",
        "difficult",
        "hard",
    }
)

_NEGATIVE = frozenset({"no", "none", "nothing", "unchanged", "not"})
_POSITIVE = frozenset({"yes", "increased", "increase", "more", "grown", "expanded"})

#: A change mask below this fraction is treated as a quiet scene.
_QUIET = 0.02
#: Above this, claiming "nothing changed" contradicts the measurement.
_LOUD = 0.10


def text_confidence(text: str) -> float:
    """Lexical confidence for a generated answer.

    A short, unhedged answer to a closed question is the high-confidence case; an
    empty or hedged answer is the low one.
    """
    if not text or not text.strip():
        return CONFIDENCE_FLOOR

    tokens = {t.strip(".,;:!?").lower() for t in text.split()}
    score = 0.78

    if tokens & _HEDGES:
        score -= 0.22
    if len(tokens) > 60:
        score -= 0.08
    elif len(tokens) <= 3:
        score += 0.06

    return round(min(max(score, CONFIDENCE_FLOOR), CONFIDENCE_CEILING), 3)


def agreement_adjustment(
    answer: str, artifacts: Mapping[str, object]
) -> tuple[float, str | None]:
    """Compare a generated answer against what the specialists measured.

    Returns an additive adjustment and, when they conflict, a warning the trace
    surfaces to the user.
    """
    tokens = {t.strip(".,;:!?").lower() for t in answer.split()}
    changed = artifacts.get("changed_area_frac")

    if isinstance(changed, (int, float)):
        says_nothing = bool(tokens & _NEGATIVE)
        says_something = bool(tokens & _POSITIVE)
        if says_nothing and changed > _LOUD:
            return -0.18, (
                f"the answer reports no change, but the change mask covers "
                f"{changed:.1%} of the scene"
            )
        if says_something and changed < _QUIET:
            return -0.15, (
                f"the answer reports change, but the change mask covers only "
                f"{changed:.1%} of the scene"
            )
        if (says_nothing and changed < _QUIET) or (says_something and changed > _QUIET):
            return 0.10, None

    sar_water = artifacts.get("sar_water_fraction")
    optical_water = artifacts.get("optical_water_fraction")
    if isinstance(sar_water, (int, float)) and isinstance(optical_water, (int, float)):
        gap = abs(sar_water - optical_water)
        if gap > 0.15:
            return -0.12, (
                f"optical and SAR disagree on water extent "
                f"({optical_water:.1%} vs {sar_water:.1%}); cloud or shadow is likely"
            )
        return 0.08, None

    return 0.0, None


def blend(values: Sequence[float | None]) -> float | None:
    """Mean of the confidences that were actually reported."""
    present = [v for v in values if v is not None]
    if not present:
        return None
    return round(
        min(max(sum(present) / len(present), CONFIDENCE_FLOOR), CONFIDENCE_CEILING), 3
    )
