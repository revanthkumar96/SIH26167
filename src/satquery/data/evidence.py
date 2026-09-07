"""Evidence-preamble synthesis for the adaptation corpus.

At inference the model never sees a bare task prompt. The controller runs its
deterministic specialists first and prepends what they measured:

    Measurements from image-analysis tools that have already run on these
    images. Treat them as reliable and do not contradict them:
    - water fraction from SAR backscatter: 0.3151
    - built-up fraction from optical NDBI: 0.2803

    <task prompt>

A model adapted only on bare prompts has never seen that. It either ignores the
measurements -- which switches off the evidence grounding that distinguishes this
system -- or, worse, having been made more confident in its own visual reading by
the fine-tune, contradicts them more often, firing the contradiction-retry
constantly and doubling latency for worse answers. This is the same class of bug
as image-normalisation skew, in the prompt channel, and it would silently waste
the entire fine-tune.

So a share of training records carry a preamble, and it is rendered by the
*same* ``format_evidence()`` the controller calls rather than by a copy of its
wording. A test asserts the two agree byte-for-byte.

**Where the numbers come from.** Not from templates over the labels: the
measurements are computed from the patch's own pixels with the same functions
that will run at inference -- ``ndwi_water``/``ndbi_builtup`` on the 12-band
stack, and Otsu over the rendered SAR composite through the serving path's own
``to_gray``. So a training preamble is not merely shaped like an inference one,
it is the one that patch would actually produce.

**The mix.** Three cases, and the third is the one that matters:

*decisive* -- the question is about water or built-up and the measurement
supports the gold answer. Teaches the model to use the evidence.

*orthogonal* -- the question is about something the measurement does not speak
to. Teaches it not to over-apply evidence to every question it sees.

*contradictory* -- never emitted. A preamble that disagrees with its own gold
answer teaches the model that the measurements are unreliable, which is the
precise opposite of what the preamble says about itself. Where the check cannot
establish agreement, the record is emitted with **no** preamble rather than a
guessed one.
"""

from __future__ import annotations

import hashlib
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Any

import numpy as np

from satquery.agent.tools.indices import (
    band_plan,
    mask_fraction,
    ndbi_builtup,
    ndwi_water,
)
from satquery.agent.tools.vlm import format_evidence

#: Share of records that carry a preamble. The design target is 40-50%: enough
#: that the model learns the format is normal, not so much that it stops coping
#: with the bare prompts it still meets on single-image runs with no precursor.
DEFAULT_PREAMBLE_RATE = 0.45

#: A fraction at or below this reads as "essentially none of the scene", and one
#: at or above the second as "unmistakably present". Between them the evidence is
#: equivocal and is not used to call a record decisive -- the gap is deliberate
#: dead space, because a borderline measurement asserted firmly is how a
#: contradiction gets trained in.
ABSENT_BELOW = 0.02
PRESENT_ABOVE = 0.10

_WATER_TERMS = (
    "water",
    "lake",
    "river",
    "sea ",
    "coastal",
    "wetland",
    "marine",
    "reservoir",
    "lagoon",
    "estuar",
    "harbour",
    "harbor",
)
_BUILTUP_TERMS = (
    "urban",
    "built-up",
    "built up",
    "builtup",
    "industrial",
    "residential",
    "construction",
    "artificial surface",
    "settlement",
    "city",
    "town",
    "airport",
    "road",
)

_AFFIRMATIVE = frozenset({"yes", "true", "present", "y"})
_NEGATIVE = frozenset({"no", "false", "absent", "none", "n"})
_WORD = re.compile(r"[a-z][a-z-]+")


class EvidenceKind(str, Enum):
    """Why a record carries the preamble it does."""

    NONE = "none"
    DECISIVE = "decisive"
    ORTHOGONAL = "orthogonal"


@dataclass(frozen=True, slots=True)
class Measurements:
    """What the specialists measure on one BigEarthNet patch.

    Keys match the controller's artifact bag exactly, because they are handed
    straight to ``format_evidence()``. A key renamed here and not there produces
    a preamble with a line silently missing.
    """

    optical_water_fraction: float | None = None
    optical_builtup_fraction: float | None = None
    sar_water_fraction: float | None = None
    sar_builtup_fraction: float | None = None
    landcover_classes: str | None = None

    def as_artifacts(self) -> dict[str, Any]:
        return {
            key: value
            for key, value in (
                ("optical_water_fraction", self.optical_water_fraction),
                ("optical_builtup_fraction", self.optical_builtup_fraction),
                ("sar_water_fraction", self.sar_water_fraction),
                ("sar_builtup_fraction", self.sar_builtup_fraction),
                ("landcover_classes", self.landcover_classes),
            )
            if value is not None
        }


# -- measuring -----------------------------------------------------------


def measure_optical(stack: np.ndarray) -> dict[str, float]:
    """NDWI and NDBI over a band stack, via the serving path's own functions.

    Takes the 12-band array rather than a rendered RGB because NDWI needs NIR and
    NDBI needs SWIR, and neither survives the true-colour composite the model is
    shown. That asymmetry is the point of the whole evidence design: the tools
    see bands the model cannot.
    """
    plan = band_plan(int(stack.shape[0]))
    if plan is None or "nir" not in plan:
        return {}

    nir = stack[plan["nir"] - 1]
    out: dict[str, float] = {}
    if "green" in plan:
        out["optical_water_fraction"] = mask_fraction(
            ndwi_water(stack[plan["green"] - 1], nir)
        )
    if "swir" in plan:
        out["optical_builtup_fraction"] = mask_fraction(
            ndbi_builtup(stack[plan["swir"] - 1], nir)
        )
    return out


def measure_sar(sar_png: str | Path, builtup_percentile: float = 95.0) -> dict[str, float]:
    """Water and built-up extent from the rendered SAR composite.

    Reads the PNG that was just written rather than the raw bands, and does it
    through ``to_gray`` and ``otsu_threshold`` -- the exact calls ``sar_indices``
    makes at inference. Measuring the same file the same way is what makes the
    number in a training preamble the number that patch really produces.
    """
    from satquery.agent.tools._imaging import otsu_threshold, to_gray

    gray = to_gray(Path(sar_png))
    threshold = otsu_threshold(gray)
    return {
        "sar_water_fraction": mask_fraction(gray < threshold),
        "sar_builtup_fraction": mask_fraction(
            gray > float(np.percentile(gray, builtup_percentile))
        ),
    }


def describe_classes(labels: Sequence[str], limit: int = 4) -> str | None:
    """Render CORINE labels the way the land-cover CNN emits them.

    Delegates to the classifier's own renderer rather than reproducing it. If
    the CNN emits this line at inference and no training record ever contained
    one, the adapted model has never seen it and the CNN's numbers reach the
    prompt to no effect -- so the format is settled before the fine-tune and
    there is exactly one function that produces it.
    """
    from satquery.cnn.labels import render_evidence_line

    return render_evidence_line(labels, limit=limit)


# -- the mixing policy ---------------------------------------------------


def _mentions(text: str, terms: Sequence[str]) -> bool:
    lowered = f" {text.lower()} "
    return any(term in lowered for term in terms)


def _polarity(answer: str) -> bool | None:
    """Whether a closed answer asserts presence, denies it, or neither."""
    tokens = set(_WORD.findall(answer.lower()))
    if tokens & _AFFIRMATIVE:
        return True
    if tokens & _NEGATIVE:
        return False
    return None


def _agrees(fraction: float | None, asserts_present: bool | None, text: str) -> bool:
    """Whether a measurement supports what the gold answer says.

    Deliberately strict. Returning False costs one preamble; returning True
    wrongly trains the model that measurements can be argued with.
    """
    if fraction is None:
        return False
    if asserts_present is None:
        # Not a yes/no answer. The answer naming the feature is the only
        # assertion available, so require the measurement to back it up.
        return fraction >= PRESENT_ABOVE if text else False
    if asserts_present:
        return fraction >= PRESENT_ABOVE
    return fraction <= ABSENT_BELOW


def _selected(sample_id: str, seed: int, rate: float) -> bool:
    """Stable per-record draw, so a rerun rebuilds the identical corpus.

    Hashed rather than drawn from a stream: preparation may be resumed, sharded
    or reordered, and a stream position would make the mixture depend on how the
    run happened to be executed.
    """
    digest = hashlib.sha256(f"{seed}:{sample_id}".encode()).digest()
    return (int.from_bytes(digest[:4], "big") / 0xFFFFFFFF) < rate


@dataclass(frozen=True, slots=True)
class EvidenceChoice:
    """The preamble chosen for one record, and why."""

    kind: EvidenceKind
    preamble: str = ""
    artifacts: Mapping[str, Any] = None  # type: ignore[assignment]

    @property
    def used(self) -> bool:
        return self.kind is not EvidenceKind.NONE and bool(self.preamble)


def choose_evidence(
    sample_id: str,
    prompt: str,
    answer: str,
    measurements: Measurements,
    seed: int = 1234,
    rate: float = DEFAULT_PREAMBLE_RATE,
) -> EvidenceChoice:
    """Decide what evidence, if any, this record should carry.

    Returns ``EvidenceKind.NONE`` whenever agreement cannot be established. That
    is the whole safety property: no record is ever emitted whose preamble
    disagrees with its own gold answer.
    """
    artifacts = measurements.as_artifacts()
    if not artifacts or not _selected(sample_id, seed, rate):
        return EvidenceChoice(EvidenceKind.NONE, artifacts={})

    text = f"{prompt} {answer}"
    about_water = _mentions(text, _WATER_TERMS)
    about_builtup = _mentions(text, _BUILTUP_TERMS)
    polarity = _polarity(answer)

    if about_water or about_builtup:
        # The question is on-topic for something we measured, so the preamble is
        # only safe if the measurement agrees with the gold answer.
        checks = []
        if about_water:
            checks.append(
                any(
                    _agrees(artifacts.get(key), polarity, answer)
                    for key in ("optical_water_fraction", "sar_water_fraction")
                )
            )
        if about_builtup:
            checks.append(
                any(
                    _agrees(artifacts.get(key), polarity, answer)
                    for key in ("optical_builtup_fraction", "sar_builtup_fraction")
                )
            )
        if not all(checks):
            return EvidenceChoice(EvidenceKind.NONE, artifacts={})
        kind = EvidenceKind.DECISIVE
    else:
        # Nothing we measured bears on the question. Included anyway, and that is
        # the point: at inference the specialists run regardless of what was
        # asked, so the model has to meet irrelevant evidence in training too or
        # it will learn that a preamble always answers the question.
        kind = EvidenceKind.ORTHOGONAL

    preamble = format_evidence(artifacts)
    if not preamble:
        return EvidenceChoice(EvidenceKind.NONE, artifacts={})
    return EvidenceChoice(kind, preamble=preamble, artifacts=artifacts)


def apply_preamble(prompt: str, preamble: str) -> str:
    """Join a preamble to a prompt exactly as the controller does.

    One function, called by preparation and asserted against ``VLMTool.run`` in
    the tests. The separator is two newlines; getting that wrong is a difference
    the model sees on every single record.
    """
    return f"{preamble}\n\n{prompt}" if preamble else prompt
