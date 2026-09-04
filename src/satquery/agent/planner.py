"""Adaptive parameter selection and conditional re-planning.

Two things the controller was not doing, both of which the problem statement
asks for and both of which are visible in the scored trace.

**Adaptive parameters.** The controller previously called every tool with no
arguments, so ``params`` was empty in every trace step: the registry enforced
``allowed_params`` but nothing ever exercised it. Parameters are now derived
from properties of the actual input, and each carries the reason it was chosen.

Only adaptations with a real justification are made. A knob tuned on a hunch
would look adaptive and mean nothing, which is worse than a documented default.

**Conditional re-planning.** A first attempt that lands in a known-weak regime
is retried once with different parameters, and *both* attempts stay in the
trace. An agent that silently re-rolls until it likes the answer is not
auditable; one that shows the first attempt, the reason, and the second is.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any

from satquery.agent.context import RunContext
from satquery.schema import Task

#: Ground area below which a change mask is treated as speckle rather than a
#: finding. One hectare is about the smallest change worth reporting from
#: 10 m imagery, and it converts to a different pixel count at every resolution
#: -- which is exactly why this cannot be a fixed fraction.
MIN_CHANGE_AREA_M2 = 10_000.0

#: A closed question wants a word. Anything longer invites the model to ramble,
#: and a rambling answer scores worse under exact match.
#:
#: The auxiliary verbs are anchored to the start of the query on purpose. Matching
#: them anywhere reads "What is going on here" -- an open question -- as closed,
#: because of the "is" in the middle.
CLOSED_QUESTION = re.compile(
    r"^\s*(is|are|was|were|does|do|did|has|have|can|will)\b"
    r"|\bhow (many|much)\b"
    r"|\b(increased?|decreased?|unchanged)\b",
    re.IGNORECASE,
)

#: Below this changed fraction the change tools are in their weak regime.
QUIET_CHANGE = 0.02
#: How far to lower an Otsu threshold on a retry.
RETRY_THRESHOLD_SCALE = 0.55
#: Hard ceiling on revisions, so re-planning cannot spin.
MAX_REVISIONS = 2


@dataclass
class PlannedStep:
    """One tool invocation, its parameters, and why they were chosen."""

    tool: str
    params: dict[str, Any] = field(default_factory=dict)
    reason: str = ""
    revises: int | None = None


def _finest_gsd(ctx: RunContext) -> float | None:
    values = [info.gsd_m for info in ctx.infos if info.gsd_m]
    return min(values) if values else None


def _scene_pixels(ctx: RunContext) -> int:
    if not ctx.infos:
        return 0
    return max(info.width * info.height for info in ctx.infos)


def change_mask_params(ctx: RunContext) -> tuple[dict[str, Any], str]:
    """Speckle floor scaled to the resolution actually supplied.

    A fixed changed-area fraction means different things at different ground
    sample distances: 0.5% of a 10 m scene is 5 hectares, the same fraction of a
    0.6 m scene is under a fifth of that. Anchoring to a ground area keeps the
    floor meaning the same thing.
    """
    gsd = _finest_gsd(ctx)
    pixels = _scene_pixels(ctx)
    if not gsd or not pixels:
        return {}, "no ground sample distance available; using tool defaults"

    scene_area = pixels * gsd * gsd
    fraction = MIN_CHANGE_AREA_M2 / scene_area
    # Eight places, not five: on a wide 10 m scene the fraction is around 1e-4,
    # and rounding to five would distort it by several percent.
    clamped = round(min(max(fraction, 0.0), 0.2), 8)
    return (
        {"min_area_frac": clamped},
        (
            f"speckle floor {clamped:.6f} of the scene, which is "
            f"{MIN_CHANGE_AREA_M2 / 10_000:.0f} ha at {gsd:.2f} m/px"
        ),
    )


def sar_params(ctx: RunContext) -> tuple[dict[str, Any], str]:
    """Built-up percentile relaxed on coarse imagery.

    Bright double-bounce returns occupy a smaller tail as pixels get coarser,
    because a 10 m pixel averages scatterers that a 1 m pixel resolves. Holding
    the percentile fixed across resolutions would report less built-up simply
    because the imagery was coarser.
    """
    gsd = _finest_gsd(ctx)
    if not gsd:
        return {}, "no ground sample distance available; using tool defaults"
    percentile = 92.0 if gsd >= 8.0 else 95.0
    return (
        {"builtup_percentile": percentile},
        f"built-up tail at p{percentile:.0f} for {gsd:.1f} m/px imagery",
    )


def vlm_params(ctx: RunContext, task: Task) -> tuple[dict[str, Any], str]:
    """Generation budget matched to what the question actually asks for."""
    if task in {Task.CAPTION, Task.CHANGE_CAPTION}:
        return {"max_new_tokens": 128}, "description task; room for two sentences"
    if task is Task.GROUNDING:
        return {"max_new_tokens": 64}, "grounding; enough for a box, no prose"
    if ctx.query and CLOSED_QUESTION.search(ctx.query):
        return (
            {"max_new_tokens": 24},
            "closed question; a short answer scores better under exact match",
        )
    return {"max_new_tokens": 48}, "open question; a short phrase"


def plan_params(tool_name: str, ctx: RunContext, task: Task) -> tuple[dict, str]:
    """Parameters for a tool, derived from this run's inputs."""
    if tool_name == "change_mask":
        return change_mask_params(ctx)
    if tool_name == "sar_indices":
        return sar_params(ctx)
    if tool_name.startswith("vlm_"):
        return vlm_params(ctx, task)
    return {}, ""


def replan_change_mask(
    outputs: dict[str, Any], previous: dict[str, Any]
) -> tuple[dict[str, Any], str] | None:
    """Retry a change mask that found almost nothing.

    Otsu assumes a bimodal histogram. When two dates differ only subtly it
    splits noise instead of signal and reports a near-empty mask, so a second
    pass at a deliberately lower threshold is worth one attempt. If that also
    finds nothing, the scene really is unchanged and the trace shows both tries.
    """
    if "threshold" in previous:
        return None  # already an explicit threshold; do not keep lowering it
    changed = outputs.get("changed_area_frac")
    threshold = outputs.get("threshold")
    if not isinstance(changed, (int, float)) or not isinstance(threshold, (int, float)):
        return None
    if changed >= QUIET_CHANGE:
        return None

    lowered = round(max(threshold * RETRY_THRESHOLD_SCALE, 0.02), 4)
    if lowered >= threshold:
        return None
    return (
        {**previous, "threshold": lowered},
        (
            f"first pass found {changed:.2%} changed, below the {QUIET_CHANGE:.0%} "
            f"floor; retrying at {lowered} instead of the Otsu split {threshold}"
        ),
    )


def replan_vlm(
    answer: str, artifacts: dict[str, Any], previous: dict[str, Any]
) -> tuple[dict[str, Any], str] | None:
    """Retry an answer that contradicts what the specialists measured.

    Reinforcement goes through the shared artifact bag, which is the same
    channel the measurements already travel on, so the retry differs only in
    how firmly the evidence is framed.
    """
    from satquery.agent.confidence import agreement_adjustment

    if previous.get("_reinforced"):
        return None
    _, disagreement = agreement_adjustment(answer, artifacts)
    if not disagreement:
        return None
    return (
        {**previous, "_reinforced": True},
        f"answer conflicts with the measurement ({disagreement}); asking again",
    )
