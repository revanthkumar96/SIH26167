"""The agentic controller.

Six stages, matching the six controller duties in the problem statement: check,
route, select, execute, fuse, report. Implemented as an explicit pipeline rather
than a graph framework -- the flow is fixed, and an explicit pipeline makes the
emitted trace a direct reading of the code rather than a rendering of a framework's
internal state.

Ordering matters: deterministic specialists run *before* the VLM so their
measurements can be injected into its prompt. That is the difference between an
evidence-grounded answer and a guess.
"""

from __future__ import annotations

import time
import uuid
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any

from satquery.agent.confidence import agreement_adjustment, blend
from satquery.agent.context import RunContext
from satquery.agent.planner import (
    MAX_REVISIONS,
    PlannedStep,
    plan_params,
    replan_change_mask,
    replan_truncated,
    replan_vlm,
)
from satquery.agent.registry import ToolRegistry
from satquery.agent.router import RouteDecision, route
from satquery.agent.tools import default_registry
from satquery.eval.backends.base import VLMBackend
from satquery.geo.checks import assign_roles, check_pair, infer_input_config
from satquery.geo.raster import RasterInfo, read_info
from satquery.schema import (
    ExecutionTrace,
    ImageRef,
    InputCheck,
    InputConfig,
    Task,
    TraceStep,
)

StepCallback = Callable[[TraceStep], None]

#: Which specialists to run ahead of the VLM, per task.
_PRECURSORS: dict[Task, tuple[str, ...]] = {
    Task.CHANGE_VQA: ("change_mask",),
    Task.CHANGE_CAPTION: ("change_mask",),
    Task.CROSSMODAL_VQA: ("optical_indices", "sar_indices"),
}

#: Specialist outputs promoted into the shared artifact bag under stable names,
#: so the prompt builder does not need to know which tool produced what.
_ARTIFACT_MAP: dict[str, dict[str, str]] = {
    "change_mask": {
        "changed_area_frac": "changed_area_frac",
        "change_location": "change_location",
        "direction": "direction",
    },
    "sar_indices": {
        "water_fraction": "sar_water_fraction",
        "builtup_fraction": "sar_builtup_fraction",
        "water_location": "sar_water_location",
    },
    "optical_indices": {
        "water_fraction": "optical_water_fraction",
        "builtup_fraction": "optical_builtup_fraction",
    },
}


class Controller:
    """Routes a query over one or two images and returns an auditable trace."""

    def __init__(
        self,
        backend: VLMBackend,
        registry: ToolRegistry | None = None,
        workroot: Path | None = None,
    ) -> None:
        self.backend = backend
        self.registry = registry or default_registry()
        self.workroot = Path(workroot or "runs/artifacts")

    # -- stage 1: check ----------------------------------------------------

    def check_inputs(
        self, paths: Sequence[Path]
    ) -> tuple[InputCheck, tuple[ImageRef, ...], tuple[RasterInfo, ...]]:
        """Inspect the uploaded set and derive its configuration."""
        infos = tuple(read_info(p) for p in paths)
        config, modalities = infer_input_config(infos)
        roles = assign_roles(config, modalities)

        images = tuple(
            ImageRef(path=info.path, modality=modality, role=role)
            for info, modality, role in zip(infos, modalities, roles, strict=True)
        )

        passed: list[str] = ["format_supported"]
        warnings: list[str] = []
        coregistered = config is InputConfig.SINGLE

        if len(infos) == 2:
            coregistered, pair_passed, pair_warnings = check_pair(infos[0], infos[1])
            passed.extend(pair_passed)
            warnings.extend(pair_warnings)
            if not coregistered:
                warnings.append(
                    "images may not be co-registered; results should be treated "
                    "as indicative"
                )

        image_payloads = [
            {**info.as_dict(), "role": str(image.role), "modality": str(image.modality)}
            for info, image in zip(infos, images, strict=True)
        ]

        check = InputCheck(
            config=config,
            images=image_payloads,
            coregistered=coregistered,
            checks_passed=tuple(passed),
            warnings=tuple(warnings),
        )
        return check, images, infos

    # -- stages 2 and 3: route and select ----------------------------------

    def plan(
        self, query: str, config: InputConfig, ctx: RunContext | None = None
    ) -> tuple[RouteDecision, list[PlannedStep]]:
        """Decide the task, then name the tools and their parameters, in order.

        Parameters are derived from the run's own inputs rather than left at
        tool defaults, and each records why it was chosen -- the trace shows the
        configuration the agent selected, not just the tools it picked.
        """
        decision = route(query, config)

        names: list[str] = [
            name
            for name in _PRECURSORS.get(decision.task, ())
            if self.registry.has(name)
        ]

        candidates = self.registry.for_task(decision.task, config)
        vlm_tools = [t.name for t in candidates if t.name.startswith("vlm_")]
        if not vlm_tools:
            raise ValueError(
                f"no tool registered for task '{decision.task}' with input "
                f"configuration '{config}'"
            )
        names.append(vlm_tools[0])

        steps: list[PlannedStep] = []
        for name in names:
            params, reason = (
                plan_params(name, ctx, decision.task) if ctx is not None else ({}, "")
            )
            steps.append(PlannedStep(tool=name, params=params, reason=reason))
        return decision, steps

    def _revision(
        self,
        planned: PlannedStep,
        outputs: dict[str, Any],
        answer: str,
        ctx: RunContext,
    ) -> PlannedStep | None:
        """A second attempt at a step, when the first landed in a weak regime."""
        if planned.tool == "change_mask":
            revised = replan_change_mask(outputs, planned.params)
        elif planned.tool.startswith("vlm_"):
            # A cut-off reply is fixed before its content is judged: re-asking a
            # half sentence for contradicting a measurement would be unfair.
            revised = replan_truncated(outputs, planned.params) or replan_vlm(
                answer, ctx.artifacts, planned.params
            )
        else:
            revised = None

        if revised is None:
            return None
        params, reason = revised
        return PlannedStep(tool=planned.tool, params=params, reason=reason)

    # -- stage 4 to 6: execute, fuse, report -------------------------------

    def run(
        self,
        query: str,
        paths: Sequence[str | Path],
        on_step: StepCallback | None = None,
        run_id: str | None = None,
    ) -> ExecutionTrace:
        """Execute end to end and return the trace. Never partially applied."""
        started = time.perf_counter()
        identifier = run_id or uuid.uuid4().hex[:12]
        resolved = [Path(p) for p in paths]

        check, images, infos = self.check_inputs(resolved)

        ctx = RunContext(
            run_id=identifier,
            query=query,
            images=images,
            infos=infos,
            input_config=check.config,
            workdir=self.workroot / identifier,
            backend=self.backend,
        )
        decision, planned = self.plan(query, check.config, ctx)

        steps: list[TraceStep] = []
        evidence: list[dict[str, Any]] = []
        confidences: list[float | None] = []
        answer = ""
        warnings = list(check.warnings)

        # The queue is mutable: a step may append one revision of itself, which
        # is what makes the plan conditional rather than fixed.
        queue: list[PlannedStep] = list(planned)
        revisions = 0
        index = 0

        while queue:
            planned_step = queue.pop(0)
            index += 1
            tool = self.registry.get(planned_step.tool)

            # Internal flags steer prompt framing; they are not tool parameters
            # and must not reach the registry's allowed_params check.
            call_params = {
                k: v for k, v in planned_step.params.items() if not k.startswith("_")
            }
            if planned_step.params.get("_reinforced"):
                ctx.artifacts["contradiction_retry"] = True

            step_started = time.perf_counter()
            result = tool.invoke(ctx, **call_params)
            duration_ms = int((time.perf_counter() - step_started) * 1000)

            for source, target in _ARTIFACT_MAP.get(planned_step.tool, {}).items():
                if source in result.outputs:
                    ctx.artifacts[target] = result.outputs[source]

            if "answer" in result.outputs:
                answer = str(result.outputs["answer"])

            evidence.extend(dict(item) for item in result.evidence)
            confidences.append(result.confidence)

            step = TraceStep(
                step=index,
                tool=tool.spec.name,
                version=tool.spec.version,
                adapter=getattr(tool, "adapter", None),
                params=dict(call_params),
                outputs=dict(result.outputs),
                confidence=result.confidence,
                duration_ms=duration_ms,
                reason=planned_step.reason,
                revises=planned_step.revises,
            )
            steps.append(step)
            if on_step is not None:
                on_step(step)

            if revisions >= MAX_REVISIONS:
                continue

            revision = self._revision(planned_step, result.outputs, answer, ctx)
            if revision is not None:
                revision.revises = index
                queue.insert(0, revision)
                revisions += 1

        # Only the last attempt of a revised step is scored, but every attempt
        # stays in the trace.
        confidences = [c for c in confidences if c is not None]
        adjustment, disagreement = agreement_adjustment(answer, ctx.artifacts)
        if disagreement:
            warnings.append(disagreement)

        confidence = blend(confidences)
        if confidence is not None:
            confidence = round(min(max(confidence + adjustment, 0.05), 0.92), 3)

        return ExecutionTrace(
            run_id=identifier,
            query=query,
            input_check=InputCheck(
                config=check.config,
                images=check.images,
                coregistered=check.coregistered,
                checks_passed=check.checks_passed,
                warnings=tuple(warnings),
            ),
            routed_task=decision.task,
            steps=tuple(steps),
            answer=answer,
            evidence=tuple(evidence),
            confidence=confidence,
            duration_ms=int((time.perf_counter() - started) * 1000),
            routing_rule=decision.rule,
            routing_confidence=decision.confidence,
        )
