"""Vision-language tools.

One class covers every VLM task; instances differ only by task, name and accepted
input configuration. That is what lets the fine-tuned LoRA adapters slot in later
without touching the controller -- an instance simply names an ``adapter`` and the
backend routes to it.

The important behaviour here is evidence injection: whatever the deterministic
specialists measured earlier in the run is prepended to the prompt, so the model
answers from a measurement rather than from the picture alone.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from satquery.agent.confidence import text_confidence
from satquery.agent.context import RunContext
from satquery.agent.registry import Tool, ToolResult
from satquery.eval.normalize import parse_bbox
from satquery.eval.prompts import build_prompt
from satquery.eval.prompts import max_new_tokens as default_max_tokens
from satquery.schema import (
    GenerationRequest,
    InputConfig,
    Sample,
    Task,
    ToolSpec,
)

#: Artifact keys worth showing the model, and how to phrase them.
_EVIDENCE_LABELS: Mapping[str, str] = {
    "changed_area_frac": "changed area (fraction of scene)",
    "change_location": "change location",
    "direction": "change direction",
    "sar_water_fraction": "water fraction from SAR backscatter",
    "sar_builtup_fraction": "built-up fraction from SAR backscatter",
    "sar_water_location": "water location from SAR",
    "optical_water_fraction": "water fraction from optical NDWI",
    "optical_builtup_fraction": "built-up fraction from optical NDBI",
}


def format_evidence(artifacts: Mapping[str, Any]) -> str:
    """Render specialist measurements as a prompt preamble."""
    lines = [
        f"- {label}: {artifacts[key]}"
        for key, label in _EVIDENCE_LABELS.items()
        if key in artifacts
    ]
    if not lines:
        return ""

    header = (
        "Measurements from image-analysis tools that have already run on these "
        "images. Treat them as reliable and do not contradict them:"
    )
    if artifacts.get("contradiction_retry"):
        # Second pass. The first answer disagreed with the measurement, so the
        # framing is firmer -- the question itself is unchanged.
        header = (
            "Your previous answer contradicted these measurements, which were "
            "computed directly from the pixels and are not in doubt. Answer "
            "again, consistently with them:"
        )
    return header + "\n" + "\n".join(lines)


class VLMTool(Tool):
    """Runs the vision-language backend for one task."""

    def __init__(
        self,
        name: str,
        task: Task,
        accepts: InputConfig,
        adapter: str | None = None,
        version: str = "1.0.0",
        summary: str = "",
    ) -> None:
        self.task = task
        self.adapter = adapter
        self.spec = ToolSpec(
            name=name,
            version=version,
            accepts=accepts,
            tasks=(task,),
            allowed_params={
                "max_new_tokens": (1, 512),
                "temperature": (0.0, 1.0),
            },
            outputs=("answer", "bbox") if task is Task.GROUNDING else ("answer",),
            summary=summary or f"Vision-language model for the {task} task.",
            kind="model",
            category="language",
            cost="heavy",
            requires=(
                "One image"
                if accepts is InputConfig.SINGLE
                else "Two co-registered images"
            ),
            emits_evidence=task is Task.GROUNDING,
            param_docs={
                "max_new_tokens": "Generation budget; vision tokens dominate cost.",
                "temperature": "Held at 0 so benchmark runs are reproducible.",
            },
        )

    def _question(self, ctx: RunContext) -> str | None:
        if self.task in {Task.CAPTION, Task.CHANGE_CAPTION}:
            return None
        return ctx.query

    def run(
        self,
        ctx: RunContext,
        max_new_tokens: int | None = None,
        temperature: float = 0.0,
    ) -> ToolResult:
        sample = Sample(
            sample_id=ctx.run_id,
            task=self.task,
            images=ctx.images,
            question=self._question(ctx),
        )
        prompt = build_prompt(sample)
        preamble = format_evidence(ctx.artifacts)
        if preamble:
            prompt = f"{preamble}\n\n{prompt}"

        budget = max_new_tokens or default_max_tokens(self.task)
        request = GenerationRequest(
            sample_id=ctx.run_id,
            prompt=prompt,
            images=ctx.paths,
            max_new_tokens=budget,
        )
        raw, meta = ctx.backend.generate_with_meta([request])[0]

        outputs: dict[str, Any] = {
            "answer": raw,
            "grounded_in_evidence": bool(preamble),
        }
        # A reply cut off at the budget is a defect, and the trace should say so
        # rather than present a half sentence as the answer.
        if meta.get("finish_reason"):
            outputs["truncated"] = meta["finish_reason"] == "length"
            outputs["tokens"] = meta.get("tokens", 0)
        if self.adapter:
            outputs["adapter"] = self.adapter

        evidence: tuple[Mapping[str, Any], ...] = ()
        if self.task is Task.GROUNDING:
            box = parse_bbox(raw, scale="milli")
            outputs["bbox"] = list(box) if box else None
            outputs["parse_ok"] = box is not None
            if box:
                evidence = (
                    {
                        "type": "bbox",
                        "bbox": list(box),
                        "image": ctx.images[0].path.name,
                        "label": ctx.query,
                    },
                )

        return ToolResult(
            outputs=outputs,
            confidence=text_confidence(raw),
            evidence=evidence,
        )


def build_vlm_tools() -> list[VLMTool]:
    """The default VLM tool set: one entry per task, all one set of weights."""
    return [
        VLMTool(
            "vlm_vqa",
            Task.VQA,
            InputConfig.SINGLE,
            summary="Answers a question about a single scene.",
        ),
        VLMTool(
            "vlm_caption",
            Task.CAPTION,
            InputConfig.SINGLE,
            summary="Describes land cover, objects and their arrangement.",
        ),
        VLMTool(
            "vlm_grounding",
            Task.GROUNDING,
            InputConfig.SINGLE,
            summary="Locates the region a phrase refers to and returns a box.",
        ),
        VLMTool(
            "vlm_change_vqa",
            Task.CHANGE_VQA,
            InputConfig.BITEMPORAL_PAIR,
            summary="Answers a question about what differs between two dates.",
        ),
        VLMTool(
            "vlm_change_caption",
            Task.CHANGE_CAPTION,
            InputConfig.BITEMPORAL_PAIR,
            summary="Describes what changed between two dates, and where.",
        ),
        VLMTool(
            "vlm_crossmodal_vqa",
            Task.CROSSMODAL_VQA,
            InputConfig.CROSSMODAL_PAIR,
            summary="Answers using an optical and a SAR view together.",
        ),
    ]
