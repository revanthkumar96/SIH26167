"""Routing, tool contracts and end-to-end controller behaviour.

Runs against the echo backend, so the whole agentic path is exercised with no GPU
and no weights.
"""

import numpy as np
import pytest
from PIL import Image

from satquery.agent.confidence import agreement_adjustment, blend, text_confidence
from satquery.agent.controller import Controller
from satquery.agent.registry import ParameterError, Tool, ToolRegistry, ToolResult
from satquery.agent.router import route
from satquery.agent.tools import default_registry
from satquery.agent.tools.vlm import format_evidence
from satquery.eval.backends import build_backend
from satquery.schema import InputConfig, Task, ToolSpec


def _png(path, size=(64, 64), colour=(30, 90, 40)):
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", size, colour).save(path)
    return path


def _half_bright(path, size=(64, 64)):
    """An image whose lower half is bright, to create a detectable change."""
    array = np.zeros((size[1], size[0], 3), dtype=np.uint8)
    array[: size[1] // 2] = 30
    array[size[1] // 2 :] = 220
    Image.fromarray(array, mode="RGB").save(path)
    return path


@pytest.fixture
def controller(tmp_path):
    return Controller(build_backend("echo"), workroot=tmp_path / "artifacts")


# -- routing ---------------------------------------------------------------


@pytest.mark.parametrize(
    ("query", "expected"),
    [
        ("Describe the land cover in this image.", Task.CAPTION),
        ("What do you see?", Task.CAPTION),
        ("Highlight the water body referred to in the query.", Task.GROUNDING),
        ("Where is the airport?", Task.GROUNDING),
        ("How many buildings are there?", Task.VQA),
        ("Is there a river?", Task.VQA),
        ("", Task.CAPTION),
    ],
)
def test_route_single_image(query, expected):
    assert route(query, InputConfig.SINGLE).task is expected


@pytest.mark.parametrize(
    ("query", "expected"),
    [
        ("What changed between these two dates?", Task.CHANGE_CAPTION),
        ("Describe the differences.", Task.CHANGE_CAPTION),
        ("Has the built-up area increased?", Task.CHANGE_VQA),
        ("Is there more water now?", Task.CHANGE_VQA),
    ],
)
def test_route_bitemporal(query, expected):
    assert route(query, InputConfig.BITEMPORAL_PAIR).task is expected


def test_crossmodal_routing_ignores_wording():
    """Only one task consumes an optical-SAR pair, so phrasing cannot change it."""
    for query in ["Describe this", "How many?", "Highlight water", ""]:
        decision = route(query, InputConfig.CROSSMODAL_PAIR)
        assert decision.task is Task.CROSSMODAL_VQA


def test_route_records_its_reason():
    decision = route("Highlight the lake", InputConfig.SINGLE)
    assert "highlight" in decision.rule
    assert 0.0 < decision.confidence <= 1.0


# -- registry and parameter enforcement ------------------------------------


class _Stub(Tool):
    spec = ToolSpec(
        name="stub",
        version="0.1.0",
        accepts=InputConfig.SINGLE,
        tasks=(Task.VQA,),
        allowed_params={"threshold": (0.0, 1.0)},
    )

    def run(self, ctx, threshold=0.5):
        return ToolResult(outputs={"threshold": threshold})


def test_registry_rejects_duplicate_names():
    registry = ToolRegistry([_Stub()])
    with pytest.raises(KeyError, match="already registered"):
        registry.register(_Stub())


def test_registry_unknown_tool_lists_known_ones():
    with pytest.raises(KeyError, match="Registered:"):
        ToolRegistry([_Stub()]).get("missing")


def test_registry_filters_by_task_and_config():
    registry = ToolRegistry([_Stub()])
    assert registry.for_task(Task.VQA, InputConfig.SINGLE)
    assert registry.for_task(Task.VQA, InputConfig.BITEMPORAL_PAIR) == []
    assert registry.for_task(Task.CAPTION, InputConfig.SINGLE) == []


def test_disallowed_parameter_is_refused_not_trusted():
    """Only permitted parameters may be configured, so enforcement lives here."""
    tool = _Stub()
    with pytest.raises(ParameterError, match="not permitted"):
        tool.invoke(None, learning_rate=0.1)
    with pytest.raises(ParameterError, match="outside"):
        tool.invoke(None, threshold=5.0)


def test_default_registry_covers_every_task():
    registry = default_registry()
    for task in Task:
        config = {
            Task.VQA: InputConfig.SINGLE,
            Task.CAPTION: InputConfig.SINGLE,
            Task.GROUNDING: InputConfig.SINGLE,
            Task.CHANGE_VQA: InputConfig.BITEMPORAL_PAIR,
            Task.CHANGE_CAPTION: InputConfig.BITEMPORAL_PAIR,
            Task.CROSSMODAL_VQA: InputConfig.CROSSMODAL_PAIR,
        }[task]
        assert registry.for_task(task, config), f"no tool serves {task}"


# -- confidence ------------------------------------------------------------


def test_confidence_never_reaches_one():
    assert text_confidence("yes") < 1.0


def test_hedged_answer_scores_lower():
    assert text_confidence("It possibly might be water") < text_confidence("water")


def test_empty_answer_is_floor():
    assert text_confidence("") == pytest.approx(0.05)


def test_agreement_penalises_contradicting_the_mask():
    adjustment, warning = agreement_adjustment("no change", {"changed_area_frac": 0.3})
    assert adjustment < 0
    assert warning and "30" in warning.replace(".0%", "%")


def test_agreement_rewards_consistency():
    adjustment, warning = agreement_adjustment(
        "no change", {"changed_area_frac": 0.001}
    )
    assert adjustment > 0
    assert warning is None


def test_optical_sar_disagreement_is_surfaced():
    adjustment, warning = agreement_adjustment(
        "some water", {"sar_water_fraction": 0.4, "optical_water_fraction": 0.05}
    )
    assert adjustment < 0
    assert warning and "cloud" in warning


def test_blend_ignores_missing_values():
    assert blend([None, None]) is None
    assert blend([0.8, None, 0.6]) == pytest.approx(0.7)


# -- evidence injection ----------------------------------------------------


def test_evidence_block_is_empty_without_measurements():
    assert format_evidence({}) == ""


def test_evidence_block_names_the_measurements():
    block = format_evidence(
        {"changed_area_frac": 0.12, "change_location": "mainly north-east"}
    )
    assert "changed area" in block
    assert "0.12" in block
    assert "north-east" in block


# -- controller end to end -------------------------------------------------


def test_single_image_run(controller, tmp_path):
    trace = controller.run("Is there water?", [_png(tmp_path / "a.png")])

    assert trace.routed_task is Task.VQA
    assert trace.input_check.config is InputConfig.SINGLE
    assert trace.answer == "yes"
    assert [s.tool for s in trace.steps] == ["vlm_vqa"]
    assert trace.routing_rule
    assert trace.confidence is not None


def test_bitemporal_run_puts_specialist_before_the_model(controller, tmp_path):
    """The change mask must run first so its measurement reaches the prompt."""
    trace = controller.run(
        "Has the built-up area increased?",
        [_png(tmp_path / "t1.png"), _half_bright(tmp_path / "t2.png")],
    )

    assert trace.routed_task is Task.CHANGE_VQA
    assert [s.tool for s in trace.steps] == ["change_mask", "vlm_change_vqa"]
    assert trace.steps[0].outputs["changed_area_frac"] > 0
    assert trace.steps[1].outputs["grounded_in_evidence"] is True
    assert any(e["type"] == "mask" for e in trace.evidence)


def test_crossmodal_run_uses_both_specialists(controller, tmp_path):
    trace = controller.run(
        "Identify built-up and water regions.",
        [_png(tmp_path / "optical.png"), _png(tmp_path / "scene_VV.png")],
    )

    assert trace.input_check.config is InputConfig.CROSSMODAL_PAIR
    assert trace.routed_task is Task.CROSSMODAL_VQA
    assert [s.tool for s in trace.steps] == [
        "optical_indices",
        "sar_indices",
        "vlm_crossmodal_vqa",
    ]


def test_optical_indices_declines_on_rgb_rather_than_inventing(controller, tmp_path):
    """A 3-band PNG has no NIR, so NDWI is reported unavailable, not faked."""
    trace = controller.run(
        "Where is the water?",
        [_png(tmp_path / "optical.png"), _png(tmp_path / "scene_VV.png")],
    )
    optical_step = next(s for s in trace.steps if s.tool == "optical_indices")
    assert optical_step.outputs["applicable"] is False
    assert "near-infrared" in optical_step.outputs["reason"]


def test_grounding_run_parses_a_box(controller, tmp_path):
    trace = controller.run("Highlight the water body", [_png(tmp_path / "a.png")])

    assert trace.routed_task is Task.GROUNDING
    step = trace.steps[-1]
    assert step.outputs["parse_ok"] is True
    assert step.outputs["bbox"] == pytest.approx([0.25, 0.25, 0.75, 0.75])
    assert trace.evidence[0]["type"] == "bbox"


def test_trace_is_json_serialisable(controller, tmp_path):
    import json

    trace = controller.run("Describe this", [_png(tmp_path / "a.png")])
    payload = json.loads(json.dumps(trace.to_dict()))

    assert payload["routed_task"] == "caption"
    assert payload["input_check"]["config"] == "single"
    assert isinstance(payload["steps"], list)


def test_every_step_names_a_tool_and_version(controller, tmp_path):
    trace = controller.run(
        "What changed?", [_png(tmp_path / "a.png"), _half_bright(tmp_path / "b.png")]
    )
    for step in trace.steps:
        assert step.tool and step.version
        assert step.duration_ms >= 0


def test_three_images_rejected(controller, tmp_path):
    paths = [_png(tmp_path / f"{i}.png") for i in range(3)]
    with pytest.raises(ValueError, match="bi-temporal pair"):
        controller.run("What is this?", paths)


def test_mismatched_pair_warns_but_still_answers(controller, tmp_path):
    trace = controller.run(
        "What changed?",
        [
            _png(tmp_path / "a.png", size=(64, 64)),
            _png(tmp_path / "b.png", size=(128, 128)),
        ],
    )
    assert trace.answer
    assert any("pixel dimensions differ" in w for w in trace.input_check.warnings)
