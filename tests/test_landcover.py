"""The land-cover CNN: labels, stem inflation, metrics, and integration.

The integration tests at the bottom are the ones worth having. CNN.md warns that
the third wiring step -- adding the tool's outputs to ``_ARTIFACT_MAP`` and
``_EVIDENCE_LABELS`` -- is easy to forget, and that forgetting it produces a tool
that runs, reports healthy numbers in the trace, and never reaches the model.
Nothing raises. The only symptom is that the evidence does not help.
"""

from __future__ import annotations

import numpy as np
import pytest

from satquery.agent.controller import _ARTIFACT_MAP, _PRECURSORS
from satquery.agent.tools import default_registry
from satquery.agent.tools.vlm import _EVIDENCE_LABELS
from satquery.cnn.labels import (
    CORINE_CLASSES,
    LabelMismatchError,
    class_index,
    encode,
    observed_classes,
    render_evidence_line,
    verify_classes,
)
from satquery.cnn.metrics import average_precision, classification_metrics
from satquery.schema import Task

torch = pytest.importorskip("torch")


# -- labels --------------------------------------------------------------


def test_class_list_is_the_19_class_nomenclature():
    assert len(CORINE_CLASSES) == 19
    assert len(set(CORINE_CLASSES)) == 19
    assert "Inland waters" in CORINE_CLASSES
    assert "Urban fabric" in CORINE_CLASSES


def test_encode_produces_a_multi_hot_row():
    vector = encode(["Arable land", "Mixed forest"])
    index = class_index()
    assert sum(vector) == 2
    assert vector[index["Arable land"]] == 1.0
    assert vector[index["Mixed forest"]] == 1.0


def test_encode_ignores_an_unknown_label():
    """One odd label must not abort a 549k-patch pass; verify_classes is the gate."""
    assert sum(encode(["Arable land", "Atlantis"])) == 1


def test_verify_classes_accepts_the_shipped_set():
    verify_classes(list(CORINE_CLASSES))


def test_verify_classes_reports_both_directions():
    """A silent mismatch offsets every logit against the wrong class name."""
    with pytest.raises(LabelMismatchError) as excinfo:
        verify_classes(["Arable land", "Atlantis"])
    message = str(excinfo.value)
    assert "Atlantis" in message
    assert "Urban fabric" in message


def test_observed_classes_is_sorted_and_deduplicated():
    assert observed_classes([["b", "a"], ["a", "c"]]) == ["a", "b", "c"]


def test_evidence_line_matches_what_preparation_wrote():
    """One renderer, or the adapted model meets a format it never trained on."""
    from satquery.data.evidence import describe_classes

    labels = ["Arable land", "Pastures", "Mixed forest"]
    assert render_evidence_line(labels) == describe_classes(labels)
    assert render_evidence_line(labels) == "arable land, pastures, mixed forest"


# -- stem inflation ------------------------------------------------------


def test_inflation_preserves_the_response_to_a_uniform_input():
    """The reason for the rescale: activation statistics must not shift.

    Without ``3 / channels`` a 12-band stem returns four times the activation
    the pretrained batch-norm downstream was fitted to, and the first epochs go
    on unlearning that instead of learning bands.
    """
    from satquery.cnn.model import inflate_stem

    weight = torch.randn(64, 3, 7, 7)
    widened = inflate_stem(weight, 12)

    assert widened.shape == (64, 12, 7, 7)
    original = (weight * torch.ones(1, 3, 7, 7)).sum(dim=(1, 2, 3))
    inflated = (widened * torch.ones(1, 12, 7, 7)).sum(dim=(1, 2, 3))
    assert torch.allclose(original, inflated, atol=1e-4)


def test_inflation_handles_a_width_that_is_not_a_multiple_of_three():
    from satquery.cnn.model import inflate_stem

    assert inflate_stem(torch.randn(64, 3, 7, 7), 14).shape == (64, 14, 7, 7)


def test_inflation_to_the_same_width_is_a_copy():
    from satquery.cnn.model import inflate_stem

    weight = torch.randn(64, 3, 7, 7)
    assert torch.allclose(inflate_stem(weight, 3), weight)


def test_inflation_rejects_a_non_conv_weight():
    from satquery.cnn.model import inflate_stem

    with pytest.raises(ValueError, match="4-D"):
        inflate_stem(torch.randn(64, 3), 12)


def test_model_accepts_twelve_bands_and_emits_one_logit_per_class():
    """Multi-label: sigmoid per class, never a softmax across them."""
    from satquery.cnn.model import build_model

    model = build_model(channels=12, num_classes=19, pretrained=False).eval()
    with torch.no_grad():
        logits = model(torch.randn(2, 12, 120, 120))
    assert logits.shape == (2, 19)


def test_model_accepts_fourteen_bands_for_optical_plus_sar():
    from satquery.cnn.model import build_model

    model = build_model(channels=14, num_classes=19, pretrained=False).eval()
    with torch.no_grad():
        assert model(torch.randn(1, 14, 120, 120)).shape == (1, 19)


# -- metrics -------------------------------------------------------------


def test_average_precision_is_one_for_a_perfect_ranking():
    scores = np.array([0.9, 0.8, 0.2, 0.1])
    targets = np.array([1, 1, 0, 0])
    assert average_precision(scores, targets) == pytest.approx(1.0)


def test_average_precision_is_undefined_without_positives():
    """nan, not zero: an absent class has no score and must not be averaged as 0."""
    result = average_precision(np.array([0.5, 0.4]), np.array([0, 0]))
    assert np.isnan(result)


def test_macro_f1_exposes_rare_class_failure_that_micro_hides():
    """The reason both are reported.

    One common class predicted perfectly and one rare class missed entirely.
    Micro F1 stays high because the common class dominates the pooled counts;
    macro halves, because it gives the rare class an equal vote.
    """
    # Negative everywhere by default: sigmoid(0) is exactly 0.5 and would
    # clear the threshold, giving the common class false positives that muddy
    # what this test is about.
    logits = np.full((100, 2), -8.0)
    targets = np.zeros((100, 2))
    logits[:90, 0] = 8.0  # common class, predicted exactly where it is true
    targets[:90, 0] = 1
    targets[:5, 1] = 1  # rare class, never predicted

    metrics = classification_metrics(logits, targets, ["common", "rare"])
    assert metrics["f1_micro"] > 0.9
    assert metrics["f1_macro"] == pytest.approx(0.5, abs=0.01)


def test_metrics_report_per_class_ap_and_support():
    logits = np.array([[2.0, -2.0], [1.0, -1.0]])
    targets = np.array([[1, 0], [1, 0]])
    metrics = classification_metrics(logits, targets, ["Inland waters", "Pastures"])
    assert metrics["ap/inland_waters"] == pytest.approx(1.0)
    assert metrics["support/inland_waters"] == 2.0
    # No positives, so no AP -- and it is not counted in the mean.
    assert "ap/pastures" not in metrics
    assert metrics["classes_scored"] == 1.0


def test_metrics_refuse_a_class_count_mismatch():
    """A silent mismatch scores every class against the wrong name."""
    with pytest.raises(ValueError, match="class names"):
        classification_metrics(np.zeros((4, 3)), np.zeros((4, 3)), ["only", "two"])


# -- the three integration steps -----------------------------------------


def test_step_one_the_tool_is_registered():
    registry = default_registry()
    assert registry.has("landcover_cnn")
    assert registry.get("landcover_cnn").spec.kind == "measurement"


def test_step_two_single_image_tasks_finally_have_a_precursor():
    """Previously VQA and captioning went straight to the model with no evidence."""
    assert "landcover_cnn" in _PRECURSORS[Task.VQA]
    assert "landcover_cnn" in _PRECURSORS[Task.CAPTION]


def test_step_two_paired_tasks_keep_their_existing_precursors():
    assert "change_mask" in _PRECURSORS[Task.CHANGE_VQA]
    assert "optical_indices" in _PRECURSORS[Task.CROSSMODAL_VQA]
    assert "sar_indices" in _PRECURSORS[Task.CROSSMODAL_VQA]


def test_step_three_outputs_reach_the_model():
    """The step CNN.md warns is easy to forget and silently makes the tool useless.

    Two hops have to line up: _ARTIFACT_MAP promotes the tool's output into the
    artifact bag, and _EVIDENCE_LABELS gives that key a phrasing. Miss either and
    the classifier runs, its numbers appear in the trace, and the prompt never
    sees them.
    """
    from satquery.agent.tools.vlm import format_evidence

    for name in (
        "landcover_cnn",
        "landcover_cnn_crossmodal",
        "landcover_cnn_bitemporal",
    ):
        promoted = _ARTIFACT_MAP[name]
        assert promoted == {"evidence_line": "landcover_classes"}
        for target in promoted.values():
            assert target in _EVIDENCE_LABELS

    rendered = format_evidence({"landcover_classes": "arable land, pastures"})
    assert "land cover classes present: arable land, pastures" in rendered


def test_every_precursor_is_actually_registered():
    """A precursor named but not registered is silently skipped when planning."""
    registry = default_registry()
    for task, names in _PRECURSORS.items():
        for name in names:
            assert registry.has(name), f"{task} names unregistered tool '{name}'"


def test_every_promoted_artifact_has_a_phrasing():
    """Generally, not just for the CNN: a promoted key with no label renders nothing."""
    for tool, mapping in _ARTIFACT_MAP.items():
        for target in mapping.values():
            assert target in _EVIDENCE_LABELS, f"{tool} promotes unlabelled '{target}'"


# -- degrading without a checkpoint --------------------------------------


def test_tool_reports_inapplicable_without_a_checkpoint(tmp_path, monkeypatch):
    """The system has to work before the classifier is trained."""
    from PIL import Image

    from satquery.agent.controller import Controller
    from satquery.eval.backends import build_backend

    monkeypatch.delenv("SATQUERY_LANDCOVER_CHECKPOINT", raising=False)
    path = tmp_path / "scene.png"
    Image.new("RGB", (32, 32), "green").save(path)

    with build_backend("echo") as backend:
        trace = Controller(backend, workroot=tmp_path / "runs").run(
            "What is here?", [path]
        )

    step = next(s for s in trace.steps if s.tool == "landcover_cnn")
    assert step.outputs["applicable"] is False
    assert "SATQUERY_LANDCOVER_CHECKPOINT" in step.outputs["reason"]
    # The run still answers: a missing classifier degrades evidence, not service.
    assert trace.answer


def test_a_trained_checkpoint_round_trips_into_the_tool(tmp_path, monkeypatch):
    """End to end: train-shaped payload in, evidence line out."""
    from PIL import Image

    from satquery.agent.context import RunContext
    from satquery.agent.tools.landcover import LandCoverTool
    from satquery.cnn.model import build_model
    from satquery.schema import ImageRef, InputConfig

    classes = list(CORINE_CLASSES)
    model = build_model(channels=12, num_classes=len(classes), pretrained=False)
    checkpoint = tmp_path / "landcover.pt"
    torch.save(
        {"state_dict": model.state_dict(), "classes": classes, "channels": 12},
        checkpoint,
    )
    monkeypatch.setenv("SATQUERY_LANDCOVER_CHECKPOINT", str(checkpoint))

    path = tmp_path / "scene.png"
    Image.new("RGB", (32, 32), "green").save(path)
    ctx = RunContext(
        run_id="r",
        query="What is here?",
        images=(ImageRef(path=path, modality="optical", role="single"),),
        infos=(),
        input_config=InputConfig.SINGLE,
        workdir=tmp_path / "work",
        backend=None,
    )

    result = LandCoverTool(InputConfig.SINGLE).run(ctx, min_confidence=0.0)
    assert result.outputs["applicable"] is True
    assert len(result.outputs["classes"]) == len(classes)
    assert result.outputs["evidence_line"]
    # Stated in the trace, because a European taxonomy over Indian terrain is a
    # caveat the reader needs rather than a detail.
    assert "Europe-trained" in result.outputs["taxonomy"]


def test_channel_fitting_repeats_panchromatic_rather_than_zero_padding():
    """A zero band is a strong signal the network was never trained to see."""
    from satquery.agent.tools.landcover import _fit_channels

    pan = np.arange(4, dtype=np.float32).reshape(1, 2, 2)
    widened = _fit_channels(pan, 12)
    assert widened.shape == (12, 2, 2)
    assert widened.min() > 0 or np.array_equal(widened[0], pan[0])
    assert np.array_equal(widened[3], pan[0])


def test_channel_fitting_truncates_a_wider_stack():
    from satquery.agent.tools.landcover import _fit_channels

    assert _fit_channels(np.zeros((14, 2, 2), dtype=np.float32), 12).shape == (12, 2, 2)
