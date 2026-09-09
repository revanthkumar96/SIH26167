"""Evidence-preamble synthesis, and its parity with the serving path.

The parity assertions here are the point of the file. If a training preamble and
the one the controller builds at inference differ by so much as a separator, the
adapted model meets an unfamiliar prompt shape on every single query -- and
nothing raises, nothing logs, the scores are just quietly worse. That is the same
class of bug as image-normalisation skew, in the prompt channel.
"""

from __future__ import annotations

import numpy as np
import pytest

from satquery.agent.tools.vlm import _EVIDENCE_LABELS, format_evidence
from satquery.data.evidence import (
    ABSENT_BELOW,
    PRESENT_ABOVE,
    EvidenceKind,
    Measurements,
    apply_preamble,
    choose_evidence,
    describe_classes,
    measure_optical,
    measure_sar,
)

WET = Measurements(optical_water_fraction=0.3151, sar_water_fraction=0.2903)
DRY = Measurements(optical_water_fraction=0.0, sar_water_fraction=0.004)
BUILT = Measurements(optical_builtup_fraction=0.2803)


def always(sample_id, seed, rate):
    return True


@pytest.fixture(autouse=True)
def _select_every_record(monkeypatch):
    """Take the sampling gate out of the way; it is tested on its own below."""
    monkeypatch.setattr("satquery.data.evidence._selected", always)


# -- parity with the serving path ----------------------------------------


def test_preamble_matches_format_evidence_byte_for_byte():
    """The synthesised preamble is the controller's own output, not a copy."""
    choice = choose_evidence("s1", "How many forests?", "three", WET)
    assert choice.preamble == format_evidence(WET.as_artifacts())


def test_rendered_turn_matches_what_vlmtool_builds(tmp_path, monkeypatch):
    """End to end: the training turn equals the prompt VLMTool sends.

    Built by actually running the tool against a stub backend and capturing the
    prompt, rather than by re-deriving the join here -- a test that repeats the
    implementation would agree with a wrong implementation.
    """
    from PIL import Image

    from satquery.agent.context import RunContext
    from satquery.agent.tools.vlm import VLMTool
    from satquery.eval.prompts import build_prompt
    from satquery.schema import ImageRef, InputConfig, Sample, Task

    path = tmp_path / "scene.png"
    Image.new("RGB", (16, 16), "green").save(path)

    captured: dict = {}

    class _Backend:
        name = "stub"

        def generate_with_meta(self, requests):
            captured["prompt"] = requests[0].prompt
            return [("a lake", {})]

    ctx = RunContext(
        run_id="r1",
        query="Is there water here?",
        images=(ImageRef(path=path, modality="optical", role="primary"),),
        infos=(),
        input_config=InputConfig.SINGLE,
        workdir=tmp_path / "work",
        backend=_Backend(),
    )
    ctx.artifacts.update(WET.as_artifacts())
    VLMTool("vlm_vqa", Task.VQA, InputConfig.SINGLE).run(ctx)

    task_prompt = build_prompt(
        Sample(
            sample_id="s",
            task=Task.VQA,
            images=(ImageRef(path=path, modality="optical", role="primary"),),
            question="Is there water here?",
        )
    )
    choice = choose_evidence("s1", task_prompt, "a lake", WET)
    # Non-vacuous: both sides must actually carry the measurement block, or the
    # equality below would hold for two identical bare prompts.
    assert choice.preamble
    assert "water fraction from optical NDWI: 0.3151" in captured["prompt"]
    assert apply_preamble(task_prompt, choice.preamble) == captured["prompt"]


def test_every_measurement_key_is_renderable():
    """A key the label map does not know renders as nothing at all.

    Measurements exists to be handed to format_evidence(), so a field added to
    one and not the other produces a preamble silently missing a line.
    """
    keys = set(
        Measurements(
            optical_water_fraction=0.1,
            optical_builtup_fraction=0.1,
            sar_water_fraction=0.1,
            sar_builtup_location="north-east quadrant",
            landcover_classes="arable land",
        ).as_artifacts()
    )
    assert keys <= set(_EVIDENCE_LABELS), sorted(keys - set(_EVIDENCE_LABELS))


def test_cnn_evidence_format_is_settled_before_the_finetune():
    """The land-cover line must already render, or it cannot be trained on."""
    rendered = format_evidence({"landcover_classes": "arable land, pastures"})
    assert "land cover classes present: arable land, pastures" in rendered


def test_apply_preamble_leaves_a_bare_prompt_alone():
    assert apply_preamble("Question?", "") == "Question?"


# -- never contradictory -------------------------------------------------


def test_affirmative_answer_with_no_water_measured_gets_no_preamble():
    """The guard that matters: never train a preamble that argues with its gold."""
    choice = choose_evidence("s1", "Is there water?", "yes", DRY)
    assert choice.kind is EvidenceKind.NONE
    assert choice.preamble == ""


def test_negative_answer_with_water_measured_gets_no_preamble():
    choice = choose_evidence("s1", "Is there water in this image?", "no", WET)
    assert choice.kind is EvidenceKind.NONE


def test_agreeing_affirmative_is_decisive():
    choice = choose_evidence("s1", "Is there water in this image?", "yes", WET)
    assert choice.kind is EvidenceKind.DECISIVE
    assert "water fraction from optical NDWI: 0.3151" in choice.preamble


def test_agreeing_negative_is_decisive():
    choice = choose_evidence("s1", "Is there water in this image?", "no", DRY)
    assert choice.kind is EvidenceKind.DECISIVE


def test_equivocal_measurement_is_not_called_decisive():
    """Between the two thresholds the evidence is not firm enough to assert."""
    middle = (ABSENT_BELOW + PRESENT_ABOVE) / 2
    borderline = Measurements(optical_water_fraction=middle)
    assert choose_evidence("s1", "Is there water?", "yes", borderline).kind is (
        EvidenceKind.NONE
    )
    assert choose_evidence("s1", "Is there water?", "no", borderline).kind is (
        EvidenceKind.NONE
    )


def test_open_answer_naming_water_needs_the_measurement_to_back_it():
    """A caption asserting a lake, on a patch with no water, gets no preamble."""
    caption = "A large lake surrounded by forest."
    assert choose_evidence("s1", "Describe.", caption, DRY).kind is EvidenceKind.NONE
    assert choose_evidence("s1", "Describe.", caption, WET).kind is (
        EvidenceKind.DECISIVE
    )


def test_builtup_topic_is_checked_against_the_builtup_measurement():
    assert choose_evidence("s1", "Is this urban?", "yes", BUILT).kind is (
        EvidenceKind.DECISIVE
    )
    assert (
        choose_evidence(
            "s1", "Is this urban?", "yes", Measurements(optical_builtup_fraction=0.001)
        ).kind
        is EvidenceKind.NONE
    )


def test_a_question_on_two_topics_needs_both_to_agree():
    """Water agrees and built-up does not, so the record carries no preamble."""
    mixed = Measurements(optical_water_fraction=0.31, optical_builtup_fraction=0.001)
    choice = choose_evidence("s1", "Is there water near the city?", "yes", mixed)
    assert choice.kind is EvidenceKind.NONE


# -- orthogonal ----------------------------------------------------------


def test_unrelated_question_still_carries_evidence():
    """At inference the specialists run whatever was asked, so training must too."""
    choice = choose_evidence("s1", "How many forest patches are there?", "three", WET)
    assert choice.kind is EvidenceKind.ORTHOGONAL
    assert "water fraction" in choice.preamble


def test_no_measurements_means_no_preamble():
    choice = choose_evidence("s1", "Anything?", "yes", Measurements())
    assert choice.kind is EvidenceKind.NONE


# -- the sampling gate ---------------------------------------------------


def test_selection_is_stable_across_runs(monkeypatch):
    """A rerun must rebuild the identical corpus, not a differently mixed one."""
    monkeypatch.undo()
    from satquery.data.evidence import _selected

    first = [_selected(f"id-{i}", 1234, 0.45) for i in range(200)]
    second = [_selected(f"id-{i}", 1234, 0.45) for i in range(200)]
    assert first == second


def test_selection_rate_lands_near_the_target(monkeypatch):
    monkeypatch.undo()
    from satquery.data.evidence import _selected

    picked = sum(_selected(f"id-{i}", 1234, 0.45) for i in range(4000))
    assert 0.42 < picked / 4000 < 0.48


def test_a_different_seed_gives_a_different_mixture(monkeypatch):
    monkeypatch.undo()
    from satquery.data.evidence import _selected

    a = [_selected(f"id-{i}", 1, 0.45) for i in range(200)]
    b = [_selected(f"id-{i}", 2, 0.45) for i in range(200)]
    assert a != b


# -- measuring -----------------------------------------------------------


def test_measure_optical_uses_the_same_functions_as_the_tool():
    """Twelve bands: NDWI from green/NIR, NDBI from SWIR/NIR."""
    rng = np.random.default_rng(0)
    stack = rng.random((12, 16, 16), dtype=np.float32)
    out = measure_optical(stack)

    from satquery.agent.tools.indices import mask_fraction, ndbi_builtup, ndwi_water

    assert out["optical_water_fraction"] == mask_fraction(
        ndwi_water(stack[2], stack[7])
    )
    assert out["optical_builtup_fraction"] == mask_fraction(
        ndbi_builtup(stack[10], stack[7])
    )


def test_measure_optical_reports_nothing_without_nir():
    """Three bands cannot yield NDWI, and inventing one would be worse."""
    assert measure_optical(np.zeros((3, 8, 8), dtype=np.float32)) == {}


def test_measure_sar_reads_the_rendered_png_through_the_serving_path(tmp_path):
    from PIL import Image

    rng = np.random.default_rng(1)
    array = rng.integers(0, 255, (32, 32, 3), dtype=np.uint8)
    path = tmp_path / "sar.png"
    Image.fromarray(array, mode="RGB").save(path)

    out = measure_sar(path)
    assert 0.0 <= out["sar_water_fraction"] <= 1.0
    # A location, not a fraction. The fraction above a percentile is ~5% on
    # every scene ever measured, so a corpus carrying it teaches a constant.
    assert isinstance(out["sar_builtup_location"], str)
    assert out["sar_builtup_location"]
    assert "sar_builtup_fraction" not in out


def test_describe_classes_renders_and_truncates():
    assert describe_classes(["Arable land", "Pastures"]) == "arable land, pastures"
    assert describe_classes([]) is None
    assert describe_classes(["a", "b", "c", "d", "e"], limit=2) == "a, b"


# -- end to end through prepare() ----------------------------------------


def test_prepare_attaches_preambles_and_records_the_kind(tmp_path):
    """The mixture reaches the JSONL, and the human turn carries the preamble."""
    import json

    pd = pytest.importorskip("pandas")

    from satquery.data.bigearthnet import prepare, write_jsonl

    frame = pd.DataFrame(
        [
            {
                "ID": "r1",
                "patch_id": "p1",
                "type": "binary",
                "category": "water",
                "input": "Is there water in this image?",
                "output": "yes",
                "country": "Lithuania",
            },
            {
                "ID": "r2",
                "patch_id": "p1",
                "type": "binary",
                "category": "forest",
                "input": "Is there forest in this image?",
                "output": "yes",
                "country": "Lithuania",
            },
        ]
    )
    images = {"p1": ("images/p1_optical.png", "images/p1_sar.png")}
    records = list(prepare(frame, images, measurements={"p1": WET}, preamble_rate=1.0))

    assert [r.evidence_kind for r in records] == ["decisive", "orthogonal"]
    for record in records:
        assert record.rendered_prompt.startswith("Measurements from image-analysis")
        assert record.prompt in record.rendered_prompt

    path = tmp_path / "train.jsonl"
    assert write_jsonl(iter(records), path) == 2
    first = json.loads(path.read_text(encoding="utf-8").splitlines()[0])
    assert first["evidence_kind"] == "decisive"
    assert (
        "water fraction from optical NDWI: 0.3151"
        in (first["conversations"][0]["value"])
    )
    assert first["conversations"][1]["value"] == "yes"


def test_prepare_without_measurements_yields_bare_prompts(tmp_path):
    """A patch we could not measure degrades the mixture, never the run."""
    pd = pytest.importorskip("pandas")

    from satquery.data.bigearthnet import prepare

    frame = pd.DataFrame(
        [
            {
                "ID": "r1",
                "patch_id": "p1",
                "type": "binary",
                "category": "water",
                "input": "Is there water?",
                "output": "yes",
                "country": "Lithuania",
            }
        ]
    )
    records = list(prepare(frame, {"p1": ("a.png", "b.png")}, measurements={}))
    assert [r.evidence_kind for r in records] == ["none"]
    assert records[0].rendered_prompt == records[0].prompt


def test_the_sar_bright_tail_fraction_is_a_constant_and_is_not_evidence():
    """Why SAR contributes a location rather than a built-up fraction.

    The tool's built-up mask is "pixels above the 95th percentile", so ~5% of
    every scene qualifies whatever it contains -- open water and a dense city
    alike. Telling the model not to contradict a constant is worse than telling
    it nothing, and training on one teaches that built-up is always 0.05.
    """
    from satquery.agent.tools.indices import mask_fraction
    from satquery.agent.tools.vlm import _EVIDENCE_LABELS

    rng = np.random.default_rng(0)
    scenes = [
        rng.normal(0.1, 0.01, (64, 64)),  # uniformly dark, no built-up at all
        rng.normal(0.5, 0.05, (64, 64)),  # uniform mid-tone
        rng.normal(0.9, 0.03, (64, 64)),  # uniformly bright
    ]
    fractions = {mask_fraction(s > float(np.percentile(s, 95.0))) for s in scenes}
    assert fractions == {0.05}, f"expected a constant, got {fractions}"

    assert "sar_builtup_fraction" not in _EVIDENCE_LABELS
    assert "sar_builtup_location" in _EVIDENCE_LABELS
