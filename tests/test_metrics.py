"""Metric behaviour, especially the properties the judging table depends on."""

import pytest

from satquery.eval.metrics.caption import bleu, caption_metrics, cider_d, rouge_l
from satquery.eval.metrics.grounding import grounding_metrics, iou
from satquery.eval.metrics.vqa import vqa_metrics


def test_vqa_perfect_and_zero():
    assert vqa_metrics(["yes", "no"], ["yes", "no"])["oa"] == 1.0
    assert vqa_metrics(["yes", "no"], ["no", "yes"])["oa"] == 0.0


def test_vqa_aa_exposes_class_imbalance():
    """A majority-class model posts high OA but poor AA -- the reason both ship."""
    predictions = ["no"] * 9 + ["no"]
    references = ["no"] * 9 + ["yes"]
    qtypes = ["presence"] * 9 + ["count"]

    metrics = vqa_metrics(predictions, references, qtypes)

    assert metrics["oa"] == pytest.approx(0.9)
    assert metrics["aa"] == pytest.approx(0.5)
    assert metrics["acc/presence"] == 1.0
    assert metrics["acc/count"] == 0.0


def test_vqa_aa_falls_back_to_oa_without_types():
    metrics = vqa_metrics(["yes", "no"], ["yes", "yes"])
    assert metrics["aa"] == metrics["oa"] == pytest.approx(0.5)


def test_vqa_length_mismatch_raises():
    with pytest.raises(ValueError, match="same length"):
        vqa_metrics(["yes"], ["yes", "no"])


def test_bleu_identical_text_is_one():
    text = ["a large forest area next to a river"]
    scores = bleu(text, [text])
    assert scores["bleu4"] == pytest.approx(1.0)


def test_bleu_brevity_penalty_punishes_short_output():
    reference = [["a large forest area next to a wide river"]]
    full = bleu(["a large forest area next to a wide river"], reference)["bleu4"]
    short = bleu(["a large forest"], reference)["bleu4"]
    assert short < full


def test_rouge_l_bounds():
    assert rouge_l(["water body"], [["water body"]]) == pytest.approx(1.0)
    assert rouge_l(["completely unrelated"], [["water body"]]) == pytest.approx(0.0)


def test_cider_rewards_matching_corpus():
    candidates = ["a river runs through farmland", "dense urban buildings"]
    references = [["a river runs through farmland"], ["dense urban buildings"]]
    matched = cider_d(candidates, references)
    swapped = cider_d(candidates[::-1], references)
    assert matched > swapped


def test_cider_empty_corpus():
    assert cider_d([], []) == 0.0


def test_caption_metrics_reports_every_prescribed_metric():
    metrics = caption_metrics(["a river"], [["a river"]])
    for key in ("bleu1", "bleu2", "bleu3", "bleu4", "rouge_l", "cider_d"):
        assert key in metrics


def test_iou_basic_cases():
    assert iou((0, 0, 1, 1), (0, 0, 1, 1)) == pytest.approx(1.0)
    assert iou((0, 0, 0.5, 0.5), (0.5, 0.5, 1, 1)) == 0.0
    assert iou(None, (0, 0, 1, 1)) == 0.0
    assert iou((0, 0, 1, 1), (0, 0, 0.5, 1)) == pytest.approx(0.5)


def test_grounding_counts_unparseable_as_miss():
    """Dropping unparseable boxes would inflate the score, so they score zero."""
    metrics = grounding_metrics(
        [(0.0, 0.0, 1.0, 1.0), None], [(0.0, 0.0, 1.0, 1.0), (0.0, 0.0, 1.0, 1.0)]
    )
    assert metrics["acc@0.5"] == pytest.approx(0.5)
    assert metrics["miou"] == pytest.approx(0.5)
    assert metrics["parse_rate"] == pytest.approx(0.5)
