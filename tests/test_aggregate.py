"""Score normalisation before aggregation.

The requirement exists because our own metrics are on different scales, so the
first test is the one that would fail without any normalisation at all: CIDEr-D
runs to 10 and everything else to 1, and a plain mean lets captioning outvote
every other criterion.
"""

from __future__ import annotations

import pytest

from satquery.eval.aggregate import (
    CRITERIA,
    METRIC_RANGES,
    NormalisationError,
    cells_from_results,
    delta_table,
    normalise_metric,
    score_matrix,
    score_model,
)
from satquery.eval.report import HEADLINE_METRIC


def cell(benchmark, task, metrics, model="base", num_samples=200):
    return {
        "model_id": model,
        "benchmark": benchmark,
        "task": task,
        "metrics": metrics,
        "num_samples": num_samples,
    }


# -- the reason normalisation is required --------------------------------


def test_cider_dominates_an_unnormalised_mean():
    """A raw mean is carried by captioning; a normalised one is not.

    Grounding here is excellent (0.9) and captioning mediocre for its scale
    (3.0 of a possible 10). Averaged raw, the mediocre number is three times the
    excellent one and sets the result.
    """
    scored = score_model(
        "m",
        [
            cell("vrsbench_caption", "caption", {"cider_d": 3.0}),
            cell("vrsbench_referring", "grounding", {"acc@0.5": 0.9}),
        ],
    )

    raw_mean = (3.0 + 0.9) / 2
    assert raw_mean > 1.0  # meaningless as a score at all

    assert scored.criteria["captioning"] == pytest.approx(0.30)
    assert scored.criteria["grounding"] == pytest.approx(0.90)
    assert scored.overall == pytest.approx(0.60)


def test_every_headline_metric_has_a_declared_range():
    """No task can be ranked by a metric the aggregator cannot normalise."""
    missing = sorted(set(HEADLINE_METRIC.values()) - set(METRIC_RANGES))
    assert not missing, f"headline metrics with no declared range: {missing}"


# -- normalisation -------------------------------------------------------


@pytest.mark.parametrize(
    ("metric", "value", "expected"),
    [
        ("oa", 0.0, 0.0),
        ("oa", 0.75, 0.75),
        ("acc@0.5", 1.0, 1.0),
        ("cider_d", 0.0, 0.0),
        ("cider_d", 5.0, 0.5),
        ("cider_d", 10.0, 1.0),
    ],
)
def test_normalise_metric_maps_onto_unit_range(metric, value, expected):
    assert normalise_metric(metric, value) == pytest.approx(expected)


def test_out_of_range_value_raises_rather_than_clamping():
    """A value past its bound means the metric moved, which is worth failing on."""
    with pytest.raises(NormalisationError, match="outside its declared range"):
        normalise_metric("oa", 1.5)


def test_unknown_metric_raises_with_an_actionable_message():
    with pytest.raises(NormalisationError, match="METRIC_RANGES"):
        normalise_metric("some_new_metric", 0.5)


def test_float_drift_at_the_bound_is_tolerated():
    """CIDEr-D can accumulate a hair past 10.0; that is arithmetic, not a bug."""
    assert normalise_metric("cider_d", 10.0 + 1e-12) == pytest.approx(1.0)


# -- criteria, not benchmarks --------------------------------------------


def test_benchmarks_are_averaged_within_their_criterion():
    """Three VRSBench splits must not outweigh one CDVQA split.

    Single-image VQA scores 0.0 across three benchmarks and change understanding
    1.0 across one. A flat mean over benchmarks gives 0.25; averaging within
    criteria first gives 0.5, which is what equal criterion weighting means.
    """
    scored = score_model(
        "m",
        [
            cell("vrsbench_vqa", "vqa", {"oa": 0.0}),
            cell("rsvqa_lr", "vqa", {"oa": 0.0}),
            cell("rsvqa_hr", "vqa", {"oa": 0.0}),
            cell("cdvqa", "change_vqa", {"oa": 1.0}),
        ],
    )
    assert scored.criteria["single_image_vqa"] == pytest.approx(0.0)
    assert scored.criteria["change_understanding"] == pytest.approx(1.0)
    assert scored.overall == pytest.approx(0.5)


def test_change_tasks_share_one_criterion():
    scored = score_model(
        "m",
        [
            cell("cdvqa", "change_vqa", {"oa": 0.4}),
            cell("cdvqa_cap", "change_caption", {"cider_d": 8.0}),
        ],
    )
    assert set(scored.criteria) == {"change_understanding"}
    assert scored.criteria["change_understanding"] == pytest.approx((0.4 + 0.8) / 2)


# -- honesty about what did not run --------------------------------------


def test_failed_cell_is_skipped_with_a_reason_not_dropped():
    """A benchmark that produced no headline metric must not lift the mean."""
    scored = score_model(
        "m",
        [
            cell("vrsbench_vqa", "vqa", {"oa": 0.8}),
            cell("vrsbench_referring", "grounding", {}),
        ],
    )
    assert scored.criteria == {"single_image_vqa": pytest.approx(0.8)}
    assert [s["benchmark"] for s in scored.skipped] == ["vrsbench_referring"]
    assert "acc@0.5" in scored.skipped[0]["reason"]


def test_missing_mandatory_criteria_are_reported_first():
    scored = score_model("m", [cell("vrsbench_caption", "caption", {"cider_d": 4.0})])

    mandatory = [name for name, (_, is_required) in CRITERIA.items() if is_required]
    assert set(scored.missing_criteria) >= set(mandatory)
    assert scored.missing_criteria[: len(mandatory)] == mandatory


def test_no_scoreable_cells_yields_no_overall():
    """An empty aggregate reports nothing rather than a confident zero."""
    scored = score_model("m", [])
    assert scored.overall is None
    assert scored.criteria == {}


# -- the base-versus-adapted artefact ------------------------------------


def test_score_matrix_groups_by_model_preserving_order():
    scores = score_matrix(
        [
            cell("cdvqa", "change_vqa", {"oa": 0.4}, model="base"),
            cell("cdvqa", "change_vqa", {"oa": 0.6}, model="adapted"),
        ]
    )
    assert [s.model for s in scores] == ["base", "adapted"]
    assert scores[1].overall == pytest.approx(0.6)


def test_delta_table_reports_the_change_against_the_first_row():
    scores = score_matrix(
        [
            cell("cdvqa", "change_vqa", {"oa": 0.40}, model="base"),
            cell("cdvqa", "change_vqa", {"oa": 0.55}, model="adapted"),
        ]
    )
    table = delta_table(scores)
    assert "+0.1500" in table
    assert "base" in table and "adapted" in table


def test_delta_table_handles_a_single_row():
    scores = score_matrix([cell("cdvqa", "change_vqa", {"oa": 0.4})])
    assert "delta" not in delta_table(scores)


# -- reading back the long-format results CSV ----------------------------


def result_row(model, benchmark, task, metric, value, timestamp="t1", n=200):
    return {
        "timestamp": timestamp,
        "benchmark": benchmark,
        "task": task,
        "backend": "vllm",
        "model": model,
        "num_samples": str(n),
        "metric": metric,
        "value": str(value),
        "duration_s": "1.0",
        "prompt_version": "1.1.0",
        "config_hash": "abc",
        "git_sha": "sha",
    }


def test_cells_from_results_pivots_metrics_back_together():
    cells = cells_from_results(
        [
            result_row("m", "cdvqa", "change_vqa", "oa", 0.4),
            result_row("m", "cdvqa", "change_vqa", "aa", 0.3),
        ]
    )
    assert len(cells) == 1
    assert cells[0]["metrics"] == {"oa": 0.4, "aa": 0.3}
    assert cells[0]["num_samples"] == 200


def test_runs_at_different_limits_stay_separate():
    """A 200-sample probe must not be averaged into a 1,000-sample headline."""
    cells = cells_from_results(
        [
            result_row("m", "cdvqa", "change_vqa", "oa", 0.40, timestamp="t1", n=200),
            result_row("m", "cdvqa", "change_vqa", "oa", 0.44, timestamp="t2", n=1000),
        ]
    )
    assert len(cells) == 2
    assert sorted(c["num_samples"] for c in cells) == [200, 1000]
