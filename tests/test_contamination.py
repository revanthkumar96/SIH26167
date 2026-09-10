"""The guard that stands between Stage B and a meaningless number.

Stage A's measured delta (+0.0760 overall) is defensible only because the model
never saw the rows it was scored on. Stage B trains on the *train* splits of the
same three benchmarks, and the file naming in those releases is close enough that
pointing a config at the wrong JSON is a one-character mistake. If that happened,
every downstream signal would look *better*: loss falls, scores rise, and the
submission claims an improvement that is recall.

So these tests check the two things that matter -- that the guard finds the
overlap, and that finding it stops the build rather than printing a warning.
"""

from __future__ import annotations

import json

import pytest

from satquery.data.contamination import (
    ContaminationError,
    ContaminationReport,
    Fingerprints,
    check_records,
    fingerprint_samples,
    image_key,
    question_key,
    read_corpus,
)
from satquery.schema import ImageRef, Modality, Sample, Task


def sample(sample_id: str, image: str, question: str | None = "how many buildings?"):
    return Sample(
        sample_id=sample_id,
        task=Task.VQA,
        images=(
            ImageRef(path=f"/data/VRSBench/Images_val/{image}", modality=Modality.RGB),
        ),
        question=question,
        answer="three",
    )


def record(record_id: str, image: str, question: str = "how many buildings?"):
    return {
        "id": record_id,
        "images": [f"train/images/{image}"],
        "conversations": [
            {"from": "human", "value": question},
            {"from": "gpt", "value": "three"},
        ],
    }


# -- identity ------------------------------------------------------------


def test_image_key_ignores_the_directory():
    """The same picture lives under different roots in train and test archives."""
    assert image_key("/data/VRSBench/Images_val/P0001.png") == image_key(
        "s3://bucket/train/P0001.png"
    )


def test_image_key_keeps_the_extension():
    """`1234.png` and `1234.tif` are different products, not the same file."""
    assert image_key("a/1234.png") != image_key("b/1234.tif")


def test_image_key_survives_windows_separators():
    assert image_key(r"C:\data\VRSBench\Images_val\P0001.PNG") == "p0001.png"


def test_question_key_normalises_whitespace_and_case():
    assert question_key("  How   MANY\nbuildings? ") == "how many buildings?"


def test_question_key_of_nothing_is_empty():
    assert question_key(None) == ""


# -- detection -----------------------------------------------------------


def test_clean_corpus_passes():
    marks = fingerprint_samples([sample("t-1", "P0001.png")])
    report = check_records(
        [record("r-1", "P9999.png", "what colour is the roof?")], marks
    )
    assert report.clean
    assert report.checked == 1
    report.raise_if_contaminated()  # must not raise


def test_shared_image_alone_is_contamination():
    """The strongest signal. A seen test image cannot be unseen by rephrasing."""
    marks = fingerprint_samples([sample("t-1", "P0001.png")])
    report = check_records(
        [record("r-1", "P0001.png", "a completely different question")], marks
    )
    assert not report.clean
    assert report.overlaps[0].kind == "image"
    assert report.overlaps[0].record_id == "r-1"


def test_shared_image_and_question_is_reported_as_the_pair():
    marks = fingerprint_samples([sample("t-1", "P0001.png")])
    report = check_records([record("r-1", "P0001.png")], marks)
    assert report.overlaps[0].kind == "image+question"


def test_shared_question_on_a_new_image_is_the_weak_signal():
    """Templated corpora reuse phrasings legitimately, so this is a flag not a verdict."""
    marks = fingerprint_samples([sample("t-1", "P0001.png")])
    report = check_records([record("r-1", "P4242.png")], marks)
    assert report.overlaps[0].kind == "question"


def test_require_pair_ignores_everything_but_the_exact_row():
    """The lenient mode exists; it is off by default and this shows why."""
    marks = fingerprint_samples([sample("t-1", "P0001.png")])
    leaked = [record("r-1", "P0001.png", "rephrased entirely")]
    assert check_records(leaked, marks).clean is False
    assert check_records(leaked, marks, require_pair=True).clean is True


def test_case_and_path_differences_do_not_hide_a_leak():
    """The realistic failure: same file, different archive layout and casing."""
    marks = fingerprint_samples([sample("t-1", "P0001.png")])
    leaked = {
        "id": "r-1",
        "images": [r"staged\VRSBench\train\P0001.PNG"],
        "conversations": [{"from": "human", "value": "anything"}],
    }
    assert not check_records([leaked], marks).clean


def test_multi_image_record_is_caught_on_its_second_image():
    """Change and cross-modal records carry two images; both must be checked."""
    marks = fingerprint_samples([sample("t-1", "after.png")])
    leaked = {
        "id": "r-1",
        "images": ["train/before.png", "train/after.png"],
        "conversations": [{"from": "human", "value": "what changed?"}],
    }
    assert not check_records([leaked], marks).clean


def test_records_without_images_or_turns_do_not_crash_the_guard():
    marks = fingerprint_samples([sample("t-1", "P0001.png")])
    report = check_records([{"id": "r-1"}, {"id": "r-2", "images": []}], marks)
    assert report.clean
    assert report.checked == 2


def test_empty_fingerprints_report_zero_benchmark_images():
    """A guard that checked nothing must not look like a guard that passed."""
    report = check_records([record("r-1", "P0001.png")], Fingerprints())
    assert report.clean
    assert report.benchmark_images == 0


# -- the part that makes it a guard rather than a lint --------------------


def test_contaminated_report_raises():
    marks = fingerprint_samples([sample("t-1", "P0001.png")])
    report = check_records([record("r-1", "P0001.png")], marks)
    with pytest.raises(ContaminationError, match="CONTAMINATED"):
        report.raise_if_contaminated()


def test_the_error_says_what_to_do_about_it():
    """An exception nobody can act on gets suppressed, and then it is a comment."""
    report = ContaminationReport(checked=1, benchmark_images=1)
    report.overlaps.append(
        check_records(
            [record("r-1", "P0001.png")],
            fingerprint_samples([sample("t-1", "P0001.png")]),
        ).overlaps[0]
    )
    with pytest.raises(ContaminationError) as caught:
        report.raise_if_contaminated()
    assert "rebuild the corpus from the train split" in str(caught.value)


def test_summary_names_the_offending_records():
    marks = fingerprint_samples([sample("t-1", "P0001.png")])
    report = check_records([record("bad-record", "P0001.png")], marks)
    assert "bad-record" in report.summary()
    assert "p0001.png" in report.summary()


def test_by_kind_counts_each_signal():
    marks = fingerprint_samples(
        [sample("t-1", "P0001.png"), sample("t-2", "P0002.png", "what colour?")]
    )
    report = check_records(
        [
            record("r-1", "P0001.png"),  # image+question
            record("r-2", "P0002.png", "unrelated"),  # image
            record("r-3", "P7777.png"),  # question
            record("r-4", "P8888.png", "unrelated"),  # clean
        ],
        marks,
    )
    assert report.by_kind == {"image+question": 1, "image": 1, "question": 1}
    assert report.checked == 4


# -- fingerprinting and corpus reading ------------------------------------


def test_fingerprint_samples_collects_images_questions_and_pairs():
    marks = fingerprint_samples([sample("t-1", "P0001.png", "How Many Buildings?")])
    assert marks.images == {"p0001.png"}
    assert marks.questions == {"how many buildings?"}
    assert marks.pairs == {("p0001.png", "how many buildings?")}
    assert len(marks) == 1


def test_captioning_samples_contribute_images_but_no_question():
    caption = Sample(
        sample_id="c-1",
        task=Task.CAPTION,
        images=(ImageRef(path="/data/Images_val/P0003.png"),),
        answer="a harbour with moored vessels",
    )
    marks = fingerprint_samples([caption])
    assert marks.images == {"p0003.png"}
    assert not marks.questions and not marks.pairs


def test_read_corpus_streams_jsonl_and_skips_blank_lines(tmp_path):
    path = tmp_path / "train.jsonl"
    path.write_text(
        json.dumps(record("r-1", "a.png"))
        + "\n\n"
        + json.dumps(record("r-2", "b.png"))
        + "\n",
        encoding="utf-8",
    )
    rows = list(read_corpus(path))
    assert [r["id"] for r in rows] == ["r-1", "r-2"]
