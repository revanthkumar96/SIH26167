"""Reading VRSBench's train file, which is not shaped like its EVAL files.

Every fixture here is copied from the real ``VRSBench_train.json`` v1.2 rather
than invented, because the failures this adapter has to avoid are all failures of
assumption: that the boxes are on the same grid as the EVAL release (they are
not), that each instruction template pairs a fixed prefix with a fixed suffix
(8,443 records disprove it), and that a file called "train" from the same
publisher can be read by the same adapter (it cannot).

None of those would raise. Each would produce a corpus that trains the model
slightly wrong, which is the expensive kind of bug here -- five GPU-hours later
it looks like the model failed to learn.
"""

from __future__ import annotations

import json

import pytest

from satquery.eval.datasets import BenchmarkConfig, load_benchmark
from satquery.eval.datasets.vrsbench_train import (
    BOX_GRID,
    VRSBenchTrainError,
    parse_box,
    strip_question,
)
from satquery.schema import Task


def conversation(human: str, gpt: str, image: str = "00002_0000.png"):
    return {
        "id": "Final_Data/v1.2",
        "image": image,
        "conversations": [
            {"from": "human", "value": human},
            {"from": "gpt", "value": gpt},
        ],
    }


def build(tmp_path, rows, tag, task):
    (tmp_path / "VRSBench_train.json").write_text(json.dumps(rows), encoding="utf-8")
    return BenchmarkConfig.from_dict(
        {
            "name": f"vrsbench_train_{tag}",
            "adapter": "vrsbench_train",
            "task": task,
            "root": str(tmp_path),
            "annotations": "VRSBench_train.json",
            "image_dir": "Images_train",
            "extra": {"task_tag": tag},
        }
    )


# -- the box grid --------------------------------------------------------


def test_boxes_are_read_on_the_0_100_grid():
    """The EVAL release is pixels on 512; this file is a 0-100 grid."""
    assert BOX_GRID == 100.0
    assert parse_box("{<45><45><59><59>}") == (0.45, 0.45, 0.59, 0.59)


def test_out_of_range_coordinates_are_clamped_not_rejected():
    """8.8% of published coordinates fall outside the grid; min -73, max 196."""
    assert parse_box("{<-73><10><196><90>}") == (0.0, 0.1, 1.0, 0.9)


def test_inverted_boxes_are_ordered():
    assert parse_box("{<59><59><45><45>}") == (0.45, 0.45, 0.59, 0.59)


def test_a_box_that_clamps_to_nothing_is_dropped():
    """An empty region teaches the model to answer a localisation with a point."""
    assert parse_box("{<120><120><150><150>}") is None
    assert parse_box("{<-40><-40><-10><-10>}") is None


def test_zero_area_box_is_dropped():
    assert parse_box("{<50><50><50><50>}") is None


def test_text_without_a_box_returns_none():
    assert parse_box("somewhere near the middle") is None
    assert parse_box("") is None


# -- instruction wrappers ------------------------------------------------


@pytest.mark.parametrize(
    "wrapped",
    [
        "The question What is the main object? can be answered using the image. "
        "A short answer is",
        "Based on the image, respond to this question with a short answer: "
        "What is the main object?.",
        "Use the provided image to answer the question: What is the main object? "
        "Provide your answer as short as possible.",
        "Given the image, answer the following question with no more than three "
        "words. What is the main object?",
        "Question: What is the main object? Short answer:",
        "Q: What is the main object? A:",
        "What is the main object?. A short answer to the question is",
        "What is the main object?",
    ],
)
def test_every_published_wrapper_recovers_the_bare_question(wrapped):
    assert strip_question(wrapped) == "What is the main object?"


def test_a_prefix_without_its_usual_suffix_is_still_stripped():
    """8,443 records carry `Question:` with no `Short answer:` after it.

    Matching fixed prefix/suffix templates as whole units leaves the unpaired
    half behind, and the model trains on "Question: is there a vehicle?".
    """
    assert strip_question("Question: Are trees present near the service area?") == (
        "Are trees present near the service area?"
    )


def test_a_suffix_without_its_usual_prefix_is_still_stripped():
    assert strip_question("How many ships are visible? Short answer:") == (
        "How many ships are visible?"
    )


def test_stripping_never_empties_a_question():
    """A question that reads like scaffolding must survive, not vanish."""
    assert strip_question("Question:") == "Question:"


def test_unwrapped_text_is_returned_unchanged():
    assert strip_question("  Is the image color or grayscale?  ") == (
        "Is the image color or grayscale?"
    )


# -- task separation -----------------------------------------------------


def test_each_tag_loads_only_its_own_rows(tmp_path):
    """All three tasks share one file; a config that did not filter would mix
    caption targets into a VQA corpus."""
    rows = [
        conversation("<image>\n[caption] Describe the image in detail", "A harbour."),
        conversation("<image>\n[vqa] How many ships are visible?", "three"),
        conversation(
            "<image>\n[refer] give me the location of <p>the left ship</p>",
            "{<10><20><30><40>}",
        ),
    ]
    for tag, task, expected in (
        ("caption", "caption", Task.CAPTION),
        ("vqa", "vqa", Task.VQA),
        ("refer", "grounding", Task.GROUNDING),
    ):
        samples = load_benchmark(build(tmp_path, rows, tag, task)).load()
        assert len(samples) == 1
        assert samples[0].task is expected


def test_referring_expression_comes_from_the_p_wrapper(tmp_path):
    rows = [
        conversation(
            "<image>\n[refer] could you tell me the location for "
            "<p>The toll station is at the center</p>?",
            "{<45><45><59><59>}",
        )
    ]
    sample = load_benchmark(build(tmp_path, rows, "refer", "grounding")).load()[0]
    assert sample.question == "The toll station is at the center"
    assert sample.bbox == (0.45, 0.45, 0.59, 0.59)


def test_caption_keeps_the_target_and_discards_the_published_instruction(tmp_path):
    """The published phrasing is a paraphrase; build_prompt supplies ours."""
    rows = [
        conversation(
            "<image>\n[caption] Could you describe the contents of this image for me?",
            "The image shows a rural area with a toll station.",
        )
    ]
    sample = load_benchmark(build(tmp_path, rows, "caption", "caption")).load()[0]
    assert sample.question is None
    assert sample.answer == "The image shows a rural area with a toll station."
    assert sample.references == (sample.answer,)


def test_the_image_token_never_reaches_the_question(tmp_path):
    rows = [conversation("<image>\n[vqa] How many ships are visible?", "three")]
    sample = load_benchmark(build(tmp_path, rows, "vqa", "vqa")).load()[0]
    assert "<image>" not in sample.question


def test_image_path_is_resolved_under_the_image_dir(tmp_path):
    rows = [conversation("<image>\n[vqa] How many?", "three", image="07777_0000.png")]
    sample = load_benchmark(build(tmp_path, rows, "vqa", "vqa")).load()[0]
    assert sample.images[0].path.name == "07777_0000.png"
    assert sample.images[0].path.parent.name == "Images_train"


def test_sample_ids_are_unique_despite_the_constant_id_field(tmp_path):
    """Every record's `id` is the literal string 'Final_Data/v1.2'."""
    rows = [conversation(f"<image>\n[vqa] Q{i}?", "a") for i in range(5)]
    samples = load_benchmark(build(tmp_path, rows, "vqa", "vqa")).load()
    assert len({s.sample_id for s in samples}) == 5


# -- refusing to guess ---------------------------------------------------


def test_a_config_without_a_task_tag_is_refused(tmp_path):
    (tmp_path / "VRSBench_train.json").write_text("[]", encoding="utf-8")
    config = BenchmarkConfig.from_dict(
        {
            "name": "vrsbench_train",
            "adapter": "vrsbench_train",
            "task": "vqa",
            "root": str(tmp_path),
            "annotations": "VRSBench_train.json",
            "image_dir": "Images_train",
        }
    )
    with pytest.raises(VRSBenchTrainError, match="task_tag"):
        load_benchmark(config).load()


def test_a_record_with_the_wrong_turn_count_is_reported(tmp_path):
    rows = [
        {
            "id": "x",
            "image": "a.png",
            "conversations": [{"from": "human", "value": "[vqa] hi"}],
        }
    ]
    with pytest.raises(VRSBenchTrainError, match="two-turn"):
        load_benchmark(build(tmp_path, rows, "vqa", "vqa")).load()


def test_untagged_rows_are_skipped_rather_than_guessed(tmp_path):
    rows = [
        conversation("<image>\nDescribe this image", "A harbour."),
        conversation("<image>\n[vqa] How many ships?", "three"),
    ]
    samples = load_benchmark(build(tmp_path, rows, "vqa", "vqa")).load()
    assert len(samples) == 1


def test_grounding_rows_without_a_parsable_box_are_skipped(tmp_path):
    rows = [
        conversation(
            "<image>\n[refer] the location of <p>the ship</p>", "somewhere on the left"
        ),
        conversation(
            "<image>\n[refer] the location of <p>the pier</p>", "{<10><20><30><40>}"
        ),
    ]
    samples = load_benchmark(build(tmp_path, rows, "refer", "grounding")).load()
    assert [s.question for s in samples] == ["the pier"]
