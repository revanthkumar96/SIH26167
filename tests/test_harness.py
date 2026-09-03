"""End-to-end harness checks against synthetic benchmarks and the echo backend.

No GPU, no weights, no downloads -- this is what CI runs.
"""

import json

import pytest

from satquery.eval.backends import build_backend
from satquery.eval.datasets import BenchmarkConfig, load_benchmark
from satquery.eval.prompts import build_prompt, max_new_tokens
from satquery.eval.report import append_results, comparison_table
from satquery.eval.runner import run_benchmark
from satquery.schema import ImageRef, InputConfig, Sample, Task, ToolSpec


def _write(path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


@pytest.fixture
def vqa_config(tmp_path):
    _write(
        tmp_path / "VRSBench_EVAL_vqa.json",
        [
            {
                "image_id": f"img{i}.png",
                "question": "Is there water in this image?",
                "ground_truth": "yes" if i % 2 == 0 else "no",
                "type": "presence",
            }
            for i in range(10)
        ],
    )
    return BenchmarkConfig.from_dict(
        {
            "name": "synthetic_vqa",
            "adapter": "vrsbench_vqa",
            "task": "vqa",
            "root": str(tmp_path),
            "annotations": "VRSBench_EVAL_vqa.json",
            "image_dir": "images",
        }
    )


def test_vqa_adapter_loads_samples(vqa_config):
    samples = load_benchmark(vqa_config).load()
    assert len(samples) == 10
    assert samples[0].task is Task.VQA
    assert samples[0].qtype == "presence"
    assert samples[0].images[0].path.name == "img0.png"


def test_missing_field_error_names_the_config_key(tmp_path):
    _write(tmp_path / "bad.json", [{"image_id": "a.png", "question": "q?"}])
    config = BenchmarkConfig.from_dict(
        {
            "name": "bad",
            "adapter": "vrsbench_vqa",
            "task": "vqa",
            "root": str(tmp_path),
            "annotations": "bad.json",
        }
    )
    with pytest.raises(KeyError, match=r"fields.answer"):
        load_benchmark(config).load()


def test_field_override_rescues_a_renamed_key(tmp_path):
    _write(
        tmp_path / "odd.json",
        [{"picture": "a.png", "query": "q?", "gold": "yes"}],
    )
    config = BenchmarkConfig.from_dict(
        {
            "name": "odd",
            "adapter": "vrsbench_vqa",
            "task": "vqa",
            "root": str(tmp_path),
            "annotations": "odd.json",
            "fields": {"image_id": "picture", "question": "query", "answer": "gold"},
        }
    )
    samples = load_benchmark(config).load()
    assert samples[0].answer == "yes"


def test_subset_is_seeded_and_identical_across_loads(vqa_config):
    vqa_config.limit = 4
    first = [s.sample_id for s in load_benchmark(vqa_config).load_subset()]
    second = [s.sample_id for s in load_benchmark(vqa_config).load_subset()]
    assert first == second
    assert len(first) == 4


def test_cdvqa_pairs_two_images(tmp_path):
    _write(
        tmp_path / "cdvqa_test.json",
        [
            {
                "img_name": "0001",
                "question": "What changed?",
                "answer": "buildings",
                "type": "change_to_what",
            }
        ],
    )
    config = BenchmarkConfig.from_dict(
        {
            "name": "synthetic_cdvqa",
            "adapter": "cdvqa",
            "task": "change_vqa",
            "root": str(tmp_path),
            "annotations": "cdvqa_test.json",
            "image_dir": ".",
            "image_suffix": ".png",
        }
    )
    sample = load_benchmark(config).load()[0]
    assert len(sample.images) == 2
    assert sample.images[0].path.parts[-2:] == ("im1", "0001.png")
    assert sample.images[1].path.parts[-2:] == ("im2", "0001.png")
    assert sample.input_config is InputConfig.BITEMPORAL_PAIR


def test_rsvqa_joins_questions_to_answers(tmp_path):
    _write(
        tmp_path / "q.json",
        {
            "questions": [
                {
                    "id": 1,
                    "img_id": 7,
                    "question": "How many?",
                    "type": "count",
                    "active": True,
                },
                {
                    "id": 2,
                    "img_id": 8,
                    "question": "Dropped?",
                    "type": "count",
                    "active": False,
                },
            ]
        },
    )
    _write(
        tmp_path / "a.json",
        {"answers": [{"id": 1, "question_id": 1, "answer": "5", "active": True}]},
    )
    config = BenchmarkConfig.from_dict(
        {
            "name": "synthetic_rsvqa",
            "adapter": "rsvqa",
            "task": "vqa",
            "root": str(tmp_path),
            "image_dir": "images",
            "image_suffix": ".tif",
            "extra": {"questions": "q.json", "answers": "a.json"},
        }
    )
    samples = load_benchmark(config).load()
    assert len(samples) == 1
    assert samples[0].answer == "5"
    assert samples[0].images[0].path.name == "7.tif"


def test_prompt_includes_question_and_budget():
    sample = Sample(
        sample_id="s1",
        task=Task.VQA,
        images=(ImageRef("a.png"),),
        question="Is there water?",
    )
    assert "Is there water?" in build_prompt(sample)
    assert max_new_tokens(Task.CAPTION) > max_new_tokens(Task.VQA)


def test_prompt_requires_a_question_when_the_task_needs_one():
    sample = Sample(sample_id="s1", task=Task.VQA, images=(ImageRef("a.png"),))
    with pytest.raises(ValueError, match="requires a question"):
        build_prompt(sample)


def test_run_benchmark_end_to_end(vqa_config, tmp_path):
    backend = build_backend("echo")
    result = run_benchmark(
        load_benchmark(vqa_config),
        backend,
        output_dir=tmp_path / "out",
        progress_every=0,
    )

    # Echo always answers "yes"; half the synthetic references are "yes".
    assert result.num_samples == 10
    assert result.metrics["oa"] == pytest.approx(0.5)
    assert (tmp_path / "out" / "metrics.json").exists()

    lines = (
        (tmp_path / "out" / "predictions.jsonl")
        .read_text(encoding="utf-8")
        .strip()
        .splitlines()
    )
    assert len(lines) == 10
    assert json.loads(lines[0])["raw_text"] == "yes"


def test_results_csv_is_long_format(vqa_config, tmp_path):
    result = run_benchmark(
        load_benchmark(vqa_config), build_backend("echo"), progress_every=0
    )
    csv_path = tmp_path / "results.csv"
    append_results([result], csv_path)
    append_results([result], csv_path)

    rows = csv_path.read_text(encoding="utf-8").strip().splitlines()
    assert rows[0].startswith("timestamp,benchmark")
    assert len(rows) == 1 + 2 * len(result.metrics)
    assert "synthetic_vqa" in comparison_table([result])


def test_tool_spec_rejects_disallowed_parameters():
    spec = ToolSpec(
        name="change_mask_cnn",
        version="1.0.0",
        accepts=InputConfig.BITEMPORAL_PAIR,
        tasks=(Task.CHANGE_VQA,),
        allowed_params={"threshold": (0.1, 0.9), "tile": {256, 512}},
    )
    assert spec.validate_params({"threshold": 0.5, "tile": 256}) == []
    assert spec.validate_params({"threshold": 2.0})
    assert spec.validate_params({"tile": 999})
    assert spec.validate_params({"learning_rate": 0.1})
