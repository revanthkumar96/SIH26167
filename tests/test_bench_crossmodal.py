"""The BigEarthNet bench split as the cross-modal benchmark.

Optical-SAR joint analysis is mandatory and was the one scored criterion with no
benchmark behind it. These tests cover the two ways wiring it up could go wrong
quietly: image order, which the cross-modal prompt asserts and a swap would
invert, and evidence leaking from the training pipeline into evaluation
questions, which would measure the preamble rather than the model.
"""

from __future__ import annotations

import json

import pytest
from PIL import Image

from satquery.eval.datasets import load_benchmark
from satquery.eval.datasets.base import BenchmarkConfig
from satquery.eval.datasets.bigearthnet import BenchSplitError
from satquery.schema import Modality, Task


def write_split(root, records):
    """A prepared bench directory, in the layout the preparation script writes."""
    images = root / "images"
    images.mkdir(parents=True, exist_ok=True)
    for record in records:
        for name in record["images"]:
            Image.new("RGB", (16, 16), "grey").save(root / name)
    (root / "bench.jsonl").write_text(
        "\n".join(json.dumps(r) for r in records) + "\n", encoding="utf-8"
    )
    return root


def record(index=0, task="binary", answer="yes", evidence_kind="none"):
    return {
        "id": f"ben-{index}",
        "patch_id": f"p{index}",
        "task": task,
        "images": [f"images/p{index}_optical.png", f"images/p{index}_sar.png"],
        "evidence_kind": evidence_kind,
        "conversations": [
            {"from": "human", "value": "Is there water in this scene?"},
            {"from": "gpt", "value": answer},
        ],
    }


def config_for(root, **extra):
    return BenchmarkConfig.from_dict(
        {
            "name": "bigearthnet_bench",
            "adapter": "bigearthnet_bench",
            "task": "crossmodal_vqa",
            "root": str(root),
            "annotations": "bench.jsonl",
            "image_dir": "images",
            "extra": extra or {"types": ["binary", "mcq"]},
        }
    )


def test_shipped_config_is_valid():
    """The committed YAML parses and names a registered adapter."""
    config = BenchmarkConfig.from_yaml("configs/bench/bigearthnet_bench.yaml")
    assert config.adapter == "bigearthnet_bench"
    assert config.task is Task.CROSSMODAL_VQA
    load_benchmark(config)  # constructs without touching disk


def test_loads_pairs_as_optical_then_sar(tmp_path):
    """Order is the contract: the prompt tells the model image 1 is optical."""
    root = write_split(tmp_path / "bench", [record(0)])
    samples = load_benchmark(config_for(root)).load()

    assert len(samples) == 1
    sample = samples[0]
    assert sample.task is Task.CROSSMODAL_VQA
    assert [image.modality for image in sample.images] == [
        Modality.OPTICAL,
        Modality.SAR,
    ]
    assert sample.images[0].path.name.endswith("_optical.png")
    assert sample.images[1].path.name.endswith("_sar.png")
    assert all(image.path.exists() for image in sample.images)
    assert sample.question == "Is there water in this scene?"
    assert sample.answer == "yes"
    assert sample.qtype == "binary"


def test_only_short_form_types_are_kept(tmp_path):
    """Captioning and boxes share the split but are scored differently."""
    root = write_split(
        tmp_path / "bench",
        [
            record(0, task="binary"),
            record(1, task="mcq", answer="c"),
            record(2, task="captioning", answer="A lake."),
            record(3, task="bounding box", answer="[10, 10, 90, 90]"),
        ],
    )
    samples = load_benchmark(config_for(root)).load()
    assert sorted(s.qtype for s in samples) == ["binary", "mcq"]


def test_types_filter_is_configurable(tmp_path):
    root = write_split(
        tmp_path / "bench", [record(0, task="binary"), record(1, task="mcq")]
    )
    samples = load_benchmark(config_for(root, types=["mcq"])).load()
    assert [s.qtype for s in samples] == ["mcq"]


def test_a_baked_in_preamble_is_refused_loudly(tmp_path):
    """Evidence in an evaluation question would measure the preamble, not the model."""
    root = write_split(tmp_path / "bench", [record(0, evidence_kind="decisive")])
    with pytest.raises(BenchSplitError, match="bare"):
        load_benchmark(config_for(root)).load()


def test_preparation_will_not_bake_evidence_into_the_bench_split():
    """The upstream guard: the flag is ignored rather than honoured on bench."""
    source = (__import__("pathlib").Path("scripts/prepare_bigearthnet.py")).read_text(
        encoding="utf-8"
    )
    assert 'preamble_rate = 0.0 if args.split == "bench"' in source


def test_a_record_without_a_pair_is_refused(tmp_path):
    broken = record(0)
    broken["images"] = ["images/p0_optical.png"]
    root = write_split(tmp_path / "bench", [broken])
    with pytest.raises(BenchSplitError, match="exactly two"):
        load_benchmark(config_for(root)).load()


def test_missing_split_names_how_to_build_it(tmp_path):
    with pytest.raises(FileNotFoundError, match=r"prepare_bigearthnet.py"):
        load_benchmark(config_for(tmp_path / "absent")).load()


def test_scores_end_to_end_through_the_harness(tmp_path):
    """A full cell: dataset to metrics, so the config is proven runnable.

    Uses the echo backend, so the numbers are meaningless -- what is being
    checked is that a cross-modal cell produces OA and AA at all, which is the
    criterion that previously had none.
    """
    from satquery.eval.backends import build_backend
    from satquery.eval.runner import run_benchmark

    root = write_split(
        tmp_path / "bench",
        [record(i, task="binary" if i % 2 else "mcq") for i in range(4)],
    )
    with build_backend("echo") as backend:
        result = run_benchmark(
            load_benchmark(config_for(root)), backend, output_dir=tmp_path / "out"
        )

    assert result.task == "crossmodal_vqa"
    assert result.num_samples == 4
    assert "oa" in result.metrics and "aa" in result.metrics
    assert (tmp_path / "out" / "metrics.json").is_file()


def test_the_criterion_is_no_longer_unscored(tmp_path):
    """The point of the whole config: crossmodal stops being a missing criterion."""
    from satquery.eval.aggregate import score_model

    scored = score_model(
        "m",
        [
            {
                "benchmark": "bigearthnet_bench",
                "task": "crossmodal_vqa",
                "metrics": {"oa": 0.42, "aa": 0.38},
                "num_samples": 200,
            }
        ],
    )
    assert "crossmodal" in scored.criteria
    assert "crossmodal" not in scored.missing_criteria
