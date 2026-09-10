"""Benchmark train splits -> Stage B corpus.

Two things are being protected. The first is train/serve parity: a record's
human turn has to be the string ``build_prompt`` produces, because that is the
string the harness and the serving path produce, and a converter that phrases
prompts its own way teaches the model a shape it will never meet again. The
second is the grounding grid -- gold boxes written as unit floats would train the
model to emit precisely the form the scorer rejects, and the whole grounding
column would read as a localisation failure instead of a units bug.

Fixtures are written in the real VRSBench and CDVQA on-disk shapes and loaded
through the real adapters, so a release-key change breaks these tests rather
than a run.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from satquery.data.contamination import (
    ContaminationError,
    Fingerprints,
    fingerprint_samples,
)
from satquery.data.evidence import Measurements, apply_preamble
from satquery.data.instruct import (
    DEFAULT_CAPS,
    MIN_EVIDENCE_SHARE,
    build_corpus,
    convert_source,
    format_box_answer,
    is_crossmodal,
    load_slice,
    mixture_of,
    mixture_warnings,
    no_measurements,
    sample_to_record,
    slice_image_prefix,
)
from satquery.eval.datasets import BenchmarkConfig, load_benchmark
from satquery.eval.prompts import build_prompt
from satquery.schema import ImageRef, Sample, Task


def write_vqa(root, rows, name="train_vqa.json", images="Images_train"):
    root.mkdir(parents=True, exist_ok=True)
    (root / name).write_text(json.dumps(rows), encoding="utf-8")
    image_dir = root / images
    image_dir.mkdir(parents=True, exist_ok=True)
    for row in rows:
        (image_dir / row["image_id"]).write_bytes(b"not really a png")
    return BenchmarkConfig.from_dict(
        {
            "name": "vrsbench_vqa",
            "adapter": "vrsbench_vqa",
            "task": "vqa",
            "root": str(root),
            "annotations": name,
            "image_dir": images,
        }
    )


def vqa_row(image_id, question="How many buildings are visible?", answer="three"):
    return {"image_id": image_id, "question": question, "ground_truth": answer}


# -- parity --------------------------------------------------------------


def test_prompt_is_the_harness_prompt_verbatim():
    """The contract. Anything else is train/serve skew in the prompt channel."""
    sample = Sample(
        sample_id="s-1",
        task=Task.VQA,
        images=(ImageRef(path="/img/P0001.png"),),
        question="How many buildings are visible?",
        answer="three",
    )
    record = sample_to_record(sample, "vrsbench_vqa")
    assert record.prompt == build_prompt(sample)
    assert "Answer the question using a single word" in record.prompt


def test_rendered_turn_joins_the_preamble_the_way_serving_does():
    sample = Sample(
        sample_id="s-1",
        task=Task.VQA,
        images=(ImageRef(path="/img/P0001.png"),),
        question="Is there water in this scene?",
        answer="yes",
    )

    def measure(_):
        return Measurements(optical_water_fraction=0.31, sar_water_fraction=0.29)

    record = sample_to_record(sample, "vrsbench_vqa", measure=measure, rate=1.0)
    assert record.preamble
    assert record.rendered_prompt == apply_preamble(record.prompt, record.preamble)
    assert (
        json.loads(record.as_jsonl())["conversations"][0]["value"]
        == record.rendered_prompt
    )


def test_no_measurements_means_no_preamble_rather_than_an_invented_one():
    """RGB benchmark images cannot support NDWI or NDBI. Saying nothing is correct."""
    sample = Sample(
        sample_id="s-1",
        task=Task.VQA,
        images=(ImageRef(path="/img/P0001.png"),),
        question="Is there water?",
        answer="yes",
    )
    record = sample_to_record(sample, "vrsbench_vqa", measure=no_measurements, rate=1.0)
    assert record.preamble == ""
    assert record.evidence_kind == "none"
    assert record.rendered_prompt == record.prompt


# -- grounding units -----------------------------------------------------


def test_grounding_answer_is_on_the_0_1000_grid():
    assert format_box_answer((0.1, 0.25, 0.5, 0.75)) == "[100, 250, 500, 750]"


def test_grounding_answer_clamps_out_of_range_coordinates():
    assert format_box_answer((-0.2, 0.0, 1.4, 1.0)) == "[0, 0, 1000, 1000]"


def test_grounding_record_answer_matches_what_the_prompt_asks_for():
    sample = Sample(
        sample_id="s-1",
        task=Task.GROUNDING,
        images=(ImageRef(path="/img/P0001.png"),),
        question="the ship at the pier",
        bbox=(0.2, 0.3, 0.4, 0.5),
    )
    record = sample_to_record(sample, "vrsbench_referring")
    assert "0-1000 grid" in record.prompt
    assert record.answer == "[200, 300, 400, 500]"


def test_grounding_sample_without_a_box_is_dropped():
    sample = Sample(
        sample_id="s-1",
        task=Task.GROUNDING,
        images=(ImageRef(path="/img/P0001.png"),),
        question="the ship at the pier",
    )
    assert sample_to_record(sample, "vrsbench_referring") is None


def test_blank_answers_are_dropped_rather_than_trained_on():
    """Stage A showed what training toward silence costs. Never emit an empty target."""
    sample = Sample(
        sample_id="s-1",
        task=Task.VQA,
        images=(ImageRef(path="/img/P0001.png"),),
        question="How many?",
        answer="   ",
    )
    assert sample_to_record(sample, "vrsbench_vqa") is None


# -- image paths ---------------------------------------------------------


def test_image_paths_are_relative_to_the_staging_root(tmp_path):
    sample = Sample(
        sample_id="s-1",
        task=Task.VQA,
        images=(ImageRef(path=tmp_path / "VRSBench" / "Images_train" / "P0001.png"),),
        question="How many?",
        answer="three",
    )
    record = sample_to_record(sample, "vrsbench_vqa", image_root=tmp_path)
    assert record.images == ("VRSBench/Images_train/P0001.png",)


def test_a_path_outside_the_root_is_kept_absolute_not_silently_mangled(tmp_path):
    sample = Sample(
        sample_id="s-1",
        task=Task.VQA,
        images=(ImageRef(path="/elsewhere/P0001.png"),),
        question="How many?",
        answer="three",
    )
    record = sample_to_record(sample, "vrsbench_vqa", image_root=tmp_path)
    assert record.images == ("/elsewhere/P0001.png",)


# -- contamination, wired into the converter -----------------------------


def test_conversion_refuses_a_split_that_overlaps_the_test_set(tmp_path):
    """The failure this whole pair of modules exists to prevent."""
    config = write_vqa(tmp_path / "VRSBench", [vqa_row("P0001.png")])
    marks = fingerprint_samples(load_benchmark(config).load())
    with pytest.raises(ContaminationError, match="not safe to train on"):
        convert_source(config, marks=marks)


def test_pointing_at_the_test_annotations_by_mistake_is_caught(tmp_path):
    """The realistic version: the config edit that was never made."""
    root = tmp_path / "VRSBench"
    test_config = write_vqa(
        root, [vqa_row("P0001.png"), vqa_row("P0002.png")], name="test.json"
    )
    marks = fingerprint_samples(load_benchmark(test_config).load())
    # "train" config that still names the test JSON.
    train_config = BenchmarkConfig.from_dict(
        {
            "name": "vrsbench_vqa",
            "adapter": "vrsbench_vqa",
            "task": "vqa",
            "root": str(root),
            "annotations": "test.json",
            "image_dir": "Images_train",
        }
    )
    with pytest.raises(ContaminationError):
        convert_source(train_config, marks=marks)


def test_drop_mode_excludes_the_overlap_and_counts_it(tmp_path):
    root = tmp_path / "VRSBench"
    config = write_vqa(
        root, [vqa_row("P0001.png"), vqa_row("P0002.png", "What colour?", "blue")]
    )
    marks = fingerprint_samples(
        [
            Sample(
                sample_id="t-1",
                task=Task.VQA,
                images=(ImageRef(path="/anywhere/P0001.png"),),
                question="irrelevant",
                answer="irrelevant",
            )
        ]
    )
    records, stats = convert_source(config, marks=marks, on_contamination="drop")
    assert stats.dropped_contaminated == 1
    assert [r.images[0].split("/")[-1] for r in records] == ["P0002.png"]


def test_there_is_no_mode_that_keeps_contaminated_rows(tmp_path):
    config = write_vqa(tmp_path / "VRSBench", [vqa_row("P0001.png")])
    with pytest.raises(ValueError, match=r"raise.*drop"):
        convert_source(config, on_contamination="ignore")


def test_no_fingerprints_means_no_check_not_a_silent_pass(tmp_path):
    """Passing marks=None is legitimate, but must not report a clean check."""
    config = write_vqa(tmp_path / "VRSBench", [vqa_row("P0001.png")])
    _, stats = convert_source(config, marks=None)
    assert stats.dropped_contaminated == 0
    assert stats.written == 1


# -- mixture hygiene -----------------------------------------------------


def test_identical_rows_are_kept_once(tmp_path):
    """Templated corpora repeat themselves; duplicates reweight the mixture."""
    config = write_vqa(
        tmp_path / "VRSBench", [vqa_row("P0001.png"), vqa_row("P0001.png")]
    )
    records, stats = convert_source(config)
    assert stats.dropped_duplicate == 1
    assert len(records) == 1


def test_the_same_question_on_a_different_image_is_not_a_duplicate(tmp_path):
    config = write_vqa(
        tmp_path / "VRSBench", [vqa_row("P0001.png"), vqa_row("P0002.png")]
    )
    records, stats = convert_source(config)
    assert stats.dropped_duplicate == 0
    assert len(records) == 2


def test_cap_is_applied_and_is_reproducible(tmp_path):
    rows = [vqa_row(f"P{i:04d}.png", f"Question {i}?") for i in range(20)]
    config = write_vqa(tmp_path / "VRSBench", rows)
    first, stats = convert_source(config, cap=5)
    again, _ = convert_source(config, cap=5)
    assert stats.written == 5 and stats.dropped_capped == 15
    assert [r.sample_id for r in first] == [r.sample_id for r in again]


def test_default_caps_cover_every_train_config():
    """A source with no cap silently dominates the mixture."""
    import glob
    from pathlib import Path

    import yaml

    found = glob.glob("configs/train/*.yaml")
    assert found, "no train configs on disk"
    for path in found:
        data = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
        name = data.get("name")
        assert name in DEFAULT_CAPS, f"{name} has no cap in DEFAULT_CAPS"


def test_train_configs_never_name_a_test_annotation():
    """The one-character mistake the whole guard exists for, caught statically."""
    import glob
    from pathlib import Path

    import yaml

    for path in glob.glob("configs/train/*.yaml"):
        data = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
        blob = " ".join(
            str(v)
            for v in (
                data.get("annotations", ""),
                data.get("root", ""),
                data.get("extra", {}),
            )
        ).lower()
        for forbidden in ("_test_", "eval", "_val_", "test.json"):
            assert forbidden not in blob, f"{path} references {forbidden!r}: {blob}"


def test_train_and_bench_configs_do_not_share_a_name():
    """Distinct names keep results, caps and reports unambiguous."""
    import glob
    from pathlib import Path

    import yaml

    def names(pattern):
        return {
            (yaml.safe_load(Path(p).read_text(encoding="utf-8")) or {}).get("name")
            for p in glob.glob(pattern)
        }

    assert not names("configs/train/*.yaml") & names("configs/bench/*.yaml")


def test_missing_image_files_are_dropped_when_required(tmp_path):
    root = tmp_path / "VRSBench"
    config = write_vqa(
        root, [vqa_row("P0001.png"), vqa_row("P0002.png", "What colour?", "blue")]
    )
    (root / "Images_train" / "P0002.png").unlink()
    _, stats = convert_source(config)
    assert stats.dropped_missing_image == 1
    assert stats.written == 1


# -- writing the corpus --------------------------------------------------


def test_build_corpus_writes_readable_jsonl(tmp_path):
    config = write_vqa(
        tmp_path / "VRSBench", [vqa_row(f"P{i:04d}.png", f"Q{i}?") for i in range(6)]
    )
    out = tmp_path / "stage-b" / "train.jsonl"
    report = build_corpus([config], out, caps={})
    assert out.exists()
    rows = [
        json.loads(x) for x in out.read_text(encoding="utf-8").splitlines() if x.strip()
    ]
    assert len(rows) == report.written == 6
    assert mixture_of(out) == {"vqa": 6}
    assert all(r["conversations"][1]["value"] for r in rows)


def test_build_corpus_shuffle_is_seeded(tmp_path):
    config = write_vqa(
        tmp_path / "VRSBench", [vqa_row(f"P{i:04d}.png", f"Q{i}?") for i in range(30)]
    )
    first = tmp_path / "a.jsonl"
    second = tmp_path / "b.jsonl"
    build_corpus([config], first, caps={})
    build_corpus([config], second, caps={})
    assert first.read_text(encoding="utf-8") == second.read_text(encoding="utf-8")


def test_report_renders_per_source_accounting(tmp_path):
    config = write_vqa(
        tmp_path / "VRSBench", [vqa_row("P0001.png"), vqa_row("P0001.png")]
    )
    report = build_corpus([config], tmp_path / "out.jsonl", caps={})
    rendered = report.render()
    assert "vrsbench_vqa" in rendered
    assert "dup=1" in rendered
    assert "total written=1" in rendered


def test_record_ids_are_namespaced_by_source(tmp_path):
    """Two benchmarks both start numbering at zero; unqualified ids would collide."""
    config = write_vqa(tmp_path / "VRSBench", [vqa_row("P0001.png")])
    records, _ = convert_source(config)
    assert records[0].sample_id.startswith("vrsbench_vqa-")


# -- the mixture: what the benchmark train splits cannot supply --------------
#
# Stage A earned +0.1750 on optical-SAR and 38% preamble coverage from
# BigEarthNet. No benchmark train split contains a cross-modal pair, and no RGB
# benchmark image can support NDWI or NDBI, so a Stage B corpus built from the
# three benchmarks alone trains both capabilities away -- silently, because the
# loss falls and the sweep succeeds either way.


def prepared_row(record_id, image="S2A_MSIL2A_26_57", evidence="decisive"):
    """A row in the shape scripts/prepare_bigearthnet.py already writes."""
    preamble = "Measured water fraction 0.31."
    return {
        "id": record_id,
        "patch_id": image,
        "task": "crossmodal_vqa",
        "images": [f"images/{image}_optical.png", f"images/{image}_sar.png"],
        "evidence_kind": evidence,
        "conversations": [
            {
                "from": "human",
                "value": preamble + "\n\n" + f"Is {record_id} wet?",
            },
            {"from": "gpt", "value": "yes"},
        ],
    }


def write_prepared(tmp_path, rows, name="bigearthnet.jsonl"):
    path = tmp_path / name
    body = "".join(json.dumps(r) + "\n" for r in rows)
    path.write_text(body, encoding="utf-8")
    return path


def test_a_prepared_slice_is_taken_as_written(tmp_path):
    """The preamble is already applied; re-rendering would apply it twice."""
    path = write_prepared(tmp_path, [prepared_row("r-1")])
    rows, stats = load_slice(path, "bigearthnet")
    assert stats.written == 1
    assert rows[0]["conversations"][0]["value"].startswith("Measured water fraction")
    assert stats.with_evidence == 1


def test_a_prepared_slice_is_capped_reproducibly(tmp_path):
    path = write_prepared(
        tmp_path, [prepared_row(f"r-{i}", f"patch{i}") for i in range(20)]
    )
    first, stats = load_slice(path, "bigearthnet", cap=5)
    again, _ = load_slice(path, "bigearthnet", cap=5)
    assert stats.written == 5 and stats.dropped_capped == 15
    assert [r["id"] for r in first] == [r["id"] for r in again]


def test_a_prepared_slice_is_contamination_checked_like_anything_else(tmp_path):
    """This is the check nobody had run. bigearthnet_bench is rendered from the
    same LMDB by the same script to the same filenames."""
    path = write_prepared(tmp_path, [prepared_row("r-1", "S2A_shared")])
    marks = Fingerprints(images={"s2a_shared_optical.png"})
    with pytest.raises(ContaminationError, match="not safe to train on"):
        load_slice(path, "bigearthnet", marks=marks)


def test_a_prepared_slice_can_drop_its_overlap_instead(tmp_path):
    path = write_prepared(
        tmp_path, [prepared_row("r-1", "S2A_shared"), prepared_row("r-2", "S2A_clean")]
    )
    marks = Fingerprints(images={"s2a_shared_optical.png"})
    rows, stats = load_slice(path, "bigearthnet", marks=marks, on_contamination="drop")
    assert stats.dropped_contaminated == 1
    assert [r["id"] for r in rows] == ["r-2"]


def test_prepared_rows_with_no_answer_are_dropped(tmp_path):
    bad = prepared_row("r-1")
    bad["conversations"][1]["value"] = "   "
    path = write_prepared(tmp_path, [bad, prepared_row("r-2", "patch2")])
    _, stats = load_slice(path, "bigearthnet")
    assert stats.dropped_empty == 1 and stats.written == 1


def test_the_slice_reaches_the_written_corpus(tmp_path):
    config = write_vqa(tmp_path / "VRSBench", [vqa_row("P0001.png")])
    path = write_prepared(tmp_path, [prepared_row("be-1")])
    out = tmp_path / "mixed.jsonl"
    report = build_corpus([config], out, caps={}, include=[("bigearthnet", path)])
    assert report.written == 2
    assert mixture_of(out) == {"vqa": 1, "crossmodal_vqa": 1}
    ids = {json.loads(x)["id"] for x in out.read_text(encoding="utf-8").splitlines()}
    assert "be-1" in ids


def test_a_benchmark_only_mixture_is_warned_about(tmp_path):
    """The gap this whole path exists to close."""
    config = write_vqa(tmp_path / "VRSBench", [vqa_row("P0001.png")])
    report = build_corpus([config], tmp_path / "out.jsonl", caps={})
    warnings = mixture_warnings(report)
    assert any("optical-SAR" in w for w in warnings)
    assert any("evidence preamble" in w for w in warnings)


def test_a_mixture_with_the_slice_is_not_warned_about(tmp_path):
    config = write_vqa(tmp_path / "VRSBench", [vqa_row("P0001.png")])
    path = write_prepared(
        tmp_path, [prepared_row(f"be-{i}", f"p{i}") for i in range(9)]
    )
    report = build_corpus(
        [config], tmp_path / "out.jsonl", caps={}, include=[("bigearthnet", path)]
    )
    assert mixture_warnings(report) == []


def test_the_evidence_warning_uses_the_documented_threshold(tmp_path):
    """Matching scripts/train_lora.py, which warns at the same 20%."""
    config = write_vqa(
        tmp_path / "VRSBench", [vqa_row(f"P{i:04d}.png", f"Q{i}?") for i in range(10)]
    )
    path = write_prepared(tmp_path, [prepared_row("be-1")])  # 1 of 11 = 9.1%
    report = build_corpus(
        [config], tmp_path / "out.jsonl", caps={}, include=[("bigearthnet", path)]
    )
    assert MIN_EVIDENCE_SHARE == 0.20
    warnings = mixture_warnings(report)
    assert not any("optical-SAR" in w for w in warnings)
    assert any("9.1%" in w for w in warnings)


def test_an_empty_corpus_is_reported_as_such(tmp_path):
    config = write_vqa(tmp_path / "VRSBench", [])
    report = build_corpus([config], tmp_path / "out.jsonl", caps={})
    assert mixture_warnings(report) == ["the corpus is empty"]


def real_bigearthnet_row():
    """The record shape actually on disk, copied from the prepared corpus.

    `task` is the *annotation type*, not a schema Task -- the bench adapter is
    what maps it to CROSSMODAL_VQA at load. A cross-modal check that reads the
    task field therefore sees "binary" and concludes the corpus has no
    optical-SAR records in it, which is exactly backwards.
    """
    patch = "S2B_MSIL2A_20170923T100019_N9999_R122_T33TWM_38_10"
    return {
        "id": "5293548",
        "patch_id": patch,
        "task": "binary",
        "images": [f"images/{patch}_optical.png", f"images/{patch}_sar.png"],
        "evidence_kind": "none",
        "conversations": [
            {
                "from": "human",
                "value": "Can you observe any complex cultivation patterns?",
            },
            {"from": "gpt", "value": "no"},
        ],
    }


def test_crossmodal_is_detected_on_the_real_record_shape():
    assert is_crossmodal(real_bigearthnet_row())


def test_crossmodal_is_not_inferred_from_the_task_field_alone():
    """The trap: the prepared corpus never says "crossmodal_vqa" anywhere."""
    row = real_bigearthnet_row()
    assert row["task"] != "crossmodal_vqa"
    assert is_crossmodal(row)


def test_a_single_image_record_is_not_crossmodal():
    assert not is_crossmodal({"task": "vqa", "images": ["Images_train/P0001.png"]})


def test_a_bitemporal_pair_is_not_crossmodal():
    """CDVQA also ships two images; two images is not the signal, SAR is."""
    assert not is_crossmodal(
        {"task": "change_vqa", "images": ["im1/07197.png", "im2/07197.png"]}
    )


def test_converter_produced_crossmodal_records_are_recognised():
    assert is_crossmodal({"task": "crossmodal_vqa", "images": ["a.png", "b.png"]})


def test_the_real_shape_clears_the_crossmodal_warning(tmp_path):
    """End to end on the shape from S3, not the fixture shape."""
    config = write_vqa(tmp_path / "VRSBench", [vqa_row("P0001.png")])
    path = tmp_path / "be.jsonl"
    rows = []
    for i in range(9):
        row = real_bigearthnet_row()
        row["id"] = f"be-{i}"
        row["patch_id"] = f"patch{i}"
        row["images"] = [f"images/patch{i}_optical.png", f"images/patch{i}_sar.png"]
        row["evidence_kind"] = "decisive"
        rows.append(row)
    path.write_text("".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")

    report = build_corpus(
        [config], tmp_path / "out.jsonl", caps={}, include=[("bigearthnet", path)]
    )
    assert report.crossmodal == 9
    assert not any("optical-SAR" in w for w in mixture_warnings(report))


# -- one root for the whole corpus ----------------------------------------
#
# A prepared slice records its images relative to itself; converted benchmark
# records are written relative to the data root. Mixed unadjusted, the corpus
# carries two roots and a third of it is unfindable -- which shows up as a
# file-not-found thousands of training steps in, not at build time.


def test_prefix_is_the_slice_location_relative_to_the_data_root(tmp_path):
    corpus = tmp_path / "prepared" / "train" / "train.jsonl"
    corpus.parent.mkdir(parents=True)
    corpus.touch()
    assert (
        slice_image_prefix(corpus, tmp_path).replace(chr(92), "/") == "prepared/train"
    )


def test_a_slice_beside_the_data_root_needs_no_prefix(tmp_path):
    corpus = tmp_path / "train.jsonl"
    corpus.touch()
    assert slice_image_prefix(corpus, tmp_path) == "."


def test_a_slice_outside_the_data_root_gets_no_prefix(tmp_path):
    corpus = tmp_path / "train.jsonl"
    corpus.touch()
    assert slice_image_prefix(corpus, tmp_path / "elsewhere") == ""


def test_no_image_root_means_no_prefix(tmp_path):
    assert slice_image_prefix(tmp_path / "train.jsonl", None) == ""


def test_mixed_corpus_resolves_every_image_from_one_root(tmp_path):
    """The bug this guards: BigEarthNet rows said images/x.png while VRSBench
    rows said Images_train/y.png, and only one of them was right."""
    root = tmp_path / "data"
    config = write_vqa(root / "VRSBench_train", [vqa_row("P0001.png")])
    slice_dir = root / "prepared" / "train"
    slice_dir.mkdir(parents=True)
    row = real_bigearthnet_row()
    (slice_dir / "train.jsonl").write_text(json.dumps(row) + "\n", encoding="utf-8")

    out = tmp_path / "mixed.jsonl"
    build_corpus(
        [config],
        out,
        caps={},
        image_root=root,
        include=[("bigearthnet", slice_dir / "train.jsonl")],
    )
    rows = [json.loads(x) for x in out.read_text(encoding="utf-8").splitlines()]
    for record in rows:
        for image in record["images"]:
            assert not Path(image).is_absolute()
            # Every path must be meaningful from the single declared root.
            assert image.split("/")[0] in {"VRSBench_train", "prepared"}, image

    be = next(r for r in rows if r["task"] == "binary")
    assert be["images"][0].startswith("prepared/train/images/")
