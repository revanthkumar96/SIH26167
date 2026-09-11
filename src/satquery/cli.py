"""Command line entry point.

    satquery bench adapters
    satquery bench validate --config configs/bench/vrsbench_vqa.yaml
    satquery bench run --config configs/bench/*.yaml --backend vllm \
        --model Qwen/Qwen2.5-VL-3B-Instruct --limit 2000
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import TYPE_CHECKING

from satquery.eval.backends import BACKENDS, BackendConfig, build_backend
from satquery.eval.datasets import BenchmarkConfig, available_adapters, load_benchmark
from satquery.eval.report import append_results, comparison_table, format_result
from satquery.eval.runner import EvalResult, run_benchmark
from satquery.models import DownloadProgress, cached_path, ensure_model

if TYPE_CHECKING:  # imported lazily at runtime to keep CLI startup cheap
    from satquery.data.contamination import Fingerprints

DEFAULT_RESULTS = Path("runs/results.csv")


def _load_configs(
    paths: list[str], limit: int | None, seed: int | None, root: str | None
) -> list[BenchmarkConfig]:
    configs: list[BenchmarkConfig] = []
    for pattern in paths:
        matches = (
            sorted(Path().glob(pattern))
            if any(c in pattern for c in "*?[")
            else [Path(pattern)]
        )
        if not matches:
            raise SystemExit(f"no benchmark config matched: {pattern}")
        for match in matches:
            config = BenchmarkConfig.from_yaml(match)
            if limit is not None:
                config.limit = limit
            if seed is not None:
                config.seed = seed
            if root is not None:
                config.root = Path(root)
            configs.append(config)
    return configs


def cmd_adapters(_: argparse.Namespace) -> int:
    for name in available_adapters():
        print(name)
    return 0


def cmd_validate(args: argparse.Namespace) -> int:
    configs = _load_configs(args.config, args.limit, args.seed, args.root)
    failures = 0
    for config in configs:
        report = load_benchmark(config).describe(probe=args.probe)
        print(json.dumps(report, indent=2, default=str))
        if report.get("error") or report.get("images_missing"):
            failures += 1
    if failures:
        print(
            f"\n{failures}/{len(configs)} benchmark(s) need attention", file=sys.stderr
        )
    return 1 if failures else 0


def cmd_run(args: argparse.Namespace) -> int:
    configs = _load_configs(args.config, args.limit, args.seed, args.root)

    backend_config = BackendConfig(
        model=args.model,
        dtype=args.dtype,
        max_side=args.max_side,
        batch_size=args.batch_size,
        tensor_parallel_size=args.tensor_parallel_size,
        gpu_memory_utilization=args.gpu_memory_utilization,
        max_model_len=args.max_model_len,
    )

    out_root = Path(args.out)
    model_slug = args.model.replace("/", "__")
    results: list[EvalResult] = []

    with build_backend(args.backend, backend_config) as backend:
        for config in configs:
            print(f"\n=== {config.name} :: {args.model} ===", flush=True)
            dataset = load_benchmark(config)
            result = run_benchmark(
                dataset,
                backend,
                output_dir=out_root / model_slug / config.name,
                batch_size=args.batch_size,
            )
            results.append(result)
            print(format_result(result), flush=True)

    append_results(results, Path(args.results))
    print("\n" + comparison_table(results))
    print(f"\nappended {len(results)} run(s) to {args.results}")
    return 0


def cmd_score(args: argparse.Namespace) -> int:
    """Normalise recorded results and aggregate them per criterion.

    Reads the shared results CSV rather than taking numbers on the command line:
    the submission's table has to be derivable from recorded runs, and a score
    typed in by hand is one nobody can reproduce.
    """
    from satquery.eval.aggregate import cells_from_results, delta_table, score_matrix
    from satquery.eval.report import read_results

    rows = read_results(Path(args.results))
    if not rows:
        print(f"no results in {args.results}. Run 'satquery bench run' first.")
        return 1

    cells = cells_from_results(rows)
    if args.model:
        cells = [c for c in cells if c["model"] in set(args.model)]
        if not cells:
            print(f"no recorded runs for {', '.join(args.model)}")
            return 1

    # Ordered so the baseline is the row deltas are taken against, which is the
    # comparison the whole adaptation programme exists to produce.
    if args.baseline:
        cells.sort(key=lambda c: c["model"] != args.baseline)

    scores = score_matrix(cells)
    print(delta_table(scores))

    for score in scores:
        provenance = {
            (c["prompt_version"], c["git_sha"], c["num_samples"])
            for c in cells
            if c["model"] == score.model
        }
        print(f"\n{score.model}")
        for entry in score.benchmarks:
            print(
                f"  {entry.benchmark:<22} {entry.metric:<10} "
                f"raw {entry.raw:8.4f}  normalised {entry.normalised:.4f}  "
                f"n={entry.num_samples}"
            )
        for versions, sha, samples in sorted(provenance):
            print(f"  prompt {versions or '?'}  git {sha or '?'}  n={samples}")
        if score.missing_criteria:
            # Named rather than counted: an unscored mandatory criterion is the
            # gap that costs most, and it should be legible without a lookup.
            print(f"  unscored: {', '.join(score.missing_criteria)}")
        for skipped in score.skipped:
            print(f"  skipped {skipped['benchmark']}: {skipped['reason']}")

    if args.json:
        Path(args.json).parent.mkdir(parents=True, exist_ok=True)
        Path(args.json).write_text(
            json.dumps([s.as_dict() for s in scores], indent=2) + "\n",
            encoding="utf-8",
        )
        print(f"\nwrote {args.json}")
    return 0


def cmd_data_list(args: argparse.Namespace) -> int:
    """Show the prescribed benchmarks and whether they are on disk."""
    from satquery.data import describe_all

    for entry in describe_all(Path(args.data_root)):
        mark = "ready" if entry["ready"] else "missing"
        print(f"{entry['name']:<12} {mark:<8} {entry['title']}")
        print(f"  source   {entry['provenance']}")
        print(f"  home     {entry['homepage']}")
        size = f"{entry['download_mb']:.0f} MB"
        if entry["optional_mb"]:
            size += f"  (+{entry['optional_mb'] / 1024:.1f} GB optional imagery)"
        if entry["shards"]:
            size = (
                f"{entry['shard_size_mb']:.0f} MB per shard, {entry['shards']} shards"
            )
        print(f"  download {size}")
        if entry["note"]:
            print(f"  note     {entry['note']}")
        print()
    return 0


def cmd_data_pull(args: argparse.Namespace) -> int:
    """Download benchmark data from its official source."""
    from satquery.data import DataProgress, pull

    seen: set[int] = set()

    def on_update(progress: DataProgress) -> None:
        percent = progress.percent
        if progress.state != "downloading" or percent is None:
            return
        band = int(percent) // 5
        if band in seen:
            return
        seen.add(band)
        print(
            f"  {percent:5.1f}%  {_human_bytes(progress.downloaded_bytes)}"
            f" / {_human_bytes(progress.total_bytes)}   {progress.current_file}",
            flush=True,
        )

    status = pull(
        args.name,
        Path(args.data_root),
        on_update,
        with_optional=args.with_images,
        shards=args.shards,
    )
    print(f"{status.state}: {status.detail}")
    return 0 if status.state == "ready" else 1


def _fingerprint_test_splits(patterns: list[str], root: str | None) -> Fingerprints:
    """Read the benchmark test splits the corpus must not touch."""
    from satquery.data.contamination import fingerprint_benchmarks

    configs = _load_configs(patterns, limit=None, seed=None, root=root)
    # Every row of every test split counts, not the seeded subset the harness
    # scores: a limit is a sampling decision, and a record that leaks a row
    # outside today's subset still poisons tomorrow's.
    for config in configs:
        config.limit = None
    marks = fingerprint_benchmarks(configs)
    for line in marks.sources:
        print(f"  test split  {line}")

    # A split that could not be read was not checked, and "clean" then means
    # "clean against whatever happened to be on this disk". The BigEarthNet
    # bench split is the one that matters most here: it is rendered from the
    # same LMDB as the training corpus, by the same script, to the same image
    # filenames -- so it is the split a training corpus is most likely to
    # overlap and the one most often absent from a machine.
    skipped = [s for s in marks.sources if "SKIPPED" in s]
    if skipped:
        print(
            f"\n  WARNING: {len(skipped)} split(s) could not be read and were NOT "
            f"checked:\n"
            + "".join(f"    {s}\n" for s in skipped)
            + "  A pass below covers only the splits listed above."
        )
    return marks


def cmd_data_instruct(args: argparse.Namespace) -> int:
    """Convert benchmark train splits into the Stage B adaptation corpus."""
    from satquery.data.instruct import (
        DEFAULT_CAPS,
        DEFAULT_SLICE_DROP_TASKS,
        build_corpus,
        mixture_warnings,
    )

    marks = None
    if not args.no_guard:
        marks = _fingerprint_test_splits(args.test_config, args.test_root)
        if not marks.images:
            raise SystemExit(
                "contamination guard found no benchmark test images. Refusing to "
                "build a corpus that cannot be checked -- pass --test-config "
                "pointing at the test configs, or --no-guard to state explicitly "
                "that you accept an unchecked corpus."
            )

    caps = dict(DEFAULT_CAPS)
    for item in args.cap or []:
        name, _, value = item.partition("=")
        caps[name.strip()] = int(value)

    include: list[tuple[str, str]] = []
    for item in args.include or []:
        name, sep, path = item.partition(chr(61))
        include.append((name, path) if sep else (Path(name).stem, name))

    configs = _load_configs(args.config, limit=None, seed=None, root=args.root)

    # Default the image root to the data directory rather than leaving paths
    # absolute. The corpus is written on one machine and trained on another, and
    # an absolute C:\Users path is not a path on the GPU box.
    image_root = Path(args.image_root) if args.image_root else Path(args.data_root)
    report = build_corpus(
        configs,
        args.out,
        marks=marks,
        image_root=image_root,
        caps=caps,
        seed=args.seed,
        require_images=not args.no_image_check,
        on_contamination="drop" if args.drop_overlap else "raise",
        include=include,
        slice_drop_tasks=(
            frozenset(args.slice_drop_task)
            if args.slice_drop_task is not None
            else DEFAULT_SLICE_DROP_TASKS
        ),
        slice_max_answer_words=args.slice_max_answer_words,
    )
    print(report.render())

    # The two ways a Stage B mixture goes wrong without anything failing.
    # Both were live until the BigEarthNet slice was wired in: no benchmark
    # train split carries an optical-SAR pair, and no RGB benchmark image
    # can support an evidence preamble, so a corpus built from the
    # benchmarks alone trains away both capabilities while every log line
    # looks healthy.
    for warning in mixture_warnings(report):
        print(f"  WARNING: {warning}")
    print(f"\nwrote {args.out}")
    return 0


def cmd_data_check(args: argparse.Namespace) -> int:
    """Check an already-built corpus against the benchmark test splits."""
    from satquery.data.contamination import check_records, read_corpus

    marks = _fingerprint_test_splits(args.test_config, args.test_root)

    # A guard that could not read its splits has not cleared the corpus, and
    # saying so only in a warning is how a run trains on an unchecked corpus
    # while its own log records that nothing was checked. Exit status is the
    # only part of this a shell script reads.
    skipped = [line for line in marks.sources if "SKIPPED" in line]
    if skipped and not args.allow_partial:
        print(
            f"REFUSING: {len(skipped)} of {len(marks.sources)} benchmark "
            f"splits could not be read, so this corpus has not been "
            f"checked against them. Fix the paths, or pass --allow-partial "
            f"to accept a check covering only the splits listed above."
        )
        return 2
    if not marks.images and not args.allow_partial:
        print("REFUSING: no benchmark images were fingerprinted at all.")
        return 2

    report = check_records(read_corpus(args.corpus), marks)
    print(report.summary())
    for overlap in report.fatal_overlaps[:20]:
        print(f"  {overlap.kind:<16} {overlap.record_id}  {overlap.detail}")
    return 0 if report.safe else 1


def _human_bytes(count: int) -> str:
    size = float(count)
    for unit in ("B", "KB", "MB", "GB"):
        if size < 1024:
            return f"{size:.1f} {unit}"
        size /= 1024
    return f"{size:.1f} TB"


def cmd_models_pull(args: argparse.Namespace) -> int:
    """Fetch weights ahead of time -- the offline-demo insurance policy."""
    reported: set[int] = set()

    def on_update(progress: DownloadProgress) -> None:
        percent = progress.percent
        if progress.state != "downloading" or percent is None:
            return
        # Report each 5% band once: a progress bar is noise in a log file, and
        # this command is most often run non-interactively.
        band = int(percent) // 5
        if band in reported:
            return
        reported.add(band)
        print(
            f"  {percent:5.1f}%  "
            f"{_human_bytes(progress.downloaded_bytes)}"
            f" / {_human_bytes(progress.total_bytes)}",
            flush=True,
        )

    print(f"resolving {args.model}")
    status = ensure_model(
        args.model,
        args.revision,
        on_update,
        not args.no_download,
        Path(args.dir) if args.dir else None,
    )
    print(f"{status.state}: {status.detail}")
    if status.path:
        print(f"path: {status.path}")
    return 0 if status.state == "ready" else 1


def cmd_models_status(args: argparse.Namespace) -> int:
    path = cached_path(args.model, args.revision, Path(args.dir) if args.dir else None)
    if path is None:
        print(f"{args.model}: not present locally")
        return 1
    print(f"{args.model}: present")
    print(f"path: {path}")
    return 0


def cmd_serve(args: argparse.Namespace) -> int:
    """Run the web application."""
    import os

    import uvicorn

    # Settings are read from the environment by the app factory, so the CLI sets
    # them rather than threading a config object through uvicorn's reloader.
    os.environ["SATQUERY_BACKEND"] = args.backend
    os.environ["SATQUERY_MODEL"] = args.model
    os.environ["SATQUERY_WORKSPACE"] = args.workspace
    os.environ["SATQUERY_PRELOAD"] = "0" if args.no_preload else "1"
    os.environ["SATQUERY_ALLOW_DOWNLOAD"] = "0" if args.no_download else "1"
    if args.dtype:
        os.environ["SATQUERY_DTYPE"] = args.dtype
    if args.revision:
        os.environ["SATQUERY_REVISION"] = args.revision

    print(f"SatQuery AI on http://{args.host}:{args.port}  (backend: {args.backend})")
    if args.backend != "echo" and not args.no_preload:
        print(f"Model {args.model} is fetched at startup if it is not already present.")
        print("Watch progress at /api/model; the UI shows it in the header.")
    uvicorn.run(
        "satquery.api.app:app",
        host=args.host,
        port=args.port,
        reload=args.reload,
        log_level=args.log_level,
    )
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="satquery", description="SatQuery AI tools")
    sub = parser.add_subparsers(dest="group", required=True)

    bench = sub.add_parser("bench", help="benchmark harness")
    bench_sub = bench.add_subparsers(dest="command", required=True)

    bench_sub.add_parser(
        "adapters", help="list registered dataset adapters"
    ).set_defaults(func=cmd_adapters)

    common = argparse.ArgumentParser(add_help=False)
    common.add_argument(
        "--config",
        nargs="+",
        required=True,
        help="benchmark config YAML path(s) or glob(s)",
    )
    common.add_argument(
        "--limit", type=int, default=None, help="override sample cap (seeded subset)"
    )
    common.add_argument("--seed", type=int, default=None, help="override subset seed")
    common.add_argument("--root", default=None, help="override dataset root")

    validate = bench_sub.add_parser(
        "validate", parents=[common], help="check a download against its config"
    )
    validate.add_argument(
        "--probe", type=int, default=200, help="how many image paths to existence-check"
    )
    validate.set_defaults(func=cmd_validate)

    run = bench_sub.add_parser("run", parents=[common], help="evaluate a model")
    run.add_argument("--backend", choices=BACKENDS, default="echo")
    run.add_argument("--model", default="echo")
    run.add_argument("--dtype", default="auto")
    run.add_argument(
        "--max-side",
        type=int,
        default=1024,
        help="cap the longest image edge; the main throughput lever",
    )
    run.add_argument("--batch-size", type=int, default=32)
    run.add_argument("--tensor-parallel-size", type=int, default=1)
    run.add_argument("--gpu-memory-utilization", type=float, default=0.90)
    run.add_argument("--max-model-len", type=int, default=None)
    run.add_argument("--out", default="runs", help="artefact output root")
    run.add_argument(
        "--results",
        default=str(DEFAULT_RESULTS),
        help="shared results CSV to append to",
    )
    run.set_defaults(func=cmd_run)

    score = bench_sub.add_parser(
        "score",
        help="normalise recorded results and aggregate them per criterion",
    )
    score.add_argument(
        "--results",
        default=str(DEFAULT_RESULTS),
        help="results CSV to aggregate",
    )
    score.add_argument(
        "--model",
        nargs="+",
        default=None,
        help="restrict to these model names (default: every model recorded)",
    )
    score.add_argument(
        "--baseline",
        default=None,
        help="model to take deltas against; sorted to the first row",
    )
    score.add_argument("--json", default=None, help="also write the scores as JSON")
    score.set_defaults(func=cmd_score)

    serve = sub.add_parser("serve", help="run the web application")
    serve.add_argument("--host", default="127.0.0.1")
    serve.add_argument("--port", type=int, default=8000)
    serve.add_argument("--backend", choices=BACKENDS, default="echo")
    serve.add_argument("--model", default="echo")
    serve.add_argument("--dtype", default=None)
    serve.add_argument("--workspace", default="runs")
    serve.add_argument("--revision", default=None, help="pin a model revision")
    serve.add_argument(
        "--no-preload",
        action="store_true",
        help="skip the startup weight fetch; load on the first query instead",
    )
    serve.add_argument(
        "--no-download",
        action="store_true",
        help="fail rather than fetch weights that are not already present",
    )
    serve.add_argument("--reload", action="store_true")
    serve.add_argument("--log-level", default="info")
    serve.set_defaults(func=cmd_serve)

    data = sub.add_parser("data", help="benchmark datasets")
    data_sub = data.add_subparsers(dest="command", required=True)

    data_list = data_sub.add_parser("list", help="show prescribed benchmarks")
    data_list.add_argument("--data-root", default="data")
    data_list.set_defaults(func=cmd_data_list)

    data_pull = data_sub.add_parser("pull", help="download from the official source")
    data_pull.add_argument("name", help="rsvqa_lr | vrsbench | cdvqa")
    data_pull.add_argument("--data-root", default="data")
    data_pull.add_argument(
        "--with-images",
        action="store_true",
        help="also fetch large imagery archives (VRSBench validation is 3.8 GB)",
    )
    data_pull.add_argument(
        "--shards",
        type=int,
        default=2,
        help="CDVQA shards to fetch; each holds 100 samples",
    )
    data_pull.set_defaults(func=cmd_data_pull)

    # The guard defaults matter more than the flags. Both commands read the
    # benchmark test splits by default and both fail on overlap, because the
    # alternative -- a corpus nobody checked -- looks identical right up until
    # the delta it produces is challenged.
    guarded = argparse.ArgumentParser(add_help=False)
    guarded.add_argument(
        "--test-config",
        nargs="+",
        default=["configs/bench/*.yaml"],
        help="benchmark configs whose TEST splits training must not touch",
    )
    guarded.add_argument("--test-root", default=None, help="override test data root")

    instruct = data_sub.add_parser(
        "instruct",
        parents=[guarded],
        help="build the Stage B corpus from benchmark train splits",
    )
    instruct.add_argument(
        "--config",
        nargs="+",
        required=True,
        help="benchmark configs pointed at TRAIN annotations",
    )
    instruct.add_argument("--out", required=True, help="output JSONL")
    instruct.add_argument("--root", default=None, help="override train data root")
    instruct.add_argument(
        "--image-root",
        default=None,
        help="make image paths relative to this (default: --data-root)",
    )
    instruct.add_argument(
        "--data-root",
        default="data",
        help="root the written image paths are relative to when --image-root is unset",
    )
    instruct.add_argument(
        "--cap",
        action="append",
        metavar="NAME=N",
        help="per-source record cap, repeatable (default: DEFAULT_CAPS)",
    )
    instruct.add_argument("--seed", type=int, default=1234)
    instruct.add_argument(
        "--slice-drop-task",
        action="append",
        metavar="TASK",
        help="drop this task from included slices, repeatable. Defaults to "
        "'captioning': the BigEarthNet slice is included for cross-modal "
        "ability, and its 96-word captions teach a length the benchmark "
        "punishes. Pass the flag with no value elsewhere to keep everything",
    )
    instruct.add_argument(
        "--slice-max-answer-words",
        type=int,
        default=None,
        help="drop included-slice records whose target is longer than this",
    )
    instruct.add_argument(
        "--include",
        action="append",
        metavar="NAME=PATH",
        help=(
            "mix in an already-prepared corpus, repeatable. Stage B needs "
            "the BigEarthNet slice: it is the only source of optical-SAR "
            "pairs and of evidence preambles"
        ),
    )
    instruct.add_argument(
        "--drop-overlap",
        action="store_true",
        help="exclude and count rows that overlap the test splits, instead of failing",
    )
    instruct.add_argument(
        "--no-image-check",
        action="store_true",
        help="keep records whose image files are not present on this machine",
    )
    instruct.add_argument(
        "--no-guard",
        action="store_true",
        help="skip the contamination check entirely (the delta becomes indefensible)",
    )
    instruct.set_defaults(func=cmd_data_instruct)

    check = data_sub.add_parser(
        "check-contamination",
        parents=[guarded],
        help="check a built corpus against the benchmark test splits",
    )
    check.add_argument("corpus", help="prepared train.jsonl")
    check.add_argument(
        "--allow-partial",
        action="store_true",
        help="accept a check covering only the splits that could be read",
    )
    check.set_defaults(func=cmd_data_check)

    models = sub.add_parser("models", help="model weights")
    models_sub = models.add_subparsers(dest="command", required=True)

    pull = models_sub.add_parser("pull", help="download weights ahead of time")
    pull.add_argument("model")
    pull.add_argument("--revision", default=None)
    pull.add_argument("--dir", default=None, help="where to put weights")
    pull.add_argument("--no-download", action="store_true")
    pull.set_defaults(func=cmd_models_pull)

    status = models_sub.add_parser("status", help="check whether weights are present")
    status.add_argument("model")
    status.add_argument("--revision", default=None)
    status.add_argument("--dir", default=None, help="where weights are kept")
    status.set_defaults(func=cmd_models_status)

    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())
