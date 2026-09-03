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

from satquery.eval.backends import BACKENDS, BackendConfig, build_backend
from satquery.eval.datasets import BenchmarkConfig, available_adapters, load_benchmark
from satquery.eval.report import append_results, comparison_table, format_result
from satquery.eval.runner import EvalResult, run_benchmark
from satquery.models import DownloadProgress, cached_path, ensure_model

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
