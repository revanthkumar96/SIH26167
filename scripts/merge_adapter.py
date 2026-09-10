"""Fold a LoRA adapter into its base and write one set of weights.

    python scripts/merge_adapter.py \
        --adapter runs/adapters/stage-b --out runs/merged-b

Exists as a repository script because the Stage A run did this with a file
written by hand on the rented box, which was then destroyed with it. The step is
load-bearing -- the merged model is what gets benchmarked and what gets
published, so "the artefact that was scored" and "the artefact that was
released" are the same object -- and a load-bearing step that lives only in one
shell history is a step nobody can repeat.

The base model and revision come from the adapter's own recorded metadata rather
than from a flag. Merging an adapter into a different revision than it was
trained against produces weights that load, run, and are quietly wrong.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from publish_adapter import load_metadata, merge_adapter


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="merge a LoRA adapter into its base")
    ap.add_argument("--adapter", required=True, type=Path)
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument(
        "--base-model",
        default=None,
        help="override the recorded base; use only to merge an adapter whose "
        "metadata is missing, and say so wherever the result is reported",
    )
    ap.add_argument("--base-revision", default=None)
    args = ap.parse_args(argv)

    if not (args.adapter / "adapter_model.safetensors").is_file():
        ap.error(f"no adapter weights at {args.adapter}")

    try:
        meta = load_metadata(args.adapter)
    except FileNotFoundError:
        meta = {}

    if args.base_model:
        meta["base_model"] = args.base_model
    if args.base_revision:
        meta["base_revision"] = args.base_revision

    missing = [k for k in ("base_model", "base_revision") if not meta.get(k)]
    if missing:
        ap.error(
            f"the adapter records no {', '.join(missing)}, and guessing it would "
            f"produce weights that load and are wrong. Pass --base-model / "
            f"--base-revision explicitly."
        )

    print(f"merging {args.adapter} -> {args.out}")
    merge_adapter(args.adapter, meta, args.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
