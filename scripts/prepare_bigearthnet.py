#!/usr/bin/env python
"""Ingest BigEarthNet.txt into instruction-tuning data.

Reads the published annotation parquet and the pre-encoded BigEarthNet v2.0
image store, drops the rows that cannot be answered from pixels, renders each
patch the way the serving path renders one, and writes ShareGPT-style JSONL.

    # bring-up on the 2.5 GB slice, no AWS involved
    python scripts/prepare_bigearthnet.py \
        --lmdb  data/BENv2_lithuania_summer.lmdb \
        --out   data/prepared/dev \
        --per-type 500

    # the real pass, against the full 155 GB store
    python scripts/prepare_bigearthnet.py \
        --lmdb data/BENv2.lmdb --out data/prepared/train --per-type 40000

Sources, both ungated and CDLA-Permissive-1.0:
  annotations  BIFOLD-BigEarthNetv2-0/BigEarthNet.txt   (467 MB parquet)
  imagery      hackelle/BigEarthNetV2-LMDB              (155.4 GB)
               hackelle/BigEarthNetV2-Lithuania-Summer-LMDB (2.5 GB slice)
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO = "BIFOLD-BigEarthNetv2-0/BigEarthNet.txt"
PARQUET = "BigEarthNet.txt.parquet"


def fetch_parquet(cache: Path) -> Path:
    """Download the annotation parquet unless it is already present."""
    local = cache / PARQUET
    if local.exists():
        print(f"annotations: {local} (cached)")
        return local
    from huggingface_hub import hf_hub_download

    cache.mkdir(parents=True, exist_ok=True)
    print(f"annotations: downloading {PARQUET} ({REPO}) ...")
    path = hf_hub_download(
        repo_id=REPO, filename=PARQUET, repo_type="dataset", local_dir=str(cache)
    )
    return Path(path)


def render_patches(
    lmdb_path: Path, patch_ids: list[str], s1_of: dict[str, str], out_dir: Path
) -> dict[str, tuple[str, ...]]:
    """Write an optical and a SAR PNG per patch, returning relative paths.

    Both go through the same percentile stretch the serving path uses, so a
    patch seen in training is rendered identically to one seen at inference.
    """
    import lmdb
    from PIL import Image
    from safetensors.numpy import load as safetensor_load

    from satquery.data.bigearthnet import (
        S2_RGB,
        PreparationError,
        sar_composite,
        stack_bands,
        to_rgb8,
    )

    out_dir.mkdir(parents=True, exist_ok=True)
    env = lmdb.open(str(lmdb_path), readonly=True, lock=False, readahead=True)
    written: dict[str, tuple[str, ...]] = {}
    misses = 0

    with env.begin(write=False) as txn:
        for i, patch in enumerate(patch_ids, 1):
            if i % 500 == 0:
                print(f"  rendered {i}/{len(patch_ids)}", flush=True)
            s2_blob = txn.get(patch.encode())
            s1_key = s1_of.get(patch)
            s1_blob = txn.get(s1_key.encode()) if s1_key else None
            if s2_blob is None or s1_blob is None:
                misses += 1
                continue
            try:
                s2 = safetensor_load(bytes(s2_blob))
                s1 = safetensor_load(bytes(s1_blob))
                optical = to_rgb8(stack_bands(s2, S2_RGB))
                bands = stack_bands(s1, ("VV", "VH"))
                sar = to_rgb8(sar_composite(bands[0], bands[1]))
            except (PreparationError, KeyError, ValueError):
                misses += 1
                continue

            o_name, s_name = f"{patch}_optical.png", f"{patch}_sar.png"
            Image.fromarray(optical, mode="RGB").save(out_dir / o_name)
            Image.fromarray(sar, mode="RGB").save(out_dir / s_name)
            written[patch] = (f"images/{o_name}", f"images/{s_name}")

    env.close()
    if misses:
        print(f"  {misses} patches skipped (not in this LMDB slice)")
    return written


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--lmdb", required=True, type=Path, help="BigEarthNet v2.0 LMDB")
    ap.add_argument("--out", required=True, type=Path, help="output directory")
    ap.add_argument(
        "--parquet", type=Path, help="annotation parquet (downloaded if omitted)"
    )
    ap.add_argument("--cache", type=Path, default=Path("data/cache"))
    ap.add_argument(
        "--split", default="train", choices=["train", "validation", "test", "bench"]
    )
    ap.add_argument("--per-type", type=int, default=25_000, help="rows per task type")
    ap.add_argument("--seed", type=int, default=1234)
    args = ap.parse_args()

    import pandas as pd

    from satquery.data.bigearthnet import (
        UNANSWERABLE_CATEGORIES,
        prepare,
        stratified_sample,
        write_jsonl,
    )

    parquet = args.parquet or fetch_parquet(args.cache)
    if not args.lmdb.exists():
        print(f"error: no LMDB at {args.lmdb}", file=sys.stderr)
        return 2

    print(f"reading {parquet} ...")
    frame = pd.read_parquet(parquet)
    total = len(frame)

    # The S1 key for a patch comes from the annotations themselves, so the
    # separate image-metadata parquet is not needed.
    s1_of = dict(zip(frame["patch_id"], frame["s1_name"], strict=False))

    frame = frame[frame["split"] == args.split]
    after_split = len(frame)

    # Drop the unanswerable rows *before* sampling, or the quota is spent on
    # rows that are then discarded and the type balance comes out wrong.
    frame = frame[
        ~frame["category"].astype(str).str.lower().isin(UNANSWERABLE_CATEGORIES)
    ]
    after_clean = len(frame)

    print(
        f"  {total:,} rows -> {after_split:,} in '{args.split}' -> "
        f"{after_clean:,} answerable ({after_split - after_clean:,} dropped)"
    )

    sampled = stratified_sample(frame, per_type=args.per_type, seed=args.seed)
    print(
        f"  sampled {len(sampled):,} rows across "
        f"{sampled['country'].nunique()} countries, {sampled['season'].nunique()} seasons"
    )
    print(sampled["type"].value_counts().to_string())

    patch_ids = sorted(set(sampled["patch_id"]))
    print(f"rendering {len(patch_ids):,} patches ...")
    images = render_patches(args.lmdb, patch_ids, s1_of, args.out / "images")

    count = write_jsonl(prepare(sampled, images), args.out / f"{args.split}.jsonl")
    manifest = {
        "split": args.split,
        "records": count,
        "patches": len(images),
        "per_type": args.per_type,
        "seed": args.seed,
        "annotations": REPO,
        "dropped_categories": sorted(UNANSWERABLE_CATEGORIES),
        "rows_before_clean": after_split,
        "rows_after_clean": after_clean,
    }
    (args.out / "manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )

    print(f"\nwrote {count:,} records to {args.out / (args.split + '.jsonl')}")
    print(f"      {len(images):,} patch pairs to {args.out / 'images'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
