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
from collections import Counter
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    import pandas as pd

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
    lmdb_path: Path,
    patch_ids: list[str],
    s1_of: dict[str, str],
    out_dir: Path,
    labels_of: dict[str, list[str]] | None = None,
) -> tuple[dict[str, tuple[str, ...]], dict[str, Any]]:
    """Write an optical and a SAR PNG per patch, and measure each one.

    Both renderings go through the same percentile stretch the serving path
    uses, so a patch seen in training is rendered identically to one seen at
    inference.

    The measurements are taken here rather than in a second pass because the
    bands are already in memory, and because the SAR numbers are read back off
    the PNG that was just written -- the same file, through the same ``to_gray``
    the tool calls. A training preamble is then not merely shaped like an
    inference one, it is the one this patch actually produces.
    """
    import lmdb
    from PIL import Image
    from safetensors.numpy import load as safetensor_load

    from satquery.data.bigearthnet import (
        S2_BANDS,
        S2_RGB,
        PreparationError,
        sar_composite,
        stack_bands,
        to_rgb8,
    )
    from satquery.data.evidence import (
        Measurements,
        describe_classes,
        measure_optical,
        measure_sar,
    )

    out_dir.mkdir(parents=True, exist_ok=True)
    env = lmdb.open(str(lmdb_path), readonly=True, lock=False, readahead=True)
    written: dict[str, tuple[str, ...]] = {}
    measured: dict[str, Any] = {}
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
                # The full 12-band stack, not the RGB composite: NDWI needs NIR
                # and NDBI needs SWIR, and neither survives true colour.
                full = stack_bands(s2, S2_BANDS)
            except (PreparationError, KeyError, ValueError):
                misses += 1
                continue

            o_name, s_name = f"{patch}_optical.png", f"{patch}_sar.png"
            Image.fromarray(optical, mode="RGB").save(out_dir / o_name)
            Image.fromarray(sar, mode="RGB").save(out_dir / s_name)
            written[patch] = (f"images/{o_name}", f"images/{s_name}")

            values = measure_optical(full)
            values.update(measure_sar(out_dir / s_name))
            measured[patch] = Measurements(
                optical_water_fraction=values.get("optical_water_fraction"),
                optical_builtup_fraction=values.get("optical_builtup_fraction"),
                sar_water_fraction=values.get("sar_water_fraction"),
                sar_builtup_fraction=values.get("sar_builtup_fraction"),
                landcover_classes=describe_classes((labels_of or {}).get(patch, [])),
            )

    env.close()
    if misses:
        print(f"  {misses} patches skipped (not in this LMDB slice)")
    return written, measured


def read_patch_metadata(path: Path) -> pd.DataFrame | None:
    """The v2.0 image metadata shipped alongside the LMDB, if it is there.

    Carries the CORINE multi-label classes and the cloud/snow flags. Optional
    rather than required: without it the run still produces a corpus, just one
    that is not filtered for cloud and carries no land-cover evidence.
    """
    import pandas as pd

    if not path.is_file():
        print(f"  no metadata.parquet at {path}; cloud/snow filtering skipped")
        return None
    return pd.read_parquet(path)


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
    ap.add_argument(
        "--metadata",
        type=Path,
        default=None,
        help="metadata.parquet shipped with the LMDB (defaults beside it)",
    )
    ap.add_argument(
        "--preamble-rate",
        type=float,
        default=None,
        help="share of records carrying an evidence preamble (default 0.45)",
    )
    ap.add_argument(
        "--keep-cloudy",
        action="store_true",
        help="do not drop cloud/shadow and snow patches; for diagnosis only",
    )
    args = ap.parse_args()

    import pandas as pd

    from satquery.data.bigearthnet import (
        UNANSWERABLE_CATEGORIES,
        as_label_list,
        prepare,
        stratified_sample,
        write_jsonl,
    )
    from satquery.eval.prompts import PROMPT_VERSION

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

    # Cloud, shadow and snow come free from the v2.0 image metadata. A patch
    # under cloud teaches nothing about land cover, and one under snow teaches
    # the wrong thing about it.
    metadata = read_patch_metadata(
        args.metadata or args.lmdb.parent / "metadata.parquet"
    )
    labels_of: dict[str, list[str]] = {}
    excluded: set[str] = set()
    if metadata is not None:
        if "labels" in metadata.columns:
            labels_of = {
                str(row["patch_id"]): as_label_list(row["labels"])
                for _, row in metadata[["patch_id", "labels"]].iterrows()
            }
        flags = [
            column
            for column in ("contains_cloud_or_shadow", "contains_seasonal_snow")
            if column in metadata.columns
        ]
        if flags and not args.keep_cloudy:
            mask = metadata[flags].fillna(False).any(axis=1)
            excluded = set(metadata.loc[mask, "patch_id"].astype(str))
            frame = frame[~frame["patch_id"].astype(str).isin(excluded)]
    after_quality = len(frame)

    # Drop the unanswerable rows *before* sampling, or the quota is spent on
    # rows that are then discarded and the type balance comes out wrong.
    frame = frame[
        ~frame["category"].astype(str).str.lower().isin(UNANSWERABLE_CATEGORIES)
    ]
    after_clean = len(frame)

    print(
        f"  {total:,} rows -> {after_split:,} in '{args.split}' -> "
        f"{after_quality:,} after cloud/snow "
        f"({after_split - after_quality:,} dropped) -> "
        f"{after_clean:,} answerable ({after_quality - after_clean:,} dropped)"
    )

    sampled = stratified_sample(frame, per_type=args.per_type, seed=args.seed)
    print(
        f"  sampled {len(sampled):,} rows across "
        f"{sampled['country'].nunique()} countries, {sampled['season'].nunique()} seasons"
    )
    print(sampled["type"].value_counts().to_string())

    patch_ids = sorted(set(sampled["patch_id"]))
    print(f"rendering and measuring {len(patch_ids):,} patches ...")
    images, measurements = render_patches(
        args.lmdb, patch_ids, s1_of, args.out / "images", labels_of
    )

    # The bench split is evaluation data, and a benchmark question with the
    # answer's evidence baked into it measures something other than the model.
    # Forced rather than left to the operator: it is one flag away from
    # silently inflating the cross-modal score we are about to report.
    preamble_rate = 0.0 if args.split == "bench" else args.preamble_rate
    if args.split == "bench" and args.preamble_rate:
        print("  bench split: --preamble-rate ignored, evaluation data stays bare")

    records = list(
        prepare(
            sampled,
            images,
            measurements=measurements,
            seed=args.seed,
            preamble_rate=preamble_rate,
        )
    )
    count = write_jsonl(iter(records), args.out / f"{args.split}.jsonl")

    mix = Counter(record.evidence_kind for record in records)
    with_evidence = count - mix.get("none", 0)
    print(
        f"  evidence preamble on {with_evidence:,}/{count:,} records "
        f"({(with_evidence / count if count else 0):.1%}): "
        f"{mix.get('decisive', 0):,} decisive, "
        f"{mix.get('orthogonal', 0):,} orthogonal"
    )

    manifest = {
        "split": args.split,
        "records": count,
        "patches": len(images),
        "per_type": args.per_type,
        "seed": args.seed,
        "annotations": REPO,
        "prompt_version": PROMPT_VERSION,
        "dropped_categories": sorted(UNANSWERABLE_CATEGORIES),
        "rows_before_clean": after_split,
        "rows_after_quality_filter": after_quality,
        "rows_after_clean": after_clean,
        "patches_excluded_cloud_or_snow": len(excluded),
        "quality_filter_applied": not args.keep_cloudy,
        # The mixture is recorded because it is a training decision, not an
        # implementation detail: a later ablation asking whether the preamble
        # helped needs to know what share of records carried one.
        "evidence_mix": dict(sorted(mix.items())),
        "evidence_rate": round(with_evidence / count, 4) if count else 0.0,
    }
    (args.out / "manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )

    print(f"\nwrote {count:,} records to {args.out / (args.split + '.jsonl')}")
    print(f"      {len(images):,} patch pairs to {args.out / 'images'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
