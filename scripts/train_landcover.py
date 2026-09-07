#!/usr/bin/env python
"""Train the multi-label land-cover CNN on BigEarthNet v2.0.

    python scripts/train_landcover.py \
        --lmdb data/BENv2.lmdb --metadata data/metadata.parquet \
        --out runs/landcover --epochs 8

Multi-label classification over the 19-class v2.0 nomenclature: one sigmoid
logit per class, binary cross-entropy, no softmax. A patch carries a *set* of
land-cover classes, and making them compete for probability mass would be
modelling something the data does not say.

549k patches at 120x120 is small by modern standards -- a few GPU hours on the
A10G, well under $5. Checkpoints stream to disk every epoch because spot
instances are reclaimed with two minutes' notice.

What this produces is a classifier, not a segmenter. BigEarthNet has no
per-pixel masks, so the output is "which classes are present, with what
confidence" over the whole patch. Do not present it as a land-cover map.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

import numpy as np


def load_split(
    metadata_path: Path, split: str, keep_cloudy: bool = False
) -> tuple[list[str], list[list[str]], list[str]]:
    """Patch ids, their labels, and the class vocabulary actually present."""
    import pandas as pd

    from satquery.cnn.labels import observed_classes

    frame = pd.read_parquet(metadata_path)
    frame = frame[frame["split"] == split]
    before = len(frame)

    # A cloudy patch teaches nothing about land cover, and a snowy one teaches
    # the wrong thing. Both flags ship with the metadata, so the filter is free.
    if not keep_cloudy:
        flags = [
            c
            for c in ("contains_cloud_or_shadow", "contains_seasonal_snow")
            if c in frame.columns
        ]
        if flags:
            frame = frame[~frame[flags].fillna(False).any(axis=1)]

    patches = [str(p) for p in frame["patch_id"]]
    labels = [list(row or []) for row in frame["labels"]]
    print(
        f"  {split}: {before:,} -> {len(patches):,} patches "
        f"({before - len(patches):,} dropped for cloud or snow)"
    )
    return patches, labels, observed_classes(labels)


class PatchDataset:
    """Reads band stacks straight out of the LMDB.

    Opened lazily per worker: an LMDB environment does not survive a fork, so
    sharing one across DataLoader workers produces silent corruption rather
    than an error.
    """

    def __init__(
        self,
        lmdb_path: Path,
        patches: list[str],
        targets: np.ndarray,
        s1_of: dict[str, str] | None = None,
    ) -> None:
        self.lmdb_path = str(lmdb_path)
        self.patches = patches
        self.targets = targets
        self.s1_of = s1_of or {}
        self._env: Any = None

    def __len__(self) -> int:
        return len(self.patches)

    def _open(self) -> Any:
        import lmdb

        if self._env is None:
            self._env = lmdb.open(
                self.lmdb_path, readonly=True, lock=False, readahead=False
            )
        return self._env

    def __getitem__(self, index: int) -> tuple[Any, Any]:
        import torch
        from safetensors.numpy import load as safetensor_load

        from satquery.data.bigearthnet import S1_BANDS, S2_BANDS, stack_bands

        patch = self.patches[index]
        with self._open().begin(write=False) as txn:
            blob = txn.get(patch.encode())
            if blob is None:
                raise KeyError(f"patch {patch} not in the LMDB")
            stack = stack_bands(safetensor_load(bytes(blob)), S2_BANDS)

            s1_key = self.s1_of.get(patch)
            if s1_key:
                sar_blob = txn.get(s1_key.encode())
                if sar_blob is not None:
                    sar = stack_bands(safetensor_load(bytes(sar_blob)), S1_BANDS)
                    stack = np.concatenate([stack, sar])

        return torch.from_numpy(normalise(stack)), torch.from_numpy(self.targets[index])


def normalise(stack: np.ndarray) -> np.ndarray:
    """Per-band standardisation.

    Computed per patch rather than from corpus statistics deliberately: the
    hidden evaluation imagery is a different sensor with a different radiometric
    range, and a model fitted to BigEarthNet's absolute band means would meet
    Cartosat values far outside them. Per-patch standardisation is the version
    that transfers.
    """
    array = np.asarray(stack, dtype=np.float32)
    mean = array.mean(axis=(1, 2), keepdims=True)
    std = array.std(axis=(1, 2), keepdims=True)
    return (array - mean) / np.maximum(std, 1e-6)


def evaluate(model: Any, loader: Any, classes: list[str], device: str) -> dict:
    import torch

    from satquery.cnn.metrics import classification_metrics

    model.eval()
    logits: list[np.ndarray] = []
    targets: list[np.ndarray] = []
    with torch.no_grad():
        for batch, target in loader:
            output = model(batch.to(device))
            logits.append(output.float().cpu().numpy())
            targets.append(target.numpy())
    model.train()
    return classification_metrics(
        np.concatenate(logits), np.concatenate(targets), classes
    )


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--lmdb", required=True, type=Path)
    ap.add_argument("--metadata", required=True, type=Path)
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--epochs", type=int, default=8)
    ap.add_argument("--batch-size", type=int, default=128)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--seed", type=int, default=1234)
    ap.add_argument(
        "--limit", type=int, default=None, help="cap patches, for smoke runs"
    )
    ap.add_argument(
        "--with-sar",
        action="store_true",
        help="append the two S1 polarisations, giving a 14-band stem",
    )
    ap.add_argument(
        "--reben-weights",
        type=Path,
        default=None,
        help="BIFOLD pretrained reBEN checkpoint, if its licence permits use",
    )
    args = ap.parse_args()

    import torch
    from torch.utils.data import DataLoader

    from satquery.cnn.labels import encode
    from satquery.cnn.model import build_model

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"device: {device}")

    print(f"reading {args.metadata} ...")
    train_patches, train_labels, observed = load_split(args.metadata, "train")
    val_patches, val_labels, _ = load_split(args.metadata, "validation")

    # The vocabulary comes from the data, not the constant. It is written into
    # the checkpoint because logit order is not recoverable from the weights,
    # and a checkpoint read under a different ordering scores every class
    # against the wrong name.
    classes = observed
    print(f"  {len(classes)} classes observed in the training split")

    if args.limit:
        train_patches, train_labels = (
            train_patches[: args.limit],
            train_labels[: args.limit],
        )
        val_patches, val_labels = val_patches[: args.limit], val_labels[: args.limit]

    s1_of: dict[str, str] = {}
    if args.with_sar:
        import pandas as pd

        meta = pd.read_parquet(args.metadata)
        s1_of = {
            str(r["patch_id"]): str(r["s1_name"])
            for _, r in meta[["patch_id", "s1_name"]].iterrows()
        }

    train_targets = np.array(
        [encode(row, classes) for row in train_labels], dtype=np.float32
    )
    val_targets = np.array(
        [encode(row, classes) for row in val_labels], dtype=np.float32
    )

    train_loader = DataLoader(
        PatchDataset(args.lmdb, train_patches, train_targets, s1_of),
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.workers,
        drop_last=True,
    )
    val_loader = DataLoader(
        PatchDataset(args.lmdb, val_patches, val_targets, s1_of),
        batch_size=args.batch_size,
        num_workers=args.workers,
    )

    channels = 14 if args.with_sar else 12
    model = build_model(
        channels=channels,
        num_classes=len(classes),
        pretrained=args.reben_weights is None,
        weights_path=str(args.reben_weights) if args.reben_weights else None,
    ).to(device)

    optimiser = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    schedule = torch.optim.lr_scheduler.CosineAnnealingLR(optimiser, T_max=args.epochs)
    criterion = torch.nn.BCEWithLogitsLoss()
    # bf16 on the A10G: Ampere, so no GradScaler and one fewer source of
    # training instability than the fp16 path a T4 would force.
    autocast = torch.autocast(
        device_type=device, dtype=torch.bfloat16, enabled=device == "cuda"
    )

    args.out.mkdir(parents=True, exist_ok=True)
    history = []

    for epoch in range(1, args.epochs + 1):
        started = time.perf_counter()
        running = 0.0
        for index, (batch, target) in enumerate(train_loader, 1):
            batch, target = batch.to(device), target.to(device)
            optimiser.zero_grad(set_to_none=True)
            with autocast:
                loss = criterion(model(batch), target)
            loss.backward()
            optimiser.step()
            running += float(loss.item())
            if index % 50 == 0:
                print(
                    f"  epoch {epoch} batch {index} loss {running / index:.4f}",
                    flush=True,
                )
        schedule.step()

        metrics = evaluate(model, val_loader, classes, device)
        elapsed = time.perf_counter() - started
        print(
            f"epoch {epoch}: loss {running / max(len(train_loader), 1):.4f}  "
            f"mAP {metrics['map']:.4f}  micro F1 {metrics['f1_micro']:.4f}  "
            f"macro F1 {metrics['f1_macro']:.4f}  ({elapsed:.0f}s)"
        )
        history.append(
            {"epoch": epoch, "loss": running / max(len(train_loader), 1), **metrics}
        )

        # Written every epoch, not only at the end: a reclaimed spot instance
        # gives two minutes' notice, and a lost eight-hour run is a lost day.
        torch.save(
            {
                "state_dict": model.state_dict(),
                "classes": list(classes),
                "channels": channels,
                "epoch": epoch,
                "metrics": metrics,
                "seed": args.seed,
            },
            args.out / "landcover.pt",
        )
        (args.out / "history.json").write_text(
            json.dumps(history, indent=2), encoding="utf-8"
        )

    final = history[-1] if history else {}
    print("\nper-class average precision:")
    for key, value in sorted(final.items()):
        if key.startswith("ap/"):
            support = final.get(f"support/{key[3:]}", 0)
            print(f"  {key[3:]:<50} {value:.4f}  (n={support:.0f})")

    print(f"\nwrote {args.out / 'landcover.pt'}")
    print(
        f"point the tool at it: SATQUERY_LANDCOVER_CHECKPOINT={args.out / 'landcover.pt'}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
