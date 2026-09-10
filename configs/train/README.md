# Stage B train-split configs

These point the adapters at the **train** annotations. They are kept in their own
directory, with their own `root`, because the one mistake that would quietly ruin
the programme is a config that says "train" and reads the file the benchmark
scores. Separate directories mean a glob can never pick up the wrong one:

    satquery bench run     --config "configs/bench/*.yaml"    # scoring
    satquery data instruct --config "configs/train/*.yaml"    # training

`satquery data instruct` checks every record against the test splits before
writing, and fails rather than warns. Do not pass `--no-guard`.

## The mixture needs BigEarthNet, and it is not optional

The three benchmarks cannot supply two of the things Stage A earned:

- **No optical–SAR pairs.** That is why `bigearthnet_bench` exists at all —
  RSVQA and VRSBench are single-image, CDVQA is bi-temporal. A Stage B corpus
  built from the benchmarks alone trains away the **+0.1750** Stage A measured
  on that criterion, and nothing in the run reports it.
- **No evidence preambles.** They need NDWI and NDBI, which need NIR and SWIR.
  A three-channel benchmark JPEG cannot produce one, so every benchmark source
  reports `evidence=0`.

So the corpus is *benchmark train splits + a slice of the existing BigEarthNet
corpus*, mixed with `--include`:

    satquery data instruct         --config "configs/train/*.yaml"         --test-config "configs/bench/*.yaml"         --include bigearthnet=data/prepared/train/train.jsonl         --out data/prepared/stage-b/train.jsonl

The slice is **rehearsal, not training**: Stage B resumes from Stage A's
adapter, so cross-modal ability is already in the weights and this is here to
stop it being trained away. `satquery data instruct` warns if either property
is missing from the mixture it just wrote.

One caveat on the preamble share. BigEarthNet was prepared at
`--preamble-rate 0.45`, giving 36.5% coverage, so a 20,000-record slice against
~52,000 benchmark records lands near 10% — below the 20% that
`scripts/train_lora.py` warns at. The fix is to re-prepare the slice with a
higher `--preamble-rate` (the ceiling is ~81%, where `choose_evidence` can
establish agreement), **not** to enlarge it, which would just rerun Stage A.

Pull the data first:

    satquery data pull vrsbench_train --with-images
    satquery data pull rsvqa_lr_train
    satquery data pull cdvqa_train --shards 660

## What each source contributes

Measured by running the converter against the real downloads, not estimated:

| config | available | cap | notes |
| --- | --- | --- | --- |
| `vrsbench_train_caption` | 20,264 | 12,000 | reference-style captions — the Stage A regression |
| `vrsbench_train_vqa` | 85,813 | 12,000 | six instruction templates, stripped and re-rendered |
| `vrsbench_train_referring` | 36,281 | 10,000 | boxes converted 0-100 → 0-1000 |
| `rsvqa_lr_train` | ~500k | 10,000 | templated; uncapped it *is* the corpus |
| `cdvqa_train` | 65,967 | 8,000 | only if the guard clears it |

About 52,000 records, roughly two hours per epoch at Stage A's measured
7.6 samples/s.

## Verified

- VRSBench train images are disjoint from the EVAL set: **0 of 20,262 shared**.
- RSVQA LR test uses tiles 232–331 of 772, so train tiles are a different id
  range in the same shared directory.
- CDVQA train and test are disjoint in shard 0 — but that is three images each,
  and CDVQA draws 122k questions from a few thousand SECOND tiles. **Unresolved
  until the guard runs on the full pull.**

- RSVQA LR train and test tiles are **disjoint**: 572 active train tiles
  spanning ids 0–771, excluding the 232–331 test block entirely. Both splits
  read the same shared `Images_LR` directory, so this is the one source where
  the guard is checking a genuine hazard rather than confirming separate
  directories.

## Not verified

CDVQA train/test tile overlap. Everything else above was checked against the
real downloads.
