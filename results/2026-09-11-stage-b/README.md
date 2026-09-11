# Stage B adaptation — measured result, 2026-09-11

Instruction tuning on the task mixture, resumed from the Stage A adapter. These
files are the source of the submission's numbers; they are in git because a
figure that lives only in a bucket is one nobody can defend.

## Provenance

| | |
| --- | --- |
| Base | `Qwen/Qwen3-VL-2B-Instruct` @ `89644892e4d85e24eaac8bacfd4f463576704203` |
| Adapter | LoRA r=32 α=64, lr 1e-4 cosine, bf16, **resumed from Stage A** |
| Adapted surfaces | language stack, vision–language projector, and the vision encoder |
| Trainable | 49,364,992 params (2.27%) |
| Corpus | 64,000 records: 34,000 VRSBench train, 10,000 RSVQA LR train, 20,000 BigEarthNet rehearsal |
| Evidence preambles | 14,390 records (22.5%), from a slice re-prepared at `--preamble-rate 0.85` |
| Training | **2.0 epochs, 2000/2000 steps**, full cosine anneal, final loss ~0.78 |
| Hardware | RTX 4080 SUPER 32 GB (vast.ai), ~$1.3 across two rentals |
| Harness | `hf` backend for both rows, n=200, seed 1234, prompt version 1.1.0 |

Training was split across two rentals and resumed from `checkpoint-1000` with
`--resume-from-checkpoint`, which restores step count, optimizer state and LR
schedule position. It is one training run, not two — and unlike Stage A, the
cosine schedule completed rather than being truncated by a time budget.

## Contamination

The guard ran on the training box against all six benchmark test splits:

    clean: 64,000 training records checked against 11,133 benchmark images,
    no image reuse (617 records share only a question string with a benchmark
    row, over a different image)

BigEarthNet train and bench were also confirmed disjoint at source: 69,051 train
patches against 978 bench patches, **zero shared images and zero shared patches**.

## Result

| criterion | base | Stage B | delta | *Stage A* |
| --- | --- | --- | --- | --- |
| Single-image VQA | 0.2250 | **0.6675** | **+0.4425** | *+0.1825* |
| Change understanding | 0.0100 | **0.3000** | **+0.2900** | *+0.0350* |
| Optical–SAR joint | 0.0000 | **0.1600** | **+0.1600** | *+0.1750* |
| Grounding | 0.0000 | 0.0000 | 0.0000 | *0.0000* |
| Captioning | 0.0128 | 0.0006 | **−0.0123** | *−0.0126* |
| **Overall** | **0.0496** | **0.2256** | **+0.1761** | *+0.0760* |

Raw headline metrics: `rsvqa_lr` OA 0.250 → **0.725**, `vrsbench_vqa` OA 0.200 →
**0.610**, `cdvqa` OA 0.010 → **0.300**, `bigearthnet_bench` OA 0.000 → **0.160**,
`vrsbench_caption` CIDEr-D 0.128 → 0.006.

Scores are normalised against each metric's own declared range before
aggregation. CIDEr-D runs to 10 and everything else to 1, so a raw mean would be
captioning alone.

## What these numbers say that Stage A's did not

**The silence is gone.** Stage A's largest caveat was that much of its gain was
the model beginning to answer at all. Empty predictions, base → Stage B:

| | base | Stage B |
| --- | --- | --- |
| `cdvqa` | 192/200 | **0** |
| `rsvqa_lr` | 108/200 | **0** |
| `vrsbench_vqa` | 72/200 | **0** |
| `vrsbench_referring` unparsed | 33/200 | **1** |

Every accuracy gain above sits on top of a model that answers every question.

**Cross-modal survived.** 0.160 against Stage A's 0.175, after training on four
new corpora. That is what the BigEarthNet rehearsal slice was for, and it worked.

## What is still wrong

**Captioning got worse, and it is the thing Stage B was built to fix.**
CIDEr-D 0.128 → 0.006, with mean caption length 66.8 → **110.9 words** against a
47.4-word reference.

The cause is a corpus error, diagnosed after the fact and now fixed in
`satquery.data.instruct`. Before the run, caption/reference alignment was checked
on the VRSBench train split alone — median 52 words against the 47.4-word
reference — and declared sound. The BigEarthNet rehearsal slice was never
re-checked for what it contributed to the *caption* distribution: its captioning
records run to a median of 96 words. The mixture's p90 was 112.

A second measurement ruled out the obvious alternative explanation. Re-running
the caption benchmark with the token budget raised from 128 to 320 gave
**CIDEr-D 0.000** — worse than the truncated 0.006. So the fault is length, not
truncation, and a larger budget would have been the wrong fix.

`DEFAULT_SLICE_DROP_TASKS` now drops `captioning` from an included slice. The
rebuilt corpus has caption median 52, p90 77, and captions over 100 words fall
from 2,302 to 113 — while keeping 20,000 optical–SAR records and *raising*
evidence coverage to 23.2%. The fix costs nothing it was included for. **It has
not yet been trained on.**

**Grounding is unsolved, but the format problem is.** Still 0.000, while
unparsed boxes fell 33 → 1. The model now emits well-formed boxes on the right
grid, in the wrong places. That is localisation failure, not a scoring artefact.

## Files

| file | contents |
| --- | --- |
| `results.csv` | long-format, one row per metric per cell, as the harness wrote it |
| `scores.json` | normalised per-criterion scores and the delta |
| `delta.txt` | the rendered comparison table |
| `quality.txt` | empty predictions, unparsed boxes and mean answer length per cell |

Predictions are not committed: they are large and regenerable from
`s3://satquery-869987460914/results/`.
