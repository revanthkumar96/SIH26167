# Stage A adaptation — measured result, 2026-09-10

The base-versus-adapted comparison the whole programme exists to produce. These
files are the source of the submission's numbers; they are in git because a
figure that lives only in a bucket or on one laptop is one nobody can defend.

## Provenance

| | |
| --- | --- |
| Base | `Qwen/Qwen3-VL-2B-Instruct` @ `89644892e4d85e24eaac8bacfd4f463576704203` |
| Adapter | LoRA r=32 α=64, lr 1e-4 cosine, bf16, effective batch 64 |
| Adapted surfaces | language stack, vision–language projector, **and the vision encoder** |
| Trainable | 49,364,992 params (2.27%) |
| Corpus | BigEarthNet.txt, 98,925 records over 69,561 patch pairs, 38% carrying an evidence preamble |
| Training | **1.0 epoch**, 13,013 s, 7.60 samples/s, final loss 0.4873 |
| Hardware | RTX 4080 SUPER 32 GB (vast.ai), ~$2.25 |
| Harness | `hf` backend for **both** rows, n=200, seed 1234, prompt version 1.1.0 |

Both sweeps were re-run after the bounding-box parser was fixed, so base and
adapted are scored by identical code. An earlier baseline measured against the
older parser was discarded rather than compared across the change.

## Result

| criterion | base | adapted | delta |
| --- | --- | --- | --- |
| Single-image VQA | 0.2250 | **0.4075** | **+0.1825** |
| Optical–SAR joint | 0.0000 | **0.1750** | **+0.1750** |
| Change understanding | 0.0100 | 0.0450 | +0.0350 |
| Grounding | 0.0000 | 0.0000 | 0.0000 |
| Captioning | 0.0128 | 0.0003 | **−0.0126** |
| **Overall** | **0.0496** | **0.1256** | **+0.0760** |

Raw headline metrics: `rsvqa_lr` OA 0.250 → 0.540, `vrsbench_vqa` OA 0.200 →
0.275, `bigearthnet_bench` OA 0.000 → 0.175, `vrsbench_caption` CIDEr-D 0.128 →
0.003.

Scores are normalised against each metric's own declared range before
aggregation, then averaged within a criterion and across criteria. CIDEr-D runs
to 10 and everything else to 1, so a raw mean would be captioning alone.

## What these numbers do not say

Read `quality.txt` alongside the table. Three things the aggregate hides:

**Captioning regressed, and not the way the plan predicted.** `FINETUNING.md`
warns terse-answer training can collapse caption length. It did the opposite —
captions grew from 66.8 to 79.9 words against a 47.4-word reference. The model
learned BigEarthNet's idiom ("expansive mixed forest, covering approximately
67,000 sqm") which does not match VRSBench's reference style. This is the cost
of training on one corpus and scoring on another, and it is what Stage B exists
to correct.

**Much of the base's weakness was silence rather than error.** Empty predictions
in the base row: `cdvqa` 192/200, `rsvqa_lr` 108/200, `vrsbench_vqa` 72/200. The
adapted model returns nothing on `rsvqa_lr` 0/200 — that recovery is most of the
+0.29 there. "It now answers at all" is a real improvement but a different claim
from "it reasons better", and the two should not be merged in a slide.

**The cross-modal gain hides a regression.** `bigearthnet_bench` improved 0.000 →
0.175 while empty answers went 0 → 134/200. On the 66 it answers it is terse and
correct; on the rest it is silent. Both facts are true and the second needs
saying.

**Grounding is genuinely unsolved.** 0.000 before and after. The parser fix did
work — unparsed boxes fell 33 → 0 — so this is localisation failure, not a
scoring artefact.

## Files

| file | contents |
| --- | --- |
| `results.csv` | long-format, one row per metric per cell, as the harness wrote it |
| `scores.json` | normalised per-criterion scores and the delta |
| `delta.txt` | the rendered comparison table |
| `quality.txt` | empty-prediction and unparsed-box counts per cell |

Predictions are not committed: they are large and regenerable from
`s3://satquery-869987460914/results/`.
