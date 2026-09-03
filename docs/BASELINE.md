# Baseline Bake-Off — Technical Plan

Phase 0 of the build. Goal: **pick the base VLM with evidence**, and produce the
zero-shot numbers that every later fine-tuning delta is measured against.

No training happens in this phase.

---

## 1. What we are deciding

| Question | Decided by |
|---|---|
| Which base VLM? | Highest normalised aggregate across the four benchmarks |
| Does it handle image *pairs* at all? | CDVQA zero-shot accuracy above chance |
| Captioning or grounding as the scored extra task? | VRSBench caption vs referring zero-shot gap |
| What does fine-tuning have to beat? | The frozen baseline table |

That third row matters: the choose-one decision should be made on measured numbers,
not on my prior. If zero-shot grounding is unexpectedly strong, the recommendation
flips.

---

## 2. Candidates

| Model | Why it is in |
|---|---|
| `Qwen2.5-VL-3B-Instruct` | Native multi-image, trained for box output, mature LoRA/vLLM support |
| `InternVL3-2B` | Strong multi-image, different vision stack — genuine alternative |
| `MiniCPM-V` (4.x) | High-resolution tiling, good for small-object RS scenes |

Reference points, not candidates: `GeoChat` and `TeoChat` — run them to know where
published RS-adapted models sit, so we can say "our fine-tune beats GeoChat by X."

Three candidates, not six. Each extra model is ~2–3 GPU hours.

---

## 3. Protocol

**Fixed across all models — the comparison is worthless otherwise.**

- Greedy decoding, `temperature=0`, fixed `max_new_tokens` per task.
- Identical prompt per task from `eval/prompts.py`. Same file the serving path uses.
- Identical seeded subset per benchmark (`--limit`, `--seed 1234`).
- Identical answer normalisation and box parsing.
- Every run writes `predictions.jsonl` + `metrics.json` + one row in `results.csv`
  tagged with model, config hash, and git SHA.

### Sample budget

| Phase | Samples per benchmark | Purpose |
|---|---|---|
| Smoke | 32 | Harness works, prompts render, parser fires |
| Bake-off | 2 000–3 000 | Model comparison |
| Final | full prescribed split | Submission numbers, winner only |

Iterating on full splits is the most common way to waste a GPU budget. The seeded
subset is statistically sufficient to rank three models; the full split is only needed
for the number you report.

---

## 4. Benchmarks and metrics

| Benchmark | Task | Metrics |
|---|---|---|
| RSVQA (LR + HR) | `vqa` | Overall accuracy; per-type accuracy (presence, comparison, count, area, rural/urban) |
| VRSBench VQA | `vqa` | Overall accuracy; per-type |
| VRSBench Caption | `caption` | BLEU-1..4, ROUGE-L, CIDEr-D, METEOR (optional) |
| VRSBench Referring | `grounding` | Acc@IoU0.5, Acc@IoU0.25, mIoU |
| CDVQA | `change_vqa` | OA, AA (average of per-type accuracies) |

BLEU, ROUGE-L and CIDEr-D are implemented in-repo so scoring needs no Java and no
network. METEOR is optional and degrades gracefully when `nltk` data is absent.

**AA vs OA** matters for CDVQA and RSVQA: class imbalance means a model that always
answers "no" can score respectable OA. AA exposes that. Report both, always.

---

## 5. Normalisation

Scoring depends as much on parsing as on the model. Both are versioned and tested.

- **Answers**: lowercase, strip articles and punctuation, collapse whitespace,
  map yes/no variants, number-words → digits, take the first line.
- **Boxes**: tolerant parser covering Qwen (`<|box_start|>(x1,y1),(x2,y2)<|box_end|>`
  and `bbox_2d` JSON, 0–1000 scale), GeoChat (`{<x1><y1><x2><y2>}`, 0–100 scale) and
  bare `[x1,y1,x2,y2]`. All normalised to xyxy in 0–1.

A parse failure is scored as wrong, never dropped — silently dropping unparseable
outputs inflates scores.

---

## 6. Backends

One interface, three implementations:

| Backend | Use |
|---|---|
| `echo` | CI and unit tests. Deterministic, no model, no GPU. |
| `hf` | Correctness reference. Slow, easy to debug. |
| `vllm` | The bake-off. 5–10× throughput on batched short answers. |

Develop against `echo`, verify on `hf` with 32 samples, run the sweep on `vllm`.

---

## 7. Compute plan (~$6 on vast.ai)

**RTX 4090, on-demand, ~15–18 h.** Ada gives bf16 + FlashAttention-2; 24 GB is ample
for 3B inference and headroom to sanity-check a 7B.

| Step | Hours |
|---|---|
| Instance setup, vLLM, benchmark image pull (~20–30 GB) | 1.5–2.5 |
| Harness debug on 32-sample subsets | 1.5 |
| 3 models × 5 benchmark tasks @ 2–3 k samples | 6–9 |
| Full prescribed splits, winner only | 2–3 |
| **Total** | **11–16 h** |

Throughput to check against: ~5–10 samples/s for short-answer VQA at 3B with vLLM
batching; ~2–4/s for captioning; roughly half for two-image pair tasks.

Cost traps: vast bills disk while the instance *exists* — destroy, don't stop. Stage
benchmark images on HF Hub so they pull once.

---

## 8. Data layout

```
data/
  VRSBench/  {Images_val/, VRSBench_EVAL_vqa.json, _Cap.json, _referring.json}
  RSVQA/     {LR/Images_LR/, LR_split_test_questions.json, …, HR/…}
  CDVQA/     {im1/, im2/, cdvqa_test.json}
```

Benchmark JSON schemas vary across releases, so every dataset adapter takes a
**field mapping** from YAML rather than hard-coding keys. Run

```
satquery bench validate --config configs/bench/<name>.yaml
```

against the real download once; it reports the keys it found and what it could not
map, and the fix is a YAML edit rather than a code change.

---

## 9. Exit criteria

- `results.csv` holds every candidate × benchmark, reproducible from one command.
- A winner is named with the margin stated.
- Captioning-vs-grounding is decided on measured numbers.
- The frozen baseline table is committed — it is the denominator for `ML_PLAN.md`.
