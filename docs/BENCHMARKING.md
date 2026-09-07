# Benchmarking

The harness scores models on the prescribed test splits with the prescribed
metrics. It is the instrument that decides whether anything else in this project
worked, so it is built to be boring, reproducible, and honest.

Its most important output is a **two-row comparison**: stock Qwen3-VL against our
adapted model, same splits, same prompts, same seed. That delta is the evidence
that mandatory scope item 1 was discharged.

---

## What is scored, and against what

| criterion | dataset / split | metrics |
| --- | --- | --- |
| Single-image VQA *(mandatory)* | VRSBench VQA, RSVQA test splits | overall accuracy; per-question-type where defined |
| Captioning | VRSBench captioning test | BLEU, METEOR, ROUGE-L, CIDEr |
| Grounding | VRSBench referring test | Acc@IoU 0.5, mIoU, Acc@IoU 0.25 |
| Change understanding *(mandatory)* | CDVQA official test | OA and AA across question types |
| Optical–SAR joint *(mandatory)* | hidden ISRO/SAC set; public co-registered pairs | task-specific |
| Agentic orchestration | observable execution trace | correct routing, valid params, auditable summary, evidence, confidence |
| System completeness | demonstration | functional checklist |

The problem statement requires captioning **or** grounding as the additional
single-image task. We implement and score **both**, and elect whichever is
stronger at submission time.

Scores from heterogeneous metrics must be **normalised before aggregation** so
no metric dominates by scale. This is a scored requirement and is **not yet
implemented** — see the gap list below.

---

## Metrics, as implemented

All in `eval/metrics/`, written directly rather than pulled from a scoring
package, so the constants are visible and auditable.

**Captioning** — `caption.py`

- BLEU with brevity penalty against the closest reference length
- ROUGE-L, F-measure with **β = 1.2**, the standard for summarisation
- CIDEr-D with **σ = 6.0**, n = 1..4, TF-IDF over the reference corpus
- METEOR, guarded — if NLTK data is unavailable it is reported absent rather
  than silently zero

**Grounding** — `grounding.py`
IoU on unit-normalised xyxy boxes, thresholds `(0.25, 0.5)`, plus mIoU. An
unparseable box counts as IoU 0 rather than being dropped — a model that fails
to emit a box has failed the task, and quietly excluding those inflates the mean.

**VQA** — `vqa.py`
Exact match after normalisation, plus per-question-type accuracy for AA.

### Answer normalisation

Exact match is brittle in exactly the ways that flatter or punish a model
unfairly. The normaliser lowercases, collapses whitespace, and strips `./-%` at
token edges — `"Yes."` must match `"yes"`. It does not do synonym expansion:
that would be scoring generosity we cannot justify against a hidden reference.

### OA versus AA

Overall accuracy is the mean over questions; average accuracy is the mean of
per-type accuracies. CDVQA question types are heavily imbalanced, so a model that
is excellent on the common type and useless on the rare ones posts a strong OA
and a poor AA. **Both are reported.** AA is the more honest number and the
problem statement asks for it.

---

## Running it

Benchmarks are declared in `configs/bench/*.yaml`, one per split, each naming its
adapter class, annotation paths, and task. Five exist:
`rsvqa_lr`, `vrsbench_vqa`, `vrsbench_caption`, `vrsbench_referring`, `cdvqa`.

The matrix runs *model-outer, dataset-inner*, writing a result file per cell and
skipping cells that already exist. A crashed sweep resumes instead of restarting,
which matters when a cell is 15 minutes of GPU time.

Defaults: `limit=200` samples, `seed=1234`, `reuse_cached=true`.

### Sample counts and what is actually runnable

| config | annotations | imagery on disk |
| --- | --- | --- |
| `cdvqa` | 200 | ✅ 200 |
| `rsvqa_lr` | 10,004 | ✅ 10,004 |
| `vrsbench_caption` | 9,350 | ❌ 0 |
| `vrsbench_referring` | 16,159 | ❌ 0 |
| `vrsbench_vqa` | 37,409 | ❌ 0 |

**Three of five cannot be scored until VRSBench imagery is pulled** — a 3.8 GB
opt-in download. This is the first blocking item for any real benchmark run.

### Cost of a sweep

At the A10G throughput expected from `AWS.md`, one model across all five configs
at `limit=200` is roughly **50–60 minutes**. Base plus adapted is about 2 hours.

Do not run full splits. `rsvqa_lr` at 10,004 samples is ~7 hours and
`vrsbench_vqa` at 37,409 is over a day. Use 200 for iteration and one
1,000-sample pass for the headline number.

---

## The cross-modal gap, and how it closes

`eval/prompts.py` records that **no public benchmark covers cross-modal
analysis** — RSVQA and VRSBench are single-image, CDVQA is bi-temporal. The
cross-modal path, one of the three mandatory capabilities, could be demonstrated
but never scored.

**BigEarthNet.txt's `bench` split closes this.** 15,029 annotations over 1,082
co-registered S1+S2 pairs, **manually verified** by the publishers. Adding it to
`configs/bench/` as a first-class benchmark makes the mandatory cross-modal
criterion measurable rather than merely demonstrable.

This is the highest-value benchmarking work available and it is cheap. Do it
before the fine-tune, so cross-modal has a before column too.

Caveat to state when reporting it: it is Sentinel at 10 m over Europe, while the
hidden set is Cartosat/RISAT over India. It measures cross-modal *reasoning*, not
performance on the evaluation domain.

---

## Prompts are part of the measurement

`eval/prompts.py` carries `PROMPT_VERSION`. Prompts are shared between the
harness and the serving path, so a benchmark number is always produced by the
prompt the deployed system uses. Changing a prompt changes scores — bump the
version and re-run the affected baselines. A comparison across a prompt change is
not a comparison.

The cross-modal prompt is deliberately *not* terse, unlike the others. Terseness
elsewhere exists to score under exact match; cross-modal is judged on the hidden
set and demonstrated interactively, where a one-word reply is a non-answer. Asked
tersely, the model returns the noun phrase from the question and reports nothing.

That prompt also carries a scar worth remembering: an early version said "where
the optical image is obscured", and the model duly invented cloud cover that was
not there. **A prompt that mentions a condition invites the model to confirm it.**
The clause was removed.

---

## Reporting honestly

- Report **AA alongside OA** wherever types are imbalanced.
- Report **n**. A number from 120 self-selected rows is not comparable to one
  from a prescribed test split, and presenting them alike is misleading.
- Report **truncation and parse-failure rates**. A caption cut off at the token
  budget and a grounding answer that emitted no box are defects the aggregate
  score hides.
- Report the **prompt version, seed, sample limit and model revision** with every
  table. A score without them cannot be reproduced or defended.

---

## Known gaps

| gap | impact |
| --- | --- |
| VRSBench imagery not downloaded | three of five configs unscoreable |
| No baseline run recorded yet | no "before" column exists |

Both need a GPU box rather than more code. `satquery data pull vrsbench
--imagery` is the first step and now works — it was broken by a module-shadowing
bug until the `satquery.data` package was reorganised.

Closed since this document was written:

- **Score normalisation before aggregation** — `eval/aggregate.py`, reachable as
  `satquery bench score`. Normalises against each metric's own declared range
  rather than the spread across the sweep, and averages within a criterion
  before averaging across criteria.
- **BigEarthNet `bench` wired as a config** — `configs/bench/bigearthnet_bench.yaml`,
  so optical–SAR joint analysis has a score rather than only a demonstration.
