# Data

Three data programmes, serving three different purposes:

| programme | source | feeds |
| --- | --- | --- |
| **A** — adaptation | BigEarthNet.txt | VLM fine-tuning (`FINETUNING.md`) |
| **B** — classification labels | BigEarthNet v2.0 | the land-cover CNN (`CNN.md`) |
| **C** — regional supervision | Sentinel over India + our own tools | closing the Europe→India gap |

A and B come from **one 155 GB download** — the reBEN LMDB mirror ships imagery
*and* `metadata.parquet` with the CORINE labels. C costs nothing but compute.

Full provenance in `SOURCES.md`.

---

## A. BigEarthNet.txt — adaptation data

464,044 co-registered S1+S2 pairs, 9,553,962 annotations, Europe only.

### The cleaning is the point

Roughly **1.39M of the 9.55M rows ask questions the pixels cannot answer**:

```
mcq/country       "Identify which of the following countries is shown in the
                   satellite image: a) Lithuania, b) Belgium, c) Austria,
                   d) Luxembourg"                              → "c"
mcq/season        "Which of the following seasons is shown in the image?"
mcq/climate zone  "choose the climate zone shown in the satellite image"
```

Nothing in a picture of farmland says *Austria*. Train on these and the model
learns to state a country confidently from land cover — and our evaluation
imagery is Indian. These categories are dropped, and dropped **before sampling**
so the per-type quota is not spent on rows that are then discarded.

Captions carry the same contamination inline, so category filtering cannot catch
them:

> "This satellite image, **captured in Austria during summer**, depicts a diverse
> landscape dominated by agricultural and forested areas **within the "cold, no
> dry season, warm summer" climate zone**. Arable land (~708,000 sqm) is the most
> prominent feature."

The second half is exactly what we want. So captions are **redacted** — the
acquisition clause and the quoted climate zone are removed, the land-cover
description is kept. Both phrasings the corpus uses are covered by tests written
against verbatim published text. A caption that still names its country after
redaction is **dropped rather than patched**: one we cannot clean confidently is
not worth the hallucination risk.

Two further filters come free from the v2.0 metadata:
`contains_cloud_or_shadow` and `contains_seasonal_snow`. A cloudy patch teaches
nothing about land cover.

**In practice both are already false everywhere.** Measured against the real
`metadata.parquet` on 2026-09-10: all **480,038** patches carry `False` for both
flags, because the v2.0 release removed those patches before publication and
kept the columns only as a record. The filter is therefore a no-op on this
release, and `patches_excluded_cloud_or_snow: 0` in a manifest is the correct
result rather than a broken filter. It stays in the pipeline because it costs
nothing and a future release may not be pre-cleaned.

### Geometry conversion

Boxes are published as `[x1 y1, x2 y2]` in 0–1 floats. Ours are integers on a
0–1000 grid. **This mismatch does not raise** — it scores zero on every grounding
item and looks like a model failure. Conversion is explicit and tested in both
directions:

```
[0.64 0.0, 1.0 0.71]   →   [640, 0, 1000, 710]
```

Prompt markup is rewritten to match how the grounding tool phrases a request at
inference time: `<point>(0.82, 0.28)</point>` → `(820, 280)`, and `<ref>` /
`</ref>` are stripped.

### Sampling

Stratified across `country` and `season` by construction. The corpus carries
those columns precisely so a draw need not be narrow — a competitor's published
adapter took every row from Lithuania in summer, the narrowest possible slice of
an already Europe-only corpus. A test asserts no country can dominate a draw.

Target: **100–200k rows over 30–50k unique pairs**, roughly 3–4% of the train
split, balanced across the four task types.

### Rendering

Optical is a true-colour B04/B03/B02 composite. SAR is VV / VH / (VV − VH) —
the standard three-plane visualisation, a *difference* rather than a ratio
because **BigEarthNet Sentinel-1 is already in dB**.

Rendering delegates to the serving path's `stretch_to_uint8` rather than
reimplementing it. Two implementations of "the same" normalisation is how
train/serve skew gets in, and it is invisible until scores are inexplicably bad.

### Running it

```bash
pip install -e ".[data,geo]"

# bring-up on the 2.5 GB slice — no AWS, minutes not hours
huggingface-cli download hackelle/BigEarthNetV2-Lithuania-Summer-LMDB \
  --repo-type dataset --local-dir data/ben-dev

python scripts/prepare_bigearthnet.py \
  --lmdb data/ben-dev/BENv2_lithuania_summer.lmdb \
  --out  data/prepared/dev --per-type 500

# the real pass, against the full store
python scripts/prepare_bigearthnet.py \
  --lmdb data/BENv2.lmdb --out data/prepared/train --per-type 40000
```

Outputs `train.jsonl`, `images/`, and a `manifest.json` recording exactly what
was dropped and why. Always verify on the small slice first.

---

## The evidence preamble — do not skip this

At inference the model receives measurements before the task prompt:

```
Measurements from image-analysis tools that have already run on these
images. Treat them as reliable and do not contradict them:
- water fraction from SAR backscatter: 0.3151
- built-up fraction from optical NDBI: 0.2803

<task prompt>
```

**If the fine-tuning data never contains this, the fine-tuned model has never
seen it.** Two quiet failure modes:

1. It ignores the measurements, and the entire evidence-grounding design — the
   thing that distinguishes this system — stops working.
2. Worse: adapted on bare prompts, it becomes *more* confident in its own visual
   reading, contradicts the measurements more often, and the contradiction-retry
   fires constantly, doubling latency for worse answers.

This is the same class of bug as image-normalisation skew, in the prompt channel.

**Fix, in the preparation pipeline:** synthesise preambles into roughly 40–50% of
training records using the *same* `format_evidence()` function the controller
calls. BigEarthNet supplies the ground truth — `area` rows state coverage,
`presence` rows state which classes are there.

Compose the mix deliberately:

- **decisive** — the answer follows from the evidence. Teaches the model to use it.
- **orthogonal** — evidence present, unrelated to the question. Teaches it not to
  over-apply.
- **never contradictory** — never train on a preamble that disagrees with the
  gold answer, or you teach it to ignore evidence.

Add a test asserting the training preamble matches `format_evidence()`
byte-for-byte so drift is caught at commit time, not at evaluation time.

---

## B. BigEarthNet v2.0 — CNN labels

Same LMDB, different annotation layer. `metadata.parquet` carries multi-label
CORINE classes per patch:

```
['Arable land', 'Broad-leaved forest', 'Mixed forest', 'Pastures']
```

Multi-label **classification**, not segmentation — there are no per-pixel masks,
so do not promise a segmentation map from it. See `CNN.md`.

---

## C. Regional supervision over India

The gap BigEarthNet cannot close: it contains **no Indian scene at all**, and
the hidden evaluation set is entirely Indian terrain.

The play is **programmatic supervision**, not classical semi-supervised learning.
SSL solves label scarcity; we are not label-scarce, we are domain-shifted. What
we lack is labelled *Indian* data — and we can generate it.

We can pull unlimited Sentinel-1/2 over India today through Planetary Computer;
the live feed already does. It has no text labels. But `optical_indices`,
`sar_indices` and `change_mask` compute water fraction, built-up fraction, change
extent and location **deterministically from physics**. Those are measurements,
not model guesses. So training triples come for free:

| generated question | answer source |
| --- | --- |
| "What fraction of this scene is water?" | NDWI / SAR Otsu |
| "Where are the water bodies?" | quadrant summary of the mask |
| "Has built-up area increased?" | `change_mask` direction |
| "Locate the largest water body." | connected component of the mask |

This is exactly how BigEarthNet.txt itself was built — text generated from label
masks — applied to Indian terrain instead of European.

### The confidence gate

Thresholding is not ground truth, and errors would be baked in. The filter that
makes this defensible: **only emit a pair when optical NDWI and SAR backscatter
agree** within tolerance. Two physically independent sensors concurring is strong
evidence the label is right, and we already compute both on every cross-modal
run. Where they disagree, the scene is ambiguous and is skipped.

Honest limits, to state rather than hide:

- It teaches *measurable* properties — water, built-up, change. Not semantics.
- It inherits our thresholds' biases, so it must be mixed with real annotated
  data rather than trained on alone.
- Sentinel is 10 m; it fixes **region**, not **resolution**. Resolution comes
  from VRSBench.

---

## Domain gaps, summarised

| gap | source of the problem | mitigation |
| --- | --- | --- |
| Region | BigEarthNet is Europe-only | programme C, over Indian Sentinel |
| Resolution | 120×120 at 10 m vs sub-metre Cartosat | VRSBench (~0.3 m) in the mixture |
| Bands | Cartosat has no SWIR → no NDBI | SAR, plus the CNN |
| Modality | text is generated from labels, templated | mix real benchmarks in Stage B |
| Time | BigEarthNet is single-date | CDVQA supplies bi-temporal |
