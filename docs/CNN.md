# The land-cover CNN

A multi-label land-cover classifier, trained on BigEarthNet v2.0, registered as
a tenth tool. It exists to answer a question the index tools cannot answer on the
imagery we will actually be evaluated on.

---

## Why it is needed

`optical_indices` computes built-up extent as NDBI:

```
NDBI = (swir − nir) / (swir + nir)
```

**Cartosat-2S has no SWIR band.** It is panchromatic, or 4-band MX (B/G/R/NIR).
So on the hidden evaluation set, `optical_indices` will correctly report
`applicable: false` for built-up, and the only remaining optical signal is —
nothing. Built-up would come from SAR alone.

A CNN learns land cover from whatever bands it is given. It needs no SWIR, works
on RGB or panchromatic input, and produces class evidence where the physics-based
index cannot be computed at all.

Two secondary benefits:

- **It gives the model richer evidence than a scalar.** "arable land (0.91),
  mixed forest (0.77)" is more useful to a captioning prompt than
  "builtup_fraction: 0.28".
- **It settles the compliance question cleanly.** The problem statement requires
  at least one *visual or vision-language* component to be adapted. A CNN trained
  on remote-sensing imagery is unambiguously an adapted visual component — a
  stronger claim than LoRA on a language model's attention blocks.

---

## What it is, precisely

**Multi-label classification, not segmentation.** BigEarthNet v2.0 labels a whole
patch with a set of CORINE classes:

```
['Arable land', 'Broad-leaved forest', 'Mixed forest', 'Pastures']
```

There are **no per-pixel masks** in the dataset, so this model cannot produce a
land-cover map. Do not promise one in a demo or a slide. It answers *which
classes are present, with what confidence*, over the analysed window.

If a spatial map is wanted later, the honest routes are class-activation
mapping for weak localisation, or a different dataset with segmentation masks.
Both are out of scope for now.

---

## Start from published weights

BIFOLD publish pretrained reBEN model weights alongside the dataset. **Confirm
availability and licence before Phase 4b** — this is marked unverified in
`SOURCES.md` — but if they are usable, fine-tuning from them instead of training
from scratch likely removes the entire training cost of this component.

Fall back to a torchvision ResNet-50 initialised from ImageNet if not.

---

## Configuration

| | |
| --- | --- |
| Architecture | ResNet-50 (or the published reBEN backbone) |
| Input | 12-band S2, optionally + 2-band S1, 120×120 |
| Head | multi-label, sigmoid, one logit per CORINE class |
| Loss | binary cross-entropy with logits |
| Precision | bf16 on the A10G |
| Data | BigEarthNet v2.0, official train split |
| Filters | drop `contains_cloud_or_shadow` and `contains_seasonal_snow` |

**The first-layer stem must be adapted** for non-3-channel input. An ImageNet
ResNet expects 3 channels; 12-band Sentinel-2 needs the stem convolution
re-initialised at 12 (or 14 with SAR). Inflating the pretrained RGB weights
across the new channels transfers better than random initialisation.

Cost: 549k patches at 120×120 is small by modern standards. A few GPU hours on
the A10G, well under **$5**.

---

## Metrics

Multi-label classification is not accuracy. Report:

- **mAP** (mean average precision) — the primary number, and what the reBEN
  literature reports, so it is comparable
- **micro / macro F1** — macro exposes rare-class failure that micro hides
- **per-class AP** — the classes that matter to our queries are water bodies,
  urban fabric, industrial units and arable land

Class imbalance is severe in CORINE. A model that ignores every rare class can
still post a respectable micro F1, which is why macro is reported alongside.

---

## Registering it as a tool

```
name            landcover_cnn
kind            measurement
accepts         single, and both pair configurations
cost            fast
outputs         classes, confidences, dominant_class, applicable
allowed_params  min_confidence (0.0–1.0)
```

Then three integration steps, per `TOOLS.md`:

1. Register in `default_registry()`.
2. Add to `_PRECURSORS` for the tasks that should have it — captioning and VQA
   are the obvious ones; it is useful on **single-image** runs, which currently
   have no precursor at all and go straight to the VLM.
3. Add its outputs to `_ARTIFACT_MAP` and `_EVIDENCE_LABELS`, or its numbers will
   be computed and never reach the model.

Step 3 is the one that is easy to forget.

Adding it to single-image tasks is a meaningful change: it means every run —
not just paired ones — carries a deterministic measurement into the prompt.
It also adds ~1 s to single-image latency, which is worth it.

---

## Consequences to keep in view

**It must be in the fine-tuning evidence mix.** If the CNN emits
`land cover classes present: ...` at inference but no training record ever
contained that line, the adapted VLM has never seen it. Whatever preamble the
CNN contributes has to be part of the synthesised evidence described in
`DATA.md`. Decide the CNN's evidence format *before* the fine-tune, not after.

**Europe-trained, like everything else from BigEarthNet.** CORINE classes are a
European taxonomy. Indian land cover has categories it does not describe well.
The classifier will be least reliable exactly where we need it most, so its
confidences should be surfaced honestly in the trace and never presented as
ground truth.

**It is not a replacement for the index tools.** Where SWIR exists, NDBI is
cheaper, deterministic, and needs no trust. The CNN is the fallback that makes
the system degrade gracefully rather than fail — which is what the hidden
evaluation set will demand of it.
