# Roadmap

Where the project stands against the problem statement, and the order of work
from here. Branch: `agent_v2`.

---

## Status against the mandatory scope

| # | Requirement | State |
| --- | --- | --- |
| 1 | Web app with agentic backend | done |
| 2 | Upload + compatibility checking | done — GeoTIFF via rasterio, co-registration IoU 0.90, GSD tolerance 0.05 |
| 3 | Single-image VQA *(mandatory)* | done |
| 4 | Captioning **or** grounding | done — **both** |
| 5 | Bi-temporal change *(mandatory)* | done |
| 6 | Optical–SAR joint *(mandatory)* | done |
| 7 | Agentic orchestration | done — six stages, enforced params, adaptive params, bounded re-planning |
| 8 | Evidence, confidence, trace, reports | done |
| 9 | Benchmark harness on prescribed splits | done — 5 configs, prescribed metrics |
| 10 | Tests and demo artefacts | done — 217 tests, 8 real scene sets |
| 11 | **Remote-sensing adaptation** *(mandatory)* | **not started** |
| 12 | Score normalisation before aggregation | not implemented |
| 13 | Cross-modal benchmark | not wired |
| 14 | VRSBench imagery | not downloaded |

Eleven of fourteen are done, and the open items are one story. The system is a
strong agentic harness running a stock model — and a stock model is the one
thing the problem statement explicitly disqualifies. Everything below exists to
close that.

---

## Both original gates are closed

**BigEarthNet.txt is available** — public, ungated, CDLA-Permissive-1.0, and the
imagery exists pre-encoded so no `rico-hdl` conversion is needed.

**Qwen3-VL trains under plain `transformers` + `peft`** — evidenced by a
third-party adapter that trained on a T4, which also gives us known-good pins.

No blocking unknown remains. The residual risk is not availability but **domain
distance**: Europe-only training data against Indian evaluation imagery, and
120×120 patches at 10 m against sub-metre Cartosat. Both are addressed by the
data mixture in `DATA.md`, not by anything that can be resolved with a decision.

---

## Phases

Effort is person-hours; cost is AWS spend. Detail lives in the linked documents.

| # | Phase | Effort | Cost | Detail |
| --- | --- | --- | --- | --- |
| 0 | Infrastructure | 4 h | $2 | `AWS.md` |
| 1 | Data preparation | 20–25 h | $5 | `DATA.md` |
| 2 | **Baseline benchmark** | 4 h | $5 | `BENCHMARKING.md` |
| 3 | Adaptation strategy | 8 h | $3 | `FINETUNING.md` |
| 4a | VLM training, stages A and B | 6 h + wall-clock | $10–20 | `FINETUNING.md` |
| 4b | CNN training | 6 h | $2 | `CNN.md` |
| 5 | Adapted benchmark | 3 h | $5 | `BENCHMARKING.md` |
| 6 | Ablation + score normalisation | 8 h | $8 | `BENCHMARKING.md` |
| 7 | Integration + serving | 10 h | $5 | `AWS.md` |
| 8 | Hidden-set robustness | 10 h | $3 | below |
| A | Application v2 | 20 h | — | below |
| **Total** | | **~100 h** | **~$50** | |

Roughly half the credit grant, leaving headroom for a reclaimed spot run, a
second training pass after ablation, and demo-day serving.

### Phase 8 — hidden-set robustness

The evaluation imagery is Cartosat-2S plus RISAT, and it will break assumptions
the code currently makes. Two need fixing with tests, not hope:

- **Band count.** Cartosat-2S is panchromatic — potentially **one band** — or
  4-band MX. The optical path assumes a 12-band Sentinel-2 stack. A 1-band PAN
  GeoTIFF must not crash `preview_bands()` or `optical_indices`; it must report
  `applicable: false` cleanly and let the run continue.
- **Index availability.** 4-band MX has NIR, so **NDWI works**. It has no SWIR,
  so **NDBI does not**. Built-up must come from SAR or the CNN.

Build a synthetic Cartosat-like fixture — 1-band and 4-band, sub-metre GSD — and
add it to the suite so this path is proven rather than assumed.

### Phase A — application v2

Independent of phases 1–6 and can run alongside. Only the `openai_compat`
backend is shared, and Phase 2 needs it.

- Full-screen Leaflet map carrying Sentinel-1 and Sentinel-2 tiles from the
  Planetary Computer `tilejson` assets already surfaced in `feed.py`.
- **The map is the input surface, not a backdrop**: draw a box, that rectangle
  becomes the analysis window via `/api/feed/load`, and evidence overlays render
  back onto the map as layers. A grounding box drawn on the map is georeferenced
  evidence; the same box in a side panel is a picture.
- Optical/SAR swipe divider rather than a toggle.
- Query as a floating widget with the trace streaming into it.
- Leaflet **vendored**, not CDN-loaded — the demo must survive bad venue wifi.
- Catalogue cut to `qwen3-vl-base` and `qwen3-vl-satquery`.

---

## Order of work

1. **Phase 0 and Phase 1 in parallel.** Infrastructure while data preparation
   runs on the 2.5 GB slice.
2. **Pull VRSBench imagery** (3.8 GB). Three of five benchmarks cannot be scored
   without it, and every later number depends on them.
3. **Wire BigEarthNet `bench` as a cross-modal config.** Cheap, and it closes a
   mandatory criterion that currently has no score at all.
4. **Phase 2 baseline.** As soon as the box serves. It de-risks the harness and
   produces the "before" column that everything afterwards is measured against.
5. **Phase 3 verification, then 4a and 4b.**
6. **Phase 5**, then **6 and 8 in parallel**, then **7**.
7. **Phase A** alongside throughout.

---

## The three things that matter most

**Run the baseline before training anything.** Without a before column,
"we fine-tuned it" is an unverifiable claim, and the base-versus-adapted
comparison is the single most important artefact the project produces.

**Get the evidence preamble into the training data.** If the adapted model never
sees the measurement preamble it will meet at inference, it either ignores the
measurements or fights them — and the evidence-grounding design that makes this
system distinctive stops working. `DATA.md` covers the mix.

**Do not skip score normalisation.** It is explicitly required, it is a small
piece of work, and marks are attached to it.

---

## Competitive note

At least a dozen forks of BigEarthNet.txt from other 2026 teams exist publicly,
and at least one working Qwen3-VL adapter has been published. The dataset is not
a differentiator.

What is: prescribed-split evaluation with prescribed metrics rather than
self-selected samples; a scored cross-modal criterion; genuine bi-temporal
capability; evidence-grounded orchestration with an auditable trace; and regional
adaptation toward the terrain the hidden set actually contains. Those are the
axes worth spending effort on.
