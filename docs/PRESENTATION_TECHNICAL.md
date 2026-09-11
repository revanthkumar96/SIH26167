# SatQuery AI — Technical Presentation Script & Reference

**Problem Statement:** SIH26167 · **Organization:** ISRO · **Hackathon:** SIH 2026  
**Duration:** ~10 minutes (~1,800 words spoken at natural pace)  
**Companion docs:** [`ARCHITECTURE.md`](ARCHITECTURE.md) · [`BENCHMARKING.md`](BENCHMARKING.md) · [`FINETUNING.md`](FINETUNING.md) · [`results/2026-09-10-stage-a/`](../results/2026-09-10-stage-a/)

---

## Timing guide

| Section | Target | Slide / demo |
| --- | --- | --- |
| 1. Introduction | 0:45 | Title + one-line value prop |
| 2. Problem statement | 1:15 | Fragmented RS AI vs agentic assistant |
| 3. Objectives & deliverables | 1:30 | Mandatory scope checklist |
| 4. Our solution | 1:30 | Measure-first architecture diagram |
| 5. Methodology & technical approach | 3:30 | Controller, tools, backends, adaptation |
| 6. Feasibility | 1:00 | Evidence table + laptop demo command |
| 7. Conclusion | 0:30 | Numbers + thank you |

---

## 1. Introduction (~45 s)

**[SLIDE: SatQuery AI — Interactive Vision-Language Assistant for Multimodal Remote Sensing]**

Good morning. We present **SatQuery AI** for **ISRO Problem Statement SIH26167**.

Remote sensing imagery supports agriculture, disaster management, urban planning, forest and water monitoring — but extracting answers still requires GIS expertise, separate pipelines for VQA, change detection, and optical–SAR fusion, and models not adapted to satellite data.

**SatQuery AI** is a **software-based agentic vision-language assistant**: users upload GeoTIFF imagery or load real Sentinel scenes, ask questions in natural language, and receive **evidence-grounded answers** with a full **execution trace** — the only surface the problem statement scores.

The delivered system is a **domain-adapted** Qwen3-VL-2B served by vLLM on an AWS G instance, fronted by the SatQuery API. Adaptation is the point: a generic VLM with no remote-sensing fine-tuning does not meet the problem statement, and the adapted model is measured against its own unadapted base on identical splits.

---

## 2. Problem statement (~1 min 15 s)

**[SLIDE: Three input configurations + disqualifier]**

The problem statement defines three **input configurations** — inferred from imagery, never declared by the user:

| Configuration | Images | Mandatory capabilities |
| --- | --- | --- |
| **Single image** | 1 optical/multispectral or SAR | VQA *(mandatory)* + captioning **or** grounding |
| **Bi-temporal pair** | 2 co-registered scenes, different dates | Change description / change-VQA *(mandatory)* |
| **Cross-modal pair** | 1 optical + 1 SAR, same geography | Joint optical–SAR reasoning *(mandatory)* |

**Formats:** GeoTIFF/TIFF for operational imagery; PNG/JPEG only for prescribed public benchmarks.

**Adaptation is mandatory.** A generic LLM/VLM without remote-sensing fine-tuning or domain adaptation **does not satisfy** the requirements. **BigEarthNet.txt** is the named adaptation dataset; **VRSBench**, **RSVQA**, and **CDVQA** are the prescribed evaluation benchmarks.

**Agentic orchestration is mandatory:** select specialist tools, enforce permitted parameters, combine outputs, attach evidence and confidence, return an auditable trace. Internal chain-of-thought is **not** evaluated.

**Representative queries** (all implemented and routable):

- Describe the land cover and major objects visible in this image.
- Highlight the water body referred to in the query.
- What changed between these two dates, and where did the change occur?
- Use the optical and SAR images together to identify built-up and water-covered regions.
- Has the built-up area increased, decreased, or remained unchanged?

---

## 3. Objectives and deliverables (~1 min 30 s)

**[SLIDE: Deliverables mapped to `docs/ROADMAP.md` status table]**

| # | Requirement | Deliverable | Status |
| --- | --- | --- | --- |
| 1 | Web app + agentic backend | FastAPI + static UI, four tabs | ✅ Done |
| 2 | Upload + compatibility | Rasterio GeoTIFF, IoU ≥ 0.90, GSD tol 0.05 | ✅ Done |
| 3 | Single-image VQA | `vlm_vqa` | ✅ Done |
| 4 | Captioning **or** grounding | **Both** `vlm_caption` + `vlm_grounding` | ✅ Done |
| 5 | Bi-temporal change | `change_mask` → change VLM tools | ✅ Done |
| 6 | Optical–SAR joint | `optical_indices` + `sar_indices` → `vlm_crossmodal_vqa` | ✅ Done |
| 7 | Agentic orchestration | Six-stage controller, registry, re-planning | ✅ Done |
| 8 | Evidence, trace, reports | WebSocket trace, mask overlays, JSON export | ✅ Done |
| 9 | Benchmark harness | 6 YAML configs, prescribed metrics, aggregation | ✅ Done |
| 10 | Tests + demo artefacts | **509 tests**, 10 bundled scene sets, 15+ GeoTIFFs | ✅ Done |
| 11 | **Remote-sensing adaptation** | Stage A LoRA on BigEarthNet.txt — **measured** | ✅ Done |
| 12 | Score normalisation | `eval/aggregate.py`, `satquery bench score` | ✅ Done |
| 13 | Cross-modal benchmark | `bigearthnet_bench.yaml` | ✅ Done |
| 14 | Stage B instruction tuning | VRSBench + RSVQA + CDVQA train mixture | 🔄 In progress |
| 15 | Land-cover CNN | Code integrated; training pending | ⏳ Planned |

**Artifacts judges can inspect:**

- Live app: `satquery serve` → `http://127.0.0.1:8000`
- Benchmark results: `results/2026-09-10-stage-a/`
- Configs: `configs/bench/*.yaml`
- Demo scenes: `runs/samples/` (Mumbai, Ujani, Sundarbans, Delhi, Chennai, …)

---

## 4. Our solution (~1 min 30 s)

**[SLIDE: `docs/ARCHITECTURE.md` — measure first, describe second]**

### Design principle

A general VLM produces fluent but unreliable statements about water extent or urban growth — it cannot **measure**. NDWI, NDBI, and Otsu change masks **measure** reliably but cannot answer in English.

```
       images ──▶ specialists ──▶ measurements ──┐
                                                 ├──▶ VLM ──▶ answer + trace
       query  ─────────────────────────────────  ┘
```

Measurements are injected into the VLM prompt as an **evidence preamble** the model is told not to contradict. Answers are grounded in pixel-derived numbers, not impressions.

### User workflow

1. **Analyse** — upload 1–2 GeoTIFFs or load a bundled scene; configuration inferred automatically.
2. **Live feed** — search Sentinel-1/2 via Microsoft Planetary Computer; load onto shared UTM grid.
3. **Benchmarks** — run model×benchmark matrix with resume and skip-existing cells.
4. **Registry** — inspect all twelve tools, permitted parameters, and constraints.

### Deployment tiers

| Tier | Stack | Use case |
| --- | --- | --- |
| **GPU serve (delivered)** | vLLM on EC2 `g6.xlarge` (L4 22.9 GB, bf16), SatQuery API in front | The live endpoint; `infra/serve-model.sh` |
| **HF inference** | PyTorch + Transformers (`pip install -e ".[hf]"`) | Correctness reference and the backend both benchmark sweeps ran through |
| **Adapted weights** | Merged LoRA on Qwen3-VL-2B, one set of weights | What is scored and what is published |

The delivered system serves the **merged** model — base plus both adaptation
stages folded in — so loading it is an ordinary `from_pretrained`, with no
adapter juggling at inference and no way for the served weights to drift from
the benchmarked ones.

---

## 5. Methodology and technical approach (~3 min 30 s)

**[SLIDE: Six stages — `src/satquery/agent/controller.py`]**

### 5.1 Six-stage controller

Matches the six controller duties in the problem statement:

| Stage | Module | What it does |
| --- | --- | --- |
| **1. Check** | `geo/raster.py` | Band count, dtype, CRS, GSD; pair IoU ≥ 0.90, GSD agreement within 0.05 |
| **2. Route** | `agent/router.py` | Rule-based task classification — **not** an LLM call (auditable, instant) |
| **3. Select** | `agent/registry.py` + `agent/planner.py` | Tool lookup for `(task, input_config)`; adaptive params; `allowed_params` enforced |
| **4. Execute** | `agent/controller.py` | Specialists first; bounded re-planning (`MAX_REVISIONS = 2`) |
| **5. Fuse** | Controller + `eval/prompts.py` | Evidence preamble from specialist outputs |
| **6. Respond** | API + WebSocket | Answer, confidence, downloadable JSON trace |

**Input configuration inference** (`schema.InputConfig`):

- 1 image → `SINGLE`
- 2 images, same modality → `BITEMPORAL_PAIR`
- 2 images, different modality (optical vs SAR hints in path/bands) → `CROSSMODAL_PAIR`

**Routing logic** (`agent/router.py`) — configuration constrains task before keywords:

```python
# Cross-modal: only one task exists for optical+SAR pairs
if input_config is CROSSMODAL_PAIR:
    return CROSSMODAL_VQA

# Bi-temporal: change caption vs change VQA by question patterns
if input_config is BITEMPORAL_PAIR:
    if matches(CHANGE_CAPTION_PATTERNS) and not matches(QUESTION_PATTERNS):
        return CHANGE_CAPTION
    return CHANGE_VQA

# Single: grounding → caption → VQA by regex priority
```

Six tasks: `VQA`, `CAPTION`, `GROUNDING`, `CHANGE_VQA`, `CHANGE_CAPTION`, `CROSSMODAL_VQA`.

**Precursor map** — specialists always run before the VLM on multi-step tasks:

```python
_PRECURSORS = {
    Task.CHANGE_VQA: ("change_mask",),
    Task.CHANGE_CAPTION: ("change_mask",),
    Task.CROSSMODAL_VQA: ("optical_indices", "sar_indices"),
}
```

### 5.2 Twelve tools (registry contract)

Every tool declares a `ToolSpec` in `agent/registry.py`:

| Field | Purpose |
| --- | --- |
| `name`, `version` | Identity in trace |
| `accepts` | Valid `InputConfig` |
| `tasks` | Routed tasks served |
| `allowed_params` | **Enforced** min–max; `ParameterError` if violated |
| `kind` | `measurement` (deterministic) or `model` (VLM) |
| `emits_evidence` | Contributes visual artefacts |

#### Measurement tools (deterministic, fast, auditable)

| Tool | File | Inputs | Key outputs |
| --- | --- | --- | --- |
| `change_mask` | `agent/tools/change.py` | Bi-temporal | `changed_area_frac`, `change_location` (quadrant), `direction`, `mask_uri` |
| `optical_indices` | `agent/tools/indices.py` | Single/cross-modal optical | NDVI, NDWI, NDBI, band stats (12-band BigEarthNet order) |
| `sar_indices` | `agent/tools/indices.py` | Cross-modal SAR | VV/VH stats; Sentinel-1 already in dB |

`change_mask` uses absolute difference + **Otsu threshold** (baseline). A Siamese CNN on LEVIR-CD is planned behind the same `ToolSpec` with a version bump — no upstream changes.

#### Model tools (VLM-backed)

| Tool | Task(s) |
| --- | --- |
| `vlm_vqa` | Single-image VQA |
| `vlm_caption` | Captioning |
| `vlm_grounding` | Text-guided grounding (0–1000 box grid) |
| `vlm_change_vqa` | Change VQA (after `change_mask`) |
| `vlm_change_caption` | Change description (after `change_mask`) |
| `vlm_crossmodal_vqa` | Optical–SAR joint (after index tools) |

**Re-planning:** if `changed_area_frac` is below quiet threshold, planner retries `change_mask` at lower `min_area_frac` (bounded).

### 5.3 Backend architecture (code decision)

`eval/backends/__init__.py` — deferred imports so base install needs no torch:

| Backend | When used | Key file |
| --- | --- | --- |
| `hf` | **Benchmark sweeps and the correctness reference** — every number in this deck | `eval/backends/hf.py` |
| `openai_compat` | The delivered endpoint: anything speaking the OpenAI API, vLLM included | `eval/backends/openai_compat.py` |
| `vllm` | In-process vLLM, optional `[vllm]` extra | `eval/backends/vllm_backend.py` |
| `echo` | CI and tests only | deterministic stub |

The application talks to `openai_compat` over HTTP and does not know what is
behind it. That is what let the same code path serve a local engine during
development and vLLM on a G instance in delivery, with no application change --
only a different `SATQUERY_VLM_BASE_URL`.

**GeoTIFF path:** `pip install -e ".[geo]"` -> Rasterio in `geo/raster.py`; PIL
fallback for benchmark PNGs.

**Lazy backend load** (`api/app.py`): the server boots before the model is reachable; the first query constructs the `Controller`, and a backend that is not up yet returns **503** with an actionable message rather than a stack trace. This matters on the G instance, where vLLM needs a minute to load 4.3 GB of weights after the API is already accepting connections.

**GeoTIFF path:** `pip install -e ".[geo]"` → Rasterio in `geo/raster.py`; PIL fallback for benchmark PNGs.

### 5.4 Data pipeline and adaptation

#### Adaptation datasets

| Role | Dataset | Details |
| --- | --- | --- |
| **Stage A** (domain) | BigEarthNet.txt | 464,044 S1+S2 pairs, 9.5M annotations; CDLA-Permissive-1.0; Hub `BIFOLD-BigEarthNetv2-0/BigEarthNet.txt` |
| **Stage B** (instruction) | VRSBench + RSVQA + CDVQA **train** + Indian Sentinel (Programme C) | Teaches exact-match answers, box format, caption style |
| **CNN labels** | BigEarthNet v2.0 LMDB | CORINE multi-label, 120×120 px @ 10 m |
| **Demo imagery** | Planetary Computer STAC | `scripts/fetch_sentinel_samples.py` — 15 GeoTIFFs, shared UTM grid per AOI |

#### Adaptation configuration (both stages measured)

| Knob | Value | Rationale |
| --- | --- | --- |
| Base model | `Qwen/Qwen3-VL-2B-Instruct` @ pinned revision `89644892…` | Problem-statement-scale VLM; pinned so a comparison stays a comparison |
| Method | **LoRA** via `transformers` + `peft`, base in **bf16** | Not QLoRA: the base is not quantised. No LLaMA-Factory dependency |
| Rank / alpha | 32 / 64 | Headroom for domain shift |
| Targets | `q,k,v,o,gate,up,down` + **projector** + **vision encoder** | SAR backscatter resembles nothing in natural-image pretraining, so language-only adaptation would not be adaptation |
| Trainable params | 49,364,992 (**2.27%**) | |
| LR / schedule | 1e-4 cosine, 3% warmup | |
| Precision | bf16 on **Ada Lovelace** (RTX 4080 SUPER 32 GB) | |
| Toolchain | `transformers==4.57.1`, `peft==0.17.1`, `accelerate==1.7.0` | Pinned: upstream is at 5.x, and an unpinned install silently changes the model implementation under the comparison |

**Stage A — domain and cross-modal** (2026-09-10). 98,925 BigEarthNet records
over 69,561 patch pairs, 1.0 epoch, ~38% carrying an evidence preamble, ~$2.25.
Teaches SAR vocabulary and optical–SAR joint reasoning.

**Stage B — task formats** (2026-09-11). 64,000 records, **2.0 epochs
(2000/2000 steps)**, ~$1.30 across two rentals. Resumed **from the Stage A
adapter**, so the two stages are one set of weights with one lineage rather
than two models. Composition:

| Source | Records | Why it is in the mixture |
| --- | --- | --- |
| VRSBench train (caption / VQA / referring) | 34,000 | The output formats the metrics reward |
| RSVQA LR train | 10,000 | Templated VQA; capped, or it becomes the corpus |
| **BigEarthNet rehearsal slice** | 20,000 | The only optical–SAR pairs and the only evidence preambles — without it Stage B trains away what Stage A earned |

Evidence preambles on **22.5%** of records, from a slice re-prepared at
`--preamble-rate 0.85`. Training was split across two rentals and resumed with
`--resume-from-checkpoint`, which restores step count, optimizer state and LR
schedule position — so the cosine schedule completed rather than being truncated
by a time budget as Stage A's was.

#### Train/serve parity (three channels)

1. **Image normalisation** — shared `stretch_to_uint8` for train and serve.
2. **Prompt format** — one `build_prompt()`, 0–1000 boxes, versioned `PROMPT_VERSION`.
3. **Evidence preamble** — the same `format_evidence()` at training and inference.

Each is a place where a silent mismatch costs accuracy with nothing in the logs.

#### Contamination guard

Stage B trains on the **train** splits of the same benchmarks it is scored on.
A single leaked row would make the delta meaningless while every other signal —
loss, sweep, score — looked *better*. `satquery data check-contamination` fails
the build on image reuse and, since a guard that cannot read its splits has not
cleared anything, **exits non-zero when a split is unreadable** rather than
passing silently.

Verified on the training box before the run:

    clean: 64,000 training records checked against 11,133 benchmark images,
    no image reuse

Question-only matches (617) are reported but not counted against the corpus:
all three benchmarks are templated, so identical phrasings recur across
unrelated scenes. BigEarthNet train and bench were separately confirmed
disjoint — 69,051 vs 978 patches, zero shared images.

### 5.5 Benchmark harness

**Entry point:** `satquery bench run --config configs/bench/*.yaml`  
**Runner:** `eval/runner.py` → `predictions.jsonl` + `metrics.json` per cell  
**Matrix:** model-outer, dataset-inner; resume on crash; `limit=200`, `seed=1234` for headline numbers  
**Aggregation:** `satquery bench score` → `eval/aggregate.py`

See **Appendix A** for full benchmark and metric tables.

---

## 6. Feasibility (~1 min)

**[SLIDE: Four pillars with evidence]**

| Challenge | Evidence in project |
| --- | --- |
| **Accuracy** | Measure-then-describe; overall **0.0496 → 0.2256** against the unadapted base on identical splits; **509 tests**; prescribed benchmark protocol |
| **Cost** | Whole programme under $15: Stage A ~$2.25, Stage B ~$1.30, data staging ~$0.30, serving $0.805/hr with a request-based idle timer |
| **Data** | BigEarthNet.txt public/ungated; 6 benchmark configs; train/test contamination checked and clean; Indian AOIs via Planetary Computer |
| **Compute** | 2B model fits an L4/A10G for serving; LoRA over 2.27% of params, not a full fine-tune; both stages trained on one rented RTX 4080S |
| **Security** | `SATQUERY_MAX_UPLOAD_MB`; no secrets in repo; security group admits one address; scoped disposable IAM for rented hosts; trace as audit surface |

**Reproducing the delivered system:**

```bash
# 1. the corpus, contamination-checked against every benchmark test split
satquery data instruct --config "configs/train/*.yaml"     --test-config "configs/bench/*.yaml"     --include bigearthnet=data/prepared/train/train.jsonl     --out data/prepared/stage-b/train.jsonl

# 2. adaptation, resumed from Stage A so it is one lineage
python scripts/train_lora.py --stage b --resume runs/adapters/stage-a     --data data/prepared/stage-b/train.jsonl --image-root data

# 3. the comparison the claim rests on: base and adapted, one session, one harness
satquery bench run --config "configs/bench/*.yaml" --backend hf     --model Qwen/Qwen3-VL-2B-Instruct --limit 200 --seed 1234
satquery bench run --config "configs/bench/*.yaml" --backend hf     --model runs/merged-b --limit 200 --seed 1234
satquery bench score --results runs/results.csv --baseline Qwen/Qwen3-VL-2B-Instruct

# 4. serve it
./infra/serve-model.sh --apply
```

---

## 7. Conclusion (~30 s)

**[SLIDE: Summary numbers]**

SatQuery AI delivers SIH26167's mandatory scope as **working software** with **measured adaptation**:

- All three input configurations + agentic orchestration + evidence trace  
- Stage A: overall **+0.076** normalised; single-image VQA **+0.183**; cross-modal **0 → 0.175**  
- 509 tests, 6 benchmarks, 10 demo scenes, real Sentinel data  

We give planners and researchers a **first pass** that is fast, traceable, and grounded in pixel measurements — not a replacement for GIS analysts, but a door that did not exist before.

Thank you. Questions welcome — we can run a live query on Mumbai or Ujani.

---

# Appendix A — Benchmarks and metrics (full reference)

## A.1 Judging criteria → benchmarks

| Criterion | Mandatory? | Benchmark config | Dataset | Task | Headline metric |
| --- | --- | --- | --- | --- | --- |
| **Single-image VQA** | ✅ Yes | `rsvqa_lr.yaml` | RSVQA-LR test | `vqa` | OA (+ AA reported) |
| **Single-image VQA** | ✅ Yes | `vrsbench_vqa.yaml` | VRSBench VQA eval | `vqa` | OA |
| **Captioning** | No (or grounding) | `vrsbench_caption.yaml` | VRSBench Cap eval | `caption` | **CIDEr-D** (headline) |
| **Grounding** | No (or captioning) | `vrsbench_referring.yaml` | VRSBench referring eval | `grounding` | **Acc@IoU 0.5** (headline) |
| **Change understanding** | ✅ Yes | `cdvqa.yaml` | CDVQA official test | `change_vqa` | OA + **AA** |
| **Optical–SAR joint** | ✅ Yes | `bigearthnet_bench.yaml` | BigEarthNet.txt `bench` split | `crossmodal_vqa` | OA (binary+mcq types) |

**Agentic orchestration** and **system completeness** are evaluated from the **execution trace** and demonstration checklist — not a single accuracy number.

## A.2 Per-benchmark details

### `rsvqa_lr` — RSVQA low-resolution

| Field | Value |
| --- | --- |
| Imagery | Sentinel-2 GeoTIFF, `data/RSVQA/LR/Images_LR/*.tif` |
| Annotations | `LR_split_test_questions.json` + `LR_split_test_answers.json` |
| Full test size | 10,004 questions |
| Question types | presence, comp, count, rural_urban |
| **Metrics** | OA, AA (per-type accuracy — **report both**; distribution skewed) |

### `vrsbench_vqa`

| Field | Value |
| --- | --- |
| Imagery | `data/VRSBench/Images_val/` (~0.3 m resolution) |
| Annotations | `VRSBench_EVAL_vqa.json` |
| Full test size | 37,409 questions |
| **Metrics** | OA |

### `vrsbench_caption`

| Field | Value |
| --- | --- |
| Annotations | `VRSBench_EVAL_Cap.json` |
| Full test size | 9,350 captions |
| **Metrics** | BLEU-1–4, ROUGE-L (β=1.2), METEOR (if NLTK available), **CIDEr-D** (σ=6.0, n=1..4) |
| Note | CIDEr-D is corpus-level; keep `limit` identical across models |

### `vrsbench_referring` (grounding)

| Field | Value |
| --- | --- |
| Annotations | `VRSBench_EVAL_referring.json` |
| Full test size | 16,159 referring expressions |
| Box format | xyxy, pixel scale; model prompted for **0–1000 milli grid** |
| **Metrics** | Acc@IoU 0.25, **Acc@IoU 0.5** (headline), mIoU, parse_rate |
| Note | Unparseable box → IoU 0 (not dropped) |

### `cdvqa` — change VQA (only bi-temporal benchmark)

| Field | Value |
| --- | --- |
| Imagery | PNG pairs `im1/` + `im2/` (SECOND-based) |
| Annotations | `cdvqa_test.json` |
| Full test size | 200 (official test in our pull) |
| **Metrics** | OA, **AA** across 8 official question types |
| Note | Only public benchmark that sends **two images** to the model |

### `bigearthnet_bench` — cross-modal (we added — closes scoring gap)

| Field | Value |
| --- | --- |
| Source | BigEarthNet.txt **`bench`** split — 15,029 annotations, 1,082 S1+S2 pairs, manually verified |
| Prepared data | `data/prepared/bench/` via `scripts/prepare_bigearthnet.py --split bench` |
| Task types scored | `binary`, `mcq` only (not caption/bbox rows) |
| **Metrics** | OA |
| Caveat | Sentinel 10 m Europe — measures cross-modal **reasoning**, not Indian Cartosat domain |

## A.3 Metric implementations (`src/satquery/eval/metrics/`)

| Module | Metrics | Notes |
| --- | --- | --- |
| `vqa.py` | OA, AA | Exact match after `normalize_answer()` — lowercase, strip `./-%` at token edges |
| `caption.py` | BLEU, ROUGE-L, METEOR, CIDEr-D | CIDEr-D scaled ×10; METEOR absent if NLTK missing |
| `grounding.py` | IoU, Acc@0.25, Acc@0.5, mIoU, parse_rate | Unit-normalised xyxy |
| `change_vqa` | Uses VQA metrics on CDVQA answers | |

## A.4 Score aggregation (`eval/aggregate.py`)

**Problem statement requirement:** normalise before aggregating.

| Step | Rule |
| --- | --- |
| 1. Normalise each raw metric | Against **declared theoretical range** (not sweep min–max) |
| 2. CIDEr-D range | [0, 10]; all other headline metrics [0, 1] |
| 3. Within criterion | Average benchmarks/tasks in same criterion |
| 4. Overall | Average across criteria |

**Criterion mapping (`TASK_CRITERION`):**

| Task | Criterion key |
| --- | --- |
| `vqa` | `single_image_vqa` |
| `caption` | `captioning` |
| `grounding` | `grounding` |
| `change_vqa`, `change_caption` | `change_understanding` |
| `crossmodal_vqa` | `crossmodal` |

**CLI:** `satquery bench score --results runs/results.csv`

---

# Appendix B — Measured results

**Provenance:** `results/2026-09-10-stage-a/` and `results/2026-09-11-stage-b/` ·
Harness: `hf` backend for every row · n=200 · seed=1234 · prompt 1.1.0 · the
baseline re-measured in the same session as each adapted row, so no comparison
crosses a code change.

## B.0 Delivered result — base vs the adapted model

| Criterion | Base | **Delivered (Stage B)** | Delta | *Stage A* |
| --- | --- | --- | --- | --- |
| Single-image VQA | 0.2250 | **0.6675** | **+0.4425** | *+0.1825* |
| Change understanding | 0.0100 | **0.3000** | **+0.2900** | *+0.0350* |
| Optical–SAR joint | 0.0000 | **0.1600** | **+0.1600** | *+0.1750* |
| Grounding | 0.0000 | 0.0000 | 0.0000 | *0.0000* |
| Captioning | 0.0128 | 0.0006 | **−0.0123** | *−0.0126* |
| **Overall** | **0.0496** | **0.2256** | **+0.1761** | *+0.0760* |

Raw: `rsvqa_lr` OA 0.250 → **0.725** · `vrsbench_vqa` OA 0.200 → **0.610** ·
`cdvqa` OA 0.010 → **0.300** · `bigearthnet_bench` OA 0.000 → **0.160** ·
`vrsbench_caption` CIDEr-D 0.128 → 0.006.

**Answering behaviour** (the caveat Stage A carried, now closed):

| Empty predictions | Base | Stage B |
| --- | --- | --- |
| `cdvqa` | 192/200 | **0** |
| `rsvqa_lr` | 108/200 | **0** |
| `vrsbench_vqa` | 72/200 | **0** |
| `vrsbench_referring` unparsed boxes | 33/200 | **1** |

## B.1 Stage A normalised criteria (2026-09-10)

| Criterion | Base | Adapted | Delta |
| --- | --- | --- | --- |
| Single-image VQA | 0.2250 | **0.4075** | **+0.1825** |
| Optical–SAR joint | 0.0000 | **0.1750** | **+0.1750** |
| Change understanding | 0.0100 | 0.0450 | +0.0350 |
| Grounding | 0.0000 | 0.0000 | 0.0000 |
| Captioning | 0.0128 | 0.0003 | −0.0126 |
| **Overall** | **0.0496** | **0.1256** | **+0.0760** |

## B.2 Raw per-benchmark headline metrics

| Benchmark | Metric | Base raw | Adapted raw |
| --- | --- | --- | --- |
| `rsvqa_lr` | OA | 0.250 | **0.540** |
| `vrsbench_vqa` | OA | 0.200 | **0.275** |
| `cdvqa` | OA | 0.010 | **0.045** |
| `bigearthnet_bench` | OA | 0.000 | **0.175** |
| `vrsbench_caption` | CIDEr-D | 0.128 | 0.003 |
| `vrsbench_referring` | Acc@0.5 | 0.000 | 0.000 |

## B.3 Quality diagnostics (`quality.txt`)

Empty predictions (base → adapted):

| Benchmark | Base empty / 200 | Adapted empty / 200 |
| --- | --- | --- |
| `rsvqa_lr` | 108 | **0** |
| `vrsbench_vqa` | 72 | 34 |
| `cdvqa` | 192 | 161 |
| `bigearthnet_bench` | 0 | 134 |
| `vrsbench_referring` | 33 (unparsed boxes) | 0 |

**Interpretation.** Stage A recovered **answering behaviour** rather than pure
reasoning — much of its RSVQA gain was silence turning into a response, which is
a real improvement but a different claim. Stage B closed that gap completely:
empty predictions went 192/200 → **0** on CDVQA, 108 → **0** on RSVQA, 72 → **0**
on VRSBench VQA, so every accuracy number now sits on a model that answers
everything.

Cross-modal **survived** at 0.160 against Stage A's 0.175 after training on four
new corpora — the rehearsal slice doing exactly what it was included for.

**Captioning did not get fixed, and the deck says so.** CIDEr-D 0.128 → 0.006,
mean length 66.8 → **110.9 words** against a 47.4-word reference. The cause was a
corpus error: caption/reference alignment was validated on the VRSBench train
split alone (median 52, close to the reference) without re-checking what the
BigEarthNet rehearsal slice contributed to the caption distribution — its
captions run to a median of 96 words. Re-running the benchmark with the token
budget raised 128 → 320 gave CIDEr-D **0.000**, *worse*, which ruled out
truncation and proved the fault is length. `DEFAULT_SLICE_DROP_TASKS` now drops
those records; the rebuilt corpus has caption median 52 and p90 77 against 57 and
112, while keeping all 20,000 optical–SAR records. **Not yet trained on.**

**Grounding** is still 0.000, but the format problem is solved — unparsed boxes
fell 33 → 1. The model emits well-formed boxes on the right grid, in the wrong
places: localisation failure, not a scoring artefact.

---

# Appendix C — Architecture in detail

Everything below is read off the running system: `default_registry()` builds
twelve tools, `_EVIDENCE_LABELS` declares nine measurement keys, and the
controller's six duties are six methods in `agent/controller.py`.

## C.1 The six controller stages

| # | Stage | Method | What it decides |
| --- | --- | --- | --- |
| 1 | **check** | `check_inputs()` | How many images, what modality, what `InputConfig`. Rejects impossible sets before any model loads |
| 2 | **route** | `plan()` → `router.route()` | Which of six `Task` values this query is, from the input configuration and the question text |
| 3 | **select** | `plan()` → registry | Which tools serve that task, filtered by `accepts` and `tasks` in each `ToolSpec` |
| 4 | **execute** | `run()` | Specialists first, VLM second — the ordering the whole design rests on |
| 5 | **fuse** | `run()` | Measurements become the evidence preamble the VLM is told not to contradict |
| 6 | **report** | `run()` | Answer plus a trace naming every tool, parameter and artefact |

`InputConfig` is **derived, never declared**: one image is `SINGLE`, two of the
same modality is `BITEMPORAL_PAIR`, optical + SAR is `CROSSMODAL_PAIR`. A user
cannot mis-declare their upload into the wrong pipeline.

## C.2 The twelve tools

**Three deterministic specialists** — no model, auditable, fast:

| Tool | Inputs | Measures |
| --- | --- | --- |
| `optical_indices` | optical / multispectral | NDWI water fraction, NDBI built-up fraction, NDVI, per-band statistics (12-band BigEarthNet order) |
| `sar_indices` | SAR | VV/VH statistics, water fraction from backscatter, location of brightest returns. Sentinel-1 arrives already in dB |
| `change_mask` | bi-temporal pair | Changed-area fraction, change quadrant, direction, and a rendered mask artefact |

**Three land-cover CNN tools** — one per input configuration
(`landcover_cnn`, `landcover_cnn_bitemporal`, `landcover_cnn_crossmodal`), so a
multi-label classifier contributes to single, change and cross-modal paths
without the caller special-casing any of them.

**Six VLM tools** — one per routed task: `vlm_vqa`, `vlm_caption`,
`vlm_grounding`, `vlm_change_vqa`, `vlm_change_caption`, `vlm_crossmodal_vqa`.
One tool per task rather than one tool with a mode flag, so each carries its own
prompt, token budget and `ToolSpec`.

Every tool declares a `ToolSpec`: `name`, `version`, `accepts` (valid
`InputConfig`), `tasks`, `allowed_params` with **enforced** min–max bounds
(`ParameterError` on violation), `kind` (`measurement` or `model`), and
`emits_evidence`. The registry is the contract; a tool that does not fit it
cannot be routed to.

## C.3 The nine measurements that reach the prompt

`format_evidence()` renders these, and only these, keyed exactly as
`data/evidence.py` writes them during corpus preparation — the same function on
both sides, because a key renamed in one place and not the other produces a
preamble with a line silently missing:

| Key | Meaning | Source |
| --- | --- | --- |
| `optical_water_fraction` | water fraction from NDWI | `optical_indices` |
| `optical_builtup_fraction` | built-up fraction from NDBI | `optical_indices` |
| `sar_water_fraction` | water fraction from SAR backscatter | `sar_indices` |
| `sar_water_location` | where that water is | `sar_indices` |
| `sar_builtup_location` | quadrant of brightest SAR returns | `sar_indices` |
| `changed_area_frac` | changed fraction of the scene | `change_mask` |
| `change_location` | quadrant of the change | `change_mask` |
| `direction` | direction of change | `change_mask` |
| `landcover_classes` | classes present | land-cover CNN |

## C.4 Why the evidence preamble is trained, not just prompted

The controller prepends measurements at inference on **every** query. A model
that first meets that format at inference has never seen it, and the preamble
reaches the prompt to no effect — or worse, the model argues with it.

So the preamble is part of the **training corpus**, at 22.5% of Stage B records,
and `choose_evidence()` enforces one safety property: it never emits a preamble
that disagrees with the record's own gold answer. Where agreement cannot be
established it emits nothing. Records whose measurements are irrelevant to the
question are deliberately included too — at inference the specialists run
regardless of what was asked, so the model must meet irrelevant evidence in
training or it learns that a preamble always answers the question.

## C.5 What is measured versus what is inferred

| Claim in an answer | Where it comes from |
| --- | --- |
| "water covers 31% of the scene" | NDWI, computed from NIR and green |
| "built-up in the north-east" | NDBI + quadrant, computed |
| "12% of the area changed" | Otsu-thresholded difference, computed |
| "this looks like a harbour with moored vessels" | the VLM, from pixels |
| "the change is consistent with construction" | the VLM, reasoning over the measurement |

The split is the design. Numbers are never generated by the model; descriptions
are never invented from numbers alone. The trace records which is which, so an
answer can be audited rather than trusted.

---

# Appendix D — Technology stack (validated)

| Layer | Technologies | Notes |
| --- | --- | --- |
| Language | Python **3.11+** | `pyproject.toml` |
| API | **FastAPI**, **Uvicorn**, **Pydantic** | Base deps |
| Arrays / imaging | **NumPy**, **Pillow** | Base; **Rasterio** optional `[geo]` |
| Inference (serving) | **vLLM** behind `openai_compat` | The delivered endpoint on `g6.xlarge` |
| Inference (reference) | **PyTorch**, **Transformers**, **Accelerate** | `[hf]` extra |
| Inference (reference) | **PyTorch**, **Transformers** via the `hf` backend | Every benchmark number in this deck |
| Fine-tuning | **PyTorch**, **peft**, **transformers** | `scripts/train_lora.py` |
| CNN | **PyTorch**, **torchvision** ResNet-50 | `src/satquery/cnn/` |
| Data hub | **huggingface-hub**, **requests** | Downloads, STAC |
| Satellite catalog | **Microsoft Planetary Computer** | `feed.py`, fetch script |
| Storage | Local `runs/` workspace; S3 on AWS | GeoTIFF uploads, models, results |

---

# Appendix E — Code map (quick reference)

| Concern | Path |
| --- | --- |
| Controller | `src/satquery/agent/controller.py` |
| Router | `src/satquery/agent/router.py` |
| Tool registry | `src/satquery/agent/registry.py` |
| Change detection | `src/satquery/agent/tools/change.py` |
| Optical/SAR indices | `src/satquery/agent/tools/indices.py` |
| GeoTIFF I/O | `src/satquery/geo/raster.py` |
| API + samples | `src/satquery/api/app.py` |
| Benchmark runner | `src/satquery/eval/runner.py` |
| Metrics | `src/satquery/eval/metrics/` |
| Aggregation | `src/satquery/eval/aggregate.py` |
| Prompts | `src/satquery/eval/prompts.py` |
| LoRA training | `scripts/train_lora.py` |
| Bench configs | `configs/bench/*.yaml` |
| Stage A results | `results/2026-09-10-stage-a/` |

---

# Appendix F — External references

| Resource | URL | Role in project |
| --- | --- | --- |
| BigEarthNet.txt paper | https://arxiv.org/abs/2603.29630 | Stage A adaptation corpus |
| VRSBench | https://vrsbench.github.io/ | VQA, caption, grounding benchmarks |
| RSVQA | https://rsvqa.sylvainlobry.com/ | Low-res Sentinel VQA |
| CDVQA | https://github.com/YZHJessica/CDVQA | Bi-temporal change VQA |
| Qwen3-VL-2B | https://huggingface.co/Qwen/Qwen3-VL-2B-Instruct | Base + adapted VLM |
| Planetary Computer | https://planetarycomputer.microsoft.com/ | Live feed + demo GeoTIFF fetch |

---

*Document version: 2026-09-11. Update Stage B rows when `results/` for Stage B is committed.*
