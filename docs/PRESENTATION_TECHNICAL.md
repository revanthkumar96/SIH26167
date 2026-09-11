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

The system runs on a **student laptop** via Ollama (quantised Qwen3-VL-2B, 1.9 GB) and scales to **GPU fine-tuning and serving** on AWS when adaptation is required.

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
| 10 | Tests + demo artefacts | **359 tests**, 10 bundled scene sets, 15+ GeoTIFFs | ✅ Done |
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
4. **Registry** — inspect nine tools, permitted parameters, and constraints.

### Deployment tiers

| Tier | Stack | Use case |
| --- | --- | --- |
| **Laptop demo** | Ollama + `qwen3-vl:2b-instruct` (4-bit, ~1.9 GB) | Hackathon demo, no GPU |
| **HF inference** | PyTorch + Transformers (`pip install -e ".[hf]"`) | Correctness reference, CPU/GPU |
| **GPU serve** | vLLM on EC2 `g5.xlarge` (A10G, bf16) | Adapted model throughput |
| **Adapted model** | Merged LoRA on Qwen3-VL-2B | Scored evaluation |

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

### 5.2 Nine tools (registry contract)

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
| `ollama` | **Default** laptop demo | `eval/backends/ollama.py` |
| `hf` | Transformers inference, **benchmark sweeps** | `eval/backends/hf.py` |
| `vllm` | EC2 throughput serving | optional `[vllm]` |
| `echo` | CI/tests only | deterministic stub |

**Ollama path:** `POST /api/chat` with base64 JPEG images (via `eval/images.load_image` — handles 12-band uint16 GeoTIFF). Default timeout 600 s (`SATQUERY_OLLAMA_TIMEOUT`).

**Lazy backend load** (`api/app.py`): server boots without GPU/Ollama; first query constructs `Controller`; failures return **503** with actionable message.

**GeoTIFF path:** `pip install -e ".[geo]"` → Rasterio in `geo/raster.py`; PIL fallback for benchmark PNGs.

### 5.4 Data pipeline and adaptation

#### Adaptation datasets

| Role | Dataset | Details |
| --- | --- | --- |
| **Stage A** (domain) | BigEarthNet.txt | 464,044 S1+S2 pairs, 9.5M annotations; CDLA-Permissive-1.0; Hub `BIFOLD-BigEarthNetv2-0/BigEarthNet.txt` |
| **Stage B** (instruction) | VRSBench + RSVQA + CDVQA **train** + Indian Sentinel (Programme C) | Teaches exact-match answers, box format, caption style |
| **CNN labels** | BigEarthNet v2.0 LMDB | CORINE multi-label, 120×120 px @ 10 m |
| **Demo imagery** | Planetary Computer STAC | `scripts/fetch_sentinel_samples.py` — 15 GeoTIFFs, shared UTM grid per AOI |

#### Stage A configuration (measured 2026-09-10)

| Knob | Value | Rationale |
| --- | --- | --- |
| Base model | `Qwen/Qwen3-VL-2B-Instruct` @ pinned revision `89644892…` | Problem-statement-scale VLM |
| Method | QLoRA / LoRA via `transformers` + `peft` | No LLaMA-Factory dependency |
| Rank / alpha | 32 / 64 | Headroom for domain shift |
| Targets | `q,k,v,o,gate,up,down` + **projector** + **vision encoder** | SAR/multispectral appearance, not language-only |
| Trainable params | 49,364,992 (2.27%) | |
| LR / schedule | 1e-4 cosine, 3% warmup | Above competitor's 1e-5 |
| Precision | bf16 (A10G/Ampere) | |
| Epochs | 1.0 | 98,925 records, 69,561 patch pairs |
| Evidence preamble | ~38% of training records | Train/serve parity with controller |
| Hardware | RTX 4080 SUPER 32 GB | ~$2.25 training cost |

#### Train/serve parity (three channels)

1. **Image normalisation** — shared `stretch_to_uint8` for train and serve.
2. **Prompt format** — 0–1000 boxes, no markup; `PROMPT_VERSION` in `eval/prompts.py`.
3. **Evidence preamble** — same `format_evidence()` at train and inference.

#### Stage B (in progress)

Fixes caption regression and grounding format using benchmark **train** splits only; **contamination guard** prevents test-split leakage.

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
| **Accuracy** | Measure-then-describe; Stage A overall **0.050 → 0.126**; 359 tests; prescribed benchmark protocol |
| **Cost** | Ollama 1.9 GB on laptop; EC2 spot ~$0.30/hr; Stage A training ~$2.25 |
| **Data** | BigEarthNet.txt public/ungated; 6 benchmark configs; Indian AOIs via Planetary Computer |
| **Compute** | 2B model on T4/A10G; QLoRA not full fine-tune; Stage B running on RTX 4080S |
| **Security** | Local-first; `SATQUERY_MAX_UPLOAD_MB`; no secrets in repo; trace as audit surface |

**Laptop demo (no GPU, no fine-tune):**

```powershell
pip install -e .
pip install -e ".[geo]"
ollama pull qwen3-vl:2b-instruct
satquery serve
```

**Honest limits:** Europe-only training vs Indian eval imagery; 10 m vs sub-metre Cartosat; VLM can still err — evidence from deterministic tools is the mitigation.

---

## 7. Conclusion (~30 s)

**[SLIDE: Summary numbers]**

SatQuery AI delivers SIH26167's mandatory scope as **working software** with **measured adaptation**:

- All three input configurations + agentic orchestration + evidence trace  
- Stage A: overall **+0.076** normalised; single-image VQA **+0.183**; cross-modal **0 → 0.175**  
- 359 tests, 6 benchmarks, 10 demo scenes, real Sentinel data  

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

# Appendix B — Stage A measured results (2026-09-10)

**Provenance:** `results/2026-09-10-stage-a/` · Harness: `hf` backend both rows · n=200 · seed=1234 · prompt 1.1.0

## B.1 Normalised criteria (headline table)

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

**Interpretation:** Stage A recovered **answering behaviour** on RSVQA (+0.29 raw OA largely from silence → response). Captioning regressed due to BigEarthNet caption idiom vs VRSBench reference style — **Stage B target**. Grounding remains **localisation failure**, not parse failure (unparsed boxes 33 → 0).

---

# Appendix C — Technology stack (validated)

| Layer | Technologies | Notes |
| --- | --- | --- |
| Language | Python **3.11+** | `pyproject.toml` |
| API | **FastAPI**, **Uvicorn**, **Pydantic** | Base deps |
| Arrays / imaging | **NumPy**, **Pillow** | Base; **Rasterio** optional `[geo]` |
| Inference (demo) | **Ollama**, **Qwen3-VL-2B-Instruct** | Default; no torch in base install |
| Inference (reference) | **PyTorch**, **Transformers**, **Accelerate** | `[hf]` extra |
| Inference (serve) | **vLLM** | `[vllm]` extra, EC2 |
| Fine-tuning | **PyTorch**, **peft**, **transformers** | `scripts/train_lora.py` |
| CNN | **PyTorch**, **torchvision** ResNet-50 | `src/satquery/cnn/` |
| Data hub | **huggingface-hub**, **requests** | Downloads, STAC |
| Satellite catalog | **Microsoft Planetary Computer** | `feed.py`, fetch script |
| Storage | Local `runs/` workspace; S3 on AWS | GeoTIFF uploads, models, results |

---

# Appendix D — Code map (quick reference)

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

# Appendix E — External references

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
