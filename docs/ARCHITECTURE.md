# SatQuery AI — System Architecture

Reference architecture for SIH26167. Written so the app track and the ML track can
build in parallel against fixed contracts.

---

## 1. System context

```
┌──────────────┐   images + NL query    ┌─────────────────────────────────────┐
│  Web client  │ ─────────────────────► │  API (FastAPI)                      │
│  (Next.js)   │ ◄───────────────────── │   • upload + validation             │
└──────────────┘   answer + evidence    │   • job lifecycle                   │
       ▲            + trace (ws)        │   • trace stream (WebSocket)        │
       │                                └──────────────┬──────────────────────┘
       │                                               │
       │                                ┌──────────────▼──────────────────────┐
       │                                │  Agentic controller                 │
       │                                │   route → check → plan → execute    │
       │                                │        → fuse → report              │
       │                                └──────────────┬──────────────────────┘
       │                                               │
       │                    ┌──────────────────────────┼──────────────────────┐
       │                    ▼                          ▼                      ▼
       │            ┌───────────────┐         ┌────────────────┐    ┌──────────────┐
       │            │ VLM service   │         │ Specialists    │    │ Geo service  │
       │            │ base + LoRA   │         │ fusion / change│    │ GeoTIFF, CRS │
       │            │ adapters      │         │ / SAR indices  │    │ COG, tiles   │
       │            └───────────────┘         └────────────────┘    └──────────────┘
       │                                               │
       └───────────────── tiles / overlays ────────────┘
```

Four processes, one contract layer. Everything below the API talks in the types
defined in `src/satquery/schema.py`.

---

## 2. Components

| Component | Responsibility | Stack |
|---|---|---|
| **Web client** | Upload, query, side-by-side pair viewer, mask/box overlays, live trace timeline, report download | Next.js, MapLibre |
| **API** | Auth-free local API, upload validation, job queue, WebSocket trace stream | FastAPI, Redis |
| **Controller** | Task routing, input compatibility checks, tool selection, execution, output fusion, confidence | LangGraph |
| **VLM service** | One frozen base + hot-swappable LoRA adapters | vLLM (multi-LoRA) |
| **Specialists** | Optical–SAR fusion encoder, change-mask CNN, deterministic SAR/optical indices | PyTorch / ONNX |
| **Geo service** | GeoTIFF I/O, CRS handling, co-registration checks, COG conversion, tile serving | rasterio, pyproj, TiTiler |
| **Eval harness** | Benchmark scoring, base-model bake-off, regression tracking | offline CLI |

The eval harness is deliberately **offline and independent** — it imports the same
backends and prompts as the serving path but never depends on the API or controller.

---

## 3. Core contracts

Everything hinges on these. They are frozen first, before any component is built.

### 3.1 Input types

```
Modality  = optical | sar | rgb
ImageRole = single | before | after | optical | sar
InputConfig = SINGLE | BITEMPORAL_PAIR | CROSSMODAL_PAIR
```

`InputConfig` is derived from the uploaded set, not declared by the user. The
controller infers it and the UI shows what it inferred — that inference is part of
the scored "input checks."

### 3.2 Tasks

```
vqa | caption | grounding | change_vqa | change_caption | crossmodal_vqa
```

Each maps to exactly one entry in the tool registry. `caption` is the declared
scored extra task; `grounding` ships as an unscored demo tool.

### 3.3 Execution trace — the scored artefact

The judging table scores "correct task/tool selection; valid parameters; auditable
summary; evidence + confidence." That makes the trace a **product surface**, not a
log. Internal reasoning is explicitly not scored and is never emitted.

```jsonc
{
  "run_id": "…",
  "query": "What changed between these two dates?",
  "input_check": {
    "config": "BITEMPORAL_PAIR",
    "images": [
      {"role": "before", "modality": "optical", "crs": "EPSG:32643",
       "size": [1024,1024], "gsd_m": 0.65, "format": "GeoTIFF"},
      {"role": "after",  "modality": "optical", "crs": "EPSG:32643",
       "size": [1024,1024], "gsd_m": 0.65, "format": "GeoTIFF"}
    ],
    "coregistered": true,
    "checks_passed": ["crs_match", "extent_overlap", "size_match", "band_count"],
    "warnings": []
  },
  "routed_task": "change_vqa",
  "steps": [
    {"step": 1, "tool": "change_mask_cnn", "version": "1.0.0",
     "params": {"threshold": 0.5, "tile": 256},
     "outputs": {"mask_uri": "…/mask.png", "changed_area_frac": 0.083},
     "confidence": 0.91, "duration_ms": 1840},
    {"step": 2, "tool": "vlm", "adapter": "change", "version": "1.0.0",
     "params": {"max_new_tokens": 64, "temperature": 0.0},
     "outputs": {"answer": "Built-up area increased in the north-east."},
     "confidence": 0.78, "duration_ms": 2210}
  ],
  "answer": "Built-up area increased in the north-east.",
  "evidence": [{"type": "mask", "uri": "…/mask.png", "label": "change"}],
  "confidence": 0.78,
  "duration_ms": 4050
}
```

Rules: every step names a tool **and its version**; every parameter that was set
appears in `params`; every visual output is addressable by URI. The report export is
this object rendered, not a re-derivation.

### 3.4 Tool registry

A tool is a declarative entry, not an ad-hoc function call:

```python
ToolSpec(
    name="change_mask_cnn",
    version="1.0.0",
    accepts=InputConfig.BITEMPORAL_PAIR,
    tasks=(Task.CHANGE_VQA, Task.CHANGE_CAPTION),
    allowed_params={"threshold": (0.1, 0.9), "tile": {256, 512}},
    outputs=("mask_uri", "changed_area_frac"),
)
```

`allowed_params` is enforced — the problem statement says "only permitted task
parameters may be configured by the agent," so the registry rejects out-of-range
values rather than trusting the controller.

---

## 4. Controller flow

Six nodes, matching the six controller duties in the problem statement:

1. **route** — classify query → task. Small classifier + rules, not a free-form LLM
   decision. Deterministic where possible, because it is scored.
2. **check** — infer `InputConfig`, validate CRS/extent/bands/format, test
   co-registration. Fail loudly with a reason the UI can render.
3. **select** — query the registry for tools matching `(task, input_config)`.
4. **execute** — run the plan; specialists first, VLM last so their outputs can be
   injected as grounded context.
5. **fuse** — combine text + spatial outputs, compute confidence.
6. **report** — emit the trace object and the downloadable report.

Specialists-before-VLM is the important ordering choice: a change mask or a SAR water
index computed deterministically becomes *evidence in the prompt*, which is what makes
the answer evidence-grounded rather than a guess.

### Task routing matrix

| Query intent | Input config | Tools, in order |
|---|---|---|
| Describe / caption | SINGLE | `vlm[caption]` |
| Question | SINGLE | `vlm[vqa]` |
| Highlight / where is | SINGLE | `vlm[grounding]` → box overlay |
| What changed | BITEMPORAL | `change_mask_cnn` → `vlm[change]` |
| Increase/decrease | BITEMPORAL | `change_mask_cnn` → `vlm[change]` |
| Built-up + water from both | CROSSMODAL | `sar_indices` + `optical_indices` → `fusion_encoder` → `vlm[crossmodal]` |

---

## 5. Confidence

Three sources, combined per step and reported honestly:

- **Specialists**: calibrated head output (mask mean probability, classifier softmax).
- **VLM**: mean token log-prob of the answer span, normalised to 0–1.
- **Agreement**: when a specialist and the VLM both answer (e.g. SAR water index says
  12 % water, VLM says "significant water"), agreement raises confidence and
  disagreement lowers it and is surfaced as a warning.

Never report a bare 1.0. A visible, moving confidence number is worth more to judges
than a confident wrong answer.

---

## 6. Deployment topology

```
Vercel ── Next.js
              │ https / wss
GPU host ── docker compose
              ├─ api        (FastAPI, uvicorn)
              ├─ vllm       (base + LoRA adapters)
              ├─ workers    (specialists, geo)
              ├─ redis      (queue + trace pubsub)
              └─ titiler    (COG tiles)
```

- **Dev**: everything local, `echo` backend, no GPU needed.
- **Demo**: single GPU host (vast.ai / college box) + Vercel frontend.
- **Fallback**: offline mode — quantized small VLM on a laptop plus cached fixtures,
  so the demo survives venue wi-fi failure.

---

## 7. Repository layout

```
src/satquery/
  schema.py              core types shared by every component
  config.py              paths and settings
  eval/                  benchmark harness (built first)
    datasets/            VRSBench, RSVQA, CDVQA adapters
    backends/            echo | hf | vllm, one interface
    metrics/             vqa, caption, grounding
    prompts.py           task → prompt, shared with serving
    normalize.py         answer normalisation and box parsing
    runner.py            dataset → backend → metrics
  agent/                 controller, registry, tools     (later)
  geo/                   GeoTIFF, CRS, co-registration   (later)
  api/                   FastAPI                         (later)
configs/bench/           one YAML per benchmark
docs/                    this file, ML_PLAN.md
tests/                   pure-python, no GPU required
```

---

## 8. Build order

1. **Contracts** — `schema.py`, trace types. Unblocks both tracks.
2. **Eval harness + baseline bake-off** ← *current phase*
3. Geo service and compatibility checks
4. Controller, registry, tools
5. API + trace streaming
6. Frontend
7. Fine-tuning (see `ML_PLAN.md`), swapped in behind the same backend interface

Steps 3–6 do not wait for step 7. The adapters drop into a running system.
