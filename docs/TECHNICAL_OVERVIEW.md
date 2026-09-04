# SatQuery AI — Technical Overview

Reference for SIH26167. Covers what the system is, how each part works, the
values it actually uses, and where the boundaries are.

Companion documents: [`ARCHITECTURE.md`](ARCHITECTURE.md) (contracts and design
rationale), [`RESEARCH.md`](RESEARCH.md) (upstream validation with line-level
provenance), [`ML_PLAN.md`](ML_PLAN.md) (fine-tuning schedule),
[`BASELINE.md`](BASELINE.md) (bake-off protocol).

---

## 1. What it is

An agentic vision–language assistant for remote sensing. A user supplies one
satellite image or a pair, asks a question in natural language, and the system:

1. infers what kind of imagery it received,
2. classifies what kind of question was asked,
3. selects tools from a registry,
4. computes every deterministic measurement it can,
5. injects those measurements into a vision–language model's prompt,
6. returns an answer with visual evidence, a confidence, and an auditable trace.

**8,134 lines** of application code, **2,223** of tests, **198** test cases.

### Scale of the neural component

Worth stating precisely, because "AI system" obscures it:

| Component | Count | Learned? |
|---|---|---|
| Vision–language models | 1 (six registry entries, one set of weights) | yes |
| Deterministic measurements | 3 | no |
| Text-only LLMs / SLMs | **0** | — |
| Trained CV models | **0** (deferred, see §14) | — |

Routing is regular expressions over the query and the inferred input
configuration. Confidence is a lexical estimate combined with specialist
agreement. Both are deterministic: task selection is scored by the judging
criteria, and a rule-based router is auditable, instant, free, and cannot
hallucinate a task that does not exist.

---

## 2. Requirement traceability

Mapping the problem statement's mandatory functional scope to implementation.

| Requirement | Where | Status |
|---|---|---|
| Single-image VQA (mandatory) | `vlm_vqa`, `configs/bench/rsvqa_lr.yaml`, `vrsbench_vqa.yaml` | done |
| One additional single-image task | `vlm_caption` (**declared, scored**) and `vlm_grounding` (shipped, unscored) | done |
| Multi-image change analysis (mandatory) | `change_mask` → `vlm_change_vqa` / `vlm_change_caption`, `configs/bench/cdvqa.yaml` | done |
| Cross-modal optical–SAR analysis (mandatory) | `optical_indices` + `sar_indices` → `vlm_crossmodal_vqa` | done |
| Agentic orchestration | `agent/controller.py`, `agent/router.py`, `agent/planner.py`, `agent/registry.py` | done |
| Input upload and compatibility checking | `geo/checks.py`, `POST /api/upload` | done |
| Visual evidence, confidence, execution summary, downloadable report | `ExecutionTrace`, `/artifacts/...`, report export | done |
| **Remote-sensing adaptation (fine-tuning)** | `ML_PLAN.md` | **deferred** |

The last row is the open requirement. The problem statement is explicit that a
generic VLM without remote-sensing adaptation does not satisfy it. The trace
schema and `VLMTool` already carry the `adapter` field the LoRA adapters will
populate, so the change is additive.

### Captioning versus grounding

The problem statement permits one additional single-image task. **Captioning is
declared.** BLEU / ROUGE-L / CIDEr-D are far more reliably earned by a
fine-tuned VLM than VRSBench referring `Acc@IoU0.5`, and because scores are
normalised before aggregation, a weak grounding number would drag the total.
Grounding still ships as a working tool because a representative query asks for
it, but it is not the scored choice.

---

## 3. Repository layout

```
src/satquery/
  schema.py            core contracts; imports no ML stack
  models.py            model catalog, weight presence, downloads
  data.py              benchmark dataset acquisition
  feed.py              live STAC search and windowed COG pulls
  cli.py               satquery bench | serve | models | data

  geo/
    raster.py          RasterInfo, band access, previews, SAR dB
    checks.py          modality, input configuration, co-registration

  agent/
    controller.py      six-stage pipeline, execution, re-planning
    router.py          query → task, rule based
    planner.py         adaptive parameters, revision conditions
    registry.py        Tool, ToolResult, ToolRegistry
    context.py         RunContext, the shared artifact bag
    confidence.py      lexical confidence, agreement, blending
    tools/
      change.py        change_mask
      indices.py       optical_indices, sar_indices
      vlm.py           six VLM tools, evidence injection
      _imaging.py      Otsu, overlays, quadrant summary

  eval/
    backends/          ollama | hf | vllm | echo behind one interface
    datasets/          RSVQA, VRSBench, CDVQA adapters
    metrics/           vqa (OA/AA), caption (BLEU/ROUGE-L/CIDEr-D), grounding
    prompts.py         task → prompt, shared with serving
    normalize.py       answer normalisation, box parsing
    runner.py          one benchmark × one backend
    matrix.py          many models × many benchmarks
    report.py          results.csv, comparison tables

  api/
    app.py             FastAPI: 22 routes
    jobs.py            in-process job store, websocket fan-out
    settings.py        environment-driven configuration
    static/            web client, no build step
```

---

## 4. Core contracts

`schema.py` deliberately imports nothing from torch, transformers or rasterio,
so it stays importable in CI and on a machine with no ML stack.

```python
Task = vqa | caption | grounding | change_vqa | change_caption | crossmodal_vqa
Modality = optical | sar | rgb
ImageRole = single | before | after | optical | sar
InputConfig = single | bitemporal_pair | crossmodal_pair
BBox = tuple[float, float, float, float]  # xyxy, normalised to [0, 1]
```

`TASK_INPUT_CONFIG` maps each task to exactly one input configuration, which is
what lets the registry answer "which tools serve this task, given this input".

Key records: `ImageRef`, `Sample`, `Prediction`, `GenerationRequest`, `ToolSpec`,
`TraceStep`, `InputCheck`, `ExecutionTrace`. All are frozen slotted dataclasses;
`jsonable()` converts any of them to JSON-safe primitives, handling enums,
`Path`, tuples and nested dataclasses.

---

## 5. Input handling

### 5.1 Raster inspection

`geo/raster.py::read_info` returns a `RasterInfo`: path, width, height, band
count, dtype, driver, CRS, bounds, affine transform, and ground sample distance.

GeoTIFF goes through rasterio; PNG/JPEG fall back to PIL and yield a
non-georeferenced record. **Rasterio is tried first for `.tif/.tiff/.jp2`** — PIL
cannot decode a 12-band uint16 stack and reports that to stderr rather than
raising, so trying it first is both wrong and noisy.

GSD is derived from the affine transform. For a geographic CRS it converts
degrees to metres as `pixel_size × 111320 × cos(latitude)`.

### 5.2 Modality inference

Filename hints outrank band counts: a user who names a file `*_VV.tif` is
communicating something the header often does not.

| Signal | Result |
|---|---|
| Filename contains `sar`, `_vv`, `_vh`, `grd`, `risat`, `sentinel-1`, `s1_` | SAR |
| Filename contains `optical`, `msi`, `cartosat`, `sentinel-2`, `s2_`, `pan`, `mss` | optical |
| ≤ 2 bands and float dtype, or exactly 2 bands | SAR |
| ≥ 4 bands | optical |
| 3 bands, georeferenced | optical |
| 3 bands, not georeferenced | RGB |

### 5.3 Input configuration

Derived from the uploaded set, never declared:

- 1 image → `SINGLE`
- 2 images, differing modality → `CROSSMODAL_PAIR`
- 2 images, same modality → `BITEMPORAL_PAIR` (upload order is acquisition order)
- 3+ images → rejected with the supported configurations named

### 5.4 Co-registration checks

| Check | Condition |
|---|---|
| `size_match` | identical pixel dimensions |
| `crs_match` | identical CRS string |
| `extent_overlap` | bounding-box IoU ≥ `0.90` |
| `gsd_match` | relative GSD difference ≤ `0.05` |

`coregistered` requires `crs_match` **and** `extent_overlap`. Benchmark PNG pairs
carry no geotransform; they are accepted on identical dimensions and the
assumption is stated as a warning rather than dressed up as a passed check.

Every conclusion lands in `checks_passed` or `warnings`, both of which reach the
trace and the UI, because the problem statement makes input checking a scored
controller duty.

### 5.5 Rendering, and the SAR decibel conversion

`preview_bands(count)` selects display bands: `(4, 3, 2)` for ≥ 12 bands (a
Sentinel-2 stack in BigEarthNet order begins B01, B02, B03, so the first three
are not true colour), `(1, 2, 3)` for 3–4 bands, greyscale below that.

`looks_like_linear_backscatter()` detects SAR in linear power — ≤ 2 bands,
floating dtype, non-negative, and `p99 / median > 8` — and converts to dB before
stretching.

> This was found by real data. A Sentinel-1 RTC tile over Mumbai ran 0.001 to 360
> with a median of 0.12. Percentile-stretching the linear values piled almost
> every pixel into the darkest bin and Otsu had nothing to separate, so SAR
> reported **94% of the scene as water**. In dB the histogram is genuinely
> bimodal and the figure drops to **31.5%**, agreeing with optical NDWI computed
> independently. A DEM is excluded by the same test: positive and single-band,
> but nothing like as skewed.

---

## 6. The agentic controller

### 6.1 Six stages

| # | Stage | Function |
|---|---|---|
| 1 | check | `Controller.check_inputs` — inspect, infer configuration, validate |
| 2 | route | `router.route` — query + configuration → one task |
| 3 | select | `Controller.plan` — registry lookup + adaptive parameters |
| 4 | execute | measurements first, model last |
| 5 | fuse | confidence blending, agreement, warnings |
| 6 | report | `ExecutionTrace` |

Implemented as an explicit pipeline rather than a graph framework. The flow is a
fixed six stages, so a framework would add a dependency and an indirection
without adding capability, and the emitted trace is a direct reading of the code
rather than a rendering of a framework's internal state.

### 6.2 Routing

Deterministic. The input configuration constrains the answer before any keyword
is read.

| Configuration | Condition | Task |
|---|---|---|
| `CROSSMODAL_PAIR` | any wording | `crossmodal_vqa` |
| `BITEMPORAL_PAIR` | empty query | `change_caption` |
| `BITEMPORAL_PAIR` | description verb, no interrogative | `change_caption` |
| `BITEMPORAL_PAIR` | otherwise | `change_vqa` |
| `SINGLE` | `highlight`, `locate`, `where is`, `mark`, `bounding box`, … | `grounding` |
| `SINGLE` | `describe`, `caption`, `summarise`, no interrogative | `caption` |
| `SINGLE` | interrogative pattern, or trailing `?` | `vqa` |
| `SINGLE` | fallback | `vqa` |

The matched rule and a routing confidence are recorded on the trace.

### 6.3 Tool registry

A tool is a declarative entry, not an ad-hoc call:

```python
ToolSpec(
    name, version, accepts: InputConfig, tasks: tuple[Task, ...],
    allowed_params: Mapping[str, tuple | set],
    outputs: tuple[str, ...],
    summary, kind, category, cost, requires, emits_evidence,
    param_docs: Mapping[str, str],
)
```

`allowed_params` is **enforced on every invocation** — the problem statement
permits the agent to configure only sanctioned parameters, so a range violation
or an unknown key raises `ParameterError` before the tool runs.

`kind` separates a deterministic `measurement` from a learned `model`. This is
the most load-bearing field in the registry: measurements are reproducible and
become the evidence a model output is checked against, so the two must not be
presented as interchangeable.

### 6.4 Adaptive parameters

Parameters are derived from the run's own inputs, and each records why. Before
this, the controller called every tool with no arguments — `params` was empty in
every trace step, so the registry enforced a guard nothing exercised, while the
judging criteria score exactly that.

**Change-mask speckle floor**, anchored to a ground area:

```
min_area_frac = MIN_CHANGE_AREA_M2 / (scene_pixels × gsd²)     clamped to [0, 0.2]
MIN_CHANGE_AREA_M2 = 10 000     (one hectare)
```

A fixed fraction silently means something different at every resolution. One
hectare is `9.5e-5` of a 1024² scene at 10 m/px and `2.6e-2` of the same scene at
0.6 m/px — a 280× difference. Anchoring to ground area keeps the floor meaning
one thing.

**SAR built-up percentile**, relaxed on coarse imagery:

```
builtup_percentile = 92.0  if gsd ≥ 8 m
                     95.0  otherwise
```

A 10 m pixel averages scatterers that a 1 m pixel resolves, so the bright
double-bounce tail compresses. Holding the percentile fixed would report less
built-up simply because the imagery was coarser.

**Generation budget**, matched to the question:

| Condition | `max_new_tokens` |
|---|---|
| captioning / change captioning | 128 |
| grounding | 64 |
| closed question | 24 |
| open question | 48 |

A closed question is detected by a leading auxiliary (`is`, `are`, `does`,
`has`, …), `how many` / `how much`, or a change verb. The auxiliaries are
anchored to the start of the string on purpose: matching them anywhere reads
"What **is** going on here" — an open question — as closed.

Only adaptations with a real justification are made. A knob tuned on a hunch
would look adaptive and mean nothing, which is worse than a documented default.

### 6.5 Conditional re-planning

The execution queue is mutable: a step may append one revision of itself.

**Change mask retry.** When the mask used an Otsu split and found less than
`QUIET_CHANGE = 0.02` of the scene, it is retried once at
`max(otsu × 0.55, 0.02)`. Otsu assumes a bimodal histogram; when two dates differ
subtly it splits noise instead of signal. A step that already carries an explicit
threshold is never lowered again, so the retry cannot chain until it finds
something in noise.

**Answer retry.** When the answer contradicts a measurement — detected by
`agreement_adjustment` — the VLM is asked once more with firmer framing. The
reinforcement travels through the same shared artifact bag the measurements use,
so only the framing changes, never the question.

Bounded by `MAX_REVISIONS = 2`. **Both attempts remain in the trace**, linked by
`revises`. An agent that silently re-rolls until it likes the answer is not
auditable; one that shows the first attempt, the reason, and the second is.

### 6.6 Evidence injection

Specialists write into `RunContext.artifacts` under stable names; the VLM tool
renders them as a prompt preamble:

```
Measurements from image-analysis tools that have already run on these images.
Treat them as reliable and do not contradict them:
- changed area (fraction of scene): 0.0696
- change location: mainly north-west
- change direction: darkened
```

On a contradiction retry the header becomes firmer:

```
Your previous answer contradicted these measurements, which were computed
directly from the pixels and are not in doubt. Answer again, consistently
with them:
```

This is why specialists run before the model, and it is the difference between
an evidence-grounded answer and a plausible one.

### 6.7 Confidence

Three sources, combined and reported honestly. **A bare 1.0 is never returned**;
values are clamped to `[0.05, 0.92]`.

*Lexical* — base `0.78`; `−0.22` if any hedge word is present (`may`, `might`,
`possibly`, `unclear`, `appears`, `seems`, …); `−0.08` above 60 tokens; `+0.06`
at three tokens or fewer.

*Specialist* — each measurement reports its own value (change mask scales with
Otsu separability; optical indices `0.85`; SAR `0.80`).

*Agreement* — an additive adjustment after blending:

| Situation | Adjustment |
|---|---|
| answer says "no change", mask covers > 10% | `−0.18` + warning |
| answer says "changed", mask covers < 2% | `−0.15` + warning |
| answer and mask agree | `+0.10` |
| optical vs SAR water differ by > 15 points | `−0.12` + cloud/shadow warning |
| optical and SAR agree on water | `+0.08` |

Disagreement is surfaced as a user-visible warning rather than averaged away.

### 6.8 Execution trace

The scored artefact, and therefore a product surface rather than a log.

```jsonc
{
  "run_id": "824bd1dbf85c",
  "query": "What changed between these two dates?",
  "input_check": {
    "config": "bitemporal_pair",
    "images": [{ "role": "before", "modality": "optical", "crs": "EPSG:32643",
                 "size": [1024, 1024], "bands": 4, "gsd_m": 10.0,
                 "format": "GTiff", "georeferenced": true }],
    "coregistered": true,
    "checks_passed": ["format_supported", "size_match", "crs_match",
                      "extent_overlap", "gsd_match"],
    "warnings": []
  },
  "routed_task": "change_caption",
  "routing_rule": "bitemporal + \\bdescribe\\b",
  "routing_confidence": 0.85,
  "steps": [
    { "step": 1, "tool": "change_mask", "version": "1.0.0",
      "params": { "min_area_frac": 0.000095 },
      "reason": "speckle floor 0.000095 of the scene, which is 1 ha at 10.00 m/px",
      "outputs": { "threshold_method": "otsu", "threshold": 0.3613,
                   "changed_area_frac": 0.0696, "change_location": "mainly north-west",
                   "direction": "darkened", "mask_uri": "artifacts/.../change_mask.png" },
      "confidence": 0.9, "duration_ms": 142, "revises": null },
    { "step": 2, "tool": "vlm_change_caption", "version": "1.0.0",
      "params": { "max_new_tokens": 128 },
      "reason": "description task; room for two sentences",
      "outputs": { "answer": "...", "grounded_in_evidence": true },
      "confidence": 0.84, "duration_ms": 2210, "revises": null }
  ],
  "answer": "...",
  "evidence": [{ "type": "mask", "uri": "artifacts/.../change_mask.png",
                 "label": "detected change" }],
  "confidence": 0.84,
  "duration_ms": 2380
}
```

Internal reasoning is neither required nor emitted — the judging criteria score
the observable trace only.

---

## 7. Tool inventory

| Tool | Kind | Accepts | Permitted parameters | Outputs |
|---|---|---|---|---|
| `change_mask` | measurement | bitemporal | `threshold` 0.02–0.9, `min_area_frac` 0.0–0.2 | changed fraction, location, direction, mask URI |
| `optical_indices` | measurement | crossmodal | `water_threshold` −1–1, `builtup_threshold` −1–1 | applicable, water/built-up fractions, mask URI |
| `sar_indices` | measurement | crossmodal | `builtup_percentile` 80–99.5 | water/built-up fractions, locations, mask URI |
| `vlm_vqa` | model | single | `max_new_tokens` 1–512, `temperature` 0–1 | answer |
| `vlm_caption` | model | single | ″ | answer |
| `vlm_grounding` | model | single | ″ | answer, bbox, parse_ok |
| `vlm_change_vqa` | model | bitemporal | ″ | answer |
| `vlm_change_caption` | model | bitemporal | ″ | answer |
| `vlm_crossmodal_vqa` | model | crossmodal | ″ | answer |

### Algorithms

**`change_mask`** — both dates to percentile-stretched greyscale, absolute
difference, Otsu threshold unless one is supplied. Reports changed fraction, a
compass-quadrant summary, and a direction from the mean intensity shift
(`brightened` / `darkened` / `mixed` / `no significant change`). Confidence
tracks how decisively the Otsu split separated the histogram.

**`optical_indices`** — NDWI `(green − nir) / (green + nir)`, NDBI
`(swir − nir) / (swir + nir)`. Band positions are resolved by band count: 12 and
13 band stacks use the Sentinel-2 layout, a 4-band stack is R/G/B/NIR. On a
3-band input it returns `applicable: false` with a reason **rather than
fabricating an index** — a benchmark PNG genuinely cannot yield NDWI, and saying
so is worth more than a plausible number.

**`sar_indices`** — linear backscatter to dB, percentile stretch, Otsu split;
the dark class is water, the bright tail is built-up. Thresholds are relative,
never absolute dB, because an uploaded GRD tile may be in dB, in linear power,
or already stretched to 8-bit, and a hard −18 dB cut would silently mean three
different things.

---

## 8. Backends

One interface, four implementations, selected by name.

| Backend | Requires | Use |
|---|---|---|
| `ollama` | Ollama server | **default** — quantised weights, laptop-friendly |
| `hf` | torch, torchvision, transformers | full precision, GPU host, published numbers |
| `vllm` | vllm | throughput on a GPU |
| `echo` | nothing | **test double**, not a product surface |

### 8.1 Preflight

Availability is checked *before* construction. A sweep that dies on
`No module named 'torch'` tells an operator nothing to do.

- **Runtime check** — importability via `find_spec`, without importing. Reports
  the missing modules and the install command. For `ollama` this additionally
  probes `GET /api/version`, because the dependency that actually fails is a
  server that is not running.
- **Host-memory check** — on a CPU host, refuses a load whose weights exceed
  available RAM (`file size × 1.25`). A 3B checkpoint needs roughly 9 GB
  resident; letting that OOM means minutes of swap thrashing before anything
  fails. Skipped when CUDA is present, and when psutil is absent, so a missing
  optional dependency never blocks a load that would have worked.

`/api/models` reports **weights present**, **runtime available** and **runnable**
as three separate facts. Conflating them let the UI offer a model it could not
load.

### 8.2 Ollama

`POST /api/chat` with `messages[].images` as an array of base64 strings —
multiple images in one message, which is what the paired tasks need.
`options.temperature` and `options.num_predict` carry the sampling settings.
Requests are sequential because Ollama serialises generation per model anyway.

Rasters are rendered through the project's own loader before encoding, so a
12-band uint16 GeoTIFF and a linear-power SAR tile both arrive as sensible JPEG.

Error bodies are unwrapped rather than discarded. Ollama nests a JSON document
inside the `error` string, and the message is often the whole diagnosis — a
model published without a vision projector reports exactly that.

### 8.3 Transformers

Validated against fetched upstream source, not documentation summaries; the
verdict table with line-level provenance is in [`RESEARCH.md`](RESEARCH.md).

- `AutoModelForImageTextToText` resolves **both** Qwen2.5-VL and InternVL:
  upstream registers both in `MODEL_FOR_IMAGE_TEXT_TO_TEXT_MAPPING_NAMES`.
  One code path, no per-model branching.
- Prompt assembly is the single documented call —
  `apply_chat_template(..., tokenize=True, return_dict=True)` tokenises *and*
  loads the imagery, so there is no second `processor(text=, images=)` step.
- Images pass by `path`; accepted content keys are `image | url | path | base64`.
- Resolution is capped by the processor's `min_pixels` / `max_pixels`, not by
  pre-resizing. Each architecture patches on its own grid, so resizing
  beforehand fights the processor.
- Only InternVL's `-hf` repositories expose this interface. The plain
  `OpenGVLab/*` repos ship a bespoke `.chat()` API behind `trust_remote_code`
  and are **not** interchangeable.

---

## 9. Model catalog

| id | Runtime | Repository / tag | Size | Licence |
|---|---|---|---|---|
| `qwen3-vl-2b` | ollama | `qwen3-vl:2b-instruct` | 1.9 GB | Apache-2.0 |
| `qwen3-vl-4b` | ollama | `qwen3-vl:4b-instruct` | 3.3 GB | Apache-2.0 |
| `qwen3-vl-8b` | ollama | `qwen3-vl:8b-instruct` | 6.1 GB | Apache-2.0 |
| `qwen25-vl-7b-ollama` | ollama | `qwen2.5vl:7b` | 6.0 GB | Apache-2.0 |
| `qwen25-vl-3b` | hf | `Qwen/Qwen2.5-VL-3B-Instruct` | 7.5 GB | Apache-2.0 |
| `internvl3-2b-hf` | hf | `OpenGVLab/InternVL3-2B-hf` | 4.4 GB | ⚠️ "other" |

Quantisation is the reason Ollama leads: a 2B vision–language model is 1.9 GB
resident against roughly 9 GB for the same class of model in full precision. On
a 16 GB laptop that is the difference between a real model and no model.

The transformers entries remain for GPU hosts and for reproducing published
numbers, where quantisation would make a benchmark score incomparable.

⚠️ InternVL3's Hub metadata reports its licence as `other`. Read the model card
before relying on it for submission; Qwen2.5-VL and Qwen3-VL are Apache-2.0.

Weights land in `runs/models/<repo--id>/` as a plain directory, not the shared
Hugging Face blob cache. That cache symlinks blobs into snapshot folders, which
needs Developer Mode or administrator rights on Windows and otherwise fails
mid-download with `WinError 1314`. A plain directory works everywhere and is
portable to an offline demo machine.

---

## 10. Benchmark harness

### 10.1 Dataset adapters

Public benchmark releases move their JSON keys between versions, so every field
is resolved through a candidate-key list that a YAML config can override.
`satquery bench validate` reports what it found and what it could not map, so a
schema mismatch is fixed by editing YAML rather than code.

| Adapter | Layout handled |
|---|---|
| `vrsbench_vqa` / `_caption` / `_referring` | one flat JSON list per task |
| `rsvqa` | questions and answers in separate files joined on id; `active: false` dropped, matching the official protocol |
| `cdvqa` | LLaVA-style conversations; images in parallel `im1/` and `im2/` directories |

### 10.2 Metrics

Implemented in-repo, so scoring needs no Java, no network and no NLTK download.

- **VQA / change VQA** — exact match after normalisation, with a containment
  fallback capped at 12 tokens so a rambling answer cannot match by accident.
  Reports **OA** and **AA**.
- **Captioning** — corpus BLEU-1..4 with clipping and brevity penalty; ROUGE-L
  as an LCS F-measure with `β = 1.2`, best reference per sample; CIDEr-D with
  tf-idf weighting, count clipping and a Gaussian length penalty (`σ = 6`,
  n = 1..4, factor 10), document frequencies computed over the evaluation set's
  own references. METEOR is optional and degrades to absent.
- **Grounding** — IoU, `Acc@IoU0.25`, `Acc@IoU0.5`, mIoU, plus a parse rate.
  **An unparseable box scores IoU 0 rather than being dropped**; dropping them
  inflates the score.

Tokenisation is a simple lowercase/strip-punctuation split rather than the PTB
tokenizer used by `pycocoevalcap`. Scores are therefore consistent *within this
repository* — which is what a bake-off needs — but should not be quoted against
published numbers without the official tooling.

### 10.3 Why AA is always reported

These benchmarks are class imbalanced. A model that always answers the majority
class posts a respectable overall accuracy; average accuracy across question
types exposes it.

Measured on the real RSVQA-LR test split, a fixed-answer baseline:

```
oa                  0.3750
aa                  0.2510
  acc/presence      0.6667
  acc/comp          0.3375
  acc/count         0.0000
  acc/rural_urban   0.0000
```

### 10.4 Multi-model matrix

Loops **model-outer, dataset-inner**, so each model is constructed once and
reused across every benchmark — loading a VLM is the expensive part. Each
`(model, benchmark)` cell writes its own result file and is skipped when that
file exists, so an interrupted sweep resumes.
*(Pattern: VLMEvalKit `run.py:624, 658, 645-672`.)*

The sample subset is **seeded and frozen before the sweep starts**, so every
model sees the identical questions. Without that the comparison would be
between different question sets and would mean nothing.

A model that fails to load errors its own row and the sweep continues; one bad
checkpoint does not discard results already earned.

---

## 11. Datasets

Fetched from the sources named in the problem statement, so a score traces back
to an official release.

| Benchmark | Source | Size | Notes |
|---|---|---|---|
| RSVQA-LR | Zenodo record `6344334` | 95 MB | Official release. 10,004 test questions across `presence`, `comp`, `count`, `rural_urban` |
| VRSBench | HF `xiang709/VRSBench` | 23 MB annotations · 3.8 GB imagery | Authors' mirror. 37,409 VQA questions. Imagery is opt-in |
| CDVQA | HF `ljx620/CDVQA` | 76 MB per shard | 100 samples/shard, 397 test shards. Eight official question types |

CDVQA's official repository distributes the question file separately from the
SECOND imagery it is built on, behind a manual download; the Hub mirror packages
both, which is what makes an automated pull possible. Shards are converted into
the original `im1/` + `im2/` layout, with images written once per `image_id`
because several questions share one pair.

---

## 12. Live satellite feed

A thin STAC client over Microsoft Planetary Computer with anonymous SAS signing —
no credentials.

- **Search** — `sentinel-2-l2a` and `sentinel-1-rtc`, newest first, over a chosen
  footprint. Browsing uses the catalogue's own `rendered_preview`, so nothing
  downloads until a scene is loaded. No tile server is run: the catalogue already
  publishes `rendered_preview` and `tilejson`, so standing up TiTiler alongside
  would be reinvention.
- **Load** — selected scenes are warped onto **one explicitly defined UTM grid**
  (zone derived from the footprint centre, origin snapped to the resolution,
  1024 px at 10 m). An optical + SAR selection therefore arrives genuinely
  co-registered — identical CRS, extent and pixel size — rather than merely
  overlapping. Sentinel-2 is pulled as the 12-band BigEarthNet ordering that the
  optical index tool maps onto directly; Sentinel-1 as VV + VH float32.

**Empty results are explained, not shown as a blank grid.** Searching Mumbai in
monsoon returns nothing under a 20% cloud threshold — the clearest acquisition is
78% cloud. The feed reports how many scenes the filter removed and points at
Sentinel-1, which is the problem statement's own argument for radar arriving
unprompted from live data.

---

## 13. API and web client

22 routes. `POST` bodies are validated by pydantic models.

| Route | Purpose |
|---|---|
| `GET /api/health` | status, settings, model state, tool count |
| `GET /api/model`, `POST /api/models/pull` | weight download state and trigger |
| `GET /api/models` | catalog with weights / runtime / runnable |
| `GET /api/tools` | the registry, with permitted parameters and docs |
| `POST /api/upload` | ≤ 2 images, format and metadata checks, preview render |
| `GET /api/samples`, `POST /api/samples/{id}/load` | bundled real scenes |
| `POST /api/query` | run the controller; returns a `run_id` |
| `WS /ws/runs/{run_id}` | live trace stream |
| `GET /api/runs/{run_id}` | one run's detail |
| `GET /api/feed/collections`, `POST /api/feed/search`, `POST /api/feed/load` | live imagery |
| `GET /api/datasets`, `POST /api/datasets/pull` | benchmark data |
| `GET /api/benchmarks`, `POST /api/benchmarks/run`, `POST /api/benchmarks/matrix` | evaluation |
| `GET /api/results` | results history |
| `GET /artifacts/{run_id}/{file}`, `GET /previews/{file}` | evidence, path-traversal guarded |

**There is no run-history endpoint.** A run is live state, not an archive;
benchmark numbers are what is worth keeping and they persist in `results.csv`.

Trace steps stream over a websocket as tools fire rather than arriving as one
payload at the end — watching the tools execute in order is the observable
agentic behaviour the problem statement asks for. The stream replays recorded
events before following live, closing the race where a client connects after the
first tool has finished.

The web client is four tabs — Analyse, Live feed, Benchmarks, Registry — with
**no build step**. The demo has to start with one command on a machine that may
have no Node toolchain, and an offline venue is a real risk. System fonts only,
for the same reason.

---

## 14. Deferred work

**Fine-tuning.** The open mandatory requirement. Plan, GPU budget and weekly
schedule are in [`ML_PLAN.md`](ML_PLAN.md): four LoRA adapters (`vqa`, `caption`,
`change`, `crossmodal`) on one frozen base, trained on BigEarthNet.txt, RSVQA,
VRSBench and CDVQA. The adapter field already exists throughout.

**Trained change detector.** The current `change_mask` is a deterministic
difference-and-threshold baseline. A Siamese CNN on LEVIR-CD drops in behind the
same `ToolSpec` with a version bump; nothing upstream changes.

**Optical–SAR fusion encoder.** A CROMA/DeCUR-style dual-tower encoder is planned
for the cross-modal path; today that path uses deterministic indices only.

### Known limitations

- The resolution domain gap is the largest technical risk: BigEarthNet is
  120×120 at 10 m, while the hidden ISRO evaluation set is Cartosat-2S
  sub-metre. Training must include high-resolution data and aggressive scale
  augmentation.
- Grounding is not the declared scored task and its accuracy is untuned.
- The job store is in-process; a multi-worker deployment would need Redis behind
  the same interface.
- Benchmark scores are internally consistent but use a simpler tokenizer than
  `pycocoevalcap`.

---

## 15. Configuration

| Variable | Default | Effect |
|---|---|---|
| `SATQUERY_BACKEND` | `ollama` | `ollama` \| `hf` \| `vllm` |
| `SATQUERY_MODEL` | `qwen3-vl:2b-instruct` | model tag or repository id |
| `SATQUERY_WORKSPACE` | `runs` | uploads, previews, artifacts, results |
| `SATQUERY_DATA_ROOT` | `data` | benchmark datasets |
| `SATQUERY_MODELS_DIR` | `runs/models` | weight store |
| `SATQUERY_MAX_SIDE` | `1024` | longest image edge sent to a model |
| `SATQUERY_PRELOAD` | `1` | fetch weights at startup, not first query |
| `SATQUERY_ALLOW_DOWNLOAD` | `1` | permit weight downloads |
| `OLLAMA_HOST` | `http://127.0.0.1:11434` | Ollama endpoint |
| `HF_HUB_OFFLINE` | unset | weights must already be present |

### Running

```bash
pip install -e .                 # base: no torch, starts anywhere
pip install -e ".[geo]"          # GeoTIFF support (rasterio)
pip install -e ".[hf]"           # transformers backend
satquery serve                   # http://127.0.0.1:8000

ollama pull qwen3-vl:2b-instruct # the default model

satquery data list               # prescribed benchmarks and their sources
satquery data pull rsvqa_lr
satquery bench validate --config configs/bench/rsvqa_lr.yaml
satquery bench run --config configs/bench/*.yaml --backend ollama \
  --model qwen3-vl:2b-instruct --limit 200
```

---

## 16. Testing

198 cases across 11 modules, all runnable with no GPU, no weights and no network.

| Module | Cases | Covers |
|---|---|---|
| `test_agent.py` | 27 | routing, registry, parameter enforcement, controller end to end |
| `test_geo.py` | 25 | raster IO, modality, configuration, co-registration, SAR dB |
| `test_api.py` | 22 | uploads, query streaming, benchmarks, path safety |
| `test_planner.py` | 15 | adaptive parameters, re-planning bounds |
| `test_models.py` | 14 | weight presence, download orchestration, startup preload |
| `test_backends.py` | 13 | runtime preflight, host-memory refusal |
| `test_ollama.py` | 12 | host resolution, presence, error unwrapping, paired images |
| `test_metrics.py` | 12 | BLEU, ROUGE-L, CIDEr-D, OA/AA, IoU |
| `test_normalize.py` | 12 | answer normalisation, box parsing across five formats |
| `test_harness.py` | 11 | dataset adapters, seeded subsets, results CSV |
| `test_feed.py` | 10 | STAC search, empty-result diagnosis, grid maths |

CI runs ruff lint, ruff format check, and the full suite on every push and pull
request, plus Gitleaks over history and the same pre-commit hooks used locally.
