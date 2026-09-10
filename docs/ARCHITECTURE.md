# Architecture

SatQuery AI answers natural-language questions about satellite imagery by
*measuring first and describing second*. Deterministic remote-sensing tools
compute quantities from pixels; a vision-language model turns those measurements
into an answer. The controller decides which tools run and in what order, and
emits a trace of everything it did.

That ordering is the whole design. It is also how operational remote sensing
actually works: measurement is physics-based and auditable, and the narrative is
written afterwards by an analyst. Here the model plays the analyst.

---

## Why this shape

A general VLM shown a satellite image will produce fluent, plausible, and
frequently wrong statements about water extent or urban growth. It has no way to
measure. Meanwhile NDWI, NDBI and Otsu thresholding measure those exact
quantities reliably, and have for decades — but cannot answer a question phrased
in English.

So each does the half it is good at:

```
       images ──▶ specialists ──▶ measurements ──┐
                                                 ├──▶ VLM ──▶ answer
       query  ─────────────────────────────────  ┘
```

Measurements are injected into the model's prompt as evidence it is told not to
contradict. An answer is therefore grounded in a number computed from the pixels
rather than in the model's impression of them.

The problem statement asks for exactly this — a registry of specialist tools, a
controller that selects among them, permitted parameters only, evidence,
confidence, and an auditable execution summary. It also states that **only the
observable execution trace is evaluated** and internal reasoning is not scored.
The trace is therefore a product surface, not a debug log.

---

## The six stages

`agent/controller.py` implements six stages matching the six controller duties
in the problem statement.

### 1. Check

Read every input raster: band count, dtype, CRS, transform, ground sample
distance. For pairs, verify they describe the same ground:

- footprint intersection-over-union ≥ `0.90`
- GSD agreement within `0.05`

Failures surface as warnings on the run rather than exceptions. A pair that is
not co-registered produces meaningless change detection, and the trace should
say so rather than quietly report a number.

### 2. Route

Classify the query into a task. Input configuration is **inferred from the
imagery, never declared by the user** — one image is `SINGLE`; two images are
`BITEMPORAL_PAIR` or `CROSSMODAL_PAIR` depending on whether their modalities
match. Asking a user to state that they have uploaded a SAR image is asking them
to know the thing the system exists to figure out.

Six tasks: `VQA`, `CAPTION`, `GROUNDING`, `CHANGE_VQA`, `CHANGE_CAPTION`,
`CROSSMODAL_VQA`.

### 3. Select

Look up tools registered for `(task, input_config)`. Specialists that precede a
task come from an explicit map:

```python
_PRECURSORS = {
    Task.CHANGE_VQA: ("change_mask",),
    Task.CHANGE_CAPTION: ("change_mask",),
    Task.CROSSMODAL_VQA: ("optical_indices", "sar_indices"),
}
```

Single-image tasks have no precursors, so a single-image run is one VLM call.

Parameters are derived from the run's own inputs at this point, not left at tool
defaults, and each records *why* it was chosen. See `TOOLS.md`.

### 4. Execute

Run the queue in order. Specialists first, so their outputs are in the artifact
bag before the VLM prompt is built. Each step is timed, scored for confidence,
and emitted as a trace step the moment it completes.

The queue is **mutable**: a step may append one revision of itself. Bounded by
`MAX_REVISIONS = 2`, so a run cannot loop.

### 5. Fuse

Promote specialist outputs into a shared artifact bag under stable names, render
them as a prompt preamble, and attach visual evidence (mask PNGs, bounding
boxes) to the result.

### 6. Report

Emit the answer, the evidence, the per-step trace, warnings, and a downloadable
report.

---

## Evidence injection

The mechanism that makes the whole thing work, in `agent/tools/vlm.py`:

```
Measurements from image-analysis tools that have already run on these
images. Treat them as reliable and do not contradict them:
- water fraction from SAR backscatter: 0.3151
- built-up fraction from optical NDBI: 0.2803

<task prompt>
```

If the model contradicts a measurement anyway, `replan_vlm` re-asks once with a
firmer framing that names the contradiction. That is the conditional
re-planning path, and it is bounded.

**This preamble is a hard constraint on fine-tuning.** A model adapted only on
bare prompts has never seen it and will either ignore the measurements or become
more confident in its own visual guess and contradict them more often. See
`FINETUNING.md` — this is the single most important train/serve parity issue in
the project.

---

## Contracts

Everything crossing a boundary is a frozen, slotted dataclass in `schema.py`.

`ToolSpec` declares what a tool accepts, the tasks it serves, its outputs, and
`allowed_params` as explicit ranges. The registry **enforces** those ranges on
every invocation — an out-of-range parameter raises `ParameterError` rather than
reaching the tool. The problem statement requires that only permitted parameters
be configurable; this is where that is true rather than merely intended.

`TraceStep` carries the tool, version, parameters, outputs, confidence,
duration, the `reason` a parameter was chosen, and `revises` when the step is a
second attempt at an earlier one.

---

## Serving

FastAPI, one process. Uploads, queries, benchmark runs, the live feed and the web
UI all come from it. Backends load **lazily on first use**, so startup needs no
GPU, no running model server and no weights — the UI comes up and reports what is
missing instead of refusing to boot.

Run state lives in an in-process `JobStore`. Trace steps stream to the browser
over a websocket, which sends a keepalive every 15 s so a long model call cannot
go silent long enough for the connection to be dropped.

Backends: `ollama` (local demo), `hf`, `vllm`, `openai_compat` (planned — the
AWS path, see `AWS.md`), and `echo` (a CI test double, never a product choice).

---

## Deliberate deviations

Two, both recorded so they read as decisions rather than omissions.

**A served single-page app, not a Next.js frontend.** The system is one Python
process with no build step. A separate Node toolchain would add a second runtime
and a compile stage to a demo that has to start reliably on unfamiliar hardware.

**An explicit pipeline, not LangGraph.** The flow is fixed and known. An explicit
pipeline makes the emitted trace a direct reading of the code rather than a
rendering of a framework's internal state — and the trace is what gets scored.

---

## Map of the code

| Path | Responsibility |
| --- | --- |
| `agent/controller.py` | the six stages |
| `agent/router.py` | query → task |
| `agent/planner.py` | adaptive parameters, re-planning |
| `agent/registry.py` | tool registry, parameter enforcement |
| `agent/tools/` | the tools themselves |
| `agent/confidence.py` | textual confidence scoring |
| `geo/raster.py` | reading, band selection, dB conversion, previews |
| `eval/` | benchmark harness, metrics, backends, prompts |
| `data/` | dataset ingestion and preparation |
| `api/` | FastAPI app, job store, settings, static UI |
