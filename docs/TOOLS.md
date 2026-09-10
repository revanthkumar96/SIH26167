# Tools

Nine tools are registered today: three deterministic **measurement** tools and
six **model** tools. A tenth, a land-cover CNN, is planned — see `CNN.md`.

The split matters. Measurement tools compute quantities from pixels and are
reproducible, fast and auditable. Model tools generate language and are none of
those things. Keeping them distinct is what lets the trace show *where a number
came from*, which is the difference between evidence and assertion.

---

## The registry contract

Every tool declares a `ToolSpec`:

| field | meaning |
| --- | --- |
| `name`, `version` | identity, both shown in the trace |
| `accepts` | which input configuration it is valid for |
| `tasks` | which routed tasks it serves |
| `outputs` | keys it promises to produce |
| `allowed_params` | **name → (min, max)**, enforced on every call |
| `kind` | `measurement` or `model` |
| `cost` | `fast` or `heavy` |
| `emits_evidence` | whether it contributes visual evidence |

`allowed_params` is enforced by `Tool.invoke`, which raises `ParameterError`
before the tool body runs. The problem statement requires that only permitted
task parameters be configurable by the agent; this is the mechanism that makes
that true rather than aspirational.

---

## Measurement tools

### `change_mask` — bi-temporal

Absolute-difference change detection between two co-registered dates,
thresholded by Otsu unless overridden.

| | |
| --- | --- |
| Accepts | `bitemporal_pair` |
| Outputs | `changed_area_frac`, `change_location`, `direction`, `mask_uri` |
| Params | `threshold` (0.02–0.9), `min_area_frac` (0.0–0.2) |
| Cost | fast |

`min_area_frac` suppresses speckle-sized components. It is derived per run
rather than fixed:

```
min_area_frac = MIN_CHANGE_AREA_M2 / (pixels × gsd²)      clamped to [0, 0.2]
MIN_CHANGE_AREA_M2 = 10_000       # a 100 m × 100 m parcel
```

So the floor is a real-world area, not a pixel count — the same setting means
the same thing at 10 m and at sub-metre resolution.

`direction` reports increase, decrease or unchanged; `QUIET_CHANGE = 0.02` is
the fraction below which a scene is called unchanged rather than noisy.

### `optical_indices` — cross-modal

NDWI and NDBI from multispectral bands.

| | |
| --- | --- |
| Accepts | `crossmodal_pair` |
| Outputs | `applicable`, `water_fraction`, `builtup_fraction`, `water_mask_uri` |
| Params | `water_threshold` (−1.0–1.0), `builtup_threshold` (−1.0–1.0) |
| Cost | fast |

```
NDWI = (green − nir)  / (green + nir)
NDBI = (swir  − nir)  / (swir  + nir)
```

**It reports itself inapplicable rather than inventing an index.** Given an RGB
image there is no NIR band, so there is no NDWI — `applicable: false` is the
honest output and the run continues without it.

This matters for the hidden evaluation set. Cartosat-2S has no SWIR band, so
**NDBI is unavailable there**. Built-up must then come from SAR, or from the
planned CNN. That is not a gap to apologise for — it is precisely the
cross-modal complementarity the problem statement is testing.

### `sar_indices` — cross-modal

Water and built-up extent from SAR backscatter in dB.

| | |
| --- | --- |
| Accepts | `crossmodal_pair` |
| Outputs | `water_fraction`, `builtup_fraction`, `water_location`, `water_mask_uri` |
| Params | `builtup_percentile` (80.0–99.5) |
| Cost | fast |

Water is a specular reflector, so it returns almost nothing to the sensor and
appears very dark — an Otsu threshold on the dark tail finds it. Built-up
surfaces produce strong double-bounce returns and appear bright, so they come
from a high percentile.

`builtup_percentile` adapts to resolution: `92.0` when GSD ≥ 8 m, `95.0` below.
Coarser pixels mix more surfaces, so the bright tail is less pure and the
threshold has to be more permissive.

**The dB question is load-bearing.** SAR products are distributed in two forms:
linear power (RTC gamma0, as Planetary Computer serves it) and decibels
(as BigEarthNet stores it). Thresholding linear power as though it were dB
produced a reading of 94% water on a Mumbai scene where the optical NDWI said
31.5%. `looks_like_linear_backscatter` detects the linear case — ≤2 bands, float
dtype, non-negative, p99/median > 8 — and converts only then. Because dB data is
negative, the guard is self-correcting on BigEarthNet input.

---

## Model tools

One class, `VLMTool`, instantiated six times. Instances differ only by task,
accepted configuration, and (eventually) which adapter they name.

| tool | accepts | task | outputs |
| --- | --- | --- | --- |
| `vlm_vqa` | single | VQA | `answer` |
| `vlm_caption` | single | CAPTION | `answer` |
| `vlm_grounding` | single | GROUNDING | `answer`, `bbox` |
| `vlm_change_vqa` | bi-temporal | CHANGE_VQA | `answer` |
| `vlm_change_caption` | bi-temporal | CHANGE_CAPTION | `answer` |
| `vlm_crossmodal_vqa` | cross-modal | CROSSMODAL_VQA | `answer` |

All accept `max_new_tokens` (1–512) and `temperature` (0.0–1.0). Temperature is
held at 0 so benchmark runs are reproducible.

Token budgets are set per task, and were corrected after observing real
truncation rather than guessed:

| task | budget | reason |
| --- | --- | --- |
| caption, change caption | 200 | the model uses ~120 tokens for two sentences |
| cross-modal | 128 | needs to cite two sensors |
| grounding | 64 | a box and nothing else |
| closed question | 24 | yes/no, a count, a direction |
| open question | 64 | |

A reply cut off at the budget is a defect, not an answer. The Ollama backend
reports `done_reason: "length"`, the tool surfaces `truncated: true` in the
trace, and `replan_truncated` retries once at double the budget. Truncation is
fixed **before** content is judged — re-asking a half sentence for contradicting
a measurement would be unfair to the model.

The per-instance `adapter` field is how fine-tuned weights slot in later without
touching the controller. A tool names an adapter; the backend routes to it.

---

## Adaptive parameters

Empty `params` in a trace is a weak showing when the criterion is "valid
parameters". Every tool that has parameters gets them derived from the run's own
properties, with a recorded reason:

| tool | derived from |
| --- | --- |
| `change_mask` | pixel count and GSD → real-world minimum area |
| `sar_indices` | GSD → percentile |
| `vlm_*` | task, and whether the question is closed-form |

`CLOSED_QUESTION` matches questions beginning *is/are/was/were/does/do/did/
has/have/can/will*, containing *how many* or *how much*, or containing
*increased/decreased/unchanged*. Those get a small budget because a long answer
to a yes/no question scores worse under exact match, not better.

---

## Re-planning

Three conditions can produce exactly one revision of a step, capped by
`MAX_REVISIONS = 2` across a run:

| condition | response |
| --- | --- |
| change mask found almost nothing | retry at `RETRY_THRESHOLD_SCALE = 0.55` of the threshold |
| reply hit the token budget | retry at double the budget |
| answer contradicts a measurement | re-ask with a firmer evidence framing |

Revisions appear in the trace with `revises` pointing at the step they replace,
so the trace shows the agent noticing and correcting rather than hiding it.

---

## Adding a tool

1. Implement `Tool` with a `ToolSpec` — declare `allowed_params` honestly, since
   the registry enforces them.
2. Register it in `agent/tools/__init__.py:default_registry()`.
3. If it should run before a VLM task, add it to `_PRECURSORS` in the controller.
4. If its outputs should reach the model, add them to `_ARTIFACT_MAP` and
   `_EVIDENCE_LABELS`.

Step 4 is the one that is easy to forget and silently makes a new measurement
tool useless: it will run, and its numbers will never reach the model.
