# OSS-Grounded Design Validation

Every load-bearing decision below was checked against fetched upstream source, not
documentation summaries. Provenance is `repo@version path:line`.

| # | Decision | Verdict | Provenance |
|---|---|---|---|
| 1 | `AutoModelForImageTextToText` serves **both** Qwen2.5-VL and InternVL | **ALIGNED** | transformers@5.16.1 `src/transformers/models/auto/modeling_auto.py:1121,1149` — both `("internvl", "InternVLForConditionalGeneration")` and `("qwen2_5_vl", "Qwen2_5_VLForConditionalGeneration")` sit in `MODEL_FOR_IMAGE_TEXT_TO_TEXT_MAPPING_NAMES` (line 1079) |
| 2 | One-call `processor.apply_chat_template(..., tokenize=True, return_dict=True)` replaces our two-step `apply_chat_template(tokenize=False)` → `processor(text=, images=)` | **ADOPT** | transformers@5.16.1 `docs/source/en/model_doc/internvl.md:102`, `qwen2_5_vl.md:95` |
| 3 | Local files passed as `{"type": "image", "path": "<abs path>"}` | **ADOPT** | transformers@5.16.1 `src/transformers/processing_utils.py:2144` — accepted keys are `["image", "url", "path", "base64"]` |
| 4 | Resolution capped through processor `min_pixels` / `max_pixels` at `from_pretrained`, not by pre-resizing ourselves | **ADOPT** | transformers@5.16.1 `docs/source/en/model_doc/qwen2_5_vl.md:218-230` |
| 5 | Batch several conversations by passing a list of message lists with `padding=True` | **ADOPT** | transformers@5.16.1 `docs/source/en/model_doc/internvl.md:177` |
| 6 | InternVL's HF-native checkpoint is the `-hf` suffix line: `OpenGVLab/InternVL3-2B-hf` | **ADOPT** | HF API `models/OpenGVLab/InternVL3-2B-hf` → `architectures: ["InternVLForConditionalGeneration"]`, `model_type: internvl`. The non-`-hf` repos use a bespoke `.chat()` API behind `trust_remote_code` and are **not** interchangeable |
| 7 | Benchmark matrix loops **model-outer, dataset-inner**, building each model once and reusing it across datasets | **ALIGNED** | VLMEvalKit@main `run.py:624` (model loop) then `:658` (dataset loop) |
| 8 | One result file per `(model, dataset)` cell, skipped when already present, so a run resumes | **ADOPT** | VLMEvalKit@main `run.py:645-672` — `get_pred_file_path(...)` then continue-if-exists |
| 9 | Live imagery via Planetary Computer's own `rendered_preview` / `tilejson` assets rather than self-hosting TiTiler | **ADOPT** (avoids **REINVENTING**) | STAC item assets carry both; `GET rendered_preview` → `HTTP 200, image/png, 2.1 MB`. PC already runs the tiler, so standing one up would be reinvention |

Convergence: decisions 2–5 are the single documented path for both model families in
the same upstream release, which is as strong a default as this gets.

## Licenses

| Project | License | Use here |
|---|---|---|
| transformers 5.16.1 | Apache-2.0 | dependency |
| VLMEvalKit (main) | Apache-2.0 | orchestration **pattern** only, no code copied |
| Qwen2.5-VL-3B-Instruct | Apache-2.0 | model weights |
| **InternVL3-2B-hf** | **"other"** | ⚠️ Hub metadata reports `other`. Read the model card terms before relying on it for the submission. Qwen2.5-VL is the unencumbered default. |

## Consequences for our code

- `HFBackend` collapses from a hand-rolled two-step to the upstream one-call form, and
  stops resizing images itself — the processor's `max_pixels` does it correctly for
  each architecture's patching scheme.
- `BackendConfig` gains `min_pixels` / `max_pixels`; `max_side` is retired for the HF
  path because pre-resizing fights the processor.
- vLLM stays available but is no longer the primary path, per the current requirement.
- The live feed is a thin STAC client plus a windowed COG reader; no tile server.
