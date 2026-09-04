# SIH26167 — SatQuery AI

Interactive vision-language assistant for multimodal remote sensing image analysis through text queries.

- **Hackathon:** Smart India Hackathon 2026
- **Organization:** ISRO
- **ID:** SIH26167

## Documentation

| Doc | What it covers |
| --- | --- |
| [`docs/TECHNICAL_OVERVIEW.md`](docs/TECHNICAL_OVERVIEW.md) | Full technical reference: control loop, tools, backends, metrics, API |
| [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md) | System design, core contracts, execution-trace schema, deployment topology |
| [`docs/BASELINE.md`](docs/BASELINE.md) | Phase 0: base-model bake-off protocol and compute plan |
| [`docs/ML_PLAN.md`](docs/ML_PLAN.md) | Fine-tuning stages, Kaggle GPU budget, weekly schedule |
| [`docs/RESEARCH.md`](docs/RESEARCH.md) | Upstream validation with line-level provenance |

## Running the application

The full agentic system runs end to end on base (zero-shot) models — no
fine-tuning required, and no GPU:

```powershell
python -m pip install -e .
ollama pull qwen3-vl:2b-instruct     # 1.9 GB, runs on CPU
satquery serve                       # http://127.0.0.1:8000
```

The default backend is **Ollama** with `qwen3-vl:2b-instruct`. Quantised weights
are what make this practical: a 2B vision-language model is 1.9 GB resident,
against roughly 9 GB for the same class of model in full precision.

### Running a real model

The `hf` backend needs its own extra — the base install deliberately has no torch,
so the app starts on a laptop with no GPU:

```powershell
pip install -e ".[hf]"
satquery serve --backend hf --model Qwen/Qwen2.5-VL-3B-Instruct
```

Both `Qwen/Qwen2.5-VL-3B-Instruct` and `OpenGVLab/InternVL3-2B-hf` load through the
same `AutoModelForImageTextToText` path, so switching model is a flag, not a code
change. Only InternVL's `-hf` repos work; the plain ones use a bespoke `.chat()` API.

Two preflights run before any weights load, because both failures are otherwise
opaque:

| Check | Failure it prevents |
| --- | --- |
| Runtime | `ModuleNotFoundError: No module named 'torch'` mid-sweep — now names the install command instead |
| Host memory | Minutes of swap thrashing before an OOM. A 3B checkpoint needs ~9 GB of RAM on CPU; the app refuses and says so |

The Benchmarks tab greys out any model it cannot actually run and explains why, so
weights being on disk is never mistaken for the model being runnable.

### Model weights

**If the model is not on disk, the server downloads it at startup** — not on the
first query, so a demo never stalls mid-question. The server answers `/api/health`
and serves the UI while the fetch runs, and the header shows live progress.

| Where | What you see |
| --- | --- |
| UI header | `downloading model 43% (5.2 GB / 12.1 GB)` |
| `GET /api/model` | `{"state": "downloading", "percent": 43.0, ...}` |
| `GET /api/health` | the same under `model` |

States are `checking → downloading → ready`, or `error` with a reason. A failed
fetch leaves the server running and reports why; queries then return **503** with
that reason rather than a stack trace.

Weights land in `runs/models/<repo--id>/` as a **plain directory**, not the shared
Hugging Face blob cache. That cache symlinks blobs into snapshot folders, which
needs Developer Mode or admin rights on Windows and otherwise fails mid-download
with `WinError 1314`. A plain directory works everywhere and is portable — copy it
to an offline demo machine and the server finds it. An existing hub cache is still
reused if you already have the model.

Pre-fetch ahead of time (the offline-demo insurance policy):

```powershell
satquery models pull Qwen/Qwen2.5-VL-3B-Instruct
satquery models status Qwen/Qwen2.5-VL-3B-Instruct
```

| Flag / variable | Effect |
| --- | --- |
| `--no-preload` / `SATQUERY_PRELOAD=0` | Load on first query instead of at startup |
| `--no-download` / `SATQUERY_ALLOW_DOWNLOAD=0` | Fail rather than fetch missing weights |
| `--revision` / `SATQUERY_REVISION` | Pin a model revision |
| `SATQUERY_MODELS_DIR` | Where weights are kept (default `runs/models`) |
| `HF_HUB_OFFLINE=1` | No network; weights must already be present |

The UI has four tabs:

| Tab | What it does |
| --- | --- |
| **Analyse** | Drop one or two images, ask a question, watch the execution trace stream live, inspect visual evidence, download a JSON report |
| **Live feed** | Search recent Sentinel-1/2 acquisitions, select one or two scenes, load them onto a shared UTM grid ready to analyse |
| **Benchmarks** | Pull the prescribed datasets, choose models, run a model-by-benchmark matrix with results in place |
| **Registry** | The tools the controller selects from, grouped by category, with every permitted parameter and its constraint |

### What the system does with your input

The input configuration is **inferred, not declared**. One image is a single
scene; two images of the same modality are bi-temporal; two of differing
modality are a co-registered optical–SAR pair. The Analyse tab shows what was
inferred and which compatibility checks passed.

| Query | Route | Tools, in order |
| --- | --- | --- |
| "Describe the land cover…" | `caption` | `vlm_caption` |
| "Highlight the water body" | `grounding` | `vlm_grounding` → box overlay |
| "How many buildings?" | `vqa` | `vlm_vqa` |
| "What changed between these dates?" | `change_caption` | `change_mask` → `vlm_change_caption` |
| "Has built-up increased?" | `change_vqa` | `change_mask` → `vlm_change_vqa` |
| optical + SAR pair | `crossmodal_vqa` | `optical_indices` + `sar_indices` → `vlm_crossmodal_vqa` |

Deterministic specialists always run **before** the VLM, and their measurements
are injected into its prompt. That is what makes an answer evidence-grounded
rather than a guess, and it is why the trace shows a changed-area fraction next
to the sentence describing it.

## Benchmark harness

Phase 0 is a zero-shot bake-off across RSVQA, VRSBench (VQA / caption / referring)
and CDVQA. No training, no weights required to develop against it.

```powershell
python -m pip install -e .          # add [hf] or [vllm] for real backends

# List dataset adapters
satquery bench adapters

# Check a download against its config before spending GPU time
satquery bench validate --config configs/bench/vrsbench_vqa.yaml

# Fetch a benchmark from its official source
satquery data pull rsvqa_lr

# Score it
satquery bench run --config configs/bench/*.yaml `
  --backend ollama --model qwen3-vl:2b-instruct --limit 200
```

Every run writes `predictions.jsonl` and `metrics.json` under `runs/`, and appends to
`runs/results.csv` — the single source for the submission's results table.

**Run `bench validate` first.** Benchmark JSON keys differ between releases; the
adapters resolve fields through candidate keys, and `validate` reports what it found
so a mismatch is fixed by editing YAML rather than code.

Put datasets under `data/` (git-ignored) as laid out in `docs/BASELINE.md`.

## Local setup

Requires Python 3.11+.

```powershell
python -m pip install pre-commit ruff pytest detect-secrets
python -m pre_commit install --hook-type pre-commit --hook-type commit-msg --hook-type pre-push
python -m pre_commit run --all-files
```

After that, every commit formats and lints staged files, rejects secrets/private keys/huge binaries, and checks Conventional Commit messages (`feat:`, `fix:`, `docs:`, `ci:`, `chore:`, ...). GitHub Actions also runs Gitleaks over history.

Do not use `--no-verify` unless you have a specific reason.

## CI

On every push to `main`, every pull request, and manual `workflow_dispatch`:

| Workflow | What it does |
| --- | --- |
| `Pre-commit` | Same hooks as local `pre-commit` |
| `CI` | Ruff lint/format, pytest when tests exist |
| `Secret scan` | Gitleaks over git history |

Dependabot opens weekly PRs for GitHub Actions and pip.

## Agent rules and triggers

Cursor loads `.cursor/rules/` automatically. Project hooks in `.cursor/hooks.json` run on:

- **sessionStart** — inject SatQuery working context
- **beforeShellExecution** — block force-push, hard reset, `git clean -f`, recursive `rm`
- **preToolUse** (Write / StrReplace / EditNotebook) — block `.env`, keys, weights, rasters
