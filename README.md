# SIH26167 — SatQuery AI

Interactive vision-language assistant for multimodal remote sensing image analysis through text queries.

- **Hackathon:** Smart India Hackathon 2026
- **Organization:** ISRO
- **ID:** SIH26167

## Documentation

| Doc | What it covers |
| --- | --- |
| [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md) | System design, core contracts, execution-trace schema, deployment topology |
| [`docs/BASELINE.md`](docs/BASELINE.md) | Phase 0: base-model bake-off protocol and compute plan |
| [`docs/ML_PLAN.md`](docs/ML_PLAN.md) | Fine-tuning stages, Kaggle GPU budget, weekly schedule |

## Running the application

The full agentic system runs end to end on base (zero-shot) models — no
fine-tuning required. It starts with no GPU and no model download:

```powershell
python -m pip install -e .
satquery serve                       # http://127.0.0.1:8000
```

That launches with the `echo` backend: no weights, deterministic replies, every
other part of the system real. Point it at an actual model when you have one:

```powershell
satquery serve --backend vllm --model Qwen/Qwen2.5-VL-3B-Instruct
```

The UI has three tabs:

| Tab | What it does |
| --- | --- |
| **Analyse** | Drop one or two images, ask a question, watch the execution trace stream live, inspect visual evidence, download a JSON report |
| **Benchmarks** | Run RSVQA / VRSBench / CDVQA against the same backend and prompts, with results streaming in |
| **Tool registry** | The predefined registry the controller selects from, including each tool's permitted parameters |

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

# Dry run with the no-model backend (no GPU, no weights)
satquery bench run --config configs/bench/*.yaml --backend echo

# The real sweep
satquery bench run --config configs/bench/*.yaml `
  --backend vllm --model Qwen/Qwen2.5-VL-3B-Instruct --limit 2000
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
