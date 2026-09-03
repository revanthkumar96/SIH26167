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
