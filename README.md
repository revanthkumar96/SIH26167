# SIH26167 — SatQuery AI

Interactive vision-language assistant for multimodal remote sensing image analysis through text queries.

- **Hackathon:** Smart India Hackathon 2026
- **Organization:** ISRO
- **ID:** SIH26167

This repo currently holds project quality gates (git hooks, Cursor rules, GitHub Actions). Application code will land under `src/`.

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
