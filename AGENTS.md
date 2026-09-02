# SIH26167 — SatQuery AI

Smart India Hackathon 2026 (ISRO). Interactive vision-language assistant for multimodal remote sensing analysis via text queries.

## Layout

- `src/` — application code (to be added)
- `tests/` — pytest
- `notebooks/` — exploration only
- `.github/workflows/` — CI, pre-commit, secret scan
- `.cursor/` — agent rules and hooks
- `.pre-commit-config.yaml` — local git hooks

## Commands

```bash
python -m pip install pre-commit ruff pytest detect-secrets
python -m pre_commit install --hook-type pre-commit --hook-type commit-msg --hook-type pre-push
python -m pre_commit run --all-files
```

Commit messages: Conventional Commits (`feat:`, `fix:`, `docs:`, `ci:`, `chore:`, …).

Do not commit secrets, `.env` files, rasters, or model weights. Never use `--no-verify` unless the user asks.
