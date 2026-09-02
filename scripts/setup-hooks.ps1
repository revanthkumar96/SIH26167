# Install local git hooks for SatQuery AI (SIH26167).
python -m pip install --upgrade pre-commit ruff pytest detect-secrets
python -m pre_commit install --hook-type pre-commit --hook-type commit-msg --hook-type pre-push
python -m pre_commit run --all-files
Write-Host "Hooks installed. Commits now run lint, secret scan, and conventional-commit checks."
