"""Inject SatQuery AI working context at session start."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from hooklib import emit, read_payload

CONTEXT = """
This repository is SIH26167 SatQuery AI (ISRO, Smart India Hackathon 2026).
Work inside the git root that contains .pre-commit-config.yaml.
Install and keep git hooks enabled: pre-commit, commit-msg, pre-push.
Do not commit secrets, satellite rasters, or model weights.
Match Conventional Commits and the Cursor rules in .cursor/rules/.
""".strip()


def main() -> int:
    read_payload()
    emit({"additional_context": CONTEXT})
    return 0


if __name__ == "__main__":
    sys.exit(main())
