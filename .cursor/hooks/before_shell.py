"""Gate destructive git and other high-risk shell commands."""

from __future__ import annotations

import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from hooklib import emit, read_payload

DENY_PATTERNS = [
    (
        r"\bgit\s+push\b.*(--force|--force-with-lease|-f)\b",
        "Force-push is blocked by project hooks.",
    ),
    (r"\bgit\s+reset\s+--hard\b", "git reset --hard is blocked by project hooks."),
    (r"\bgit\s+clean\s+-[a-zA-Z]*f", "git clean -f is blocked by project hooks."),
    (r"\bgit\s+checkout\s+\.(?:\s|$)", "git checkout . is blocked by project hooks."),
    (r"\bgit\s+restore\s+\.(?:\s|$)", "git restore . is blocked by project hooks."),
    (
        r"\bgit\s+branch\s+-D\b",
        "Deleting branches with -D is blocked by project hooks.",
    ),
    (
        r"\brm\s+(-[a-zA-Z]*r[a-zA-Z]*f|-[a-zA-Z]*f[a-zA-Z]*r)\b",
        "Recursive force-delete is blocked.",
    ),
]

ASK_PATTERNS = [
    (r"\bgit\s+push\b", "Review this git push before it runs."),
    (r"\bgit\s+commit\b.*--no-verify\b", "Skipping git hooks needs explicit approval."),
]


def _command(payload: dict) -> str:
    for key in ("command", "cmd"):
        value = payload.get(key)
        if isinstance(value, str):
            return value
    tool_input = payload.get("tool_input")
    if isinstance(tool_input, dict):
        value = tool_input.get("command")
        if isinstance(value, str):
            return value
    return ""


def main() -> int:
    command = _command(read_payload())
    for pattern, message in DENY_PATTERNS:
        if re.search(pattern, command, flags=re.IGNORECASE):
            emit(
                {
                    "permission": "deny",
                    "user_message": message,
                    "agent_message": message
                    + " Ask the user before using a destructive git command.",
                }
            )
            return 0
    for pattern, message in ASK_PATTERNS:
        if re.search(pattern, command, flags=re.IGNORECASE):
            emit(
                {
                    "permission": "ask",
                    "user_message": message,
                    "agent_message": message,
                }
            )
            return 0
    emit({"permission": "allow"})
    return 0


if __name__ == "__main__":
    sys.exit(main())
