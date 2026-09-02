"""Block writes to secret and credential files."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from hooklib import emit, read_payload

BLOCKED_NAMES = {
    ".env",
    ".env.local",
    ".env.production",
    "credentials.json",
    "service-account.json",
    "secrets.toml",
    "id_rsa",
    "id_ed25519",
}

BLOCKED_SUFFIXES = (
    ".pem",
    ".p12",
    ".pfx",
    ".key",
    ".ckpt",
    ".pt",
    ".pth",
    ".tif",
    ".tiff",
    ".npy",
    ".npz",
    ".h5",
    ".hdf5",
)


def _paths(payload: dict) -> list[str]:
    found: list[str] = []
    for key in ("file_path", "path", "target_notebook"):
        value = payload.get(key)
        if isinstance(value, str):
            found.append(value)
    tool_input = payload.get("tool_input")
    if isinstance(tool_input, dict):
        for key in ("path", "file_path", "target_notebook"):
            value = tool_input.get(key)
            if isinstance(value, str):
                found.append(value)
    return found


def _blocked(path: str) -> bool:
    name = Path(path).name.lower()
    if name in BLOCKED_NAMES:
        return True
    lowered = path.replace("\\", "/").lower()
    if any(lowered.endswith(suffix) for suffix in BLOCKED_SUFFIXES):
        return True
    return "/.ssh/" in lowered


def main() -> int:
    for path in _paths(read_payload()):
        if _blocked(path):
            message = (
                f"Refusing to write {path}. Secrets, keys, weights, and rasters "
                "must stay out of the repo unless the user explicitly overrides."
            )
            emit(
                {
                    "permission": "deny",
                    "user_message": message,
                    "agent_message": message,
                }
            )
            return 0
    emit({"permission": "allow"})
    return 0


if __name__ == "__main__":
    sys.exit(main())
