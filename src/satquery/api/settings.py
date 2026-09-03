"""Runtime settings for the application.

Everything is environment-overridable so the same image runs on a laptop with the
echo backend and on a GPU host with vLLM, with no code change.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path


def _env(name: str, default: str) -> str:
    return os.environ.get(name, default)


def _flag(name: str, default: bool) -> bool:
    raw = os.environ.get(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


@dataclass
class Settings:
    """Application configuration."""

    #: echo (no model, no GPU) | hf (correctness reference) | vllm (production)
    backend: str = field(default_factory=lambda: _env("SATQUERY_BACKEND", "echo"))
    model: str = field(default_factory=lambda: _env("SATQUERY_MODEL", "echo"))
    dtype: str = field(default_factory=lambda: _env("SATQUERY_DTYPE", "auto"))
    max_side: int = field(
        default_factory=lambda: int(_env("SATQUERY_MAX_SIDE", "1024"))
    )
    workspace: Path = field(
        default_factory=lambda: Path(_env("SATQUERY_WORKSPACE", "runs"))
    )
    bench_config_dir: Path = field(
        default_factory=lambda: Path(_env("SATQUERY_BENCH_CONFIGS", "configs/bench"))
    )
    #: Where benchmark datasets are downloaded to and read from.
    data_root: Path = field(
        default_factory=lambda: Path(_env("SATQUERY_DATA_ROOT", "data"))
    )
    max_upload_mb: int = field(
        default_factory=lambda: int(_env("SATQUERY_MAX_UPLOAD_MB", "200"))
    )
    #: Fetch the weights at startup rather than on the first query, so a demo
    #: never stalls for minutes on its first question.
    preload: bool = field(default_factory=lambda: _flag("SATQUERY_PRELOAD", True))
    allow_download: bool = field(
        default_factory=lambda: _flag("SATQUERY_ALLOW_DOWNLOAD", True)
    )
    revision: str | None = field(
        default_factory=lambda: os.environ.get("SATQUERY_REVISION") or None
    )

    @property
    def uploads_dir(self) -> Path:
        return self.workspace / "uploads"

    @property
    def previews_dir(self) -> Path:
        return self.workspace / "previews"

    @property
    def artifacts_dir(self) -> Path:
        return self.workspace / "artifacts"

    @property
    def samples_dir(self) -> Path:
        """Bundled demonstration scenes."""
        return self.workspace / "samples"

    @property
    def models_dir(self) -> Path:
        """Where weights are materialised. Portable: copy it to a demo machine."""
        override = os.environ.get("SATQUERY_MODELS_DIR")
        return Path(override) if override else self.workspace / "models"

    @property
    def results_csv(self) -> Path:
        return self.workspace / "results.csv"

    def ensure_dirs(self) -> None:
        for directory in (self.uploads_dir, self.previews_dir, self.artifacts_dir):
            directory.mkdir(parents=True, exist_ok=True)

    def describe(self) -> dict[str, object]:
        return {
            "backend": self.backend,
            "model": self.model,
            "dtype": self.dtype,
            "max_side": self.max_side,
            "workspace": str(self.workspace),
            "preload": self.preload,
            "allow_download": self.allow_download,
        }
