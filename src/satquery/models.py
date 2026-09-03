"""Model presence checks and downloading.

The server must be usable on a machine that has never seen the weights: if the
configured model is not on disk, it is fetched at startup rather than on the first
query, so a demo does not stall for minutes on its first question.

Deliberately independent of torch, transformers and vLLM. Presence is a question
about the Hugging Face cache, and answering it must not require importing a deep
learning stack.
"""

from __future__ import annotations

import os
import threading
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

ProgressCallback = Callable[["DownloadProgress"], None]

#: Backends that need weights. Everything else starts instantly.
MODEL_BACKENDS = frozenset({"hf", "vllm"})

#: Models the UI can select and benchmark. Every vision-language entry resolves
#: through AutoModelForImageTextToText on the HF backend -- upstream registers both
#: families in one mapping, so they need no per-model code.
#:   pattern: transformers@5.16.1 src/transformers/models/auto/modeling_auto.py:1121,1149
#:
#: InternVL is listed only in its "-hf" form. The plain OpenGVLab repos ship a
#: bespoke .chat() API behind trust_remote_code and are not interchangeable.
MODEL_CATALOG: tuple[dict[str, Any], ...] = (
    {
        "id": "echo",
        "label": "Echo baseline",
        "backend": "echo",
        "model": "echo",
        "params": "—",
        "size_gb": 0.0,
        "license": "n/a",
        "description": (
            "Deterministic stub. No weights, no GPU -- used for CI, offline "
            "fallback, and proving the pipeline independently of any model."
        ),
        "tags": ("baseline", "no-gpu"),
    },
    {
        "id": "qwen25-vl-3b",
        "label": "Qwen2.5-VL 3B",
        "backend": "hf",
        "model": "Qwen/Qwen2.5-VL-3B-Instruct",
        "params": "3B",
        "size_gb": 7.5,
        "license": "Apache-2.0",
        "description": (
            "Primary vision-language model. Native interleaved multi-image input "
            "and trained box output, which the paired and grounding tasks need."
        ),
        "tags": ("vlm", "recommended", "multi-image", "grounding"),
    },
    {
        "id": "internvl3-2b-hf",
        "label": "InternVL3 2B",
        "backend": "hf",
        "model": "OpenGVLab/InternVL3-2B-hf",
        "params": "2B",
        "size_gb": 4.4,
        "license": "see model card",
        "description": (
            "Comparison model for the bake-off. Different vision stack, so it is "
            "a genuine alternative rather than a variant."
        ),
        "tags": ("vlm", "comparison", "multi-image"),
    },
    {
        "id": "internvl3-8b-hf",
        "label": "InternVL3 8B",
        "backend": "hf",
        "model": "OpenGVLab/InternVL3-8B-hf",
        "params": "8B",
        "size_gb": 16.0,
        "license": "see model card",
        "description": "Higher-capacity InternVL, when GPU memory allows.",
        "tags": ("vlm", "comparison", "large"),
    },
    {
        "id": "qwen25-vl-7b",
        "label": "Qwen2.5-VL 7B",
        "backend": "hf",
        "model": "Qwen/Qwen2.5-VL-7B-Instruct",
        "params": "7B",
        "size_gb": 16.6,
        "license": "Apache-2.0",
        "description": "Higher-capacity Qwen, when GPU memory allows.",
        "tags": ("vlm", "large"),
    },
)

#: Kept for the older attribute name used elsewhere.
RECOMMENDED_MODELS = MODEL_CATALOG


def catalog_entry(model_id: str) -> dict[str, Any] | None:
    """Look a catalog entry up by its short id."""
    return next((m for m in MODEL_CATALOG if m["id"] == model_id), None)


#: Weight shards and the config/tokenizer files needed alongside them. Excluding
#: the other serialisation formats roughly halves the download for repos that
#: ship both safetensors and .bin.
DEFAULT_ALLOW_PATTERNS = (
    "*.safetensors",
    "*.safetensors.index.json",
    "*.json",
    "*.model",
    "*.txt",
    "*.py",
)


@dataclass
class DownloadProgress:
    """Snapshot of an in-flight download, safe to serialise to a client."""

    state: str = "idle"  # idle | checking | downloading | ready | error | skipped
    model: str = ""
    detail: str = ""
    downloaded_bytes: int = 0
    total_bytes: int = 0
    path: str | None = None

    @property
    def percent(self) -> float | None:
        if self.total_bytes <= 0:
            return None
        return round(100.0 * self.downloaded_bytes / self.total_bytes, 1)

    def as_dict(self) -> dict[str, Any]:
        return {
            "state": self.state,
            "model": self.model,
            "detail": self.detail,
            "downloaded_bytes": self.downloaded_bytes,
            "total_bytes": self.total_bytes,
            "percent": self.percent,
            "path": self.path,
        }


def repo_total_bytes(model: str, revision: str | None = None) -> int:
    """Total download size in bytes, or 0 when it cannot be determined.

    Asked of the hub up front rather than inferred from progress bars:
    ``snapshot_download`` only routes its outer file-count bar through
    ``tqdm_class``, so per-file byte counts are not observable that way.
    """
    try:
        from fnmatch import fnmatch

        from huggingface_hub import HfApi

        info = HfApi().model_info(model, revision=revision, files_metadata=True)
    except Exception:
        return 0

    total = 0
    for sibling in info.siblings or []:
        name = sibling.rfilename
        if any(fnmatch(name, pattern) for pattern in DEFAULT_ALLOW_PATTERNS):
            total += sibling.size or 0
    return total


def directory_bytes(directory: Path) -> int:
    """Bytes currently on disk under a directory, ignoring transient errors."""
    total = 0
    if not directory.is_dir():
        return 0
    for path in directory.rglob("*"):
        try:
            if path.is_file():
                total += path.stat().st_size
        except OSError:
            continue
    return total


class _SizePoller:
    """Reports download progress by watching the target directory grow.

    Independent of hub internals, so it keeps working across releases and
    whichever transfer backend is in play.
    """

    def __init__(
        self,
        directory: Path,
        progress: DownloadProgress,
        on_update: ProgressCallback | None,
        interval: float = 0.5,
    ) -> None:
        self._directory = directory
        self._progress = progress
        self._on_update = on_update
        self._interval = interval
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, daemon=True)

    def _run(self) -> None:
        while not self._stop.wait(self._interval):
            self._progress.downloaded_bytes = directory_bytes(self._directory)
            if self._on_update is not None:
                self._on_update(self._progress)

    def __enter__(self) -> _SizePoller:
        self._thread.start()
        return self

    def __exit__(self, *exc: object) -> None:
        self._stop.set()
        self._thread.join(timeout=2.0)


def _silent_tqdm():
    """A tqdm subclass that renders nothing, for use as ``tqdm_class``."""
    from tqdm.auto import tqdm as base_tqdm

    class _SilentTqdm(base_tqdm):
        def __init__(self, *args: Any, **kwargs: Any) -> None:
            kwargs["disable"] = True
            super().__init__(*args, **kwargs)

    return _SilentTqdm


#: Where weights land when no explicit directory is given.
DEFAULT_MODELS_DIR = Path("runs/models")


def needs_model(backend: str) -> bool:
    return backend in MODEL_BACKENDS


def local_model_dir(model: str, models_dir: Path | None = None) -> Path:
    """Directory a repo's weights are materialised into.

    Weights go into a plain directory rather than the shared hub cache. The hub
    cache symlinks blobs into snapshot folders, which needs Developer Mode or
    admin rights on Windows and otherwise fails mid-download with WinError 1314.
    A plain directory works on every platform and is portable: copy it to an
    offline demo machine and the server finds it.
    """
    root = Path(models_dir) if models_dir else DEFAULT_MODELS_DIR
    return root / model.replace("/", "--")


def _looks_complete(directory: Path) -> bool:
    """A directory counts as present when it holds a config and some weights."""
    if not directory.is_dir():
        return False
    if not (directory / "config.json").is_file():
        return False
    return any(directory.glob("*.safetensors")) or any(directory.glob("*.bin"))


def is_local_path(model: str) -> bool:
    """True when the identifier points at a directory rather than a hub repo."""
    return Path(model).expanduser().is_dir()


def offline_mode() -> bool:
    """Whether the environment forbids network access to the hub."""
    return os.environ.get("HF_HUB_OFFLINE", "").strip().lower() in {"1", "true", "yes"}


def cached_path(
    model: str, revision: str | None = None, models_dir: Path | None = None
) -> Path | None:
    """Local path for a model already fully present, else ``None``."""
    expanded = Path(model).expanduser()
    if expanded.is_dir():
        return expanded

    target = local_model_dir(model, models_dir)
    if _looks_complete(target):
        return target

    # Someone may already hold the model in the shared hub cache from other work.
    # Reuse it rather than downloading gigabytes a second time.
    try:
        from huggingface_hub import snapshot_download
    except ImportError:
        return None

    try:
        return Path(
            snapshot_download(
                model,
                revision=revision,
                local_files_only=True,
                allow_patterns=list(DEFAULT_ALLOW_PATTERNS),
            )
        )
    except Exception:
        # Any failure here means "not usable from cache" -- a missing repo, a
        # partial download, or no cache at all. The caller downloads instead.
        return None


def describe_catalog(models_dir: Path | None = None) -> list[dict[str, Any]]:
    """Return the recommended model list with local presence and size."""
    entries: list[dict[str, Any]] = []
    for spec in RECOMMENDED_MODELS:
        backend = str(spec["backend"])
        model = str(spec["model"])
        ready = backend == "echo"
        path: Path | None = None
        size_bytes = 0
        if needs_model(backend):
            path = cached_path(model, models_dir=models_dir)
            ready = path is not None
            if not ready:
                partial = local_model_dir(model, models_dir)
                size_bytes = directory_bytes(partial)
        entry = dict(spec)
        entry["ready"] = ready
        entry["path"] = str(path) if path else None
        entry["size_bytes"] = size_bytes
        entries.append(entry)
    return entries


def ensure_model(
    model: str,
    revision: str | None = None,
    on_update: ProgressCallback | None = None,
    allow_download: bool = True,
    models_dir: Path | None = None,
) -> DownloadProgress:
    """Make sure the weights are on disk, downloading them if they are not.

    Never raises: the result carries the outcome, because a failed download must
    leave the server running and able to report why rather than refusing to start.
    """
    progress = DownloadProgress(state="checking", model=model)

    def emit() -> None:
        if on_update is not None:
            on_update(progress)

    emit()

    existing = cached_path(model, revision, models_dir)
    if existing is not None:
        progress.state = "ready"
        progress.path = str(existing)
        progress.detail = "already present"
        emit()
        return progress

    if not allow_download:
        progress.state = "error"
        progress.detail = "model is not present and downloads are disabled"
        emit()
        return progress

    if offline_mode():
        progress.state = "error"
        progress.detail = (
            "model is not present and HF_HUB_OFFLINE is set; "
            "pre-fetch it with 'satquery models pull'"
        )
        emit()
        return progress

    try:
        from huggingface_hub import snapshot_download
    except ImportError:
        progress.state = "error"
        progress.detail = (
            "huggingface_hub is not installed; install the 'hf' or 'vllm' extra"
        )
        emit()
        return progress

    target = local_model_dir(model, models_dir)
    target.parent.mkdir(parents=True, exist_ok=True)

    progress.state = "downloading"
    progress.detail = f"fetching {model}"
    progress.total_bytes = repo_total_bytes(model, revision)
    progress.downloaded_bytes = directory_bytes(target)
    emit()

    try:
        with _SizePoller(target, progress, on_update):
            path = snapshot_download(
                model,
                revision=revision,
                allow_patterns=list(DEFAULT_ALLOW_PATTERNS),
                local_dir=str(target),
                # Progress comes from the poller, so hub's own bar is just noise
                # in a server log.
                tqdm_class=_silent_tqdm(),
            )
    except Exception as exc:
        progress.state = "error"
        progress.detail = f"{type(exc).__name__}: {exc}"
        emit()
        return progress

    progress.downloaded_bytes = directory_bytes(target)

    progress.state = "ready"
    progress.path = str(path)
    progress.detail = "downloaded"
    emit()
    return progress
