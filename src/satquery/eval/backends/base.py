"""Backend interface shared by the eval harness and the serving path.

Candidates in the bake-off come from different model families with different prompt
formats, so requests are expressed as OpenAI-style chat messages and each backend
applies its own model's chat template. That keeps the comparison fair: no candidate
wins or loses because of a hand-written prompt-format hack.
"""

from __future__ import annotations

import base64
import io
from abc import ABC, abstractmethod
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from satquery.schema import GenerationRequest


@dataclass(frozen=True, slots=True)
class BackendConfig:
    """Everything a backend needs, kept model-agnostic."""

    model: str
    dtype: str = "auto"
    max_side: int | None = 1024
    batch_size: int = 8
    temperature: float = 0.0
    tensor_parallel_size: int = 1
    gpu_memory_utilization: float = 0.90
    max_model_len: int | None = None
    trust_remote_code: bool = True
    extra: dict[str, Any] | None = None


def build_messages(request: GenerationRequest, max_side: int | None) -> list[dict]:
    """OpenAI-style messages with images inlined as base64 data URLs.

    Images come first, then the text, matching how every candidate's chat template
    expects interleaved multimodal turns.
    """
    from satquery.eval.images import load_image

    content: list[dict[str, Any]] = []
    for image_path in request.images:
        image = load_image(image_path, max_side=max_side)
        buffer = io.BytesIO()
        image.save(buffer, format="JPEG", quality=92)
        encoded = base64.b64encode(buffer.getvalue()).decode("ascii")
        content.append(
            {
                "type": "image_url",
                "image_url": {"url": f"data:image/jpeg;base64,{encoded}"},
            }
        )
    content.append({"type": "text", "text": request.prompt})
    return [{"role": "user", "content": content}]


def load_pil_images(paths: Sequence[Path], max_side: int | None) -> list:
    from satquery.eval.images import load_image

    return [load_image(p, max_side=max_side) for p in paths]


class VLMBackend(ABC):
    """Turns generation requests into raw text, in order."""

    name: str = "base"

    def __init__(self, config: BackendConfig) -> None:
        self.config = config

    @abstractmethod
    def generate(self, requests: Sequence[GenerationRequest]) -> list[str]:
        """Return one raw output string per request, order preserved."""

    def close(self) -> None:  # noqa: B027 - optional override
        """Release GPU memory. Safe to call more than once."""

    def __enter__(self) -> VLMBackend:
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()

    def describe(self) -> dict[str, Any]:
        return {"backend": self.name, "model": self.config.model}
