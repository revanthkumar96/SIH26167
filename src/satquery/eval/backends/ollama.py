"""Ollama backend.

The practical way to run a vision-language model on a laptop. Ollama serves
4-bit quantised weights, so ``qwen3-vl:2b-instruct`` is 1.9 GB resident against
the ~9 GB the same class of model needs in fp32 through transformers. That
difference is what makes a real model usable on a demo machine at all.

Talks to ``POST /api/chat``, whose message carries an ``images`` array of
base64 strings -- multiple images per message, which is exactly what the paired
tasks need.
  pattern: ollama/ollama docs/api.md -- messages[].images, options.temperature,
  options.num_predict

Requests go one at a time: Ollama serialises generation per model anyway, so
batching here would only add latency and memory pressure.
"""

from __future__ import annotations

import base64
import io
import os
from collections.abc import Sequence
from typing import Any

from satquery.eval.backends.base import BackendConfig, VLMBackend
from satquery.schema import GenerationRequest

DEFAULT_HOST = "http://127.0.0.1:11434"
#: Generous: a cold model load on CPU is slow, and the first call pays for it.
REQUEST_TIMEOUT = 600


def host_url() -> str:
    """Ollama endpoint, honouring the same variable the CLI uses."""
    raw = os.environ.get("OLLAMA_HOST", DEFAULT_HOST).strip()
    if not raw:
        return DEFAULT_HOST
    if not raw.startswith(("http://", "https://")):
        raw = f"http://{raw}"
    return raw.rstrip("/")


def server_reachable(timeout: float = 2.0) -> tuple[bool, str]:
    """Whether an Ollama server answers, and its version."""
    import requests

    try:
        response = requests.get(f"{host_url()}/api/version", timeout=timeout)
        response.raise_for_status()
        return True, str(response.json().get("version", "unknown"))
    except Exception as exc:
        return False, f"{type(exc).__name__}: {exc}"


def installed_models(timeout: float = 5.0) -> list[str]:
    """Model tags pulled on the server."""
    import requests

    try:
        response = requests.get(f"{host_url()}/api/tags", timeout=timeout)
        response.raise_for_status()
        return [m["name"] for m in response.json().get("models", [])]
    except Exception:
        return []


def model_present(model: str) -> bool:
    """True when the tag is pulled. ``qwen3-vl:2b`` also matches ``…:2b``."""
    available = installed_models()
    if model in available:
        return True
    # Ollama appends ":latest" when a tag is omitted.
    return ":" not in model and f"{model}:latest" in available


def _error_message(text: str) -> str:
    """Pull the human-readable message out of an Ollama error body.

    Errors arrive as JSON, sometimes with a second JSON document nested inside
    the ``error`` string, so one level of unwrapping is worth doing.
    """
    import json

    try:
        payload = json.loads(text)
    except (ValueError, TypeError):
        return text.strip()[:400]

    error = payload.get("error", payload) if isinstance(payload, dict) else payload
    if isinstance(error, str):
        try:
            inner = json.loads(error)
            if isinstance(inner, dict):
                error = inner.get("error", inner)
        except (ValueError, TypeError):
            return error.strip()[:400]
    if isinstance(error, dict):
        return str(error.get("message", error))[:400]
    return str(error)[:400]


def encode_image(path: Any, max_side: int | None) -> str:
    """Render any supported raster to base64 JPEG.

    Goes through our own loader rather than reading bytes directly: a 12-band
    uint16 GeoTIFF is not something Ollama can decode, and the loader already
    handles band selection and the SAR decibel conversion.
    """
    from satquery.eval.images import load_image

    image = load_image(path, max_side=max_side)
    buffer = io.BytesIO()
    image.save(buffer, format="JPEG", quality=92)
    return base64.b64encode(buffer.getvalue()).decode("ascii")


class OllamaBackend(VLMBackend):
    name = "ollama"

    def __init__(self, config: BackendConfig) -> None:
        super().__init__(config)
        import requests

        self._session = requests.Session()
        self._host = host_url()

        reachable, detail = server_reachable()
        if not reachable:
            raise RuntimeError(
                f"no Ollama server at {self._host} ({detail}). Start it with "
                f"'ollama serve', or set OLLAMA_HOST to point at one."
            )
        if not model_present(config.model):
            available = ", ".join(installed_models()[:8]) or "none"
            raise RuntimeError(
                f"model '{config.model}' is not pulled. Run "
                f"'ollama pull {config.model}'. Currently pulled: {available}"
            )

    def _chat(self, request: GenerationRequest) -> tuple[str, dict[str, Any]]:
        payload: dict[str, Any] = {
            "model": self.config.model,
            "stream": False,
            "messages": [
                {
                    "role": "user",
                    "content": request.prompt,
                    "images": [
                        encode_image(path, self.config.max_side)
                        for path in request.images
                    ],
                }
            ],
            "options": {
                "temperature": self.config.temperature,
                "num_predict": request.max_new_tokens,
            },
        }

        response = self._session.post(
            f"{self._host}/api/chat", json=payload, timeout=REQUEST_TIMEOUT
        )
        if response.status_code >= 400:
            # Ollama puts a useful diagnosis in the body -- a model without a
            # vision projector reports exactly that. raise_for_status() would
            # throw it away and leave only "500 Server Error".
            raise RuntimeError(
                f"Ollama returned {response.status_code} for "
                f"'{self.config.model}': {_error_message(response.text)}"
            )

        body = response.json()
        text = str(body.get("message", {}).get("content", "")).strip()
        meta = {
            # "length" means the budget was hit and the answer is cut off.
            "finish_reason": body.get("done_reason", ""),
            "tokens": body.get("eval_count", 0),
        }
        return text, meta

    def generate(self, requests_: Sequence[GenerationRequest]) -> list[str]:
        return [text for text, _ in self.generate_with_meta(requests_)]

    def generate_with_meta(
        self, requests_: Sequence[GenerationRequest]
    ) -> list[tuple[str, dict[str, Any]]]:
        return [self._chat(request) for request in requests_]

    def close(self) -> None:
        session = getattr(self, "_session", None)
        if session is not None:
            session.close()

    def describe(self) -> dict[str, Any]:
        return {"backend": self.name, "model": self.config.model, "host": self._host}
