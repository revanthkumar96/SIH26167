"""OpenAI-compatible chat-completions backend.

The AWS path. vLLM's ``openai.api_server`` serves the base model and any LoRA
adapters passed to ``--lora-modules`` as separate ``model`` names on one process,
which is exactly the shape the benchmark matrix wants: base and adapted become
two rows against one running server rather than two loads of a 2B model.

It is close to free to build because ``build_messages()`` already emits
OpenAI-format content with base64 ``image_url`` parts -- the same function the
vLLM in-process backend uses -- so nothing here reformats a prompt. That matters
beyond convenience: a hand-written prompt-format difference between the backend
that measured the baseline and the one that measures the adapted model would
make the comparison meaningless.

Because it speaks plain OpenAI, the same code path works against a local vLLM,
the EC2 box over an SSH tunnel, or any other compatible server.

    SATQUERY_BACKEND=openai_compat
    SATQUERY_VLM_BASE_URL=http://<host>:8000/v1
    SATQUERY_VLM_API_KEY=<token>
    SATQUERY_MODEL=qwen3-vl-satquery

  pattern: vllm docs/source/serving/openai_compatible_server.md -- /v1/models,
  /v1/chat/completions, and the --lora-modules adapter naming
"""

from __future__ import annotations

import os
from collections.abc import Sequence
from typing import Any

from satquery.eval.backends.base import BackendConfig, VLMBackend, build_messages
from satquery.schema import GenerationRequest

DEFAULT_BASE_URL = "http://127.0.0.1:8000/v1"
#: A cold vLLM server compiling CUDA graphs can take minutes to answer its first
#: request, and a paired-image prompt is not fast even when warm.
DEFAULT_REQUEST_TIMEOUT = 600


def base_url() -> str:
    """Endpoint root, with the ``/v1`` suffix supplied if it was omitted."""
    raw = os.environ.get("SATQUERY_VLM_BASE_URL", DEFAULT_BASE_URL).strip()
    if not raw:
        raw = DEFAULT_BASE_URL
    if not raw.startswith(("http://", "https://")):
        raw = f"http://{raw}"
    raw = raw.rstrip("/")
    return raw if raw.endswith("/v1") else f"{raw}/v1"


def api_key() -> str:
    """Bearer token. vLLM accepts any value when started without ``--api-key``."""
    return os.environ.get("SATQUERY_VLM_API_KEY", "").strip() or "EMPTY"


def request_timeout() -> float:
    raw = os.environ.get("SATQUERY_VLM_TIMEOUT", "").strip()
    if not raw:
        return float(DEFAULT_REQUEST_TIMEOUT)
    try:
        value = float(raw)
    except ValueError:
        return float(DEFAULT_REQUEST_TIMEOUT)
    return value if value > 0 else float(DEFAULT_REQUEST_TIMEOUT)


def _headers() -> dict[str, str]:
    return {"Authorization": f"Bearer {api_key()}", "Content-Type": "application/json"}


def server_reachable(timeout: float = 5.0) -> tuple[bool, str]:
    """Whether a compatible server answers, and what it is serving."""
    import requests

    try:
        response = requests.get(
            f"{base_url()}/models", headers=_headers(), timeout=timeout
        )
        response.raise_for_status()
        served = [m.get("id", "?") for m in response.json().get("data", [])]
        return True, ", ".join(served) or "no models listed"
    except Exception as exc:
        return False, f"{type(exc).__name__}: {exc}"


def served_models(timeout: float = 5.0) -> list[str]:
    """Model names the server exposes, base and LoRA adapters alike."""
    import requests

    try:
        response = requests.get(
            f"{base_url()}/models", headers=_headers(), timeout=timeout
        )
        response.raise_for_status()
        return [str(m.get("id", "")) for m in response.json().get("data", []) if m]
    except Exception:
        return []


def _error_message(text: str) -> str:
    """The human-readable part of an OpenAI-shaped error body."""
    import json

    try:
        payload = json.loads(text)
    except (ValueError, TypeError):
        return text.strip()[:400]
    if isinstance(payload, dict):
        error = payload.get("error", payload)
        if isinstance(error, dict):
            return str(error.get("message", error))[:400]
        return str(error)[:400]
    return str(payload)[:400]


class OpenAICompatBackend(VLMBackend):
    name = "openai_compat"

    def __init__(self, config: BackendConfig) -> None:
        super().__init__(config)
        import requests

        self._session = requests.Session()
        self._base = base_url()

        reachable, detail = server_reachable()
        if not reachable:
            raise RuntimeError(
                f"no OpenAI-compatible server at {self._base} ({detail}). Start "
                f"vLLM with 'python -m vllm.entrypoints.openai.api_server', or "
                f"set SATQUERY_VLM_BASE_URL to point at one."
            )

        # Checked at construction rather than on the first request: a sweep that
        # dies 40 minutes in because an adapter name was misspelled has wasted
        # the GPU hours it already spent.
        available = served_models()
        if available and config.model not in available:
            raise RuntimeError(
                f"model '{config.model}' is not served at {self._base}. "
                f"Available: {', '.join(available)}. A LoRA adapter must be "
                f"passed to vLLM as '--lora-modules <name>=<path>' to appear here."
            )

    def _complete(self, request: GenerationRequest) -> tuple[str, dict[str, Any]]:
        import requests

        payload: dict[str, Any] = {
            "model": self.config.model,
            "messages": build_messages(request, self.config.max_side),
            "max_tokens": request.max_new_tokens,
            "temperature": self.config.temperature,
        }

        timeout = request_timeout()
        try:
            response = self._session.post(
                f"{self._base}/chat/completions",
                json=payload,
                headers=_headers(),
                timeout=timeout,
            )
        except requests.exceptions.ReadTimeout as exc:
            raise RuntimeError(
                f"'{self.config.model}' produced no reply within {timeout:.0f}s at "
                f"{self._base}. A cold vLLM server can spend minutes compiling "
                f"CUDA graphs before its first token; raise the ceiling with "
                f"SATQUERY_VLM_TIMEOUT=<seconds> if that is what this is."
            ) from exc

        if response.status_code >= 400:
            # The body carries the actionable part -- a context-length overflow
            # names the limit and the request size. raise_for_status() would
            # discard it and leave only "400 Client Error".
            raise RuntimeError(
                f"{self._base} returned {response.status_code} for "
                f"'{self.config.model}': {_error_message(response.text)}"
            )

        body = response.json()
        choices = body.get("choices") or []
        if not choices:
            raise RuntimeError(
                f"{self._base} returned no choices for '{self.config.model}': "
                f"{_error_message(response.text)}"
            )

        text = str(choices[0].get("message", {}).get("content") or "").strip()
        usage = body.get("usage") or {}
        meta = {
            # "length" means the budget was hit and the answer is cut off, which
            # the trace reports rather than presenting half a sentence.
            "finish_reason": str(choices[0].get("finish_reason") or ""),
            "tokens": int(usage.get("completion_tokens", 0) or 0),
        }
        return text, meta

    def generate(self, requests_: Sequence[GenerationRequest]) -> list[str]:
        return [text for text, _ in self.generate_with_meta(requests_)]

    def generate_with_meta(
        self, requests_: Sequence[GenerationRequest]
    ) -> list[tuple[str, dict[str, Any]]]:
        # Sequential: vLLM batches continuously on its own side, so issuing
        # concurrent requests here would add client complexity without adding
        # throughput.
        return [self._complete(request) for request in requests_]

    def close(self) -> None:
        session = getattr(self, "_session", None)
        if session is not None:
            session.close()

    def describe(self) -> dict[str, Any]:
        return {
            "backend": self.name,
            "model": self.config.model,
            "base_url": self._base,
        }
