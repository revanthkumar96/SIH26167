"""Deterministic no-model backend for CI and harness development.

Lets the whole pipeline -- datasets, prompts, parsing, metrics, reporting -- be
exercised on a laptop with no GPU and no weights. Its replies are shaped to match
each task so the parsers are genuinely tested rather than bypassed.
"""

from __future__ import annotations

from collections.abc import Sequence

from satquery.eval.backends.base import BackendConfig, VLMBackend
from satquery.schema import GenerationRequest


class EchoBackend(VLMBackend):
    name = "echo"

    def __init__(self, config: BackendConfig | None = None) -> None:
        super().__init__(config or BackendConfig(model="echo"))
        self.calls = 0

    def generate(self, requests: Sequence[GenerationRequest]) -> list[str]:
        self.calls += 1
        return [self._reply(r) for r in requests]

    @staticmethod
    def _reply(request: GenerationRequest) -> str:
        prompt = request.prompt.lower()
        if "bounding box" in prompt:
            return "[250, 250, 750, 750]"
        if "describe" in prompt:
            return (
                "An agricultural area with rectangular fields, a road running "
                "north to south, and a small water body in the north-east."
            )
        return "yes"
