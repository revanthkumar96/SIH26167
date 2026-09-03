"""vLLM backend -- the one that actually runs the sweep.

Batched short-answer generation is 5-10x faster here than under transformers, which
is the difference between a $6 bake-off and a $30 one. ``LLM.chat`` is used rather
than raw prompts so each candidate's own chat template handles multimodal turns.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

from satquery.eval.backends.base import (
    BackendConfig,
    VLMBackend,
    build_messages,
)
from satquery.schema import GenerationRequest


class VLLMBackend(VLMBackend):
    name = "vllm"

    def __init__(self, config: BackendConfig) -> None:
        super().__init__(config)
        from vllm import LLM

        extra: dict[str, Any] = dict(config.extra or {})
        max_images = extra.pop("max_images", 2)

        self.llm = LLM(
            model=config.model,
            dtype=config.dtype,
            trust_remote_code=config.trust_remote_code,
            tensor_parallel_size=config.tensor_parallel_size,
            gpu_memory_utilization=config.gpu_memory_utilization,
            max_model_len=config.max_model_len,
            # Pair tasks send two images; the engine must reserve slots for them.
            limit_mm_per_prompt={"image": max_images},
            **extra,
        )

    def generate(self, requests: Sequence[GenerationRequest]) -> list[str]:
        from vllm import SamplingParams

        if not requests:
            return []

        conversations = [build_messages(r, self.config.max_side) for r in requests]
        params = [
            SamplingParams(
                temperature=self.config.temperature,
                max_tokens=r.max_new_tokens,
            )
            for r in requests
        ]

        outputs = self.llm.chat(conversations, params)
        return [output.outputs[0].text.strip() for output in outputs]

    def close(self) -> None:
        if getattr(self, "llm", None) is not None:
            del self.llm
