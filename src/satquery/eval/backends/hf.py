"""Transformers backend -- the correctness reference.

Slow but easy to debug, and it works for any model with a chat template. Use it to
verify a new candidate on ~32 samples, then run the full sweep on vLLM.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

from satquery.eval.backends.base import (
    BackendConfig,
    VLMBackend,
    load_pil_images,
)
from satquery.schema import GenerationRequest


def _resolve_dtype(name: str) -> Any:
    import torch

    if name == "auto":
        # T4 and P100 have no bf16; picking it silently would fall back to fp32
        # and halve throughput.
        return torch.float16 if torch.cuda.is_available() else torch.float32
    return {
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
        "float32": torch.float32,
    }[name]


class HFBackend(VLMBackend):
    name = "hf"

    def __init__(self, config: BackendConfig) -> None:
        super().__init__(config)
        import torch
        from transformers import AutoModelForImageTextToText, AutoProcessor

        self._torch = torch
        self.processor = AutoProcessor.from_pretrained(
            config.model, trust_remote_code=config.trust_remote_code
        )
        self.model = AutoModelForImageTextToText.from_pretrained(
            config.model,
            dtype=_resolve_dtype(config.dtype),
            device_map="auto" if torch.cuda.is_available() else None,
            trust_remote_code=config.trust_remote_code,
        )
        self.model.eval()

    def _messages(self, request: GenerationRequest) -> list[dict]:
        content: list[dict[str, Any]] = [{"type": "image"} for _ in request.images]
        content.append({"type": "text", "text": request.prompt})
        return [{"role": "user", "content": content}]

    def generate(self, requests: Sequence[GenerationRequest]) -> list[str]:
        outputs: list[str] = []
        batch_size = max(self.config.batch_size, 1)
        for start in range(0, len(requests), batch_size):
            outputs.extend(self._generate_batch(requests[start : start + batch_size]))
        return outputs

    def _generate_batch(self, batch: Sequence[GenerationRequest]) -> list[str]:
        torch = self._torch
        texts = [
            self.processor.apply_chat_template(
                self._messages(r), tokenize=False, add_generation_prompt=True
            )
            for r in batch
        ]
        images = [load_pil_images(r.images, self.config.max_side) for r in batch]

        inputs = self.processor(
            text=texts,
            images=images if any(images) else None,
            return_tensors="pt",
            padding=True,
        ).to(self.model.device)

        max_new = max(r.max_new_tokens for r in batch)
        with torch.inference_mode():
            generated = self.model.generate(
                **inputs,
                max_new_tokens=max_new,
                do_sample=False,
                temperature=None,
                top_p=None,
            )

        trimmed = [
            out[len(inp) :]
            for inp, out in zip(inputs["input_ids"], generated, strict=True)
        ]
        return [
            text.strip()
            for text in self.processor.batch_decode(trimmed, skip_special_tokens=True)
        ]

    def close(self) -> None:
        model = getattr(self, "model", None)
        if model is not None:
            del self.model
        if self._torch.cuda.is_available():
            self._torch.cuda.empty_cache()
