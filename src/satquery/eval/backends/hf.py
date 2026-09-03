"""Transformers backend for Qwen2.5-VL and InternVL.

One code path serves both families: upstream registers
``Qwen2_5_VLForConditionalGeneration`` and ``InternVLForConditionalGeneration`` in the
same ``MODEL_FOR_IMAGE_TEXT_TO_TEXT_MAPPING_NAMES``, so ``AutoModelForImageTextToText``
resolves either checkpoint.
  pattern: transformers@5.16.1 src/transformers/models/auto/modeling_auto.py:1079,1121,1149

Prompt assembly is the single-call upstream form -- ``apply_chat_template`` tokenises
*and* loads the imagery, so there is no second ``processor(text=, images=)`` step and no
hand-rolled image loading.
  pattern: transformers@5.16.1 docs/source/en/model_doc/internvl.md:102

Resolution is capped by the processor's ``min_pixels`` / ``max_pixels`` rather than by
resizing beforehand. Each architecture patches images on its own grid, so pre-resizing
fights the processor instead of helping it.
  pattern: transformers@5.16.1 docs/source/en/model_doc/qwen2_5_vl.md:218-230

Note on InternVL checkpoints: only the ``-hf`` suffixed repos (for example
``OpenGVLab/InternVL3-2B-hf``) expose this interface. The plain repos ship a bespoke
``.chat()`` API behind ``trust_remote_code`` and are not interchangeable.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

from satquery.eval.backends.base import BackendConfig, VLMBackend
from satquery.schema import GenerationRequest

#: Auto classes to try, newest name first. Older transformers releases only carry
#: ``AutoModelForVision2Seq``, and failing on the import would make the backend
#: unusable on an environment that has not been upgraded yet.
_AUTO_CLASS_NAMES = ("AutoModelForImageTextToText", "AutoModelForVision2Seq")


def resolve_dtype(name: str) -> Any:
    """Torch dtype for a configured name.

    ``auto`` picks fp16 on GPU: the T4 and P100 cards this is most likely to meet
    have no bf16, and selecting it there silently falls back to fp32.
    """
    import torch

    if name == "auto":
        return torch.float16 if torch.cuda.is_available() else torch.float32
    return {
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
        "float32": torch.float32,
    }[name]


def auto_model_class() -> Any:
    """The multimodal auto class available in the installed transformers."""
    import transformers

    for name in _AUTO_CLASS_NAMES:
        candidate = getattr(transformers, name, None)
        if candidate is not None:
            return candidate
    raise RuntimeError(
        f"transformers {transformers.__version__} exposes none of "
        f"{', '.join(_AUTO_CLASS_NAMES)}; upgrade with "
        f"pip install -U 'transformers>=4.49'"
    )


def build_messages(request: GenerationRequest) -> list[dict[str, Any]]:
    """One user turn: every image, then the prompt text.

    Images are referenced by path and left for the processor to load.
      pattern: transformers@5.16.1 src/transformers/processing_utils.py:2144
    """
    content: list[dict[str, Any]] = [
        {"type": "image", "path": str(path.resolve())} for path in request.images
    ]
    content.append({"type": "text", "text": request.prompt})
    return [{"role": "user", "content": content}]


class HFBackend(VLMBackend):
    name = "hf"

    def __init__(self, config: BackendConfig) -> None:
        super().__init__(config)
        import torch
        from transformers import AutoProcessor

        self._torch = torch

        processor_kwargs: dict[str, Any] = {
            "trust_remote_code": config.trust_remote_code
        }
        if config.min_pixels:
            processor_kwargs["min_pixels"] = config.min_pixels
        if config.max_pixels:
            processor_kwargs["max_pixels"] = config.max_pixels

        self.processor = AutoProcessor.from_pretrained(config.model, **processor_kwargs)
        self.model = auto_model_class().from_pretrained(
            config.model,
            dtype=resolve_dtype(config.dtype),
            device_map="auto" if torch.cuda.is_available() else None,
            trust_remote_code=config.trust_remote_code,
        )
        self.model.eval()

    def generate(self, requests: Sequence[GenerationRequest]) -> list[str]:
        outputs: list[str] = []
        batch_size = max(self.config.batch_size, 1)
        for start in range(0, len(requests), batch_size):
            outputs.extend(self._generate_batch(requests[start : start + batch_size]))
        return outputs

    def _generate_batch(self, batch: Sequence[GenerationRequest]) -> list[str]:
        torch = self._torch
        conversations = [build_messages(request) for request in batch]

        inputs = self.processor.apply_chat_template(
            conversations,
            add_generation_prompt=True,
            tokenize=True,
            return_dict=True,
            return_tensors="pt",
            padding=True,
        ).to(self.model.device)

        with torch.inference_mode():
            generated = self.model.generate(
                **inputs,
                max_new_tokens=max(r.max_new_tokens for r in batch),
                do_sample=False,
            )

        # Trim the prompt so only the completion is decoded.
        prompt_length = inputs["input_ids"].shape[1]
        return [
            text.strip()
            for text in self.processor.batch_decode(
                generated[:, prompt_length:], skip_special_tokens=True
            )
        ]

    def close(self) -> None:
        if getattr(self, "model", None) is not None:
            del self.model
        if getattr(self, "processor", None) is not None:
            del self.processor
        if self._torch.cuda.is_available():
            self._torch.cuda.empty_cache()
