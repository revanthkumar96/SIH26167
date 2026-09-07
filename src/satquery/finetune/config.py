"""Fine-tuning configuration, as data rather than as flags.

Every number here is recorded in the adapter's own metadata at the end of a run,
because a benchmark score that cannot be tied to the configuration that produced
it cannot be defended -- and the final evaluation is against a hidden reference
we do not get to argue with.

Two settings are deliberate corrections to a published competitor adapter, and
both are commented where they sit: the vision-language projector is in
``target_modules`` (see ``targets.py``), and the learning rate is 1e-4 rather
than 1e-5.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any

#: Pinned so a silent upstream reweight cannot invalidate every number measured
#: against it. Verified 2026-09-06.
BASE_MODEL = "Qwen/Qwen3-VL-2B-Instruct"
# The public commit hash of the model repo on the Hub, not a credential --
# the entropy scanner cannot tell those apart.
BASE_REVISION = "89644892e4d85e24eaac8bacfd4f463576704203"  # pragma: allowlist secret

#: Known-good pins from a third-party run of this exact base on a Tesla T4.
TOOLCHAIN_PINS: dict[str, str] = {
    "transformers": "4.57.1",
    "peft": "0.17.1",
    "accelerate": "1.7.0",
    "qwen-vl-utils": "0.0.14",
}


@dataclass(frozen=True, slots=True)
class LoRASettings:
    """Adapter shape and optimisation."""

    rank: int = 32
    alpha: int = 64
    dropout: float = 0.05

    #: An order of magnitude above the 1e-5 the competitor adapter used. At
    #: 1e-5 over ~375 optimiser steps an adapter has barely moved, which is the
    #: cheapest single correction available to us. 1e-4 is the conventional
    #: LoRA range.
    learning_rate: float = 1e-4
    warmup_ratio: float = 0.03
    scheduler: str = "cosine"

    #: Ampere, so bf16 and no fp16 GradScaler -- one fewer source of training
    #: instability than the T4 path would force.
    precision: str = "bfloat16"

    batch_size: int = 4
    gradient_accumulation: int = 16  # effective batch 64
    epochs: int = 1

    #: The primary memory and throughput lever. The T4 run capped this at
    #: 200,704 -- about 256 vision tokens -- which is thin for grounding and
    #: consistent with its reported mIoU of 0.362. Raised here because 24 GB
    #: allows it, and grounding is what needs the resolution.
    max_pixels: int = 512 * 28 * 28
    min_pixels: int = 4 * 28 * 28
    max_sequence_length: int = 4096

    include_projector: bool = True
    seed: int = 1234

    @property
    def effective_batch(self) -> int:
        return self.batch_size * self.gradient_accumulation


@dataclass(frozen=True, slots=True)
class StageConfig:
    """One training stage: what it trains on and what it is for."""

    name: str
    dataset: str
    purpose: str
    lora: LoRASettings = field(default_factory=LoRASettings)

    def as_dict(self) -> dict[str, Any]:
        return {
            "stage": self.name,
            "dataset": self.dataset,
            "purpose": self.purpose,
            "base_model": BASE_MODEL,
            "base_revision": BASE_REVISION,
            "toolchain": dict(TOOLCHAIN_PINS),
            "lora": asdict(self.lora),
            "effective_batch": self.lora.effective_batch,
        }


def stage_a(**overrides: Any) -> StageConfig:
    """Domain and cross-modal adaptation on BigEarthNet.txt.

    Teaches SAR vocabulary, backscatter intuition and joint optical-SAR
    reasoning. This is the stage that discharges mandatory scope item 1 using
    the dataset the problem statement names.
    """
    return StageConfig(
        name="stage-a",
        dataset="bigearthnet.txt",
        purpose="remote-sensing domain and cross-modal adaptation",
        lora=LoRASettings(**overrides),
    )


def stage_b(**overrides: Any) -> StageConfig:
    """Instruction tuning on the task mixture.

    Teaches the output formats the metrics reward -- terse answers for
    exact-match VQA, 0-1000 boxes for grounding, one or two sentences for
    captioning. Worth more raw points than stage A, because a model that knows
    the answer but phrases it wrongly scores zero under exact match and three of
    the five scored criteria are exact-match or n-gram based.

    It is also the only source of bi-temporal capability: BigEarthNet is
    single-date, so no amount of stage A produces change understanding.
    """
    defaults: dict[str, Any] = {"epochs": 2}
    defaults.update(overrides)
    return StageConfig(
        name="stage-b",
        dataset="vrsbench + rsvqa + cdvqa + evidence records",
        purpose="output formats the metrics reward, and bi-temporal capability",
        lora=LoRASettings(**defaults),
    )
