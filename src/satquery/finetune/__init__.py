"""LoRA adaptation of the vision-language model.

Imports are deferred: torch and peft are only needed on the training box, and
``satquery`` must stay importable without them.
"""

from satquery.finetune.config import LoRASettings, StageConfig, stage_a, stage_b
from satquery.finetune.targets import (
    ProjectorNotFoundError,
    freeze_vision_encoder,
    resolve_targets,
)

__all__ = [
    "LoRASettings",
    "ProjectorNotFoundError",
    "StageConfig",
    "freeze_vision_encoder",
    "resolve_targets",
    "stage_a",
    "stage_b",
]
