"""Choosing which modules LoRA adapts.

The single most consequential decision in the fine-tune, and the one a published
competitor adapter got wrong. Targeting only ``q,k,v,o,gate,up,down`` adapts the
language model and nothing else, so the *visual* domain shift -- what SAR
backscatter looks like, how a multispectral false-colour composite reads -- is
never learned at all. The vision-language projector is where that shift lands.

It also decides a compliance question. The problem statement asks for a visual
or vision-language component to be adapted; LoRA confined to a language model's
attention blocks is the weakest defensible reading of that.

Module names are **discovered from the loaded model, not hardcoded**. Qwen3-VL's
projector has been called ``merger``, ``visual.merger`` and ``mlp`` across
releases, and a name that silently matches nothing would reproduce exactly the
competitor's bug while looking like it had been fixed. So the projector is
located by walking the module tree, and ``resolve_targets`` raises when it finds
none rather than quietly training a language-only adapter.
"""

from __future__ import annotations

import re
from collections.abc import Iterable
from typing import Any

#: Attention and MLP projections in the language stack. Suffix-matched, which is
#: how peft's target_modules works.
LANGUAGE_TARGETS: tuple[str, ...] = (
    "q_proj",
    "k_proj",
    "v_proj",
    "o_proj",
    "gate_proj",
    "up_proj",
    "down_proj",
)

#: Module-path fragments that mark the vision-to-language bridge. Matched against
#: the full dotted path, so they select the projector without also selecting the
#: vision encoder's own attention blocks -- which stay frozen.
_PROJECTOR_HINTS = ("merger", "projector", "multi_modal_projector", "mm_projector")

#: Anything under the vision tower that is *not* the projector. Excluded because
#: full vision-encoder tuning needs far more data than we have and invites
#: forgetting general visual competence.
_VISION_TOWER = re.compile(r"(^|\.)(visual|vision_tower|vision_model)\.")


class ProjectorNotFoundError(RuntimeError):
    """No vision-language projector matched, so LoRA would be language-only."""


def is_projector(path: str) -> bool:
    """Whether a dotted module path names part of the vision-language bridge."""
    lowered = path.lower()
    return any(hint in lowered for hint in _PROJECTOR_HINTS)


def in_vision_tower(path: str) -> bool:
    return bool(_VISION_TOWER.search(path))


def linear_module_paths(model: Any) -> list[str]:
    """Dotted paths of every ``nn.Linear`` in the model."""
    from torch import nn

    return [
        name for name, module in model.named_modules() if isinstance(module, nn.Linear)
    ]


def find_projector_targets(paths: Iterable[str]) -> list[str]:
    """Leaf names of the linear layers making up the projector.

    Returns leaf names rather than full paths because peft suffix-matches, and a
    full path would pin the adapter to one release's exact module layout.
    """
    found: list[str] = []
    for path in paths:
        if not is_projector(path):
            continue
        leaf = path.rsplit(".", 1)[-1]
        # A projector MLP is often `mlp.0` / `mlp.2`, whose leaves are digits.
        # Those cannot be suffix-matched usefully, so keep the parent segment.
        if leaf.isdigit():
            parts = path.split(".")
            leaf = ".".join(parts[-2:])
        if leaf not in found:
            found.append(leaf)
    return found


def resolve_targets(model: Any, include_projector: bool = True) -> list[str]:
    """The full ``target_modules`` list for this model.

    Raises when the projector cannot be located. That is deliberate: a run that
    silently trains a language-only adapter is the failure being corrected here,
    and it is indistinguishable from a successful one until the benchmark comes
    back flat.
    """
    paths = linear_module_paths(model)
    present = {path.rsplit(".", 1)[-1] for path in paths}
    targets = [name for name in LANGUAGE_TARGETS if name in present]

    if not targets:
        raise ProjectorNotFoundError(
            "none of the expected language projections were found; the model "
            f"exposes leaves such as {sorted(present)[:12]}"
        )

    if not include_projector:
        return targets

    projector = find_projector_targets(paths)
    if not projector:
        candidates = sorted({p for p in paths if in_vision_tower(p)})[:12]
        raise ProjectorNotFoundError(
            "no vision-language projector matched, so LoRA would adapt the "
            "language model only -- which is the exact mistake this "
            "configuration exists to correct. Vision-side linear layers found: "
            f"{candidates}. Extend _PROJECTOR_HINTS to name this model's bridge, "
            "or pass include_projector=False to accept a language-only adapter "
            "deliberately."
        )
    return targets + projector


def freeze_vision_encoder(model: Any) -> int:
    """Freeze the vision tower except its projector. Returns tensors frozen.

    LoRA already leaves base weights frozen, so this matters for the case where
    a caller unfreezes something, and as an explicit statement of intent: the
    encoder is not being tuned, the bridge to the language model is.
    """
    frozen = 0
    for name, parameter in model.named_parameters():
        if in_vision_tower(name) and not is_projector(name):
            parameter.requires_grad_(False)
            frozen += 1
    return frozen
