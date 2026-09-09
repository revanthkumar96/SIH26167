"""LoRA target selection and configuration.

The tests that matter here are about the projector. A published competitor
adapter targeted only the language model's attention and MLP projections, so the
visual domain shift was never learned -- and crucially, that run looked
successful throughout. The failure surfaces as a flat benchmark, weeks later.

So target resolution refuses to produce a language-only adapter silently, and
these tests pin that refusal.
"""

from __future__ import annotations

import pytest

from satquery.finetune.config import (
    BASE_MODEL,
    BASE_REVISION,
    LoRASettings,
    stage_a,
    stage_b,
)
from satquery.finetune.targets import (
    LANGUAGE_TARGETS,
    ProjectorNotFoundError,
    find_projector_targets,
    find_vision_encoder_targets,
    freeze_vision_encoder,
    in_vision_tower,
    is_projector,
    resolve_targets,
)

torch = pytest.importorskip("torch")
nn = torch.nn


def fake_vlm(projector: bool = True, projector_name: str = "merger"):
    """A module tree shaped like Qwen3-VL, without downloading 2B parameters."""

    class Block(nn.Module):
        def __init__(self):
            super().__init__()
            self.q_proj = nn.Linear(8, 8)
            self.k_proj = nn.Linear(8, 8)
            self.v_proj = nn.Linear(8, 8)
            self.o_proj = nn.Linear(8, 8)
            self.gate_proj = nn.Linear(8, 8)
            self.up_proj = nn.Linear(8, 8)
            self.down_proj = nn.Linear(8, 8)

    class VisionBlock(nn.Module):
        """A ViT block: fused qkv and fc1/fc2, not the language stack's names."""

        def __init__(self):
            super().__init__()
            self.qkv = nn.Linear(8, 24)
            self.proj = nn.Linear(8, 8)
            self.fc1 = nn.Linear(8, 16)
            self.fc2 = nn.Linear(16, 8)

    class Visual(nn.Module):
        def __init__(self):
            super().__init__()
            self.blocks = nn.ModuleList([VisionBlock()])
            if projector:
                setattr(
                    self,
                    projector_name,
                    nn.Sequential(nn.Linear(8, 8), nn.GELU(), nn.Linear(8, 8)),
                )

    class Model(nn.Module):
        def __init__(self):
            super().__init__()
            self.visual = Visual()
            self.language_model = nn.ModuleList([Block()])

    return Model()


# -- the projector, which is the whole point -----------------------------


def test_projector_is_included_by_default():
    targets = resolve_targets(fake_vlm())
    assert set(LANGUAGE_TARGETS) <= set(targets)
    assert any("mlp" in t or "merger" in t for t in targets), targets


def test_a_missing_projector_raises_rather_than_training_language_only():
    """The competitor's bug, made loud.

    A run that silently adapts only the language model is indistinguishable
    from a successful one until the benchmark comes back flat.
    """
    with pytest.raises(ProjectorNotFoundError, match="language model only"):
        resolve_targets(fake_vlm(projector=False))


def test_language_only_can_still_be_chosen_deliberately():
    """An ablation may want it -- but now it takes turning off both halves.

    Switching off the projector alone no longer yields a language-only adapter,
    because the vision encoder is adapted by default. That is the point: the
    weakest reading of the requirement should take two deliberate acts, not one
    forgotten flag.
    """
    targets = resolve_targets(
        fake_vlm(projector=False),
        include_projector=False,
        include_vision_encoder=False,
    )
    assert set(targets) == set(LANGUAGE_TARGETS)


@pytest.mark.parametrize("name", ["merger", "projector", "mm_projector"])
def test_projector_is_found_under_the_names_releases_have_used(name):
    """Module names have moved between releases; a hardcoded one would miss."""
    targets = resolve_targets(fake_vlm(projector_name=name))
    assert len(targets) > len(LANGUAGE_TARGETS)


def test_numeric_leaves_keep_their_parent_segment():
    """`mlp.0` and `mlp.2` cannot be suffix-matched as "0" and "2"."""
    found = find_projector_targets(["visual.merger.mlp.0", "visual.merger.mlp.2"])
    assert found == ["mlp.0", "mlp.2"]


@pytest.mark.parametrize(
    ("path", "expected"),
    [
        ("visual.merger.mlp.0", True),
        ("model.multi_modal_projector.linear_1", True),
        ("visual.blocks.0.attn.q_proj", False),
        ("language_model.layers.0.self_attn.q_proj", False),
    ],
)
def test_projector_detection(path, expected):
    assert is_projector(path) is expected


@pytest.mark.parametrize(
    ("path", "expected"),
    [
        ("visual.blocks.0.attn.q_proj", True),
        ("vision_tower.encoder.layer.0", True),
        ("language_model.layers.0.self_attn.q_proj", False),
    ],
)
def test_vision_tower_detection(path, expected):
    assert in_vision_tower(path) is expected


# -- freezing ------------------------------------------------------------


def test_vision_encoder_is_frozen_but_the_projector_is_not():
    """Full ViT tuning needs more data than we have and invites forgetting."""
    model = fake_vlm()
    assert freeze_vision_encoder(model) > 0

    trainable = {name for name, p in model.named_parameters() if p.requires_grad}
    assert any("merger" in name for name in trainable), (
        "the projector must stay trainable -- it is where the visual domain shift lands"
    )
    assert not any("visual.blocks" in name for name in trainable), (
        "the vision encoder's own blocks must be frozen"
    )


def test_language_model_is_untouched_by_freezing():
    model = fake_vlm()
    freeze_vision_encoder(model)
    assert all(
        p.requires_grad
        for name, p in model.named_parameters()
        if name.startswith("language_model")
    )


# -- configuration -------------------------------------------------------


def test_base_revision_is_pinned():
    """A silent upstream reweight would invalidate every measured number."""
    assert len(BASE_REVISION) == 40
    assert BASE_MODEL == "Qwen/Qwen3-VL-2B-Instruct"


def test_learning_rate_is_the_corrected_one():
    """1e-5 over ~375 steps barely moves an adapter; 1e-4 is the LoRA range."""
    assert LoRASettings().learning_rate == pytest.approx(1e-4)


def test_rank_and_alpha_give_headroom_for_a_real_domain_shift():
    settings = LoRASettings()
    assert settings.rank == 32
    assert settings.alpha == 64


def test_effective_batch_is_in_the_intended_range():
    assert 64 <= LoRASettings().effective_batch <= 128


def test_max_pixels_exceeds_the_resolution_starved_t4_setting():
    """200,704 is ~256 vision tokens, which starves grounding."""
    assert LoRASettings().max_pixels > 200_704


def test_precision_is_bf16_not_fp16():
    """Ampere, so no GradScaler and one fewer instability than the T4 path."""
    assert LoRASettings().precision == "bfloat16"


def test_stage_b_trains_longer_than_stage_a():
    assert stage_b().lora.epochs >= stage_a().lora.epochs


def test_stage_metadata_records_everything_a_score_needs():
    """A number that cannot be tied to its configuration cannot be defended."""
    payload = stage_a().as_dict()
    assert payload["base_revision"] == BASE_REVISION
    assert payload["toolchain"]["transformers"] == "4.57.1"
    assert payload["lora"]["seed"] == 1234
    assert payload["lora"]["include_projector"] is True
    assert payload["effective_batch"] == 64


def test_overrides_reach_the_settings():
    assert stage_a(epochs=3).lora.epochs == 3
    assert stage_b(learning_rate=5e-5).lora.learning_rate == pytest.approx(5e-5)


# -- the vision encoder, not just the bridge -----------------------------


def test_the_vision_encoder_is_adapted_by_default():
    """Three things are adapted: language, projector, and the encoder itself.

    A projector alone can only re-mix features the encoder computed under
    natural-image assumptions. Sentinel false colour and SAR backscatter are not
    natural images.
    """
    targets = resolve_targets(fake_vlm())
    vision_side = [t for t in targets if t not in LANGUAGE_TARGETS]

    assert {"qkv", "proj", "fc1", "fc2"} <= set(vision_side), vision_side
    assert any("merger" in t or "mlp" in t for t in vision_side), vision_side


def test_the_encoder_can_be_left_out_deliberately():
    """An ablation may want projector-only -- by asking, not by default."""
    targets = resolve_targets(fake_vlm(), include_vision_encoder=False)
    assert "qkv" not in targets and "fc1" not in targets
    assert any("merger" in t or "mlp" in t for t in targets)


def test_the_projector_is_not_counted_as_encoder():
    """The bridge and the tower are different claims and must not be conflated."""
    found = find_vision_encoder_targets(
        [
            "visual.blocks.0.attn.qkv",
            "visual.merger.mlp.0",
            "language_model.layers.0.self_attn.q_proj",
        ]
    )
    assert found == ["qkv"]


def test_language_leaf_names_are_not_duplicated():
    """A ViT sharing a leaf name with the language stack must not appear twice."""
    targets = resolve_targets(fake_vlm())
    assert len(targets) == len(set(targets))


def test_base_weights_stay_frozen_while_the_adapter_trains():
    """LoRA over the encoder is not full fine-tuning of it.

    The documented objection to tuning a ViT is about updating every weight.
    Base weights are frozen; what learns is a rank-32 adapter that can be
    removed.
    """
    model = fake_vlm()
    freeze_vision_encoder(model)
    encoder_base = [
        p.requires_grad for n, p in model.named_parameters() if "visual.blocks" in n
    ]
    assert encoder_base and not any(encoder_base)


def test_vision_encoder_is_on_by_default_in_the_shipped_config():
    assert LoRASettings().include_vision_encoder is True
    assert stage_a().as_dict()["lora"]["include_vision_encoder"] is True
