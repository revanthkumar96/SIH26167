"""Target discovery against the real Qwen3-VL architecture.

Every other test of ``resolve_targets`` runs against a module tree we invented,
which can only confirm that the code agrees with our guess. This one loads the
pinned model's own config and checks the names that actually exist.

It found three things the synthetic tests could not:

  * the vision MLP is ``linear_fc1``/``linear_fc2``, not the ``fc1``/``fc2`` we
    had assumed -- discovery handled it, a hardcoded list would not have;
  * ``linear_fc1`` names *both* the projector and the encoder's own MLP, so the
    two lists concatenated to a target set with duplicates in it;
  * Qwen3-VL has ``deepstack_merger_list`` alongside ``merger`` -- extra bridge
    layers that a narrower hint list would have missed.

Skipped unless the pinned toolchain is installed, so it stays quiet on a laptop
and runs on the training host where it matters. Only the config is downloaded
(a few KB); the model is built at a fraction of its real size because module
*names* are what is under test, not weights.
"""

from __future__ import annotations

import pytest

transformers = pytest.importorskip("transformers", minversion="4.57.0")
peft = pytest.importorskip("peft")
torch = pytest.importorskip("torch")

if not hasattr(transformers, "Qwen3VLForConditionalGeneration"):
    pytest.skip("transformers is too old for Qwen3-VL", allow_module_level=True)

from satquery.finetune.config import BASE_MODEL, BASE_REVISION  # noqa: E402
from satquery.finetune.targets import (  # noqa: E402
    LANGUAGE_TARGETS,
    freeze_vision_encoder,
    is_projector,
    linear_module_paths,
    resolve_targets,
)

pytestmark = pytest.mark.slow


@pytest.fixture(scope="module")
def tiny_config():
    """The pinned model's own config, fetched once. A few KB, no weights."""
    from transformers import AutoConfig

    try:
        config = AutoConfig.from_pretrained(BASE_MODEL, revision=BASE_REVISION)
    except Exception as exc:  # offline, or the Hub is unreachable
        pytest.skip(f"cannot fetch {BASE_MODEL} config: {type(exc).__name__}")

    config.text_config.num_hidden_layers = 2
    config.text_config.hidden_size = 128
    config.text_config.intermediate_size = 256
    config.text_config.num_attention_heads = 4
    config.text_config.num_key_value_heads = 2
    config.vision_config.depth = 2
    config.vision_config.hidden_size = 128
    config.vision_config.intermediate_size = 256
    config.vision_config.num_heads = 4
    config.vision_config.deepstack_visual_indexes = [0, 1]
    return config


@pytest.fixture
def tiny_model(tiny_config):
    """A fresh model per test.

    get_peft_model wraps its argument in place, so a shared instance leaves
    later tests inspecting an already-adapted model -- which is how this fixture
    came to be split in two.
    """
    from transformers import Qwen3VLForConditionalGeneration

    return Qwen3VLForConditionalGeneration._from_config(tiny_config)


def test_the_real_module_names_are_discovered(tiny_model):
    """The names we would have hardcoded were wrong; discovery found the real ones."""
    targets = resolve_targets(tiny_model)

    assert set(LANGUAGE_TARGETS) <= set(targets)
    # Real Qwen3-VL vision blocks: fused qkv, an output proj, and an MLP whose
    # layers are linear_fc1/linear_fc2 rather than the fc1/fc2 we assumed.
    assert {"qkv", "proj", "linear_fc1", "linear_fc2"} <= set(targets), targets


def test_the_target_list_has_no_duplicates(tiny_model):
    """The projector and the encoder MLP share leaf names on this model.

    Concatenating the two discovered lists repeated them, which is how this was
    found. peft would tolerate it; a target list that cannot be read at face
    value is still a defect.
    """
    targets = resolve_targets(tiny_model)
    assert len(targets) == len(set(targets)), targets


def test_the_projector_is_actually_present(tiny_model):
    """Both the merger and the DeepStack mergers are recognised as the bridge."""
    projector_paths = [p for p in linear_module_paths(tiny_model) if is_projector(p)]

    assert projector_paths, "no projector matched on the real architecture"
    assert any("visual.merger" in p for p in projector_paths)
    assert any("deepstack_merger_list" in p for p in projector_paths)


def test_no_vision_target_reaches_the_language_stack(tiny_model):
    """peft matches whole path segments, so `proj` must not catch `q_proj`.

    If it did, the vision targets would silently pull in the entire language
    model and the reported split between the two would be meaningless.
    """
    paths = linear_module_paths(tiny_model)
    for target in ("qkv", "proj", "linear_fc1", "linear_fc2"):
        matched = [p for p in paths if p.endswith(f".{target}")]
        assert matched, f"{target} matched nothing"
        assert all(".visual." in p for p in matched), (
            f"{target} reaches outside the vision tower: "
            f"{[p for p in matched if '.visual.' not in p][:3]}"
        )


def test_peft_attaches_adapters_to_all_three_surfaces(tiny_model):
    """The claim, end to end: language, vision encoder and projector all adapted.

    Asserted on the adapters peft actually created, not on the target list we
    handed it -- a name that matches nothing produces no adapter and no error.
    """
    from peft import LoraConfig, get_peft_model

    targets = resolve_targets(tiny_model)
    freeze_vision_encoder(tiny_model)
    adapted = get_peft_model(
        tiny_model,
        LoraConfig(
            r=8,
            lora_alpha=16,
            lora_dropout=0.05,
            target_modules=targets,
            task_type="CAUSAL_LM",
            bias="none",
        ),
    )

    lora = [name for name, _ in adapted.named_parameters() if "lora_" in name]
    encoder = [n for n in lora if ".visual." in n and ".blocks." in n]
    projector = [n for n in lora if "merger" in n]
    language = [n for n in lora if ".visual." not in n]

    assert language, "no adapter on the language stack"
    assert encoder, "no adapter on the vision encoder -- the frozen-ViT regression"
    assert projector, "no adapter on the projector -- the competitor's mistake"


def test_only_adapters_are_trainable(tiny_model):
    """Base weights stay frozen; what learns is removable."""
    from peft import LoraConfig, get_peft_model

    adapted = get_peft_model(
        tiny_model,
        LoraConfig(
            r=8,
            lora_alpha=16,
            target_modules=resolve_targets(tiny_model),
            task_type="CAUSAL_LM",
            bias="none",
        ),
    )
    trainable = [n for n, p in adapted.named_parameters() if p.requires_grad]
    assert trainable
    assert all("lora_" in n for n in trainable), [
        n for n in trainable if "lora_" not in n
    ][:5]
