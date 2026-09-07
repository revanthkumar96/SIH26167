"""ResNet-50 adapted for multi-band remote-sensing input.

An ImageNet ResNet expects three channels. BigEarthNet gives twelve Sentinel-2
bands, or fourteen with the two Sentinel-1 polarisations, so the stem
convolution has to be rebuilt at the new width.

The interesting decision is what to put in it. Reinitialising randomly throws
away the only thing the pretrained stem knows -- edge and colour-opponent
filters that transfer perfectly well to satellite imagery -- so instead the RGB
kernels are *inflated* across the new channels: each pretrained filter is tiled
and rescaled so the layer's response to a uniform input is unchanged. That
preserves the activation statistics the rest of the network was trained under,
which random init does not.
  pattern: the standard I3D-style inflation, Carreira & Zisserman 2017 s3.1,
  applied across channels rather than time

The head is one sigmoid logit per class with BCE-with-logits, because a patch
carries a *set* of land-cover classes rather than one. Softmax here would force
the classes to compete for probability mass that is not theirs to share.
"""

from __future__ import annotations

from typing import Any

from satquery.cnn.labels import CORINE_CLASSES


def inflate_stem(weight: Any, channels: int) -> Any:
    """Widen a ``(out, 3, k, k)`` conv kernel to ``(out, channels, k, k)``.

    Scaled by ``3 / channels`` so the summed response to a uniform input matches
    the original. Without the rescale a 12-band stem produces roughly four times
    the activation the pretrained batch-norm statistics downstream expect, and
    the first epochs are spent unlearning that rather than learning bands.
    """
    import torch

    if weight.dim() != 4:
        raise ValueError(f"expected a 4-D conv weight, got shape {tuple(weight.shape)}")
    source = weight.shape[1]
    if channels == source:
        return weight.clone()
    if channels < 1:
        raise ValueError(f"channels must be positive, got {channels}")

    # Tile the RGB kernels across the target width, truncating the final repeat
    # when the counts do not divide evenly.
    repeats = (channels + source - 1) // source
    tiled = weight.repeat(1, repeats, 1, 1)[:, :channels]
    return (tiled * (source / channels)).contiguous().to(torch.float32)


def build_model(
    channels: int = 12,
    num_classes: int = len(CORINE_CLASSES),
    pretrained: bool = True,
    weights_path: str | None = None,
) -> Any:
    """A ResNet-50 with a widened stem and a multi-label head.

    ``weights_path`` loads a reBEN checkpoint if one is available -- BIFOLD
    publish pretrained weights alongside the dataset, and starting from a model
    that has already seen Sentinel-2 removes most of this component's training
    cost. Availability and licence are unverified, so ImageNet is the documented
    fallback rather than an afterthought.
    """
    import torch
    from torch import nn
    from torchvision.models import ResNet50_Weights, resnet50

    weights = (
        ResNet50_Weights.IMAGENET1K_V2 if pretrained and not weights_path else None
    )
    model = resnet50(weights=weights)

    original = model.conv1.weight.data
    model.conv1 = nn.Conv2d(
        channels, 64, kernel_size=7, stride=2, padding=3, bias=False
    )
    if weights is not None:
        model.conv1.weight.data = inflate_stem(original, channels)

    model.fc = nn.Linear(model.fc.in_features, num_classes)

    if weights_path:
        state = torch.load(weights_path, map_location="cpu", weights_only=True)
        # Tolerated because a reBEN checkpoint may carry a different head width;
        # the stem and trunk are what we are actually after.
        missing, unexpected = model.load_state_dict(state, strict=False)
        if missing or unexpected:
            print(
                f"  loaded {weights_path} with {len(missing)} missing and "
                f"{len(unexpected)} unexpected keys"
            )
    return model
