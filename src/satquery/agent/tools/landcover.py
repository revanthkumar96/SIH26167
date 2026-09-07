"""Land-cover classification tool.

Exists because of a band that is not there. ``optical_indices`` derives built-up
extent from NDBI, which needs SWIR -- and Cartosat-2S has none. On the hidden
evaluation set the index tool will correctly report itself inapplicable for
built-up, and without this tool the only remaining optical signal is nothing at
all.

A CNN learns land cover from whatever bands it is given. It needs no SWIR, works
on RGB or panchromatic input, and produces class evidence exactly where the
physics-based index cannot be computed.

Two things it is not. It is **classification, not segmentation**: BigEarthNet
has no per-pixel masks, so this cannot produce a land-cover map and must not be
demonstrated as one. And it is **not a replacement for the index tools**: where
SWIR exists, NDBI is cheaper, deterministic and needs no trust. This is the
fallback that makes the system degrade gracefully rather than fail, which is
what the hidden set will demand of it.

Its confidences are surfaced honestly and never presented as ground truth. CORINE
is a European taxonomy trained on European scenes, so the classifier will be
least reliable exactly where we need it most.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

from satquery.agent.context import RunContext
from satquery.agent.registry import Tool, ToolResult
from satquery.cnn.labels import CORINE_CLASSES, render_evidence_line
from satquery.schema import ImageRole, InputConfig, Task, ToolSpec

#: Where the trained checkpoint lives. Absent by default: the tool reports
#: itself inapplicable rather than failing a run, so the system works before the
#: classifier is trained and keeps working if a demo machine lacks it.
CHECKPOINT_ENV = "SATQUERY_LANDCOVER_CHECKPOINT"


def checkpoint_path() -> Path | None:
    raw = os.environ.get(CHECKPOINT_ENV, "").strip()
    if not raw:
        return None
    path = Path(raw)
    return path if path.is_file() else None


class LandCoverTool(Tool):
    """Multi-label land-cover classes over the analysed window."""

    def __init__(self, accepts: InputConfig, name: str = "landcover_cnn") -> None:
        self.accepts = accepts
        self._model: Any = None
        self._classes: tuple[str, ...] = CORINE_CLASSES
        self.spec = ToolSpec(
            name=name,
            version="1.0.0",
            accepts=accepts,
            tasks=(
                Task.VQA,
                Task.CAPTION,
                Task.GROUNDING,
                Task.CHANGE_VQA,
                Task.CHANGE_CAPTION,
                Task.CROSSMODAL_VQA,
            ),
            allowed_params={"min_confidence": (0.0, 1.0)},
            outputs=(
                "applicable",
                "classes",
                "confidences",
                "dominant_class",
                "evidence_line",
            ),
            summary=(
                "Multi-label land-cover classes from a CNN. Works without SWIR, "
                "so it produces built-up evidence where NDBI cannot be computed."
            ),
            kind="measurement",
            category="land-cover",
            cost="fast",
            requires="Any optical image; no specific band set.",
            emits_evidence=True,
            param_docs={
                "min_confidence": (
                    "Sigmoid probability below which a class is not reported."
                )
            },
        )

    def _load(self) -> Any:
        """Load the checkpoint once, lazily. Returns ``None`` when absent."""
        if self._model is not None:
            return self._model
        path = checkpoint_path()
        if path is None:
            return None

        import torch

        from satquery.cnn.model import build_model

        payload = torch.load(path, map_location="cpu", weights_only=True)
        # The class list travels with the weights: logit order is not
        # recoverable from the tensors, so a checkpoint read under a different
        # ordering scores every class against the wrong name.
        self._classes = tuple(payload.get("classes") or CORINE_CLASSES)
        model = build_model(
            channels=int(payload.get("channels", 12)),
            num_classes=len(self._classes),
            pretrained=False,
        )
        model.load_state_dict(payload["state_dict"])
        model.eval()
        self._model = model
        return model

    def run(self, ctx: RunContext, min_confidence: float = 0.35) -> ToolResult:
        model = self._load()
        if model is None:
            return ToolResult(
                outputs={
                    "applicable": False,
                    "reason": (
                        f"no land-cover checkpoint; set {CHECKPOINT_ENV} to one "
                        f"trained by scripts/train_landcover.py"
                    ),
                },
                confidence=None,
            )

        import numpy as np
        import torch

        from satquery.geo.raster import read_bands

        image = (
            ctx.require_role(ImageRole.OPTICAL)
            if self.accepts is InputConfig.CROSSMODAL_PAIR
            else ctx.images[0]
        )
        bands = read_bands(image.path)

        expected = int(model.conv1.in_channels)
        array = _fit_channels(np.asarray(bands, dtype=np.float32), expected)
        tensor = torch.from_numpy(array).unsqueeze(0)

        with torch.no_grad():
            probabilities = torch.sigmoid(model(tensor))[0].numpy()

        ranked = sorted(
            zip(self._classes, probabilities.tolist(), strict=True),
            key=lambda item: item[1],
            reverse=True,
        )
        kept = [(name, score) for name, score in ranked if score >= min_confidence]

        outputs: dict[str, Any] = {
            "applicable": True,
            "classes": [name for name, _ in kept],
            "confidences": {name: round(score, 4) for name, score in kept},
            "dominant_class": kept[0][0] if kept else None,
            # The string the preamble carries, rendered by the same function
            # that rendered it into the training corpus. _ARTIFACT_MAP promotes
            # this key, not the list above.
            "evidence_line": render_evidence_line([name for name, _ in kept]),
            # Named so the trace shows what the numbers are a claim about. The
            # taxonomy is European and the evaluation terrain is not.
            "taxonomy": "CORINE (BigEarthNet v2.0, Europe-trained)",
        }
        return ToolResult(
            outputs=outputs,
            # The model's own top probability, not a fixed number: a run where
            # nothing scored above the threshold should read as uncertain.
            confidence=round(float(ranked[0][1]), 3) if ranked else None,
        )


def _fit_channels(bands: Any, expected: int) -> Any:
    """Coerce a band stack to the width the checkpoint was trained at.

    Panchromatic input is repeated across channels rather than padded with
    zeros: a zero band is not "no information", it is a strong signal that the
    normalisation downstream was never trained to see. Extra bands are dropped
    from the end, which is where the ones we do not use sit in Sentinel order.
    """
    import numpy as np

    have = bands.shape[0]
    if have == expected:
        return bands
    if have > expected:
        return bands[:expected]
    repeats = (expected + have - 1) // have
    return np.tile(bands, (repeats, 1, 1))[:expected]


def build_landcover_tools() -> list[LandCoverTool]:
    """One instance per input configuration the registry keys on."""
    return [
        LandCoverTool(InputConfig.SINGLE, "landcover_cnn"),
        LandCoverTool(InputConfig.CROSSMODAL_PAIR, "landcover_cnn_crossmodal"),
        LandCoverTool(InputConfig.BITEMPORAL_PAIR, "landcover_cnn_bitemporal"),
    ]
