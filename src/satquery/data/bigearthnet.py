"""BigEarthNet.txt ingestion, cleaning and instruction-format preparation.

Turns the published corpus into training records our own harness and serving
path agree on. Three jobs, in order: read the annotations, *remove the ones that
would teach the model to lie*, and render imagery with exactly the normalisation
inference uses.

The middle job is the one that matters and the reason this module exists rather
than a twenty-line script. BigEarthNet.txt is built over Europe and a fifth of
it asks questions whose answers are not in the pixels:

  mcq/country       "Identify which of the following countries is shown"
  mcq/season        "Which of the following seasons is shown in the image?"
  mcq/climate zone  "choose the climate zone shown in the satellite image"

Together those are ~1.39M of 9.55M rows. A model trained on them learns to state
a country confidently from a picture of farmland, and the evaluation set is
Indian. Captions carry the same contamination inline -- "captured in Austria
during summer ... within the 'cold, no dry season, warm summer' climate zone" --
so they are redacted rather than dropped, since the land-cover half is exactly
what we want.

  provenance: BIFOLD-BigEarthNetv2-0/BigEarthNet.txt, ben_txt_datamodule.py
  (LMDB layout, band names, S1/S2 key mapping)
"""

from __future__ import annotations

import json
import re
from collections.abc import Iterator, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

#: Sentinel-2 bands as stored, in the published order.
S2_BANDS = (
    "B01",
    "B02",
    "B03",
    "B04",
    "B05",
    "B06",
    "B07",
    "B08",
    "B8A",
    "B09",
    "B11",
    "B12",
)
#: Sentinel-1 bands. Values are **already in dB** -- the published statistics
#: give VV a mean of -12.64 and VH -19.35. Do not apply a linear-to-dB pass.
S1_BANDS = ("VV", "VH")

#: True-colour composite, matching what the serving path shows a model.
S2_RGB = ("B04", "B03", "B02")

#: Categories whose answer is not determinable from the imagery. Training on
#: these produces confident geographic hallucination on unseen regions, which is
#: precisely the failure mode that would wreck us on Indian evaluation scenes.
UNANSWERABLE_CATEGORIES = frozenset({"country", "season", "climate zone"})

#: Grounding boxes are published as ``[x1 y1, x2 y2]`` with 0-1 floats. Ours are
#: integers on a 0-1000 grid (see eval/prompts.py). Mismatch here does not raise,
#: it silently scores zero on every grounding item, so conversion is explicit.
_BOX = re.compile(
    r"\[\s*([-+0-9.eE]+)\s+([-+0-9.eE]+)\s*,\s*([-+0-9.eE]+)\s+([-+0-9.eE]+)\s*\]"
)
_POINT = re.compile(r"<point>\s*\(\s*([-+0-9.eE]+)\s*,\s*([-+0-9.eE]+)\s*\)\s*</point>")
_REF = re.compile(r"</?ref>")

#: Caption redaction. The climate zone is quote-delimited so it goes first and
#: unambiguously; the acquisition clause is then bounded by its own commas.
_CLIMATE_CLAUSE = re.compile(r'\s*(?:with)?in the\s*"[^"]*"\s*climate zone', re.I)
_CAPTURED_CLAUSE = re.compile(r",\s*captured[^,]*,\s*", re.I)
_TIDY = re.compile(r"\s{2,}")


class PreparationError(RuntimeError):
    """Raised when a record cannot be prepared and should be skipped."""


@dataclass(frozen=True, slots=True)
class Record:
    """One prepared instruction-tuning example.

    ``preamble`` is the measurement block the controller prepends at inference.
    It is stored separately from ``prompt`` so the mixture can be audited after
    the fact -- how many records carried evidence, of which kind -- while the
    rendered turn joins them exactly the way serving does.
    """

    sample_id: str
    patch_id: str
    task: str
    prompt: str
    answer: str
    images: tuple[str, ...]
    preamble: str = ""
    evidence_kind: str = "none"

    @property
    def rendered_prompt(self) -> str:
        """The human turn as the model sees it, preamble included."""
        from satquery.data.evidence import apply_preamble

        return apply_preamble(self.prompt, self.preamble)

    def as_jsonl(self) -> str:
        return json.dumps(
            {
                "id": self.sample_id,
                "patch_id": self.patch_id,
                "task": self.task,
                "images": list(self.images),
                # Recorded alongside the turn so a later ablation can split the
                # corpus by evidence kind without re-deriving the choice.
                "evidence_kind": self.evidence_kind,
                "conversations": [
                    {"from": "human", "value": self.rendered_prompt},
                    {"from": "gpt", "value": self.answer},
                ],
            },
            ensure_ascii=False,
        )


# -- cleaning ------------------------------------------------------------


def is_unanswerable(category: Any) -> bool:
    """Whether a row asks for something the pixels cannot support."""
    return str(category).strip().lower() in UNANSWERABLE_CATEGORIES


def redact_caption(caption: str, country: str | None = None) -> str | None:
    """Strip acquisition metadata from a caption, or reject it.

    The land-cover content is worth keeping; the country/season/climate-zone
    preamble is not, because no model can see it. Anything still naming the
    country after redaction is dropped rather than patched -- a caption we
    cannot clean confidently is not worth the hallucination risk.
    """
    text = _CLIMATE_CLAUSE.sub("", caption)
    text = _CAPTURED_CLAUSE.sub(" ", text)
    text = _TIDY.sub(" ", text).strip()
    text = re.sub(r"\s+([,.])", r"\1", text)

    if country and re.search(rf"\b{re.escape(str(country))}\b", text, re.I):
        return None
    return text or None


# -- grounding geometry --------------------------------------------------


def parse_box(text: str) -> tuple[float, float, float, float]:
    """Parse the published ``[x1 y1, x2 y2]`` form into unit floats."""
    match = _BOX.search(text or "")
    if not match:
        raise PreparationError(f"no bounding box in {text!r}")
    x1, y1, x2, y2 = (float(g) for g in match.groups())
    if x2 < x1:
        x1, x2 = x2, x1
    if y2 < y1:
        y1, y2 = y2, y1
    return x1, y1, x2, y2


def to_milli_box(box: Sequence[float]) -> list[int]:
    """Unit box to our 0-1000 integer grid, matching eval/prompts.py."""
    return [round(min(max(v, 0.0), 1.0) * 1000) for v in box]


def rewrite_prompt(text: str) -> str:
    """Convert the published markup into the convention our prompts use.

    ``<point>(0.82, 0.28)</point>`` becomes a 0-1000 coordinate pair and the
    ``<ref>`` wrapper is dropped, so a prepared prompt reads the same way the
    grounding tool phrases one at inference time.
    """

    def _point(match: re.Match[str]) -> str:
        x, y = float(match.group(1)), float(match.group(2))
        return f"({round(x * 1000)}, {round(y * 1000)})"

    return _REF.sub("", _POINT.sub(_point, text or "")).strip()


# -- imagery -------------------------------------------------------------


def stack_bands(
    payload: dict[str, np.ndarray], bands: Sequence[str], size: int = 120
) -> np.ndarray:
    """Stack named bands to ``(len(bands), size, size)``.

    Bands arrive at their native 10/20/60 m resolutions and therefore different
    array shapes, so each is resampled to a common grid before stacking.
    """
    planes = []
    for name in bands:
        if name not in payload:
            raise PreparationError(f"band {name} missing from patch")
        plane = np.asarray(payload[name], dtype=np.float32)
        if plane.shape[-2:] != (size, size):
            plane = _resample_nearest(plane, size)
        planes.append(plane)
    return np.stack(planes)


def _resample_nearest(plane: np.ndarray, size: int) -> np.ndarray:
    """Nearest-neighbour resample, matching the published datamodule default."""
    h, w = plane.shape[-2:]
    rows = (np.arange(size) * h / size).astype(np.int64).clip(0, h - 1)
    cols = (np.arange(size) * w / size).astype(np.int64).clip(0, w - 1)
    return plane[np.ix_(rows, cols)]


def sar_composite(vv: np.ndarray, vh: np.ndarray) -> np.ndarray:
    """VV / VH / VV-VH as a three-plane stack.

    The standard SAR visualisation. Inputs are already in dB, so the difference
    plane is the log-domain equivalent of the VV/VH ratio.
    """
    return np.stack([vv, vh, vv - vh])


def to_rgb8(stack: np.ndarray) -> np.ndarray:
    """Percentile-stretch a 3-plane stack to uint8 HWC.

    Delegates to the serving path's stretch so a patch rendered for training is
    byte-identical to the same patch rendered at inference. Two implementations
    of "the same" normalisation is how train/serve skew gets in.
    """
    from satquery.geo.raster import stretch_to_uint8

    return np.dstack([stretch_to_uint8(plane) for plane in stack[:3]])


# -- sampling ------------------------------------------------------------


def stratified_sample(
    frame: Any,
    per_type: int,
    seed: int = 1234,
    columns: Sequence[str] = ("type", "country", "season"),
) -> Any:
    """Sample ``per_type`` rows for each task type, spread across strata.

    The pilot adapter published by another team drew 1,500 rows per type from a
    single country in a single season. The corpus carries ``country``, ``season``
    and ``climate_zone`` precisely so that does not have to happen: sampling
    within each stratum keeps boreal Finland and Mediterranean Portugal both
    represented, which is the only breadth available in a Europe-only dataset.
    """
    import pandas as pd

    present = [c for c in columns if c in frame.columns]
    if "type" not in present:
        raise PreparationError("frame has no 'type' column to stratify on")
    strata_cols = [c for c in present if c != "type"]

    chunks = []
    for _, group in frame.groupby("type", sort=True):
        # Explicit iteration rather than groupby.apply: apply would have to drop
        # the grouping columns to avoid a pandas deprecation, and `country` is
        # needed downstream to redact captions.
        subsets = (
            [g for _, g in group.groupby(strata_cols, sort=True)]
            if strata_cols
            else [group]
        )
        take = max(per_type // max(len(subsets), 1), 1)
        picked = pd.concat(
            [s.sample(min(len(s), take), random_state=seed) for s in subsets],
            ignore_index=True,
        )
        if len(picked) > per_type:
            picked = picked.sample(per_type, random_state=seed)
        chunks.append(picked)

    return pd.concat(chunks, ignore_index=True)


# -- record construction -------------------------------------------------


def build_record(
    row: Any,
    images: Sequence[str],
    sample_id: str | None = None,
    measurements: Any = None,
    seed: int = 1234,
    preamble_rate: float | None = None,
) -> Record:
    """Turn one annotation row into a prepared record, or raise to skip it.

    When ``measurements`` are supplied the record may also carry an evidence
    preamble; see ``data/evidence.py`` for the mixing policy and why a record
    whose measurement disagrees with its gold answer gets none.
    """
    task = str(row["type"]).strip()
    category = row.get("category") if hasattr(row, "get") else row["category"]

    if is_unanswerable(category):
        raise PreparationError(f"category {category!r} is not visually answerable")

    prompt = rewrite_prompt(str(row["input"]))
    answer = str(row["output"]).strip()

    if task == "captioning":
        cleaned = redact_caption(
            answer, row.get("country") if hasattr(row, "get") else None
        )
        if cleaned is None:
            raise PreparationError(
                "caption could not be cleaned of acquisition metadata"
            )
        answer = cleaned
    elif task == "bounding box":
        answer = str(to_milli_box(parse_box(answer)))

    if not prompt or not answer:
        raise PreparationError("empty prompt or answer")

    identifier = str(sample_id if sample_id is not None else row["ID"])
    preamble, kind = "", "none"
    if measurements is not None:
        from satquery.data.evidence import DEFAULT_PREAMBLE_RATE, choose_evidence

        choice = choose_evidence(
            identifier,
            prompt,
            answer,
            measurements,
            seed=seed,
            rate=DEFAULT_PREAMBLE_RATE if preamble_rate is None else preamble_rate,
        )
        if choice.used:
            preamble, kind = choice.preamble, choice.kind.value

    return Record(
        sample_id=identifier,
        patch_id=str(row["patch_id"]),
        task=task,
        prompt=prompt,
        answer=answer,
        images=tuple(images),
        preamble=preamble,
        evidence_kind=kind,
    )


def prepare(
    frame: Any,
    image_paths: dict[str, tuple[str, ...]],
    measurements: dict[str, Any] | None = None,
    seed: int = 1234,
    preamble_rate: float | None = None,
) -> Iterator[Record]:
    """Yield prepared records, skipping rows that cannot be cleaned.

    ``measurements`` maps a patch id to what the specialists measured on it. A
    patch with no entry simply yields records with no preamble, so a partial
    measurement pass degrades the mixture rather than failing the run.
    """
    for row in frame.to_dict("records"):
        patch = str(row["patch_id"])
        if patch not in image_paths:
            continue
        try:
            yield build_record(
                row,
                image_paths[patch],
                measurements=(measurements or {}).get(patch),
                seed=seed,
                preamble_rate=preamble_rate,
            )
        except PreparationError:
            continue


def write_jsonl(records: Iterator[Record], path: str | Path) -> int:
    """Write records to JSONL, returning the count."""
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    written = 0
    with target.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(record.as_jsonl() + "\n")
            written += 1
    return written
