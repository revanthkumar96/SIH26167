"""The BigEarthNet v2.0 label set.

BigEarthNet v2.0 labels each patch with a subset of the 19-class nomenclature
that the v2.0 release consolidated the original 43 CORINE classes into. The list
below is that nomenclature, in the order the reBEN literature reports it.

**It is a default, not the authority.** ``metadata.parquet`` ships with the image
store and carries the actual ``labels`` column, so training derives its class
list from the data and writes it into the checkpoint; this constant is what the
tool falls back to and what tests pin. A hardcoded list that silently disagrees
with the data it is scoring is the same failure as the runtime table that
predated the Ollama backend, and it would show up as a permanently-zero average
precision on whichever classes had shifted position -- so ``verify_classes``
exists to make the disagreement loud.

Class order matters beyond bookkeeping: it fixes which logit means which class,
so a checkpoint trained under one ordering and read under another is not
recoverable from the weights alone.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence

#: The 19-class BigEarthNet v2.0 nomenclature. Verify against the shipped
#: metadata before trusting it for a scored run -- see ``verify_classes``.
CORINE_CLASSES: tuple[str, ...] = (
    "Urban fabric",
    "Industrial or commercial units",
    "Arable land",
    "Permanent crops",
    "Pastures",
    "Complex cultivation patterns",
    "Land principally occupied by agriculture, with significant areas of "
    "natural vegetation",
    "Agro-forestry areas",
    "Broad-leaved forest",
    "Coniferous forest",
    "Mixed forest",
    "Natural grassland and sparsely vegetated areas",
    "Moors, heathland and sclerophyllous vegetation",
    "Transitional woodland/shrub",
    "Beaches, dunes, sands",
    "Inland wetlands",
    "Coastal wetlands",
    "Inland waters",
    "Marine waters",
)

#: Classes our queries actually turn on. Reported individually because the
#: aggregate hides them: CORINE is severely imbalanced, and a model that never
#: predicts coastal wetlands can still post a respectable micro F1.
CLASSES_OF_INTEREST: tuple[str, ...] = (
    "Urban fabric",
    "Industrial or commercial units",
    "Inland waters",
    "Marine waters",
    "Arable land",
)


class LabelMismatchError(ValueError):
    """The data's class set is not the one the model was built for."""


def class_index(classes: Sequence[str] | None = None) -> dict[str, int]:
    """Map class name to logit position."""
    return {name: i for i, name in enumerate(classes or CORINE_CLASSES)}


def encode(labels: Iterable[str], classes: Sequence[str] | None = None) -> list[float]:
    """One multi-hot target row.

    Unknown labels are ignored rather than raising: a single unrecognised label
    should not abort a 549k-patch pass. ``verify_classes`` is the place that
    fails loudly, once, before training starts.
    """
    index = class_index(classes)
    vector = [0.0] * len(index)
    for label in labels:
        position = index.get(str(label).strip())
        if position is not None:
            vector[position] = 1.0
    return vector


def indices_to_labels(
    indices: Iterable[int], classes: Sequence[str] | None = None
) -> list[str]:
    names = list(classes or CORINE_CLASSES)
    return [names[i] for i in indices if 0 <= i < len(names)]


def render_evidence_line(labels: Iterable[str], limit: int = 4) -> str | None:
    """Render class names as the evidence preamble carries them.

    One implementation, called by the tool at inference and by the corpus
    preparation that trains the model to expect the line. Two renderings of
    "the same" list is how the adapted model ends up meeting a format it never
    saw -- the same failure the preamble itself exists to prevent.

    Truncated because the preamble is a prompt, not a report: nineteen classes
    at four decimal places would crowd out the question.
    """
    cleaned = [str(label).strip() for label in labels if str(label).strip()]
    if not cleaned:
        return None
    return ", ".join(cleaned[:limit]).lower()


def observed_classes(rows: Iterable[Iterable[str]]) -> list[str]:
    """Every distinct label present in the data, sorted for a stable ordering."""
    seen: set[str] = set()
    for row in rows:
        seen.update(str(label).strip() for label in row if str(label).strip())
    return sorted(seen)


def verify_classes(
    observed: Sequence[str], expected: Sequence[str] | None = None
) -> None:
    """Fail loudly when the shipped labels are not the ones we encode.

    Called once before training. Silence here would mean every logit is offset
    against the class it is supposed to mean, and the only symptom would be
    inexplicably poor per-class average precision.
    """
    reference = set(expected or CORINE_CLASSES)
    found = set(observed)
    if found == reference:
        return

    missing = sorted(reference - found)
    unexpected = sorted(found - reference)
    raise LabelMismatchError(
        "BigEarthNet label set does not match CORINE_CLASSES. "
        f"Absent from the data: {missing or 'none'}. "
        f"Present but unknown: {unexpected or 'none'}. "
        "Pass the observed classes through explicitly, and record them in the "
        "checkpoint, rather than editing the constant to match."
    )
