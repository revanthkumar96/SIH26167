"""VQA and change-VQA accuracy, including the OA / AA split.

Overall accuracy (OA) alone is misleading on RSVQA and CDVQA: both are class
imbalanced, so a model that always answers "no" can post a respectable OA. Average
accuracy (AA) -- the unweighted mean of per-question-type accuracies -- exposes that.
The judging table asks for both, so both are always reported.
"""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Sequence

from satquery.eval.normalize import answers_match


def vqa_metrics(
    predictions: Sequence[str],
    references: Sequence[str],
    qtypes: Sequence[str | None] | None = None,
) -> dict[str, float]:
    """Overall accuracy, per-type accuracy, and the unweighted per-type mean."""
    if len(predictions) != len(references):
        raise ValueError("predictions and references must be the same length")
    if not predictions:
        return {"oa": 0.0, "aa": 0.0, "n": 0.0}

    types = list(qtypes) if qtypes is not None else [None] * len(predictions)
    if len(types) != len(predictions):
        raise ValueError("qtypes must match predictions in length")

    correct = 0
    per_type: dict[str, list[int]] = defaultdict(list)

    for pred, ref, qtype in zip(predictions, references, types, strict=True):
        hit = int(answers_match(pred, ref))
        correct += hit
        if qtype:
            per_type[qtype].append(hit)

    results: dict[str, float] = {
        "oa": correct / len(predictions),
        "n": float(len(predictions)),
    }

    if per_type:
        accuracies = {}
        for qtype, hits in sorted(per_type.items()):
            accuracy = sum(hits) / len(hits)
            accuracies[qtype] = accuracy
            results[f"acc/{qtype}"] = accuracy
            results[f"n/{qtype}"] = float(len(hits))
        results["aa"] = sum(accuracies.values()) / len(accuracies)
    else:
        results["aa"] = results["oa"]

    return results
