"""Multi-label classification metrics.

Multi-label classification is not accuracy, and reporting accuracy here would
flatter badly: with 19 classes and two or three positive per patch, a model that
predicts nothing at all is right about 87% of the time per label.

So: mean average precision as the primary number, because that is what the reBEN
literature reports and it makes our result comparable; micro and macro F1
together, because CORINE is severely imbalanced and macro is what exposes the
rare-class failure micro hides; and per-class AP, because the classes our queries
actually turn on -- water bodies, urban fabric, industrial units, arable land --
are not the classes that dominate the average.

Written here rather than pulled from scikit-learn for the same reason the caption
metrics are: the constants stay visible and the scoring path has no dependency
that has to be installed on the box before a number can be produced.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np


def average_precision(scores: np.ndarray, targets: np.ndarray) -> float:
    """Area under the precision-recall curve for one class.

    The interpolation-free definition used by the detection literature: the mean
    of precision at each rank where a positive is retrieved. Returns ``nan`` for
    a class with no positives -- an undefined score, which is different from a
    zero and must not be averaged as one.
    """
    positives = int(targets.sum())
    if positives == 0:
        return float("nan")

    order = np.argsort(-scores, kind="stable")
    hits = targets[order]
    cumulative = np.cumsum(hits)
    ranks = np.arange(1, len(hits) + 1)
    precision_at_hit = cumulative[hits > 0] / ranks[hits > 0]
    return float(precision_at_hit.sum() / positives)


def _counts(pred: np.ndarray, target: np.ndarray) -> tuple[int, int, int]:
    tp = int(np.sum((pred == 1) & (target == 1)))
    fp = int(np.sum((pred == 1) & (target == 0)))
    fn = int(np.sum((pred == 0) & (target == 1)))
    return tp, fp, fn


def _f1(tp: int, fp: int, fn: int) -> float:
    denominator = 2 * tp + fp + fn
    return (2.0 * tp / denominator) if denominator else 0.0


def classification_metrics(
    logits: np.ndarray,
    targets: np.ndarray,
    classes: Sequence[str],
    threshold: float = 0.5,
) -> dict[str, float]:
    """Score a multi-label run.

    ``logits`` and ``targets`` are ``(samples, classes)``. Logits are passed
    through a sigmoid here rather than by the caller, so the threshold always
    means a probability.
    """
    if logits.shape != targets.shape:
        raise ValueError(
            f"logits {logits.shape} and targets {targets.shape} must match"
        )
    if logits.shape[1] != len(classes):
        raise ValueError(
            f"{logits.shape[1]} logits against {len(classes)} class names; a "
            f"mismatch here silently scores every class against the wrong label"
        )

    probabilities = 1.0 / (1.0 + np.exp(-logits))
    predictions = (probabilities >= threshold).astype(np.int8)
    targets = targets.astype(np.int8)

    results: dict[str, float] = {"n": float(len(targets))}

    per_class: list[float] = []
    macro_f1: list[float] = []
    for index, name in enumerate(classes):
        ap = average_precision(probabilities[:, index], targets[:, index])
        key = name.lower().replace(" ", "_").replace(",", "")[:48]
        if not np.isnan(ap):
            results[f"ap/{key}"] = float(ap)
            per_class.append(float(ap))
        # Support is reported so a reader can tell a hard class from an absent
        # one; a macro mean over classes with three positives is not a score.
        support = int(targets[:, index].sum())
        results[f"support/{key}"] = float(support)
        if support:
            macro_f1.append(_f1(*_counts(predictions[:, index], targets[:, index])))

    results["map"] = float(np.mean(per_class)) if per_class else 0.0
    results["f1_micro"] = _f1(*_counts(predictions.ravel(), targets.ravel()))
    results["f1_macro"] = float(np.mean(macro_f1)) if macro_f1 else 0.0
    results["classes_scored"] = float(len(per_class))
    return results
