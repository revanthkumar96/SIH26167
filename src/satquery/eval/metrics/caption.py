"""Captioning metrics: BLEU, ROUGE-L and CIDEr-D.

Implemented in-repo so scoring needs no Java, no network and no NLTK download.
METEOR is optional and returns ``None`` when NLTK data is unavailable.

Tokenisation here is a simple lowercase/strip-punctuation split rather than the PTB
tokenizer used by ``pycocoevalcap``. Scores are therefore consistent *within this
repo* -- which is what the bake-off needs -- but may differ by a small margin from
numbers published in papers. Use the official tooling for any cross-paper claim.
"""

from __future__ import annotations

import math
import re
from collections import Counter, defaultdict
from collections.abc import Sequence

_PUNCT = re.compile(r"[^\w\s]")
_SIGMA = 6.0
_MAX_N = 4


def tokenize(text: str) -> list[str]:
    return _PUNCT.sub(" ", text.lower()).split()


def _ngram_counts(tokens: Sequence[str], n: int) -> Counter[tuple[str, ...]]:
    return Counter(tuple(tokens[i : i + n]) for i in range(len(tokens) - n + 1))


def _closest_ref_len(cand_len: int, ref_lens: Sequence[int]) -> int:
    return min(ref_lens, key=lambda r: (abs(r - cand_len), r))


def bleu(
    candidates: Sequence[str],
    references: Sequence[Sequence[str]],
    max_n: int = _MAX_N,
) -> dict[str, float]:
    """Corpus-level BLEU-1..N with clipping and brevity penalty."""
    if len(candidates) != len(references):
        raise ValueError("candidates and references must be the same length")

    clipped = [0] * max_n
    totals = [0] * max_n
    cand_total = 0
    ref_total = 0

    for cand, refs in zip(candidates, references, strict=True):
        cand_tokens = tokenize(cand)
        ref_token_lists = [tokenize(r) for r in refs] or [[]]
        cand_total += len(cand_tokens)
        ref_total += _closest_ref_len(
            len(cand_tokens), [len(r) for r in ref_token_lists]
        )

        for n in range(1, max_n + 1):
            cand_counts = _ngram_counts(cand_tokens, n)
            if not cand_counts:
                continue
            max_ref: Counter[tuple[str, ...]] = Counter()
            for ref_tokens in ref_token_lists:
                for gram, count in _ngram_counts(ref_tokens, n).items():
                    if count > max_ref[gram]:
                        max_ref[gram] = count
            clipped[n - 1] += sum(min(c, max_ref[g]) for g, c in cand_counts.items())
            totals[n - 1] += sum(cand_counts.values())

    brevity = 1.0
    if cand_total == 0:
        brevity = 0.0
    elif cand_total < ref_total:
        brevity = math.exp(1.0 - ref_total / cand_total)

    scores: dict[str, float] = {}
    log_sum = 0.0
    for n in range(1, max_n + 1):
        precision = clipped[n - 1] / totals[n - 1] if totals[n - 1] else 0.0
        log_sum += math.log(precision) if precision > 0 else -math.inf
        geo_mean = math.exp(log_sum / n) if log_sum != -math.inf else 0.0
        scores[f"bleu{n}"] = brevity * geo_mean
    return scores


def _lcs_length(a: Sequence[str], b: Sequence[str]) -> int:
    if not a or not b:
        return 0
    prev = [0] * (len(b) + 1)
    for token_a in a:
        curr = [0]
        for j, token_b in enumerate(b):
            curr.append(
                prev[j] + 1 if token_a == token_b else max(curr[j], prev[j + 1])
            )
        prev = curr
    return prev[-1]


def rouge_l(
    candidates: Sequence[str],
    references: Sequence[Sequence[str]],
    beta: float = 1.2,
) -> float:
    """Mean sentence-level ROUGE-L F-measure, taking the best reference per sample."""
    if not candidates:
        return 0.0
    total = 0.0
    for cand, refs in zip(candidates, references, strict=True):
        cand_tokens = tokenize(cand)
        best = 0.0
        for ref in refs:
            ref_tokens = tokenize(ref)
            lcs = _lcs_length(cand_tokens, ref_tokens)
            if lcs == 0:
                continue
            precision = lcs / len(cand_tokens)
            recall = lcs / len(ref_tokens)
            denom = recall + beta**2 * precision
            if denom > 0:
                best = max(best, (1 + beta**2) * precision * recall / denom)
        total += best
    return total / len(candidates)


def cider_d(
    candidates: Sequence[str],
    references: Sequence[Sequence[str]],
    max_n: int = _MAX_N,
) -> float:
    """CIDEr-D with tf-idf weighting, count clipping and a length penalty.

    Document frequencies are computed over the evaluation set's own references, as
    in the reference implementation, so the score is only meaningful for a full
    corpus -- not for a single sample.
    """
    num_images = len(candidates)
    if num_images == 0:
        return 0.0

    doc_freq: dict[tuple[str, ...], int] = defaultdict(int)
    for refs in references:
        seen: set[tuple[str, ...]] = set()
        for ref in refs:
            tokens = tokenize(ref)
            for n in range(1, max_n + 1):
                seen.update(_ngram_counts(tokens, n))
        for gram in seen:
            doc_freq[gram] += 1

    log_num_images = math.log(max(num_images, 1))

    def to_vec(
        tokens: Sequence[str],
    ) -> tuple[list[dict[tuple[str, ...], float]], list[float], int]:
        vectors: list[dict[tuple[str, ...], float]] = [{} for _ in range(max_n)]
        norms = [0.0] * max_n
        for n in range(1, max_n + 1):
            for gram, count in _ngram_counts(tokens, n).items():
                idf = log_num_images - math.log(max(1.0, doc_freq.get(gram, 0)))
                value = count * idf
                vectors[n - 1][gram] = value
                norms[n - 1] += value**2
        return vectors, [math.sqrt(x) for x in norms], len(tokens)

    total = 0.0
    for cand, refs in zip(candidates, references, strict=True):
        if not refs:
            continue
        cand_vec, cand_norm, cand_len = to_vec(tokenize(cand))
        per_n = [0.0] * max_n
        for ref in refs:
            ref_vec, ref_norm, ref_len = to_vec(tokenize(ref))
            delta = float(cand_len - ref_len)
            for n in range(max_n):
                value = sum(
                    min(weight, ref_vec[n].get(gram, 0.0)) * ref_vec[n].get(gram, 0.0)
                    for gram, weight in cand_vec[n].items()
                )
                if cand_norm[n] > 0 and ref_norm[n] > 0:
                    value /= cand_norm[n] * ref_norm[n]
                per_n[n] += value * math.exp(-(delta**2) / (2 * _SIGMA**2))
        total += 10.0 * sum(per_n) / (max_n * len(refs))
    return total / num_images


def meteor(
    candidates: Sequence[str], references: Sequence[Sequence[str]]
) -> float | None:
    """METEOR via NLTK if its data is present, else ``None``.

    Optional by design: the bake-off must not fail because a wordnet download was
    unavailable on an ephemeral GPU box.
    """
    try:
        from nltk.translate.meteor_score import meteor_score
    except ImportError:
        return None
    try:
        scores = [
            meteor_score([tokenize(r) for r in refs], tokenize(cand))
            for cand, refs in zip(candidates, references, strict=True)
        ]
    except LookupError:
        return None
    return sum(scores) / len(scores) if scores else 0.0


def caption_metrics(
    candidates: Sequence[str],
    references: Sequence[Sequence[str]],
    with_meteor: bool = False,
) -> dict[str, float]:
    """All captioning metrics prescribed by the judging table."""
    results: dict[str, float] = dict(bleu(candidates, references))
    results["rouge_l"] = rouge_l(candidates, references)
    results["cider_d"] = cider_d(candidates, references)
    if with_meteor:
        score = meteor(candidates, references)
        if score is not None:
            results["meteor"] = score
    return results
