"""
Normalized Discounted Cumulative Gain (nDCG) for ranked differential diagnosis.

Used for:
    Stage 3 — differential_diagnosis  (ranking quality of the ordered diagnosis list)

nDCG rewards placing the correct diagnosis higher in the ranked list.
A perfect score of 1.0 means the predicted order exactly matches the ground-truth order.

Condition matching uses embedding cosine similarity (via a PubMed-fine-tuned sentence
encoder) rather than exact string equality, so paraphrased but clinically equivalent
condition names still contribute to the ranking score — e.g. "type 2 DM" will be
matched to the GT condition "Type 2 diabetes mellitus" and receive its relevance weight.

Depends on scikit-learn's ndcg_score utility and sentence-transformers.
Install: pip install scikit-learn sentence-transformers
"""

from __future__ import annotations

import numpy as np
from sklearn.metrics import ndcg_score

from shared.scoring.embeddings import (
    CONDITION_SIMILARITY_THRESHOLD,
    cosine_similarity,
)


def _embed_relevance(condition: str, relevance_map: dict[str, float]) -> float:
    """Return the relevance score for a condition using embedding similarity fallback.

    Tries an exact key lookup first (fast path for perfect matches).  If that
    misses, finds the GT condition with the highest cosine similarity and returns
    its relevance weight when similarity meets CONDITION_SIMILARITY_THRESHOLD.
    Returns 0.0 when no match is close enough.
    """
    key = condition.lower().strip()
    if key in relevance_map:
        return relevance_map[key]

    best_sim = 0.0
    best_rel = 0.0
    for gt_cond, rel in relevance_map.items():
        sim = cosine_similarity(condition, gt_cond)
        if sim > best_sim:
            best_sim = sim
            best_rel = rel

    return best_rel if best_sim >= CONDITION_SIMILARITY_THRESHOLD else 0.0


def score_differential_ndcg(
    predicted: list[dict],
    expected: list[dict],
    k: int | None = None,
) -> float:
    """Compute nDCG@k for a ranked differential diagnosis list.

    Args:
        predicted: Ordered list of differential diagnosis dicts from model output,
                   each with at least a "condition" key. Rank is implied by list order.
        expected:  Ordered list of differential diagnosis dicts from ground truth.
        k:         Cut-off rank. If None, uses full list length.

    Returns:
        nDCG score as a float in [0.0, 1.0].
    """
    if not predicted or not expected:
        return 0.0

    # Build relevance lookup: normalised condition string -> score
    # Rank 0 in expected = most relevant = highest relevance score
    n = len(expected)
    expected_relevance: dict[str, float] = {
        d.get("condition", "").lower().strip(): float(n - i)
        for i, d in enumerate(expected)
    }

    # Build relevance vector aligned to predicted order using embedding lookup
    pred_relevances = [
        _embed_relevance(d.get("condition", ""), expected_relevance)
        for d in predicted
    ]

    # Build ideal relevance vector (expected order, descending)
    ideal_relevances = sorted(expected_relevance.values(), reverse=True)
    # Pad or truncate to match predicted length
    ideal_padded = (ideal_relevances + [0.0] * len(predicted))[:len(predicted)]

    # Edge cases: all-zero relevance or single-item list
    if not any(ideal_padded) or not any(pred_relevances):
        return 0.0
    if len(ideal_padded) < 2:
        return 1.0 if pred_relevances[0] > 0 else 0.0

    y_true  = np.array([ideal_padded])
    y_score = np.array([pred_relevances])

    if k is not None:
        return float(ndcg_score(y_true, y_score, k=k))
    return float(ndcg_score(y_true, y_score))
