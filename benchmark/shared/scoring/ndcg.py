"""
Normalized Discounted Cumulative Gain (nDCG) for ranked differential diagnosis.

Used for:
    Stage 3 — differential_diagnosis  (ranking quality of the ordered diagnosis list)

nDCG rewards placing the correct diagnosis higher in the ranked list.
A perfect score of 1.0 means the predicted order exactly matches the ground-truth order.

Depends on scikit-learn's ndcg_score utility.
Install: pip install scikit-learn
"""

from __future__ import annotations

import numpy as np
from sklearn.metrics import ndcg_score


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

    # Build relevance lookup: condition -> score (higher = more relevant)
    # Rank 0 in expected = most relevant = highest relevance score
    n = len(expected)
    expected_relevance = {
        d.get("condition", "").lower().strip(): (n - i)
        for i, d in enumerate(expected)
    }

    # Build relevance vector aligned to predicted order
    pred_relevances = [
        expected_relevance.get(d.get("condition", "").lower().strip(), 0)
        for d in predicted
    ]

    # Build ideal relevance vector (expected order, descending)
    ideal_relevances = sorted(expected_relevance.values(), reverse=True)
    # Pad or truncate to match predicted length
    ideal_padded = (ideal_relevances + [0] * len(predicted))[:len(predicted)]

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
