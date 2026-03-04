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

import numpy as np  # noqa: F401  (imported to validate install)
from sklearn.metrics import ndcg_score  # noqa: F401


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

    Raises:
        NotImplementedError: Always — scoring logic not yet implemented.

    TODO:
        1. Build a relevance score vector for the predicted list:
           relevance[i] = (len(expected) - expected_rank) if condition in expected else 0
           where expected_rank is the 0-based position in the expected list.
        2. Build the ideal relevance vector from the expected list order.
        3. Reshape both into 2D arrays and call sklearn.metrics.ndcg_score(ideal, predicted, k=k).
    """
    raise NotImplementedError("TODO: implement nDCG scoring for ranked differential diagnosis")
