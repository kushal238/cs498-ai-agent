"""
Concept-level F1 scoring for structured output stages.

Used for:
    Stage 3 — differential_diagnosis  (condition name match)
    Stage 4 — normalized_medications  (ingredient name match)
    Stage 5 — drug_interactions       (drug pair match)

Precision = |predicted ∩ expected| / |predicted|
Recall    = |predicted ∩ expected| / |expected|
F1        = 2 * P * R / (P + R)

Matching is case-insensitive exact string match by default; fuzzy matching is a TODO.

Depends on scikit-learn for macro/micro aggregation utilities.
Install: pip install scikit-learn
"""

from __future__ import annotations

import numpy as np  # noqa: F401  (imported to validate install)
from sklearn.metrics import precision_recall_fscore_support  # noqa: F401


def concept_f1(predicted: list[str], expected: list[str]) -> dict[str, float]:
    """Compute precision, recall, and F1 for a flat list of predicted vs expected concepts.

    Args:
        predicted: List of predicted concept strings (e.g. condition names, ingredient names).
        expected:  List of ground-truth concept strings.

    Returns:
        Dict with keys "precision", "recall", "f1".
    """
    pred_set = {s.lower().strip() for s in predicted if s}
    exp_set  = {s.lower().strip() for s in expected if s}

    if not pred_set and not exp_set:
        return {"precision": 1.0, "recall": 1.0, "f1": 1.0}

    intersection = pred_set & exp_set
    precision = len(intersection) / len(pred_set) if pred_set else 0.0
    recall    = len(intersection) / len(exp_set)  if exp_set  else 0.0
    f1 = (2 * precision * recall / (precision + recall)
          if (precision + recall) > 0 else 0.0)

    return {"precision": precision, "recall": recall, "f1": f1}


def score_differential_diagnosis(
    predicted: list[dict],
    expected: list[dict],
) -> dict[str, float]:
    """Score Stage 3 differential diagnosis output using concept F1 on condition names.

    Args:
        predicted: List of dicts with at least a "condition" key (model output).
        expected:  List of dicts with at least a "condition" key (ground truth).

    Returns:
        Dict with keys "precision", "recall", "f1".
    """
    pred_conditions = [d.get("condition", "") for d in predicted]
    exp_conditions  = [d.get("condition", "") for d in expected]
    return concept_f1(pred_conditions, exp_conditions)


def score_normalized_medications(
    predicted: list[dict],
    expected: list[dict],
) -> dict[str, float]:
    """Score Stage 4 medication normalization using concept F1 on ingredient names.

    Args:
        predicted: List of normalized medication dicts from model output.
        expected:  List of normalized medication dicts from ground truth.

    Returns:
        Dict with keys "precision", "recall", "f1".
    """
    pred_ingredients = [d.get("ingredient") or "" for d in predicted]
    exp_ingredients  = [d.get("ingredient") or "" for d in expected]
    return concept_f1(pred_ingredients, exp_ingredients)


def score_drug_interactions(
    predicted: list[dict],
    expected: list[dict],
) -> dict[str, float]:
    """Score Stage 5 drug interaction output using concept F1 on (drug_a, drug_b) pairs.

    Args:
        predicted: List of drug interaction dicts from model output.
        expected:  List of drug interaction dicts from ground truth.

    Returns:
        Dict with keys "precision", "recall", "f1".
    """
    def pair_key(d: dict) -> str:
        return "|".join(sorted([
            d.get("drug_a", "").lower().strip(),
            d.get("drug_b", "").lower().strip(),
        ]))

    pred_pairs = [pair_key(d) for d in predicted]
    exp_pairs  = [pair_key(d) for d in expected]
    return concept_f1(pred_pairs, exp_pairs)
