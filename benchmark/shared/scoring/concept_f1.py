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

    Raises:
        NotImplementedError: Always — scoring logic not yet implemented.

    TODO: Lowercase both lists, compute set intersection, then:
          precision = len(intersection) / len(predicted) if predicted else 0.0
          recall    = len(intersection) / len(expected)  if expected  else 0.0
          f1        = 2*p*r / (p+r) if (p+r) > 0 else 0.0
    """
    raise NotImplementedError("TODO: implement concept-level F1 scoring")


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

    Raises:
        NotImplementedError: Always — delegates to concept_f1.

    TODO: Extract condition names from both lists, call concept_f1().
    """
    raise NotImplementedError("TODO: implement differential diagnosis concept F1")


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

    Raises:
        NotImplementedError: Always — delegates to concept_f1.

    TODO: Extract ingredient names (lowercased) from both lists, call concept_f1().
    """
    raise NotImplementedError("TODO: implement medication normalization concept F1")


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

    Raises:
        NotImplementedError: Always — delegates to concept_f1.

    TODO: Represent each interaction as a frozenset({drug_a, drug_b}) for
          order-insensitive matching, then call concept_f1() on the string
          representations.
    """
    raise NotImplementedError("TODO: implement drug interaction pair concept F1")
