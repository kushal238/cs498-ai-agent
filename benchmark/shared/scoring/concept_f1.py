"""
Concept-level F1 scoring for structured output stages.

Used for:
    Stage 3 — differential_diagnosis  (condition name match)
    Stage 4 — normalized_medications  (ingredient name match)
    Stage 5 — drug_interactions       (drug pair match)

Precision = |predicted ∩ expected| / |predicted|
Recall    = |predicted ∩ expected| / |expected|
F1        = 2 * P * R / (P + R)

Differential diagnosis uses embedding cosine similarity (via a PubMed-fine-tuned
sentence encoder) so that clinically equivalent condition names receive credit
regardless of phrasing — e.g. "type 2 DM" matches "Type 2 diabetes mellitus".
Medication and drug-pair stages continue to use exact token-bag matching, which
is appropriate for canonical ingredient names returned by RxNorm.

Depends on scikit-learn for macro/micro aggregation utilities and
sentence-transformers for condition embeddings.
Install: pip install scikit-learn sentence-transformers
"""

from __future__ import annotations

import re

import numpy as np  # noqa: F401  (imported to validate install)
from sklearn.metrics import precision_recall_fscore_support  # noqa: F401

from shared.scoring.embeddings import (
    CONDITION_SIMILARITY_THRESHOLD,
    best_match_similarity,
)


def _tokenize_concepts(concepts: list[str]) -> set[str]:
    """Convert concept strings into a normalized alphanumeric token set."""
    tokens: set[str] = set()
    for concept in concepts:
        if not concept:
            continue
        normalized = concept.lower().strip()
        tokens.update(re.findall(r"[a-z0-9]+", normalized))
    return tokens


def concept_f1(predicted: list[str], expected: list[str]) -> dict[str, float]:
    """Compute precision, recall, and F1 over whitespace-tokenized concept strings.

    Args:
        predicted: List of predicted concept strings (e.g. ingredient names).
        expected:  List of ground-truth concept strings.

    Returns:
        Dict with keys "precision", "recall", "f1".
    """
    pred_set = _tokenize_concepts(predicted)
    exp_set  = _tokenize_concepts(expected)

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
    """Score Stage 3 differential diagnosis using embedding-based semantic matching.

    Each predicted condition is embedded with a biomedical sentence encoder and
    matched to the closest expected condition by cosine similarity.  A pair is
    counted as a match when similarity meets or exceeds CONDITION_SIMILARITY_THRESHOLD
    (default 0.70, overridable via SCORING_EMBED_THRESHOLD env var).

    This handles paraphrases, abbreviations, and different specificity levels,
    e.g. "type 2 DM" matching "Type 2 diabetes mellitus", or "brachial artery
    pseudoaneurysm" matching "iatrogenic pseudoaneurysm of the right distal
    brachial artery".

    Args:
        predicted: List of dicts with at least a "condition" key (model output).
        expected:  List of dicts with at least a "condition" key (ground truth).

    Returns:
        Dict with keys "precision", "recall", "f1".
    """
    pred_conditions = [d.get("condition", "") for d in predicted]
    exp_conditions  = [d.get("condition", "") for d in expected]

    if not pred_conditions and not exp_conditions:
        return {"precision": 1.0, "recall": 1.0, "f1": 1.0}

    matched_pred = sum(
        1 for pc in pred_conditions
        if best_match_similarity(pc, exp_conditions) >= CONDITION_SIMILARITY_THRESHOLD
    )
    matched_exp = sum(
        1 for ec in exp_conditions
        if best_match_similarity(ec, pred_conditions) >= CONDITION_SIMILARITY_THRESHOLD
    )

    precision = matched_pred / len(pred_conditions) if pred_conditions else 0.0
    recall    = matched_exp  / len(exp_conditions)  if exp_conditions  else 0.0
    f1 = (2 * precision * recall / (precision + recall)
          if (precision + recall) > 0 else 0.0)

    return {"precision": precision, "recall": recall, "f1": f1}


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
    """Score Stage 5 drug interaction output.

    Two sub-scores are combined:

    1. Pair F1 — concept F1 on (drug_a, drug_b) pair keys (order-insensitive).
       Measures whether the agent identified the correct drug pairs.

    2. Recommendation ROUGE-L — for each matched pair, computes ROUGE-L between
       the predicted recommendation and the ground-truth recommendation, then
       averages across all matched pairs. Unmatched predicted pairs contribute
       0.0 to this average. This distinguishes a clean clinical sentence from a
       raw pasted FDA label table, which pair F1 alone cannot.

    Args:
        predicted: List of drug interaction dicts from model output.
        expected:  List of drug interaction dicts from ground truth.

    Returns:
        Dict with keys "precision", "recall", "f1", "recommendation_rougeL".
    """
    from shared.scoring.rouge_score import score_rouge  # noqa: PLC0415 (avoid circular at module level)

    def pair_key(d: dict) -> str:
        return "|".join(sorted([
            d.get("drug_a", "").lower().strip(),
            d.get("drug_b", "").lower().strip(),
        ]))

    pred_pairs = [pair_key(d) for d in predicted]
    exp_pairs  = [pair_key(d) for d in expected]
    pair_scores = concept_f1(pred_pairs, exp_pairs)

    # Build a lookup from pair key → recommendation for the expected interactions
    exp_rec_map: dict[str, str] = {
        pair_key(d): (d.get("recommendation") or "")
        for d in expected
    }

    # For each predicted interaction that has a matching GT pair, score the
    # recommendation text with ROUGE-L.
    rouge_scores: list[float] = []
    for pred_dict, pk in zip(predicted, pred_pairs):
        if pk in exp_rec_map:
            pred_rec = pred_dict.get("recommendation") or ""
            gt_rec   = exp_rec_map[pk]
            rougeL   = score_rouge(pred_rec, gt_rec, metrics=["rougeL"])["rougeL"]
            rouge_scores.append(rougeL)
        else:
            rouge_scores.append(0.0)

    # If there were no predicted interactions, score depends on whether any were expected.
    # predicted=[] + expected=[] → both correct → 1.0
    # predicted=[] + expected!=[] → agent missed everything → 0.0
    if not rouge_scores:
        rec_rougeL = 0.0 if expected else 1.0
    else:
        rec_rougeL = sum(rouge_scores) / len(rouge_scores)

    return {**pair_scores, "recommendation_rougeL": rec_rougeL}
