"""
Unit tests for benchmark/shared/scoring/ modules.

Run with:
    pytest benchmark/tests/test_scoring.py -v

No network access required — all tests use local data.
"""

from __future__ import annotations

import sys
from pathlib import Path

# Allow running from repo root without installing
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

import pytest

from benchmark.shared.scoring.rouge_score import score_rouge, score_stage_text
from benchmark.shared.scoring.concept_f1 import (
    concept_f1,
    score_differential_diagnosis,
    score_normalized_medications,
    score_drug_interactions,
)
from benchmark.shared.scoring.ndcg import score_differential_ndcg


# ---------------------------------------------------------------------------
# ROUGE
# ---------------------------------------------------------------------------

class TestScoreRouge:
    def test_identical_strings(self):
        text = "The patient presents with chest pain and shortness of breath."
        scores = score_rouge(text, text)
        assert scores["rouge1"] == pytest.approx(1.0, abs=1e-3)
        assert scores["rougeL"] == pytest.approx(1.0, abs=1e-3)

    def test_empty_hypothesis_returns_zeros(self):
        scores = score_rouge("", "some reference text")
        assert scores == {"rouge1": 0.0, "rouge2": 0.0, "rougeL": 0.0}

    def test_empty_reference_returns_zeros(self):
        scores = score_rouge("some hypothesis", "")
        assert scores == {"rouge1": 0.0, "rouge2": 0.0, "rougeL": 0.0}

    def test_both_empty_returns_zeros(self):
        scores = score_rouge("", "")
        assert scores == {"rouge1": 0.0, "rouge2": 0.0, "rougeL": 0.0}

    def test_partial_overlap(self):
        hyp = "patient has chest pain"
        ref = "patient presents with severe chest pain and dyspnea"
        scores = score_rouge(hyp, ref)
        assert 0.0 < scores["rouge1"] < 1.0

    def test_custom_metrics(self):
        scores = score_rouge("hello world", "hello world", metrics=["rouge1"])
        assert set(scores.keys()) == {"rouge1"}

    def test_returns_dict_with_default_keys(self):
        scores = score_rouge("a b c", "a b c d")
        assert set(scores.keys()) == {"rouge1", "rouge2", "rougeL"}
        assert all(isinstance(v, float) for v in scores.values())


class TestScoreStageText:
    def test_returns_stage_key(self):
        result = score_stage_text("transcription_cleaned", "text", "text")
        assert result["stage"] == "transcription_cleaned"

    def test_includes_rouge_keys(self):
        result = score_stage_text("clinical_summary", "hello", "hello world")
        assert "rouge1" in result
        assert "rougeL" in result

    def test_perfect_match(self):
        text = "Patient: I have a headache. Doctor: How long?"
        result = score_stage_text("transcription_cleaned", text, text)
        assert result["rouge1"] == pytest.approx(1.0, abs=1e-3)


# ---------------------------------------------------------------------------
# Concept F1
# ---------------------------------------------------------------------------

class TestConceptF1:
    def test_perfect_match(self):
        result = concept_f1(["angina", "GERD", "pericarditis"],
                            ["angina", "GERD", "pericarditis"])
        assert result == {"precision": 1.0, "recall": 1.0, "f1": 1.0}

    def test_both_empty(self):
        result = concept_f1([], [])
        assert result == {"precision": 1.0, "recall": 1.0, "f1": 1.0}

    def test_no_overlap(self):
        result = concept_f1(["angina"], ["GERD"])
        assert result == {"precision": 0.0, "recall": 0.0, "f1": 0.0}

    def test_partial_overlap(self):
        result = concept_f1(["angina", "GERD"], ["angina", "pericarditis"])
        assert result["precision"] == pytest.approx(0.5)
        assert result["recall"] == pytest.approx(0.5)
        assert result["f1"] == pytest.approx(0.5)

    def test_case_insensitive(self):
        result = concept_f1(["Angina"], ["angina"])
        assert result["f1"] == pytest.approx(1.0)

    def test_empty_predicted(self):
        result = concept_f1([], ["angina"])
        assert result["precision"] == 0.0
        assert result["recall"] == 0.0
        assert result["f1"] == 0.0

    def test_empty_expected(self):
        result = concept_f1(["angina"], [])
        assert result["precision"] == 0.0
        assert result["recall"] == 0.0
        assert result["f1"] == 0.0

    def test_duplicates_treated_as_set(self):
        # Duplicates shouldn't inflate scores — set-based matching
        result = concept_f1(["angina", "angina"], ["angina"])
        assert result["f1"] == pytest.approx(1.0)


class TestScoreDifferentialDiagnosis:
    PRED = [
        {"condition": "Unstable Angina", "pmid": "12345", "rationale": "..."},
        {"condition": "GERD", "pmid": None, "rationale": "..."},
    ]
    GT = [
        {"condition": "Unstable Angina", "pmid": "99999", "rationale": "..."},
        {"condition": "Pulmonary Embolism", "pmid": "88888", "rationale": "..."},
    ]

    def test_partial_match(self):
        result = score_differential_diagnosis(self.PRED, self.GT)
        assert result["precision"] == pytest.approx(0.5)
        assert result["recall"] == pytest.approx(0.5)

    def test_perfect_match(self):
        result = score_differential_diagnosis(self.PRED, self.PRED)
        assert result["f1"] == pytest.approx(1.0)

    def test_empty_both(self):
        result = score_differential_diagnosis([], [])
        assert result["f1"] == pytest.approx(1.0)


class TestScoreNormalizedMedications:
    PRED = [
        {"original": "warfarin 5mg", "rxnorm_id": "11289", "ingredient": "warfarin"},
        {"original": "aspirin 81mg", "rxnorm_id": "1191", "ingredient": "aspirin"},
    ]
    GT = [
        {"original": "warfarin 5mg daily", "rxnorm_id": "11289", "ingredient": "warfarin"},
        {"original": "aspirin 81mg", "rxnorm_id": "1191", "ingredient": "aspirin"},
        {"original": "metoprolol 25mg", "rxnorm_id": "41493", "ingredient": "metoprolol"},
    ]

    def test_partial_match(self):
        result = score_normalized_medications(self.PRED, self.GT)
        assert result["precision"] == pytest.approx(1.0)
        assert result["recall"] == pytest.approx(2 / 3, abs=1e-3)

    def test_none_ingredient_handled(self):
        pred = [{"original": "unknown drug", "rxnorm_id": None, "ingredient": None}]
        gt   = [{"original": "warfarin", "rxnorm_id": "11289", "ingredient": "warfarin"}]
        result = score_normalized_medications(pred, gt)
        assert result["f1"] == 0.0


class TestScoreDrugInteractions:
    PRED = [{"drug_a": "warfarin", "drug_b": "aspirin", "severity": "major", "recommendation": "..."}]
    GT   = [{"drug_a": "aspirin", "drug_b": "warfarin", "severity": "major", "recommendation": "..."}]

    def test_order_insensitive(self):
        # aspirin|warfarin == warfarin|aspirin
        result = score_drug_interactions(self.PRED, self.GT)
        assert result["f1"] == pytest.approx(1.0)

    def test_no_overlap(self):
        pred = [{"drug_a": "warfarin", "drug_b": "aspirin", "severity": "major", "recommendation": ""}]
        gt   = [{"drug_a": "metoprolol", "drug_b": "digoxin", "severity": "minor", "recommendation": ""}]
        result = score_drug_interactions(pred, gt)
        assert result["f1"] == 0.0

    def test_empty_both(self):
        result = score_drug_interactions([], [])
        assert result["f1"] == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# nDCG
# ---------------------------------------------------------------------------

class TestScoreDifferentialNdcg:
    EXPECTED = [
        {"condition": "Unstable Angina"},
        {"condition": "GERD"},
        {"condition": "Pericarditis"},
    ]

    def test_perfect_order(self):
        score = score_differential_ndcg(self.EXPECTED, self.EXPECTED)
        assert score == pytest.approx(1.0, abs=1e-3)

    def test_reversed_order(self):
        reversed_pred = list(reversed(self.EXPECTED))
        score = score_differential_ndcg(reversed_pred, self.EXPECTED)
        assert 0.0 <= score < 1.0

    def test_empty_predicted(self):
        assert score_differential_ndcg([], self.EXPECTED) == 0.0

    def test_empty_expected(self):
        assert score_differential_ndcg(self.EXPECTED, []) == 0.0

    def test_no_matching_conditions(self):
        pred = [{"condition": "Appendicitis"}, {"condition": "Pneumonia"}]
        score = score_differential_ndcg(pred, self.EXPECTED)
        assert score == 0.0

    def test_single_item_match(self):
        pred = [{"condition": "Unstable Angina"}]
        gt   = [{"condition": "Unstable Angina"}]
        score = score_differential_ndcg(pred, gt)
        assert score == pytest.approx(1.0)

    def test_single_item_no_match(self):
        pred = [{"condition": "Appendicitis"}]
        gt   = [{"condition": "Unstable Angina"}]
        score = score_differential_ndcg(pred, gt)
        assert score == 0.0

    def test_ndcg_at_k(self):
        # With k=1, only the top prediction matters
        pred_correct = [{"condition": "Unstable Angina"}, {"condition": "Appendicitis"}]
        pred_wrong   = [{"condition": "Appendicitis"}, {"condition": "Unstable Angina"}]
        score_good = score_differential_ndcg(pred_correct, self.EXPECTED, k=1)
        score_bad  = score_differential_ndcg(pred_wrong,   self.EXPECTED, k=1)
        assert score_good > score_bad
