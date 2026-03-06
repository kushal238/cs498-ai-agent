"""
Unit tests for benchmark/harness/harness.py (host-side only, no Docker required).

Tests case discovery, ground truth loading, and score_case() using mock data.

Run with:
    pytest benchmark/tests/test_harness.py -v
"""

from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from benchmark.harness.harness import (
    discover_cases,
    load_input,
    load_ground_truth,
    score_case,
)

# ---------------------------------------------------------------------------
# Sample data
# ---------------------------------------------------------------------------

SAMPLE_INPUT = {
    "case_id": "test_001",
    "patient_transcript": "Doctor: How are you? Patient: Chest pain.",
    "chart_notes": "45 y/o male.",
    "patient_history": "HTN.",
    "medication_list": ["aspirin 81mg"],
}

SAMPLE_GROUND_TRUTH = {
    "case_id": "test_001",
    "transcription_cleaned": "Doctor: How are you? Patient: I have chest pain.",
    "clinical_summary": "45-year-old male with hypertension presents with chest pain.",
    "differential_diagnosis": [
        {"condition": "Unstable Angina", "pmid": "12345", "rationale": "Classic presentation."},
        {"condition": "GERD", "pmid": "67890", "rationale": "Common alternative."},
    ],
    "normalized_medications": [
        {"original": "aspirin 81mg", "rxnorm_id": "1191", "ingredient": "aspirin"},
    ],
    "drug_interactions": [],
    "final_report": {
        "subjective": "Patient reports chest pain.",
        "objective": "BP 140/90, HR 80.",
        "assessment": "Likely unstable angina.",
        "plan": "Admit for workup, start anticoagulation.",
    },
}

SAMPLE_PREDICTION = {
    "case_id": "test_001",
    "transcription_cleaned": "Doctor: How are you? Patient: I have chest pain.",
    "clinical_summary": "45-year-old male with hypertension presenting with chest pain.",
    "differential_diagnosis": [
        {"condition": "Unstable Angina", "pmid": "12345", "rationale": "..."},
    ],
    "normalized_medications": [
        {"original": "aspirin 81mg", "rxnorm_id": "1191", "ingredient": "aspirin"},
    ],
    "drug_interactions": [],
    "final_report": {
        "subjective": "Patient reports chest pain.",
        "objective": "BP 140/90.",
        "assessment": "Unstable angina.",
        "plan": "Admit for workup.",
    },
}


# ---------------------------------------------------------------------------
# Case discovery
# ---------------------------------------------------------------------------

class TestDiscoverCases:
    def test_finds_dirs_with_input_json(self, tmp_path):
        case_dir = tmp_path / "case_001"
        case_dir.mkdir()
        (case_dir / "input.json").write_text(json.dumps(SAMPLE_INPUT))

        # Create a dir without input.json — should not be discovered
        (tmp_path / "not_a_case").mkdir()

        cases = discover_cases(tmp_path)
        assert len(cases) == 1
        assert cases[0] == case_dir

    def test_returns_sorted(self, tmp_path):
        for name in ["case_003", "case_001", "case_002"]:
            d = tmp_path / name
            d.mkdir()
            (d / "input.json").write_text("{}")
        cases = discover_cases(tmp_path)
        names = [c.name for c in cases]
        assert names == sorted(names)

    def test_empty_dir_returns_empty(self, tmp_path):
        assert discover_cases(tmp_path) == []


class TestLoadInput:
    def test_loads_json(self, tmp_path):
        case_dir = tmp_path / "case_001"
        case_dir.mkdir()
        (case_dir / "input.json").write_text(json.dumps(SAMPLE_INPUT))
        result = load_input(case_dir)
        assert result["case_id"] == "test_001"


class TestLoadGroundTruth:
    def test_loads_from_ground_truths_dir(self, tmp_path, monkeypatch):
        # Patch GT_DIR to a temp directory
        import benchmark.harness.harness as harness_module
        gt_dir = tmp_path / "ground_truths"
        gt_dir.mkdir()
        (gt_dir / "test_001.json").write_text(json.dumps(SAMPLE_GROUND_TRUTH))
        monkeypatch.setattr(harness_module, "GT_DIR", gt_dir)

        result = load_ground_truth("test_001")
        assert result["case_id"] == "test_001"

    def test_missing_ground_truth_returns_none(self, tmp_path, monkeypatch):
        import benchmark.harness.harness as harness_module
        monkeypatch.setattr(harness_module, "GT_DIR", tmp_path)
        result = load_ground_truth("nonexistent_case")
        assert result is None


# ---------------------------------------------------------------------------
# Score case
# ---------------------------------------------------------------------------

class TestScoreCase:
    def test_returns_all_stage_keys(self):
        scores = score_case(SAMPLE_PREDICTION, SAMPLE_GROUND_TRUTH)
        assert "transcription_cleanup" in scores
        assert "clinical_summarization" in scores
        assert "differential_diagnosis" in scores
        assert "medication_normalization" in scores
        assert "drug_interaction_check" in scores
        assert "final_report_generation" in scores

    def test_transcription_rouge_scores(self):
        scores = score_case(SAMPLE_PREDICTION, SAMPLE_GROUND_TRUTH)
        tc = scores["transcription_cleanup"]
        assert "rouge1" in tc
        assert 0.0 <= tc["rouge1"] <= 1.0
        assert 0.0 <= tc["rougeL"] <= 1.0

    def test_perfect_transcription_score(self):
        perfect_pred = {**SAMPLE_PREDICTION,
                        "transcription_cleaned": SAMPLE_GROUND_TRUTH["transcription_cleaned"]}
        scores = score_case(perfect_pred, SAMPLE_GROUND_TRUTH)
        assert scores["transcription_cleanup"]["rouge1"] == pytest.approx(1.0, abs=1e-3)

    def test_differential_diagnosis_has_ndcg(self):
        scores = score_case(SAMPLE_PREDICTION, SAMPLE_GROUND_TRUTH)
        assert "ndcg" in scores["differential_diagnosis"]

    def test_final_report_has_soap_sections(self):
        scores = score_case(SAMPLE_PREDICTION, SAMPLE_GROUND_TRUTH)
        report_scores = scores["final_report_generation"]
        for section in ("subjective", "objective", "assessment", "plan"):
            assert f"{section}_rougeL" in report_scores

    def test_missing_fields_handled_gracefully(self):
        empty_pred = {"case_id": "test_001"}
        # Should not raise — missing fields default to empty strings/lists
        scores = score_case(empty_pred, SAMPLE_GROUND_TRUTH)
        assert scores["transcription_cleanup"]["rouge1"] == 0.0
        assert scores["differential_diagnosis"]["f1"] == 0.0

    def test_stub_output_scores_zero(self):
        """A stub pipeline (all None/[]) should score 0 everywhere."""
        stub_pred = {
            "case_id": "test_001",
            "transcription_cleaned": None,
            "clinical_summary": None,
            "differential_diagnosis": [],
            "normalized_medications": [],
            "drug_interactions": [],
            "final_report": {"subjective": None, "objective": None,
                             "assessment": None, "plan": None},
        }
        scores = score_case(stub_pred, SAMPLE_GROUND_TRUTH)
        assert scores["transcription_cleanup"]["rouge1"] == 0.0
        assert scores["clinical_summarization"]["rouge1"] == 0.0
        assert scores["differential_diagnosis"]["f1"] == 0.0
