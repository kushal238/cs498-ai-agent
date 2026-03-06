"""
Unit tests for benchmark/runner/langgraph_runner.py.

Tests schema validation, case loading, and stub pipeline execution.
No network access required.

Run with:
    pytest benchmark/tests/test_pipeline.py -v
"""

from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path

import pytest

# Allow running from repo root
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "runner"))

from benchmark.runner.langgraph_runner import (
    load_schema,
    validate_input,
    run_pipeline,
    node_transcription_cleanup,
    node_clinical_summarization,
    node_differential_diagnosis,
    node_medication_normalization,
    node_drug_interaction_check,
    node_final_report_generation,
    WORKFLOW_STAGES,
)

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

MINIMAL_INPUT = {
    "case_id": "test_001",
    "data_source": "synthetic",
    "difficulty": "moderate",
    "patient_transcript": "Doctor: How are you? Patient: I have chest pain.",
    "chart_notes": "45 y/o male, HTN.",
    "patient_history": {
        "age": 45,
        "sex": "male",
        "chief_complaint": "chest pain",
        "known_conditions": ["hypertension"],
        "known_allergies": [],
    },
    "medication_list": ["aspirin 81mg", "metoprolol 25mg"],
}


# ---------------------------------------------------------------------------
# Schema validation
# ---------------------------------------------------------------------------

class TestValidateInput:
    def test_valid_input_passes(self):
        schema_path = (
            Path(__file__).resolve().parent.parent / "shared" / "schemas" / "input_schema.json"
        )
        schema = load_schema(schema_path)
        # Should not raise
        validate_input(MINIMAL_INPUT, schema)

    def test_missing_required_field_raises(self):
        import jsonschema
        schema_path = (
            Path(__file__).resolve().parent.parent / "shared" / "schemas" / "input_schema.json"
        )
        schema = load_schema(schema_path)
        bad_input = {k: v for k, v in MINIMAL_INPUT.items() if k != "case_id"}
        with pytest.raises(jsonschema.ValidationError):
            validate_input(bad_input, schema)

    def test_wrong_type_raises(self):
        import jsonschema
        schema_path = (
            Path(__file__).resolve().parent.parent / "shared" / "schemas" / "input_schema.json"
        )
        schema = load_schema(schema_path)
        bad_input = {**MINIMAL_INPUT, "medication_list": "aspirin"}  # should be list
        with pytest.raises(jsonschema.ValidationError):
            validate_input(bad_input, schema)


# ---------------------------------------------------------------------------
# Node stubs
# ---------------------------------------------------------------------------

class TestNodeStubs:
    """All stub nodes must return partial state dicts with the correct keys."""

    def test_transcription_cleanup_returns_key(self):
        result = node_transcription_cleanup(MINIMAL_INPUT)
        assert "transcription_cleaned" in result

    def test_clinical_summarization_returns_key(self):
        state = {**MINIMAL_INPUT, "transcription_cleaned": None}
        result = node_clinical_summarization(state)
        assert "clinical_summary" in result

    def test_differential_diagnosis_returns_list(self):
        state = {**MINIMAL_INPUT, "clinical_summary": None}
        result = node_differential_diagnosis(state)
        assert "differential_diagnosis" in result
        assert isinstance(result["differential_diagnosis"], list)

    def test_medication_normalization_returns_list(self):
        result = node_medication_normalization(MINIMAL_INPUT)
        assert "normalized_medications" in result
        assert isinstance(result["normalized_medications"], list)

    def test_drug_interaction_check_returns_list(self):
        state = {**MINIMAL_INPUT, "normalized_medications": []}
        result = node_drug_interaction_check(state)
        assert "drug_interactions" in result
        assert isinstance(result["drug_interactions"], list)

    def test_final_report_generation_returns_dict(self):
        state = {
            **MINIMAL_INPUT,
            "transcription_cleaned": None,
            "clinical_summary": None,
            "differential_diagnosis": [],
            "normalized_medications": [],
            "drug_interactions": [],
        }
        result = node_final_report_generation(state)
        assert "final_report" in result
        report = result["final_report"]
        assert isinstance(report, dict)
        for section in ("subjective", "objective", "assessment", "plan"):
            assert section in report


# ---------------------------------------------------------------------------
# Full pipeline
# ---------------------------------------------------------------------------

class TestRunPipeline:
    def test_returns_all_output_keys(self):
        result = run_pipeline(MINIMAL_INPUT)
        expected_keys = {
            "transcription_cleaned",
            "clinical_summary",
            "differential_diagnosis",
            "normalized_medications",
            "drug_interactions",
            "final_report",
        }
        assert expected_keys.issubset(set(result.keys()))

    def test_input_fields_preserved(self):
        result = run_pipeline(MINIMAL_INPUT)
        assert result["case_id"] == MINIMAL_INPUT["case_id"]
        assert result["medication_list"] == MINIMAL_INPUT["medication_list"]

    def test_workflow_stages_constant(self):
        assert len(WORKFLOW_STAGES) == 6
        assert WORKFLOW_STAGES[0] == "transcription_cleanup"
        assert WORKFLOW_STAGES[-1] == "final_report_generation"


# ---------------------------------------------------------------------------
# Case loader
# ---------------------------------------------------------------------------

class TestLoadCase:
    def test_loads_valid_case(self):
        from benchmark.runner.langgraph_runner import load_case
        cases_dir = Path(__file__).resolve().parent.parent / "cases"
        case_dirs = [p for p in cases_dir.iterdir() if p.is_dir() and (p / "input.json").exists()]
        if not case_dirs:
            pytest.skip("No cases found in benchmark/cases/")
        result = load_case(case_dirs[0])
        assert "case_id" in result

    def test_missing_input_raises(self):
        from benchmark.runner.langgraph_runner import load_case
        with tempfile.TemporaryDirectory() as tmp:
            with pytest.raises(FileNotFoundError):
                load_case(tmp)
