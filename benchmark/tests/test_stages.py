"""Tests for individual stage scratchpad consumption."""
import sys
from pathlib import Path
from unittest.mock import patch, MagicMock

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "runner"))

import pytest
import stage_summarization
import stage_diagnosis


def _make_summarization_result(summary="test summary"):
    m = MagicMock()
    m.reasoning = "looked at transcript"
    m.confidence = "high"
    m.clinical_summary = summary
    return m


def _make_diagnosis_result():
    from stage_diagnosis import DiagnosisResult, _Diagnosis
    dx = _Diagnosis(condition="Unstable Angina", rationale="chest pain with exertion")
    return DiagnosisResult(
        reasoning="clinical reasoning",
        confidence="high",
        diagnoses=[dx],
    )


class TestSummarizationScratchpad:
    def test_scratchpad_included_in_user_message_when_present(self):
        with patch("llm_client.chat_structured", return_value=_make_summarization_result()) as mock_chat:
            context = {
                "transcription_cleaned": "Doctor: How are you? Patient: chest pain.",
                "chart_notes": "BP 140/90",
                "patient_history": {"age": 55, "sex": "M", "chief_complaint": "chest pain",
                                    "known_conditions": [], "known_allergies": []},
                "scratchpad_summary": "[transcription] Removed filler words. (confidence: high)",
            }
            stage_summarization.run(context)

            messages = mock_chat.call_args[0][0]
            user_content = next(m["content"] for m in messages if m["role"] == "user")
            assert "Prior stage reasoning" in user_content
            assert "[transcription]" in user_content

    def test_scratchpad_absent_when_not_in_context(self):
        with patch("llm_client.chat_structured", return_value=_make_summarization_result()) as mock_chat:
            context = {
                "transcription_cleaned": "Doctor: How are you? Patient: chest pain.",
                "chart_notes": "BP 140/90",
                "patient_history": {"age": 55, "sex": "M", "chief_complaint": "chest pain",
                                    "known_conditions": [], "known_allergies": []},
            }
            stage_summarization.run(context)

            messages = mock_chat.call_args[0][0]
            user_content = next(m["content"] for m in messages if m["role"] == "user")
            assert "Prior stage reasoning" not in user_content


class TestDiagnosisScratchpad:
    def test_scratchpad_included_in_user_message_when_present(self):
        with patch("llm_client.chat_structured", return_value=_make_diagnosis_result()) as mock_chat:
            with patch("stage_diagnosis.search_pubmed", return_value=["12345678"]):
                context = {
                    "clinical_summary": "55yo M with chest pain.",
                    "patient_history": {"age": 55, "sex": "M", "chief_complaint": "chest pain",
                                        "known_conditions": [], "known_allergies": []},
                    "scratchpad_summary": (
                        "[transcription] Cleaned dialogue. (confidence: high)\n"
                        "[summarization] Key findings extracted. (confidence: high)"
                    ),
                }
                stage_diagnosis.run(context)

                messages = mock_chat.call_args[0][0]
                user_content = next(m["content"] for m in messages if m["role"] == "user")
                assert "Prior stage reasoning" in user_content
                assert "[summarization]" in user_content

    def test_scratchpad_absent_when_not_in_context(self):
        with patch("llm_client.chat_structured", return_value=_make_diagnosis_result()) as mock_chat:
            with patch("stage_diagnosis.search_pubmed", return_value=[]):
                context = {
                    "clinical_summary": "55yo M with chest pain.",
                    "patient_history": {"age": 55, "sex": "M", "chief_complaint": "chest pain",
                                        "known_conditions": [], "known_allergies": []},
                }
                stage_diagnosis.run(context)

                messages = mock_chat.call_args[0][0]
                user_content = next(m["content"] for m in messages if m["role"] == "user")
                assert "Prior stage reasoning" not in user_content
