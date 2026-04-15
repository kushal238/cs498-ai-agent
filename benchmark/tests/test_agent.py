import sys
from pathlib import Path
from unittest.mock import patch
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "runner"))


def _make_stage_fn(output_key, output_value):
    def stage_fn(context):
        return {
            "reasoning": "mock reasoning",
            "confidence": "high",
            "output": {output_key: output_value},
        }
    return stage_fn


MOCK_STAGE_MAP = {
    "transcription": _make_stage_fn("transcription_cleaned", "clean"),
    "summarization": _make_stage_fn("clinical_summary", "summary"),
    "diagnosis": _make_stage_fn("differential_diagnosis", []),
    "medications": _make_stage_fn("normalized_medications", []),
    "interactions": _make_stage_fn("drug_interactions", []),
    "report": _make_stage_fn("final_report", {
        "subjective": "s", "objective": "o", "assessment": "a", "plan": "p"
    }),
}


def test_agent_run_returns_all_output_fields():
    import executor
    executor.STAGE_MAP = MOCK_STAGE_MAP

    from agent import ClinicalAgent
    task = {
        "case_id": "test_01",
        "patient_transcript": "raw",
        "chart_notes": "notes",
        "medication_list": ["warfarin"],
        "patient_history": {"age": 45},
    }
    result = ClinicalAgent().run(task)
    assert result["case_id"] == "test_01"
    assert result["transcription_cleaned"] == "clean"
    assert result["clinical_summary"] == "summary"
    assert result["differential_diagnosis"] == []
    assert result["normalized_medications"] == []
    assert result["drug_interactions"] == []
    assert result["final_report"]["subjective"] == "s"


def test_agent_skips_med_stages_when_no_medications():
    import executor
    from state import StepStatus
    executor.STAGE_MAP = MOCK_STAGE_MAP

    from agent import ClinicalAgent
    task = {
        "case_id": "test_02",
        "patient_transcript": "raw",
        "chart_notes": "notes",
        "medication_list": [],
        "patient_history": {"age": 45},
    }
    agent = ClinicalAgent()
    agent.run(task)
    skipped = [s for s in agent._last_state.plan.steps if s.status == StepStatus.SKIPPED]
    assert {s.stage for s in skipped} == {"medications", "interactions"}


def test_agent_scratchpad_populated_after_run():
    import executor
    executor.STAGE_MAP = MOCK_STAGE_MAP

    from agent import ClinicalAgent
    task = {
        "case_id": "test_03",
        "patient_transcript": "raw",
        "chart_notes": "notes",
        "medication_list": ["aspirin"],
        "patient_history": {"age": 45},
    }
    agent = ClinicalAgent()
    agent.run(task)
    # 6 stages all ran, 6 scratchpad entries
    assert len(agent._last_state.memory.scratchpad) == 6
    assert all(e.reasoning == "mock reasoning" for e in agent._last_state.memory.scratchpad)
