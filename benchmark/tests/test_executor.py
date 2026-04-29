import sys
from pathlib import Path
from unittest.mock import patch
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "runner"))

import executor
from state import AgentState, PlanStep, StepStatus


def _make_state(task=None):
    return AgentState(task=task or {"case_id": "test"})


def _make_pending_step(stage="transcription"):
    return PlanStep(stage=stage)


def _good_stage_fn(context):
    return {
        "reasoning": "looked good",
        "confidence": "high",
        "output": {"transcription_cleaned": "clean text"},
    }


def test_execute_success_writes_to_working_memory():
    executor.STAGE_MAP = {"transcription": _good_stage_fn}
    step = _make_pending_step("transcription")
    state = _make_state()
    executor.execute(step, state)
    assert state.memory.working_memory.get("transcription_cleaned") == "clean text"


def test_execute_success_writes_to_scratchpad():
    executor.STAGE_MAP = {"transcription": _good_stage_fn}
    step = _make_pending_step("transcription")
    state = _make_state()
    executor.execute(step, state)
    assert len(state.memory.scratchpad) == 1
    assert state.memory.scratchpad[0].reasoning == "looked good"
    assert state.memory.scratchpad[0].confidence == "high"


def test_execute_success_marks_step_success():
    executor.STAGE_MAP = {"transcription": _good_stage_fn}
    step = _make_pending_step("transcription")
    state = _make_state()
    executor.execute(step, state)
    assert step.status == StepStatus.SUCCESS


def test_execute_retries_on_exception():
    call_count = {"n": 0}

    def flaky_then_ok(context):
        call_count["n"] += 1
        if call_count["n"] < 2:
            raise RuntimeError("temporary failure")
        return {
            "reasoning": "recovered",
            "confidence": "medium",
            "output": {"transcription_cleaned": "recovered text"},
        }

    executor.STAGE_MAP = {"transcription": flaky_then_ok}
    step = _make_pending_step("transcription")
    state = _make_state()

    with patch("executor.time.sleep"):
        executor.execute(step, state)

    assert step.status == StepStatus.SUCCESS
    assert step.attempts == 2


def test_execute_marks_failed_after_max_attempts():
    def always_fails(context):
        raise RuntimeError("always broken")

    executor.STAGE_MAP = {"transcription": always_fails}
    step = _make_pending_step("transcription")
    state = _make_state()

    with patch("executor.time.sleep"):
        executor.execute(step, state)

    assert step.status == StepStatus.FAILED
    assert step.attempts == 3


def test_execute_applies_fallback_on_failure():
    def always_fails(context):
        raise RuntimeError("broken")

    executor.STAGE_MAP = {"diagnosis": always_fails}
    step = _make_pending_step("diagnosis")
    state = _make_state()

    with patch("executor.time.sleep"):
        executor.execute(step, state)

    assert state.memory.working_memory.get("differential_diagnosis") == []


def test_execute_skipped_step_returns_empty():
    step = PlanStep(stage="medications", status=StepStatus.SKIPPED,
                    skipped_reason="no meds")
    state = _make_state()
    result = executor.execute(step, state)
    assert result == {}
    assert step.status == StepStatus.SKIPPED


def test_execute_writes_execution_log():
    executor.STAGE_MAP = {"transcription": _good_stage_fn}
    step = _make_pending_step("transcription")
    state = _make_state()
    executor.execute(step, state)
    assert len(state.memory.execution_log) == 1
    assert state.memory.execution_log[0].event == "success"


def test_execute_injects_scratchpad_for_non_transcription_stage():
    """Stages after transcription should receive scratchpad_summary in context."""
    from state import ScratchEntry

    captured = {}

    def capture_fn(context):
        captured.update(context)
        return {
            "reasoning": "ok",
            "confidence": "high",
            "output": {"clinical_summary": "test summary"},
        }

    executor.STAGE_MAP = {"summarization": capture_fn}
    step = _make_pending_step("summarization")
    state = _make_state()
    state.memory.scratchpad.append(
        ScratchEntry(stage="transcription", reasoning="cleaned dialogue", confidence="high")
    )
    executor.execute(step, state)

    assert "scratchpad_summary" in captured
    assert "transcription" in captured["scratchpad_summary"]
    assert "cleaned dialogue" in captured["scratchpad_summary"]


def test_execute_does_not_inject_scratchpad_for_transcription_stage():
    """The transcription stage runs first and should not receive a scratchpad."""
    captured = {}

    def capture_fn(context):
        captured.update(context)
        return {
            "reasoning": "ok",
            "confidence": "high",
            "output": {"transcription_cleaned": "clean text"},
        }

    executor.STAGE_MAP = {"transcription": capture_fn}
    step = _make_pending_step("transcription")
    state = _make_state()
    executor.execute(step, state)

    assert "scratchpad_summary" not in captured


def test_execute_does_not_inject_scratchpad_for_transcription_stage_even_when_populated():
    """Transcription must not receive scratchpad even if prior entries exist."""
    from state import ScratchEntry
    captured = {}

    def capture_fn(context):
        captured.update(context)
        return {
            "reasoning": "ok",
            "confidence": "high",
            "output": {"transcription_cleaned": "clean text"},
        }

    executor.STAGE_MAP = {"transcription": capture_fn}
    step = _make_pending_step("transcription")
    state = _make_state()
    state.memory.scratchpad.append(
        ScratchEntry(stage="some_prior", reasoning="some reasoning", confidence="high")
    )
    executor.execute(step, state)

    assert "scratchpad_summary" not in captured


def test_execute_does_not_inject_empty_scratchpad():
    """If the scratchpad is empty, scratchpad_summary should not be injected."""
    captured = {}

    def capture_fn(context):
        captured.update(context)
        return {
            "reasoning": "ok",
            "confidence": "high",
            "output": {"clinical_summary": "summary"},
        }

    executor.STAGE_MAP = {"summarization": capture_fn}
    step = _make_pending_step("summarization")
    state = _make_state()
    # scratchpad is empty — no entries
    executor.execute(step, state)

    assert "scratchpad_summary" not in captured


def test_run_tool_decision_returns_empty_list_when_llm_requests_no_tools():
    """If LLM decides no tools are needed, returns an empty list."""
    from unittest.mock import MagicMock

    class _NoTools:
        tool_calls = []
        reasoning = "no tools needed"

    with patch("llm_client.chat_structured", return_value=_NoTools()):
        result = executor._run_tool_decision(
            stage="summarization",
            context={"clinical_summary": "test"},
            tool_manifest={"pubmed_search": "Search PubMed for a condition"},
        )
    assert result == []


def test_run_tool_decision_returns_requested_tool_calls():
    """If LLM requests tools, they are returned as a list of dicts."""
    from unittest.mock import MagicMock

    class _ToolCall:
        name = "pubmed_search"
        args = {"query": "unstable angina"}

    class _WithTools:
        tool_calls = [_ToolCall()]
        reasoning = "need citation"

    with patch("llm_client.chat_structured", return_value=_WithTools()):
        result = executor._run_tool_decision(
            stage="diagnosis",
            context={"clinical_summary": "test"},
            tool_manifest={"pubmed_search": "Search PubMed"},
        )
    assert len(result) == 1
    assert result[0]["name"] == "pubmed_search"
    assert result[0]["args"] == {"query": "unstable angina"}


def test_run_tool_decision_returns_empty_list_on_llm_error():
    """If the LLM call fails, returns empty list (no-op fallback)."""
    with patch("llm_client.chat_structured", side_effect=Exception("API error")):
        result = executor._run_tool_decision(
            stage="diagnosis",
            context={},
            tool_manifest={"pubmed_search": "Search PubMed"},
        )
    assert result == []
