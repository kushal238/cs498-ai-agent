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
