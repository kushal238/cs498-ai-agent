import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "runner"))

from state import (
    AgentState, AgentMemory, AgentPlan, PlanStep,
    ScratchEntry, LogEntry, StepStatus,
)


def test_plan_step_defaults():
    step = PlanStep(stage="transcription")
    assert step.status == StepStatus.PENDING
    assert step.attempts == 0
    assert step.max_attempts == 3
    assert step.result is None
    assert step.error is None


def test_plan_next_step_returns_first_pending():
    plan = AgentPlan(steps=[
        PlanStep(stage="transcription", status=StepStatus.SUCCESS),
        PlanStep(stage="summarization"),
        PlanStep(stage="diagnosis"),
    ])
    assert plan.next_step().stage == "summarization"


def test_plan_next_step_returns_none_when_all_done():
    plan = AgentPlan(steps=[
        PlanStep(stage="transcription", status=StepStatus.SUCCESS),
        PlanStep(stage="summarization", status=StepStatus.SKIPPED),
    ])
    assert plan.next_step() is None


def test_plan_is_complete_false_when_pending():
    plan = AgentPlan(steps=[
        PlanStep(stage="transcription", status=StepStatus.SUCCESS),
        PlanStep(stage="summarization"),
    ])
    assert not plan.is_complete()


def test_plan_is_complete_true_when_all_terminal():
    plan = AgentPlan(steps=[
        PlanStep(stage="transcription", status=StepStatus.SUCCESS),
        PlanStep(stage="summarization", status=StepStatus.SKIPPED),
        PlanStep(stage="diagnosis", status=StepStatus.FAILED),
    ])
    assert plan.is_complete()


def test_agent_state_get_context_merges_task_and_working_memory():
    state = AgentState(task={"case_id": "c1", "patient_transcript": "raw"})
    state.memory.working_memory["transcription_cleaned"] = "cleaned"
    ctx = state.get_context()
    assert ctx["case_id"] == "c1"
    assert ctx["patient_transcript"] == "raw"
    assert ctx["transcription_cleaned"] == "cleaned"


def test_agent_state_task_not_mutated_by_get_context():
    state = AgentState(task={"case_id": "c1"})
    state.memory.working_memory["extra"] = "value"
    _ = state.get_context()
    assert "extra" not in state.task


def test_scratch_entry_has_timestamp():
    entry = ScratchEntry(stage="transcription", reasoning="looks clean", confidence="high")
    assert entry.timestamp is not None
    assert entry.stage == "transcription"


def test_log_entry_has_timestamp():
    entry = LogEntry(stage="transcription", event="success", detail="done")
    assert entry.timestamp is not None
