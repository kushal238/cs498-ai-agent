"""Executes a single PlanStep with retry logic, scratchpad writing, and fallback."""
from __future__ import annotations

import time
import traceback
from typing import Callable

import jsonschema

from state import AgentState, PlanStep, StepStatus, ScratchEntry, LogEntry
from validator import validate

# Populated by agent.py at startup: maps stage name → stage run() function
STAGE_MAP: dict[str, Callable[[dict], dict]] = {}

BACKOFF_SECONDS = [1, 2, 4]

# Fallback outputs applied when a stage exhausts all retries
_FALLBACKS: dict[str, dict] = {
    "summarization": {"clinical_summary": ""},
    "diagnosis": {"differential_diagnosis": []},
    "medications": {"normalized_medications": []},
    "interactions": {"drug_interactions": []},
    "report": {
        "final_report": {
            "subjective": "", "objective": "",
            "assessment": "", "plan": "",
        }
    },
}


def execute(step: PlanStep, state: AgentState) -> dict:
    """Run a single PlanStep, handling retries and memory updates.

    On success: writes output to working_memory, reasoning to scratchpad,
                and a success entry to execution_log.
    On exhausted retries: marks step FAILED, applies fallback output, logs error.

    Returns the stage output dict, or {} if the step was skipped or failed.
    """
    if step.status == StepStatus.SKIPPED:
        return {}

    step.status = StepStatus.RUNNING
    stage_fn = STAGE_MAP[step.stage]
    last_error = ""

    for attempt in range(step.max_attempts):
        step.attempts += 1
        try:
            context = state.get_context()
            if step.stage == "report":
                context["scratchpad_summary"] = _format_scratchpad(state.memory.scratchpad)
            if "_validation_error" in state.task:
                context["_validation_error"] = state.task.pop("_validation_error")

            result = stage_fn(context)
            validate(step.stage, result["output"])

            state.memory.working_memory.update(result["output"])
            state.memory.scratchpad.append(ScratchEntry(
                stage=step.stage,
                reasoning=result.get("reasoning", ""),
                confidence=result.get("confidence", "unknown"),
            ))
            state.memory.execution_log.append(LogEntry(
                stage=step.stage,
                event="success",
                detail=f"completed on attempt {step.attempts}",
            ))
            step.status = StepStatus.SUCCESS
            step.result = result["output"]
            return result["output"]

        except jsonschema.ValidationError as exc:
            last_error = exc.message
            state.memory.execution_log.append(LogEntry(
                stage=step.stage,
                event="validation_retry",
                detail=last_error,
            ))
            state.task["_validation_error"] = last_error
            if attempt >= 1:   # only one validation retry
                break

        except Exception as exc:
            last_error = str(exc)
            is_last = attempt >= step.max_attempts - 1
            print(f"[Executor] {step.stage} attempt {step.attempts} failed: {last_error}", file=__import__('sys').stderr)
            state.memory.execution_log.append(LogEntry(
                stage=step.stage,
                event="error" if is_last else "retry",
                detail=last_error,
            ))
            if not is_last:
                time.sleep(BACKOFF_SECONDS[min(attempt, len(BACKOFF_SECONDS) - 1)])

    # All attempts exhausted
    step.status = StepStatus.FAILED
    step.error = last_error
    _apply_fallback(step.stage, state)
    return {}


def _format_scratchpad(scratchpad: list[ScratchEntry]) -> str:
    return "\n".join(
        f"[{e.stage}] {e.reasoning} (confidence: {e.confidence})"
        for e in scratchpad
    )


def _apply_fallback(stage: str, state: AgentState) -> None:
    """Write a safe empty fallback into working_memory for the failed stage."""
    if stage == "transcription":
        fallback = {"transcription_cleaned": state.task.get("patient_transcript", "")}
    else:
        fallback = _FALLBACKS.get(stage, {})
    state.memory.working_memory.update(fallback)
