"""Executes a single PlanStep with retry logic, scratchpad writing, and fallback."""
from __future__ import annotations

import time
import traceback
from typing import Callable

import jsonschema
from pydantic import BaseModel, Field

import llm_client
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


class _ToolCallRequest(BaseModel):
    name: str = Field(description="Tool name from the manifest")
    args: dict = Field(description="Arguments to pass to the tool", default_factory=dict)


class _ToolDecision(BaseModel):
    reasoning: str = Field(description="Why these tools are or are not needed")
    tool_calls: list[_ToolCallRequest] = Field(
        description="Tool calls to make before running this stage. Empty list means no calls needed.",
        default_factory=list,
    )


# Stage-level tool manifests — empty dict means no tools available for that stage
TOOL_MANIFESTS: dict[str, dict[str, str]] = {
    "transcription":  {},
    "summarization":  {},
    "diagnosis":      {"pubmed_search": "Search PubMed for supporting citations for a condition name. Args: {query: str}"},
    "medications":    {"rxnorm_lookup": "Look up the RxNorm CUI and ingredient name for a drug. Args: {drug: str}"},
    "interactions":   {"check_interactions": "Check drug interactions for a list of RxCUI IDs. Args: {rxcui_ids: list[str]}"},
    "report":         {},
}


def _run_tool_decision(
    stage: str,
    context: dict,
    tool_manifest: dict[str, str],
) -> list[dict]:
    """Ask the LLM whether it needs any tool calls before running a stage.

    Args:
        stage:         Stage name, used for logging.
        context:       Current agent context dict.
        tool_manifest: Map of tool_name → one-line description available to this stage.

    Returns:
        List of {"name": str, "args": dict} tool call requests.
        Returns [] if no tools are needed or if the LLM call fails.
    """
    if not tool_manifest:
        return []

    manifest_text = "\n".join(f"- {name}: {desc}" for name, desc in tool_manifest.items())
    context_summary = str(context)[:800]  # keep prompt short

    messages = [
        {
            "role": "system",
            "content": (
                "You are a tool-selection assistant. Decide whether any tool calls "
                "are needed before running the next stage. Only request tools if they "
                "will meaningfully improve the output. Return an empty tool_calls list "
                "if no tools are needed."
            ),
        },
        {
            "role": "user",
            "content": (
                f"Stage: {stage}\n\n"
                f"Available tools:\n{manifest_text}\n\n"
                f"Current context (truncated):\n{context_summary}"
            ),
        },
    ]
    try:
        decision = llm_client.chat_structured(messages, _ToolDecision, model="gpt-4o-mini")
        return [{"name": tc.name, "args": tc.args} for tc in decision.tool_calls]
    except Exception as exc:
        import sys
        print(f"[Executor] _run_tool_decision({stage}) failed: {exc}", file=sys.stderr)
        return []


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
            if step.stage != "transcription" and state.memory.scratchpad:
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
