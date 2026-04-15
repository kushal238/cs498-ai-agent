"""ClinicalAgent: main run loop wiring plan → executor → memory."""
from __future__ import annotations

import sys

import executor
import planner
import stage_diagnosis
import stage_interactions
import stage_medications
import stage_report
import stage_summarization
import stage_transcription
from state import AgentState

# Default stage map wired at import time.
# Tests may override executor.STAGE_MAP before importing this module;
# we only populate it here when it is still empty so mocks are preserved.
_DEFAULT_STAGE_MAP = {
    "transcription": stage_transcription.run,
    "summarization": stage_summarization.run,
    "diagnosis": stage_diagnosis.run,
    "medications": stage_medications.run,
    "interactions": stage_interactions.run,
    "report": stage_report.run,
}
if not executor.STAGE_MAP:
    executor.STAGE_MAP = _DEFAULT_STAGE_MAP


class ClinicalAgent:
    """Agentic clinical workflow processor.

    Builds a conditional plan from the task, executes each step with
    retry logic, and accumulates outputs in layered memory. The final
    output matches the ground_truth_schema field set.
    """

    def __init__(self) -> None:
        self._last_state: AgentState | None = None

    def run(self, task: dict) -> dict:
        """Run the full clinical workflow for one task dict.

        Args:
            task: Validated input dict from input_schema.json.

        Returns:
            Output dict matching ground_truth_schema.json.
        """
        state = AgentState(task=dict(task))
        state.plan = planner.create_plan(task)
        self._last_state = state

        while not state.plan.is_complete():
            step = state.plan.next_step()
            if step is None:
                break
            print(f"[Agent] Stage: {step.stage}", file=sys.stderr)
            executor.execute(step, state)

        return {
            "case_id": task.get("case_id"),
            "transcription_cleaned": state.memory.working_memory.get("transcription_cleaned"),
            "clinical_summary": state.memory.working_memory.get("clinical_summary"),
            "differential_diagnosis": state.memory.working_memory.get("differential_diagnosis", []),
            "normalized_medications": state.memory.working_memory.get("normalized_medications", []),
            "drug_interactions": state.memory.working_memory.get("drug_interactions", []),
            "final_report": state.memory.working_memory.get("final_report", {}),
        }
