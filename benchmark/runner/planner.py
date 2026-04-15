"""Builds an AgentPlan from a task dict, conditionally skipping medication stages."""
from __future__ import annotations

from state import AgentPlan, PlanStep, StepStatus

STAGE_ORDER = [
    "transcription",
    "summarization",
    "diagnosis",
    "medications",
    "interactions",
    "report",
]

# Stages that require at least one medication in the input
MED_DEPENDENT = {"medications", "interactions"}


def create_plan(task: dict) -> AgentPlan:
    """Build an AgentPlan from a task dict.

    Medication normalization (stage 4) and drug interaction checking (stage 5)
    are skipped when the input medication_list is empty or absent.
    """
    has_medications = bool(task.get("medication_list"))
    steps = []
    for stage in STAGE_ORDER:
        if stage in MED_DEPENDENT and not has_medications:
            steps.append(PlanStep(
                stage=stage,
                status=StepStatus.SKIPPED,
                skipped_reason="no medications in input",
            ))
        else:
            steps.append(PlanStep(stage=stage))
    return AgentPlan(steps=steps)
