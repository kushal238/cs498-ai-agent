import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "runner"))

from planner import create_plan
from state import StepStatus

STAGE_ORDER = [
    "transcription", "summarization", "diagnosis",
    "medications", "interactions", "report",
]


def test_full_plan_when_medications_present():
    task = {"medication_list": ["warfarin", "aspirin"]}
    plan = create_plan(task)
    stages = [s.stage for s in plan.steps]
    assert stages == STAGE_ORDER
    assert all(s.status == StepStatus.PENDING for s in plan.steps)


def test_med_stages_skipped_when_no_medications():
    task = {"medication_list": []}
    plan = create_plan(task)
    skipped = {s.stage for s in plan.steps if s.status == StepStatus.SKIPPED}
    assert skipped == {"medications", "interactions"}


def test_med_stages_skipped_when_medication_list_absent():
    task = {}
    plan = create_plan(task)
    skipped = {s.stage for s in plan.steps if s.status == StepStatus.SKIPPED}
    assert skipped == {"medications", "interactions"}


def test_skipped_steps_have_reason():
    task = {"medication_list": []}
    plan = create_plan(task)
    for step in plan.steps:
        if step.status == StepStatus.SKIPPED:
            assert step.skipped_reason is not None
            assert len(step.skipped_reason) > 0


def test_plan_has_six_steps():
    plan = create_plan({"medication_list": ["metformin"]})
    assert len(plan.steps) == 6


def test_plan_order_is_correct():
    plan = create_plan({"medication_list": ["metformin"]})
    assert [s.stage for s in plan.steps] == STAGE_ORDER
