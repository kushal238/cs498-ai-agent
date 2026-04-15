from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Optional


class StepStatus(Enum):
    PENDING = "pending"
    RUNNING = "running"
    SUCCESS = "success"
    FAILED = "failed"
    SKIPPED = "skipped"


@dataclass
class ScratchEntry:
    stage: str
    reasoning: str
    confidence: str
    timestamp: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )


@dataclass
class LogEntry:
    stage: str
    event: str   # "success" | "retry" | "validation_retry" | "error" | "skipped"
    detail: str
    timestamp: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )


@dataclass
class PlanStep:
    stage: str
    status: StepStatus = StepStatus.PENDING
    attempts: int = 0
    max_attempts: int = 3
    result: Optional[dict] = None
    error: Optional[str] = None
    skipped_reason: Optional[str] = None


@dataclass
class AgentPlan:
    steps: list[PlanStep] = field(default_factory=list)

    def next_step(self) -> Optional[PlanStep]:
        """Return the first PENDING step, or None if all steps are terminal."""
        for step in self.steps:
            if step.status == StepStatus.PENDING:
                return step
        return None

    def is_complete(self) -> bool:
        """True when every step is in a terminal state (SUCCESS, SKIPPED, FAILED)."""
        if not self.steps:
            return False
        terminal = {StepStatus.SUCCESS, StepStatus.SKIPPED, StepStatus.FAILED}
        return all(s.status in terminal for s in self.steps)


@dataclass
class AgentMemory:
    working_memory: dict[str, Any] = field(default_factory=dict)
    scratchpad: list[ScratchEntry] = field(default_factory=list)
    execution_log: list[LogEntry] = field(default_factory=list)


@dataclass
class AgentState:
    task: dict[str, Any]                    # original input — never mutated
    plan: AgentPlan = field(default_factory=AgentPlan)
    memory: AgentMemory = field(default_factory=AgentMemory)

    def get_context(self) -> dict:
        """Merge original task inputs with all accumulated stage outputs.

        Returns a shallow copy — top-level keys from working_memory win on collision.
        Callers must not mutate nested objects in the returned dict.
        """
        return {**self.task, **self.memory.working_memory}
