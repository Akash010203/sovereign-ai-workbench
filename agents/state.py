"""
agents/state.py — Task state machine for the agent lifecycle.

AGENT LIFECYCLE STATES
-----------------------
    PENDING   → task received, not yet started
    PLANNING  → planner is building the step list
    RUNNING   → executor is working through steps
    VERIFYING → verifier is checking results
    DONE      → task completed successfully
    FAILED    → task failed after exhausting retries
    CANCELLED → user or system cancelled the task
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Optional


class TaskStatus(str, Enum):
    PENDING   = "pending"
    PLANNING  = "planning"
    RUNNING   = "running"
    VERIFYING = "verifying"
    DONE      = "done"
    FAILED    = "failed"
    CANCELLED = "cancelled"


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


@dataclass
class AgentStep:
    """One step in the agent's plan."""
    step_number:  int
    description:  str
    tool_name:    Optional[str] = None
    tool_args:    dict = field(default_factory=dict)
    result:       Any = None
    success:      Optional[bool] = None
    error:        Optional[str] = None
    timestamp:    str = field(default_factory=_utc_now)


@dataclass
class TaskState:
    """Complete state of one agent task execution."""
    task_id:       str
    user_input:    str
    status:        TaskStatus = TaskStatus.PENDING
    model_name:    str = ""
    category:      str = ""
    plan:          list[AgentStep] = field(default_factory=list)
    current_step:  int = 0
    memory:        list[dict] = field(default_factory=list)   # conversation history
    artifacts:     list[dict] = field(default_factory=list)   # generated files
    final_answer:  Optional[str] = None
    error:         Optional[str] = None
    created_at:    str = field(default_factory=_utc_now)
    updated_at:    str = field(default_factory=_utc_now)

    def update_status(self, status: TaskStatus) -> None:
        self.status = status
        self.updated_at = _utc_now()

    def add_artifact(self, name: str, path: str, artifact_type: str) -> None:
        self.artifacts.append({
            "name": name, "path": path,
            "type": artifact_type,
            "created_at": _utc_now(),
        })

    def to_dict(self) -> dict:
        return {
            "task_id":      self.task_id,
            "user_input":   self.user_input,
            "status":       self.status.value,
            "model_name":   self.model_name,
            "category":     self.category,
            "current_step": self.current_step,
            "steps_total":  len(self.plan),
            "artifacts":    self.artifacts,
            "final_answer": self.final_answer,
            "error":        self.error,
            "created_at":   self.created_at,
            "updated_at":   self.updated_at,
        }
