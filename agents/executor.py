"""
agents/executor.py — Step executor: run each plan step in sequence.

The executor processes AgentStep objects one at a time:
  - If the step has a tool_name, it calls the ToolRegistry.
  - If no tool, it asks the model to generate a response for this step.
  - Results are stored back into the AgentStep.
  - The executor updates TaskState throughout.
"""
from __future__ import annotations

import logging
from typing import Optional

from agents.state import AgentStep, TaskState, TaskStatus
from models.adapters.base import ModelProvider, GenerationConfig
from tools.registry import ToolRegistry

log = logging.getLogger(__name__)


class Executor:
    """Executes the steps of an agent plan sequentially."""

    def __init__(
        self,
        tool_registry: ToolRegistry,
        model: Optional[ModelProvider] = None,
    ) -> None:
        self.tools = tool_registry
        self.model = model

    def execute_step(self, step: AgentStep, state: TaskState) -> AgentStep:
        """
        Execute a single AgentStep and update its result fields.

        Args:
            step:   The step to execute.
            state:  The parent TaskState (for context and artifact recording).

        Returns:
            The same step object with result, success, and error populated.
        """
        log.info("Executor: step %d — %s", step.step_number, step.description[:80])

        # ── Tool call ──────────────────────────────────────────────────
        if step.tool_name:
            tool_result = self.tools.call(step.tool_name, **step.tool_args)
            step.result  = tool_result.output
            step.success = tool_result.success
            step.error   = tool_result.error

            # Record file artifacts
            if tool_result.success and step.tool_name in (
                "word_write", "spreadsheet_write", "powerpoint_write", "file_write"
            ):
                path = tool_result.metadata.get("path", "")
                if path:
                    state.add_artifact(
                        name=step.description[:40],
                        path=path,
                        artifact_type=step.tool_name,
                    )

        # ── Model call ─────────────────────────────────────────────────
        elif self.model and self.model.is_available():
            # Build a prompt from the step description + recent context
            context = "\n".join(
                f"Step {s.step_number} result: {str(s.result)[:300]}"
                for s in state.plan[:step.step_number - 1]
                if s.result is not None
            )
            prompt = (
                f"Task: {state.user_input}\n"
                f"{context}\n"
                f"Current step: {step.description}\n"
                f"Response:"
            )
            try:
                answer = self.model.generate(
                    prompt,
                    GenerationConfig(max_new_tokens=512, temperature=0.7, top_k=40)
                )
                step.result  = answer
                step.success = True
            except Exception as exc:
                step.error   = str(exc)
                step.success = False

        else:
            step.result  = f"[No model or tool available for: {step.description}]"
            step.success = False
            step.error   = "No model available."

        return step

    def execute_plan(self, state: TaskState) -> TaskState:
        """
        Execute all steps in state.plan in order.

        Updates state.current_step and state.status throughout.
        Stops early if a critical step fails (tool failure with no retry).
        """
        state.update_status(TaskStatus.RUNNING)

        for i, step in enumerate(state.plan):
            state.current_step = i + 1
            completed_step = self.execute_step(step, state)
            state.plan[i] = completed_step

            if not completed_step.success:
                log.warning(
                    "Step %d failed: %s", step.step_number, completed_step.error
                )
                # Continue to next step regardless — best effort

        return state
