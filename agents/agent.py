"""
agents/agent.py — Top-level agent orchestrator.

Combines: Router → Planner → Executor → Verifier into one callable.

AGENT LIFECYCLE
---------------
    user_input
        │
        ▼ TaskRouter.route()
    RoutingDecision (category, model, tools)
        │
        ▼ Planner.plan()
    list[AgentStep]
        │
        ▼ Executor.execute_plan()
    TaskState (with step results)
        │
        ▼ build final_answer from step results
        │
        ▼ Verifier.verify()
    TaskState (status=DONE or FAILED)
"""
from __future__ import annotations

import logging
import re
import uuid
from typing import Optional

from agents.executor import Executor
from agents.memory import AgentMemory
from agents.planner import Planner
from agents.state import TaskState, TaskStatus
from agents.verifier import Verifier
from models.registry import ModelRegistry
from router.router import TaskRouter
from tools.registry import ToolRegistry

log = logging.getLogger(__name__)


class Agent:
    """
    SovereignAI top-level agent — receives a user request and
    returns a completed TaskState with results and artifacts.
    """

    def __init__(
        self,
        model_registry: ModelRegistry,
        tool_registry: ToolRegistry,
    ) -> None:
        self.model_registry = model_registry
        self.tool_registry  = tool_registry
        self.router   = TaskRouter(model_registry)
        self.memory   = AgentMemory(max_turns=10)
        self.verifier = Verifier()

    def run(self, user_input: str) -> TaskState:
        """
        Process a user request end-to-end.

        Args:
            user_input: Raw text from the user.

        Returns:
            Completed TaskState with final_answer and artifacts.
        """
        task_id = str(uuid.uuid4())[:8]
        log.info("Agent run: task_id=%s  input='%s'", task_id, user_input[:80])

        # ── 1. Route ───────────────────────────────────────────────────
        decision = self.router.route(user_input)

        state = TaskState(
            task_id=task_id,
            user_input=user_input,
            model_name=decision.model_name,
            category=decision.category.value,
        )
        state.update_status(TaskStatus.PLANNING)

        # ── 2. Plan ────────────────────────────────────────────────────
        planner = Planner(self.tool_registry)
        state.plan = planner.plan(state, model=decision.model)
        log.info("Agent: %d steps planned.", len(state.plan))

        # Set tool args from decision for steps that need them
        for step in state.plan:
            if step.tool_name and not step.tool_args:
                # Default: pass user_input as the primary argument
                step.tool_args = self._default_tool_args(step.tool_name, user_input)

        # ── 3. Execute ─────────────────────────────────────────────────
        executor = Executor(self.tool_registry, model=decision.model)
        state = executor.execute_plan(state)

        # ── 4. Build final answer ──────────────────────────────────────
        state.final_answer = self._compile_answer(state, decision.model)

        # ── 5. Verify ─────────────────────────────────────────────────
        state.update_status(TaskStatus.VERIFYING)
        state = self.verifier.verify(state)

        # ── 6. Store in memory ─────────────────────────────────────────
        self.memory.add_user(user_input)
        self.memory.add_assistant(state.final_answer or "")

        log.info("Agent done: status=%s  artifacts=%d",
                 state.status.value, len(state.artifacts))
        return state

    def _default_tool_args(self, tool_name: str, user_input: str) -> dict:
        """Provide sensible default args for each tool based on user input."""
        defaults: dict[str, dict] = {
            "calculator": {"expression": self._calculation_expression(user_input)},
            "code_sandbox": {"code": "# Generated code will be placed here\nprint('Hello from sandbox')"},
            "rag_search": {"query": user_input},
        }
        return defaults.get(tool_name, {})

    @staticmethod
    def _calculation_expression(user_input: str) -> str:
        """Extract a calculator-safe expression from a natural-language request."""
        expression = user_input.lower().strip()
        expression = re.sub(r"^(?:please\s+)?(?:calculate|compute|evaluate|what is)\s+", "", expression)
        replacements = {
            "multiplied by": "*", "times": "*", "x": "*",
            "divided by": "/", "plus": "+", "minus": "-",
        }
        for phrase, symbol in replacements.items():
            expression = re.sub(rf"\b{re.escape(phrase)}\b", symbol, expression)
        return expression.rstrip("?. ")

    def _compile_answer(
        self, state: TaskState, model: Optional[object]
    ) -> str:
        """Compile step results into a final answer string."""
        # Collect meaningful step results
        parts = []
        for step in state.plan:
            if step.result and step.success:
                result_str = str(step.result)
                if len(result_str) > 600:
                    result_str = result_str[:600] + "..."
                parts.append(f"**{step.description}**\n{result_str}")

        if parts:
            return "\n\n".join(parts)
        if model and hasattr(model, "is_available") and model.is_available():
            try:
                from models.adapters.base import GenerationConfig
                prompt = f"{self.memory.format_for_prompt()}\nUser: {state.user_input}\nAssistant:"
                return model.generate(prompt, GenerationConfig(max_new_tokens=256))
            except Exception:
                pass
        return f"Unable to complete task: {state.error or 'no model or tool produced output.'}"
