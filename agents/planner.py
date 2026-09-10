"""
agents/planner.py — Multi-step task planner.

WHAT THE PLANNER DOES
----------------------
Given a user request and a routing decision (which model, which tools),
the planner generates a structured list of steps to execute.

HOW PLANNING WORKS (in this project)
--------------------------------------
The planner uses a TEMPLATE-BASED approach for the SIH demo scenarios,
combined with model-generated steps for general tasks.

HONESTY DECLARATION: the planner does NOT use complex reasoning.
For well-defined SIH demo scenarios, the steps are deterministic
templates. For general tasks, the planner asks the language model to
produce a numbered step list and parses it — simple and transparent.
"""
from __future__ import annotations

import logging
import re
from typing import Optional

from agents.state import AgentStep, TaskState
from models.adapters.base import ModelProvider, GenerationConfig
from router.task_classifier import TaskCategory
from tools.registry import ToolRegistry

log = logging.getLogger(__name__)

# ── Template plans for deterministic SIH demo scenarios ───────────────────────

_DEMO_PLANS: dict[TaskCategory, list[dict]] = {
    TaskCategory.OCR: [
        {"description": "Extract text from the uploaded document using local OCR.",
         "tool": "ocr"},
        {"description": "Analyze the extracted text and identify key findings.",
         "tool": None},
        {"description": "Generate a structured approval note as a Word document.",
         "tool": "word_write"},
    ],
    TaskCategory.CODING: [
        {"description": "Understand the coding requirement and write Python code.",
         "tool": None},
        {"description": "Execute the code in the local sandbox and capture output.",
         "tool": "code_sandbox"},
        {"description": "Verify output correctness and report results.",
         "tool": None},
    ],
    TaskCategory.RETRIEVAL: [
        {"description": "Search the local knowledge base for relevant documents.",
         "tool": "rag_search"},
        {"description": "Synthesize retrieved content into a cited answer.",
         "tool": None},
    ],
    TaskCategory.CALCULATION: [
        {"description": "Parse the calculation request and evaluate the expression.",
         "tool": "calculator"},
        {"description": "Format the result clearly.",
         "tool": None},
    ],
    TaskCategory.SPREADSHEET: [
        {"description": "Process the data and perform calculations.",
         "tool": "calculator"},
        {"description": "Write results to an Excel spreadsheet.",
         "tool": "spreadsheet_write"},
    ],
}


class Planner:
    """
    Generates a step-by-step plan for an agent task.

    For recognized demo scenarios, uses deterministic template plans.
    For other tasks, uses the selected model to generate a step list.
    """

    def __init__(
        self,
        tool_registry: ToolRegistry,
    ) -> None:
        self.tool_registry = tool_registry

    def plan(
        self,
        state: TaskState,
        model: Optional[ModelProvider] = None,
    ) -> list[AgentStep]:
        """
        Build the execution plan for the given task state.

        Args:
            state:  TaskState with user_input and category set.
            model:  The selected ModelProvider (used for open-ended planning).

        Returns:
            List of AgentStep objects with description and tool assignments.
        """
        category = TaskCategory(state.category) if state.category else TaskCategory.UNKNOWN

        # ── 1. Use deterministic template if available ─────────────────
        template = _DEMO_PLANS.get(category)
        if template is not None:
            log.info("Planner: using template plan for category '%s'", category.value)
            return [
                AgentStep(
                    step_number=i + 1,
                    description=step["description"],
                    tool_name=step.get("tool"),
                )
                for i, step in enumerate(template)
            ]

        # ── 2. Ask the model to generate steps (general tasks) ─────────
        if model is not None and model.is_available():
            prompt = (
                f"You are an AI assistant. Break down this task into numbered steps:\n"
                f"Task: {state.user_input}\n"
                f"Available tools: {', '.join(t['name'] for t in self.tool_registry.list_tools())}\n"
                f"List the steps:"
            )
            try:
                plan_text = model.generate(
                    prompt,
                    GenerationConfig(max_new_tokens=256, temperature=0.3, top_k=20)
                )
                steps = self._parse_numbered_list(plan_text, state.user_input)
                if steps:
                    log.info("Planner: model generated %d steps.", len(steps))
                    return steps
            except Exception as exc:
                log.warning("Model-based planning failed: %s — using fallback.", exc)

        # ── 3. Fallback: single-step direct answer ─────────────────────
        log.info("Planner: falling back to single-step plan.")
        return [AgentStep(step_number=1, description=f"Answer: {state.user_input}")]

    @staticmethod
    def _parse_numbered_list(text: str, fallback_description: str) -> list[AgentStep]:
        """Parse a numbered list from model output into AgentStep objects."""
        steps = []
        for match in re.finditer(r"(\d+)[.)]\s*(.+)", text):
            num  = int(match.group(1))
            desc = match.group(2).strip()
            if desc:
                steps.append(AgentStep(step_number=num, description=desc))
        return steps
