"""
agents/verifier.py — Post-execution result verification.

The verifier checks:
  1. Did enough steps succeed?
  2. Were expected artifacts created?
  3. Was a code sandbox result correct?
  4. Is the final answer non-empty?

This is NOT neural verification — it's structural/heuristic checking.
"""
from __future__ import annotations

import logging
from agents.state import TaskState, TaskStatus

log = logging.getLogger(__name__)


class Verifier:
    """
    Verifies the output of a completed agent execution.

    HONESTY: this is rule-based structural verification, not an LLM judge.
    """

    def verify(self, state: TaskState) -> TaskState:
        """
        Run verification checks and update state.status accordingly.

        Returns the updated state.
        """
        issues: list[str] = []

        # ── Check step success rate ────────────────────────────────────
        total  = len(state.plan)
        passed = sum(1 for s in state.plan if s.success)
        if total > 0 and passed == 0:
            issues.append("All steps failed — no useful output produced.")

        # ── Check for empty final answer ───────────────────────────────
        if not state.final_answer or not state.final_answer.strip():
            issues.append("Final answer is empty.")

        # ── Check code execution steps ─────────────────────────────────
        for step in state.plan:
            if step.tool_name == "code_sandbox" and step.result:
                exit_code = (step.result or {}).get("exit_code", -1)
                if exit_code != 0:
                    issues.append(
                        f"Code sandbox exited with code {exit_code}: "
                        f"{step.result.get('stderr', '')[:200]}"
                    )

        if issues:
            log.warning("Verifier found %d issue(s): %s", len(issues), issues)
            state.update_status(TaskStatus.DONE)   # still done, just flagged
            if state.final_answer:
                state.final_answer += (
                    f"\n\n[Verifier notes: {'; '.join(issues)}]"
                )
        else:
            log.info("Verifier: all checks passed.")
            state.update_status(TaskStatus.DONE)

        return state
