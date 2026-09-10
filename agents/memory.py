"""
agents/memory.py — Short-term and long-term memory for the agent.

SHORT-TERM MEMORY: the conversation window (last N turns).
Used to provide context to the model on each step.

LONG-TERM MEMORY: not implemented in Phase 11 (will use SQLite in
Phase 17 for persistent conversation history).
"""
from __future__ import annotations

from collections import deque
from typing import Optional


class AgentMemory:
    """
    Maintains the agent's conversational context window.

    Args:
        max_turns: Maximum number of (user, assistant) turn pairs to keep.
    """

    def __init__(self, max_turns: int = 10) -> None:
        self._turns: deque[dict] = deque(maxlen=max_turns * 2)

    def add_user(self, content: str) -> None:
        self._turns.append({"role": "user", "content": content})

    def add_assistant(self, content: str) -> None:
        self._turns.append({"role": "assistant", "content": content})

    def add_tool_result(self, tool_name: str, result: str) -> None:
        self._turns.append({
            "role": "tool",
            "tool": tool_name,
            "content": result,
        })

    def get_context_window(self) -> list[dict]:
        return list(self._turns)

    def format_for_prompt(self, system_prompt: str = "") -> str:
        """
        Format the conversation history as a text prompt for the model.

        Returns a simple role-labeled format:
            [System]: ...
            [User]: ...
            [Assistant]: ...
        """
        lines = []
        if system_prompt:
            lines.append(f"[System]: {system_prompt}")
        for turn in self._turns:
            role = turn["role"].capitalize()
            if turn["role"] == "tool":
                lines.append(f"[Tool: {turn['tool']}]: {turn['content']}")
            else:
                lines.append(f"[{role}]: {turn['content']}")
        lines.append("[Assistant]:")
        return "\n".join(lines)

    def clear(self) -> None:
        self._turns.clear()

    def __len__(self) -> int:
        return len(self._turns)
