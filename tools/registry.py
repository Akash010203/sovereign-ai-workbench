"""
tools/registry.py — Tool registry with audit logging.

Every tool call in the system goes through this registry.
This ensures:
  - No ad-hoc tool calls bypass logging/auditing
  - Each tool is validated before execution
  - Security policies can be enforced at this layer
  - Tool results are structured and consistent
"""
from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Optional

log = logging.getLogger(__name__)


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


@dataclass
class ToolResult:
    """Standardized output from any tool call."""
    tool_name:  str
    success:    bool
    output:     Any
    error:      Optional[str] = None
    metadata:   dict = field(default_factory=dict)
    timestamp:  str = field(default_factory=_utc_now)


class BaseTool(ABC):
    """Abstract base class for all SovereignAI tools."""

    @property
    @abstractmethod
    def name(self) -> str:
        """Unique tool identifier, e.g. 'calculator'."""

    @property
    @abstractmethod
    def description(self) -> str:
        """One-sentence description for the agent planner."""

    @abstractmethod
    def run(self, **kwargs) -> ToolResult:
        """Execute the tool with the given keyword arguments."""

    def __repr__(self) -> str:
        return f"Tool({self.name})"


class ToolRegistry:
    """
    Registry of all available tools.

    Every tool call is logged here — providing an audit trail of what
    the agent did, which is a security requirement for the air-gapped
    confidential-data environment.
    """

    def __init__(self) -> None:
        self._tools: dict[str, BaseTool] = {}
        self._audit_log: list[dict] = []

    def register(self, tool: BaseTool) -> None:
        self._tools[tool.name] = tool
        log.info("Tool registered: %s", tool.name)

    def get(self, name: str) -> Optional[BaseTool]:
        return self._tools.get(name)

    def call(self, tool_name: str, **kwargs) -> ToolResult:
        """
        Look up and execute a tool, logging the call and result.

        Args:
            tool_name: The tool's name string.
            **kwargs:  Tool-specific arguments.

        Returns:
            ToolResult with output or error details.
        """
        tool = self._tools.get(tool_name)
        if tool is None:
            err = f"Tool '{tool_name}' not found in registry."
            log.error(err)
            result = ToolResult(tool_name=tool_name, success=False, output=None, error=err)
        else:
            try:
                result = tool.run(**kwargs)
            except Exception as exc:
                log.error("Tool '%s' raised exception: %s", tool_name, exc)
                result = ToolResult(
                    tool_name=tool_name, success=False, output=None,
                    error=str(exc)
                )

        # Audit log entry
        entry = {
            "timestamp":  result.timestamp,
            "tool":       tool_name,
            "args":       {k: str(v)[:200] for k, v in kwargs.items()},
            "success":    result.success,
            "error":      result.error,
        }
        self._audit_log.append(entry)
        log.info("Tool call: %s  success=%s", tool_name, result.success)
        return result

    def get_audit_log(self) -> list[dict]:
        return list(self._audit_log)

    def list_tools(self) -> list[dict]:
        return [{"name": t.name, "description": t.description}
                for t in self._tools.values()]
