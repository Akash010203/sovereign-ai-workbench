"""
tools/filesystem.py — Local filesystem read/write tool.

SECURITY CONTROLS
-----------------
- Only paths inside the configured workspace root are allowed.
- Path traversal attacks (../../etc/passwd) are blocked.
- The allowlist is checked before any file operation.
"""
from __future__ import annotations

from pathlib import Path
from typing import Optional

from tools.registry import BaseTool, ToolResult


class FileReadTool(BaseTool):
    """Read a file from the local workspace."""

    def __init__(self, workspace_root: str | Path) -> None:
        self._root = Path(workspace_root).resolve()

    @property
    def name(self) -> str:
        return "file_read"

    @property
    def description(self) -> str:
        return "Read a text file from the local workspace. Args: path (str)."

    def _safe_path(self, path: str) -> Optional[Path]:
        """Resolve path and reject if it escapes the workspace root."""
        try:
            resolved = (self._root / path).resolve()
            resolved.relative_to(self._root)  # raises ValueError if outside root
            return resolved
        except (ValueError, Exception):
            return None

    def run(self, path: str = "") -> ToolResult:
        safe = self._safe_path(path)
        if safe is None:
            return ToolResult(
                tool_name=self.name, success=False, output=None,
                error=f"Path '{path}' is outside the workspace root or invalid."
            )
        if not safe.exists():
            return ToolResult(
                tool_name=self.name, success=False, output=None,
                error=f"File not found: {path}"
            )
        content = safe.read_text(encoding="utf-8", errors="replace")
        return ToolResult(
            tool_name=self.name, success=True,
            output=content, metadata={"path": str(safe), "size_bytes": safe.stat().st_size}
        )


class FileWriteTool(BaseTool):
    """Write text to a file inside the local workspace."""

    def __init__(self, workspace_root: str | Path) -> None:
        self._root = Path(workspace_root).resolve()

    @property
    def name(self) -> str:
        return "file_write"

    @property
    def description(self) -> str:
        return "Write text content to a file. Args: path (str), content (str)."

    def _safe_path(self, path: str) -> Optional[Path]:
        try:
            resolved = (self._root / path).resolve()
            resolved.relative_to(self._root)
            return resolved
        except Exception:
            return None

    def run(self, path: str = "", content: str = "") -> ToolResult:
        safe = self._safe_path(path)
        if safe is None:
            return ToolResult(
                tool_name=self.name, success=False, output=None,
                error=f"Path '{path}' is outside the workspace or invalid."
            )
        safe.parent.mkdir(parents=True, exist_ok=True)
        safe.write_text(content, encoding="utf-8")
        return ToolResult(
            tool_name=self.name, success=True,
            output=f"Written {len(content)} chars to {path}",
            metadata={"path": str(safe)}
        )
