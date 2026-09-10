"""
security/permissions.py — Filesystem permission allowlist.

Only files and directories inside the workspace root are accessible.
"""
from __future__ import annotations

from pathlib import Path


class PermissionManager:
    """Enforces filesystem access restrictions."""

    def __init__(self, workspace_root: str | Path) -> None:
        self._root = Path(workspace_root).resolve()

    def is_allowed(self, path: str | Path) -> bool:
        """Return True if path is within the workspace root."""
        try:
            resolved = Path(path).resolve()
            resolved.relative_to(self._root)
            return True
        except ValueError:
            return False

    def check(self, path: str | Path) -> None:
        """Raise PermissionError if path is not allowed."""
        if not self.is_allowed(path):
            raise PermissionError(
                f"Access denied: '{path}' is outside workspace '{self._root}'."
            )
