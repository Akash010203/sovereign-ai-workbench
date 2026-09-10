"""
tools/code_sandbox.py — Isolated code execution sandbox.

SECURITY MODEL
--------------
Generated code is NEVER executed directly in the main application
process. Instead, it is written to a temp file and executed in a
subprocess with:
  - A configurable timeout (default 10s)
  - stdout and stderr captured (not visible to the network)
  - No access to the main process memory, imports, or secrets
  - Working directory set to an isolated temp folder
  - The main process's environment variables are NOT inherited

This prevents a malicious or buggy LLM-generated script from:
  - Reading environment variables (API keys, secrets)
  - Importing and modifying main process state
  - Running forever (timeout kills it)

HONESTY NOTE: this sandbox provides process-level isolation only.
It does NOT use a container, VM, or OS-level seccomp/namespaces.
For a production deployment, wrapping in a container is recommended.
"""
from __future__ import annotations

import subprocess
import sys
import tempfile
from pathlib import Path

from tools.registry import BaseTool, ToolResult


class CodeSandboxTool(BaseTool):
    """
    Execute Python code in an isolated subprocess.

    Args:
        timeout_seconds: Kill the process after this many seconds.
        allowed_imports: If set, reject code that imports anything not in this list.
    """

    def __init__(
        self,
        timeout_seconds: int = 10,
        allowed_imports: list[str] | None = None,
    ) -> None:
        self._timeout = timeout_seconds
        self._allowed = set(allowed_imports) if allowed_imports else None

    @property
    def name(self) -> str:
        return "code_sandbox"

    @property
    def description(self) -> str:
        return (
            "Execute Python code in an isolated subprocess. "
            "Args: code (str). Returns stdout, stderr, exit_code."
        )

    def run(self, code: str = "") -> ToolResult:
        if not code.strip():
            return ToolResult(tool_name=self.name, success=False, output=None,
                              error="No code provided.")

        # ── Import allowlist check ─────────────────────────────────────
        if self._allowed is not None:
            import ast
            try:
                tree = ast.parse(code)
            except SyntaxError as e:
                return ToolResult(tool_name=self.name, success=False, output=None,
                                  error=f"Syntax error in code: {e}")
            for node in ast.walk(tree):
                if isinstance(node, (ast.Import, ast.ImportFrom)):
                    mods = ([a.name for a in node.names]
                            if isinstance(node, ast.Import)
                            else [node.module or ""])
                    for mod in mods:
                        top = mod.split(".")[0]
                        if top and top not in self._allowed:
                            return ToolResult(
                                tool_name=self.name, success=False, output=None,
                                error=f"Import '{mod}' is not in the allowed list."
                            )

        # ── Write code to temp file and execute ───────────────────────
        with tempfile.TemporaryDirectory() as tmpdir:
            code_file = Path(tmpdir) / "sandbox_script.py"
            code_file.write_text(code, encoding="utf-8")

            try:
                proc = subprocess.run(
                    [sys.executable, str(code_file)],
                    capture_output=True,
                    text=True,
                    timeout=self._timeout,
                    cwd=tmpdir,
                    env={},  # empty environment — no secrets exposed
                )
                return ToolResult(
                    tool_name=self.name,
                    success=(proc.returncode == 0),
                    output={
                        "stdout":    proc.stdout,
                        "stderr":    proc.stderr,
                        "exit_code": proc.returncode,
                    },
                    metadata={"timeout_s": self._timeout}
                )
            except subprocess.TimeoutExpired:
                return ToolResult(
                    tool_name=self.name, success=False, output=None,
                    error=f"Code execution timed out after {self._timeout}s."
                )
            except Exception as exc:
                return ToolResult(tool_name=self.name, success=False, output=None,
                                  error=str(exc))
