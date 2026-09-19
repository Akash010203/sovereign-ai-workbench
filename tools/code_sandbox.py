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
import ast
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
        # Only modules with no filesystem, process, or socket capabilities are
        # permitted.  Callers cannot widen this boundary by passing ``os``.
        safe_imports = {"datetime", "json", "math", "random", "statistics", "sys", "time"}
        requested = set(allowed_imports) if allowed_imports is not None else safe_imports
        self._allowed = requested & safe_imports

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

        try:
            tree = ast.parse(code)
        except SyntaxError as e:
            return ToolResult(tool_name=self.name, success=False, output=None,
                              error=f"Syntax error in code: {e}")

        policy_error = self._validate_ast(tree)
        if policy_error:
            return ToolResult(tool_name=self.name, success=False, output=None,
                              error=policy_error)

        # ── Write code to temp file and execute ───────────────────────
        with tempfile.TemporaryDirectory() as tmpdir:
            code_file = Path(tmpdir) / "sandbox_script.py"
            code_file.write_text(code, encoding="utf-8")

            try:
                proc = subprocess.run(
                    [sys.executable, "-I", str(code_file)],
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

    def _validate_ast(self, tree: ast.AST) -> str | None:
        """Reject syntax that can escape the constrained Python subprocess."""
        forbidden_names = {
            "__builtins__", "__import__", "breakpoint", "compile", "eval",
            "exec", "getattr", "globals", "input", "locals", "open",
            "setattr", "delattr", "vars",
        }
        forbidden_nodes = (ast.ClassDef, ast.Delete, ast.Global, ast.Nonlocal, ast.With)
        for node in ast.walk(tree):
            if isinstance(node, forbidden_nodes):
                return f"Syntax '{type(node).__name__}' is not allowed in the sandbox."
            if isinstance(node, ast.Name) and node.id in forbidden_names:
                return f"Builtin '{node.id}' is not allowed in the sandbox."
            if isinstance(node, ast.Attribute) and node.attr.startswith("__"):
                return "Dunder attribute access is not allowed in the sandbox."
            if isinstance(node, (ast.Import, ast.ImportFrom)):
                modules = ([alias.name for alias in node.names]
                           if isinstance(node, ast.Import)
                           else [node.module or ""])
                for module in modules:
                    top = module.split(".")[0]
                    if not top or top not in self._allowed:
                        return f"Import '{module}' is not in the allowed list."
        return None
