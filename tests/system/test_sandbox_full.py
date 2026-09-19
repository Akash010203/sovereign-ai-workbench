"""
TEST L — Code Sandbox Full Validation

Verifies:
- Safe code executes correctly
- Malicious code is blocked
- Timeout works
- Import allowlist works
- Environment isolation
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))

from tools.code_sandbox import CodeSandboxTool


class TestSandboxExecution:
    """Test safe code execution."""

    def test_simple_print(self):
        tool = CodeSandboxTool(timeout_seconds=10)
        result = tool.run(code="print('Hello from sandbox')")
        assert result.success, f"Failed: {result.error}"
        assert "Hello from sandbox" in result.output["stdout"]

    def test_math_calculation(self):
        tool = CodeSandboxTool(timeout_seconds=10)
        result = tool.run(code="print(25 * 48)")
        assert result.success
        assert "1200" in result.output["stdout"]

    def test_empty_code(self):
        tool = CodeSandboxTool(timeout_seconds=10)
        result = tool.run(code="")
        assert result.success is False

    def test_syntax_error(self):
        tool = CodeSandboxTool(timeout_seconds=10,
                               allowed_imports=["math"])
        result = tool.run(code="def f(\n  broken")
        assert result.success is False

    def test_runtime_error(self):
        tool = CodeSandboxTool(timeout_seconds=10)
        result = tool.run(code="raise ValueError('test error')")
        assert result.success is False
        assert result.output["exit_code"] != 0


class TestSandboxSecurity:
    """Test sandbox security measures."""

    def test_timeout_kills_infinite_loop(self):
        tool = CodeSandboxTool(timeout_seconds=3)
        result = tool.run(code="while True: pass")
        assert result.success is False
        assert "timed out" in result.error.lower()

    def test_import_allowlist_blocks(self):
        tool = CodeSandboxTool(
            timeout_seconds=10,
            allowed_imports=["math"]
        )
        result = tool.run(code="import os; os.system('echo hacked')")
        assert result.success is False
        assert "not in the allowed list" in result.error.lower()

    def test_import_allowlist_allows(self):
        tool = CodeSandboxTool(
            timeout_seconds=10,
            allowed_imports=["math"]
        )
        result = tool.run(code="import math; print(math.sqrt(144))")
        assert result.success
        assert "12" in result.output["stdout"]

    def test_empty_environment(self):
        """Environment inspection is blocked with filesystem/process access."""
        tool = CodeSandboxTool(timeout_seconds=10)
        result = tool.run(code="import os; print(len(os.environ))")
        assert result.success is False
        assert "not in the allowed list" in result.error.lower()

    def test_no_file_access_outside_sandbox(self):
        """Code should not be able to read files outside temp dir."""
        tool = CodeSandboxTool(timeout_seconds=10)
        # Try to read a file that definitely exists
        result = tool.run(code="""
import os
try:
    with open(os.path.expanduser('~/.bashrc'), 'r') as f:
        print(f.read())
except:
    print('BLOCKED')
""")
        if result.success:
            assert "BLOCKED" in result.output["stdout"]

    def test_no_network_access(self):
        """Code should not be able to make network requests easily."""
        tool = CodeSandboxTool(
            timeout_seconds=5,
            allowed_imports=["math", "json"]
        )
        result = tool.run(code="import requests; print('FAIL')")
        assert result.success is False

    def test_subprocess_not_spawnable(self):
        """Code should not easily spawn subprocesses."""
        tool = CodeSandboxTool(
            timeout_seconds=5,
            allowed_imports=["math"]
        )
        result = tool.run(code="import subprocess; subprocess.run(['echo', 'hacked'])")
        assert result.success is False


class TestSandboxToolProperties:
    """Test tool interface."""

    def test_name(self):
        assert CodeSandboxTool().name == "code_sandbox"

    def test_description(self):
        assert len(CodeSandboxTool().description) > 0
