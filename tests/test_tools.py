"""
tests/test_tools.py — Unit tests for the tools module.

Tests each tool class for:
  - Correct name and description properties
  - Success on valid input
  - Proper error handling on bad input
  - ToolResult structure and fields
"""
from __future__ import annotations

import sys
import tempfile
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from tools.registry import ToolRegistry, BaseTool, ToolResult
from tools.calculator import CalculatorTool
from tools.filesystem import FileReadTool, FileWriteTool
from tools.code_sandbox import CodeSandboxTool
from tools.pdf import PDFTextExtractTool
from tools.ocr import OCRTool
from tools.spreadsheet import SpreadsheetReadTool, SpreadsheetWriteTool
from tools.word import WordWriteTool
from tools.powerpoint import PowerPointWriteTool
from tools.rag_search import RagSearchTool
from tools import build_default_tool_registry


# ─────────────────────────────────────────────────────────────────────────────
# Fixtures
# ─────────────────────────────────────────────────────────────────────────────

@pytest.fixture
def workspace(tmp_path):
    """Create a temporary workspace directory with test files."""
    (tmp_path / "hello.txt").write_text("Hello, World!", encoding="utf-8")
    (tmp_path / "subdir").mkdir()
    (tmp_path / "subdir" / "nested.txt").write_text("nested content", encoding="utf-8")
    return tmp_path


@pytest.fixture
def registry(workspace):
    return build_default_tool_registry(workspace_root=str(workspace))


# ─────────────────────────────────────────────────────────────────────────────
# 1. ToolRegistry
# ─────────────────────────────────────────────────────────────────────────────

class TestToolRegistry:
    def test_register_and_list(self, registry):
        tools = registry.list_tools()
        names = [t["name"] for t in tools]
        assert "calculator" in names
        assert "file_read" in names
        assert "file_write" in names
        assert "code_sandbox" in names

    def test_call_missing_tool(self, registry):
        result = registry.call("nonexistent_tool")
        assert not result.success
        assert "not found" in result.error.lower()

    def test_audit_log(self, registry):
        registry.call("calculator", expression="2+2")
        log = registry.get_audit_log()
        assert len(log) >= 1
        assert log[-1]["tool"] == "calculator"
        assert log[-1]["success"] is True


# ─────────────────────────────────────────────────────────────────────────────
# 2. CalculatorTool
# ─────────────────────────────────────────────────────────────────────────────

class TestCalculatorTool:
    def setup_method(self):
        self.calc = CalculatorTool()

    def test_name(self):
        assert self.calc.name == "calculator"
        assert len(self.calc.description) > 10

    def test_basic_arithmetic(self):
        result = self.calc.run(expression="2 + 3 * 4")
        assert result.success
        assert result.output == 14

    def test_power(self):
        result = self.calc.run(expression="2 ** 10")
        assert result.success
        assert result.output == 1024

    def test_math_functions(self):
        result = self.calc.run(expression="sqrt(144)")
        assert result.success
        assert result.output == 12.0

    def test_pi_constant(self):
        result = self.calc.run(expression="pi")
        assert result.success
        assert abs(result.output - 3.14159) < 0.001

    def test_darcy_weisbach(self):
        """The demo scenario 4 formula."""
        result = self.calc.run(
            expression="(0.02 * 50 * 1000 * 0.637 * 0.637) / (2 * 0.1)"
        )
        assert result.success
        assert isinstance(result.output, (int, float))
        assert result.output > 0

    def test_empty_expression(self):
        result = self.calc.run(expression="")
        assert not result.success
        assert result.error

    def test_unsafe_expression_rejected(self):
        """Ensure __import__ and exec-like calls are blocked."""
        result = self.calc.run(expression="__import__('os').system('echo hi')")
        assert not result.success

    def test_division_by_zero(self):
        result = self.calc.run(expression="1/0")
        assert not result.success


# ─────────────────────────────────────────────────────────────────────────────
# 3. FileReadTool / FileWriteTool
# ─────────────────────────────────────────────────────────────────────────────

class TestFileTools:
    def test_read_existing_file(self, workspace):
        reader = FileReadTool(workspace)
        result = reader.run(path="hello.txt")
        assert result.success
        assert result.output == "Hello, World!"

    def test_read_nested_file(self, workspace):
        reader = FileReadTool(workspace)
        result = reader.run(path="subdir/nested.txt")
        assert result.success
        assert result.output == "nested content"

    def test_read_missing_file(self, workspace):
        reader = FileReadTool(workspace)
        result = reader.run(path="does_not_exist.txt")
        assert not result.success
        assert "not found" in result.error.lower()

    def test_path_traversal_blocked(self, workspace):
        reader = FileReadTool(workspace)
        result = reader.run(path="../../etc/passwd")
        assert not result.success
        assert "outside" in result.error.lower()

    def test_write_new_file(self, workspace):
        writer = FileWriteTool(workspace)
        result = writer.run(path="output.txt", content="test data")
        assert result.success
        assert (workspace / "output.txt").read_text() == "test data"

    def test_write_creates_subdirectory(self, workspace):
        writer = FileWriteTool(workspace)
        result = writer.run(path="newdir/deep/file.txt", content="deep")
        assert result.success
        assert (workspace / "newdir" / "deep" / "file.txt").exists()

    def test_write_path_traversal_blocked(self, workspace):
        writer = FileWriteTool(workspace)
        result = writer.run(path="../../../tmp/evil.txt", content="hack")
        assert not result.success


# ─────────────────────────────────────────────────────────────────────────────
# 4. CodeSandboxTool
# ─────────────────────────────────────────────────────────────────────────────

class TestCodeSandboxTool:
    def setup_method(self):
        self.sandbox = CodeSandboxTool(timeout_seconds=5)

    def test_simple_execution(self):
        result = self.sandbox.run(code="print('Hello from sandbox')")
        assert result.success
        assert "Hello from sandbox" in result.output["stdout"]

    def test_captures_stderr(self):
        result = self.sandbox.run(code="import sys; sys.stderr.write('err msg')")
        assert result.success
        assert "err msg" in result.output["stderr"]

    def test_exit_code_nonzero(self):
        result = self.sandbox.run(code="raise ValueError('boom')")
        assert not result.success
        assert result.output["exit_code"] != 0

    def test_empty_code(self):
        result = self.sandbox.run(code="")
        assert not result.success

    def test_timeout(self):
        sandbox = CodeSandboxTool(timeout_seconds=2)
        result = sandbox.run(code="import time; time.sleep(10)")
        assert not result.success
        assert "timed out" in result.error.lower()

    def test_allowed_imports_whitelist(self):
        sandbox = CodeSandboxTool(timeout_seconds=5, allowed_imports=["math"])
        result = sandbox.run(code="import os")
        assert not result.success
        assert "not in the allowed" in result.error.lower()

    def test_allowed_import_accepted(self):
        sandbox = CodeSandboxTool(timeout_seconds=5, allowed_imports=["math"])
        result = sandbox.run(code="import math\nprint(math.sqrt(4))")
        assert result.success


# ─────────────────────────────────────────────────────────────────────────────
# 5. Document generation tools — basic instantiation + name
# ─────────────────────────────────────────────────────────────────────────────

class TestDocumentTools:
    def test_pdf_tool_name(self):
        assert PDFTextExtractTool().name == "pdf"

    def test_ocr_tool_name(self):
        assert OCRTool().name == "ocr"

    def test_spreadsheet_read_name(self):
        assert SpreadsheetReadTool().name == "spreadsheet_read"

    def test_spreadsheet_write_name(self):
        assert SpreadsheetWriteTool().name == "spreadsheet_write"

    def test_word_write_name(self):
        assert WordWriteTool().name == "word_write"

    def test_powerpoint_write_name(self):
        assert PowerPointWriteTool().name == "powerpoint_write"

    def test_pdf_missing_file(self):
        result = PDFTextExtractTool().run(path="/nonexistent/file.pdf")
        assert not result.success

    def test_ocr_missing_file(self):
        result = OCRTool().run(path="/nonexistent/file.png")
        assert not result.success


# ─────────────────────────────────────────────────────────────────────────────
# 6. RagSearchTool
# ─────────────────────────────────────────────────────────────────────────────

class TestRagSearchTool:
    def test_name(self):
        tool = RagSearchTool(retriever=None)
        assert tool.name == "rag_search"

    def test_empty_query(self):
        tool = RagSearchTool(retriever=None)
        result = tool.run(query="")
        assert not result.success
        assert "empty" in result.error.lower()


# ─────────────────────────────────────────────────────────────────────────────
# 7. build_default_tool_registry
# ─────────────────────────────────────────────────────────────────────────────

def test_default_registry_has_all_tools():
    registry = build_default_tool_registry(workspace_root=".")
    tools = {t["name"] for t in registry.list_tools()}
    expected = {
        "calculator", "file_read", "file_write", "pdf", "ocr",
        "spreadsheet_read", "spreadsheet_write",
        "word_write", "powerpoint_write", "code_sandbox",
    }
    assert expected.issubset(tools), f"Missing tools: {expected - tools}"


def test_registry_includes_rag_when_retriever_provided():
    """When a retriever is passed, rag_search should be registered."""
    class FakeRetriever:
        pass
    registry = build_default_tool_registry(
        workspace_root=".", retriever=FakeRetriever()
    )
    names = {t["name"] for t in registry.list_tools()}
    assert "rag_search" in names


def test_registry_excludes_rag_when_no_retriever():
    registry = build_default_tool_registry(workspace_root=".")
    names = {t["name"] for t in registry.list_tools()}
    assert "rag_search" not in names
