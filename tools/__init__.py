"""tools/__init__.py — Tool package exports."""
from tools.registry import ToolRegistry, BaseTool, ToolResult
from tools.calculator import CalculatorTool
from tools.filesystem import FileReadTool, FileWriteTool
from tools.pdf import PDFTextExtractTool
from tools.ocr import OCRTool
from tools.spreadsheet import SpreadsheetReadTool, SpreadsheetWriteTool
from tools.word import WordWriteTool
from tools.powerpoint import PowerPointWriteTool
from tools.code_sandbox import CodeSandboxTool


def build_default_tool_registry(workspace_root: str = ".") -> ToolRegistry:
    """Build and return the default tool registry with all tools registered."""
    registry = ToolRegistry()
    registry.register(CalculatorTool())
    registry.register(FileReadTool(workspace_root))
    registry.register(FileWriteTool(workspace_root))
    registry.register(PDFTextExtractTool())
    registry.register(OCRTool())
    registry.register(SpreadsheetReadTool())
    registry.register(SpreadsheetWriteTool())
    registry.register(WordWriteTool())
    registry.register(PowerPointWriteTool())
    registry.register(CodeSandboxTool(timeout_seconds=10))
    return registry


__all__ = [
    "ToolRegistry", "BaseTool", "ToolResult",
    "CalculatorTool", "FileReadTool", "FileWriteTool",
    "PDFTextExtractTool", "OCRTool",
    "SpreadsheetReadTool", "SpreadsheetWriteTool",
    "WordWriteTool", "PowerPointWriteTool", "CodeSandboxTool",
    "build_default_tool_registry",
]
