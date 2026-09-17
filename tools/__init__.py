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
from tools.rag_search import RagSearchTool


def build_default_tool_registry(
    workspace_root: str = ".",
    retriever=None,
) -> ToolRegistry:
    """Build and return the default tool registry with all tools registered.

    Args:
        workspace_root: Root directory for filesystem tools.
        retriever: Optional rag.retriever.Retriever instance. When provided,
                   the ``rag_search`` tool is registered so the agent can
                   search the knowledge base through the tool pipeline.
    """
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
    if retriever is not None:
        registry.register(RagSearchTool(retriever))
    return registry


__all__ = [
    "ToolRegistry", "BaseTool", "ToolResult",
    "CalculatorTool", "FileReadTool", "FileWriteTool",
    "PDFTextExtractTool", "OCRTool",
    "SpreadsheetReadTool", "SpreadsheetWriteTool",
    "WordWriteTool", "PowerPointWriteTool", "CodeSandboxTool",
    "RagSearchTool",
    "build_default_tool_registry",
]
