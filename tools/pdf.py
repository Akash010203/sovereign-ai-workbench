"""tools/pdf.py — Local PDF text extraction tool (no cloud OCR)."""
from __future__ import annotations

from pathlib import Path
from tools.registry import BaseTool, ToolResult


class PDFTextExtractTool(BaseTool):
    """
    Extract plain text from a PDF file using pdfminer.six (local, offline).

    Requires: pip install pdfminer.six
    This is pure Python, no cloud service, no API.
    For scanned PDFs (images inside PDF), see tools/ocr.py.
    """

    @property
    def name(self) -> str:
        return "pdf"

    @property
    def description(self) -> str:
        return "Extract text from a PDF file. Args: path (str). Returns text content."

    def run(self, path: str = "") -> ToolResult:
        pdf_path = Path(path)
        if not pdf_path.exists():
            return ToolResult(tool_name=self.name, success=False, output=None,
                              error=f"PDF file not found: {path}")
        try:
            from pdfminer.high_level import extract_text
            text = extract_text(str(pdf_path))
            if not text.strip():
                return ToolResult(
                    tool_name=self.name, success=True,
                    output="",
                    metadata={"path": path, "note": "No text layer found — try OCR tool for scanned PDFs."}
                )
            return ToolResult(
                tool_name=self.name, success=True, output=text.strip(),
                metadata={"path": path, "char_count": len(text)}
            )
        except ImportError:
            return ToolResult(
                tool_name=self.name, success=False, output=None,
                error="pdfminer.six not installed. Run: pip install pdfminer.six"
            )
        except Exception as exc:
            return ToolResult(tool_name=self.name, success=False, output=None,
                              error=str(exc))
