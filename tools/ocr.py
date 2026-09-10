"""
tools/ocr.py — Local OCR tool using Tesseract (offline).

Tesseract OCR is a free, open-source OCR engine maintained by Google.
It runs entirely locally — no cloud API calls.
Install: https://github.com/UB-Mannheim/tesseract/wiki (Windows installer)
Then: pip install pytesseract Pillow pdf2image

PIPELINE FOR SCANNED PDF:
  scanned PDF → pdf2image (page images) → Tesseract → text per page → joined
"""
from __future__ import annotations

from pathlib import Path
from tools.registry import BaseTool, ToolResult


class OCRTool(BaseTool):
    """
    Extract text from images or scanned PDFs using local Tesseract OCR.
    All processing is on-device — no cloud API.
    """

    @property
    def name(self) -> str:
        return "ocr"

    @property
    def description(self) -> str:
        return (
            "Run local OCR on an image or scanned PDF. "
            "Args: path (str). Supports PNG, JPG, TIFF, PDF."
        )

    def run(self, path: str = "") -> ToolResult:
        file_path = Path(path)
        if not file_path.exists():
            return ToolResult(tool_name=self.name, success=False, output=None,
                              error=f"File not found: {path}")

        suffix = file_path.suffix.lower()

        try:
            import pytesseract
            from PIL import Image
        except ImportError:
            return ToolResult(
                tool_name=self.name, success=False, output=None,
                error="pytesseract/Pillow not installed. Run: pip install pytesseract Pillow"
            )

        try:
            if suffix == ".pdf":
                # Convert PDF pages to images, then OCR each
                try:
                    from pdf2image import convert_from_path
                except ImportError:
                    return ToolResult(
                        tool_name=self.name, success=False, output=None,
                        error="pdf2image not installed. Run: pip install pdf2image"
                    )
                images = convert_from_path(str(file_path), dpi=200)
                pages_text = []
                for i, img in enumerate(images):
                    page_text = pytesseract.image_to_string(img, lang="eng")
                    pages_text.append(f"--- Page {i+1} ---\n{page_text}")
                full_text = "\n\n".join(pages_text)
            else:
                img = Image.open(file_path)
                full_text = pytesseract.image_to_string(img, lang="eng")

            return ToolResult(
                tool_name=self.name, success=True, output=full_text.strip(),
                metadata={"path": path, "char_count": len(full_text)}
            )
        except Exception as exc:
            return ToolResult(tool_name=self.name, success=False, output=None,
                              error=f"OCR failed: {exc}")
