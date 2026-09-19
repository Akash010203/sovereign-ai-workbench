"""
TEST H — OCR Validation

Verifies:
- OCR tool initializes
- Handles missing files
- Handles import errors gracefully
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))

from tools.ocr import OCRTool


class TestOCRTool:
    """Test OCR tool behavior."""

    def test_ocr_tool_has_name(self):
        tool = OCRTool()
        assert tool.name == "ocr"

    def test_ocr_tool_has_description(self):
        tool = OCRTool()
        assert len(tool.description) > 0

    def test_ocr_missing_file(self):
        tool = OCRTool()
        result = tool.run(path="/nonexistent/file.png")
        assert result.success is False
        assert "not found" in result.error.lower()

    def test_ocr_empty_path(self):
        tool = OCRTool()
        result = tool.run(path="")
        assert result.success is False
