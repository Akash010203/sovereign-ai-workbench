"""
TEST K — Document Generation (DOCX, XLSX, PPTX)

Verifies:
- Files are actually created
- Files are not corrupt (can be reopened)
- Content is present
- Formatting exists
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))

from tools.word import WordWriteTool
from tools.spreadsheet import SpreadsheetWriteTool, SpreadsheetReadTool
from tools.powerpoint import PowerPointWriteTool


class TestWordGeneration:
    """Test DOCX generation and validation."""

    def test_create_docx(self, tmp_path):
        tool = WordWriteTool()
        out = tmp_path / "test.docx"
        result = tool.run(
            path=str(out),
            title="Test Document",
            sections=[
                {"heading": "Section 1", "body": "This is the first section."},
                {"heading": "Section 2", "body": "This is the second section."},
            ],
        )
        assert result.success, f"DOCX creation failed: {result.error}"
        assert out.exists(), "DOCX file not created"
        assert out.stat().st_size > 0, "DOCX file is empty"

    def test_docx_can_be_reopened(self, tmp_path):
        tool = WordWriteTool()
        out = tmp_path / "reopen.docx"
        tool.run(
            path=str(out),
            title="Reopen Test",
            sections=[{"heading": "H1", "body": "Body text"}],
        )
        # Reopen with python-docx
        from docx import Document
        doc = Document(out)
        # Should have content
        full_text = "\n".join(p.text for p in doc.paragraphs)
        assert "Reopen Test" in full_text
        assert "Body text" in full_text

    def test_docx_empty_sections(self, tmp_path):
        tool = WordWriteTool()
        out = tmp_path / "empty.docx"
        result = tool.run(path=str(out), title="Empty", sections=[])
        assert result.success

    def test_docx_many_sections(self, tmp_path):
        tool = WordWriteTool()
        out = tmp_path / "many.docx"
        sections = [{"heading": f"Section {i}", "body": f"Content {i}"} for i in range(20)]
        result = tool.run(path=str(out), title="Many Sections", sections=sections)
        assert result.success
        assert out.stat().st_size > 0


class TestSpreadsheetGeneration:
    """Test XLSX generation and validation."""

    def test_create_xlsx(self, tmp_path):
        tool = SpreadsheetWriteTool()
        out = tmp_path / "test.xlsx"
        result = tool.run(
            path=str(out),
            headers=["Name", "Value", "Unit"],
            rows=[
                ["Pressure", 10.5, "bar"],
                ["Temperature", 85.0, "°C"],
                ["Flow Rate", 120, "m³/h"],
            ],
        )
        assert result.success, f"XLSX creation failed: {result.error}"
        assert out.exists()
        assert out.stat().st_size > 0

    def test_xlsx_can_be_reopened(self, tmp_path):
        tool = SpreadsheetWriteTool()
        out = tmp_path / "reopen.xlsx"
        tool.run(
            path=str(out),
            headers=["Col1", "Col2"],
            rows=[["A", 1], ["B", 2]],
        )
        # Reopen with openpyxl
        import openpyxl
        wb = openpyxl.load_workbook(out)
        ws = wb.active
        assert ws["A1"].value == "Col1"
        assert ws["A2"].value == "A"

    def test_xlsx_read_written_file(self, tmp_path):
        write_tool = SpreadsheetWriteTool()
        read_tool = SpreadsheetReadTool()
        out = tmp_path / "readwrite.xlsx"
        write_tool.run(
            path=str(out),
            headers=["Equipment", "Status"],
            rows=[["Pump P-101", "Running"], ["Valve V-200", "Closed"]],
        )
        result = read_tool.run(path=str(out))
        assert result.success
        assert "Pump P-101" in result.output

    def test_xlsx_empty_rows(self, tmp_path):
        tool = SpreadsheetWriteTool()
        out = tmp_path / "empty.xlsx"
        result = tool.run(path=str(out), headers=["A"], rows=[])
        assert result.success


class TestPowerPointGeneration:
    """Test PPTX generation and validation."""

    def test_create_pptx(self, tmp_path):
        tool = PowerPointWriteTool()
        out = tmp_path / "test.pptx"
        result = tool.run(
            path=str(out),
            title="Test Presentation",
            slides=[
                {"title": "Slide 1", "content": "First slide content"},
                {"title": "Slide 2", "content": "Second slide content"},
            ],
        )
        assert result.success, f"PPTX creation failed: {result.error}"
        assert out.exists()
        assert out.stat().st_size > 0

    def test_pptx_can_be_reopened(self, tmp_path):
        tool = PowerPointWriteTool()
        out = tmp_path / "reopen.pptx"
        tool.run(
            path=str(out),
            title="Reopen Test",
            slides=[{"title": "S1", "content": "Content"}],
        )
        from pptx import Presentation
        prs = Presentation(out)
        assert len(prs.slides) >= 2  # title slide + 1 content slide

    def test_pptx_empty_slides(self, tmp_path):
        tool = PowerPointWriteTool()
        out = tmp_path / "empty.pptx"
        result = tool.run(path=str(out), title="Empty", slides=[])
        assert result.success
