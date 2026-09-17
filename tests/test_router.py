"""
tests/test_router.py — Unit tests for the task router and classifier.

Tests that the rule-based classifier correctly categorizes
representative inputs for all SIH demo scenarios.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from router.task_classifier import classify_task, TaskCategory


# ─────────────────────────────────────────────────────────────────────────────
# 1. TaskCategory classification
# ─────────────────────────────────────────────────────────────────────────────

class TestClassifyTask:
    """Test that representative inputs are classified correctly."""

    @pytest.mark.parametrize("text,expected", [
        ("write a Python function to sort a list", TaskCategory.CODING),
        ("write code to parse a CSV file", TaskCategory.CODING),
        ("debug this Python script", TaskCategory.CODING),
        ("create a class for data validation", TaskCategory.CODING),
    ])
    def test_coding_classification(self, text, expected):
        assert classify_task(text) == expected

    @pytest.mark.parametrize("text,expected", [
        ("calculate the pressure drop across the pipeline", TaskCategory.CALCULATION),
        ("compute the flow rate through valve V-201", TaskCategory.CALCULATION),
        ("what is 2 + 3 * 4", TaskCategory.CALCULATION),
        ("formula for pipe stress", TaskCategory.CALCULATION),
    ])
    def test_calculation_classification(self, text, expected):
        assert classify_task(text) == expected

    @pytest.mark.parametrize("text,expected", [
        ("search for the previous maintenance report", TaskCategory.RETRIEVAL),
        ("find the document about compressor maintenance", TaskCategory.RETRIEVAL),
        ("what does our SOP say about seal replacement", TaskCategory.RETRIEVAL),
        ("look up the procedure for valve inspection", TaskCategory.RETRIEVAL),
    ])
    def test_retrieval_classification(self, text, expected):
        assert classify_task(text) == expected

    @pytest.mark.parametrize("text,expected", [
        ("extract text from the scanned document", TaskCategory.OCR),
        ("OCR this maintenance logbook page", TaskCategory.OCR),
        ("read the scanned handwritten page", TaskCategory.OCR),
    ])
    def test_ocr_classification(self, text, expected):
        assert classify_task(text) == expected

    @pytest.mark.parametrize("text,expected", [
        ("create a spreadsheet with the sensor data", TaskCategory.SPREADSHEET),
        ("export to Excel format", TaskCategory.SPREADSHEET),
        ("parse this CSV data", TaskCategory.SPREADSHEET),
    ])
    def test_spreadsheet_classification(self, text, expected):
        assert classify_task(text) == expected

    @pytest.mark.parametrize("text,expected", [
        ("summarize the quarterly report", TaskCategory.SUMMARIZATION),
        ("give me a brief overview of this document", TaskCategory.SUMMARIZATION),
    ])
    def test_summarization_classification(self, text, expected):
        assert classify_task(text) == expected

    @pytest.mark.parametrize("text,expected", [
        ("run this code and show output", TaskCategory.CODE_EXECUTION),
        ("execute the analysis script", TaskCategory.CODE_EXECUTION),
    ])
    def test_code_execution_classification(self, text, expected):
        assert classify_task(text) == expected

    def test_unknown_classification(self):
        result = classify_task("xyzzy foobar gibberish 42")
        assert result == TaskCategory.UNKNOWN

    def test_case_insensitive(self):
        assert classify_task("CALCULATE 2+2") == TaskCategory.CALCULATION
        assert classify_task("Write A Python Script") == TaskCategory.CODING


# ─────────────────────────────────────────────────────────────────────────────
# 2. TaskRouter integration
# ─────────────────────────────────────────────────────────────────────────────

class TestTaskRouter:
    def test_router_returns_decision(self):
        from router.router import TaskRouter
        from models.registry import ModelRegistry

        registry = ModelRegistry()
        router = TaskRouter(registry)
        decision = router.route("calculate 2+2")
        assert decision.category == TaskCategory.CALCULATION
        assert decision.reason  # should have a non-empty reason string

    def test_router_decision_has_model_name(self):
        from router.router import TaskRouter
        from models.registry import ModelRegistry

        registry = ModelRegistry()
        router = TaskRouter(registry)
        decision = router.route("hello there")
        assert isinstance(decision.model_name, str)
