"""
router/task_classifier.py — Rule-based task classification.

HONESTY DECLARATION
--------------------
This classifier is RULE-BASED (keyword + heuristic pattern matching).
It is NOT a neural classifier.  This is explicitly stated here and in
the BUILD_STATUS.md honesty ledger, as required by the project contract.

A rule-based router is appropriate at this stage because:
  1. The demonstration corpus is too small to train a reliable
     neural classifier.
  2. For the SIH demo, the task categories are well-defined and
     keyword-distinct (code requests contain "def", "python", "code"
     etc.; OCR requests mention "scan", "image", "PDF" etc.)
  3. Rule-based routing is transparent and explainable to judges.
  4. A learned neural classifier can replace this later without
     changing any other component, since the router sits behind
     the TaskCategory interface.

CATEGORIES
----------
See TaskCategory enum below for the full list.
"""
from __future__ import annotations

import re
from enum import Enum


class TaskCategory(str, Enum):
    """All task categories the platform can route to."""
    GENERAL_CHAT        = "general_chat"
    SUMMARIZATION       = "summarization"
    CODING              = "coding"
    DOCUMENT_ANALYSIS   = "document_analysis"
    RETRIEVAL           = "retrieval"
    SPREADSHEET         = "spreadsheet"
    CALCULATION         = "calculation"
    IMAGE_ANALYSIS      = "image_analysis"
    OCR                 = "ocr"
    ENGINEERING         = "engineering_document"
    CODE_EXECUTION      = "code_execution"
    UNKNOWN             = "unknown"


# ── Keyword rule sets (order matters — first match wins) ───────────────────────

_RULES: list[tuple[TaskCategory, list[str]]] = [
    (TaskCategory.CODE_EXECUTION, [
        r"\brun\b.*\bcode\b", r"\bexecute\b", r"\btest\b.*\bscript\b",
        r"\bsandbox\b", r"\brun\b.*\bscript\b", r"\brun\b.*\banalysis\b",
    ]),
    (TaskCategory.CODING, [
        r"\bwrite\b.*\bcode\b", r"\bpython\b", r"\bfunction\b", r"\bclass\b",
        r"\bdef\b", r"\bscript\b", r"\bbug\b", r"\bdebug\b", r"\bprogram\b",
        r"\bsql\b", r"\balgorithm\b",
    ]),
    (TaskCategory.OCR, [
        r"\bocr\b", r"\bscan(ned)?\b", r"\bhandwrit\b", r"\bextract text\b",
        r"\bread.*image\b", r"\bimage.*text\b",
    ]),
    (TaskCategory.IMAGE_ANALYSIS, [
        r"\bimage\b", r"\bphoto\b", r"\bpicture\b", r"\bdiagram\b",
        r"\bvisual\b", r"\bdrawing\b", r"\bp&id\b", r"\bpiping\b",
    ]),
    (TaskCategory.SPREADSHEET, [
        r"\bspreadsheet\b", r"\bexcel\b", r"\bxlsx?\b", r"\btable\b.*\bdata\b",
        r"\bcsv\b",
    ]),
    (TaskCategory.CALCULATION, [
        r"\bcalculate\b", r"\bcompute\b", r"\bmath\b", r"\bformula\b",
        r"\barithmetic\b", r"\bequation\b", r"\bpressure drop\b",
        r"\bflow rate\b", r"\bwhat is \d", r"\bhow much\b.*\d",
    ]),
    (TaskCategory.SUMMARIZATION, [
        r"\bsummariz(e|ation)\b", r"\bsummar(y|ize)\b", r"\btldr\b",
        r"\bbrief\b", r"\boverview\b", r"\bkey point\b",
    ]),
    (TaskCategory.RETRIEVAL, [
        r"\bfind\b.*\bdocument\b", r"\bsearch\b", r"\bknowledge base\b",
        r"\bmanual\b", r"\bsop\b", r"\bprocedure\b", r"\bprevious report\b",
        r"\bbased on our\b", r"\bretriev\b", r"\bfind\b.*\breport\b",
        r"\blook up\b", r"\bsearch for\b",
    ]),
    (TaskCategory.ENGINEERING, [
        r"\binspection\b", r"\bpressure\b", r"\bvalve\b", r"\bpump\b",
        r"\bpipeline\b", r"\bcompressor\b", r"\bflange\b", r"\bmaintenance\b",
        r"\btechnical report\b", r"\bengineering\b",
    ]),
    (TaskCategory.DOCUMENT_ANALYSIS, [
        r"\bdocument\b", r"\bpdf\b", r"\breport\b", r"\bletter\b",
        r"\bmemo\b", r"\banalyze\b.*\bfile\b",
    ]),
    (TaskCategory.GENERAL_CHAT, [
        r"\bhello\b", r"\bhi\b", r"\bhow are\b", r"\bwhat is\b",
        r"\bwho are\b", r"\bhelp me\b",
    ]),
]


def classify_task(user_input: str) -> TaskCategory:
    """
    Classify ``user_input`` into a TaskCategory using rule-based matching.

    HONESTLY LABELLED: rule-based keyword matching, NOT neural classification.

    Args:
        user_input: Raw user message text.

    Returns:
        The best-matching TaskCategory, or UNKNOWN if no rule matches.
    """
    text = user_input.lower()
    for category, patterns in _RULES:
        for pat in patterns:
            if re.search(pat, text):
                return category
    return TaskCategory.UNKNOWN
