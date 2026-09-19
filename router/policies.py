"""
router/policies.py — Model selection policies for each task category.

A policy maps a TaskCategory → the preferred model name(s) to use,
with a fallback chain if the preferred model is unavailable.

STRUCTURE
---------
Each entry in ROUTING_POLICY is:
    TaskCategory → {
        "preferred": [model_name, ...],   # ordered by preference
        "tools":     [tool_name, ...],    # tools to enable for this task
        "notes":     str,                 # why this choice was made
    }
"""
from __future__ import annotations

from router.task_classifier import TaskCategory


# ── Routing policy table ───────────────────────────────────────────────────────
# preferred: list of model names in order of preference.
# The router will use the first available one.
# "custom_minilm_v1" is the sole generative model in this fully custom build.

ROUTING_POLICY: dict[TaskCategory, dict] = {
    TaskCategory.CODING: {
        "preferred": ["custom_minilm_v1"],
        "tools":     ["code_sandbox", "filesystem"],
        "notes":     "The custom MiniLLM handles code tasks alongside deterministic local tools.",
    },
    TaskCategory.CODE_EXECUTION: {
        "preferred": ["custom_minilm_v1"],
        "tools":     ["code_sandbox"],
        "notes":     "Execution always goes through the sandboxed code runner.",
    },
    TaskCategory.SUMMARIZATION: {
        "preferred": ["custom_minilm_v1"],
        "tools":     ["pdf", "filesystem"],
        "notes":     "The custom MiniLLM synthesizes summaries from local document text.",
    },
    TaskCategory.DOCUMENT_ANALYSIS: {
        "preferred": ["custom_minilm_v1"],
        "tools":     ["pdf", "ocr", "filesystem"],
        "notes":     "Document analysis may need OCR for scanned PDFs.",
    },
    TaskCategory.ENGINEERING: {
        "preferred": ["custom_minilm_v1"],
        "tools":     ["pdf", "ocr", "filesystem", "spreadsheet"],
        "notes":     "Engineering docs combine OCR + domain-specific analysis.",
    },
    TaskCategory.OCR: {
        "preferred": ["custom_minilm_v1"],
        "tools":     ["ocr", "pdf"],
        "notes":     "OCR tasks use the local OCR pipeline and custom MiniLLM.",
    },
    TaskCategory.IMAGE_ANALYSIS: {
        "preferred": ["custom_minilm_v1"],
        "tools":     ["ocr"],
        "notes":     "Image analysis uses locally extracted text and the custom MiniLLM.",
    },
    TaskCategory.RETRIEVAL: {
        "preferred": ["custom_minilm_v1"],
        "tools":     ["rag_search", "filesystem"],
        "notes":     "RAG retrieval uses the local vector index.",
    },
    TaskCategory.SPREADSHEET: {
        "preferred": ["custom_minilm_v1"],
        "tools":     ["spreadsheet", "calculator", "filesystem"],
        "notes":     "Spreadsheet tasks need data analysis capability.",
    },
    TaskCategory.CALCULATION: {
        "preferred": ["custom_minilm_v1"],
        "tools":     ["calculator"],
        "notes":     "Simple calculation uses the local calculator tool.",
    },
    TaskCategory.GENERAL_CHAT: {
        "preferred": ["custom_minilm_v1"],
        "tools":     [],
        "notes":     "The custom MiniLLM handles general chat.",
    },
    TaskCategory.UNKNOWN: {
        "preferred": ["custom_minilm_v1"],
        "tools":     [],
        "notes":     "Fallback: use custom MiniLLM.",
    },
}
