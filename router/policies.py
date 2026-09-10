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
# "custom_minilm_v1" is always the fallback of last resort (always loaded).

ROUTING_POLICY: dict[TaskCategory, dict] = {
    TaskCategory.CODING: {
        "preferred": ["ollama/deepseek-coder:6.7b", "ollama/phi3:mini", "custom_minilm_v1"],
        "tools":     ["code_sandbox", "filesystem"],
        "notes":     "Code tasks need a strong reasoning model; deepseek-coder is preferred.",
    },
    TaskCategory.CODE_EXECUTION: {
        "preferred": ["ollama/deepseek-coder:6.7b", "ollama/phi3:mini", "custom_minilm_v1"],
        "tools":     ["code_sandbox"],
        "notes":     "Execution always goes through the sandboxed code runner.",
    },
    TaskCategory.SUMMARIZATION: {
        "preferred": ["ollama/phi3:mini", "ollama/mistral:7b-instruct-q4_0", "custom_minilm_v1"],
        "tools":     ["pdf", "filesystem"],
        "notes":     "Summarization needs a capable language model; phi3-mini is compact and fast.",
    },
    TaskCategory.DOCUMENT_ANALYSIS: {
        "preferred": ["ollama/phi3:mini", "ollama/mistral:7b-instruct-q4_0", "custom_minilm_v1"],
        "tools":     ["pdf", "ocr", "filesystem"],
        "notes":     "Document analysis may need OCR for scanned PDFs.",
    },
    TaskCategory.ENGINEERING: {
        "preferred": ["ollama/phi3:mini", "ollama/mistral:7b-instruct-q4_0", "custom_minilm_v1"],
        "tools":     ["pdf", "ocr", "filesystem", "spreadsheet"],
        "notes":     "Engineering docs combine OCR + domain-specific analysis.",
    },
    TaskCategory.OCR: {
        "preferred": ["ollama/llava:7b", "custom_minilm_v1"],
        "tools":     ["ocr", "pdf"],
        "notes":     "OCR tasks use the local vision model (llava) or the local OCR pipeline.",
    },
    TaskCategory.IMAGE_ANALYSIS: {
        "preferred": ["ollama/llava:7b", "ollama/phi3:mini", "custom_minilm_v1"],
        "tools":     ["ocr"],
        "notes":     "Vision tasks need a multimodal model; llava is local and open-weight.",
    },
    TaskCategory.RETRIEVAL: {
        "preferred": ["ollama/phi3:mini", "ollama/mistral:7b-instruct-q4_0", "custom_minilm_v1"],
        "tools":     ["rag_search", "filesystem"],
        "notes":     "RAG retrieval uses the local vector index (Phase 12).",
    },
    TaskCategory.SPREADSHEET: {
        "preferred": ["ollama/phi3:mini", "custom_minilm_v1"],
        "tools":     ["spreadsheet", "calculator", "filesystem"],
        "notes":     "Spreadsheet tasks need data analysis capability.",
    },
    TaskCategory.CALCULATION: {
        "preferred": ["ollama/phi3:mini", "custom_minilm_v1"],
        "tools":     ["calculator"],
        "notes":     "Simple calculation uses the local calculator tool.",
    },
    TaskCategory.GENERAL_CHAT: {
        "preferred": ["ollama/phi3:mini", "ollama/mistral:7b-instruct-q4_0", "custom_minilm_v1"],
        "tools":     [],
        "notes":     "General chat needs a capable instruction-following model.",
    },
    TaskCategory.UNKNOWN: {
        "preferred": ["ollama/phi3:mini", "custom_minilm_v1"],
        "tools":     [],
        "notes":     "Fallback: try general chat model.",
    },
}
