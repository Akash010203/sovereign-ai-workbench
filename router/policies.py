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
        "preferred": ["custom_minilm_v1", "ollama/deepseek-coder:6.7b", "ollama/phi3:mini"],
        "tools":     ["code_sandbox", "filesystem"],
        "notes":     "Custom MiniLLM handles code tasks; upgrade to deepseek-coder when Ollama is available.",
    },
    TaskCategory.CODE_EXECUTION: {
        "preferred": ["custom_minilm_v1", "ollama/deepseek-coder:6.7b", "ollama/phi3:mini"],
        "tools":     ["code_sandbox"],
        "notes":     "Execution always goes through the sandboxed code runner.",
    },
    TaskCategory.SUMMARIZATION: {
        "preferred": ["custom_minilm_v1", "ollama/phi3:mini", "ollama/mistral:7b-instruct-q4_0"],
        "tools":     ["pdf", "filesystem"],
        "notes":     "Custom MiniLLM is used first; phi3-mini is a future upgrade path.",
    },
    TaskCategory.DOCUMENT_ANALYSIS: {
        "preferred": ["custom_minilm_v1", "ollama/phi3:mini", "ollama/mistral:7b-instruct-q4_0"],
        "tools":     ["pdf", "ocr", "filesystem"],
        "notes":     "Document analysis may need OCR for scanned PDFs.",
    },
    TaskCategory.ENGINEERING: {
        "preferred": ["custom_minilm_v1", "ollama/phi3:mini", "ollama/mistral:7b-instruct-q4_0"],
        "tools":     ["pdf", "ocr", "filesystem", "spreadsheet"],
        "notes":     "Engineering docs combine OCR + domain-specific analysis.",
    },
    TaskCategory.OCR: {
        "preferred": ["custom_minilm_v1", "ollama/llava:7b"],
        "tools":     ["ocr", "pdf"],
        "notes":     "OCR tasks use the local OCR pipeline; llava is an optional upgrade.",
    },
    TaskCategory.IMAGE_ANALYSIS: {
        "preferred": ["custom_minilm_v1", "ollama/llava:7b", "ollama/phi3:mini"],
        "tools":     ["ocr"],
        "notes":     "Image analysis; llava is an optional multimodal upgrade.",
    },
    TaskCategory.RETRIEVAL: {
        "preferred": ["custom_minilm_v1", "ollama/phi3:mini", "ollama/mistral:7b-instruct-q4_0"],
        "tools":     ["rag_search", "filesystem"],
        "notes":     "RAG retrieval uses the local vector index.",
    },
    TaskCategory.SPREADSHEET: {
        "preferred": ["custom_minilm_v1", "ollama/phi3:mini"],
        "tools":     ["spreadsheet", "calculator", "filesystem"],
        "notes":     "Spreadsheet tasks need data analysis capability.",
    },
    TaskCategory.CALCULATION: {
        "preferred": ["custom_minilm_v1", "ollama/phi3:mini"],
        "tools":     ["calculator"],
        "notes":     "Simple calculation uses the local calculator tool.",
    },
    TaskCategory.GENERAL_CHAT: {
        "preferred": ["custom_minilm_v1", "ollama/phi3:mini", "ollama/mistral:7b-instruct-q4_0"],
        "tools":     [],
        "notes":     "Custom MiniLLM handles general chat; Ollama models are upgrades.",
    },
    TaskCategory.UNKNOWN: {
        "preferred": ["custom_minilm_v1", "ollama/phi3:mini"],
        "tools":     [],
        "notes":     "Fallback: use custom MiniLLM.",
    },
}
