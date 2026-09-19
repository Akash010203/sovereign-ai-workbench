# SovereignAI — Component Inventory & Honest Classification

## Classification Key

| Code | Meaning |
|------|---------|
| A | TRAINED FROM SCRATCH |
| B | CUSTOM ARCHITECTURE + SELF-TRAINED WEIGHTS |
| C | LOCAL PRETRAINED MODEL |
| D | PRETRAINED MODEL WRAPPER |
| E | EXTERNAL API |
| F | RULE-BASED |
| G | MOCK / PLACEHOLDER |

---

## Component Registry

### 1. Custom Tokenizer (Byte-level BPE)
| Field | Value |
|---|---|
| Component | ByteLevelBPETokenizer |
| Implementation | Custom BPE from scratch (tokenizer/tokenizer.py + trainer.py) |
| Architecture | Byte-level Byte Pair Encoding |
| Weights source | Self-trained on project corpus |
| Training source | Local corpus (demo + blended) |
| Inference source | Local |
| External dependency | None |
| Internet requirement | No |
| **Classification** | **A — TRAINED FROM SCRATCH** |

### 2. Custom MiniLLM (Decoder-only Transformer)
| Field | Value |
|---|---|
| Component | MiniLLM |
| Implementation | Custom decoder-only Transformer (models/custom_minilm/) |
| Architecture | Embedding → N×[RMSNorm → CausalAttention+RoPE → RMSNorm → SwiGLU FFN] → RMSNorm → LM Head |
| Weights source | Self-trained (random init → trained on OpenOrca/blended corpus) |
| Training source | Local GPU (RTX 4050) |
| Inference source | Local GPU/CPU |
| External dependency | PyTorch |
| Internet requirement | No |
| **Classification** | **B — CUSTOM ARCHITECTURE + SELF-TRAINED WEIGHTS** |
| Notes | Three sizes: small (~5M), medium (~12M), industrial_80m (~83M). Weights are self-trained but initialized randomly, not transferred from another model. |

### 3. Task Router
| Field | Value |
|---|---|
| Component | TaskRouter + TaskClassifier |
| Implementation | Keyword regex pattern matching (router/task_classifier.py) |
| Architecture | Rule-based (NOT neural) |
| Weights source | N/A |
| Internet requirement | No |
| **Classification** | **F — RULE-BASED** |
| Notes | Honestly labelled in source code and docs as rule-based. |

### 4. Agent System
| Field | Value |
|---|---|
| Component | Agent (Planner → Executor → Verifier) |
| Implementation | Procedural orchestration (agents/) |
| Architecture | Rule-based planning with tool dispatch |
| Weights source | N/A |
| Internet requirement | No |
| **Classification** | **F — RULE-BASED** |

### 5. RAG Embedding Engine
| Field | Value |
|---|---|
| Component | TFIDFEmbedder |
| Implementation | Hashed TF-IDF in `rag/embeddings.py` |
| Architecture | Deterministic 512-dimensional bag-of-words vector |
| Weights source | None |
| Training source | N/A |
| Inference source | Local process |
| External dependency | None |
| Internet requirement | No |
| **Classification** | **F — RULE-BASED** |
| Notes | Used directly for every RAG query; no neural embedding model is loaded. |

### 6. RAG Vector Index
| Field | Value |
|---|---|
| Component | LocalVectorIndex |
| Implementation | Custom cosine-similarity vector store (rag/index.py) |
| Architecture | Brute-force nearest neighbor |
| Internet requirement | No |
| **Classification** | **F — RULE-BASED** |

### 7. OCR
| Field | Value |
|---|---|
| Component | OCRTool |
| Implementation | Wrapper around Tesseract (pytesseract) |
| Architecture | Pretrained OCR engine |
| Weights source | Tesseract's trained models |
| Internet requirement | No |
| **Classification** | **C — LOCAL PRETRAINED MODEL** |

### 8. Vision
| Field | Value |
|---|---|
| Component | N/A |
| Implementation | No dedicated vision model found |
| **Classification** | **NOT_IMPLEMENTED** |
| Notes | Image analysis routes exist in the router, but no dedicated vision model. OCR handles scanned documents. |

### 9. Image Generation
| Field | Value |
|---|---|
| Component | N/A |
| Implementation | No image generation system found |
| **Classification** | **NOT_IMPLEMENTED** |

### 10. Document Generation (DOCX)
| Field | Value |
|---|---|
| Component | WordWriteTool |
| Implementation | python-docx wrapper (tools/word.py) |
| Internet requirement | No |
| **Classification** | **F — RULE-BASED** |

### 11. Document Generation (XLSX)
| Field | Value |
|---|---|
| Component | SpreadsheetWriteTool / SpreadsheetReadTool |
| Implementation | openpyxl wrapper (tools/spreadsheet.py) |
| Internet requirement | No |
| **Classification** | **F — RULE-BASED** |

### 12. Document Generation (PPTX)
| Field | Value |
|---|---|
| Component | PowerPointWriteTool |
| Implementation | python-pptx wrapper (tools/powerpoint.py) |
| Internet requirement | No |
| **Classification** | **F — RULE-BASED** |

### 13. Code Sandbox
| Field | Value |
|---|---|
| Component | CodeSandboxTool |
| Implementation | subprocess isolation (tools/code_sandbox.py) |
| Architecture | Process-level isolation (NOT container) |
| Internet requirement | No |
| **Classification** | **F — RULE-BASED** |
| Notes | Honestly labelled as "process-level isolation only". |

### 14. Database
| Field | Value |
|---|---|
| Component | Database (SQLite) |
| Implementation | sqlite3 stdlib (database/db.py) |
| Internet requirement | No |
| **Classification** | **F — RULE-BASED** |

### 15. Security / Network Monitor
| Field | Value |
|---|---|
| Component | NetworkMonitor |
| Implementation | socket.connect patching (security/network_monitor.py) |
| Internet requirement | No |
| **Classification** | **F — RULE-BASED** |
| Notes | Honestly labelled as Python-level only, not OS-level. |

## Summary

| Classification | Count | Components |
|---|---|---|
| A — Trained from scratch | 1 | Tokenizer |
| B — Custom + self-trained | 1 | MiniLLM |
| C — Local pretrained | 1 | OCR |
| D — Pretrained wrapper | 0 | None |
| F — Rule-based | 8 | Router, Agent, Vector Index, Doc Gen (×3), Sandbox, DB, Security |
| NOT_IMPLEMENTED | 2 | Vision (standalone), Image Generation |
| G — Mock/Placeholder | 0 | None detected |
