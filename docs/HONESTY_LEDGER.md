# SovereignAI — Honesty Ledger

> **SIH 2026, Problem Statement SIH 117**
> Akash | B.Tech CS/AI-ML
> Last updated: Phase 24 (all phases complete)

This document exists to prevent misleading judges. Every major component
is labelled by what it actually is. You can point to this during your
judge presentation to demonstrate intellectual honesty — which is itself
a winning quality.

---

## LEGEND

| Label | Meaning |
|-------|---------|
| ✅ CUSTOM FROM SCRATCH | Implemented in this project, zero reliance on library magic |
| ⚡ OPEN-WEIGHT (HONEST) | Pre-trained by another organisation, used as-is |
| 🔧 RULE-BASED | Deterministic logic, no ML involved |
| 📦 LIBRARY WRAPPER | Thin wrapper over an installed package |

---

## COMPONENT BREAKDOWN

### 1. Tokenizer
**Label: ✅ CUSTOM FROM SCRATCH**

- Byte-level BPE (Byte-Pair Encoding) — the merge table is built from scratch
- `tokenizer/tokenizer.py` — encode(), decode(), special tokens
- `tokenizer/trainer.py` — counts byte pairs, selects merges, builds vocab
- No use of `tokenizers`, `sentencepiece`, or `tiktoken`
- Verified with 14 unit tests, round-trip exact on emoji and Unicode

### 2. Neural Network Sub-modules
**Label: ✅ CUSTOM FROM SCRATCH**

| Module | What's custom |
|--------|--------------|
| `normalization.py` | RMSNorm — formula implemented in PyTorch, no `nn.LayerNorm` |
| `rotary.py` | RoPE — sinusoidal position encodings applied per head, from formula |
| `attention.py` | Causal self-attention — QKV projection, scaled dot-product, causal mask, all manual |
| `ffn.py` | SwiGLU FFN — gate × up projections with SiLU, down projection |
| `block.py` | Pre-norm Transformer block — residual connections, layer ordering |
| `model.py` | MiniLLM — embedding → blocks → final norm → tied LM head |

> **What PyTorch IS used for**: tensor operations (matmul, softmax, backward), CUDA transfer.
> PyTorch is the execution engine. The *architecture logic* is ours.

### 3. Training Pipeline
**Label: ✅ CUSTOM FROM SCRATCH**

- `training/dataset.py` — sliding-window tokenized dataset, custom DataLoader wrapper
- `training/scheduler.py` — cosine-with-warmup LR schedule, implemented from formula
- `training/trainer.py` — full training loop: forward → loss → backward → AdamW → clip → checkpoint
- Uses `torch.optim.AdamW` (PyTorch built-in) — the optimizer itself is not reimplemented
- Uses `torch.cuda.amp.GradScaler` for mixed precision
- All checkpoint/resume logic is custom

### 4. Text Generation
**Label: ✅ CUSTOM FROM SCRATCH**

- `models/custom_minilm/generate.py`
- Temperature scaling, Top-k, Top-p (nucleus) sampling — all implemented explicitly
- Context window sliding for long prompts — custom
- Uses `torch.multinomial` for sampling from the final probability distribution

### 5. Model Abstraction Layer
**Label: ✅ CUSTOM FROM SCRATCH**

- `ModelProvider` abstract class — designed and written here
- `CustomMiniLLMAdapter` — wraps and runs our own model in-process
- `ModelRegistry` — register/lookup/availability check

### 6. Task Router
**Label: 🔧 RULE-BASED**

- `router/task_classifier.py` — keyword + regex pattern matching
- NOT a neural classifier (there is insufficient data to train one reliably)
- Correctly classifies 95%+ of SIH demo queries (verified by RoutingEvaluator)
- Explicitly labelled as rule-based in every file it appears in

### 7. Tool System
**Label: ✅ CUSTOM (framework) + 📦 LIBRARY WRAPPER (individual tools)**

| Tool | What's custom | Library used |
|------|--------------|-------------|
| ToolRegistry | Design, audit logging, error handling | (none) |
| CalculatorTool | AST-safe evaluator, whitelist | Python `ast` stdlib |
| FileReadTool / FileWriteTool | Path traversal security check | (none) |
| PDFTextExtractTool | Thin wrapper | `pdfminer.six` |
| OCRTool | Multi-format dispatch (image/PDF) | `pytesseract`, `Pillow` |
| SpreadsheetReadTool/WriteTool | Thin wrapper | `openpyxl` |
| WordWriteTool | Thin wrapper | `python-docx` |
| PowerPointWriteTool | Thin wrapper | `python-pptx` |
| CodeSandboxTool | Subprocess isolation, timeout, env scrubbing | `subprocess` stdlib |

### 8. Agent System
**Label: ✅ CUSTOM FROM SCRATCH**

- `agents/state.py` — TaskStatus state machine, AgentStep, TaskState
- `agents/memory.py` — conversation window (deque-based)
- `agents/planner.py` — template plans for demo scenarios + model-based planning
- `agents/executor.py` — sequential step runner, tool dispatch, model dispatch
- `agents/verifier.py` — structural post-execution checks (not neural)
- `agents/agent.py` — orchestrator wiring all of the above

### 9. RAG (Retrieval-Augmented Generation)
**Label: ✅ CUSTOM FROM SCRATCH**

| Component | Label |
|-----------|-------|
| `rag/chunking.py` — sentence-boundary chunker | ✅ CUSTOM |
| `rag/index.py` — flat cosine similarity index + JSON persistence | ✅ CUSTOM |
| `rag/ingest.py` — document ingestion pipeline | ✅ CUSTOM |
| `rag/retriever.py` — query embed → search → ranked results | ✅ CUSTOM |
| `rag/citations.py` — citation formatting + RAG prompt builder | ✅ CUSTOM |
| `TFIDFEmbedder` — bag-of-words, pure Python | ✅ CUSTOM |

> The vector index is NOT Pinecone, Chroma, Weaviate, or any cloud/external DB.
> It is a Python dict + cosine similarity in 30 lines of code.

### 10. Security & Air-Gap
**Label: ✅ CUSTOM FROM SCRATCH**

- `security/permissions.py` — filesystem allowlist with path traversal protection
- `security/audit.py` — JSON audit log for every tool call and model call
- `security/network_monitor.py` — patches `socket.socket.connect` to intercept external TCP
- `security/offline_mode.py` — connectivity check + timestamped proof log

> The network monitor patches at the Python level only.
> For production air-gap assurance, OS-level firewall or network namespaces are needed.
> This is documented in the code. It is sufficient for SIH demo purposes.

### 11. Database
**Label: ✅ CUSTOM (schema + repositories) + 📦 LIBRARY (sqlite3 stdlib)**

- Schema designed for this project's data model (9 tables)
- Repository pattern: ConversationRepository, TaskRepository
- Uses Python's built-in `sqlite3` — no ORM, no cloud database

### 12. Backend
**Label: ✅ CUSTOM**

- `app/backend/app.py` — Flask application with 10 REST endpoints
- All endpoints custom-written, wiring together all subsystems
- Uses Flask (open-source Python library) as the HTTP layer

### 13. Web UI
**Label: ✅ CUSTOM**

- `app/frontend/index.html` — single-file HTML/CSS/JS, no React, no Vue, no framework
- 6-panel workbench: Chat, Agent, Knowledge Base, Models, Security, Audit Log
- Pure vanilla CSS + JS, Inter + JetBrains Mono fonts from Google Fonts

### 14. Evaluation Metrics
**Label: ✅ CUSTOM FROM SCRATCH**

- BLEU (from formula — no sacrebleu)
- ROUGE-1 (from formula — no rouge-score)
- Perplexity (from cross-entropy — using the custom MiniLLM)
- Routing accuracy (over labeled test set)
- RAG retrieval precision@K

---

## WHAT WAS NOT BUILT FROM SCRATCH (and that's correct)

| Component | Why it's appropriate |
|-----------|---------------------|
| PyTorch | A numerical computation library — like using NumPy |
| Flask | HTTP library — like using requests |
| sqlite3 | Built into Python stdlib |
| pytesseract/Tesseract | OCR engine — we wrap it, not reimplement it |

---

## JUDGE Q&A PREPARATION

**Q: Is this really from scratch?**
> The architecture is. Every Transformer math operation — RoPE, RMSNorm, SwiGLU,
> causal attention — is written in Python/PyTorch, not called from a Transformers library.
> We use PyTorch as a tensor math engine, the same way a C programmer uses libc.

**Q: Why is the custom MiniLLM a 5M parameter model?**
> Because we're training it on a demo corpus on a laptop GPU (RTX 4050, 6GB VRAM).
> The purpose of the custom model is to prove we understand Transformer architecture from
> first principles — not to compete with GPT-4 in generation quality.
> The platform routes language tasks to the locally trained custom MiniLLM and deterministic local tools.

**Q: How do you prove no data is leaving the machine?**
> Run Scenario 5. The NetworkMonitor patches Python's socket layer.
> The custom MiniLLM runs in-process, with no model-server connection.
> The audit log shows every tool call and model call. The network log shows zero
> external TCP connection attempts.

**Q: What problem does this solve for SIH 117?**
> SIH 117 asks for an AI workbench for confidential industrial knowledge work —
> specifically where data cannot leave the organization's network.
> This platform enables: document analysis, knowledge retrieval, report generation,
> engineering calculations, and code assistance — all without a single byte leaving
> the machine.
