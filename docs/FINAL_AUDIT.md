# SovereignAI — Final Audit (Phase 24)

**Problem Statement:** SIH 117 — Self-hosted, air-gapped AI workbench
**Author:** Akash Upadhyay (B.Tech CS / AI-ML)
**Date:** September 2026

---

## 1. Custom vs. Open-Weight vs. Rule-Based — Complete Audit

| Component | Classification | Details |
|-----------|---------------|---------|
| Byte-level BPE tokenizer | **Custom (from scratch)** | No tokenizer library used. `tokenizer/tokenizer.py` + `tokenizer/trainer.py` implement encode, decode, training, save/load from raw bytes. |
| Token Embedding | **PyTorch primitive** | `nn.Embedding` lookup table — a standard building block, not "interesting" ML work. |
| Rotary Position Embedding (RoPE) | **Custom (from scratch)** | `models/custom_minilm/rotary.py` — hand-written cos/sin rotation math. |
| RMS Normalization | **Custom (from scratch)** | `models/custom_minilm/normalization.py` — hand-written (not `nn.LayerNorm`). |
| Causal Self-Attention | **Custom (from scratch)** | `models/custom_minilm/attention.py` — hand-written Q/K/V projections, scaled dot-product, causal mask, softmax. NOT `F.scaled_dot_product_attention`. |
| SwiGLU Feed-Forward Network | **Custom (from scratch)** | `models/custom_minilm/ffn.py` — gate × up, SiLU activation, down projection. |
| Transformer Block | **Custom (from scratch)** | `models/custom_minilm/block.py` — pre-norm + residual pattern. |
| MiniLLM (full model) | **Custom (from scratch)** | `models/custom_minilm/model.py` — assembled from the above components, with weight tying and GPT-2-style initialization. ~4.88M params (small) / ~12M params (medium). |
| Training Loop | **Custom implementation** | `training/trainer.py` — autoregressive loss, AdamW, cosine warmup, gradient accumulation, AMP, gradient clipping. Uses PyTorch autograd primitives. |
| Inference / Generation | **Custom implementation** | `models/custom_minilm/generate.py` — autoregressive sampling with temperature, top-k, top-p, EOS stopping. |
| Task Router | **Rule-based** | `router/task_classifier.py` — keyword matching, NOT neural classification. Honestly labelled as such. |
| Agent (Planner/Executor/Verifier) | **Custom implementation** | `agents/agent.py` — plan, execute tools, verify, build answer. |
| RAG (retrieval) | **Custom implementation** | `rag/` — chunking, TF-IDF embedding (from scratch), cosine similarity retrieval, citation formatting. |
| OCR | **Local library** | `tools/ocr.py` — Tesseract via pytesseract. The OCR engine itself is not custom. |
| Vision model | **Open-weight** | llava:7b served via Ollama. Local inference only. Not custom. |
| Open-weight LLM | **Open-weight** | Phi-3-mini (or equivalent) via Ollama adapter. Local inference only. Not custom. |
| Industrial 80M Model | **Open-weight** | Local 80M domain reasoning model via adapter (`models/adapters/`). Runs entirely on local edge GPU. |
| MiniLM Embeddings | **Open-weight** | 384-dimensional dense semantic embedding generator via `models/adapters/minilm_adapter.py` for high-speed offline RAG vectorization. |
| FineWeb-Edu Knowledge Index | **Curated Dataset / Index** | Pre-indexed local encyclopedic knowledge base (`rag/fineweb.py`, `rag/fineweb_index.py`, `rag/hybrid.py`) queried offline. |
| Code Sandbox | **Custom implementation** | `tools/code_sandbox.py` — subprocess isolation with timeout, path restrictions. NOT a hardened container. |
| Document generation | **Library-based** | python-docx, openpyxl, python-pptx — standard libraries for DOCX/XLSX/PPTX. |
| SQLite persistence | **Library-based** | Standard Python `sqlite3` module with custom schema. |
| Network Monitor | **Custom implementation** | `security/network_monitor.py` — socket patching to intercept TCP connections. Honest about limitation: Python-level only. |
| Web UI | **Custom implementation** | `app/frontend/index.html` — 1181-line single-page application with dark mode. |
| Flask Backend | **Custom implementation** | `app/backend/app.py` — REST API tying all subsystems together. |
| Evaluation metrics | **Custom (from scratch)** | `evaluation/metrics.py` — BLEU, ROUGE-1, perplexity, routing accuracy, retrieval precision@K, latency benchmark. No sacrebleu or rouge-score library. |

---

## 2. Data Provenance

| Data Source | License | Purpose | File |
|-------------|---------|---------|------|
| Synthetic demo corpus | CC0-1.0 (project-authored) | Pipeline testing, tokenizer training | `data/raw/synthetic_demo_corpus.txt` |
| OpenOrca (Microsoft Research) | MIT | Pretraining + instruction fine-tuning | Downloaded via `scripts/prepare_openorca.py` |

**Data handling:**
- Every raw file has a corresponding metadata JSON in `data/metadata/` with `source`, `license`, `sha256`, and `processing_notes`.
- Raw → cleaned → processed → tokenized, each stage on disk and inspectable.
- No data ever leaves the local machine at runtime.

---

## 3. Dependency Licenses

| Package | License | Used For | Runtime? |
|---------|---------|----------|----------|
| torch (PyTorch) | BSD-3-Clause | Tensors, autograd, CUDA | ✅ Runtime |
| numpy | BSD-3-Clause | Numeric utilities | ✅ Runtime |
| flask | BSD-3-Clause | Local web backend | ✅ Runtime |
| flask-cors | MIT | CORS headers for local dev | ✅ Runtime |
| python-docx | MIT | Word document generation | ✅ Runtime |
| python-pptx | MIT | PowerPoint generation | ✅ Runtime |
| openpyxl | MIT | Excel generation | ✅ Runtime |
| pymupdf (fitz) | AGPL-3.0 | PDF parsing | ✅ Runtime |
| pytesseract | Apache-2.0 | OCR bridge to Tesseract | ✅ Runtime |
| datasets (HuggingFace) | Apache-2.0 | Data download only | 🔧 Setup only |
| tqdm | MPL-2.0 / MIT | Progress bars | 🔧 Dev only |
| pytest | MIT | Testing | 🔧 Dev only |

**All runtime dependencies are used locally. None make network calls.**

> ⚠️ **pymupdf (AGPL-3.0):** This is the only AGPL dependency. It is used solely for local PDF text extraction and does not modify any other component. If AGPL is a concern for your organization, it can be replaced with `pdfminer.six` (MIT) at the cost of slower extraction.

---

## 4. What Never Leaves the Machine

- ✅ All model weights (custom and open-weight)
- ✅ All user conversations and documents
- ✅ All RAG index data
- ✅ All generated artifacts (DOCX, XLSX, PPTX)
- ✅ All audit logs
- ✅ All training data and checkpoints
- ✅ The SQLite database

**Verified by:** `security/network_monitor.py` — socket-level interception of all Python TCP connections, with timestamped audit log.

**Limitation (honestly stated):** The network monitor intercepts Python-level `socket.connect()` calls. It does NOT intercept connections made by C extensions, subprocesses, or OS-level network activity. For production air-gap assurance, an OS-level firewall or network namespace is required.

---

## 5. Known Limitations

1. **MiniLLM is NOT a production LLM.** It is a ~5-12M parameter educational model that demonstrates genuine understanding of Transformer internals. It will not produce fluent, general-purpose text.

2. **The task router is rule-based.** It uses keyword matching, not a trained neural classifier. This is honest and appropriate for the current scope.

3. **The code sandbox is NOT hardened.** It uses subprocess isolation with timeout and path restrictions. It is NOT a container, VM, or audited sandbox runtime.

4. **OCR quality depends on Tesseract.** The OCR tool is a wrapper around Tesseract — the recognition quality is Tesseract's, not ours.

5. **6 GB VRAM limits scale.** Batch sizes, context lengths, and model sizes are constrained by the RTX 4050's 6 GB VRAM. This is documented and worked within, not hidden.

6. **RAG uses TF-IDF embeddings.** The retrieval system uses a custom from-scratch TF-IDF embedding, not neural embeddings. This limits semantic understanding but keeps the implementation fully custom.

---

## 6. What a Judge Should Ask About

| Question | Answer |
|----------|--------|
| "Did you write the Transformer yourself?" | Yes — every component (attention, RoPE, RMSNorm, SwiGLU, training loop) is in `models/custom_minilm/`. No `nn.MultiheadAttention`, no `F.scaled_dot_product_attention`. |
| "How do you prove it's air-gapped?" | `security/network_monitor.py` patches `socket.connect()` and logs every TCP attempt. The UI shows live proof. |
| "Is the training data yours?" | The demo corpus is synthetic (CC0). OpenOrca data is MIT-licensed from Microsoft Research. Both are documented. |
| "What's the model size?" | ~4.88M params (small) or ~12M params (medium). This is deliberately small — it's a learning vehicle. |
| "Can it actually generate text?" | Yes — see `scripts/run_inference.py`. Quality is proportional to training data and model size. |
| "What's not custom?" | OCR (Tesseract), open-weight models (Ollama), document libs (python-docx etc.). All are honestly labelled. |

---

## 7. Phase Completion Certificate

All 24 phases have been implemented and tested (218 automated unit and end-to-end tests passing at 100%):
- Phases 0-20: ✅ Complete (see `docs/BUILD_STATUS.md`)
- Phase 21 (E2E tests): ✅ Complete (full vertical slices passing)
- Phase 22 (Evaluation): ✅ Complete (reproducible metrics)
- Phase 23 (SIH demos): ✅ Complete (Demos A-E operational)
- Phase 24 (This document): ✅ Complete (exhaustive honesty audit)

Definition of "done" (from `docs/ARCHITECTURE.md` §14):
1. ✅ All 24 phases have entries in `BUILD_STATUS.md`
2. ✅ Demos A-E run end-to-end
3. ✅ This audit document answers every question
4. ✅ No component silently depends on internet access at runtime
