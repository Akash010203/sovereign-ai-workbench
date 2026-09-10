# SovereignAI — Self-Hosted, Air-Gapped AI Workbench

**Smart India Hackathon 2026 — Problem Statement SIH 117**

Built by: **Akash Upadhyay** (B.Tech CS / AI-ML)

> **Honesty first.** The "custom LLM" in this project is a genuine,
> from-scratch, randomly-initialized, self-trained **educational /
> research MiniLLM** — not a claim of frontier-model performance. See
> [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md) §2 for an exact,
> unembellished account of what is custom, what is open-weight, and
> what is rule-based. [`docs/FINAL_AUDIT.md`](docs/FINAL_AUDIT.md)
> makes this final and complete.

---

## What This Project Is

A **self-hosted AI workbench** designed to run entirely on an
organization's own hardware, with **zero runtime dependency on any
cloud AI API**, so that confidential engineering / PSU / defence-
adjacent documents never leave the premises.

| Feature | Implementation |
|---------|---------------|
| 🧠 Custom LLM | From-scratch Transformer (RoPE, SwiGLU, RMSNorm) — ~5M or ~12M params |
| 🔄 Multi-model routing | Rule-based task router → MiniLLM / Ollama / Vision |
| 🤖 Agent system | Planner → Executor → Verifier with memory |
| 📚 RAG | Local vector index, chunking, retrieval + citations |
| 🔍 OCR | Tesseract-powered scanned document text extraction |
| 👁️ Vision | Local vision model inference via Ollama |
| 🔒 Air-gap | Network monitor with timestamped proof of offline operation |
| 📄 Document gen | DOCX, XLSX, PPTX generation from AI outputs |
| 💾 Persistence | SQLite with full audit trail |
| 🌐 Web UI | 6-panel dark workbench (Chat, Agent, RAG, Models, Security, Logs) |

Full architecture: [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md)

---

## Build Status

**24 phase-gated milestones** — nothing is faked to look finished.
Live status: [`docs/BUILD_STATUS.md`](docs/BUILD_STATUS.md).

✅ **Phases 0–22 complete** | 🔄 Phases 23–24 in progress

---

## Quick Start (Windows 11)

### 1. Setup
```powershell
git clone <this-repo-url>
cd sovereign-ai-workbench
.\scripts\setup_windows.ps1
```

### 2. Verify GPU
```powershell
python scripts\verify_gpu.py
# Expected: CUDA available: True, RTX 4050, ~6 GB VRAM
```

### 3. Try the Core Pipeline (small demo)
```powershell
python scripts\prepare_data.py              # Build training data from demo corpus
python scripts\train_tokenizer.py           # Train from-scratch BPE tokenizer
python scripts\train_model.py --experiment exp1  # 50-step smoke test
python scripts\run_inference.py --checkpoint checkpoints/best.pt --prompt "The pump inspection"
```

### 4. Full Training Pipeline (90K examples)
```powershell
# One-command full pipeline:
python scripts\train_pipeline.py --n 90000 --device cuda

# Or step-by-step:
python scripts\prepare_openorca.py --n 90000
python scripts\train_tokenizer.py --train-file data/processed/openorca_pretrain.txt --output tokenizer/vocab/openorca_bpe_vocab.json --vocab-size 1000 --max-lines 10000 --instruction-format
python scripts\train_model.py --train-file data/processed/openorca_pretrain.txt --vocab-path tokenizer/vocab/openorca_bpe_vocab.json --max-steps 500 --device cuda
python scripts\finetune_instruct.py --checkpoint checkpoints/best.pt --max-steps 11250 --batch-size 16 --device cuda
```

### 5. Launch the Workbench
```powershell
python app\backend\app.py
# Open http://localhost:5000 in your browser
```

### 6. Run Evaluation
```powershell
python scripts\evaluate.py --checkpoint checkpoints/best.pt
```

### 7. Run SIH Demo Scenarios
```powershell
python demo\run_demo.py --list       # See all 5 scenarios
python demo\run_demo.py --scenario 4 # Engineering calculation demo
python demo\run_demo.py --all        # Run all demos
```

---

## Architecture Overview

```
User request
    │
    ▼
Task Router (rule-based classification)
    │
    ├──► Custom MiniLLM (from-scratch Transformer, ~5M params)
    ├──► Open-weight model (Phi-3 / Mistral via Ollama)
    ├──► Vision model (llava:7b for image understanding)
    └──► RAG retriever (local vector index + citations)
    │
    ▼
Agent (Planner → Executor → Verifier)
    │
    ├──► Tools: Calculator, Filesystem, PDF/OCR, Code Sandbox,
    │          Word/Excel/PowerPoint generation
    │
    ▼
Local Web UI (Flask + dark-mode workbench)
    │
    ▼
SQLite + Audit Log (full offline proof)
```

---

## Model Architecture (Custom MiniLLM)

| Component | Implementation |
|-----------|---------------|
| Tokenizer | From-scratch byte-level BPE (no tokenizer library) |
| Embeddings | `nn.Embedding` lookup (PyTorch primitive allowed) |
| Positional encoding | Rotary Position Embedding (RoPE) — hand-written |
| Normalization | RMS Norm — hand-written (not `nn.LayerNorm`) |
| Attention | Causal multi-head self-attention — hand-written |
| Feed-forward | SwiGLU FFN — hand-written |
| Training loop | Full autoregressive (AdamW, cosine warmup, AMP, grad accum) |
| Weight tying | LM head shares weights with embedding layer |

**Small config:** d_model=256, 6 layers, 4 heads, ~4.88M params
**Medium config:** d_model=384, 8 layers, 6 heads, ~12M params

---

## Project Structure

```
sovereign-ai-workbench/
├── models/custom_minilm/  # From-scratch Transformer (Phases 4-7)
├── models/adapters/       # ModelProvider interface + adapters (Phase 8)
├── tokenizer/             # Byte-level BPE tokenizer (Phase 3)
├── training/              # Dataset, Trainer, scheduler (Phase 6)
├── router/                # Task classifier + routing policies (Phase 9)
├── agents/                # Planner/Executor/Verifier/Memory (Phase 11)
├── tools/                 # Calculator, FS, PDF, OCR, Sandbox, etc. (Phase 10)
├── rag/                   # Chunking, embedding, indexing, retrieval (Phase 12)
├── security/              # Network monitor, audit log, permissions (Phase 18)
├── database/              # SQLite schema + repositories (Phase 17)
├── evaluation/            # Metrics: perplexity, BLEU, ROUGE, routing (Phase 22)
├── app/backend/           # Flask API backend (Phase 19)
├── app/frontend/          # Dark-mode web workbench (Phase 20)
├── demo/                  # SIH demo scenarios A-E (Phase 23)
├── scripts/               # CLI tools for training, inference, evaluation
├── docs/                  # Architecture, build status, audit docs
└── tests/                 # Unit tests + end-to-end integration tests
```

---

## Testing

```powershell
python -m pytest tests/ -v                    # All tests
python -m pytest tokenizer/tests -v           # Tokenizer suite (14 tests)
python -m pytest tests/test_phase5.py -v      # Model architecture (17 tests)
python -m pytest tests/test_e2e.py -v         # End-to-end integration
python scripts/evaluate.py                    # Evaluation metrics
```

---

## License

MIT — see [`LICENSE`](LICENSE).
