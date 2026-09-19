# SovereignAI — System Map

## Data Flow Architecture

```
USER INPUT (text/file)
    │
    ▼
┌─────────────────────┐
│   Frontend (HTML)    │  app/frontend/index.html
│   localhost:5000     │
└─────────┬───────────┘
          │ HTTP POST /api/chat or /api/agent
          ▼
┌─────────────────────┐
│  Flask Backend       │  app/backend/app.py
│  localhost:5000      │
└─────────┬───────────┘
          │
    ┌─────┴─────┐
    │           │
    ▼           ▼
┌────────┐  ┌────────────┐
│ Router │  │ Agent      │
│ (rules)│  │ (orchestr) │
└───┬────┘  └──┬─────────┘
    │          │
    ▼          ▼
┌─────────────────────┐
│  Model Registry     │  models/registry.py
│  └─ MiniLLM         │  models/custom_minilm/
└─────────┬───────────┘
          │
          ▼
┌─────────────────────┐
│  Tool Registry      │  tools/
│  ├─ Calculator      │  tools/calculator.py
│  ├─ Code Sandbox    │  tools/code_sandbox.py
│  ├─ OCR (Tesseract) │  tools/ocr.py
│  ├─ RAG Search      │  tools/rag_search.py
│  ├─ Word Writer     │  tools/word.py
│  ├─ Excel Writer    │  tools/spreadsheet.py
│  ├─ PowerPoint      │  tools/powerpoint.py
│  ├─ PDF Reader      │  tools/pdf.py
│  └─ Filesystem      │  tools/filesystem.py
└─────────┬───────────┘
          │
          ▼
┌─────────────────────┐
│  RAG System         │  rag/
│  ├─ Embeddings      │  rag/embeddings.py (custom TF-IDF)
│  ├─ Vector Index    │  rag/index.py
│  ├─ Chunking        │  rag/chunking.py
│  ├─ Retriever       │  rag/retriever.py
│  ├─ Citations       │  rag/citations.py
│  └─ FineWeb Index   │  rag/fineweb_index.py
└─────────────────────┘
          │
          ▼
┌─────────────────────┐
│  Database (SQLite)  │  database/db.py
│  ├─ Conversations   │  database/repositories/conversations.py
│  └─ Schema          │  database/schema.sql
└─────────────────────┘
          │
          ▼
┌─────────────────────┐
│  Security           │  security/
│  ├─ Network Monitor │  security/network_monitor.py
│  ├─ Offline Mode    │  security/offline_mode.py
│  ├─ Audit Logger    │  security/audit.py
│  └─ Permissions     │  security/permissions.py
└─────────────────────┘
          │
          ▼
     OUTPUT (text, files, documents)
```

## Component Dependencies

| Component | Depends On | Internet Required |
|---|---|---|
| Tokenizer | None | No |
| MiniLLM Model | Tokenizer, PyTorch | No |
| Training | Model, Tokenizer, Dataset | No (after data download) |
| Router | Model Registry | No |
| Agent | Router, Tools, Model | No |
| RAG Embeddings | Custom TF-IDF | No |
| RAG Index | Embeddings | No |
| OCR | Tesseract binary | No |
| Document Gen | python-docx/openpyxl/python-pptx | No |
| Code Sandbox | subprocess, Python | No |
| Database | sqlite3 (stdlib) | No |
| Backend | Flask | No |
| Frontend | Browser | No |
| Security | socket (stdlib) | No |

## Checkpoint Lineage

```
exp1_smoke (step 1-50)      → 5M params, vocab=600, demo corpus
exp4_openorca (step 100-30000) → 12M params, vocab=8000, OpenOrca
industrial_80m (step 5-20000)  → 83M params, vocab=16000, blended corpus
best.pt                      → symlink to best industrial_80m checkpoint
```
