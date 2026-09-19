# SovereignAI — Data Flow

> Generated: 2026-09-19 by forensic audit.

## CRITICAL DISTINCTION

Training data and RAG data are **separate pipelines** with **separate purposes**.
They must NEVER be silently mixed.

---

## Training Data Flow (Pipeline B — Model Weight Training)

```
HuggingFace Datasets (one-time download, then offline)
│
├── FineWeb-Edu (HuggingFaceFW/fineweb-edu)
├── OpenOrca (Open-Orca/OpenOrca)
├── Nemotron
└── Maintenance domain dataset
        │
        ▼
    data/raw/hf_sources/           ← Raw downloaded cache
        │
        ▼
    scripts/prepare_blended_corpus.py
    scripts/prepare_openorca.py
        │
        ├── Cleaning
        ├── Deduplication
        ├── Quality filtering
        └── Train/validation split
        │
        ▼
    data/processed/
    ├── blended_train.txt          ← 3.29 GB (~1.6B tokens)
    ├── blended_val.txt            ← 66 MB
    ├── openorca_instruct.jsonl    ← 416 MB
    ├── openorca_pretrain.txt      ← 490 MB
    └── openorca_val.txt           ← 9.9 MB
        │
        ▼
    tokenizer/vocab/blended_16k.json   ← 16K BPE tokenizer
        │
        ▼
    training/dataset.py            ← Tokenize to IDs, create batches
        │
        ▼
    training/trainer.py            ← Train MiniLLM model
        │
        ▼
    checkpoints/
    ├── industrial_80m_best.pt     ← Best model weights
    ├── industrial_80m_*_step*.pt  ← Intermediate checkpoints
    └── exp4_openorca_*.pt         ← Earlier experiments
```

**Output**: Trained model weights (`.pt` files)
**Purpose**: Teaching the custom LLM to predict next tokens
**Size**: ~102 GB of checkpoints from multiple experiment series

---

## RAG Data Flow (Pipeline A — Knowledge Retrieval)

### User Knowledge (Small, Interactive)

```
    User uploads documents via frontend
    ├── .txt, .md, .csv, .json
    ├── .pdf  → pdfminer extraction
    ├── .docx → python-docx extraction
    ├── .xlsx → openpyxl extraction
    └── .pptx → python-pptx extraction
        │
        ▼
    data/user_knowledge/           ← Uploaded file storage
        │
        ▼
    rag/ingest.py (DocumentIngester)
        │
        ├── Text extraction
        ├── rag/chunking.py        ← 400-char chunks, 80-char overlap
        └── rag/embeddings.py      ← TF-IDF or SentenceTransformer
        │
        ▼
    rag/index.py (LocalVectorIndex)
        │
        ▼
    data/user_knowledge_index.json ← JSON-persisted vector index
        │
        ▼
    rag/retriever.py               ← Cosine similarity search
        │
        ▼
    rag/citations.py               ← Format citations
        │
        ▼
    app/backend/app.py             ← Build RAG prompt, feed to model
```

### Bundled Demo Knowledge

```
    data/demo/
    ├── compressor_seal_replacement_sop.txt
    ├── equipment_inspection_report.txt
    ├── maintenance_log_scan.txt
    └── pressure_sensor_data.csv
        │
        ▼
    scripts/ingest_rag.py          ← Batch ingestion script
        │
        ▼
    data/rag_index.json            ← 10,975 chunks (384-dim, SentenceTransformer)
    data/rag_index_tfidf.json      ← 10,975 chunks (512-dim, TF-IDF)
```

### FineWeb Knowledge Store (Large, Precomputed)

```
    HuggingFaceFW/fineweb-edu      ← Streamed from HuggingFace (one-time)
        │
        ▼
    scripts/ingest_fineweb_knowledge.py
        │
        ├── Stream ~1.3M documents
        ├── Chunk to 384-token pieces
        ├── Embed with SentenceTransformer (384-dim)
        └── Shard to 50,000-chunk files
        │
        ▼
    data/knowledge/fineweb_edu_1p6b_embed/
    ├── chunks-00000..00104.jsonl   ← Text chunks (JSONL)
    ├── embeddings-00000..00104.f32 ← Float32 embedding vectors
    └── fineweb_edu_manifest.json   ← 5,240,427 chunks, 384-dim
        │
        ▼
    rag/fineweb_index.py (FineWebDiskIndex)
        │
        ├── Memory-mapped vector search
        ├── Bounded RAM (one shard at a time)
        └── Exact cosine/inner-product search
        │
        ▼
    rag/hybrid.py (HybridRetriever)
        │
        ├── Local index results
        ├── FineWeb semantic results
        └── FineWeb lexical results
        │
        ▼
    Merged, sorted by score → top-k results
```

---

## What MUST NOT Happen

| ❌ Anti-Pattern | Why |
|---|---|
| Insert 1.6B training tokens into RAG vector DB | OOM, wrong purpose, corrupts retrieval |
| Use RAG demo documents as training data | Cross-contamination, test leakage |
| Move `data/processed/blended_train.txt` into `data/knowledge/` | Confuses training with retrieval |
| Delete FineWeb knowledge shards to "save space" | Destroys 17 GB of precomputed embeddings |
| Rebuild FineWeb index with TF-IDF embedder | Incompatible with existing 384-dim shards |

## What IS Correct

| ✅ Pattern | Current State |
|---|---|
| FineWeb-Edu used for BOTH training AND RAG (separate pipelines) | Correct — training via `blended_train.txt`, RAG via `fineweb_edu_1p6b_embed/` |
| RAG index dimension matches embedder dimension | **BROKEN** — current TF-IDF (512) ≠ existing indexes (384) |
| User knowledge separated from bundled knowledge | Correct — `user_knowledge_index.json` vs `rag_index.json` |
| Training checkpoints gitignored | Correct |
| Training data gitignored | Correct |
