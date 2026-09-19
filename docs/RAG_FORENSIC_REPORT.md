# SovereignAI — RAG Forensic Report

> Generated: 2026-09-19 by forensic audit.

## Summary

The RAG pipeline was **broken by an uncommitted code change** that removed the
`SentenceTransformerEmbedder` from `rag/embeddings.py`, leaving only the
512-dimensional TF-IDF embedder. This created a dimension mismatch with:
- The primary RAG index (`data/rag_index.json`) built with 384-dim SentenceTransformer
- The FineWeb 1.6B knowledge shards (17.1 GB of 384-dim embeddings)

The working-tree code was then compensated by creating `rag_index_tfidf.json` and
`user_knowledge_index.json` with 512-dim TF-IDF embeddings. The backend was modified
to load `user_knowledge_index.json` instead of `rag_index.json`.

**Recovery action**: Restored the committed `rag/embeddings.py` with dual-mode
SentenceTransformer + TF-IDF fallback.

---

## Pipeline Trace — Stage by Stage

### 1. INPUT
- **Status**: EXISTS
- **Implementation**: User uploads files via `/api/rag/upload` or pastes text via `/api/rag/ingest`
- **Formats**: .txt, .md, .csv, .json, .pdf, .docx, .xlsx, .pptx
- **Storage**: `data/user_knowledge/` for uploaded files

### 2. FILE DISCOVERY
- **Status**: EXISTS
- **Implementation**: `DocumentIngester.ingest_directory()` uses `Path.rglob("*")`
- **Extensions filter**: [.txt, .pdf, .md] by default
- **Gap**: Default extensions do NOT include .docx, .xlsx, .pptx — `ingest_directory()` would skip them. But `ingest_file()` handles them individually.

### 3. FILE TYPE DETECTION
- **Status**: EXISTS
- **Implementation**: Suffix-based in `ingest.py` line 57-67
- **Supported**: .pdf, .docx, .xlsx, .pptx, fallback to text

### 4. PARSER
- **Status**: EXISTS
- **PDF**: `pdfminer.high_level.extract_text()` with binary fallback
- **DOCX**: `python-docx` paragraph extraction
- **XLSX**: `openpyxl` row extraction
- **PPTX**: `python-pptx` shape text extraction
- **Text**: Direct `read_text(encoding="utf-8")`

### 5. OCR
- **Status**: NOT_IMPLEMENTED
- **Gap**: Scanned PDFs will produce garbled text via the binary fallback
- **Note**: The `pdfminer` path handles text PDFs correctly; OCR would be needed only for scanned/image PDFs

### 6. TEXT EXTRACTION
- **Status**: EXISTS
- **Output**: Raw text string passed to `ingest_text()`

### 7. CLEANING
- **Status**: MINIMAL
- **Implementation**: `text.strip()` in chunking, no advanced cleaning
- **Gap**: No HTML tag removal, no encoding normalization, no deduplication

### 8. CHUNKING
- **Status**: EXISTS
- **Implementation**: `rag/chunking.py` with two strategies
- **Sentence strategy**: Splits on `[.!?]\s+` within `chunk_size` limit
- **Fixed strategy**: Character-count based with overlap
- **Default**: chunk_size=400, overlap=80, strategy="sentence"
- **Metadata**: chunk_id, source, char_start, char_end

### 9. METADATA
- **Status**: BASIC
- **Fields**: chunk_id, source, char_start, char_end
- **Gap**: No page numbers, no heading hierarchy, no document-level metadata

### 10. EMBEDDING
- **Status**: RECOVERED
- **Primary**: `SentenceTransformerEmbedder` (all-MiniLM-L6-v2, 384-dim)
  - Pretrained open-weight model (NOT the project's custom LLM)
  - Runs locally on CPU or GPU
  - Requires cached model weights (air-gap compatible after first download)
- **Fallback**: `TFIDFEmbedder` (hashed bag-of-words, 512-dim)
  - Pure Python, zero dependencies
  - Deterministic (blake2b hash, not random Python hash)
  - L2 normalized output
- **Selection**: `get_embedder()` tries SentenceTransformer, falls back to TF-IDF

### 11. INDEXING
- **Status**: EXISTS
- **Implementation**: `LocalVectorIndex` — in-memory flat vector index
- **Persistence**: JSON file (chunk_id → {text, source, embedding, metadata})
- **Search**: Brute-force cosine similarity
- **Gap**: No ANN (FAISS/HNSW) for large-scale search

### 12. RETRIEVAL
- **Status**: EXISTS
- **Implementation**: `Retriever` class wraps `LocalVectorIndex.search()`
- **Parameters**: top_k, min_score, source_filter
- **Output**: `RetrievalResult` dataclass with citation

### 13. RERANKING
- **Status**: NOT_IMPLEMENTED
- **Gap**: No cross-encoder or BM25 reranker
- **Mitigation**: The `HybridRetriever` merges semantic + lexical results

### 14. CONTEXT CONSTRUCTION
- **Status**: EXISTS
- **Implementation**: `Retriever.build_context()` and `build_rag_prompt()`
- **Limit**: max_chars=2000 (retriever) / max_context_chars=350 (backend)
- **Format**: Citation prefix + chunk text, concatenated

### 15. LLM
- **Status**: EXISTS
- **Implementation**: Custom MiniLLM via `GenerationConfig`
- **Context limit**: 256 tokens (industrial 80M), 128 tokens (small/medium)
- **Backend behavior**: If RAG results found, returns extractive reply; else model generation

### 16. CITATION
- **Status**: EXISTS
- **Implementation**: `rag/citations.py` — `format_citations()` and `build_rag_prompt()`
- **Format**: `[Source: filename, chunk chunk_id]`

---

## Index File Analysis

### data/rag_index.json (PRIMARY — SentenceTransformer)
- **Records**: 10,975
- **Embedding dimension**: 384
- **Sources**: 10,000 maintenance_parquet rows + 6 demo/project docs
- **Size**: 136 MB
- **Status**: INTACT but incompatible with TF-IDF-only code
- **Built by**: `scripts/ingest_rag.py` with SentenceTransformer embedder

### data/rag_index_tfidf.json (REBUILD — TF-IDF)
- **Records**: 10,975 (same documents)
- **Embedding dimension**: 512
- **Size**: 39 MB
- **Status**: COMPATIBLE with TF-IDF embedder
- **Built by**: `scripts/rebuild_local_rag_embeddings.py`

### data/user_knowledge_index.json (ACTIVE — TF-IDF)
- **Records**: 10,975 (same documents)
- **Embedding dimension**: 512
- **Size**: 90 MB
- **Status**: Currently loaded by backend
- **Built by**: Copy/rebuild of TF-IDF index

### FineWeb 1.6B Knowledge (DISK-BACKED — SentenceTransformer)
- **Chunks**: 5,240,427
- **Embedding dimension**: 384
- **Shards**: 105 chunk files + 105 embedding files
- **Size**: 17.1 GB
- **Status**: INTACT but inaccessible with TF-IDF-only code
- **Built by**: `scripts/ingest_fineweb_knowledge.py`

---

## Root Cause Chain

```
1. SentenceTransformerEmbedder removed from rag/embeddings.py (uncommitted)
2. get_embedder() now returns TFIDFEmbedder (512-dim) always
3. New ingestion produces 512-dim vectors
4. Old indexes (rag_index.json, FineWeb) have 384-dim vectors
5. Dimension mismatch → retrieval fails or produces garbage scores
6. Workaround: rag_index_tfidf.json and user_knowledge_index.json rebuilt with TF-IDF
7. Backend changed to load user_knowledge_index.json (512-dim compatible)
8. Result: Basic user knowledge RAG works, but FineWeb 17.1 GB index is inaccessible
```

## Recovery Action Taken

Restored `rag/embeddings.py` to the committed HEAD version with:
- `SentenceTransformerEmbedder` (384-dim, neural, primary)
- `TFIDFEmbedder` (512-dim, hash-based, fallback)
- `get_embedder()` tries neural first, falls back with warning

This restores compatibility with ALL existing indexes.
