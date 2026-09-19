# SovereignAI — RAG Configuration

> Generated: 2026-09-19 by forensic audit.

## Current Configuration

| Parameter | Value | Location |
|---|---|---|
| `chunk_size` | 400 characters | `rag/ingest.py:31`, `rag/chunking.py:41` |
| `chunk_overlap` | 80 characters | `rag/ingest.py:32`, `rag/chunking.py:42` |
| `chunking_strategy` | "sentence" | `rag/chunking.py:43` |
| `embedding_model` (primary) | all-MiniLM-L6-v2 | `rag/embeddings.py:SentenceTransformerEmbedder` |
| `embedding_dimension` (primary) | 384 | SentenceTransformer output |
| `embedding_model` (fallback) | TF-IDF hash | `rag/embeddings.py:TFIDFEmbedder` |
| `embedding_dimension` (fallback) | 512 | `TFIDFEmbedder.__init__` |
| `distance_metric` | Cosine similarity | `rag/index.py:_cosine` |
| `top_k` (retrieval) | 5 | `rag/retriever.py:41` |
| `top_k` (backend chat) | 2 | `app/backend/app.py:262` |
| `min_score` (backend chat) | 0.20 | `app/backend/app.py:262` |
| `max_context_chars` (retriever) | 2000 | `rag/retriever.py:79` |
| `max_context_chars` (backend) | 350 | `app/backend/app.py:271` |
| `reranker` | None | Not implemented |
| `query_rewriting` | None | Not implemented |
| `hybrid_search` | Yes | `rag/hybrid.py:HybridRetriever` |
| `BM25` | No | Lexical search uses TF-IDF, not BM25 |
| `metadata_filters` | source_filter only | `rag/index.py:93` |

## FineWeb Knowledge Configuration

| Parameter | Value |
|---|---|
| `target_tokens` | 1,600,000,000 |
| `chunk_tokens` | 384 |
| `overlap_tokens` | 48 |
| `chars_per_token` | 4.0 |
| `min_document_chars` | 200 |
| `shard_size` | 50,000 chunks per file |
| `embedding_dimension` | 384 (SentenceTransformer) |
| `total_chunks` | 5,240,427 |
| `source_documents` | 1,319,405 (from 1,319,408 seen) |

## Index Configuration

| Index | Type | Persistence | Search Method |
|---|---|---|---|
| LocalVectorIndex | In-memory flat | JSON file | Brute-force cosine |
| FineWebDiskIndex | Disk-backed sharded | F32 binary + JSONL | Memory-mapped exact search |

## Backend RAG Behavior

1. On `/api/chat`: If `use_rag` is not explicitly false AND index has chunks:
   - Embed query → search user knowledge index (top_k=2, min_score=0.20)
   - Filter results with `_has_local_evidence()` (checks for subject-word overlap)
   - If results found: return extractive reply (no model generation)
   - If no results: generate with custom MiniLLM
2. On `/api/rag/search`: Direct retrieval with configurable top_k
3. On `/api/rag/ingest`: Text → chunks → embed → add to index → save
4. On `/api/rag/upload`: File → extract → ingest pipeline → save
