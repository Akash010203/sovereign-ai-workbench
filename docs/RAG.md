# Retrieval-Augmented Generation (RAG) & Semantic Search

The **Sovereign AI Workbench** includes an entirely local, self-contained **Retrieval-Augmented Generation (RAG)** pipeline designed to ingest, index, and retrieve enterprise documents without sending data to third-party vector databases or external embedding APIs.

---

## 1. Why RAG? (Dynamic Knowledge vs. Model Retraining)

A fundamental architectural principle of this workbench is:
$$\text{Model Weights} \ne \text{Knowledge Base}$$

- **Model weights** store language synthesis, grammar, and reasoning patterns.
- **RAG index** stores dynamic, fast-changing documents (SOPs, equipment manuals, inspection logs).

When a new manual is added, the system **does not retrain the LLM**. Instead, it indexes the document in seconds, making it immediately retrievable.

---

## 2. RAG Architecture & Modules

The RAG subsystem is organized under `rag/`:

```
rag/
├── ingest.py           # Ingests PDF, DOCX, TXT, and Markdown files
├── chunking.py         # Semantic & sliding-window text chunking with token overlap
├── embeddings.py       # Local vector embedding generator (TF-IDF & dense)
├── index.py            # Vector index storage with fast cosine similarity search
├── retriever.py        # Top-K relevance retrieval & reciprocal rank fusion
├── citations.py        # Source attribution and paragraph reference tracking
├── fineweb.py          # FineWeb-Edu dataset chunk reader and streaming iterator
├── fineweb_index.py    # Offline index loader for 1.6B pre-indexed embeddings
└── hybrid.py           # Hybrid fusion combining user document RAG with FineWeb-Edu
```

---

## 3. End-to-End Document Lifecycle

```
Local Document (PDF / DOCX / TXT)
              │
              ▼
    Document Ingestion (`ingest.py`)
              │
              ▼
   Text Cleaning & Normalization
              │
              ▼
    Chunking with Overlap (`chunking.py`)
              │
              ▼
     Vector Embeddings (`embeddings.py`)
              │
      ┌───────┴────────┐
      ▼                ▼
 Vector Index     SQL Database
 (Embeddings)      (Document Metadata & Chunks)
```

---

## 4. Synergy with SQL Database

Semantic vector search and structured SQL queries complement each other:

| Feature | Vector Retrieval (`rag/index.py`) | Relational Database (`database/`) |
|---|---|---|
| **Query Type** | "What are the vibration limits for Pump P-101?" | `SELECT * FROM chunks WHERE doc_id = 42;` |
| **Strengths** | Semantic similarity, conceptual matching | Exact filtering, dates, statuses, IDs |
| **Output** | Top-K text passages with similarity scores | Deterministic, structured tabular records |

When a query is processed, vector search identifies relevant semantic chunk IDs, while SQL verifies document access permissions, timestamps, and metadata before building the final prompt context.
