"""
rag/index.py — Local vector index (from scratch, no cloud vector DB).

WHAT IS A VECTOR INDEX?
------------------------
A vector index stores (chunk_id, embedding_vector) pairs and answers
"nearest neighbor" queries: given a query vector, return the K stored
vectors most similar to it.

SIMILARITY METRIC
-----------------
We use cosine similarity:
    cosine(a, b) = (a · b) / (|a| × |b|)

When both vectors are L2-normalized (which our embedder does), this is
equivalent to the dot product — making the computation cheap.

IMPLEMENTATION
--------------
This is a pure-Python in-memory index using a flat (brute-force) search.
For the demo corpus size (dozens to a few hundred documents), brute-force
is perfectly fast.  A FAISS or HNSW index could replace this for
production without changing the rest of the code.

HONESTY NOTE: This is NOT a cloud vector database (Pinecone, Weaviate,
Chroma, etc.).  It is a simple local file-backed dictionary.

PERSISTENCE
-----------
The index is saved to a JSON file and reloaded on startup.  No cloud,
no external services.
"""
from __future__ import annotations

import json
import logging
import math
from pathlib import Path
from typing import Optional

from rag.chunking import TextChunk

log = logging.getLogger(__name__)


def _cosine(a: list[float], b: list[float]) -> float:
    """Cosine similarity, with an explicit guard for incompatible indices."""
    if len(a) != len(b):
        raise ValueError(f"Embedding dimension mismatch: query={len(a)}, record={len(b)}")
    a_norm = math.sqrt(sum(x * x for x in a))
    b_norm = math.sqrt(sum(y * y for y in b))
    if not a_norm or not b_norm:
        return 0.0
    return sum(x * y for x, y in zip(a, b)) / (a_norm * b_norm)


class LocalVectorIndex:
    """
    In-memory flat vector index with JSON persistence.

    Stores:
        chunk_id → {text, source, embedding, metadata}
    """

    def __init__(self) -> None:
        self._store: dict[str, dict] = {}   # chunk_id → record

    # ── Indexing ───────────────────────────────────────────────────────

    def add(self, chunk: TextChunk, embedding: list[float]) -> None:
        """Add or update one chunk with its embedding vector."""
        self._store[chunk.chunk_id] = {
            "chunk_id":  chunk.chunk_id,
            "source":    chunk.source,
            "text":      chunk.text,
            "embedding": embedding,
            "metadata":  chunk.metadata,
        }

    def add_batch(
        self, chunks: list[TextChunk], embeddings: list[list[float]]
    ) -> None:
        for chunk, emb in zip(chunks, embeddings):
            self.add(chunk, emb)

    # ── Retrieval ──────────────────────────────────────────────────────

    def search(
        self,
        query_embedding: list[float],
        top_k: int = 5,
        source_filter: Optional[str] = None,
    ) -> list[dict]:
        """
        Return the top-k most similar chunks to the query embedding.

        Args:
            query_embedding: The embedded query vector.
            top_k:           Number of results to return.
            source_filter:   If set, only return chunks from this source.

        Returns:
            List of dicts sorted by similarity (descending), each containing
            chunk_id, source, text, score, metadata.
        """
        candidates = []
        for record in self._store.values():
            if source_filter and record["source"] != source_filter:
                continue
            try:
                score = _cosine(query_embedding, record["embedding"])
            except ValueError as exc:
                log.warning("Skipping incompatible RAG record %s: %s", record["chunk_id"], exc)
                continue
            candidates.append((score, record))

        candidates.sort(key=lambda x: x[0], reverse=True)
        return [
            {**rec, "score": round(score, 4)}
            for score, rec in candidates[:top_k]
        ]

    # ── Persistence ────────────────────────────────────────────────────

    def save(self, path: str | Path) -> None:
        """Persist the index to a JSON file."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps({"records": list(self._store.values())}, indent=2),
            encoding="utf-8",
        )
        log.info("Index saved: %d chunks → %s", len(self._store), path)

    def load(self, path: str | Path) -> None:
        """Load a previously saved index."""
        path = Path(path)
        if not path.exists():
            log.warning("Index file not found: %s — starting empty.", path)
            return
        data = json.loads(path.read_text(encoding="utf-8"))
        for rec in data.get("records", []):
            self._store[rec["chunk_id"]] = rec
        log.info("Index loaded: %d chunks from %s", len(self._store), path)

    def __len__(self) -> int:
        return len(self._store)
