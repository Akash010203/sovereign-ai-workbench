"""rag/retriever.py — Query the vector index and return ranked results with citations."""
from __future__ import annotations

import logging
from dataclasses import dataclass, field

from rag.embeddings import get_embedder
from rag.index import LocalVectorIndex

log = logging.getLogger(__name__)


@dataclass
class RetrievalResult:
    """One retrieved chunk with citation information."""
    chunk_id:  str
    source:    str
    text:      str
    score:     float
    metadata:  dict = field(default_factory=dict)

    def citation(self) -> str:
        """Return a short citation string for this result."""
        return f"[Source: {self.source}, chunk {self.chunk_id}]"


class Retriever:
    """
    Query the local vector index and return relevant chunks.

    Used by the agent's rag_search tool and the RAG pipeline.
    """

    def __init__(self, index: LocalVectorIndex) -> None:
        self.index    = index
        self.embedder = get_embedder()

    def retrieve(
        self,
        query: str,
        top_k: int = 5,
        min_score: float = 0.0,
        source_filter: str | None = None,
    ) -> list[RetrievalResult]:
        """
        Retrieve the top-k most relevant chunks for ``query``.

        Args:
            query:         The user's question or search query.
            top_k:         Maximum results to return.
            min_score:     Minimum cosine similarity to include.
            source_filter: Optional: only return results from this source.

        Returns:
            List of RetrievalResult sorted by descending similarity.
        """
        if len(self.index) == 0:
            log.warning("RAG index is empty — no results returned.")
            return []

        query_emb = self.embedder.embed(query)
        raw = self.index.search(query_emb, top_k=top_k, source_filter=source_filter)

        results = [
            RetrievalResult(
                chunk_id=r["chunk_id"],
                source=r["source"],
                text=r["text"],
                score=r["score"],
                metadata=r.get("metadata", {}),
            )
            for r in raw
            if r["score"] >= min_score
        ]
        log.info("Retrieved %d chunks for query='%s'", len(results), query[:60])
        return results

    def build_context(
        self, results: list[RetrievalResult], max_chars: int = 2000
    ) -> str:
        """
        Concatenate retrieved chunks into a context string for the model.

        Each chunk is prefixed with its citation.  Total length is capped
        at max_chars to fit within model context windows.
        """
        parts = []
        total = 0
        for r in results:
            citation = r.citation()
            snippet  = f"{citation}\n{r.text}"
            if total + len(snippet) > max_chars:
                break
            parts.append(snippet)
            total += len(snippet)
        return "\n\n".join(parts)
