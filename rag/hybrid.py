"""Combine the small editable local RAG index with the FineWeb knowledge store."""
from __future__ import annotations

from rag.retriever import RetrievalResult, Retriever
from rag.fineweb_index import FineWebRetriever


class HybridRetriever:
    def __init__(self, local: Retriever, fineweb: FineWebRetriever | None = None) -> None:
        self.local = local
        self.fineweb = fineweb

    def retrieve(self, query: str, top_k: int = 5, min_score: float = 0.0, source_filter: str | None = None) -> list[RetrievalResult]:
        results = self.local.retrieve(query, top_k=top_k, min_score=min_score, source_filter=source_filter)
        if self.fineweb is not None and not source_filter:
            results.extend(self.fineweb.retrieve(query, top_k=top_k, min_score=min_score))
        return sorted(results, key=lambda item: item.score, reverse=True)[:top_k]

    def build_context(self, results: list[RetrievalResult], max_chars: int = 2000) -> str:
        return self.local.build_context(results, max_chars=max_chars)
