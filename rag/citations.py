"""rag/citations.py — Format citation strings for RAG answers."""
from __future__ import annotations

from rag.retriever import RetrievalResult


def format_citations(results: list[RetrievalResult]) -> str:
    """Return a formatted bibliography-style citation block."""
    if not results:
        return ""
    lines = ["\n**Sources:**"]
    for i, r in enumerate(results, 1):
        lines.append(f"  [{i}] {r.source} (relevance: {r.score:.2f})")
    return "\n".join(lines)


def build_rag_prompt(
    query: str,
    results: list[RetrievalResult],
    system: str = "",
    max_context_chars: int = 2000,
) -> str:
    """
    Build a RAG-augmented prompt for the small MiniLLM context window.

    Keeps the prefix minimal (no verbose system message) so that the
    retrieved context + question fit within the model's 128-token limit.
    Uses the same Q:/A: format that the adapter's instruction prompt uses.
    """
    # Build context string (without needing a Retriever instance)
    parts = []
    total = 0
    for r in results:
        snippet = f"[{r.source}]: {r.text}"
        if total + len(snippet) > max_context_chars:
            break
        parts.append(snippet)
        total += len(snippet)
    context = "\n".join(parts)

    # Keep it tight: context block + Q/A marker only
    return (
        f"Context:\n{context}\n\n"
        f"Q: {query}\n"
        f"A:"
    )
