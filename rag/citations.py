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

    Keeps the prefix minimal so retrieved text + question fit in the small
    local model's context. The adapter supplies the final chat role markers.
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

    # Keep it tight: context block + plain question only.
    return (
        f"Context:\n{context}\n\n"
        f"Question: {query}\n"
        "Answer from the context only:"
    )
