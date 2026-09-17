"""
tools/rag_search.py — RAG search tool for the agent executor.

Wraps the local RAG Retriever as a BaseTool so the agent can call
`rag_search` through the ToolRegistry — the same way it calls
`calculator` or `code_sandbox`.

This bridge is necessary because the planner generates steps with
tool_name="rag_search", and the executor resolves tools via
ToolRegistry.call().
"""
from __future__ import annotations

from typing import Optional

from tools.registry import BaseTool, ToolResult


class RagSearchTool(BaseTool):
    """
    Search the local RAG knowledge base for relevant document chunks.

    Uses the project's custom vector index (no cloud vector DB) and
    returns ranked results with source citations.
    """

    def __init__(self, retriever) -> None:
        """
        Args:
            retriever: An instance of rag.retriever.Retriever.
                       Accepts Any to avoid circular import at module level.
        """
        self._retriever = retriever

    @property
    def name(self) -> str:
        return "rag_search"

    @property
    def description(self) -> str:
        return (
            "Search the local knowledge base for relevant documents. "
            "Args: query (str), top_k (int, default 5). "
            "Returns ranked text chunks with source citations."
        )

    def run(
        self,
        query: str = "",
        top_k: int = 5,
        min_score: float = 0.0,
        source_filter: Optional[str] = None,
    ) -> ToolResult:
        if not query.strip():
            return ToolResult(
                tool_name=self.name, success=False, output=None,
                error="Empty query — provide a search question.",
            )

        try:
            results = self._retriever.retrieve(
                query=query,
                top_k=top_k,
                min_score=min_score,
                source_filter=source_filter,
            )
        except Exception as exc:
            return ToolResult(
                tool_name=self.name, success=False, output=None,
                error=f"RAG search failed: {exc}",
            )

        if not results:
            return ToolResult(
                tool_name=self.name, success=True,
                output="No relevant documents found in the knowledge base.",
                metadata={"query": query, "results_count": 0},
            )

        # Format results as structured text the agent can use
        formatted_parts = []
        for i, r in enumerate(results, 1):
            citation = r.citation()
            formatted_parts.append(
                f"Result {i} {citation} (score: {r.score:.2f}):\n{r.text}"
            )
        output_text = "\n\n".join(formatted_parts)

        # Build citations summary
        citations = "\n".join(
            f"  [{i}] {r.source} (relevance: {r.score:.2f})"
            for i, r in enumerate(results, 1)
        )

        return ToolResult(
            tool_name=self.name,
            success=True,
            output=output_text,
            metadata={
                "query": query,
                "results_count": len(results),
                "citations": citations,
                "sources": [
                    {"source": r.source, "chunk_id": r.chunk_id, "score": r.score}
                    for r in results
                ],
            },
        )
