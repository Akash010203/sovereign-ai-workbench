"""rag/__init__.py"""
from rag.chunking import chunk_text, TextChunk
from rag.index import LocalVectorIndex
from rag.embeddings import get_embedder
from rag.ingest import DocumentIngester
from rag.retriever import Retriever, RetrievalResult
from rag.citations import format_citations, build_rag_prompt
from rag.fineweb import FineWebChunkConfig, FineWebKnowledgeIngestor
from rag.fineweb_index import FineWebDiskIndex, FineWebRetriever
from rag.hybrid import HybridRetriever
__all__ = [
    "chunk_text", "TextChunk",
    "LocalVectorIndex",
    "get_embedder",
    "DocumentIngester",
    "Retriever", "RetrievalResult",
    "format_citations", "build_rag_prompt",
    "FineWebChunkConfig", "FineWebKnowledgeIngestor",
    "FineWebDiskIndex", "FineWebRetriever", "HybridRetriever",
]
