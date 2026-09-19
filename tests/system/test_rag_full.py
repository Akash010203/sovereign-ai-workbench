"""
TEST G — Local RAG Full Validation

Verifies:
- Document ingestion
- Chunking
- Embedding
- Retrieval
- Citation formatting
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))


class TestRAGChunking:
    """Test document chunking."""

    def test_chunk_text(self):
        from rag.chunking import chunk_text
        text = "Hello world. " * 100
        chunks = chunk_text(text, source="test_doc", chunk_size=200, overlap=50)
        assert len(chunks) > 1
        for chunk in chunks:
            assert len(chunk.text) <= 250  # Allow some margin

    def test_chunk_empty(self):
        from rag.chunking import chunk_text
        chunks = chunk_text("", source="empty", chunk_size=200, overlap=50)
        assert len(chunks) == 0

    def test_chunk_preserves_content(self):
        from rag.chunking import chunk_text
        text = "Word1 Word2 Word3 Word4 Word5"
        chunks = chunk_text(text, source="test", chunk_size=1000)
        combined = " ".join(c.text for c in chunks)
        for word in ["Word1", "Word2", "Word3", "Word4", "Word5"]:
            assert word in combined

    def test_chunk_has_metadata(self):
        from rag.chunking import chunk_text
        chunks = chunk_text("Test content here.", source="doc.pdf", chunk_size=1000)
        assert len(chunks) > 0
        assert chunks[0].source == "doc.pdf"
        assert chunks[0].chunk_id is not None


class TestRAGEmbeddings:
    """Test embedding system."""

    def test_embedder_available(self):
        from rag.embeddings import get_embedder
        embedder = get_embedder()
        assert embedder is not None

    def test_embed_produces_vector(self):
        from rag.embeddings import get_embedder
        embedder = get_embedder()
        vec = embedder.embed("Hello world")
        assert isinstance(vec, list)
        assert len(vec) > 0
        assert all(isinstance(v, float) for v in vec)

    def test_embed_batch(self):
        from rag.embeddings import get_embedder
        embedder = get_embedder()
        vecs = embedder.embed_batch(["Hello", "World", "Test"])
        assert len(vecs) == 3
        assert len(vecs[0]) == len(vecs[1]) == len(vecs[2])

    def test_similar_texts_have_higher_similarity(self):
        from rag.embeddings import get_embedder
        import math
        embedder = get_embedder()

        v1 = embedder.embed("The pump is running")
        v2 = embedder.embed("The pump is operating")
        v3 = embedder.embed("The weather is sunny today")

        def cosine(a, b):
            dot = sum(x * y for x, y in zip(a, b))
            na = math.sqrt(sum(x * x for x in a))
            nb = math.sqrt(sum(x * x for x in b))
            return dot / (na * nb) if na * nb > 0 else 0

        sim_related = cosine(v1, v2)
        sim_unrelated = cosine(v1, v3)
        assert sim_related > sim_unrelated, (
            f"Related similarity ({sim_related:.3f}) not higher than "
            f"unrelated ({sim_unrelated:.3f})"
        )


class TestRAGIndex:
    """Test vector index operations."""

    def test_create_index(self):
        from rag.index import LocalVectorIndex
        idx = LocalVectorIndex()
        assert len(idx) == 0

    def test_add_and_search(self):
        from rag.index import LocalVectorIndex
        from rag.chunking import TextChunk
        from rag.embeddings import get_embedder

        embedder = get_embedder()
        idx = LocalVectorIndex()

        texts = [
            "The pump requires regular maintenance every 6 months.",
            "Safety valves must be inspected annually.",
            "Compressor oil should be changed every 3000 hours.",
        ]
        for i, text in enumerate(texts):
            vec = embedder.embed(text)
            chunk = TextChunk(chunk_id=f"chunk_{i}", source=f"doc_{i}", text=text)
            idx.add(chunk, vec)

        assert len(idx) == 3

        query_vec = embedder.embed("pump maintenance schedule")
        results = idx.search(query_vec, top_k=2)
        assert len(results) > 0
        assert any("pump" in r["text"].lower() for r in results)

    def test_save_and_load(self, tmp_path):
        from rag.index import LocalVectorIndex
        from rag.chunking import TextChunk
        from rag.embeddings import get_embedder

        embedder = get_embedder()
        idx = LocalVectorIndex()

        vec = embedder.embed("Test document")
        chunk = TextChunk(chunk_id="c1", source="test", text="Test document")
        idx.add(chunk, vec)

        save_path = tmp_path / "test_index.json"
        idx.save(save_path)

        loaded = LocalVectorIndex()
        loaded.load(save_path)
        assert len(loaded) == 1


class TestRAGRetriever:
    """Test the retriever pipeline."""

    def test_retriever_end_to_end(self):
        from rag.index import LocalVectorIndex
        from rag.chunking import TextChunk
        from rag.retriever import Retriever
        from rag.embeddings import get_embedder

        embedder = get_embedder()
        idx = LocalVectorIndex()

        texts = [
            "Pump P-101 pressure reading: 10.5 bar",
            "Valve V-200 was inspected on 2024-01-15",
            "Compressor C-300 vibration exceeded threshold",
        ]
        for i, text in enumerate(texts):
            vec = embedder.embed(text)
            chunk = TextChunk(chunk_id=f"c_{i}", source=f"report_{i}", text=text)
            idx.add(chunk, vec)

        retriever = Retriever(idx)
        results = retriever.retrieve("What is the pump pressure?", top_k=2)
        assert len(results) > 0
        # Retriever returns RetrievalResult objects with .text attribute
        assert any("pump" in r.text.lower() or "P-101" in r.text for r in results)


class TestRAGCitations:
    """Test citation formatting."""

    def test_citation_format(self):
        from rag.citations import format_citations
        from rag.retriever import RetrievalResult

        results = [RetrievalResult(
            chunk_id="c1", source="SOP.pdf", text="Test text", score=0.85
        )]
        formatted = format_citations(results)
        assert isinstance(formatted, str)
        assert len(formatted) > 0
        assert "SOP.pdf" in formatted


class TestRAGIngest:
    """Test document ingestion."""

    def test_ingest_text(self):
        from rag.index import LocalVectorIndex
        from rag.ingest import DocumentIngester

        idx = LocalVectorIndex()
        ingester = DocumentIngester(idx)
        n = ingester.ingest_text(
            "This is a test document about pump maintenance procedures.",
            source="test_doc.txt"
        )
        assert n > 0
        assert len(idx) > 0
