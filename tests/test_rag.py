"""
tests/test_rag.py — Unit tests for the RAG pipeline.

Tests: chunking → embedding → indexing → retrieval → citations.
All tests run offline using the local vector index.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from rag.chunking import chunk_text, TextChunk
from rag.index import LocalVectorIndex
from rag.citations import format_citations, build_rag_prompt
from rag.retriever import Retriever, RetrievalResult


# ─────────────────────────────────────────────────────────────────────────────
# 1. Chunking
# ─────────────────────────────────────────────────────────────────────────────

class TestChunking:
    def test_fixed_chunking(self):
        text = "A" * 1000
        chunks = chunk_text(text, source="test.txt", chunk_size=200,
                            overlap=50, strategy="fixed")
        assert len(chunks) > 1
        for c in chunks:
            assert isinstance(c, TextChunk)
            assert len(c.text) <= 200

    def test_sentence_chunking(self):
        text = "First sentence. Second sentence. Third sentence. Fourth sentence."
        chunks = chunk_text(text, source="test.txt", chunk_size=40,
                            overlap=10, strategy="sentence")
        assert len(chunks) >= 1
        for c in chunks:
            assert isinstance(c, TextChunk)
            assert c.source == "test.txt"

    def test_empty_text(self):
        chunks = chunk_text("", source="empty.txt")
        assert chunks == []

    def test_chunk_ids_unique(self):
        text = "Some text. " * 50
        chunks = chunk_text(text, source="doc.txt", chunk_size=100, overlap=20)
        ids = [c.chunk_id for c in chunks]
        assert len(ids) == len(set(ids)), "Chunk IDs must be unique"

    def test_chunk_has_metadata(self):
        text = "Hello world. This is a test document."
        chunks = chunk_text(text, source="meta.txt", chunk_size=500)
        assert len(chunks) >= 1
        assert chunks[0].char_start == 0


# ─────────────────────────────────────────────────────────────────────────────
# 2. Vector Index
# ─────────────────────────────────────────────────────────────────────────────

class TestLocalVectorIndex:
    def test_add_and_search(self):
        index = LocalVectorIndex()
        chunk = TextChunk(chunk_id="c1", source="s1", text="hello")
        embedding = [1.0, 0.0, 0.0]
        index.add(chunk, embedding)

        results = index.search([1.0, 0.0, 0.0], top_k=1)
        assert len(results) == 1
        assert results[0]["chunk_id"] == "c1"
        assert results[0]["score"] > 0.99  # cosine of identical vectors = 1

    def test_top_k_ordering(self):
        index = LocalVectorIndex()
        # Add two chunks with different similarities to query
        index.add(TextChunk(chunk_id="far", source="s", text="far"), [0.0, 1.0, 0.0])
        index.add(TextChunk(chunk_id="close", source="s", text="close"), [0.9, 0.1, 0.0])

        results = index.search([1.0, 0.0, 0.0], top_k=2)
        assert results[0]["chunk_id"] == "close"  # higher similarity

    def test_source_filter(self):
        index = LocalVectorIndex()
        index.add(TextChunk(chunk_id="a1", source="doc_a", text="a"), [1.0, 0.0])
        index.add(TextChunk(chunk_id="b1", source="doc_b", text="b"), [1.0, 0.0])

        results = index.search([1.0, 0.0], top_k=10, source_filter="doc_a")
        assert all(r["source"] == "doc_a" for r in results)

    def test_empty_index_returns_nothing(self):
        index = LocalVectorIndex()
        results = index.search([1.0, 0.0], top_k=5)
        assert results == []

    def test_save_and_load(self, tmp_path):
        index = LocalVectorIndex()
        index.add(TextChunk(chunk_id="c1", source="s", text="hi"), [1.0, 0.0])

        save_path = tmp_path / "index.json"
        index.save(save_path)
        assert save_path.exists()

        loaded = LocalVectorIndex()
        loaded.load(save_path)
        assert len(loaded) == 1

        results = loaded.search([1.0, 0.0], top_k=1)
        assert results[0]["chunk_id"] == "c1"

    def test_len(self):
        index = LocalVectorIndex()
        assert len(index) == 0
        index.add(TextChunk(chunk_id="c1", source="s", text="t"), [1.0])
        assert len(index) == 1


# ─────────────────────────────────────────────────────────────────────────────
# 3. Retriever
# ─────────────────────────────────────────────────────────────────────────────

class TestRetriever:
    @pytest.fixture
    def populated_index(self):
        """Create an index with known embeddings for testing."""
        index = LocalVectorIndex()
        # Two chunks with distinct directions
        index.add(
            TextChunk(chunk_id="pump", source="manual.txt",
                      text="The pump P-201 requires quarterly lubrication."),
            [0.9, 0.1, 0.0],
        )
        index.add(
            TextChunk(chunk_id="valve", source="sop.txt",
                      text="The valve V-108 packing should be replaced annually."),
            [0.1, 0.9, 0.0],
        )
        return index

    def test_empty_index_warning(self):
        index = LocalVectorIndex()
        retriever = Retriever(index)
        results = retriever.retrieve("anything")
        assert results == []

    def test_build_context(self):
        results = [
            RetrievalResult(chunk_id="c1", source="s1", text="First chunk.", score=0.9),
            RetrievalResult(chunk_id="c2", source="s2", text="Second chunk.", score=0.8),
        ]
        retriever = Retriever(LocalVectorIndex())
        context = retriever.build_context(results)
        assert "First chunk" in context
        assert "Second chunk" in context

    def test_build_context_max_chars(self):
        results = [
            RetrievalResult(chunk_id="c1", source="s1", text="A" * 500, score=0.9),
            RetrievalResult(chunk_id="c2", source="s2", text="B" * 500, score=0.8),
        ]
        retriever = Retriever(LocalVectorIndex())
        context = retriever.build_context(results, max_chars=100)
        # Should truncate to fit within limit
        assert len(context) <= 600  # some overhead from citations


# ─────────────────────────────────────────────────────────────────────────────
# 4. Citations
# ─────────────────────────────────────────────────────────────────────────────

class TestCitations:
    def test_format_citations(self):
        results = [
            RetrievalResult(chunk_id="c1", source="manual.txt", text="...", score=0.95),
            RetrievalResult(chunk_id="c2", source="sop.txt", text="...", score=0.82),
        ]
        text = format_citations(results)
        assert "manual.txt" in text
        assert "sop.txt" in text
        assert "[1]" in text
        assert "[2]" in text

    def test_empty_citations(self):
        assert format_citations([]) == ""

    def test_build_rag_prompt(self):
        results = [
            RetrievalResult(chunk_id="c1", source="doc.txt", text="Relevant info.", score=0.9),
        ]
        prompt = build_rag_prompt("What is the procedure?", results)
        assert "Relevant info" in prompt
        assert "Question:" in prompt
        assert "Answer from the context only:" in prompt

    def test_retrieval_result_citation(self):
        r = RetrievalResult(chunk_id="c42", source="manual.pdf", text="...", score=0.88)
        cite = r.citation()
        assert "manual.pdf" in cite
        assert "c42" in cite


# ─────────────────────────────────────────────────────────────────────────────
# 5. DocumentIngester (integration)
# ─────────────────────────────────────────────────────────────────────────────

class TestDocumentIngester:
    def test_ingest_text(self):
        """Ingest a text string and verify chunks are added to the index."""
        from rag.ingest import DocumentIngester

        index = LocalVectorIndex()
        ingester = DocumentIngester(index)
        n = ingester.ingest_text(
            "This is a test document about pump maintenance. "
            "The pump requires regular lubrication every 90 days.",
            source="test_doc.txt",
        )
        assert n >= 1
        assert len(index) >= 1

    def test_ingest_file(self, tmp_path):
        """Ingest a file from disk."""
        from rag.ingest import DocumentIngester

        test_file = tmp_path / "test.txt"
        test_file.write_text(
            "Equipment inspection procedures for gas processing units.",
            encoding="utf-8",
        )
        index = LocalVectorIndex()
        ingester = DocumentIngester(index)
        n = ingester.ingest_file(test_file)
        assert n >= 1
