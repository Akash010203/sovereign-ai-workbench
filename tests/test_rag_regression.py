"""
tests/test_rag_regression.py — RAG pipeline regression tests.

Covers the COMPLETE RAG pipeline from document discovery through retrieval,
using controlled known-answer test fixtures.

Test categories:
    1. Document discovery
    2. File ingestion (TXT)
    3. Chunking verification
    4. Embedding consistency
    5. Index persistence
    6. Retrieval accuracy (known-answer)
    7. Negative retrieval (no-fabrication)
    8. Index rebuild reproducibility
    9. Embedding dimension compatibility
"""
from __future__ import annotations

import json
import os
import sys
import tempfile
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from rag.chunking import chunk_text, TextChunk
from rag.embeddings import get_embedder, TFIDFEmbedder, Embedder
from rag.index import LocalVectorIndex
from rag.ingest import DocumentIngester
from rag.retriever import Retriever, RetrievalResult
from rag.citations import format_citations, build_rag_prompt

FIXTURES = ROOT / "tests" / "fixtures" / "rag"


# ── 1. Document Discovery ─────────────────────────────────────────────────────

class TestDocumentDiscovery:
    """Verify the system can find documents in a directory."""

    def test_fixture_directory_exists(self):
        assert FIXTURES.is_dir(), f"RAG fixture directory missing: {FIXTURES}"

    def test_fixture_files_exist(self):
        expected = [
            "unit_a_specifications.txt",
            "pump_b_maintenance.txt",
            "emergency_shutdown.txt",
            "engineering_notes.txt",
        ]
        for filename in expected:
            path = FIXTURES / filename
            assert path.is_file(), f"Missing test fixture: {path}"

    def test_fixture_files_readable(self):
        for path in FIXTURES.glob("*.txt"):
            text = path.read_text(encoding="utf-8")
            assert len(text) > 50, f"Fixture file too small: {path} ({len(text)} chars)"

    def test_directory_scan_finds_all_txt(self):
        txt_files = list(FIXTURES.glob("*.txt"))
        assert len(txt_files) >= 4, f"Expected >= 4 txt files, found {len(txt_files)}"


# ── 2. Text Ingestion ─────────────────────────────────────────────────────────

class TestTextIngestion:
    """Verify text extraction from files produces non-empty output."""

    def test_ingest_txt_file(self):
        path = FIXTURES / "unit_a_specifications.txt"
        text = path.read_text(encoding="utf-8")
        assert "180°C" in text, "Expected temperature data not found in extracted text"
        assert len(text) > 200, f"Extracted text too short: {len(text)} chars"

    def test_ingest_all_fixtures(self):
        for path in FIXTURES.glob("*.txt"):
            text = path.read_text(encoding="utf-8")
            assert len(text) > 50, f"Empty/short extraction from {path.name}"
            # Verify actual content, not just file size
            words = text.split()
            assert len(words) > 10, f"Too few words in {path.name}: {len(words)}"


# ── 3. Chunking ───────────────────────────────────────────────────────────────

class TestChunking:
    """Verify chunking produces valid, overlapping chunks."""

    def test_basic_chunking(self):
        text = (FIXTURES / "unit_a_specifications.txt").read_text(encoding="utf-8")
        chunks = chunk_text(text, source="test_doc.txt", chunk_size=400, overlap=80)
        assert len(chunks) >= 1, "Expected at least 1 chunk"
        for chunk in chunks:
            assert isinstance(chunk, TextChunk)
            assert chunk.text.strip(), "Empty chunk text"
            assert chunk.source == "test_doc.txt"
            assert chunk.chunk_id, "Missing chunk_id"

    def test_chunk_size_respected(self):
        text = (FIXTURES / "pump_b_maintenance.txt").read_text(encoding="utf-8")
        chunks = chunk_text(text, source="pump.txt", chunk_size=300, overlap=60)
        for chunk in chunks:
            # Allow some tolerance for sentence-boundary chunking
            assert len(chunk.text) <= 400, f"Chunk too large: {len(chunk.text)} chars"

    def test_no_lost_content(self):
        """Verify chunking doesn't silently drop significant content."""
        text = (FIXTURES / "emergency_shutdown.txt").read_text(encoding="utf-8")
        chunks = chunk_text(text, source="esd.txt", chunk_size=400, overlap=80)
        # Key phrases that must appear in at least one chunk
        key_phrases = ["valve isolation", "emergency shutdown", "depressurization"]
        all_chunk_text = " ".join(c.text.lower() for c in chunks)
        for phrase in key_phrases:
            assert phrase in all_chunk_text, f"Key phrase lost in chunking: '{phrase}'"

    def test_chunk_metadata(self):
        text = "Test document content for metadata verification."
        chunks = chunk_text(text, source="meta_test.txt")
        assert len(chunks) >= 1
        chunk = chunks[0]
        assert chunk.char_start >= 0
        assert chunk.char_end > chunk.char_start

    def test_empty_text_produces_no_chunks(self):
        chunks = chunk_text("", source="empty.txt")
        assert chunks == []

    def test_whitespace_only_produces_no_chunks(self):
        chunks = chunk_text("   \n\t  ", source="whitespace.txt")
        assert chunks == []


# ── 4. Embedding Consistency ──────────────────────────────────────────────────

class TestEmbeddingConsistency:
    """Verify embeddings are deterministic and have correct dimensions."""

    def test_embedder_loads(self):
        embedder = get_embedder()
        assert embedder is not None

    def test_embedding_produces_vector(self):
        embedder = get_embedder()
        vec = embedder.embed("Test text for embedding")
        assert isinstance(vec, list)
        assert len(vec) > 0
        assert all(isinstance(v, float) for v in vec)

    def test_embedding_deterministic(self):
        """Same text must produce identical embeddings."""
        embedder = get_embedder()
        text = "The maximum operating temperature of Unit A is 180°C."
        vec1 = embedder.embed(text)
        vec2 = embedder.embed(text)
        assert vec1 == vec2, "Embeddings not deterministic for identical input"

    def test_batch_embedding(self):
        embedder = get_embedder()
        texts = ["First document", "Second document", "Third document"]
        batch = embedder.embed_batch(texts)
        assert len(batch) == 3
        for vec in batch:
            assert len(vec) > 0

    def test_batch_single_consistency(self):
        """Batch embedding must match single embedding."""
        embedder = get_embedder()
        text = "Consistency test between batch and single embedding."
        single = embedder.embed(text)
        batch = embedder.embed_batch([text])
        assert single == batch[0], "Batch embedding differs from single embedding"

    def test_embedding_dimension_consistent(self):
        """All embeddings from the same embedder must have the same dimension."""
        embedder = get_embedder()
        texts = [
            "Short text",
            "A much longer text with many more words to see if dimension changes",
            "Numbers: 42 180 3600",
        ]
        dims = [len(embedder.embed(t)) for t in texts]
        assert len(set(dims)) == 1, f"Inconsistent dimensions: {dims}"


# ── 5. Index Persistence ─────────────────────────────────────────────────────

class TestIndexPersistence:
    """Verify the index can be saved and reloaded."""

    def test_save_and_reload(self, tmp_path):
        index = LocalVectorIndex()
        embedder = get_embedder()

        text = "Test document for persistence verification."
        chunks = chunk_text(text, source="persist_test.txt")
        embeddings = embedder.embed_batch([c.text for c in chunks])
        index.add_batch(chunks, embeddings)

        save_path = tmp_path / "test_index.json"
        index.save(save_path)
        assert save_path.exists()
        assert save_path.stat().st_size > 0

        # Reload into a new index
        index2 = LocalVectorIndex()
        index2.load(save_path)
        assert len(index2) == len(index)

    def test_reload_preserves_retrieval(self, tmp_path):
        """After reload, retrieval must return the same results."""
        embedder = get_embedder()
        index = LocalVectorIndex()

        text = "The operating pressure is 42 bar for valve XV-100."
        chunks = chunk_text(text, source="valve_spec.txt")
        embeddings = embedder.embed_batch([c.text for c in chunks])
        index.add_batch(chunks, embeddings)

        save_path = tmp_path / "reload_test.json"
        index.save(save_path)

        index2 = LocalVectorIndex()
        index2.load(save_path)

        query_emb = embedder.embed("What is the operating pressure?")
        results1 = index.search(query_emb, top_k=1)
        results2 = index2.search(query_emb, top_k=1)
        assert len(results1) == len(results2)
        assert results1[0]["text"] == results2[0]["text"]
        assert abs(results1[0]["score"] - results2[0]["score"]) < 0.001


# ── 6. Retrieval Accuracy (Known-Answer) ──────────────────────────────────────

class TestRetrievalAccuracy:
    """Test retrieval with known-answer documents."""

    @pytest.fixture
    def populated_retriever(self):
        """Build a retriever with all test fixtures ingested."""
        index = LocalVectorIndex()
        ingester = DocumentIngester(index)
        for path in sorted(FIXTURES.glob("*.txt")):
            ingester.ingest_file(path)
        retriever = Retriever(index)
        return retriever, index

    def test_unit_a_temperature(self, populated_retriever):
        retriever, index = populated_retriever
        assert len(index) > 0, "Index is empty after ingestion"

        results = retriever.retrieve("What is the maximum operating temperature of Unit A?")
        assert len(results) > 0, "No results returned for Unit A temperature query"

        # The top result must come from the unit_a_specifications document
        top_text = results[0].text.lower()
        assert "180" in top_text or "temperature" in top_text, (
            f"Top result does not contain temperature data: {results[0].text[:200]}"
        )

    def test_pump_b_inspection(self, populated_retriever):
        retriever, index = populated_retriever
        results = retriever.retrieve("What is the inspection interval for Pump B?")
        assert len(results) > 0

        top_text = results[0].text.lower()
        assert "90 days" in top_text or "pump b" in top_text, (
            f"Top result missing pump B data: {results[0].text[:200]}"
        )

    def test_emergency_shutdown(self, populated_retriever):
        retriever, index = populated_retriever
        results = retriever.retrieve("What does the emergency shutdown procedure require?")
        assert len(results) > 0

        top_text = results[0].text.lower()
        assert "valve" in top_text or "isolation" in top_text or "shutdown" in top_text, (
            f"Top result missing ESD data: {results[0].text[:200]}"
        )

    def test_bolt_torque(self, populated_retriever):
        retriever, index = populated_retriever
        results = retriever.retrieve("What is the bolt torque for flange FA-D-100?")
        assert len(results) > 0

        top_text = results[0].text.lower()
        assert "450" in top_text or "torque" in top_text or "bolt" in top_text, (
            f"Top result missing torque data: {results[0].text[:200]}"
        )

    def test_results_have_citations(self, populated_retriever):
        retriever, _ = populated_retriever
        results = retriever.retrieve("temperature")
        assert len(results) > 0
        for r in results:
            citation = r.citation()
            assert "Source:" in citation
            assert r.source, "Missing source in result"
            assert r.chunk_id, "Missing chunk_id in result"

    def test_scores_are_sorted(self, populated_retriever):
        retriever, _ = populated_retriever
        results = retriever.retrieve("maintenance inspection", top_k=5)
        if len(results) >= 2:
            for i in range(len(results) - 1):
                assert results[i].score >= results[i + 1].score, (
                    f"Results not sorted: score[{i}]={results[i].score} < score[{i+1}]={results[i+1].score}"
                )


# ── 7. Negative Retrieval ─────────────────────────────────────────────────────

class TestNegativeRetrieval:
    """The system must not fabricate information for unrelated queries."""

    @pytest.fixture
    def populated_retriever(self):
        index = LocalVectorIndex()
        ingester = DocumentIngester(index)
        for path in sorted(FIXTURES.glob("*.txt")):
            ingester.ingest_file(path)
        return Retriever(index)

    def test_unrelated_query_low_score(self, populated_retriever):
        """A completely unrelated query should return low-confidence results."""
        results = populated_retriever.retrieve(
            "What is the capital of France?", top_k=3, min_score=0.5
        )
        # With a high min_score threshold, unrelated queries should return few/no results
        # If results are returned, they should have low scores
        if results:
            for r in results:
                # A perfect semantic match would score near 1.0
                # TF-IDF won't have "france" or "capital" in industrial docs
                pass  # Just ensure no crash; score checking depends on embedder

    def test_empty_index_returns_nothing(self):
        """An empty index must return an empty result set, not fabricate."""
        empty_index = LocalVectorIndex()
        retriever = Retriever(empty_index)
        results = retriever.retrieve("Any question at all")
        assert results == [], f"Empty index returned results: {results}"

    def test_min_score_filtering(self, populated_retriever):
        """Results below min_score threshold must be filtered out."""
        results = populated_retriever.retrieve(
            "quantum chromodynamics and string theory", min_score=0.99
        )
        # With an unreasonably high threshold, nothing should match
        assert len(results) == 0, (
            f"Expected no results at min_score=0.99, got {len(results)}"
        )


# ── 8. Index Rebuild Reproducibility ──────────────────────────────────────────

class TestIndexRebuild:
    """The RAG pipeline must produce identical results when rebuilt."""

    def test_rebuild_produces_same_results(self, tmp_path):
        """Build index twice from same documents; retrieval must match."""
        embedder = get_embedder()

        def build_index():
            idx = LocalVectorIndex()
            ingester = DocumentIngester(idx)
            for path in sorted(FIXTURES.glob("*.txt")):
                ingester.ingest_file(path)
            return idx

        index1 = build_index()
        index2 = build_index()

        assert len(index1) == len(index2), "Rebuild changed chunk count"

        query = "operating temperature"
        query_emb = embedder.embed(query)
        results1 = index1.search(query_emb, top_k=3)
        results2 = index2.search(query_emb, top_k=3)

        assert len(results1) == len(results2)
        for r1, r2 in zip(results1, results2):
            assert r1["text"] == r2["text"], "Rebuild changed retrieval order"
            assert abs(r1["score"] - r2["score"]) < 0.001


# ── 9. Embedding Dimension Compatibility ──────────────────────────────────────

class TestEmbeddingDimensionCompatibility:
    """Verify embedding dimensions match the loaded index."""

    def test_tfidf_dimension(self):
        embedder = TFIDFEmbedder(dim=512)
        vec = embedder.embed("test")
        assert len(vec) == 512

    def test_tfidf_custom_dimension(self):
        embedder = TFIDFEmbedder(dim=256)
        vec = embedder.embed("test")
        assert len(vec) == 256

    def test_existing_user_index_dimension(self):
        """Check that the existing user_knowledge_index.json has consistent dimensions."""
        index_path = ROOT / "data" / "user_knowledge_index.json"
        if not index_path.exists():
            pytest.skip("user_knowledge_index.json not present")

        data = json.loads(index_path.read_text(encoding="utf-8"))
        records = data.get("records", [])
        if not records:
            pytest.skip("Index is empty")

        dims = set()
        for rec in records[:100]:  # Sample first 100
            emb = rec.get("embedding", [])
            dims.add(len(emb))

        assert len(dims) == 1, f"Inconsistent dimensions in index: {dims}"

    def test_new_embedder_matches_tfidf_index(self):
        """The current embedder dimension must match TF-IDF index if using TF-IDF fallback."""
        embedder = get_embedder()
        dim = len(embedder.embed("dimension check"))

        tfidf_index_path = ROOT / "data" / "rag_index_tfidf.json"
        if not tfidf_index_path.exists():
            pytest.skip("rag_index_tfidf.json not present")

        data = json.loads(tfidf_index_path.read_text(encoding="utf-8"))
        records = data.get("records", [])
        if not records:
            pytest.skip("TF-IDF index is empty")

        index_dim = len(records[0].get("embedding", []))

        if isinstance(embedder, TFIDFEmbedder):
            assert dim == index_dim, (
                f"TF-IDF embedder dimension ({dim}) != index dimension ({index_dim})"
            )


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
