from __future__ import annotations

import json

from rag.fineweb import FineWebChunkConfig, FineWebKnowledgeIngestor, chunk_document


class _Embedder:
    def embed_batch(self, texts):
        return [[float(len(text)), 1.0] for text in texts]


def _rows():
    return [
        {"text": "alpha " * 30, "url": "https://example.test/a", "score": 0.9},
        {"text": "beta " * 30, "url": "https://example.test/b", "score": 0.8},
    ]


def test_chunk_document_is_deterministic_and_records_provenance():
    config = FineWebChunkConfig(target_tokens=100, chunk_tokens=10, overlap_tokens=2, min_document_chars=1)
    first = chunk_document("alpha " * 20, source="fineweb-edu/default", document_id="doc", config=config, url="u")
    second = chunk_document("alpha " * 20, source="fineweb-edu/default", document_id="doc", config=config, url="u")

    assert [chunk["chunk_id"] for chunk in first] == [chunk["chunk_id"] for chunk in second]
    assert all(chunk["metadata"]["dataset"] == "HuggingFaceFW/fineweb-edu" for chunk in first)
    assert all(chunk["metadata"]["estimated_tokens"] > 0 for chunk in first)


def test_ingestion_writes_embedding_sidecars_and_resumes(tmp_path):
    config = FineWebChunkConfig(target_tokens=10_000, chunk_tokens=10, overlap_tokens=2, min_document_chars=1, shard_size=2)
    ingestor = FineWebKnowledgeIngestor(tmp_path, config, embedder=_Embedder(), embedding_batch_size=2)
    first = ingestor.ingest(_rows(), resume=False, max_source_rows=1, checkpoint_every_documents=1)
    second = ingestor.ingest(_rows(), resume=True, checkpoint_every_documents=1)

    assert first.source_rows_seen == 1
    assert second.source_rows_seen == 2
    assert second.embedding_dimension == 2
    records = [json.loads(line) for file in tmp_path.glob("chunks-*.jsonl") for line in file.read_text(encoding="utf-8").splitlines()]
    assert records
    assert len({record["chunk_id"] for record in records}) == len(records)
    assert all((tmp_path / record["embedding"]["file"]).exists() for record in records)
    manifest = json.loads((tmp_path / "fineweb_edu_manifest.json").read_text(encoding="utf-8"))
    assert manifest["stats"]["chunks_written"] == len(records)
