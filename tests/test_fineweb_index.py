from __future__ import annotations

import json
from array import array

from rag.fineweb_index import FineWebDiskIndex, FineWebRetriever


class _Embedder:
    def embed(self, text):
        return [1.0, 0.0]


def test_disk_index_searches_vector_shard_and_reads_only_matched_records(tmp_path):
    (tmp_path / "fineweb_edu_manifest.json").write_text(json.dumps({
        "stats": {"chunks_written": 2, "embedding_dimension": 2},
    }), encoding="utf-8")
    records = [
        {"chunk_id": "a", "source": "fineweb-edu/default", "text": "first", "metadata": {}},
        {"chunk_id": "b", "source": "fineweb-edu/default", "text": "second", "metadata": {}},
    ]
    (tmp_path / "chunks-00000.jsonl").write_text(
        "".join(json.dumps(record) + "\n" for record in records), encoding="utf-8",
    )
    with (tmp_path / "embeddings-00000.f32").open("wb") as handle:
        array("f", [0.1, 1.0, 1.0, 0.0]).tofile(handle)

    index = FineWebDiskIndex(tmp_path)
    assert index.status.available
    assert len(index) == 2
    result = FineWebRetriever(index, embedder=_Embedder()).retrieve("question", top_k=1)
    assert [item.chunk_id for item in result] == ["b"]
    assert result[0].score == 1.0
