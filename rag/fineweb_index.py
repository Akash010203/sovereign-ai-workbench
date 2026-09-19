"""Disk-backed retrieval over sharded FineWeb embedding files.

FineWeb-Edu has millions of vectors, so it must not be imported into the
project's JSON ``LocalVectorIndex``.  This reader memory-maps one vector shard
at a time, retains only a small top-k heap, and lazily reads the matching JSONL
records.  It is exact cosine/inner-product search for normalized vectors and
uses bounded RAM; an external ANN service can replace it later without
changing the ingestion format.
"""
from __future__ import annotations

import heapq
import json
import logging
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from rag.embeddings import Embedder, get_embedder
from rag.retriever import RetrievalResult

log = logging.getLogger(__name__)


@dataclass(frozen=True)
class FineWebIndexStatus:
    available: bool
    chunks: int = 0
    dimension: int | None = None
    mode: str = "unavailable"
    detail: str = "FineWeb knowledge shards have not been created."


class FineWebDiskIndex:
    """Search completed ``ingest_fineweb_knowledge.py`` output without RAM growth."""

    def __init__(self, root: str | Path) -> None:
        self.root = Path(root)
        self.manifest_path = self.root / "fineweb_edu_manifest.json"
        self._status = self._read_status()

    def _read_status(self) -> FineWebIndexStatus:
        if not self.manifest_path.is_file():
            return FineWebIndexStatus(False)
        try:
            manifest = json.loads(self.manifest_path.read_text(encoding="utf-8"))
            stats = manifest["stats"]
            count = int(stats["chunks_written"])
            dimension = int(stats["embedding_dimension"] or 0)
            if count <= 0 or dimension <= 0:
                return FineWebIndexStatus(False, detail="FineWeb manifest has no completed embedded chunks.")
            vector_files = sorted(self.root.glob("embeddings-*.f32"))
            if not vector_files:
                return FineWebIndexStatus(False, detail="FineWeb embedding shards are missing.")
            return FineWebIndexStatus(
                True, count, dimension, "disk_exact",
                "Memory-mapped exact search across local FineWeb embedding shards.",
            )
        except (OSError, ValueError, KeyError, TypeError, json.JSONDecodeError) as exc:
            return FineWebIndexStatus(False, detail=f"FineWeb manifest cannot be read: {exc}")

    @property
    def status(self) -> FineWebIndexStatus:
        return self._status

    def __len__(self) -> int:
        return self._status.chunks if self._status.available else 0

    def search(self, query_embedding: list[float], top_k: int = 5) -> list[dict]:
        """Return top-k records using a bounded-memory scan of vector shards."""
        if not self._status.available or top_k <= 0:
            return []
        query = np.asarray(query_embedding, dtype=np.float32)
        if query.ndim != 1 or query.size != self._status.dimension:
            raise ValueError(
                f"Embedding dimension mismatch: query={query.size}, FineWeb={self._status.dimension}"
            )

        candidates: list[tuple[float, int, int]] = []
        vector_bytes = self._status.dimension * np.dtype("<f4").itemsize
        for vector_path in sorted(self.root.glob("embeddings-*.f32")):
            shard_name = vector_path.stem.rsplit("-", 1)[-1]
            chunk_path = self.root / f"chunks-{shard_name}.jsonl"
            if not chunk_path.is_file():
                log.warning("Skipping FineWeb vector shard without chunk metadata: %s", vector_path.name)
                continue
            size = vector_path.stat().st_size
            if size % vector_bytes:
                log.warning("Skipping incomplete FineWeb vector shard: %s", vector_path.name)
                continue
            vector_count = size // vector_bytes
            if not vector_count:
                continue
            matrix = np.memmap(vector_path, dtype="<f4", mode="r", shape=(vector_count, self._status.dimension))
            scores = matrix @ query
            local_k = min(top_k, vector_count)
            selected = np.argpartition(scores, -local_k)[-local_k:]
            shard_number = int(shard_name)
            for row_number in selected:
                entry = (float(scores[row_number]), shard_number, int(row_number))
                if len(candidates) < top_k:
                    heapq.heappush(candidates, entry)
                elif entry[0] > candidates[0][0]:
                    heapq.heapreplace(candidates, entry)
            del scores, matrix

        ordered = sorted(candidates, reverse=True)
        wanted: dict[int, set[int]] = {}
        for _, shard_number, row_number in ordered:
            wanted.setdefault(shard_number, set()).add(row_number)
        records: dict[tuple[int, int], dict] = {}
        for shard_number, rows in wanted.items():
            chunk_path = self.root / f"chunks-{shard_number:05d}.jsonl"
            with chunk_path.open("r", encoding="utf-8") as handle:
                for row_number, line in enumerate(handle):
                    if row_number in rows:
                        records[(shard_number, row_number)] = json.loads(line)
                        if len(records) == sum(map(len, wanted.values())):
                            break
        return [
            {**records[(shard, row)], "score": round(score, 4)}
            for score, shard, row in ordered
            if (shard, row) in records
        ]


class FineWebRetriever:
    """Embed a query locally then search the disk-backed FineWeb knowledge base."""

    def __init__(self, index: FineWebDiskIndex, embedder: Embedder | None = None) -> None:
        self.index = index
        self.embedder = embedder or get_embedder()

    def retrieve(self, query: str, top_k: int = 5, min_score: float = 0.0) -> list[RetrievalResult]:
        query_embedding = self.embedder.embed(query)
        if len(query_embedding) != self.index.status.dimension:
            log.warning(
                "FineWeb index uses %s-dimensional legacy embeddings; current local "
                "embedder uses %s dimensions. Rebuild the index before searching it.",
                self.index.status.dimension, len(query_embedding),
            )
            return []
        raw = self.index.search(query_embedding, top_k=top_k)
        return [
            RetrievalResult(
                chunk_id=record["chunk_id"], source=record["source"], text=record["text"],
                score=record["score"], metadata=record.get("metadata", {}),
            )
            for record in raw if record["score"] >= min_score
        ]
