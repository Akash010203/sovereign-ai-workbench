"""Resumable, disk-bounded FineWeb-Edu knowledge-chunk ingestion.

This module deliberately does not try to put a web-scale corpus in
``LocalVectorIndex``.  That index is a JSON-backed, brute-force demo index;
millions of vectors belong in sharded files and a production ANN index.  The
output here is a portable knowledge-chunk manifest plus optional float32
embedding shards suitable for an ANN/vector-store build step.
"""
from __future__ import annotations

import hashlib
import json
import math
import os
import re
import tempfile
import logging
from array import array
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable, Iterator, Protocol


DEFAULT_DATASET = "HuggingFaceFW/fineweb-edu"
DEFAULT_CONFIG = "default"
log = logging.getLogger(__name__)


class BatchEmbedder(Protocol):
    def embed_batch(self, texts: list[str]) -> list[list[float]]: ...


@dataclass(frozen=True)
class FineWebChunkConfig:
    """Settings that define both the output and its resume compatibility."""

    target_tokens: int = 1_600_000_000
    chunk_tokens: int = 384
    overlap_tokens: int = 48
    chars_per_token: float = 4.0
    min_document_chars: int = 200
    shard_size: int = 50_000
    dataset_id: str = DEFAULT_DATASET
    dataset_config: str = DEFAULT_CONFIG

    def __post_init__(self) -> None:
        if self.target_tokens <= 0 or self.chunk_tokens <= 0:
            raise ValueError("target_tokens and chunk_tokens must be positive")
        if not 0 <= self.overlap_tokens < self.chunk_tokens:
            raise ValueError("overlap_tokens must be non-negative and less than chunk_tokens")
        if self.chars_per_token <= 0 or self.min_document_chars < 0 or self.shard_size <= 0:
            raise ValueError("chars_per_token and shard_size must be positive; min_document_chars cannot be negative")


@dataclass
class IngestionStats:
    source_rows_seen: int = 0
    accepted_documents: int = 0
    skipped_documents: int = 0
    chunks_written: int = 0
    source_estimated_tokens: int = 0
    retrieval_chunk_estimated_tokens: int = 0
    embedding_dimension: int | None = None


def normalize_text(value: Any) -> str:
    """Normalize text without changing its language or inventing content."""
    return re.sub(r"\s+", " ", str(value or "")).strip()


def estimate_tokens(text: str, chars_per_token: float = 4.0) -> int:
    """A transparent estimate, not a substitute for the model tokenizer."""
    return max(1, math.ceil(len(text) / chars_per_token)) if text else 0


def chunk_document(
    text: str,
    *,
    source: str,
    document_id: str,
    config: FineWebChunkConfig,
    url: str = "",
    score: Any = None,
) -> list[dict[str, Any]]:
    """Create deterministic, overlapping, character-budgeted knowledge chunks.

    FineWeb records do not carry tokenization for this project's custom
    tokenizer.  A character budget therefore keeps chunking streaming-safe
    while the recorded ``estimated_tokens`` makes that approximation explicit.
    """
    text = normalize_text(text)
    if len(text) < config.min_document_chars:
        return []

    target_chars = max(1, int(config.chunk_tokens * config.chars_per_token))
    overlap_chars = int(config.overlap_tokens * config.chars_per_token)
    chunks: list[dict[str, Any]] = []
    start = 0
    chunk_number = 0
    while start < len(text):
        proposed_end = min(len(text), start + target_chars)
        end = proposed_end
        # Keep chunks readable by choosing a whitespace boundary when possible.
        if proposed_end < len(text):
            boundary = text.rfind(" ", start + max(1, target_chars // 2), proposed_end)
            if boundary > start:
                end = boundary
        piece = text[start:end].strip()
        if not piece:
            break
        digest = hashlib.sha256(
            f"{document_id}:{start}:{piece}".encode("utf-8")
        ).hexdigest()
        chunks.append({
            "chunk_id": f"fineweb-edu-{digest[:24]}",
            "source": source,
            "text": piece,
            "char_start": start,
            "char_end": end,
            "metadata": {
                "dataset": config.dataset_id,
                "config": config.dataset_config,
                "document_id": document_id,
                "document_chunk": chunk_number,
                "url": url,
                "score": score,
                "estimated_tokens": estimate_tokens(piece, config.chars_per_token),
                "content_sha256": hashlib.sha256(piece.encode("utf-8")).hexdigest(),
            },
        })
        chunk_number += 1
        if end >= len(text):
            break
        start = max(end - overlap_chars, start + 1)
    return chunks


class ShardedKnowledgeWriter:
    """Append JSONL chunk shards and matching float32 vector shards."""

    def __init__(self, output_dir: Path, shard_size: int, start_shard: int = 0, start_count: int = 0) -> None:
        self.output_dir = output_dir
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.shard_size = shard_size
        self.shard_number = start_shard
        self.shard_count = start_count
        self._chunk_file = None
        self._vector_file = None
        self._open_shard()

    @property
    def chunk_path(self) -> Path:
        return self.output_dir / f"chunks-{self.shard_number:05d}.jsonl"

    @property
    def vector_path(self) -> Path:
        return self.output_dir / f"embeddings-{self.shard_number:05d}.f32"

    def _open_shard(self) -> None:
        self._chunk_file = self.chunk_path.open("a", encoding="utf-8")
        self._vector_file = self.vector_path.open("ab")

    def _rotate(self) -> None:
        self.close()
        self.shard_number += 1
        self.shard_count = 0
        self._open_shard()

    def write(self, chunks: list[dict[str, Any]], embeddings: list[list[float]] | None = None) -> int:
        if embeddings is not None and len(chunks) != len(embeddings):
            raise ValueError("Each chunk must have exactly one embedding")
        for index, chunk in enumerate(chunks):
            if self.shard_count >= self.shard_size:
                self._rotate()
            if embeddings is not None:
                vector = array("f", embeddings[index])
                dimension = len(vector)
                byte_offset = self._vector_file.tell()
                vector.tofile(self._vector_file)
                chunk["embedding"] = {
                    "file": self.vector_path.name,
                    "byte_offset": byte_offset,
                    "dimension": dimension,
                    "dtype": "float32",
                }
            self._chunk_file.write(json.dumps(chunk, ensure_ascii=False) + "\n")
            self.shard_count += 1
        self._chunk_file.flush()
        self._vector_file.flush()
        return len(chunks)

    def close(self) -> None:
        if self._chunk_file is not None:
            self._chunk_file.close()
            self._chunk_file = None
        if self._vector_file is not None:
            self._vector_file.close()
            self._vector_file = None


class FineWebKnowledgeIngestor:
    """Turn a streaming FineWeb iterator into resumable local knowledge shards."""

    CHECKPOINT_NAME = "fineweb_edu_checkpoint.json"
    MANIFEST_NAME = "fineweb_edu_manifest.json"

    def __init__(
        self,
        output_dir: str | Path,
        config: FineWebChunkConfig | None = None,
        *,
        embedder: BatchEmbedder | None = None,
        embedding_batch_size: int = 64,
    ) -> None:
        self.output_dir = Path(output_dir)
        self.config = config or FineWebChunkConfig()
        self.embedder = embedder
        self.embedding_batch_size = embedding_batch_size
        if embedding_batch_size <= 0:
            raise ValueError("embedding_batch_size must be positive")

    @property
    def checkpoint_path(self) -> Path:
        return self.output_dir / self.CHECKPOINT_NAME

    @property
    def manifest_path(self) -> Path:
        return self.output_dir / self.MANIFEST_NAME

    def _load_checkpoint(self, resume: bool) -> tuple[IngestionStats, int, int]:
        if not resume or not self.checkpoint_path.exists():
            return IngestionStats(), 0, 0
        data = json.loads(self.checkpoint_path.read_text(encoding="utf-8"))
        if data.get("config") != asdict(self.config):
            raise ValueError("Checkpoint configuration differs. Use a new output directory or disable --resume.")
        return IngestionStats(**data["stats"]), int(data["active_shard"]), int(data["active_shard_records"])

    def _save_checkpoint(self, stats: IngestionStats, writer: ShardedKnowledgeWriter) -> None:
        self.output_dir.mkdir(parents=True, exist_ok=True)
        payload = {
            "config": asdict(self.config),
            "stats": asdict(stats),
            "active_shard": writer.shard_number,
            "active_shard_records": writer.shard_count,
        }
        # Atomic replacement means Ctrl+C never leaves a malformed checkpoint.
        with tempfile.NamedTemporaryFile("w", encoding="utf-8", dir=self.output_dir, delete=False) as temp:
            json.dump(payload, temp, indent=2)
            temp_name = temp.name
        os.replace(temp_name, self.checkpoint_path)

    def _save_manifest(self, stats: IngestionStats) -> None:
        payload = {
            "dataset": self.config.dataset_id,
            "dataset_config": self.config.dataset_config,
            "purpose": "RAG knowledge chunks; this does not train or modify the language model weights.",
            "token_count_kind": "estimated from normalized character count; retokenize with the training tokenizer for an exact training budget.",
            "config": asdict(self.config),
            "stats": asdict(stats),
            "chunk_format": "JSONL TextChunk-compatible records",
            "embedding_format": "optional little-endian float32 sidecar referenced by each JSONL record",
        }
        self.manifest_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    def ingest(
        self,
        rows: Iterable[dict[str, Any]],
        *,
        resume: bool = True,
        max_source_rows: int | None = None,
        checkpoint_every_documents: int = 100,
    ) -> IngestionStats:
        """Ingest a supplied stream.

        On resume the source stream is replayed and completed source rows are
        skipped.  This preserves deterministic chunk IDs without retaining the
        source corpus locally.  Checkpoints are per document by default, so an
        interruption may at worst replay the final document; consumers should
        treat ``chunk_id`` as the deduplication key.
        """
        stats, shard, shard_records = self._load_checkpoint(resume)
        writer = ShardedKnowledgeWriter(self.output_dir, self.config.shard_size, shard, shard_records)
        rows_to_skip = stats.source_rows_seen
        if rows_to_skip:
            log.info(
                "Resuming at source row %d: replaying prior stream rows without chunking or embedding.",
                rows_to_skip,
            )
        docs_since_checkpoint = 0
        try:
            for row_number, row in enumerate(rows):
                if row_number < rows_to_skip:
                    if (row_number + 1) % 50_000 == 0:
                        log.info("Resume replay: reached source row %d/%d.", row_number + 1, rows_to_skip)
                    continue
                if max_source_rows is not None and stats.source_rows_seen >= max_source_rows:
                    break
                stats.source_rows_seen += 1
                text = normalize_text(row.get("text"))
                source = f"fineweb-edu/{self.config.dataset_config}"
                url = normalize_text(row.get("url"))
                document_id = hashlib.sha256(
                    f"{url}:{row_number}:{text[:256]}".encode("utf-8")
                ).hexdigest()[:24]
                chunks = chunk_document(
                    text,
                    source=source,
                    document_id=document_id,
                    config=self.config,
                    url=url,
                    score=row.get("score"),
                )
                if not chunks:
                    stats.skipped_documents += 1
                else:
                    stats.accepted_documents += 1
                    # Target the amount of unique source text, not the larger
                    # total after overlap has intentionally duplicated context.
                    stats.source_estimated_tokens += estimate_tokens(text, self.config.chars_per_token)
                    for first in range(0, len(chunks), self.embedding_batch_size):
                        batch = chunks[first:first + self.embedding_batch_size]
                        vectors = self.embedder.embed_batch([item["text"] for item in batch]) if self.embedder else None
                        if vectors:
                            dimensions = {len(vector) for vector in vectors}
                            if len(dimensions) != 1:
                                raise ValueError("Embedder returned inconsistent vector dimensions")
                            dimension = dimensions.pop()
                            if stats.embedding_dimension not in (None, dimension):
                                raise ValueError("Embedder dimension changed during ingestion")
                            stats.embedding_dimension = dimension
                        writer.write(batch, vectors)
                        stats.chunks_written += len(batch)
                        stats.retrieval_chunk_estimated_tokens += sum(
                            item["metadata"]["estimated_tokens"] for item in batch
                        )
                docs_since_checkpoint += 1
                if docs_since_checkpoint >= checkpoint_every_documents:
                    self._save_checkpoint(stats, writer)
                    self._save_manifest(stats)
                    progress = 100 * stats.source_estimated_tokens / self.config.target_tokens
                    log.info(
                        "FineWeb checkpoint: rows=%d chunks=%d source_tokens≈%d/%d (%.2f%%)",
                        stats.source_rows_seen,
                        stats.chunks_written,
                        stats.source_estimated_tokens,
                        self.config.target_tokens,
                        progress,
                    )
                    docs_since_checkpoint = 0
                if stats.source_estimated_tokens >= self.config.target_tokens:
                    break
        finally:
            self._save_checkpoint(stats, writer)
            self._save_manifest(stats)
            writer.close()
        return stats


def stream_fineweb_edu(split: str = "train", config_name: str = DEFAULT_CONFIG) -> Iterator[dict[str, Any]]:
    """Open FineWeb-Edu lazily; importing this module never starts a download."""
    try:
        from datasets import load_dataset
    except ImportError as exc:
        raise RuntimeError("Install the data dependencies: pip install -r requirements.txt") from exc
    yield from load_dataset(DEFAULT_DATASET, config_name, split=split, streaming=True)
