"""rag/ingest.py — Document ingestion pipeline: text → chunks → embeddings → index."""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

from rag.chunking import chunk_text, TextChunk
from rag.embeddings import get_embedder
from rag.index import LocalVectorIndex

log = logging.getLogger(__name__)


class DocumentIngester:
    """
    Ingest documents into the local RAG knowledge base.

    Pipeline:
        file (txt/pdf) → extract text → chunk → embed → add to index
    """

    def __init__(self, index: LocalVectorIndex) -> None:
        self.index    = index
        self.embedder = get_embedder()

    def ingest_text(
        self,
        text: str,
        source: str,
        chunk_size: int = 400,
        overlap: int = 80,
    ) -> int:
        """
        Ingest a raw text string into the index.

        Returns the number of chunks added.
        """
        chunks = chunk_text(text, source, chunk_size, overlap)
        if not chunks:
            return 0
        texts      = [c.text for c in chunks]
        embeddings = self.embedder.embed_batch(texts)
        self.index.add_batch(chunks, embeddings)
        log.info("Ingested '%s': %d chunks", source, len(chunks))
        return len(chunks)

    def ingest_file(self, path: str | Path, chunk_size: int = 400) -> int:
        """
        Ingest a plain-text or PDF file.

        For PDFs, text is extracted with pdfminer if available.
        """
        path = Path(path)
        source = path.name

        if path.suffix.lower() == ".pdf":
            text = self._extract_pdf(path)
        else:
            text = path.read_text(encoding="utf-8", errors="replace")

        return self.ingest_text(text, source=source, chunk_size=chunk_size)

    def ingest_directory(self, directory: str | Path, extensions: list[str] = None) -> int:
        """Ingest all matching files in a directory."""
        extensions = extensions or [".txt", ".pdf", ".md"]
        total = 0
        for file_path in Path(directory).rglob("*"):
            if file_path.suffix.lower() in extensions:
                try:
                    total += self.ingest_file(file_path)
                except Exception as exc:
                    log.warning("Failed to ingest %s: %s", file_path, exc)
        return total

    @staticmethod
    def _extract_pdf(path: Path) -> str:
        try:
            from pdfminer.high_level import extract_text
            return extract_text(str(path)) or ""
        except ImportError:
            log.warning("pdfminer not available — reading PDF as binary text fallback.")
            return path.read_bytes().decode("utf-8", errors="replace")
