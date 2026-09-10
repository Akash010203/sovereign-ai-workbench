"""
rag/chunking.py — Split documents into chunks for RAG indexing.

WHY CHUNKING?
-------------
Embedding models and context windows have size limits.
A 50-page manual cannot be fed whole to the model.
Instead, we split it into overlapping chunks of ~200-500 tokens,
embed each chunk, and at query time retrieve only the most
relevant chunks.

CHUNKING STRATEGIES (implemented here)
---------------------------------------
1. FIXED-SIZE: split on character count with overlap.
   Simple and fast. Used for initial RAG.

2. SENTENCE-BOUNDARY: split on sentence endings (., !, ?)
   before the hard character limit.
   Better for coherence.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field


@dataclass
class TextChunk:
    """One chunk of a document, ready for embedding."""
    chunk_id:   str
    source:     str          # filename or document identifier
    text:       str
    char_start: int = 0
    char_end:   int = 0
    metadata:   dict = field(default_factory=dict)


def chunk_text(
    text: str,
    source: str,
    chunk_size: int = 400,
    overlap: int = 80,
    strategy: str = "sentence",
) -> list[TextChunk]:
    """
    Split ``text`` into overlapping TextChunk objects.

    Args:
        text:       The full document text to chunk.
        source:     Identifier for the source document.
        chunk_size: Target character size per chunk.
        overlap:    Number of characters shared between adjacent chunks.
        strategy:   'fixed' or 'sentence'.

    Returns:
        List of TextChunk objects.
    """
    text = text.strip()
    if not text:
        return []

    if strategy == "sentence":
        return _sentence_chunks(text, source, chunk_size, overlap)
    return _fixed_chunks(text, source, chunk_size, overlap)


def _fixed_chunks(
    text: str, source: str, chunk_size: int, overlap: int
) -> list[TextChunk]:
    chunks = []
    start  = 0
    idx    = 0
    while start < len(text):
        end  = min(start + chunk_size, len(text))
        chunk_text_str = text[start:end]
        chunks.append(TextChunk(
            chunk_id=f"{source}_chunk{idx:04d}",
            source=source,
            text=chunk_text_str,
            char_start=start,
            char_end=end,
        ))
        idx   += 1
        start += chunk_size - overlap
    return chunks


def _sentence_chunks(
    text: str, source: str, chunk_size: int, overlap: int
) -> list[TextChunk]:
    """Split on sentence boundaries within the chunk_size limit."""
    sentence_ends = [m.end() for m in re.finditer(r"[.!?]\s+", text)]
    sentence_ends.append(len(text))

    chunks = []
    idx    = 0
    start  = 0

    while start < len(text):
        # Find the largest chunk ≤ chunk_size ending at a sentence boundary
        end = start + chunk_size
        if end >= len(text):
            end = len(text)
        else:
            # Walk back to the last sentence boundary before end
            candidates = [s for s in sentence_ends if s <= end and s > start]
            if candidates:
                end = candidates[-1]

        chunk_text_str = text[start:end].strip()
        if chunk_text_str:
            chunks.append(TextChunk(
                chunk_id=f"{source}_chunk{idx:04d}",
                source=source,
                text=chunk_text_str,
                char_start=start,
                char_end=end,
            ))
            idx += 1
        start = max(end - overlap, start + 1)

    return chunks
