"""Fast, local lexical fallback for the disk-backed FineWeb knowledge corpus.

The existing corpus was built with a retired 384-dimensional embedding format.
This reader searches the local JSONL text directly, so it remains useful while
the corpus is rebuilt with the project's custom embedding implementation.
"""
from __future__ import annotations

import json
import re
import subprocess
from pathlib import Path

from rag.retriever import RetrievalResult

_STOP_WORDS = {
    "about", "after", "also", "and", "are", "can", "could", "does", "for",
    "from", "have", "how", "into", "is", "its", "of", "on", "please", "the",
    "this", "to", "what", "when", "where", "which", "with", "would", "you",
}


def _tokens(text: str) -> list[str]:
    """Return meaningful, spelling-normalized search terms."""
    normalized = text.lower().replace("calliper", "caliper")
    words = re.findall(r"[a-z0-9]+", normalized)
    return [word.rstrip("s") for word in words if len(word) >= 4 and word not in _STOP_WORDS]


class FineWebLexicalRetriever:
    """Search locally stored FineWeb chunks with ripgrep and rank token overlap."""

    def __init__(self, root: str | Path) -> None:
        self.root = Path(root)

    def retrieve(self, query: str, top_k: int = 2) -> list[RetrievalResult]:
        terms = list(dict.fromkeys(_tokens(query)))
        if not terms or not self.root.is_dir():
            return []

        pattern = "|".join(re.escape(term) for term in terms)
        try:
            completed = subprocess.run(
                [
                    "rg", "--json", "--ignore-case", "--max-count", "1",
                    "--glob", "chunks-*.jsonl", pattern, str(self.root),
                ],
                capture_output=True,
                text=True,
                encoding="utf-8",
                errors="replace",
                timeout=12,
                check=False,
            )
        except (FileNotFoundError, subprocess.TimeoutExpired):
            return []

        candidates: list[RetrievalResult] = []
        seen: set[str] = set()
        phrase = " ".join(terms)
        for line in completed.stdout.splitlines():
            try:
                event = json.loads(line)
                if event.get("type") != "match":
                    continue
                record = json.loads(event["data"]["lines"]["text"])
            except (KeyError, TypeError, json.JSONDecodeError):
                continue
            text = str(record.get("text", ""))
            text_terms = set(_tokens(text))
            matched = sum(term in text_terms for term in terms)
            if not matched or record.get("chunk_id") in seen:
                continue
            seen.add(record["chunk_id"])
            phrase_position = text.lower().replace("calliper", "caliper").find(phrase)
            phrase_bonus = 0.5 if phrase_position >= 0 else 0.0
            early_match_bonus = (
                0.25 * (1 - phrase_position / max(len(text), 1))
                if phrase_position >= 0 else 0.0
            )
            candidates.append(RetrievalResult(
                chunk_id=record["chunk_id"],
                source=record.get("source", "fineweb-edu/local"),
                text=text,
                score=(matched / len(terms)) + phrase_bonus + early_match_bonus,
                metadata={**record.get("metadata", {}), "retrieval_mode": "lexical"},
            ))

        return sorted(candidates, key=lambda item: item.score, reverse=True)[:top_k]
