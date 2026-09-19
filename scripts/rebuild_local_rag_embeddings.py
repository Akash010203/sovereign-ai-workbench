"""Rebuild the editable local RAG index with the project's custom TF-IDF vectors.

The source index remains untouched. This migration writes a sibling index so a
previous embedding backend can be retired without losing any document text.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from rag.embeddings import get_embedder


def rebuild(source: Path, destination: Path) -> int:
    """Copy records while replacing only their embedding vectors."""
    payload = json.loads(source.read_text(encoding="utf-8"))
    records = payload.get("records", [])
    embedder = get_embedder()

    for number, record in enumerate(records, 1):
        record["embedding"] = embedder.embed(record.get("text", ""))
        if number % 1000 == 0:
            print(f"Re-embedded {number:,}/{len(records):,} local chunks")

    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps({"records": records}), encoding="utf-8")
    return len(records)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=ROOT / "data" / "rag_index.json")
    parser.add_argument("--output", type=Path, default=ROOT / "data" / "rag_index_tfidf.json")
    args = parser.parse_args()

    if not args.source.is_file():
        raise SystemExit(f"Source index not found: {args.source}")
    count = rebuild(args.source, args.output)
    print(f"Wrote {count:,} custom TF-IDF vectors to {args.output}")


if __name__ == "__main__":
    main()
