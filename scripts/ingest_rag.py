"""
scripts/ingest_rag.py — Bulk-ingest documents into the RAG knowledge base.

This script populates the local vector index with domain knowledge so
the AI workbench can answer questions using real documents rather than
relying solely on the generative model.

Usage:
    python scripts/ingest_rag.py                        # Ingest everything
    python scripts/ingest_rag.py --demo-only             # Only demo files
    python scripts/ingest_rag.py --maintenance-only      # Only maintenance parquet
    python scripts/ingest_rag.py --max-maintenance 5000  # Limit parquet rows

Sources ingested:
    1. data/demo/*.txt         — Demo SOPs, inspection reports, logs
    2. data/cleaned/*.txt      — Cleaned corpus files
    3. data/raw/hf_sources/maintenance/maintenance.parquet — 70MB of maintenance work instructions
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from rag.chunking import chunk_text
from rag.embeddings import get_embedder
from rag.index import LocalVectorIndex
from rag.ingest import DocumentIngester

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-7s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("ingest_rag")


def ingest_text_files(
    ingester: DocumentIngester,
    directories: list[Path],
    extensions: list[str] = None,
) -> int:
    """Ingest all matching text files from the given directories."""
    extensions = extensions or [".txt", ".md"]
    total = 0
    for directory in directories:
        if not directory.exists():
            log.warning("Directory not found: %s — skipping.", directory)
            continue
        for fpath in sorted(directory.rglob("*")):
            if fpath.suffix.lower() in extensions and fpath.stat().st_size > 50:
                try:
                    n = ingester.ingest_file(fpath)
                    log.info("  ✓ %s → %d chunks", fpath.name, n)
                    total += n
                except Exception as exc:
                    log.warning("  ✗ %s — %s", fpath.name, exc)
    return total


def ingest_maintenance_parquet(
    ingester: DocumentIngester,
    parquet_path: Path,
    max_rows: int = 10000,
) -> int:
    """Extract maintenance work instructions from the HuggingFace parquet file."""
    if not parquet_path.exists():
        log.warning("Maintenance parquet not found: %s", parquet_path)
        return 0

    try:
        import pyarrow.parquet as pq
    except ImportError:
        log.warning("pyarrow not installed — trying pandas fallback")
        try:
            import pandas as pd
            df = pd.read_parquet(parquet_path)
        except ImportError:
            log.error(
                "Neither pyarrow nor pandas available. "
                "Install: pip install pyarrow  OR  pip install pandas"
            )
            return 0
    else:
        table = pq.read_table(parquet_path)
        # Convert to list of dicts for processing
        df = table.to_pandas()

    log.info("Maintenance parquet: %d rows total, processing up to %d", len(df), max_rows)

    # Find the text column(s) — maintenance datasets use domain-specific names
    text_cols = []
    # Priority: combine description columns for richer context
    maintenance_cols = [
        "WorkOrderDescription", "OperationDescription",
        "Maintenance_activity_type", "Equipment_ID", "OrderType",
    ]
    for col in maintenance_cols:
        if col in df.columns:
            text_cols.append(col)

    # Fallback to generic column names
    if not text_cols:
        for candidate in ["text", "instruction", "output", "response", "content", "input"]:
            if candidate in df.columns:
                text_cols = [candidate]
                break
    if not text_cols:
        # Use the first string column
        for col in df.columns:
            if df[col].dtype == "object":
                text_cols = [col]
                break
    if not text_cols:
        log.error("Could not identify text column in parquet. Columns: %s", list(df.columns))
        return 0

    log.info("Using columns %s for ingestion", text_cols)

    total_chunks = 0
    rows = df.head(max_rows)
    batch_size = 100
    batch_texts = []
    batch_sources = []

    for idx, row in rows.iterrows():
        # Combine all text columns into one rich document
        parts = []
        for col in text_cols:
            val = str(row.get(col, "")).strip()
            if val and val.lower() != "nan":
                parts.append(f"{col}: {val}")
        text = " | ".join(parts)

        if len(text) < 50:  # Skip trivially short entries
            continue

        # Build a richer source identifier
        source = f"maintenance_parquet_row{idx}"
        batch_texts.append(text)
        batch_sources.append(source)

        if len(batch_texts) >= batch_size:
            n = _ingest_batch(ingester, batch_texts, batch_sources)
            total_chunks += n
            batch_texts.clear()
            batch_sources.clear()
            if total_chunks % 500 == 0:
                log.info("  ... %d chunks ingested so far", total_chunks)

    # Final batch
    if batch_texts:
        total_chunks += _ingest_batch(ingester, batch_texts, batch_sources)

    return total_chunks


def _ingest_batch(
    ingester: DocumentIngester,
    texts: list[str],
    sources: list[str],
) -> int:
    """Ingest a batch of texts.

    Uses a large chunk_size since maintenance work order descriptions
    are typically short (~100-300 chars) and should NOT be split.
    """
    total = 0
    for text, source in zip(texts, sources):
        try:
            # Large chunk_size = keep each row as a single chunk (no splitting)
            n = ingester.ingest_text(text, source=source, chunk_size=2000, overlap=0)
            total += n
        except Exception as exc:
            log.warning("Failed to ingest %s: %s", source, exc)
    return total


def main() -> None:
    parser = argparse.ArgumentParser(description="Bulk RAG ingestion")
    parser.add_argument("--demo-only", action="store_true",
                        help="Only ingest demo + cleaned text files")
    parser.add_argument("--maintenance-only", action="store_true",
                        help="Only ingest maintenance parquet")
    parser.add_argument("--max-maintenance", type=int, default=5000,
                        help="Max rows from maintenance parquet (default: 5000)")
    parser.add_argument("--fresh", action="store_true",
                        help="Start with empty index (don't load existing)")
    parser.add_argument("--index-path", type=str,
                        default=str(ROOT / "data" / "rag_index.json"),
                        help="Path to save/load the RAG index")
    args = parser.parse_args()

    index_path = Path(args.index_path)
    index = LocalVectorIndex()

    if not args.fresh and index_path.exists():
        index.load(index_path)
        log.info("Loaded existing index: %d chunks", len(index))
    else:
        log.info("Starting with empty index.")

    ingester = DocumentIngester(index)
    start = time.time()

    total_new = 0

    # 1. Text files (demo + cleaned)
    if not args.maintenance_only:
        log.info("─── Ingesting text files ───")
        text_dirs = [
            ROOT / "data" / "demo",
            ROOT / "data" / "cleaned",
        ]
        n = ingest_text_files(ingester, text_dirs)
        total_new += n
        log.info("Text files: %d chunks added.", n)

    # 2. Maintenance parquet
    if not args.demo_only:
        parquet_path = ROOT / "data" / "raw" / "hf_sources" / "maintenance" / "maintenance.parquet"
        if parquet_path.exists():
            log.info("─── Ingesting maintenance parquet (%d max rows) ───", args.max_maintenance)
            n = ingest_maintenance_parquet(ingester, parquet_path, max_rows=args.max_maintenance)
            total_new += n
            log.info("Maintenance parquet: %d chunks added.", n)
        else:
            log.info("Maintenance parquet not found, skipping.")

    # Save
    index.save(index_path)
    elapsed = time.time() - start
    log.info("═══ Done: %d new chunks, %d total in index, %.1fs elapsed ═══",
             total_new, len(index), elapsed)


if __name__ == "__main__":
    main()
