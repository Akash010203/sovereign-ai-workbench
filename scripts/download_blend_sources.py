"""Download the blend's source files once, resumably, for offline preparation.

This avoids ``datasets`` streaming/range-cache failures on unstable networks.
The default downloads the exact files required by the 4-source blend:
the maintenance Parquet file, OpenOrca's 1M-GPT4-Augmented Parquet file,
the NVIDIA SFT/code JSONL file, and a subset of FineWeb-Edu.

FineWeb-Edu (HuggingFaceFW/fineweb-edu) is a curated educational web text
corpus.  Because it is very large (1.3T tokens), we stream a configurable
subset (default 400K rows ≈ 1.6B tokens) and save it locally as Parquet.
"""
from __future__ import annotations

import argparse
import logging
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

# Direct-download sources (single files)
SOURCES = {
    "maintenance": ("Jvachier/industrial-maintenance-synthetic", ["maintenance.parquet"]),
    "nemotron": ("nvidia/Llama-Nemotron-Post-Training-Dataset", ["SFT/code/code_v1.1.jsonl"]),
    "openorca": ("Open-Orca/OpenOrca", ["1M-GPT4-Augmented.parquet"]),
}


def download_sources(destination: Path, all_nemotron: bool = False) -> None:
    try:
        from huggingface_hub import snapshot_download
    except ImportError as exc:
        raise RuntimeError("Install dependencies first: pip install -r requirements.txt") from exc

    for name, (repo_id, patterns) in SOURCES.items():
        selected_patterns = ["**"] if name == "nemotron" and all_nemotron else patterns
        local_dir = destination / name
        logging.info("Downloading %s into %s (resume is automatic)", repo_id, local_dir)
        snapshot_download(
            repo_id=repo_id,
            repo_type="dataset",
            allow_patterns=selected_patterns,
            local_dir=local_dir,
            max_workers=1,  # Stable for a laptop and an interrupted connection.
        )


def download_fineweb_edu(destination: Path, max_rows: int = 400_000) -> None:
    """Stream a subset of FineWeb-Edu and save it as a local Parquet file.

    FineWeb-Edu is too large to download in full (~1.3T tokens).  This
    streams ``max_rows`` rows from the 'default' configuration and writes
    them to a Parquet file that the blending pipeline can consume offline.
    """
    try:
        from datasets import load_dataset
    except ImportError as exc:
        raise RuntimeError("pip install datasets") from exc

    output_dir = destination / "fineweb_edu"
    output_dir.mkdir(parents=True, exist_ok=True)
    output_file = output_dir / "fineweb_edu_subset.parquet"

    if output_file.exists():
        logging.info("FineWeb-Edu subset already exists at %s — skipping.", output_file)
        return

    logging.info(
        "Streaming %d rows from HuggingFaceFW/fineweb-edu (this may take a while)...",
        max_rows,
    )

    # Stream the dataset to avoid loading the entire thing into memory.
    ds = load_dataset(
        "HuggingFaceFW/fineweb-edu",
        "default",
        split="train",
        streaming=True,
    )

    rows = []
    for i, row in enumerate(ds):
        if i >= max_rows:
            break
        # Keep only the text column + metadata we need
        rows.append({
            "text": row.get("text", ""),
            "score": row.get("score", 0.0),     # Educational quality score
            "url": row.get("url", ""),
        })
        if (i + 1) % 10_000 == 0:
            logging.info("  ... streamed %d / %d rows", i + 1, max_rows)

    logging.info("Writing %d rows to %s ...", len(rows), output_file)
    try:
        import pyarrow as pa
        import pyarrow.parquet as pq
        table = pa.Table.from_pylist(rows)
        pq.write_table(table, output_file)
    except ImportError:
        # Fallback: use pandas
        import pandas as pd
        pd.DataFrame(rows).to_parquet(output_file, index=False)

    logging.info("FineWeb-Edu subset saved: %s (%d rows)", output_file, len(rows))


def main() -> None:
    parser = argparse.ArgumentParser(description="Download blend source files for offline preparation")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "data" / "raw" / "hf_sources")
    parser.add_argument("--all-nemotron", action="store_true",
                        help="Download every NVIDIA dataset file (~130 GB), not recommended for an 80M model.")
    parser.add_argument("--fineweb-rows", type=int, default=400_000,
                        help="Number of FineWeb-Edu rows to stream (default: 400000, ~1.6B tokens)")
    parser.add_argument("--skip-fineweb", action="store_true",
                        help="Skip downloading FineWeb-Edu (use existing 3-source blend)")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

    download_sources(args.output_dir, args.all_nemotron)

    if not args.skip_fineweb:
        download_fineweb_edu(args.output_dir, max_rows=args.fineweb_rows)

    print(f"Download complete: {args.output_dir}")


if __name__ == "__main__":
    main()

