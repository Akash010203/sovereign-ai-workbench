"""Download the blend's source files once, resumably, for offline preparation.

This avoids ``datasets`` streaming/range-cache failures on unstable networks.
The default downloads the exact three files required by the 45/35/20 blend:
the maintenance Parquet file, OpenOrca's 1M-GPT4-Augmented Parquet file, and
the whole NVIDIA SFT/code JSONL file.  It does *not* download NVIDIA's complete
~130 GB release.  Use --all-nemotron only when that is explicitly desired.
"""
from __future__ import annotations

import argparse
import logging
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

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


def main() -> None:
    parser = argparse.ArgumentParser(description="Download blend source files for offline preparation")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "data" / "raw" / "hf_sources")
    parser.add_argument("--all-nemotron", action="store_true",
                        help="Download every NVIDIA dataset file (~130 GB), not recommended for an 80M model.")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    download_sources(args.output_dir, args.all_nemotron)
    print(f"Download complete: {args.output_dir}")


if __name__ == "__main__":
    main()
