"""
Phase 2 - data pipeline.

    RAW  ->  CLEANED  ->  FILTERED  ->  TRAIN / VAL SPLIT  ->  metadata

Every raw ``.txt`` file in ``data/raw/`` is processed independently for
cleaning, but all filtered lines are pooled together before the
train/validation split, so the demo corpus is treated as one dataset
even if it later grows to multiple source files.

Nothing is downloaded automatically here - see ``data/raw/README.md``
for provenance policy.

Run:
    python scripts/prepare_data.py
"""
from __future__ import annotations

import hashlib
import json
import re
import sys
from pathlib import Path

# Make sure the project root is importable when this script is run
# directly as `python scripts/prepare_data.py` from anywhere.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from core.config import ensure_directories, get_settings  # noqa: E402
from core.corpus import read_lines, write_lines  # noqa: E402
from core.logging_setup import setup_logging  # noqa: E402

logger = setup_logging(__name__)

MIN_LINE_CHARS = 8
MAX_LINE_CHARS = 500
VAL_SPLIT_RATIO = 0.10


def clean_lines(raw_lines: list[str]) -> list[str]:
    """Collapse repeated whitespace and drop lines that become empty."""
    cleaned = []
    for line in raw_lines:
        line = re.sub(r"\s+", " ", line).strip()
        if line:
            cleaned.append(line)
    return cleaned


def filter_lines(cleaned_lines: list[str]) -> list[str]:
    """Keep only lines within a sane length range for training text."""
    return [
        line
        for line in cleaned_lines
        if MIN_LINE_CHARS <= len(line) <= MAX_LINE_CHARS
    ]


def split_train_val(
    lines: list[str], val_ratio: float
) -> tuple[list[str], list[str]]:
    if not lines:
        return [], []
    val_size = max(1, int(len(lines) * val_ratio))
    return lines[val_size:], lines[:val_size]


def sha256_of_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_metadata(settings, raw_path: Path, stage_counts: dict[str, int]) -> None:
    metadata = {
        "source_file": str(raw_path.relative_to(settings.paths.root)),
        "source": "synthetic - authored for this project (SIH 2026 demo)",
        "license": "CC0-1.0 (project-owned synthetic text)",
        "sha256": sha256_of_file(raw_path),
        "processing_notes": (
            f"Lines normalized (whitespace collapsed), filtered to "
            f"{MIN_LINE_CHARS}-{MAX_LINE_CHARS} characters, pooled with "
            f"any other raw files, then split into train/validation "
            f"with a {VAL_SPLIT_RATIO:.0%} validation ratio."
        ),
        "stage_counts": stage_counts,
    }
    out_path = settings.paths.data_metadata / f"{raw_path.stem}.json"
    out_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    logger.info("Wrote metadata: %s", out_path)


def main() -> None:
    settings = get_settings()
    ensure_directories(settings)

    raw_files = sorted(settings.paths.data_raw.glob("*.txt"))
    if not raw_files:
        logger.error(
            "No .txt files found in %s. Add a raw corpus first "
            "(see data/raw/README.md).",
            settings.paths.data_raw,
        )
        return

    pooled_filtered_lines: list[str] = []

    for raw_path in raw_files:
        raw_lines = read_lines(raw_path)
        cleaned = clean_lines(raw_lines)
        write_lines(settings.paths.data_cleaned / raw_path.name, cleaned)

        filtered = filter_lines(cleaned)
        pooled_filtered_lines.extend(filtered)

        write_metadata(
            settings,
            raw_path,
            stage_counts={
                "raw_lines": len(raw_lines),
                "cleaned_lines": len(cleaned),
                "filtered_lines": len(filtered),
            },
        )

    train_lines, val_lines = split_train_val(pooled_filtered_lines, VAL_SPLIT_RATIO)
    write_lines(settings.paths.data_processed / "train.txt", train_lines)
    write_lines(settings.paths.data_processed / "val.txt", val_lines)

    logger.info(
        "Processed %d raw file(s) -> %d train line(s), %d val line(s).",
        len(raw_files),
        len(train_lines),
        len(val_lines),
    )
    logger.info("Train file: %s", settings.paths.data_processed / "train.txt")
    logger.info("Val file  : %s", settings.paths.data_processed / "val.txt")


if __name__ == "__main__":
    main()
