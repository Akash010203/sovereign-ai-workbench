"""Shared helpers for reading and writing plain-text corpora."""
from __future__ import annotations

from pathlib import Path


def read_lines(path: str | Path) -> list[str]:
    """Read a UTF-8 text file and return non-empty, stripped lines."""
    text = Path(path).read_text(encoding="utf-8")
    return [line.strip() for line in text.splitlines() if line.strip()]


def write_lines(path: str | Path, lines: list[str]) -> None:
    """Write `lines` to `path`, one per line, creating parent dirs."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")
