"""Centralized logging setup used across every SovereignAI phase."""
from __future__ import annotations

import logging
import sys

from core.config import get_settings


def setup_logging(name: str, level: int = logging.INFO) -> logging.Logger:
    """Configure and return a logger that writes to console + a log file.

    Args:
        name: logger name, typically ``__name__`` of the calling module.
        level: logging level (default ``INFO``).
    """
    # Windows PowerShell sessions can still expose a cp1252 text stream.
    # Logging must never crash or emit a logging traceback merely because a
    # message contains a Unicode arrow from a path/status line.
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(errors="backslashreplace")
    settings = get_settings()
    settings.paths.logs.mkdir(parents=True, exist_ok=True)

    logger = logging.getLogger(name)
    logger.setLevel(level)

    if logger.handlers:
        # Avoid duplicate handlers if setup_logging() is called twice
        # for the same logger name (e.g. imported from two scripts).
        return logger

    formatter = logging.Formatter(
        "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s"
    )

    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    file_handler = logging.FileHandler(
        settings.paths.logs / "sovereign_ai.log", encoding="utf-8"
    )
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    return logger
