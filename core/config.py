"""
Project-wide configuration for SovereignAI.

Every phase should import paths and settings from here instead of
hard-coding strings again. This keeps the whole 24-phase project
consistent and makes it obvious, in one place, where everything lives
on disk.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent


@dataclass(frozen=True)
class Paths:
    """Every directory the project reads from or writes to."""

    root: Path = PROJECT_ROOT

    data_raw: Path = PROJECT_ROOT / "data" / "raw"
    data_cleaned: Path = PROJECT_ROOT / "data" / "cleaned"
    data_processed: Path = PROJECT_ROOT / "data" / "processed"
    data_metadata: Path = PROJECT_ROOT / "data" / "metadata"
    data_demo: Path = PROJECT_ROOT / "data" / "demo"

    tokenizer_vocab: Path = PROJECT_ROOT / "tokenizer" / "vocab"

    checkpoints: Path = PROJECT_ROOT / "checkpoints"
    logs: Path = PROJECT_ROOT / "logs"


@dataclass(frozen=True)
class Settings:
    """The single source of truth for project-wide settings."""

    paths: Paths = field(default_factory=Paths)

    # Phase 3 default. Small on purpose: the demo corpus is tiny, so a
    # huge vocabulary would just memorize it instead of learning
    # genuine sub-word merges.
    default_tokenizer_vocab_size: int = 600

    random_seed: int = 42


def get_settings() -> Settings:
    """Return the shared project settings object.

    A plain function (rather than a global singleton variable) keeps
    this easy to reason about and easy to override in tests.
    """
    return Settings()


def ensure_directories(settings: Settings | None = None) -> None:
    """Create every directory referenced by `Settings.paths` if missing."""
    settings = settings or get_settings()
    for value in vars(settings.paths).values():
        Path(value).mkdir(parents=True, exist_ok=True)
