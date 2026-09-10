"""
Phase 3 acceptance script - trains the byte-level BPE tokenizer on the
processed demo corpus and saves it to tokenizer/vocab/.

Run:
    python scripts/train_tokenizer.py
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from core.config import ensure_directories, get_settings  # noqa: E402
from core.corpus import read_lines  # noqa: E402
from core.logging_setup import setup_logging  # noqa: E402
from tokenizer.tokenizer import ByteLevelBPETokenizer  # noqa: E402

logger = setup_logging(__name__)


def main() -> None:
    settings = get_settings()
    ensure_directories(settings)

    train_path = settings.paths.data_processed / "train.txt"
    if not train_path.exists():
        logger.error(
            "%s not found. Run `python scripts/prepare_data.py` first.",
            train_path,
        )
        return

    corpus_lines = read_lines(train_path)
    logger.info("Training BPE tokenizer on %d lines...", len(corpus_lines))

    tokenizer = ByteLevelBPETokenizer.train(
        corpus_lines,
        vocab_size=settings.default_tokenizer_vocab_size,
    )

    out_path = settings.paths.tokenizer_vocab / "demo_bpe_vocab.json"
    tokenizer.save(out_path)
    logger.info("Saved tokenizer vocabulary -> %s", out_path)
    logger.info("Final vocab size: %d", tokenizer.vocab_size)
    logger.info("Learned merges:   %d", len(tokenizer.vocab.merges))

    sample = corpus_lines[0] if corpus_lines else "hello sovereign ai"
    ids = tokenizer.encode(sample)
    decoded = tokenizer.decode(ids)
    raw_byte_len = len(sample.encode("utf-8"))
    logger.info("Sample text        : %r", sample)
    logger.info("Raw UTF-8 bytes     : %d", raw_byte_len)
    logger.info("Encoded token count : %d", len(ids))
    logger.info("Encoded (first 20) : %s", ids[:20])
    logger.info("Decoded text        : %r", decoded)
    logger.info("Round-trip OK       : %s", decoded == sample)


if __name__ == "__main__":
    main()
