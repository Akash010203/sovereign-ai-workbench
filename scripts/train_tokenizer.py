"""
Phase 3 acceptance script - trains the byte-level BPE tokenizer on the
processed demo corpus and saves it to tokenizer/vocab/.

Run:
    python scripts/train_tokenizer.py
"""
from __future__ import annotations

import sys
import argparse
from itertools import islice
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from core.config import ensure_directories, get_settings  # noqa: E402
from core.corpus import read_lines  # noqa: E402
from core.logging_setup import setup_logging  # noqa: E402
from tokenizer.tokenizer import ByteLevelBPETokenizer, DEFAULT_SPECIAL_TOKENS  # noqa: E402

logger = setup_logging(__name__)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train the project byte-level BPE tokenizer")
    p.add_argument("--train-file", default="data/processed/train.txt")
    p.add_argument("--output", default="tokenizer/vocab/demo_bpe_vocab.json")
    p.add_argument("--vocab-size", type=int, default=None)
    p.add_argument("--max-lines", type=int, default=0,
                   help="Use at most this many nonempty lines (0 = all).")
    p.add_argument("--instruction-format", action="store_true",
                   help="Reserve system/user/assistant control tokens for SFT.")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    settings = get_settings()
    ensure_directories(settings)

    train_path = Path(args.train_file)
    if not train_path.is_absolute():
        train_path = settings.paths.root / train_path
    if not train_path.exists():
        logger.error(
            "%s not found. Run `python scripts/prepare_data.py` first.",
            train_path,
        )
        return

    corpus_lines = read_lines(train_path)
    if args.max_lines > 0:
        corpus_lines = list(islice(corpus_lines, args.max_lines))
    if not corpus_lines:
        logger.error("No non-empty lines found in %s", train_path)
        return
    logger.info("Training BPE tokenizer on %d lines...", len(corpus_lines))

    special_tokens = list(DEFAULT_SPECIAL_TOKENS)
    if args.instruction_format:
        special_tokens.extend(["<|system|>", "<|user|>", "<|assistant|>"])
    tokenizer = ByteLevelBPETokenizer.train(
        corpus_lines,
        vocab_size=args.vocab_size or settings.default_tokenizer_vocab_size,
        special_tokens=special_tokens,
    )

    out_path = Path(args.output)
    if not out_path.is_absolute():
        out_path = settings.paths.root / out_path
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
