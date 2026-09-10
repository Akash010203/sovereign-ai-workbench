"""
Byte-level Byte Pair Encoding (BPE) tokenizer -- implemented from
scratch for the SovereignAI project (SIH 2026, PS: SIH 117).

WHAT ARE TOKENS, AND WHY DO WE NEED THEM?
------------------------------------------
A neural network only understands numbers, not letters. A tokenizer is
the translation layer: it turns text into a list of integers ("token
IDs") the model can consume, and turns the model's predicted integers
back into text. How you choose to split text into pieces (tokens)
directly affects how efficiently the model can represent language.

WHY BYTE-LEVEL?
---------------
Every possible piece of text, in any language, is ultimately a
sequence of UTF-8 bytes. If the tokenizer's starting alphabet is the
256 possible byte values, ANY string can always be represented -- there
is never an "unknown character" it cannot encode. This is the same
idea used by modern tokenizers (e.g. the GPT-2 family), and it keeps
the base vocabulary small and completely dependency-free (no external
tokenizer library is used anywhere in this file).

WHAT DOES BPE DO?
-----------------
Byte Pair Encoding starts from single bytes and repeatedly merges the
MOST FREQUENT adjacent pair of symbols into one new symbol. After
enough merges, common sub-words (" the", "ing", " and") become single
tokens, while rare sequences fall back to individual bytes. This is
what makes BPE efficient: common text needs far fewer tokens than raw
bytes would.

VOCABULARY LAYOUT
------------------
    id  0 .. len(special_tokens)-1   -> special tokens (<pad>, <unk>, ...)
    id  len(special_tokens) .. +255  -> the 256 raw byte values
    id  len(special_tokens)+256 ..   -> learned merges, in the order
                                        they were learned (earliest
                                        merge = lowest "rank")

This module implements ENCODING and DECODING using an already-trained
vocabulary + merge list. *Learning* the merges happens in
``tokenizer/trainer.py``.
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path

# A simplified, dependency-free "pre-tokenizer" pattern: an optional
# leading space followed by a run of word characters, OR an optional
# leading space followed by a run of punctuation/symbol characters, OR
# a run of leftover whitespace.
#
# Real production tokenizers (e.g. GPT-2) use a more elaborate
# Unicode-property-aware regex and an external `regex` package. This
# version is a deliberate simplification: it keeps the implementation
# understandable and dependency-free while still splitting normal
# English/technical text into sensible chunks before byte-pair merging
# is applied.
PRETOKEN_PATTERN = re.compile(r" ?\w+| ?[^\sA-Za-z0-9_]+|\s+")

DEFAULT_SPECIAL_TOKENS = ["<pad>", "<unk>", "<bos>", "<eos>"]

BASE_BYTE_VOCAB_SIZE = 256


@dataclass
class TokenizerVocab:
    """Everything needed to encode/decode: specials, bytes, merges."""

    special_tokens: list[str]
    id_to_bytes: dict[int, bytes]
    merges: list[tuple[bytes, bytes]]

    def __post_init__(self) -> None:
        self.special_to_id: dict[str, int] = {
            token: index for index, token in enumerate(self.special_tokens)
        }
        self.bytes_to_id: dict[bytes, int] = {
            raw_bytes: token_id for token_id, raw_bytes in self.id_to_bytes.items()
        }
        self.merge_rank: dict[tuple[bytes, bytes], int] = {
            pair: rank for rank, pair in enumerate(self.merges)
        }

    @property
    def vocab_size(self) -> int:
        return len(self.special_tokens) + len(self.id_to_bytes)


class ByteLevelBPETokenizer:
    """Encode/decode text using a trained byte-level BPE vocabulary."""

    def __init__(self, vocab: TokenizerVocab):
        self.vocab = vocab

    # ------------------------------------------------------------------
    # Construction
    # ------------------------------------------------------------------
    @classmethod
    def train(
        cls,
        corpus_lines: list[str],
        vocab_size: int,
        special_tokens: list[str] | None = None,
        min_pair_frequency: int = 2,
    ) -> "ByteLevelBPETokenizer":
        """Train a brand-new tokenizer on ``corpus_lines``.

        See ``tokenizer/trainer.py`` for the actual learning algorithm.
        """
        from tokenizer.trainer import train_bpe  # local import: avoids a
        # circular import between tokenizer.py <-> trainer.py

        vocab = train_bpe(
            corpus_lines,
            vocab_size=vocab_size,
            special_tokens=special_tokens,
            min_pair_frequency=min_pair_frequency,
        )
        return cls(vocab)

    @classmethod
    def load(cls, path: str | Path) -> "ByteLevelBPETokenizer":
        """Load a tokenizer previously saved with :meth:`save`."""
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        id_to_bytes = {
            int(token_id): bytes(byte_values)
            for token_id, byte_values in data["id_to_bytes"].items()
        }
        merges = [
            (bytes(left), bytes(right)) for left, right in data["merges"]
        ]
        vocab = TokenizerVocab(
            special_tokens=data["special_tokens"],
            id_to_bytes=id_to_bytes,
            merges=merges,
        )
        return cls(vocab)

    def save(self, path: str | Path) -> None:
        """Persist this tokenizer's vocabulary + merges as JSON.

        Byte sequences are stored as plain lists of integers (0-255)
        since JSON has no native "bytes" type -- this keeps the file
        human-inspectable and avoids any Unicode-escaping tricks.
        """
        data = {
            "special_tokens": self.vocab.special_tokens,
            "id_to_bytes": {
                str(token_id): list(raw_bytes)
                for token_id, raw_bytes in self.vocab.id_to_bytes.items()
            },
            "merges": [
                [list(left), list(right)] for left, right in self.vocab.merges
            ],
        }
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(data, indent=2), encoding="utf-8")

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    @property
    def vocab_size(self) -> int:
        return self.vocab.vocab_size

    def encode(
        self, text: str, add_bos: bool = False, add_eos: bool = False
    ) -> list[int]:
        """Convert ``text`` into a list of token IDs.

        Shape note: the result is a 1-D Python list of length T (the
        number of tokens produced). When batched later for training,
        this becomes the ``[B, T]`` tensor described in
        ``docs/ARCHITECTURE.md``.
        """
        ids: list[int] = []
        if add_bos:
            ids.append(self.vocab.special_to_id["<bos>"])

        for piece in PRETOKEN_PATTERN.findall(text):
            piece_bytes = piece.encode("utf-8")
            for symbol in self._apply_bpe_to_word(piece_bytes):
                ids.append(self.vocab.bytes_to_id[symbol])

        if add_eos:
            ids.append(self.vocab.special_to_id["<eos>"])
        return ids

    def decode(self, ids: list[int]) -> str:
        """Convert a list of token IDs back into text.

        Special tokens are silently dropped from the output text (they
        are structural markers, not content).
        """
        special_ids = set(self.vocab.special_to_id.values())
        raw = bytearray()
        for token_id in ids:
            if token_id in special_ids:
                continue
            raw.extend(self.vocab.id_to_bytes[token_id])
        return bytes(raw).decode("utf-8", errors="replace")

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------
    def _apply_bpe_to_word(self, word_bytes: bytes) -> list[bytes]:
        """Apply learned merges to one pre-tokenized word's bytes.

        Starts as a list of individual bytes; repeatedly merges the
        adjacent pair with the LOWEST merge rank (earliest-learned =
        most broadly useful) until no learned merge applies to what is
        left. This mirrors, symbol-for-symbol, the order merges were
        *learned* in during training.
        """
        symbols: list[bytes] = [bytes([b]) for b in word_bytes]

        while len(symbols) > 1:
            pairs = list(zip(symbols[:-1], symbols[1:]))
            ranked_candidates = [
                (self.vocab.merge_rank[pair], index)
                for index, pair in enumerate(pairs)
                if pair in self.vocab.merge_rank
            ]
            if not ranked_candidates:
                break

            _, best_index = min(ranked_candidates)
            merged = symbols[best_index] + symbols[best_index + 1]
            symbols = symbols[:best_index] + [merged] + symbols[best_index + 2 :]

        return symbols
