"""
training/dataset.py — Tokenized dataset for MiniLLM training.

HOW AUTOREGRESSIVE TRAINING DATA IS STRUCTURED
------------------------------------------------
Language-model training uses a simple trick: every token in a sequence
is both an *input* (what the model sees) and a *target* (what it must
predict — the very next token).

Given a tokenized sequence of length N:
    tokens = [t0, t1, t2, t3, ..., t_{N-1}]

We create sliding windows of length (block_size + 1):
    window = [t_i, t_{i+1}, ..., t_{i+block_size}]

Then split each window into:
    input_ids = window[:-1]   = [t_i,   ..., t_{i+block_size-1}]
    targets   = window[1:]    = [t_{i+1},..., t_{i+block_size}  ]

At every position k in the sequence, the model sees input_ids[0..k]
and must predict targets[k] (= the next token).  The causal mask in
the attention layer enforces that at position k, only positions 0..k
are visible.

SHAPE PRODUCED
--------------
    input_ids : [block_size]  (int64)
    targets   : [block_size]  (int64)

The DataLoader batches these to [B, block_size].
"""
from __future__ import annotations

import json
import logging
import math
from pathlib import Path
from typing import Optional

import torch
from torch.utils.data import Dataset, DataLoader, IterableDataset

log = logging.getLogger(__name__)


class TokenizedTextDataset(Dataset):
    """
    Sliding-window dataset built from a plain-text file.

    The file is read line by line, each line is encoded with the
    tokenizer, and the token IDs are concatenated into one long flat
    sequence.  A sliding window of (block_size + 1) is stepped across
    that sequence to produce (input, target) pairs.

    Args:
        text_path:   Path to a plain-text file (one document per line).
        tokenizer:   A loaded ``ByteLevelBPETokenizer`` instance.
        block_size:  Number of tokens per training sample (= max_seq_len).
        stride:      Step between successive windows.  Defaults to
                     block_size (non-overlapping).  Set < block_size
                     for overlapping windows on tiny corpora.
        add_bos_eos: Whether to wrap each line with <bos>/<eos> tokens.
    """

    def __init__(
        self,
        text_path: str | Path,
        tokenizer,
        block_size: int,
        stride: Optional[int] = None,
        add_bos_eos: bool = True,
    ) -> None:
        self.block_size = block_size
        stride = stride if stride is not None else block_size

        # ── Encode the entire corpus into a flat token sequence ─────────
        text_path = Path(text_path)
        lines = [
            line.strip()
            for line in text_path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]

        all_ids: list[int] = []
        for line in lines:
            all_ids.extend(
                tokenizer.encode(line, add_bos=add_bos_eos, add_eos=add_bos_eos)
            )

        if len(all_ids) < block_size + 1:
            # Corpus too short for even one window — repeat it.
            repeats = math.ceil((block_size + 2) / max(len(all_ids), 1))
            all_ids = all_ids * repeats
            log.warning(
                "Corpus produced only %d tokens, fewer than block_size+1=%d. "
                "Repeating corpus %d× to produce training windows.",
                len(all_ids) // repeats,
                block_size + 1,
                repeats,
            )

        self.token_ids = torch.tensor(all_ids, dtype=torch.long)

        # ── Build the window start positions ─────────────────────────────
        total = len(self.token_ids)
        self.starts = list(range(0, total - block_size, stride))
        log.info(
            "Dataset: %d raw tokens, block_size=%d, stride=%d → %d samples",
            total,
            block_size,
            stride,
            len(self.starts),
        )

    def __len__(self) -> int:
        return len(self.starts)

    def __getitem__(self, idx: int):
        start = self.starts[idx]
        chunk = self.token_ids[start : start + self.block_size + 1]
        return chunk[:-1].clone(), chunk[1:].clone()


class StreamingTokenizedTextDataset(IterableDataset):
    """Tokenize a line-oriented corpus incrementally instead of storing it.

    This is intended for the multi-million-row blended corpus.  Only a small
    token buffer (at most a few documents) exists in host RAM, and no dataset
    records are transferred to VRAM until a DataLoader batch is consumed.
    The existing materialized dataset remains faster for small demo corpora.
    """

    def __init__(self, text_path: str | Path, tokenizer, block_size: int, add_bos_eos: bool = True) -> None:
        super().__init__()
        self.text_path = Path(text_path)
        self.tokenizer = tokenizer
        self.block_size = block_size
        self.add_bos_eos = add_bos_eos

    def __iter__(self):
        token_buffer: list[int] = []
        with self.text_path.open("r", encoding="utf-8") as corpus_file:
            for line in corpus_file:
                text = line.strip()
                if not text:
                    continue
                token_buffer.extend(self.tokenizer.encode(
                    text, add_bos=self.add_bos_eos, add_eos=self.add_bos_eos
                ))
                while len(token_buffer) >= self.block_size + 1:
                    chunk = torch.tensor(token_buffer[: self.block_size + 1], dtype=torch.long)
                    del token_buffer[: self.block_size]
                    yield chunk[:-1], chunk[1:]


def make_dataloader(
    dataset: TokenizedTextDataset,
    batch_size: int,
    shuffle: bool = True,
    num_workers: int = 0,
) -> DataLoader:
    """Create a standard DataLoader for a TokenizedTextDataset."""
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        drop_last=True,   # keeps all batches exactly batch_size
    )


def make_streaming_dataloader(dataset: StreamingTokenizedTextDataset, batch_size: int) -> DataLoader:
    """Create a memory-bounded loader for a ``StreamingTokenizedTextDataset``."""
    return DataLoader(
        dataset,
        batch_size=batch_size,
        num_workers=0,  # Multiple workers would duplicate a sequential corpus.
        pin_memory=torch.cuda.is_available(),
        drop_last=True,
    )
