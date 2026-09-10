"""
Token embedding layer for the custom MiniLLM.

WHAT THIS DOES
--------------
Turns a batch of token IDs, shape ``[B, T]`` (B = batch size,
T = sequence length), into a batch of dense vectors, shape
``[B, T, C]`` (C = d_model), by looking each ID up in a learned weight
matrix of shape ``[vocab_size, C]``.

This is deliberately a thin wrapper around ``torch.nn.Embedding`` -- a
lookup table is exactly what that primitive is, and storing/learning
one is not something that needs to be reinvented (PyTorch primitives
like this are explicitly allowed by the project's build contract).
What makes this project's Transformer "from scratch" is everything
*around* this lookup table: the attention math, the normalization, the
positional encoding, and the feed-forward network, all implemented
explicitly by hand in the other files in this package.
"""
from __future__ import annotations

import torch
import torch.nn as nn

from models.custom_minilm.config import MiniLLMConfig


class TokenEmbedding(nn.Module):
    def __init__(self, config: MiniLLMConfig):
        super().__init__()
        self.embedding = nn.Embedding(config.vocab_size, config.d_model)

    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        """``token_ids``: ``[B, T]`` (int64) -> returns ``[B, T, C]``."""
        if token_ids.ndim != 2:
            raise ValueError(
                f"token_ids must have shape [B, T], got {tuple(token_ids.shape)}"
            )
        return self.embedding(token_ids)

    @property
    def weight(self) -> torch.Tensor:
        """Exposed so the language-model head can tie its weights to
        this table in Phase 5 -- a standard, parameter-saving trick,
        not something that needs its own from-scratch implementation.
        """
        return self.embedding.weight
