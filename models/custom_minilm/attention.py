"""
Causal multi-head self-attention -- implemented from scratch. This
file does NOT use ``torch.nn.MultiheadAttention`` and does NOT use
``torch.nn.functional.scaled_dot_product_attention``; the score /
softmax / weighted-sum math is written out explicitly below, exactly
as required by the project's build contract.

WHAT ATTENTION DOES (intuition)
---------------------------------
For every token, attention asks: "given everything I've seen so far,
which earlier tokens should I pay the most attention to, and how much
of their information should I mix into my own representation?"
Concretely, every token produces a Query vector (what am I looking
for?), a Key vector (what do I offer?), and a Value vector (what
information do I carry?). The dot product of a Query with every Key
gives a similarity score; softmax turns those scores into weights that
sum to 1; and the weighted sum of Values becomes the token's new,
context-aware representation.

WHY CAUSAL?
-----------
This is a *language model*: at generation time, token t must only be
predicted from tokens 1..t, never from the future. The causal mask
enforces this during training too, so the model never "cheats" by
looking ahead at the answer. ``tests/test_attention.py`` checks this
directly: changing a future token must never change an earlier
token's output.

WHY MULTIPLE HEADS?
--------------------
Splitting ``d_model`` into several smaller "heads" lets different
heads specialize in different kinds of relationships, then the heads'
outputs are concatenated back together before the final projection.

SHAPES (matches the notation used throughout docs/ARCHITECTURE.md)
-----------------------------------------------------------------------
    x                  : [B, T, C]             (C = d_model)
    q, k, v (post-proj): [B, T, C]  -> reshaped to [B, H, T, D]
    attention scores    : [B, H, T, T]
    attention output     : [B, H, T, D] -> merged back to [B, T, C]
"""
from __future__ import annotations

import math

import torch
import torch.nn as nn

from models.custom_minilm.config import MiniLLMConfig
from models.custom_minilm.rotary import RotaryEmbedding


class CausalSelfAttention(nn.Module):
    def __init__(self, config: MiniLLMConfig):
        super().__init__()
        self.n_heads = config.n_heads
        self.head_dim = config.head_dim
        self.d_model = config.d_model

        self.q_proj = nn.Linear(config.d_model, config.d_model, bias=config.bias)
        self.k_proj = nn.Linear(config.d_model, config.d_model, bias=config.bias)
        self.v_proj = nn.Linear(config.d_model, config.d_model, bias=config.bias)
        self.o_proj = nn.Linear(config.d_model, config.d_model, bias=config.bias)

        self.rope = RotaryEmbedding(
            head_dim=config.head_dim,
            max_seq_len=config.max_seq_len,
            theta=config.rope_theta,
        )

        self.attn_dropout = nn.Dropout(config.dropout)
        self.resid_dropout = nn.Dropout(config.dropout)

        # True where key position (column) is *after* query position
        # (row) -- those entries get masked out before softmax.
        causal_mask = torch.triu(
            torch.ones(config.max_seq_len, config.max_seq_len, dtype=torch.bool),
            diagonal=1,
        )
        self.register_buffer("causal_mask", causal_mask, persistent=False)

    def _split_heads(
        self, x: torch.Tensor, batch_size: int, seq_len: int
    ) -> torch.Tensor:
        # [B, T, C] -> [B, T, H, D] -> [B, H, T, D]
        return x.view(batch_size, seq_len, self.n_heads, self.head_dim).transpose(
            1, 2
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size, seq_len, _ = x.shape

        q = self._split_heads(self.q_proj(x), batch_size, seq_len)
        k = self._split_heads(self.k_proj(x), batch_size, seq_len)
        v = self._split_heads(self.v_proj(x), batch_size, seq_len)

        q = self.rope(q)
        k = self.rope(k)

        scores = (q @ k.transpose(-2, -1)) / math.sqrt(self.head_dim)  # [B,H,T,T]

        mask = self.causal_mask[:seq_len, :seq_len]
        scores = scores.masked_fill(mask, float("-inf"))

        attn_weights = torch.softmax(scores, dim=-1)
        attn_weights = self.attn_dropout(attn_weights)

        out = attn_weights @ v  # [B, H, T, D]
        out = out.transpose(1, 2).contiguous().view(
            batch_size, seq_len, self.d_model
        )

        out = self.o_proj(out)
        return self.resid_dropout(out)
