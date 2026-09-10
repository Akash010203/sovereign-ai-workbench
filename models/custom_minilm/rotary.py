"""
Rotary Position Embedding (RoPE) -- implemented from scratch.

WHY DOES A TRANSFORMER NEED POSITION INFORMATION AT ALL?
----------------------------------------------------------
Self-attention on its own has no notion of order: it looks at every
token pair the same way regardless of where they sit in the sequence.
Without extra help, "the pump failed after the valve" and "the valve
failed after the pump" would attend identically. RoPE injects
*relative* position information directly into the query/key vectors
before the attention dot product is computed.

HOW ROPE WORKS (intuition)
----------------------------
Each adjacent pair of dimensions in a query/key vector is treated as a
2-D point, and RoPE *rotates* that point by an angle proportional to
the token's position (different pairs rotate at different
frequencies -- early pairs rotate quickly and capture short-range
position differences, later pairs rotate slowly and capture long-range
ones). Two vectors rotated by the *same relative* amount still have
the same dot-product relationship, which is exactly what lets
attention naturally learn "how far apart are these two tokens".

Crucially, rotation never changes a vector's length -- only its
direction -- so RoPE injects position information without distorting
the magnitude of the query/key vectors it is applied to. This is
checked directly in ``tests/test_rotary.py``.

SHAPES
------
    x               : [B, H, T, D]   (B=batch, H=heads, T=seq_len, D=head_dim)
    cos / sin cache : [max_seq_len, D/2]
    output          : [B, H, T, D]   (same shape as input)
"""
from __future__ import annotations

import torch
import torch.nn as nn


class RotaryEmbedding(nn.Module):
    def __init__(self, head_dim: int, max_seq_len: int, theta: float = 10000.0):
        super().__init__()
        if head_dim % 2 != 0:
            raise ValueError(f"head_dim ({head_dim}) must be even for RoPE.")

        # One rotation frequency per pair of dimensions. theta controls
        # how quickly frequencies decay across pairs.
        inv_freq = 1.0 / (
            theta ** (torch.arange(0, head_dim, 2, dtype=torch.float32) / head_dim)
        )
        positions = torch.arange(max_seq_len, dtype=torch.float32)
        freqs = torch.outer(positions, inv_freq)  # [max_seq_len, head_dim/2]

        # Buffers, not parameters: these are computed once from theta
        # and never learned.
        self.register_buffer("cos", freqs.cos(), persistent=False)
        self.register_buffer("sin", freqs.sin(), persistent=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply rotary position encoding: ``[B, H, T, D] -> [B, H, T, D]``."""
        seq_len = x.size(-2)
        cos = self.cos[:seq_len].to(dtype=x.dtype).unsqueeze(0).unsqueeze(0)
        sin = self.sin[:seq_len].to(dtype=x.dtype).unsqueeze(0).unsqueeze(0)

        x_even = x[..., 0::2]
        x_odd = x[..., 1::2]

        rotated_even = x_even * cos - x_odd * sin
        rotated_odd = x_even * sin + x_odd * cos

        # Interleave the rotated even/odd halves back into the
        # original dimension order: [re0, ro0, re1, ro1, ...].
        return torch.stack((rotated_even, rotated_odd), dim=-1).flatten(-2)
