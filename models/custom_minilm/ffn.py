"""
SwiGLU Feed-Forward Network — implemented from scratch for the
SovereignAI custom MiniLLM.

WHAT IS A FEED-FORWARD NETWORK (FFN) IN A TRANSFORMER?
-------------------------------------------------------
After the attention sub-layer has mixed information *across* positions
(letting every token look at every earlier token), the FFN processes
each position *independently*: the same learned weights are applied to
each token's vector one at a time, with no cross-position communication.

This sounds redundant after attention, but the FFN is actually where
most of the model's capacity lives.  The rule of thumb is that the
hidden dimension d_ff is roughly 4× d_model (classic Transformer) or
2.67× d_model (SwiGLU, see below), making the FFN the majority of
parameters per block.

Think of attention as "routing information between positions" and the
FFN as "processing / transforming that combined information per token".

WHY SwiGLU INSTEAD OF ReLU / GELU?
-----------------------------------
The original 2017 Transformer used ReLU: FFN(x) = max(0, W1·x + b)·W2.
Modern LLMs (LLaMA, PaLM, Mistral) use SwiGLU instead, because
empirically it trains faster to the same loss.

SwiGLU is a *gated* activation:

    FFN_SwiGLU(x) = (SiLU(Gate(x))  ⊙  Up(x))  ·  Down

where:
    Gate(x) = W_gate · x      [B, T, d_ff]
    Up(x)   = W_up   · x      [B, T, d_ff]
    SiLU(z) = z · sigmoid(z)  (the Swish-1 function)
    ⊙       = element-wise multiplication (the "gating")
    Down    = W_down × [B, T, d_ff] → [B, T, d_model]

The gate learns *which neurons to suppress*; the up-projection learns
*what information to amplify*. Multiplying them together lets the
network express richer non-linearities than a simple ReLU.

WHY d_ff = 683 (≈ 2.67 × 256)?
---------------------------------
The SwiGLU variant uses TWO d_ff-sized projections (gate + up) plus
one down projection, instead of the classic one W1 + one W2.  To keep
the total FLOPs (and parameter count) roughly the same as a classic
4× hidden-size FFN, the d_ff must be scaled by 2/3:

    classic  : d_ff = 4 × 256 = 1024   → 1024 + 1024 matrices
    SwiGLU   : d_ff = (4 × 2/3) × 256 ≈ 683 → 683 + 683 + 683 matrices

683 × 3 = 2049 ≈ 2 × 1024 = 2048 (within 1 neuron of equivalent FLOPs).

The SwiGLU formula is from Noam Shazeer (2020):
    https://arxiv.org/abs/2002.05202

SHAPES
------
    x      : [B, T, d_model]    (input)
    gate   : [B, T, d_ff]
    up     : [B, T, d_ff]
    hidden : [B, T, d_ff]       (after gating)
    output : [B, T, d_model]    (same shape as input)
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from models.custom_minilm.config import MiniLLMConfig


class SwiGLUFFN(nn.Module):
    """
    SwiGLU feed-forward block.

    Three weight matrices (no bias by default, matching config.bias):
        w_gate : d_model → d_ff
        w_up   : d_model → d_ff
        w_down : d_ff   → d_model

    The activation function is SiLU (≡ Swish-1), implemented via
    ``torch.nn.functional.silu``.  The gating ⊙ is a plain
    element-wise multiply — no extra library required.
    """

    def __init__(self, config: MiniLLMConfig) -> None:
        super().__init__()
        self.w_gate = nn.Linear(config.d_model, config.d_ff, bias=config.bias)
        self.w_up   = nn.Linear(config.d_model, config.d_ff, bias=config.bias)
        self.w_down = nn.Linear(config.d_ff,   config.d_model, bias=config.bias)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: ``[B, T, d_model]``

        Returns:
            ``[B, T, d_model]``  — same shape, each token processed
            independently through the gated non-linearity.
        """
        # Gate branch: SiLU activates the "gate" (what to keep)
        gate = F.silu(self.w_gate(x))   # [B, T, d_ff]

        # Up branch: linear projection of the same input
        up = self.w_up(x)               # [B, T, d_ff]

        # Gating: element-wise product — neurons the gate suppresses
        # (≈ 0) get zeroed out; neurons it amplifies (> 0) pass through
        hidden = gate * up              # [B, T, d_ff]
        hidden = self.dropout(hidden)

        # Project back to d_model
        return self.w_down(hidden)      # [B, T, d_model]
