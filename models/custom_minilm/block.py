"""
Transformer block — the single, repeating unit of the MiniLLM.

HOW A TRANSFORMER BLOCK IS ORGANIZED
--------------------------------------
A classic 2017 Transformer used *post-norm* residual connections:

    output = LayerNorm(x + Sublayer(x))

Modern LLMs (LLaMA, Mistral, this project) use *pre-norm* instead:

    output = x + Sublayer(LayerNorm(x))

Pre-norm is easier to train at the start: the sub-layers receive
a normalized signal even before any weights have been properly
calibrated.  The residual bypass (+x) keeps gradients from
vanishing as depth increases.

A SINGLE BLOCK does exactly two things in sequence:

    1. Self-attention sub-layer  (mixes information across positions)
    2. Feed-forward sub-layer    (processes each position independently)

Each sub-layer is wrapped in:
    a.  a pre-norm  (RMSNorm applied to the *input* before the sub-layer)
    b.  a residual  (the sub-layer's output is *added back* to the input)

Mermaid diagram of one block:

    x  ──┬── RMSNorm ──> CausalSelfAttention ──> + ──> y
         │                                        ↑
         └────────────────────────────────────────┘ (residual)

    y  ──┬── RMSNorm ──> SwiGLUFFN ─────────────> + ──> z
         │                                        ↑
         └────────────────────────────────────────┘ (residual)

SHAPES
------
    x, y, z : [B, T, d_model]   — all the same shape in and out
"""
from __future__ import annotations

import torch
import torch.nn as nn

from models.custom_minilm.config import MiniLLMConfig
from models.custom_minilm.normalization import RMSNorm
from models.custom_minilm.attention import CausalSelfAttention
from models.custom_minilm.ffn import SwiGLUFFN


class TransformerBlock(nn.Module):
    """
    One pre-norm Transformer block:
        pre-norm → attention → residual → pre-norm → FFN → residual.

    Args:
        config: shared ``MiniLLMConfig`` instance — all sizing comes
                from here; no ad-hoc magic numbers inside this class.
    """

    def __init__(self, config: MiniLLMConfig) -> None:
        super().__init__()

        # Pre-normalization layers — one before attention, one before FFN.
        # Both use RMSNorm, our custom from-scratch implementation.
        self.norm1 = RMSNorm(config.d_model, eps=config.rms_norm_eps)
        self.norm2 = RMSNorm(config.d_model, eps=config.rms_norm_eps)

        # Sub-layers — both are our own from-scratch implementations.
        self.attn = CausalSelfAttention(config)
        self.ffn  = SwiGLUFFN(config)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: ``[B, T, d_model]``

        Returns:
            ``[B, T, d_model]``  — same shape, richer content.
        """
        # 1. Attention sub-layer: pre-norm + residual
        #    x is first normalized, then attended, then added back.
        x = x + self.attn(self.norm1(x))   # [B, T, d_model]

        # 2. FFN sub-layer: pre-norm + residual
        x = x + self.ffn(self.norm2(x))    # [B, T, d_model]

        return x
