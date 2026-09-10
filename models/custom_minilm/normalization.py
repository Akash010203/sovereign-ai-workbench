"""
RMSNorm -- Root Mean Square Layer Normalization, implemented from
scratch (no ``torch.nn.LayerNorm`` used anywhere here).

WHY NORMALIZE AT ALL?
----------------------
As a Transformer gets deeper, the scale of activations can drift and
make training unstable. Re-scaling activations before each sub-layer
keeps gradients well-behaved and training stable.

WHY RMSNorm INSTEAD OF LayerNorm?
-----------------------------------
Standard LayerNorm normalizes by both mean AND variance (it re-centers
to zero mean, then rescales). RMSNorm skips the re-centering step and
only rescales by the root-mean-square of the activations. In practice
this works about as well for Transformers, is simpler, and is cheaper
to compute -- which is why modern LLMs (including the one this project
is inspired by) favour it over classic LayerNorm.

THE FORMULA
------------
    RMSNorm(x) = x / sqrt(mean(x^2) + eps) * weight

``weight`` is a learned, per-channel scale (initialized to 1s), so the
network can still learn to scale each channel up or down *after*
normalizing -- normalization on its own only fixes the *overall* scale
of each position's vector, not the relative importance of each
channel.

SHAPE
-----
    x  : [..., d_model]  ->  same shape, RMS-normalized per position
"""
from __future__ import annotations

import torch
import torch.nn as nn


class RMSNorm(nn.Module):
    def __init__(self, d_model: int, eps: float = 1e-5):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(d_model))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        mean_square = x.pow(2).mean(dim=-1, keepdim=True)
        normalized = x * torch.rsqrt(mean_square + self.eps)
        return normalized * self.weight
