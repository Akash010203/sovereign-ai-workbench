"""
Configuration for SovereignAI's custom "MiniLLM" -- a small,
from-scratch, decoder-only Transformer.

Every other file under ``models/custom_minilm/`` takes one of these
config objects and never re-invents its own copy of these numbers.
Keeping them in one place makes the model's size an explicit, single
decision -- easy to shrink further or grow later (see
``docs/ARCHITECTURE.md`` Section 9 for why this project starts small).

DEFAULT SIZING
--------------
The defaults below land at roughly 6-7M parameters once the full model
is assembled in Phase 5 -- chosen to comfortably fit an RTX 4050
laptop GPU with 6 GB of VRAM for the staged experiments described in
the project's build contract (10-50 steps, then 100-500 steps, then a
longer demonstration run).
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass
class MiniLLMConfig:
    vocab_size: int  # must match the trained tokenizer's vocab_size (Phase 3)

    max_seq_len: int = 128

    d_model: int = 256
    n_layers: int = 6
    n_heads: int = 4
    d_ff: int = 683  # SwiGLU hidden size (Phase 5); roughly 2.67 x d_model

    dropout: float = 0.0

    rms_norm_eps: float = 1e-5
    rope_theta: float = 10000.0

    bias: bool = False

    def __post_init__(self) -> None:
        if self.d_model % self.n_heads != 0:
            raise ValueError(
                f"d_model ({self.d_model}) must be divisible by "
                f"n_heads ({self.n_heads})."
            )

    @property
    def head_dim(self) -> int:
        return self.d_model // self.n_heads
