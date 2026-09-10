"""
MiniLLM — the complete, from-scratch decoder-only Transformer for
SovereignAI (SIH 2026, Problem Statement SIH 117).

FULL ARCHITECTURE OVERVIEW
---------------------------
The model is a decoder-only (GPT-style) Transformer.  "Decoder-only"
means there is no cross-attention encoder; the model learns to predict
the next token purely from the preceding tokens it has generated
(autoregressive language modelling).

Token IDs → Embedding → [Block × N] → Final RMSNorm → LM Head → Logits

Step by step:

    1. TOKEN EMBEDDING  [B, T] → [B, T, C]
       Each integer token ID is looked up in a learned weight table of
       shape [vocab_size, d_model].  The result is a continuous vector
       that the model can do arithmetic on.

    2. N TRANSFORMER BLOCKS  [B, T, C] → [B, T, C]  (×n_layers)
       Each block runs:
           • pre-norm  (RMSNorm)
           • causal multi-head self-attention + RoPE + residual
           • pre-norm  (RMSNorm)
           • SwiGLU FFN + residual
       See ``block.py`` for the per-block diagram.

    3. FINAL NORMALIZATION  [B, T, C] → [B, T, C]
       One last RMSNorm applied to the output of the final block
       before the projection to vocabulary logits.  Without this,
       the logit magnitudes can grow unbounded during training.

    4. LANGUAGE MODEL HEAD  [B, T, C] → [B, T, vocab_size]
       A single nn.Linear (no bias) projects d_model → vocab_size.

       WEIGHT TYING: the LM head's weight matrix is the *same tensor*
       as the token embedding matrix (transposed).  This is a standard
       trick that:
           (a) saves ~vocab_size × d_model ≈ 600 × 256 ≈ 153 K
               parameters (significant at ~6 M total),
           (b) encourages token representations to be similar in input
               and output space, which aids training.

    5. LOGITS & LOSS  [B, T, vocab_size]
       At training time we compute cross-entropy loss between the
       logits at every position t and the *next* token at position t+1
       (the targets are the input sequence shifted left by one).

CAUSAL MASKING (WHERE IT ACTUALLY LIVES)
-----------------------------------------
The causal mask lives inside ``CausalSelfAttention.__init__`` and is
applied during every attention forward pass.  The model itself does not
need to pass it in explicitly — every block already enforces it.

SHAPE LEGEND (used throughout this codebase & docs/ARCHITECTURE.md):
    B           = batch size
    T           = sequence length (≤ config.max_seq_len)
    C           = d_model (embedding / hidden dimension)
    H           = n_heads
    D           = head_dim = C // H
    vocab_size  = tokenizer vocabulary size

DEFAULT PARAMETER COUNT (with MiniLLMConfig defaults):
    d_model=256, n_layers=6, n_heads=4, d_ff=683, vocab_size=600

    Component               |  Parameters
    ─────────────────────────────────────────────────────
    Token embedding         |  vocab_size × d_model  = 153 600
    Per block — attention   |  4 × d_model²           = 262 144
    Per block — FFN         |  3 × d_model × d_ff     = 524 544
    Per block — norms (×2)  |  2 × d_model             =    512
    All 6 blocks            |  6 × 787 200            = 4 723 200
    Final norm              |                             256
    LM head (tied, no bias) |  shared with embedding ≈   0
    ─────────────────────────────────────────────────────
    TOTAL (approx)          |  ~4.88 M parameters

This sits well within the 6 GB VRAM budget on an RTX 4050.
(Actual count: run ``model.count_parameters()`` after instantiation.)
"""
from __future__ import annotations

import torch
import torch.nn as nn

from models.custom_minilm.config import MiniLLMConfig
from models.custom_minilm.embeddings import TokenEmbedding
from models.custom_minilm.normalization import RMSNorm
from models.custom_minilm.block import TransformerBlock


class MiniLLM(nn.Module):
    """
    Custom decoder-only Transformer language model.

    This is the entire model assembled from the components built in
    Phase 4 (embeddings, RMSNorm, RoPE, attention) and Phase 5
    (SwiGLU FFN, TransformerBlock).

    Starts from *randomly-initialized* weights.  The weights are
    learned by the training loop in Phase 6 on the project's own
    corpus.

    Explicitly labelled: Custom Educational / Research MiniLLM.
    It is intentionally tiny (~5 M parameters) and will not match
    the quality of frontier LLMs trained on billions of tokens.
    Its purpose is to demonstrate that every mathematical component
    of a Transformer — attention, normalization, positional encoding,
    feed-forward, training loop — was built and trained in-house.

    Args:
        config: ``MiniLLMConfig`` instance carrying all hyperparameters.
    """

    def __init__(self, config: MiniLLMConfig) -> None:
        super().__init__()
        self.config = config

        # ── 1. Token Embedding ─────────────────────────────────────────
        # [B, T] → [B, T, d_model]
        self.token_embedding = TokenEmbedding(config)

        # ── 2. Transformer Blocks ──────────────────────────────────────
        # Each block is identical in *structure* but has its own weights.
        # N = config.n_layers blocks in total.
        self.blocks = nn.ModuleList(
            [TransformerBlock(config) for _ in range(config.n_layers)]
        )

        # ── 3. Final Layer Normalization ───────────────────────────────
        # Applied after the last block, before the LM head.
        self.final_norm = RMSNorm(config.d_model, eps=config.rms_norm_eps)

        # ── 4. Language Model Head ─────────────────────────────────────
        # Projects hidden states → vocabulary logits.
        # bias=False: standard practice for tied-weight LM heads.
        self.lm_head = nn.Linear(config.d_model, config.vocab_size, bias=False)

        # Weight tying: share the embedding matrix with the LM head.
        # After this assignment, updating embedding weights also updates
        # lm_head.weight, and vice versa.  This halves the gradient
        # information paths for the vocabulary matrix, which often
        # improves training stability when vocab_size is small.
        self.lm_head.weight = self.token_embedding.weight

        # ── Weight Initialization ──────────────────────────────────────
        self._init_weights()

    def _init_weights(self) -> None:
        """
        Apply sensible default initializations.

        Follows the GPT-2 paper convention:
          - nn.Linear weights → Normal(0, 0.02)
          - nn.Embedding weights → Normal(0, 0.02)
          - Residual-path projections scaled down by 1/sqrt(n_layers)
            to keep the residual stream variance stable at init.
        """
        std = 0.02
        residual_scale = std / (self.config.n_layers ** 0.5)

        for name, module in self.named_modules():
            if isinstance(module, nn.Linear):
                # Scale down the output projections that feed directly
                # into the residual stream (o_proj and w_down).
                if name.endswith(("o_proj", "w_down")):
                    nn.init.normal_(module.weight, mean=0.0, std=residual_scale)
                else:
                    nn.init.normal_(module.weight, mean=0.0, std=std)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Embedding):
                nn.init.normal_(module.weight, mean=0.0, std=std)

    # ──────────────────────────────────────────────────────────────────
    # Forward pass
    # ──────────────────────────────────────────────────────────────────

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        """
        Run the full model forward pass.

        Args:
            input_ids: ``[B, T]`` integer tensor of token IDs.
                       T must be ≤ config.max_seq_len.

        Returns:
            logits: ``[B, T, vocab_size]`` — raw (un-softmaxed) scores
                    for each vocabulary token at every position.  To
                    get a probability distribution, apply softmax along
                    dim=-1.  For training, pass directly to
                    ``compute_loss()``.
        """
        batch_size, seq_len = input_ids.shape
        if seq_len > self.config.max_seq_len:
            raise ValueError(
                f"Input sequence length {seq_len} exceeds "
                f"config.max_seq_len={self.config.max_seq_len}."
            )

        # [B, T] → [B, T, C]
        x = self.token_embedding(input_ids)

        # [B, T, C] through each of the N Transformer blocks
        for block in self.blocks:
            x = block(x)

        # Final normalization: [B, T, C] → [B, T, C]
        x = self.final_norm(x)

        # Project to vocabulary: [B, T, C] → [B, T, vocab_size]
        logits = self.lm_head(x)

        return logits

    # ──────────────────────────────────────────────────────────────────
    # Loss
    # ──────────────────────────────────────────────────────────────────

    def compute_loss(
        self, input_ids: torch.Tensor, targets: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute the language-modelling cross-entropy loss.

        In autoregressive language modelling, each token must predict
        the *next* token.  Given a sequence [t0, t1, t2, t3]:
            • input  = [t0, t1, t2, t3]   → model produces logits
            • target = [t1, t2, t3, t4]   → what the model should predict

        The targets must therefore be the input shifted one position to
        the left, padding the final position as needed.

        Args:
            input_ids: ``[B, T]`` — the model's input.
            targets:   ``[B, T]`` — the target token at each position.
                       Positions with target == -100 are *ignored*
                       by ``F.cross_entropy`` (standard PyTorch
                       convention for masked positions / padding).

        Returns:
            loss: scalar tensor — the mean cross-entropy loss over all
                  non-masked positions.  Differentiable, ready for
                  ``.backward()``.
        """
        logits = self.forward(input_ids)  # [B, T, vocab_size]

        # Reshape for cross-entropy: [B*T, vocab_size] vs [B*T]
        batch_size, seq_len, vocab_size = logits.shape
        loss = torch.nn.functional.cross_entropy(
            logits.view(batch_size * seq_len, vocab_size),
            targets.view(batch_size * seq_len),
            ignore_index=-100,
        )
        return loss

    # ──────────────────────────────────────────────────────────────────
    # Utilities
    # ──────────────────────────────────────────────────────────────────

    def count_parameters(self) -> int:
        """
        Return the total number of *trainable* parameters.

        For a model with weight-tying, this correctly avoids
        double-counting the shared embedding / LM-head matrix.
        """
        # Use a set of tensor ids to avoid counting tied weights twice.
        seen_ids: set[int] = set()
        total = 0
        for param in self.parameters():
            if id(param) not in seen_ids:
                seen_ids.add(id(param))
                total += param.numel()
        return total

    def parameter_summary(self) -> str:
        """
        Return a human-readable parameter count summary.

        Example output:
            MiniLLM parameter summary
            ─────────────────────────────────────────────
            token_embedding         : 153,600
            blocks.0                : 787,200
            blocks.1                : 787,200
            ...
            final_norm              :     256
            lm_head (tied, skipped) :       0
            ─────────────────────────────────────────────
            TOTAL (unique params)   : 4,876,000
        """
        # Keep CLI output compatible with Windows consoles still configured
        # for cp1252. Decorative box-drawing characters otherwise abort a
        # training run before its first optimizer step.
        lines = ["MiniLLM parameter summary", "-" * 48]
        seen_ids: set[int] = set()
        seen_data: set[int] = set()

        for name, module in self.named_children():
            count = 0
            for param in module.parameters():
                data_ptr = param.data_ptr()
                if data_ptr not in seen_data:
                    seen_data.add(data_ptr)
                    count += param.numel()
            tied_note = ""
            if name == "lm_head":
                tied_note = " (tied to token_embedding)"
                count = 0
            lines.append(f"  {name:<28} : {count:>10,}{tied_note}")

        total = self.count_parameters()
        lines.append("-" * 48)
        lines.append(f"  {'TOTAL (unique parameters)':<28} : {total:>10,}")
        return "\n".join(lines)
