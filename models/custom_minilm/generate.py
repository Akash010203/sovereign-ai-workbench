"""
models/custom_minilm/generate.py — Autoregressive text generation.

HOW AUTOREGRESSIVE GENERATION WORKS
--------------------------------------
At each step, the model sees all tokens generated so far and produces a
probability distribution over the vocabulary.  We pick the next token
from that distribution and append it.  Repeat until:
  - An <eos> token is produced, OR
  - We've generated ``max_new_tokens`` tokens

SAMPLING STRATEGIES
--------------------
1. GREEDY  — always pick the single highest-probability token.
   Fast and deterministic, but often repetitive.

2. TEMPERATURE  — divide all logits by T before softmax:
   - T < 1 → sharper distribution (more "confident", less creative)
   - T > 1 → flatter distribution (more random / creative)
   - T = 1 → unmodified model distribution

3. TOP-K  — zero out all tokens except the k highest-probability ones,
   then sample from the remaining k.  Prevents the model from sampling
   from the long tail of very unlikely tokens.

4. TOP-P (nucleus sampling) — keep the smallest set of tokens whose
   cumulative probability ≥ p, then sample from that set.  The set
   size adapts: on high-confidence steps only a few tokens make the
   cut; on uncertain steps many do.

These strategies can be combined: temperature → top-k → top-p → sample.
"""
from __future__ import annotations

import torch
import torch.nn.functional as F

from models.custom_minilm.model import MiniLLM


@torch.no_grad()
def generate(
    model: MiniLLM,
    prompt_ids: list[int],
    max_new_tokens: int = 64,
    temperature: float = 1.0,
    top_k: int = 0,
    top_p: float = 1.0,
    eos_id: int | None = None,
    seed: int | None = None,
    device: str | torch.device = "cpu",
) -> list[int]:
    """
    Generate tokens autoregressively from a prompt.

    Args:
        model:          A trained MiniLLM in eval mode.
        prompt_ids:     List of integer token IDs to condition on.
        max_new_tokens: Maximum number of NEW tokens to generate.
        temperature:    Softmax temperature (1.0 = unchanged).
        top_k:          If > 0, only sample from top-k tokens.
        top_p:          If < 1.0, apply nucleus (top-p) sampling.
        eos_id:         Token ID that stops generation early.
        seed:           If provided, makes generation deterministic.
        device:         torch.device (string or object).

    Returns:
        List of generated token IDs (does NOT include the prompt).
    """
    if seed is not None:
        torch.manual_seed(seed)

    device = torch.device(device) if isinstance(device, str) else device
    model = model.to(device).eval()

    max_ctx = model.config.max_seq_len

    # Start with the prompt as the context
    ctx = list(prompt_ids)
    generated: list[int] = []

    for _ in range(max_new_tokens):
        # Truncate context to max_seq_len if needed (sliding window)
        window = ctx[-max_ctx:]
        input_ids = torch.tensor([window], dtype=torch.long, device=device)

        # Forward pass → logits for the LAST position only
        logits = model(input_ids)          # [1, T, vocab_size]
        next_logits = logits[0, -1, :]     # [vocab_size]

        # ── Temperature ───────────────────────────────────────────
        if temperature != 1.0 and temperature > 0.0:
            next_logits = next_logits / temperature

        # ── Top-k ─────────────────────────────────────────────────
        if top_k > 0:
            top_k_clamped = min(top_k, next_logits.size(-1))
            top_values, _ = torch.topk(next_logits, top_k_clamped)
            cutoff = top_values[-1]
            next_logits = next_logits.masked_fill(next_logits < cutoff, float("-inf"))

        # ── Top-p (nucleus) ───────────────────────────────────────
        if top_p < 1.0:
            probs = F.softmax(next_logits, dim=-1)
            sorted_probs, sorted_indices = torch.sort(probs, descending=True)
            cumulative = torch.cumsum(sorted_probs, dim=-1)
            # Remove tokens once cumulative prob exceeds top_p
            sorted_indices_to_remove = cumulative - sorted_probs > top_p
            sorted_probs[sorted_indices_to_remove] = 0.0
            # Scatter back to original order
            probs = torch.zeros_like(probs).scatter_(0, sorted_indices, sorted_probs)
            next_token_id = torch.multinomial(probs, num_samples=1).item()
        elif temperature == 0.0:
            # Greedy
            next_token_id = int(torch.argmax(next_logits).item())
        else:
            probs = F.softmax(next_logits, dim=-1)
            next_token_id = int(torch.multinomial(probs, num_samples=1).item())

        generated.append(next_token_id)
        ctx.append(next_token_id)

        if eos_id is not None and next_token_id == eos_id:
            break

    return generated
