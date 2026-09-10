"""
training/evaluation.py — Validation loss loop for MiniLLM.

PURPOSE
-------
After every N training steps we want a snapshot of how the model
performs on text it has NOT been trained on (the validation split).
This tells us:
  - Is the model actually learning, or just memorizing the training set?
  - Has it started to overfit? (val loss rises while train loss falls)
  - Is the training loop numerically stable?

HOW IT WORKS
------------
The validation loop runs exactly like the training forward pass but:
  - ``torch.no_grad()`` — no gradient computation (saves memory)
  - the optimizer does NOT step
  - the model is put into eval() mode (disables dropout, if any)
  - we average the cross-entropy loss over all validation batches

PERPLEXITY
----------
Perplexity is ``exp(cross_entropy_loss)``  and is a common metric for
language models.  A model guessing uniformly over 600 tokens has
perplexity ≈ 600.  A perfectly-trained model has perplexity 1.
Lower is better.
"""
from __future__ import annotations

import logging
import math

import torch
from torch.utils.data import DataLoader

from models.custom_minilm.model import MiniLLM

log = logging.getLogger(__name__)


@torch.no_grad()
def evaluate(
    model: MiniLLM,
    val_loader: DataLoader,
    device: torch.device,
    max_val_batches: int = 50,
) -> dict:
    """
    Compute validation cross-entropy loss and perplexity.

    Args:
        model:            The MiniLLM (any eval-mode-able nn.Module works).
        val_loader:       DataLoader over the validation TokenizedTextDataset.
        device:           torch.device to run inference on.
        max_val_batches:  Cap to keep validation fast (useful when the
                          val set is large relative to the training budget).

    Returns:
        dict with keys:
            ``val_loss``        (float) — mean cross-entropy
            ``val_perplexity``  (float) — exp(val_loss)
            ``n_batches``       (int)   — how many batches were evaluated
    """
    model.eval()
    total_loss = 0.0
    n_batches = 0

    for batch_idx, (input_ids, targets) in enumerate(val_loader):
        if batch_idx >= max_val_batches:
            break

        input_ids = input_ids.to(device)
        targets   = targets.to(device)

        loss = model.compute_loss(input_ids, targets)
        total_loss += loss.item()
        n_batches += 1

    if n_batches == 0:
        log.warning("Validation loader is empty — returning loss=inf.")
        return {"val_loss": float("inf"), "val_perplexity": float("inf"), "n_batches": 0}

    val_loss = total_loss / n_batches
    val_ppl  = math.exp(min(val_loss, 20.0))  # clamp to avoid overflow

    log.info("Validation — loss: %.4f | perplexity: %.2f | batches: %d",
             val_loss, val_ppl, n_batches)

    model.train()   # restore training mode
    return {"val_loss": val_loss, "val_perplexity": val_ppl, "n_batches": n_batches}
