"""
training/scheduler.py — Learning rate scheduler for MiniLLM training.

WHY DO WE NEED A LEARNING RATE SCHEDULE?
-----------------------------------------
Training a neural network with a fixed learning rate is rarely optimal:
- Too high at the start → chaotic early updates, poor convergence
- Too high at the end   → the optimizer overshoots the minimum and bounces

A good schedule:
  1. Warms up from near-zero to the peak LR over the first few steps
     (lets the optimizer calibrate before taking large steps)
  2. Decays gradually after warmup so the model fine-tunes into a minimum

This project uses a standard cosine-with-warmup schedule, the same
shape used by most modern LLM training runs.

THE FORMULA
-----------
Step 0..warmup_steps:
    lr = peak_lr * (step / warmup_steps)          # linear warmup

Step warmup_steps..max_steps:
    progress = (step - warmup_steps) / (max_steps - warmup_steps)
    cosine   = 0.5 * (1 + cos(π × progress))     # 1 → 0
    lr = min_lr + (peak_lr - min_lr) * cosine    # peak_lr → min_lr

This is implemented as a PyTorch LambdaLR scheduler so it integrates
cleanly with the optimizer and checkpoint/resume machinery.
"""
from __future__ import annotations

import math

from torch.optim import Optimizer
from torch.optim.lr_scheduler import LambdaLR


def cosine_with_warmup(
    optimizer: Optimizer,
    warmup_steps: int,
    max_steps: int,
    min_lr_ratio: float = 0.1,
) -> LambdaLR:
    """
    Build a cosine-decay-with-linear-warmup LambdaLR scheduler.

    Args:
        optimizer:     The AdamW optimizer to attach the schedule to.
        warmup_steps:  Number of steps over which LR rises from 0 to peak.
        max_steps:     Total training steps (warm-up + decay period).
        min_lr_ratio:  The floor of the LR as a fraction of peak_lr.
                       Default 0.1 means LR decays to 10% of its peak.

    Returns:
        A PyTorch LambdaLR scheduler.  Call ``scheduler.step()`` once
        per optimizer step.
    """
    if max_steps <= 0:
        raise ValueError(f"max_steps must be > 0, got {max_steps}.")
    if warmup_steps < 0:
        raise ValueError(f"warmup_steps must be >= 0, got {warmup_steps}.")
    warmup_steps = min(warmup_steps, max_steps)

    def lr_lambda(current_step: int) -> float:
        # ── Linear warmup ────────────────────────────────────────────
        if current_step < warmup_steps:
            return float(current_step + 1) / float(max(warmup_steps, 1))

        # ── Cosine decay ─────────────────────────────────────────────
        decay_steps = max_steps - warmup_steps
        step_into_decay = current_step - warmup_steps
        if decay_steps == 0:
            return 1.0
        progress = float(step_into_decay) / float(decay_steps)
        cosine_factor = 0.5 * (1.0 + math.cos(math.pi * progress))
        return min_lr_ratio + (1.0 - min_lr_ratio) * cosine_factor

    return LambdaLR(optimizer, lr_lambda)
