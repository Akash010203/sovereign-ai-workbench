"""
training/trainer.py — Complete autoregressive training loop for MiniLLM.

WHAT THE TRAINING LOOP DOES (step by step)
--------------------------------------------
1. Load a batch of (input_ids, targets) from the DataLoader.
2. Move tensors to the GPU (if available).
3. Forward pass: model(input_ids) → logits [B, T, vocab_size].
4. Cross-entropy loss: compare logits with targets.
5. Divide loss by gradient_accumulation_steps (for memory management).
6. Backward pass: loss.backward() — computes gradients for all parameters.
7. Accumulate gradients across N micro-steps before updating.
8. Gradient clipping: prevents any single gradient from being huge,
   which would cause a destructive weight update.
9. Optimizer step: AdamW updates all parameters.
10. Scheduler step: adjusts the learning rate.
11. Log the step's loss, LR, and timing.
12. Every eval_every steps: run validation.
13. Every save_every steps: save a checkpoint.

GRADIENT ACCUMULATION
---------------------
The RTX 4050 has 6 GB VRAM.  If a batch of batch_size=8, block_size=128
doesn't fit in memory, we can simulate a larger effective batch by
accumulating gradients over N micro-steps before calling optimizer.step().

Effective batch size = batch_size × gradient_accumulation_steps

MIXED PRECISION
---------------
With ``use_amp=True``, PyTorch's GradScaler and autocast (bfloat16 or
float16) reduce VRAM usage and can speed up GPU matrix multiplications.
float32 is always used for loss computation and optimizer state.

CHECKPOINT FORMAT (saved every save_every steps)
-------------------------------------------------
    {
      "step":            int,
      "model_state":     OrderedDict,
      "optimizer_state": dict,
      "scheduler_state": dict,
      "config":          MiniLLMConfig as dict,
      "tokenizer_path":  str,
      "train_loss":      float,
      "val_loss":        float | None,
      "experiment":      str,
      "timestamp":       ISO-8601 str,
    }
"""
from __future__ import annotations

import datetime
import json
import logging
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Optional

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from models.custom_minilm.config import MiniLLMConfig
from models.custom_minilm.model import MiniLLM
from training.evaluation import evaluate
from training.scheduler import cosine_with_warmup

log = logging.getLogger(__name__)


@dataclass
class TrainingConfig:
    """All hyperparameters for one training run."""

    # Core
    experiment_name: str = "exp1_smoke"
    max_steps: int = 50                     # Experiment 1 default
    batch_size: int = 4
    gradient_accumulation_steps: int = 2    # effective batch = 8

    # Optimizer (AdamW)
    learning_rate: float = 3e-4
    weight_decay: float = 0.1
    beta1: float = 0.9
    beta2: float = 0.95
    grad_clip: float = 1.0                  # gradient clipping norm

    # Schedule
    warmup_steps: int = 10
    min_lr_ratio: float = 0.1

    # Evaluation & checkpointing
    eval_every: int = 25
    save_every: int = 50
    max_val_batches: int = 20

    # Precision
    use_amp: bool = True                    # auto mixed precision

    # Paths
    checkpoint_dir: str = "checkpoints"
    log_dir: str = "logs"

    # Tokenizer reference (stored in checkpoint)
    tokenizer_path: str = "tokenizer/vocab/demo_bpe_vocab.json"


class Trainer:
    """
    Trains a MiniLLM from scratch (or resumes from a checkpoint).

    Usage::

        trainer = Trainer(model_config, training_config, tokenizer)
        trainer.fit(train_loader, val_loader)
    """

    def __init__(
        self,
        model_cfg: MiniLLMConfig,
        train_cfg: TrainingConfig,
        tokenizer,
        resume_from: Optional[str | Path] = None,
        device_override: Optional[torch.device] = None,
    ) -> None:
        self.model_cfg  = model_cfg
        self.train_cfg  = train_cfg
        self.tokenizer  = tokenizer

        # ── Device ────────────────────────────────────────────────────
        if device_override is not None:
            self.device = device_override
        else:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        log.info("Training device: %s", self.device)

        # ── Model ─────────────────────────────────────────────────────
        self.model = MiniLLM(model_cfg).to(self.device)
        log.info("Model parameters (unique): %s",
                 f"{self.model.count_parameters():,}")

        # ── Optimizer ─────────────────────────────────────────────────
        # Separate weight-decay and non-weight-decay parameters.
        # Embedding weights and biases are NOT decayed — standard practice.
        decay_params = [
            p for n, p in self.model.named_parameters()
            if p.requires_grad and p.ndim >= 2
        ]
        nodecay_params = [
            p for n, p in self.model.named_parameters()
            if p.requires_grad and p.ndim < 2
        ]
        self.optimizer = torch.optim.AdamW(
            [
                {"params": decay_params,   "weight_decay": train_cfg.weight_decay},
                {"params": nodecay_params, "weight_decay": 0.0},
            ],
            lr=train_cfg.learning_rate,
            betas=(train_cfg.beta1, train_cfg.beta2),
        )

        # ── Scheduler ─────────────────────────────────────────────────
        self.scheduler = cosine_with_warmup(
            self.optimizer,
            warmup_steps=train_cfg.warmup_steps,
            max_steps=train_cfg.max_steps,
            min_lr_ratio=train_cfg.min_lr_ratio,
        )

        # ── Mixed precision ────────────────────────────────────────────
        amp_enabled = train_cfg.use_amp and self.device.type == "cuda"
        self.scaler = torch.amp.GradScaler(self.device.type, enabled=amp_enabled)
        # bfloat16 is preferred on Ampere+ (RTX 30xx / 40xx); falls back to float16
        self.amp_dtype = (
            torch.bfloat16
            if (amp_enabled and torch.cuda.is_bf16_supported())
            else torch.float16
        )
        log.info("AMP enabled: %s  dtype: %s", amp_enabled,
                 str(self.amp_dtype) if amp_enabled else "float32")

        # ── State ─────────────────────────────────────────────────────
        self.global_step: int = 0
        self.best_val_loss: float = float("inf")
        self.best_checkpoint: Optional[Path] = None

        # ── Directories ───────────────────────────────────────────────
        self.ckpt_dir = Path(train_cfg.checkpoint_dir)
        self.ckpt_dir.mkdir(parents=True, exist_ok=True)
        Path(train_cfg.log_dir).mkdir(parents=True, exist_ok=True)

        # ── Resume ────────────────────────────────────────────────────
        if resume_from is not None:
            self._load_checkpoint(Path(resume_from))

    # ──────────────────────────────────────────────────────────────────
    # Main training loop
    # ──────────────────────────────────────────────────────────────────

    def fit(
        self,
        train_loader: DataLoader,
        val_loader: Optional[DataLoader] = None,
    ) -> dict:
        """
        Run the training loop for ``max_steps`` optimizer steps.

        Returns a summary dict with final train/val loss and metadata.
        """
        cfg = self.train_cfg
        self.model.train()

        run_log: list[dict] = []
        train_loss_accum = 0.0
        micro_step_count = 0
        t_start = time.time()

        log.info(
            "=== Training START  experiment=%s  max_steps=%d  device=%s ===",
            cfg.experiment_name, cfg.max_steps, self.device,
        )

        # Infinite cycling iterator over the train loader
        def cycle(loader):
            while True:
                for batch in loader:
                    yield batch

        loader_iter = cycle(train_loader)
        last_val_loss: Optional[float] = None

        while self.global_step < cfg.max_steps:
            step_t0 = time.time()
            self.optimizer.zero_grad(set_to_none=True)

            # ── Gradient accumulation micro-steps ─────────────────────
            for micro_step in range(cfg.gradient_accumulation_steps):
                input_ids, targets = next(loader_iter)
                input_ids = input_ids.to(self.device)
                targets   = targets.to(self.device)

                with torch.autocast(
                    device_type=self.device.type,
                    dtype=self.amp_dtype,
                    enabled=self.scaler.is_enabled(),
                ):
                    loss = self.model.compute_loss(input_ids, targets)
                    loss = loss / cfg.gradient_accumulation_steps

                self.scaler.scale(loss).backward()
                train_loss_accum += loss.item()
                micro_step_count += 1

            # ── Gradient clipping ─────────────────────────────────────
            self.scaler.unscale_(self.optimizer)
            grad_norm = nn.utils.clip_grad_norm_(
                self.model.parameters(), cfg.grad_clip
            )

            # ── Optimizer + scheduler step ────────────────────────────
            self.scaler.step(self.optimizer)
            self.scaler.update()
            self.scheduler.step()

            self.global_step += 1
            step_elapsed = time.time() - step_t0

            # Average loss for this optimizer step
            step_loss = train_loss_accum / micro_step_count
            train_loss_accum = 0.0
            micro_step_count = 0

            current_lr = self.scheduler.get_last_lr()[0]

            log.info(
                "step %4d/%d | loss: %.4f | lr: %.2e | grad_norm: %.3f | %.2fs",
                self.global_step, cfg.max_steps,
                step_loss, current_lr, grad_norm, step_elapsed,
            )

            step_record = {
                "step": self.global_step,
                "train_loss": round(step_loss, 6),
                "lr": round(current_lr, 8),
                "grad_norm": round(float(grad_norm), 4),
                "elapsed_s": round(step_elapsed, 3),
            }

            # ── Validation ────────────────────────────────────────────
            if (
                val_loader is not None
                and self.global_step % cfg.eval_every == 0
            ):
                val_result = evaluate(
                    self.model, val_loader, self.device, cfg.max_val_batches
                )
                last_val_loss = val_result["val_loss"]
                step_record["val_loss"]       = round(last_val_loss, 6)
                step_record["val_perplexity"] = round(val_result["val_perplexity"], 3)
                if last_val_loss < self.best_val_loss:
                    self.best_val_loss = last_val_loss
                    self.best_checkpoint = self._save_checkpoint(
                        step_loss, last_val_loss, tag="best"
                    )

            run_log.append(step_record)

            # ── Checkpoint ────────────────────────────────────────────
            if self.global_step % cfg.save_every == 0:
                self._save_checkpoint(step_loss, last_val_loss)

        # ── Final checkpoint ──────────────────────────────────────────
        final_checkpoint = self._save_checkpoint(step_loss, last_val_loss, tag="final")

        total_elapsed = time.time() - t_start
        log.info(
            "=== Training DONE  steps=%d  total_time=%.1fs ===",
            self.global_step, total_elapsed,
        )

        # Save the JSON training log
        log_path = Path(cfg.log_dir) / f"{cfg.experiment_name}_log.json"
        log_path.write_text(json.dumps(run_log, indent=2), encoding="utf-8")

        summary = {
            "experiment": cfg.experiment_name,
            "model_params": self.model.count_parameters(),
            "total_steps":  self.global_step,
            "final_train_loss": run_log[-1]["train_loss"],
            "best_val_loss":    round(self.best_val_loss, 6),
            "device":           str(self.device),
            "total_time_s":     round(total_elapsed, 2),
            "checkpoint_dir":   str(self.ckpt_dir),
            "final_checkpoint": str(final_checkpoint),
            "best_checkpoint":  str(self.best_checkpoint or final_checkpoint),
        }
        return summary

    # ──────────────────────────────────────────────────────────────────
    # Checkpoint helpers
    # ──────────────────────────────────────────────────────────────────

    def _save_checkpoint(
        self,
        train_loss: float,
        val_loss: Optional[float],
        tag: str = "",
    ) -> Path:
        """Save a full checkpoint containing everything needed to resume."""
        ts = datetime.datetime.now().strftime("%Y%m%dT%H%M%S")
        suffix = f"_{tag}" if tag else ""
        fname  = f"{self.train_cfg.experiment_name}_step{self.global_step:05d}{suffix}.pt"
        path   = self.ckpt_dir / fname

        ckpt = {
            "step":            self.global_step,
            "model_state":     self.model.state_dict(),
            "optimizer_state": self.optimizer.state_dict(),
            "scheduler_state": self.scheduler.state_dict(),
            "scaler_state":    self.scaler.state_dict(),
            "model_config":    asdict(self.model_cfg),
            "train_config":    asdict(self.train_cfg),
            "tokenizer_path":  self.train_cfg.tokenizer_path,
            "train_loss":      train_loss,
            "val_loss":        val_loss,
            "experiment":      self.train_cfg.experiment_name,
            "timestamp":       ts,
        }
        torch.save(ckpt, path)
        log.info("Checkpoint saved → %s", path)
        return path

    def _load_checkpoint(self, path: Path) -> None:
        """Load a checkpoint and restore all training state."""
        log.info("Resuming from checkpoint: %s", path)
        ckpt = torch.load(path, map_location=self.device, weights_only=False)

        self.model.load_state_dict(ckpt["model_state"])
        self.optimizer.load_state_dict(ckpt["optimizer_state"])
        self.scheduler.load_state_dict(ckpt["scheduler_state"])
        if "scaler_state" in ckpt:
            self.scaler.load_state_dict(ckpt["scaler_state"])
        self.global_step = ckpt["step"]
        log.info("Resumed from step %d", self.global_step)
