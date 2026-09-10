"""
scripts/finetune_instruct.py — Instruction fine-tuning of the custom MiniLLM.

WHAT IS INSTRUCTION FINE-TUNING?
---------------------------------
After pretraining (learning language from raw text), the model has learned
word patterns but doesn't know HOW to answer questions helpfully.

Instruction fine-tuning teaches it to follow the prompt format:
    <|system|> ... <|user|> ... <|assistant|> ...

The model is shown (question, answer) pairs and trained to predict the
ANSWER given the QUESTION.  This is called "supervised fine-tuning" (SFT).

REQUIREMENT: Run AFTER pretraining.
    1. python scripts\\train_model.py             ← pretrain first
    2. python scripts\\prepare_openorca.py        ← get instruction data
    3. python scripts\\finetune_instruct.py       ← fine-tune (this script)

HONESTY NOTE
------------
This is standard SFT as described in InstructGPT, Alpaca, and Orca papers.
The custom model architecture and tokenizer are ours.
The training procedure is our implementation.
The data (OpenOrca) is from Microsoft Research (MIT licensed).
"""
from __future__ import annotations

import argparse
import json
import logging
import math
import sys
import time
from dataclasses import asdict
from pathlib import Path

import torch
from torch.utils.data import Dataset, DataLoader

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from core.config import get_settings, ensure_directories
from core.logging_setup import setup_logging
from models.custom_minilm.checkpoint import load_model_from_checkpoint
from models.custom_minilm.config import MiniLLMConfig
from models.custom_minilm.model import MiniLLM
from tokenizer.tokenizer import ByteLevelBPETokenizer
from training.scheduler import cosine_with_warmup


def _save_checkpoint(
    model,
    optimizer,
    step: int,
    loss: float,
    path: str,
    *,
    tokenizer_path: str,
    **meta,
) -> None:
    """Save a fine-tuning checkpoint in the same format as the Trainer."""
    torch.save({
        "model_state_dict":     model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "step":                 step,
        "loss":                 loss,
        # ``MiniLLM`` stores its configuration as ``model.config``.  Keeping
        # it here is essential: without it, a fine-tuned medium checkpoint
        # cannot be reconstructed for inference.
        "model_config":         asdict(model.config),
        "tokenizer_path":       tokenizer_path,
        **meta,
    }, path)

log = logging.getLogger(__name__)

# ─────────────────────────────────────────────────────────────────────────────
# Instruction dataset — lazy-loading for memory efficiency at 90K+ scale
# ─────────────────────────────────────────────────────────────────────────────

class InstructDataset(Dataset):
    """
    Dataset for instruction fine-tuning.

    Each example is a full prompt:
        <|system|>\\n{sys}\\n<|user|>\\n{q}\\n<|assistant|>\\n{r}

    The loss is computed ONLY on the response tokens (after <|assistant|>).
    This is standard SFT practice — we don't penalise the model for
    "predicting" the question it already received.

    MEMORY-EFFICIENT DESIGN (for 90K+ examples):
    Instead of loading all lines into RAM at init time, we read line offsets
    and load individual examples on demand in __getitem__.
    """

    # Special token strings (must be in the tokenizer vocab or will be split)
    SYS_TOKEN   = "<|system|>"
    USER_TOKEN  = "<|user|>"
    ASST_TOKEN  = "<|assistant|>"

    def __init__(
        self,
        jsonl_path: Path,
        tokenizer: ByteLevelBPETokenizer,
        max_seq_len: int = 256,
    ) -> None:
        self.tokenizer   = tokenizer
        self.max_seq_len = max_seq_len
        self.jsonl_path  = Path(jsonl_path)

        # Build an index of byte offsets for each line — O(n) but uses
        # only ~720KB for 90K lines (8 bytes per offset) vs ~500MB for
        # loading all examples as Python dicts.
        self._offsets: list[int] = []
        with self.jsonl_path.open("rb") as f:
            while True:
                offset = f.tell()
                line = f.readline()
                if not line:
                    break
                line_stripped = line.strip()
                if line_stripped:
                    self._offsets.append(offset)

        log.info("InstructDataset: %d examples indexed from %s (lazy-loading)",
                 len(self._offsets), jsonl_path)

    def __len__(self) -> int:
        return len(self._offsets)

    def _read_example(self, idx: int) -> dict:
        """Read a single example from disk by seeking to its offset."""
        with self.jsonl_path.open("r", encoding="utf-8") as f:
            f.seek(self._offsets[idx])
            line = f.readline()
            return json.loads(line)

    def _encode(self, text: str) -> list[int]:
        return self.tokenizer.encode(text)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        ex = self._read_example(idx)

        sys_text  = ex.get("system",   "You are a helpful assistant.")
        user_text = ex.get("question", "")
        asst_text = ex.get("response", "")

        # Build prefix (system + user) — no loss here
        prefix_text = (
            f"{self.SYS_TOKEN}\n{sys_text}\n"
            f"{self.USER_TOKEN}\n{user_text}\n"
            f"{self.ASST_TOKEN}\n"
        )
        prefix_ids = self._encode(prefix_text)
        asst_ids   = self._encode(asst_text)
        eos_id     = self.tokenizer.vocab.special_to_id.get("<eos>", 0)
        full_ids   = prefix_ids + asst_ids + [eos_id]

        # Keep both prompt and answer when an example is longer than the
        # context window.  Naively taking ``full_ids[:max_seq_len + 1]`` can
        # discard the entire answer for long OpenOrca prompts, leaving a batch
        # with every target masked and producing a NaN cross-entropy loss.
        max_total = self.max_seq_len + 1
        if len(full_ids) > max_total:
            prefix_budget = min(len(prefix_ids), max_total // 2)
            response_budget = max_total - prefix_budget
            prefix_ids = prefix_ids[-prefix_budget:]
            asst_ids = asst_ids[: max(response_budget - 1, 1)]
            full_ids = prefix_ids + asst_ids + [eos_id]

        input_ids = torch.tensor(full_ids[:-1], dtype=torch.long)
        targets   = torch.tensor(full_ids[1:],  dtype=torch.long)

        # Mask prefix tokens from loss: set to -100 so cross_entropy ignores them
        prefix_len = min(len(prefix_ids), len(input_ids))
        targets[:prefix_len - 1] = -100    # mask all except the first assistant token

        # Pad to max_seq_len
        seq_len = input_ids.size(0)
        if seq_len < self.max_seq_len:
            pad = self.max_seq_len - seq_len
            input_ids = torch.cat([input_ids, torch.zeros(pad, dtype=torch.long)])
            targets   = torch.cat([targets,   torch.full((pad,), -100, dtype=torch.long)])

        return input_ids, targets


# ─────────────────────────────────────────────────────────────────────────────
# Fine-tuning loop
# ─────────────────────────────────────────────────────────────────────────────

def finetune(
    checkpoint_path: str,
    instruct_jsonl:  str,
    output_dir:      str,
    max_steps:       int  = 300,
    batch_size:      int  = 8,
    lr:              float = 5e-5,
    warmup_steps:    int  = 30,
    max_seq_len:     int | None = None,
    device:          str  = "cpu",
    gradient_accum:  int  = 4,
    log_every:       int  = 20,
    save_every:      int  = 100,
    eval_split:      float = 0.05,
) -> None:
    """Run supervised fine-tuning on instruction data."""

    ensure_directories()

    # ── Load checkpoint ───────────────────────────────────────────────
    ckpt_path = Path(checkpoint_path)
    if not ckpt_path.exists():
        log.warning("No checkpoint found at %s — initialising fresh model.", ckpt_path)
        # Fresh init (for testing without a pretrained model)
        tok_path  = ROOT / "tokenizer" / "vocab" / "demo_bpe_vocab.json"
        tokenizer = ByteLevelBPETokenizer.load(tok_path)
        cfg       = MiniLLMConfig(vocab_size=tokenizer.vocab_size)
        model     = MiniLLM(cfg)
    else:
        model, meta = load_model_from_checkpoint(str(ckpt_path), device=device)
        tok_path    = meta.get("tokenizer_path") or "tokenizer/vocab/demo_bpe_vocab.json"
        tokenizer   = ByteLevelBPETokenizer.load(ROOT / tok_path)
        log.info("Loaded checkpoint: step=%d", meta["step"])

    model = model.to(device)
    model.train()

    # A checkpoint's RoPE cache and causal mask are sized at model creation.
    # Fine-tuning at a longer length would otherwise fail only on the first
    # batch after a lengthy setup.
    max_seq_len = max_seq_len or model.config.max_seq_len
    if max_seq_len > model.config.max_seq_len:
        raise ValueError(
            f"--max-seq-len={max_seq_len} exceeds the checkpoint context "
            f"length ({model.config.max_seq_len}). Re-run pretraining with "
            "--model-size medium, or set --max-seq-len to the checkpoint limit."
        )

    # ── Dataset & loader ──────────────────────────────────────────────
    full_ds = InstructDataset(
        Path(instruct_jsonl), tokenizer, max_seq_len=max_seq_len
    )

    # Split into train / val
    total = len(full_ds)
    if total < 2:
        raise ValueError("Instruction dataset needs at least two valid examples.")
    if gradient_accum < 1:
        raise ValueError("gradient_accum must be at least 1.")
    if batch_size < 1:
        raise ValueError("batch_size must be at least 1.")
    val_size = max(int(total * eval_split), 1)
    val_size = min(val_size, total - 1)
    train_size = total - val_size
    train_ds, val_ds = torch.utils.data.random_split(full_ds, [train_size, val_size])

    log.info("Dataset split: %d train, %d val (%.0f%% held out)",
             train_size, val_size, eval_split * 100)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, drop_last=True,
                              num_workers=0, pin_memory=(device == "cuda"))
    val_loader   = DataLoader(val_ds,   batch_size=batch_size, shuffle=False, drop_last=False,
                              num_workers=0, pin_memory=(device == "cuda"))

    # Auto-calculate max_steps if not explicitly set via CLI
    if max_steps <= 0:
        max_steps = max(1, (train_size // batch_size) * 3)  # ~3 epochs
        log.info("Auto max_steps: %d (3 epochs × %d batches)", max_steps, train_size // batch_size)

    # ── Optimizer & scheduler ─────────────────────────────────────────
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    # ``max_steps`` counts dataloader micro-batches for backward
    # compatibility. The scheduler advances only when the optimizer advances.
    optimizer_steps = math.ceil(max_steps / gradient_accum)
    warmup_steps = min(warmup_steps, optimizer_steps)
    scheduler = cosine_with_warmup(optimizer, warmup_steps, optimizer_steps)
    scaler    = torch.amp.GradScaler(device, enabled=(device == "cuda"))

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    log.info(
        "Fine-tuning: max_steps=%d  batch=%d  lr=%.2e  device=%s  grad_accum=%d",
        max_steps, batch_size, lr, device, gradient_accum,
    )

    # ── Training loop ─────────────────────────────────────────────────
    import torch.nn.functional as F

    step         = 0
    running_loss = 0.0
    loss_count   = 0
    best_val_loss = float("inf")
    optimizer.zero_grad()
    data_iter = iter(train_loader)
    t_start = time.time()

    while step < max_steps:
        # Cycle through data
        try:
            input_ids, targets = next(data_iter)
        except StopIteration:
            data_iter = iter(train_loader)
            input_ids, targets = next(data_iter)

        input_ids = input_ids.to(device)
        targets   = targets.to(device)

        # Forward
        if scaler.is_enabled():
            with torch.amp.autocast(device):
                logits = model(input_ids)
                B, T, V = logits.shape
                loss = F.cross_entropy(
                    logits.reshape(B * T, V),
                    targets.reshape(B * T),
                    ignore_index=-100,          # ignore masked prefix tokens
                )
            scaler.scale(loss / gradient_accum).backward()
        else:
            logits = model(input_ids)
            B, T, V = logits.shape
            loss = F.cross_entropy(
                logits.reshape(B * T, V),
                targets.reshape(B * T),
                ignore_index=-100,
            )
            (loss / gradient_accum).backward()

        running_loss += loss.item()
        loss_count += 1

        if (step + 1) % gradient_accum == 0:
            if scaler.is_enabled():
                scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            if scaler.is_enabled():
                scaler.step(optimizer)
                scaler.update()
            else:
                optimizer.step()
            scheduler.step()
            optimizer.zero_grad()

        # Logging
        if step % log_every == 0 and step > 0:
            avg = running_loss / max(loss_count, 1)
            lr_now = scheduler.get_last_lr()[0]
            elapsed = time.time() - t_start
            eta = (elapsed / step) * (max_steps - step) if step > 0 else 0
            log.info(
                "step %5d / %d  loss=%.4f  lr=%.2e  elapsed=%.0fs  ETA=%.0fs",
                step, max_steps, avg, lr_now, elapsed, eta,
            )
            running_loss = 0.0
            loss_count = 0

        # Validation
        if (step + 1) % (save_every * 2) == 0 and len(val_ds) > 0:
            val_loss = _evaluate_val(model, val_loader, device)
            log.info("step %5d  val_loss=%.4f  (best=%.4f)", step + 1, val_loss, best_val_loss)
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_path = output_dir / "finetune_best.pt"
                _save_checkpoint(
                    model, optimizer, step + 1, val_loss, str(best_path),
                    tokenizer_path=str(tok_path),
                )
                log.info("New best val checkpoint → %s", best_path)

        # Checkpoint
        if (step + 1) % save_every == 0 or step == max_steps - 1:
            ckpt_out = output_dir / f"finetune_step{step+1:05d}.pt"
            _save_checkpoint(
                model, optimizer, step + 1, loss.item(), str(ckpt_out),
                tokenizer_path=str(tok_path),
            )
            log.info("Checkpoint saved -> %s", ckpt_out)

        step += 1

    # Apply a final partial accumulation group. This matters when max_steps is
    # not divisible by gradient_accum (for example the documented 11,250).
    if step % gradient_accum:
        # Losses were divided by the configured accumulation count. Restore
        # the scale that the smaller final group should have had.
        partial_steps = step % gradient_accum
        scale = gradient_accum / partial_steps
        for parameter in model.parameters():
            if parameter.grad is not None:
                parameter.grad.mul_(scale)
        if scaler.is_enabled():
            scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        if scaler.is_enabled():
            scaler.step(optimizer)
            scaler.update()
        else:
            optimizer.step()
        scheduler.step()
        optimizer.zero_grad()

    # Final save
    final_path = output_dir / "finetune_final.pt"
    _save_checkpoint(
        model, optimizer, step, loss.item(), str(final_path),
        tokenizer_path=str(tok_path),
    )

    total_time = time.time() - t_start
    log.info("Fine-tuning complete in %.1fs. Final model -> %s", total_time, final_path)
    if best_val_loss < float("inf"):
        log.info("Best validation loss: %.4f  (saved as finetune_best.pt)", best_val_loss)

    print("\n" + "=" * 60)
    print("FINE-TUNING COMPLETE")
    print("=" * 60)
    print(f"  Total steps:     {step:,}")
    print(f"  Total time:      {total_time:.1f}s")
    print(f"  Final loss:      {loss.item():.4f}")
    print(f"  Best val loss:   {best_val_loss:.4f}" if best_val_loss < float("inf") else "")
    print(f"  Final model:     {final_path}")
    print(f"  Best model:      {output_dir / 'finetune_best.pt'}")
    print("=" * 60)


@torch.no_grad()
def _evaluate_val(model, val_loader, device) -> float:
    """Run a quick validation pass and return average loss."""
    import torch.nn.functional as F
    model.eval()
    total, count = 0.0, 0
    for input_ids, targets in val_loader:
        input_ids, targets = input_ids.to(device), targets.to(device)
        logits = model(input_ids)
        B, T, V = logits.shape
        loss = F.cross_entropy(
            logits.reshape(B * T, V), targets.reshape(B * T), ignore_index=-100,
        )
        total += loss.item()
        count += 1
        if count >= 20:  # cap at 20 batches for speed
            break
    model.train()
    return total / max(count, 1)


def parse_args():
    p = argparse.ArgumentParser(description="Instruction fine-tune the custom MiniLLM")
    p.add_argument("--checkpoint",
                   default="checkpoints/best.pt",
                   help="Pretrained MiniLLM checkpoint to start from")
    p.add_argument("--instruct-jsonl",
                   default="data/processed/openorca_instruct.jsonl",
                   help="OpenOrca instruction JSONL from prepare_openorca.py")
    p.add_argument("--output-dir",     default="checkpoints/finetuned")
    p.add_argument("--max-steps",      type=int,   default=300)
    p.add_argument("--batch-size",     type=int,   default=8)
    p.add_argument("--lr",             type=float, default=5e-5)
    p.add_argument("--warmup-steps",   type=int,   default=30)
    p.add_argument("--gradient-accum", type=int,   default=4)
    p.add_argument(
        "--max-seq-len", type=int, default=None,
        help="Context length (defaults to the checkpoint's maximum).",
    )
    p.add_argument("--eval-split",     type=float, default=0.05,
                   help="Fraction of data to hold out for validation (default: 5%%)")
    p.add_argument("--save-every",     type=int,   default=100)
    p.add_argument("--log-every",      type=int,   default=20)
    p.add_argument("--device",         default="")
    return p.parse_args()


def main():
    setup_logging("finetune_instruct")
    args = parse_args()

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    log.info("Device: %s", device)

    if not Path(args.instruct_jsonl).exists():
        log.error("Instruction JSONL not found: %s", args.instruct_jsonl)
        log.error("Run first: python scripts\\prepare_openorca.py")
        sys.exit(1)

    finetune(
        checkpoint_path = args.checkpoint,
        instruct_jsonl  = args.instruct_jsonl,
        output_dir      = args.output_dir,
        max_steps       = args.max_steps,
        batch_size      = args.batch_size,
        lr              = args.lr,
        warmup_steps    = args.warmup_steps,
        max_seq_len     = args.max_seq_len,
        device          = device,
        gradient_accum  = args.gradient_accum,
        log_every       = args.log_every,
        save_every      = args.save_every,
        eval_split      = args.eval_split,
    )


if __name__ == "__main__":
    main()
