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
import sys
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


def _save_checkpoint(model, optimizer, step: int, loss: float, path: str, **meta) -> None:
    """Save a fine-tuning checkpoint in the same format as the Trainer."""
    import torch
    torch.save({
        "model_state_dict":     model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "step":                 step,
        "loss":                 loss,
        "model_config":         model.cfg.__dict__ if hasattr(model, "cfg") else {},
        **meta,
    }, path)

log = logging.getLogger(__name__)

# ─────────────────────────────────────────────────────────────────────────────
# Instruction dataset
# ─────────────────────────────────────────────────────────────────────────────

class InstructDataset(Dataset):
    """
    Dataset for instruction fine-tuning.

    Each example is a full prompt:
        <|system|>\\n{sys}\\n<|user|>\\n{q}\\n<|assistant|>\\n{r}

    The loss is computed ONLY on the response tokens (after <|assistant|>).
    This is standard SFT practice — we don't penalise the model for
    "predicting" the question it already received.
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
        self.examples: list[dict] = []

        raw = jsonl_path.read_text(encoding="utf-8").strip().splitlines()
        for line in raw:
            try:
                self.examples.append(json.loads(line))
            except Exception:
                pass
        log.info("InstructDataset: %d examples from %s", len(self.examples), jsonl_path)

    def __len__(self) -> int:
        return len(self.examples)

    def _encode(self, text: str) -> list[int]:
        return self.tokenizer.encode(text)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        ex = self.examples[idx]

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

        # Truncate to max_seq_len
        full_ids = full_ids[: self.max_seq_len + 1]

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
    max_seq_len:     int  = 256,
    device:          str  = "cpu",
    gradient_accum:  int  = 4,
    log_every:       int  = 20,
    save_every:      int  = 100,
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
        tok_path    = meta.get("tokenizer_path", "tokenizer/vocab/demo_bpe_vocab.json")
        tokenizer   = ByteLevelBPETokenizer.load(ROOT / tok_path)
        log.info("Loaded checkpoint: step=%d  loss=%.4f", meta["step"], meta.get("loss", 0))

    model = model.to(device)
    model.train()

    # ── Dataset & loader ──────────────────────────────────────────────
    ds = InstructDataset(
        Path(instruct_jsonl), tokenizer, max_seq_len=max_seq_len
    )
    loader = DataLoader(ds, batch_size=batch_size, shuffle=True, drop_last=True)

    # ── Optimizer & scheduler ─────────────────────────────────────────
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    scheduler = cosine_with_warmup(optimizer, warmup_steps, max_steps)
    scaler    = torch.amp.GradScaler(device) if device == "cuda" else None

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    log.info(
        "Fine-tuning: max_steps=%d  batch=%d  lr=%.2e  device=%s",
        max_steps, batch_size, lr, device
    )

    # ── Training loop ─────────────────────────────────────────────────
    import torch.nn.functional as F

    step         = 0
    running_loss = 0.0
    optimizer.zero_grad()
    data_iter = iter(loader)

    while step < max_steps:
        # Cycle through data
        try:
            input_ids, targets = next(data_iter)
        except StopIteration:
            data_iter = iter(loader)
            input_ids, targets = next(data_iter)

        input_ids = input_ids.to(device)
        targets   = targets.to(device)

        # Forward
        if scaler:
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

        if (step + 1) % gradient_accum == 0:
            if scaler:
                scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            if scaler:
                scaler.step(optimizer)
                scaler.update()
            else:
                optimizer.step()
            scheduler.step()
            optimizer.zero_grad()

        # Logging
        if step % log_every == 0:
            avg = running_loss / max(log_every, 1)
            lr_now = scheduler.get_last_lr()[0]
            log.info("step %4d / %d  loss=%.4f  lr=%.2e", step, max_steps, avg, lr_now)
            running_loss = 0.0

        # Checkpoint
        if (step + 1) % save_every == 0 or step == max_steps - 1:
            ckpt_out = output_dir / f"finetune_step{step+1:05d}.pt"
            _save_checkpoint(model, optimizer, step + 1, loss.item(), str(ckpt_out))
            log.info("Checkpoint saved -> %s", ckpt_out)

        step += 1

    # Final save
    final_path = output_dir / "finetune_final.pt"
    _save_checkpoint(model, optimizer, step, loss.item(), str(final_path))
    log.info("Fine-tuning complete. Final model -> %s", final_path)


def parse_args():
    p = argparse.ArgumentParser(description="Instruction fine-tune the custom MiniLLM")
    p.add_argument("--checkpoint",
                   default="checkpoints/best.pt",
                   help="Pretrained MiniLLM checkpoint to start from")
    p.add_argument("--instruct-jsonl",
                   default="data/processed/openorca_instruct.jsonl",
                   help="OpenOrca instruction JSONL from prepare_openorca.py")
    p.add_argument("--output-dir",   default="checkpoints/finetuned")
    p.add_argument("--max-steps",    type=int,   default=300)
    p.add_argument("--batch-size",   type=int,   default=8)
    p.add_argument("--lr",           type=float, default=5e-5)
    p.add_argument("--max-seq-len",  type=int,   default=256)
    p.add_argument("--device",       default="")
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
        max_seq_len     = args.max_seq_len,
        device          = device,
    )


if __name__ == "__main__":
    main()
