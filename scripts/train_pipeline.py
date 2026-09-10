"""
scripts/train_pipeline.py — One-command full training pipeline.

Orchestrates the full OpenOrca training workflow:
    Step 1: Download data from OpenOrca
    Step 2: Train a matching tokenizer on a bounded sample
    Step 3: Pretrain MiniLLM on raw text
    Step 4: Instruction fine-tune on Q&A pairs

Usage:
    python scripts\\train_pipeline.py --n 90000 --device cuda
    python scripts\\train_pipeline.py --n 5000 --device cpu    # quick test
    python scripts\\train_pipeline.py --skip-download --device cuda  # if data already downloaded

This is a convenience wrapper — each step can also be run independently.
"""
from __future__ import annotations

import argparse
import logging
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
log = logging.getLogger("train_pipeline")


# ── ANSI colours ─────────────────────────────────────────────────────────────
CYAN  = "\033[96m"
GREEN = "\033[92m"
RED   = "\033[91m"
BOLD  = "\033[1m"
RESET = "\033[0m"


def banner(step: int, total: int, text: str) -> None:
    print(f"\n{CYAN}{BOLD}{'='*60}{RESET}")
    print(f"{CYAN}{BOLD}  STEP {step}/{total}: {text}{RESET}")
    print(f"{CYAN}{BOLD}{'='*60}{RESET}\n")


def run_step(cmd: list[str], step_name: str) -> bool:
    """Run a subprocess and return True on success."""
    log.info("Running: %s", " ".join(cmd))
    try:
        result = subprocess.run(
            cmd,
            cwd=str(ROOT),
            check=True,
            text=True,
        )
        return True
    except subprocess.CalledProcessError as e:
        log.error("%s failed with exit code %d", step_name, e.returncode)
        return False


def parse_args():
    p = argparse.ArgumentParser(description="SovereignAI Full Training Pipeline")
    p.add_argument("--n", type=int, default=5000,
                   help="Number of OpenOrca examples to download (default: 5000)")
    p.add_argument("--device", default="auto",
                   help="Training device: 'cuda', 'cpu', or 'auto' (default: auto)")
    p.add_argument("--model-size", choices=["small", "medium"], default="small",
                   help="Model size: 'small' (~5M) or 'medium' (~12M)")
    p.add_argument("--vocab-path", default="tokenizer/vocab/openorca_bpe_vocab.json",
                   help="Tokenizer path shared by pretraining and fine-tuning")
    p.add_argument("--tokenizer-vocab-size", type=int, default=1000)
    p.add_argument("--tokenizer-max-lines", type=int, default=10000,
                   help="Lines sampled to train the BPE tokenizer")

    # Steps to skip
    p.add_argument("--skip-download", action="store_true",
                   help="Skip step 1 (data already downloaded)")
    p.add_argument("--skip-tokenizer", action="store_true",
                   help="Skip step 2 (matching tokenizer already exists)")
    p.add_argument("--skip-pretrain", action="store_true",
                   help="Skip step 3 (already have a pretrained checkpoint)")
    p.add_argument("--skip-finetune", action="store_true",
                   help="Skip step 4")

    # Step-specific overrides
    p.add_argument("--pretrain-steps", type=int, default=500,
                   help="Number of pretraining steps (default: 500)")
    p.add_argument("--finetune-steps", type=int, default=0,
                   help="Number of fine-tuning steps (0 = auto-calculate from data)")
    p.add_argument("--finetune-batch", type=int, default=16,
                   help="Fine-tuning batch size (default: 16)")
    p.add_argument("--finetune-lr", type=float, default=5e-5,
                   help="Fine-tuning learning rate (default: 5e-5)")

    return p.parse_args()


def main():
    args = parse_args()
    t_start = time.time()

    # Auto-detect device
    if args.device == "auto":
        try:
            import torch
            device = "cuda" if torch.cuda.is_available() else "cpu"
        except ImportError:
            device = "cpu"
    else:
        device = args.device

    python = sys.executable  # Use the same Python that's running this script

    total_steps = 4
    results = {}

    # ── STEP 1: Download data ─────────────────────────────────────────
    if not args.skip_download:
        banner(1, total_steps, "Download & prepare OpenOrca data")
        success = run_step([
            python, "scripts/prepare_openorca.py",
            "--n", str(args.n),
        ], "Data download")
        results["download"] = success
        if not success:
            print(f"\n{RED}Step 1 failed. Fix the error and re-run.{RESET}")
            sys.exit(1)
    else:
        log.info("Skipping download (--skip-download)")
        results["download"] = "skipped"

    # ── STEP 2: Tokenizer ─────────────────────────────────────────────
    vocab_path = ROOT / args.vocab_path
    if not args.skip_tokenizer:
        banner(2, total_steps, "Train a BPE tokenizer for OpenOrca")
        pretrain_file = ROOT / "data" / "processed" / "openorca_pretrain.txt"
        success = run_step([
            python, "scripts/train_tokenizer.py",
            "--train-file", str(pretrain_file),
            "--output", str(vocab_path),
            "--vocab-size", str(args.tokenizer_vocab_size),
            "--max-lines", str(args.tokenizer_max_lines),
            "--instruction-format",
        ], "Tokenizer training")
        results["tokenizer"] = success
        if not success:
            print(f"\n{RED}Step 2 failed. Check the error above.{RESET}")
            sys.exit(1)
    elif not vocab_path.exists():
        log.error("Tokenizer not found: %s", vocab_path)
        sys.exit(1)
    else:
        log.info("Skipping tokenizer (--skip-tokenizer)")
        results["tokenizer"] = "skipped"

    # ── STEP 3: Pretrain ──────────────────────────────────────────────
    if not args.skip_pretrain:
        banner(3, total_steps, "Pretrain MiniLLM on raw text corpus")

        pretrain_file = ROOT / "data" / "processed" / "openorca_pretrain.txt"
        if not pretrain_file.exists():
            # Fall back to demo corpus
            pretrain_file = ROOT / "data" / "processed" / "train.txt"
            log.warning("OpenOrca pretrain file not found, using demo corpus: %s", pretrain_file)

        cmd = [
            python, "scripts/train_model.py",
            "--experiment", "exp4",
            "--train-file", str(pretrain_file),
            "--max-steps", str(args.pretrain_steps),
            "--device", device,
            "--model-size", args.model_size,
            "--vocab-path", str(vocab_path),
        ]
        success = run_step(cmd, "Pretraining")
        results["pretrain"] = success
        if not success:
            print(f"\n{RED}Step 2 failed. Check the error above.{RESET}")
            sys.exit(1)
    else:
        log.info("Skipping pretraining (--skip-pretrain)")
        results["pretrain"] = "skipped"

    # ── STEP 4: Fine-tune ─────────────────────────────────────────────
    if not args.skip_finetune:
        banner(4, total_steps, "Instruction fine-tune on Q&A pairs")

        instruct_file = ROOT / "data" / "processed" / "openorca_instruct.jsonl"
        if not instruct_file.exists():
            log.error("Instruction JSONL not found: %s", instruct_file)
            log.error("Run step 1 first (or remove --skip-download)")
            sys.exit(1)

        # Auto-calculate steps if not specified
        ft_steps = args.finetune_steps
        if ft_steps <= 0:
            # ~3 epochs: count lines, divide by batch_size, multiply by 3
            n_lines = sum(1 for _ in instruct_file.open())
            ft_steps = max(100, (n_lines // args.finetune_batch) * 3)
            log.info("Auto-calculated fine-tune steps: %d (%d examples × 3 epochs / batch %d)",
                     ft_steps, n_lines, args.finetune_batch)

        checkpoint = ROOT / "checkpoints" / "best.pt"
        if not checkpoint.exists():
            # Try to find any checkpoint
            ckpts = sorted(ROOT.glob("checkpoints/*.pt"))
            if ckpts:
                checkpoint = ckpts[-1]
                log.warning("best.pt not found, using: %s", checkpoint)
            else:
                log.warning("No pretrained checkpoint found — fine-tuning from scratch!")

        cmd = [
            python, "scripts/finetune_instruct.py",
            "--checkpoint", str(checkpoint),
            "--instruct-jsonl", str(instruct_file),
            "--max-steps", str(ft_steps),
            "--batch-size", str(args.finetune_batch),
            "--lr", str(args.finetune_lr),
            "--device", device,
        ]
        success = run_step(cmd, "Fine-tuning")
        results["finetune"] = success
        if not success:
            print(f"\n{RED}Step 4 failed. Check the error above.{RESET}")
            sys.exit(1)
    else:
        log.info("Skipping fine-tuning (--skip-finetune)")
        results["finetune"] = "skipped"

    # ── Summary ───────────────────────────────────────────────────────
    elapsed = time.time() - t_start

    print(f"\n{GREEN}{BOLD}{'═'*60}{RESET}")
    print(f"{GREEN}{BOLD}  TRAINING PIPELINE COMPLETE{RESET}")
    print(f"{GREEN}{BOLD}{'═'*60}{RESET}")
    print(f"  Device:       {device}")
    print(f"  Model size:   {args.model_size}")
    print(f"  Data rows:    {args.n:,}")
    print(f"  Total time:   {elapsed:.1f}s ({elapsed/60:.1f} min)")
    print()
    for step_name, result in results.items():
        icon = "OK" if result is True else ("SKIP" if result == "skipped" else "FAIL")
        print(f"  {icon} {step_name}")
    print()
    print("  Checkpoints:")
    print("    Pretrained  -> checkpoints/best.pt")
    print("    Fine-tuned  -> checkpoints/finetuned/finetune_final.pt")
    print("    Best (val)  -> checkpoints/finetuned/finetune_best.pt")
    print(f"{GREEN}{BOLD}{'='*60}{RESET}")


if __name__ == "__main__":
    main()
