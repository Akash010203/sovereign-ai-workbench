"""
scripts/train_model.py — Run MiniLLM training experiments.

THREE EXPERIMENT PRESETS + CUSTOM OVERRIDES
--------------------------------------------
Experiment 1  (--experiment exp1):  10-50 steps — smoke test.
              Proves the full pipeline (data → tokenizer → model →
              optimizer → checkpoint) executes without errors.

Experiment 2  (--experiment exp2): 100-500 steps — learning test.
              Proves the loss curve descends.  By step ~200 the model
              should show noticeably lower loss than at initialization.

Experiment 3  (--experiment exp3): configurable longer run for
              the actual demonstration corpus.

Experiment 4  (--experiment exp4): OpenOrca-scale pretraining.
              Configured for 90K+ rows with larger batch size and
              gradient accumulation.

Custom overrides: use --max-steps, --batch-size, --lr, --device to
override any preset's values.

Usage (Windows PowerShell):
    python scripts\\train_model.py --experiment exp1
    python scripts\\train_model.py --experiment exp2 --resume checkpoints/exp1_step0050_final.pt
    python scripts\\train_model.py --train-file data/processed/openorca_pretrain.txt --max-steps 500 --device cuda
    python scripts\\train_model.py --experiment exp4 --train-file data/processed/openorca_pretrain.txt --device cuda
"""
from __future__ import annotations

import argparse
import logging
import shutil
import sys
from pathlib import Path

# ── Allow imports from project root ──────────────────────────────────────────
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from core.config import get_settings, ensure_directories
from core.logging_setup import setup_logging
from models.custom_minilm.config import MiniLLMConfig
from tokenizer.tokenizer import ByteLevelBPETokenizer
from training.dataset import TokenizedTextDataset, make_dataloader
from training.trainer import Trainer, TrainingConfig


# ── Experiment presets ────────────────────────────────────────────────────────

EXPERIMENTS = {
    "exp1": TrainingConfig(
        experiment_name="exp1_smoke",
        max_steps=50,
        batch_size=4,
        gradient_accumulation_steps=2,
        learning_rate=3e-4,
        warmup_steps=5,
        eval_every=25,
        save_every=50,
    ),
    "exp2": TrainingConfig(
        experiment_name="exp2_learning",
        max_steps=300,
        batch_size=4,
        gradient_accumulation_steps=2,
        learning_rate=3e-4,
        warmup_steps=30,
        eval_every=50,
        save_every=100,
    ),
    "exp3": TrainingConfig(
        experiment_name="exp3_demo",
        max_steps=1000,
        batch_size=8,
        gradient_accumulation_steps=4,
        learning_rate=2e-4,
        warmup_steps=100,
        eval_every=100,
        save_every=250,
    ),
    "exp4": TrainingConfig(
        experiment_name="exp4_openorca",
        max_steps=500,
        batch_size=8,
        gradient_accumulation_steps=4,
        learning_rate=2e-4,
        warmup_steps=50,
        eval_every=100,
        save_every=250,
    ),
}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train SovereignAI MiniLLM")
    p.add_argument(
        "--experiment", choices=list(EXPERIMENTS.keys()), default="exp1",
        help="Experiment preset to run (default: exp1 = 50-step smoke test)",
    )
    p.add_argument(
        "--resume", type=str, default=None,
        help="Path to a checkpoint to resume training from.",
    )
    p.add_argument(
        "--vocab-path", type=str,
        default="tokenizer/vocab/demo_bpe_vocab.json",
        help="Path to the trained tokenizer vocab JSON.",
    )
    p.add_argument(
        "--train-file", type=str,
        default="data/processed/train.txt",
        help="Path to the training text file.",
    )
    p.add_argument(
        "--val-file", type=str, default="",
        help=("Optional validation text file. If omitted, use openorca_val.txt "
              "when training on OpenOrca; otherwise use the demo validation set."),
    )
    # ── CLI overrides — these override the preset values ────────────────
    p.add_argument(
        "--max-steps", type=int, default=None,
        help="Override the preset's max training steps.",
    )
    p.add_argument(
        "--batch-size", type=int, default=None,
        help="Override the preset's batch size.",
    )
    p.add_argument(
        "--lr", type=float, default=None,
        help="Override the preset's learning rate.",
    )
    p.add_argument(
        "--device", type=str, default=None,
        help="Force device: 'cuda' or 'cpu'. Default: auto-detect.",
    )
    p.add_argument(
        "--model-size", choices=["small", "medium"], default="small",
        help="Model size preset: 'small' (~5M params) or 'medium' (~12M params).",
    )
    p.add_argument(
        "--export-checkpoint", type=str, default="checkpoints/best.pt",
        help=("Stable path for the selected validation checkpoint (or final checkpoint "
              "when validation is unavailable). Set to an empty string to skip export."),
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()
    setup_logging("train_model")
    ensure_directories()

    log = logging.getLogger("train_model")
    settings = get_settings()

    # ── Load tokenizer ────────────────────────────────────────────────
    vocab_path = ROOT / args.vocab_path
    if not vocab_path.exists():
        log.error(
            "Tokenizer vocab not found at %s. "
            "Run: python scripts\\train_tokenizer.py", vocab_path
        )
        sys.exit(1)

    log.info("Loading tokenizer from %s", vocab_path)
    tokenizer = ByteLevelBPETokenizer.load(vocab_path)
    log.info("Vocab size: %d", tokenizer.vocab_size)

    # ── Model config ──────────────────────────────────────────────────
    if args.model_size == "medium":
        model_cfg = MiniLLMConfig.medium(vocab_size=tokenizer.vocab_size)
    else:
        model_cfg = MiniLLMConfig.small(vocab_size=tokenizer.vocab_size)

    log.info(
        "Model config (%s): d_model=%d, n_layers=%d, n_heads=%d, d_ff=%d, max_seq_len=%d",
        args.model_size, model_cfg.d_model, model_cfg.n_layers, model_cfg.n_heads,
        model_cfg.d_ff, model_cfg.max_seq_len,
    )

    # ── Training config ───────────────────────────────────────────────
    train_cfg = EXPERIMENTS[args.experiment]
    train_cfg.tokenizer_path = str(vocab_path)

    # Apply CLI overrides
    if args.max_steps is not None:
        train_cfg.max_steps = args.max_steps
    if args.batch_size is not None:
        train_cfg.batch_size = args.batch_size
    if args.lr is not None:
        train_cfg.learning_rate = args.lr

    log.info(
        "Experiment: %s | max_steps=%d | batch=%d x accum=%d | lr=%.2e",
        train_cfg.experiment_name, train_cfg.max_steps,
        train_cfg.batch_size, train_cfg.gradient_accumulation_steps,
        train_cfg.learning_rate,
    )

    # ── Datasets ──────────────────────────────────────────────────────
    block_size = model_cfg.max_seq_len

    train_path = ROOT / args.train_file
    if args.val_file:
        val_path = ROOT / args.val_file
    elif train_path.name == "openorca_pretrain.txt":
        val_path = train_path.with_name("openorca_val.txt")
    else:
        val_path = ROOT / "data" / "processed" / "val.txt"

    if not train_path.exists():
        log.error("Training file not found: %s", train_path)
        sys.exit(1)

    log.info("Building training dataset from %s", train_path)
    train_ds = TokenizedTextDataset(
        train_path, tokenizer, block_size=block_size, stride=block_size // 2
    )

    val_ds = None
    val_loader = None
    if val_path.exists():
        log.info("Building validation dataset from %s", val_path)
        val_ds = TokenizedTextDataset(
            val_path, tokenizer, block_size=block_size, stride=block_size
        )

    train_loader = make_dataloader(train_ds, batch_size=train_cfg.batch_size)
    if val_ds is not None and len(val_ds) > 0:
        val_loader = make_dataloader(
            val_ds, batch_size=train_cfg.batch_size, shuffle=False
        )
        log.info("Val dataset: %d samples", len(val_ds))
    else:
        log.warning("Validation dataset empty or unavailable — skipping val eval.")

    log.info("Train dataset: %d samples", len(train_ds))

    # ── Device override ───────────────────────────────────────────────
    import torch
    if args.device:
        device_override = torch.device(args.device)
    else:
        device_override = None

    # ── Trainer ───────────────────────────────────────────────────────
    trainer = Trainer(
        model_cfg=model_cfg,
        train_cfg=train_cfg,
        tokenizer=tokenizer,
        resume_from=args.resume,
        device_override=device_override,
    )

    # ── Print parameter summary ───────────────────────────────────────
    print(trainer.model.parameter_summary())

    # ── Train ─────────────────────────────────────────────────────────
    summary = trainer.fit(train_loader, val_loader)

    # The long-form experiment checkpoint names are useful for provenance, but
    # callers (the backend and the documented fine-tuning command) need one
    # stable path. Export the actual best validation state, falling back to the
    # final state when no validation file was supplied.
    if args.export_checkpoint:
        export_path = ROOT / args.export_checkpoint
        export_path.parent.mkdir(parents=True, exist_ok=True)
        selected = Path(summary["best_checkpoint"])
        if selected.resolve() != export_path.resolve():
            shutil.copy2(selected, export_path)
        summary["export_checkpoint"] = str(export_path)
        log.info("Selected checkpoint exported -> %s", export_path)

    # ── Final report ──────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("TRAINING COMPLETE")
    print("=" * 60)
    for k, v in summary.items():
        print(f"  {k:<25}: {v}")
    print("=" * 60)
    print(f"\nCheckpoints saved in: {summary['checkpoint_dir']}")
    print(f"Training log    in  : logs/{train_cfg.experiment_name}_log.json")


if __name__ == "__main__":
    main()
