"""
scripts/train_model.py — Run MiniLLM training experiments.

THREE EXPERIMENTS
-----------------
Experiment 1  (--experiment exp1):  10-50 steps — smoke test.
              Proves the full pipeline (data → tokenizer → model →
              optimizer → checkpoint) executes without errors.

Experiment 2  (--experiment exp2): 100-500 steps — learning test.
              Proves the loss curve descends.  By step ~200 the model
              should show noticeably lower loss than at initialization.

Experiment 3  (--experiment exp3): configurable longer run for
              the actual demonstration corpus.

Usage (Windows PowerShell):
    python scripts\\train_model.py --experiment exp1
    python scripts\\train_model.py --experiment exp2
    python scripts\\train_model.py --experiment exp2 --resume checkpoints/exp1_step0050_final.pt
"""
from __future__ import annotations

import argparse
import logging
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
        "--val-file", type=str,
        default="data/processed/val.txt",
        help="Path to the validation text file.",
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
    model_cfg = MiniLLMConfig(vocab_size=tokenizer.vocab_size)
    log.info(
        "Model config: d_model=%d, n_layers=%d, n_heads=%d, d_ff=%d, max_seq_len=%d",
        model_cfg.d_model, model_cfg.n_layers, model_cfg.n_heads,
        model_cfg.d_ff, model_cfg.max_seq_len,
    )

    # ── Training config ───────────────────────────────────────────────
    train_cfg = EXPERIMENTS[args.experiment]
    train_cfg.tokenizer_path = str(vocab_path)
    log.info(
        "Experiment: %s | max_steps=%d | batch=%d × accum=%d",
        train_cfg.experiment_name, train_cfg.max_steps,
        train_cfg.batch_size, train_cfg.gradient_accumulation_steps,
    )

    # ── Datasets ──────────────────────────────────────────────────────
    block_size = model_cfg.max_seq_len

    train_path = ROOT / args.train_file
    val_path   = ROOT / args.val_file

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

    # ── Trainer ───────────────────────────────────────────────────────
    trainer = Trainer(
        model_cfg=model_cfg,
        train_cfg=train_cfg,
        tokenizer=tokenizer,
        resume_from=args.resume,
    )

    # ── Print parameter summary ───────────────────────────────────────
    print(trainer.model.parameter_summary())

    # ── Train ─────────────────────────────────────────────────────────
    summary = trainer.fit(train_loader, val_loader)

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
