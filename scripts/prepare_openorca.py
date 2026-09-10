"""
scripts/prepare_openorca.py — Download and prepare OpenOrca data for SovereignAI training.

WHAT THIS SCRIPT DOES
---------------------
1. Streams N rows from Open-Orca/OpenOrca (no full dataset RAM hit)
2. Converts each row into TWO formats:
   a. PRETRAINING format  → raw text, used by train_model.py
   b. INSTRUCTION format  → prompt/response pairs, used by fine-tuning

USAGE
-----
    python scripts\\prepare_openorca.py                     # 5000 examples
    python scripts\\prepare_openorca.py --n 90000           # 90K examples
    python scripts\\prepare_openorca.py --n 2000            # fewer examples
    python scripts\\prepare_openorca.py --filter-technical  # only tech/science rows

INSTALL REQUIREMENT
-------------------
    pip install datasets tqdm

WHAT TO DO AFTER THIS SCRIPT
-----------------------------
    python scripts\\train_model.py  --train-file data/processed/openorca_pretrain.txt --max-steps 500 --device cuda
    # OR for instruction fine-tuning (after pretraining):
    python scripts\\finetune_instruct.py --device cuda

HONESTY NOTE
------------
OpenOrca was created by Microsoft Research.
The data is licensed under MIT / CC-BY.
We are using a user-specified number of the 4.2 million examples for educational purposes.
The fine-tuned model weights are derived from OpenOrca data — this will be
documented in the honesty ledger.
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
# Avoid diagnostic logging failures in legacy Windows console encodings.
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(errors="backslashreplace")
log = logging.getLogger(__name__)

# ── Technical/science keywords for domain filtering ─────────────────────────
TECHNICAL_KEYWORDS = {
    "engine", "motor", "pressure", "flow", "valve", "sensor", "pipe",
    "mechanical", "electrical", "circuit", "chemical", "physics", "math",
    "calcul", "formula", "equation", "analyse", "analysis", "report",
    "procedure", "safety", "inspection", "maintenance", "technical",
    "system", "data", "measure", "process", "temperature", "force",
    "python", "code", "function", "algorithm", "program", "software",
    "science", "engineering", "industrial", "plant", "compressor", "pump",
}


def is_technical(row: dict) -> bool:
    """Return True if the row is likely a technical/engineering/science query."""
    text = (
        row.get("system_prompt", "") + " " +
        row.get("question", "") + " " +
        row.get("response", "")
    ).lower()
    return any(kw in text for kw in TECHNICAL_KEYWORDS)


def row_to_pretrain_text(row: dict) -> str:
    """
    Convert one OpenOrca row to raw pretraining text.

    For pretraining (language modelling), we just concatenate the
    instruction + response as natural text.  The model learns the
    language distribution.
    """
    q = row.get("question", "").strip()
    r = row.get("response", "").strip()
    if not q or not r:
        return ""
    return f"{q}\n{r}"


def row_to_instruct_text(row: dict) -> str:
    """
    Convert one OpenOrca row to instruction-tuning format.

    Format:
        <|system|>
        {system_prompt}
        <|user|>
        {question}
        <|assistant|>
        {response}

    This matches the prompt format used by phi-3 and most local instruction models.
    """
    sys_prompt = row.get("system_prompt", "You are a helpful assistant.").strip()
    question   = row.get("question", "").strip()
    response   = row.get("response", "").strip()
    if not question or not response:
        return ""
    return (
        f"<|system|>\n{sys_prompt}\n"
        f"<|user|>\n{question}\n"
        f"<|assistant|>\n{response}"
    )


def download_and_prepare(
    n: int = 5000,
    filter_technical: bool = False,
    output_dir: Path = ROOT / "data" / "raw",
    max_field_length: int = 0,
    validation_ratio: float = 0.02,
) -> dict:
    """
    Stream n rows from OpenOrca, convert to both formats, save to disk.

    MEMORY-EFFICIENT: Writes directly to disk instead of accumulating
    in RAM. For 90K rows this avoids the ~500MB RAM spike.

    Returns a dict with stats.
    """
    try:
        from datasets import load_dataset
    except ImportError:
        log.error("datasets not installed. Run: pip install datasets")
        sys.exit(1)

    # Optional progress bar
    try:
        from tqdm import tqdm
        use_tqdm = True
    except ImportError:
        use_tqdm = False

    output_dir.mkdir(parents=True, exist_ok=True)
    if not 0.0 <= validation_ratio < 1.0:
        raise ValueError("validation_ratio must be in [0.0, 1.0).")

    pretrain_path = ROOT / "data" / "processed" / "openorca_pretrain.txt"
    val_path = ROOT / "data" / "processed" / "openorca_val.txt"
    instruct_path = ROOT / "data" / "processed" / "openorca_instruct.jsonl"
    metadata_path = ROOT / "data" / "metadata" / "openorca.json"
    pretrain_path.parent.mkdir(parents=True, exist_ok=True)
    metadata_path.parent.mkdir(parents=True, exist_ok=True)

    log.info("Streaming OpenOrca (n=%d, filter_technical=%s)…", n, filter_technical)

    # ── Stream from HuggingFace ─────────────────────────────────────────────
    dataset_stream = load_dataset(
        "Open-Orca/OpenOrca",
        split="train",
        streaming=True,
    )

    total_seen     = 0
    total_kept     = 0

    # Stream-write to disk instead of accumulating in RAM
    pbar = tqdm(total=n, desc="Collecting examples", unit="row") if use_tqdm else None

    holdout_every = round(1 / validation_ratio) if validation_ratio else 0
    train_rows = 0
    val_rows = 0
    with pretrain_path.open("w", encoding="utf-8") as pretrain_f, \
         val_path.open("w", encoding="utf-8") as val_f, \
         instruct_path.open("w", encoding="utf-8") as instruct_f:

        for row in dataset_stream:
            total_seen += 1
            if total_kept >= n:
                break

            # Optional domain filter
            if filter_technical and not is_technical(row):
                continue

            pretrain = row_to_pretrain_text(row)
            instruct = row_to_instruct_text(row)

            if not pretrain or not instruct:
                continue

            # Reserve a deterministic document-level validation split for
            # pretraining. The instruction data has its own random split in
            # finetune_instruct.py.
            is_val = holdout_every and (total_kept + 1) % holdout_every == 0
            (val_f if is_val else pretrain_f).write(pretrain + "\n")
            if is_val:
                val_rows += 1
            else:
                train_rows += 1

            # Write instruction JSONL directly to disk
            instruct_row = {
                "system":    row.get("system_prompt", ""),
                "question":  row.get("question", ""),
                "response":  row.get("response", ""),
            }
            # Apply field length limits if specified
            if max_field_length > 0:
                instruct_row["system"]   = instruct_row["system"][:max_field_length]
                instruct_row["question"] = instruct_row["question"][:max_field_length]
                instruct_row["response"] = instruct_row["response"][:max_field_length * 2]

            instruct_f.write(json.dumps(instruct_row, ensure_ascii=False) + "\n")

            total_kept += 1

            if pbar:
                pbar.update(1)
            elif total_kept % 500 == 0:
                log.info("  Collected %d / %d examples (seen %d rows)…",
                         total_kept, n, total_seen)

    if pbar:
        pbar.close()

    log.info("Pretraining corpus: %d train / %d validation examples → %s", train_rows, val_rows, pretrain_path)
    log.info("Instruction JSONL:  %d rows  → %s", total_kept, instruct_path)

    # ── Save metadata ───────────────────────────────────────────────────
    stats = {
        "source":          "Open-Orca/OpenOrca",
        "license":         "MIT (see https://huggingface.co/datasets/Open-Orca/OpenOrca)",
        "rows_requested":  n,
        "rows_seen":       total_seen,
        "rows_kept":       total_kept,
        "pretrain_train_rows": train_rows,
        "pretrain_val_rows": val_rows,
        "validation_ratio": validation_ratio,
        "filter_technical": filter_technical,
        "pretrain_file":   str(pretrain_path),
        "val_file":        str(val_path),
        "instruct_file":   str(instruct_path),
        "honesty_note": (
            f"These {total_kept:,} examples are from the OpenOrca dataset by Microsoft Research. "
            "They are used for fine-tuning / pretraining data only. "
            "The custom MiniLLM model architecture remains our original work. "
            "Data source and license are documented here per the project honesty contract."
        ),
    }
    metadata_path.write_text(json.dumps(stats, indent=2), encoding="utf-8")
    log.info("Metadata saved → %s", metadata_path)

    return stats


def parse_args():
    p = argparse.ArgumentParser(description="Prepare OpenOrca training data")
    p.add_argument("--n",                type=int,  default=5000,  help="Number of examples to collect")
    p.add_argument("--filter-technical", action="store_true",       help="Only keep technical/engineering rows")
    p.add_argument("--max-field-length", type=int, default=0,
                   help="Max chars per field (0 = no limit). E.g. 1000 to cap response length.")
    p.add_argument("--validation-ratio", type=float, default=0.02,
                   help="Document fraction held out for pretraining validation (default: 0.02).")
    return p.parse_args()


def main():
    args  = parse_args()
    stats = download_and_prepare(
        n=args.n,
        filter_technical=args.filter_technical,
        max_field_length=args.max_field_length,
        validation_ratio=args.validation_ratio,
    )

    print("\n" + "=" * 60)
    print("OpenOrca Data Preparation — DONE")
    print("=" * 60)
    print(f"  Rows collected:    {stats['rows_kept']:,}")
    print(f"  Rows scanned:      {stats['rows_seen']:,}")
    print(f"  Domain filter:     {'YES (technical)' if stats['filter_technical'] else 'NO (all topics)'}")
    print()
    print("  Pretraining text -> data/processed/openorca_pretrain.txt")
    print("  Validation text  -> data/processed/openorca_val.txt")
    print("  Instruction JSONL -> data/processed/openorca_instruct.jsonl")
    print()
    print("  NEXT STEP — pretrain MiniLLM on this corpus:")
    print("    python scripts\\train_model.py --train-file data/processed/openorca_pretrain.txt --max-steps 500 --device cuda")
    print()
    print("  AFTER THAT — instruction fine-tune:")
    print("    python scripts\\finetune_instruct.py --max-steps 11250 --batch-size 16 --device cuda")
    print("=" * 60)


if __name__ == "__main__":
    main()
