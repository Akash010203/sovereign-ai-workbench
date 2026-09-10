"""
scripts/prepare_openorca.py — Download and prepare OpenOrca data for SovereignAI training.

WHAT THIS SCRIPT DOES
---------------------
1. Streams 5,000 rows from Open-Orca/OpenOrca (no 4.1GB RAM hit)
2. Converts each row into TWO formats:
   a. PRETRAINING format  → raw text, used by train_model.py
   b. INSTRUCTION format  → prompt/response pairs, used by fine-tuning

USAGE
-----
    python scripts\\prepare_openorca.py                     # 5000 examples
    python scripts\\prepare_openorca.py --n 2000            # fewer examples
    python scripts\\prepare_openorca.py --filter-technical  # only tech/science rows

INSTALL REQUIREMENT
-------------------
    pip install datasets

WHAT TO DO AFTER THIS SCRIPT
-----------------------------
    python scripts\\train_model.py  --train-file data/processed/openorca_pretrain.txt
    # OR for instruction fine-tuning (after pretraining):
    python scripts\\finetune_instruct.py

HONESTY NOTE
------------
OpenOrca was created by Microsoft Research.
The data is licensed under MIT / CC-BY.
We are using 5,000 of the 4.2 million examples for educational purposes.
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
) -> dict:
    """
    Stream n rows from OpenOrca, convert to both formats, save to disk.

    Returns a dict with stats.
    """
    try:
        from datasets import load_dataset
    except ImportError:
        log.error("datasets not installed. Run: pip install datasets")
        sys.exit(1)

    output_dir.mkdir(parents=True, exist_ok=True)
    pretrain_path = ROOT / "data" / "processed" / "openorca_pretrain.txt"
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

    pretrain_lines = []
    instruct_rows  = []
    total_seen     = 0
    total_kept     = 0

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

        pretrain_lines.append(pretrain)
        instruct_rows.append({
            "system":    row.get("system_prompt", "")[:500],
            "question":  row.get("question", "")[:500],
            "response":  row.get("response", "")[:1000],
        })
        total_kept += 1

        if total_kept % 500 == 0:
            log.info("  Collected %d / %d examples (seen %d rows)…",
                     total_kept, n, total_seen)

    # ── Save pretraining corpus ─────────────────────────────────────────────
    pretrain_path.write_text("\n".join(pretrain_lines), encoding="utf-8")
    log.info("Pretraining corpus: %d lines → %s", len(pretrain_lines), pretrain_path)

    # ── Save instruction JSONL ──────────────────────────────────────────────
    with instruct_path.open("w", encoding="utf-8") as f:
        for row in instruct_rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    log.info("Instruction JSONL:  %d rows  → %s", len(instruct_rows), instruct_path)

    # ── Save metadata ───────────────────────────────────────────────────────
    stats = {
        "source":          "Open-Orca/OpenOrca",
        "license":         "MIT (see https://huggingface.co/datasets/Open-Orca/OpenOrca)",
        "rows_requested":  n,
        "rows_seen":       total_seen,
        "rows_kept":       total_kept,
        "filter_technical": filter_technical,
        "pretrain_file":   str(pretrain_path),
        "instruct_file":   str(instruct_path),
        "honesty_note": (
            "These 5,000 examples are from the OpenOrca dataset by Microsoft Research. "
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
    return p.parse_args()


def main():
    args  = parse_args()
    stats = download_and_prepare(n=args.n, filter_technical=args.filter_technical)

    print("\n" + "=" * 60)
    print("OpenOrca Data Preparation — DONE")
    print("=" * 60)
    print(f"  Rows collected:    {stats['rows_kept']:,}")
    print(f"  Rows scanned:      {stats['rows_seen']:,}")
    print(f"  Domain filter:     {'YES (technical)' if stats['filter_technical'] else 'NO (all topics)'}")
    print()
    print("  Pretraining text -> data/processed/openorca_pretrain.txt")
    print("  Instruction JSONL -> data/processed/openorca_instruct.jsonl")
    print()
    print("  NEXT STEP — pretrain MiniLLM on this corpus:")
    print("    python scripts\\train_model.py --train-file data/processed/openorca_pretrain.txt --max-steps 500")
    print()
    print("  AFTER THAT — instruction fine-tune:")
    print("    python scripts\\finetune_instruct.py")
    print("=" * 60)


if __name__ == "__main__":
    main()
