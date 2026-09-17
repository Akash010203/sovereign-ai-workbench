"""Create a disk-backed, ratio-controlled corpus from three HF datasets.

The datasets are read with ``streaming=True``: neither the multi-million-row
NVIDIA dataset nor the 2.1M-row maintenance dataset is loaded into RAM.  The
result is line-oriented text that existing tokenizer and training scripts can
consume.  A document-level holdout is created at the same time.

The default 45/35/20 blend intentionally gives industrial maintenance enough
weight to become a real speciality without erasing general instruction and
reasoning behaviour.  It is not a claim that any blend makes an 80M model
"perfect"; a model at this scale still needs extensive token exposure and
evaluation.

Example (RTX 4050 / 16 GB RAM):
    python scripts/prepare_blended_corpus.py --total-rows 300000
    python scripts/train_tokenizer.py --train-file data/processed/blended_train.txt \\
        --output tokenizer/vocab/blended_16k.json --vocab-size 16000 --max-lines 250000 \\
        --instruction-format
    python scripts/train_model.py --experiment industrial_4050 --model-size industrial-80m \\
        --vocab-path tokenizer/vocab/blended_16k.json --train-file data/processed/blended_train.txt \\
        --val-file data/processed/blended_val.txt --device cuda
"""
from __future__ import annotations

import argparse
import hashlib
import json
import logging
import os
import sys
from collections.abc import Iterable, Iterator
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

log = logging.getLogger("prepare_blended_corpus")

DEFAULT_WEIGHTS = {"maintenance": 0.45, "nemotron": 0.35, "openorca": 0.20}
NEMOTRON_CODE_URL = (
    "https://huggingface.co/datasets/nvidia/Llama-Nemotron-Post-Training-Dataset/"
    "resolve/main/SFT/code/code_v1.1.jsonl"
)


def _clean(value: Any) -> str:
    """Turn arbitrary dataset fields into compact, line-safe text."""
    if value is None:
        return ""
    if isinstance(value, str):
        return " ".join(value.split())
    return " ".join(str(value).split())


def _conversation(value: Any) -> str:
    """Render common chat-list schemas without assuming one exact schema."""
    if not isinstance(value, list):
        return _clean(value)
    parts: list[str] = []
    for message in value:
        if not isinstance(message, dict):
            continue
        role = _clean(message.get("role") or message.get("from") or "user")
        content = _clean(message.get("content") or message.get("value"))
        if content:
            parts.append(f"<|{role}|> {content}")
    return " ".join(parts)


def nemotron_to_text(row: dict[str, Any]) -> str:
    """Format NVIDIA SFT rows, whose input is normally a chat-message list."""
    prompt = _conversation(row.get("input") or row.get("messages") or row.get("prompt"))
    answer = _clean(row.get("output") or row.get("response") or row.get("chosen"))
    if not prompt or not answer:
        return ""
    return f"{prompt} <|assistant|> {answer}"


def stream_jsonl(url: str) -> Iterator[dict[str, Any]]:
    """Read JSONL sequentially, bypassing fsspec's ranged-file cache.

    The NVIDIA code split is a very large Xet-backed JSONL object.  Some
    Windows/network combinations repeatedly retry HTTP range reads (206) via
    ``datasets`` before producing a row.  A normal sequential HTTP stream is
    both lower-overhead and reliable for this append-only use case.
    """
    try:
        import requests
    except ImportError as exc:
        raise RuntimeError("Install dependencies first: pip install -r requirements.txt") from exc

    headers = {"User-Agent": "sovereign-ai-workbench/1.0"}
    if token := os.environ.get("HF_TOKEN"):
        headers["Authorization"] = f"Bearer {token}"
    log.info("Opening NVIDIA JSONL as a sequential stream (no HTTP range cache)")
    try:
        with requests.get(url, headers=headers, stream=True, timeout=(20, 300)) as response:
            response.raise_for_status()
            for line_number, raw_line in enumerate(response.iter_lines(chunk_size=128 * 1024), start=1):
                if not raw_line:
                    continue
                try:
                    yield json.loads(raw_line)
                except json.JSONDecodeError as exc:
                    raise RuntimeError(f"Invalid JSON in NVIDIA stream at line {line_number}.") from exc
    except requests.RequestException as exc:
        raise RuntimeError(
            "The NVIDIA sequential download was interrupted. Check your network, "
            "then rerun the command; no completed corpus is published on failure."
        ) from exc


def stream_local_parquet(path: Path) -> Iterator[dict[str, Any]]:
    """Read a downloaded Parquet file in small record batches."""
    try:
        import pyarrow.parquet as pq
    except ImportError as exc:
        raise RuntimeError("Install dependencies first: pip install -r requirements.txt") from exc
    if not path.is_file():
        raise FileNotFoundError(f"Local source file not found: {path}")
    log.info("Reading local source: %s", path)
    parquet_file = pq.ParquetFile(path)
    for batch in parquet_file.iter_batches(batch_size=1_024):
        yield from batch.to_pylist()


def stream_local_jsonl(path: Path) -> Iterator[dict[str, Any]]:
    """Read a local JSONL source one record at a time."""
    if not path.is_file():
        raise FileNotFoundError(f"Local source file not found: {path}")
    log.info("Reading local source: %s", path)
    with path.open("r", encoding="utf-8") as source_file:
        for line_number, line in enumerate(source_file, start=1):
            if line.strip():
                try:
                    yield json.loads(line)
                except json.JSONDecodeError as exc:
                    raise RuntimeError(f"Invalid JSON in local NVIDIA source at line {line_number}.") from exc


def local_source_streams(source_dir: Path) -> dict[str, Iterator[dict[str, Any]]]:
    """Return streams from files made by download_blend_sources.py."""
    return {
        "maintenance": stream_local_parquet(source_dir / "maintenance" / "maintenance.parquet"),
        "nemotron": stream_local_jsonl(source_dir / "nemotron" / "SFT" / "code" / "code_v1.1.jsonl"),
        "openorca": stream_local_parquet(source_dir / "openorca" / "1M-GPT4-Augmented.parquet"),
    }


def maintenance_to_text(row: dict[str, Any]) -> str:
    """Convert a work order into an instruction/completion training example."""
    description = _clean(row.get("WorkOrderDescription") or row.get("description"))
    operation = _clean(row.get("OperationDescription") or row.get("operation"))
    equipment = _clean(row.get("Equipment_ID") or row.get("equipment_id"))
    order_type = _clean(row.get("OrderType") or row.get("order_type"))
    if not description or not operation:
        return ""
    context = " ".join(part for part in (
        f"Equipment: {equipment}." if equipment else "",
        f"Order type: {order_type}." if order_type else "",
    ) if part)
    return (
        f"<|user|> Create a safe maintenance work instruction. {context} "
        f"Issue: {description} <|assistant|> {operation}"
    )


def openorca_to_text(row: dict[str, Any]) -> str:
    system = _clean(row.get("system_prompt") or "You are a helpful assistant.")
    question = _clean(row.get("question") or row.get("instruction"))
    response = _clean(row.get("response") or row.get("output"))
    if not question or not response:
        return ""
    return f"<|system|> {system} <|user|> {question} <|assistant|> {response}"


def interleave_weighted(streams: dict[str, Iterator[dict[str, Any]]], counts: dict[str, int]):
    """Yield source rows in their requested proportions without buffering.

    At each position, choose the source with the lowest emitted fraction.  In
    contrast to concatenating sources (or yielding one row from each), this
    keeps all three distributions present throughout corpus construction.
    """
    remaining = counts.copy()
    while any(remaining.values()):
        candidates = [name for name, value in remaining.items() if value > 0]
        # Largest remaining fraction == smallest emitted fraction.  The tuple
        # gives deterministic tie-breaking in the documented source order.
        name = max(candidates, key=lambda source: (remaining[source] / counts[source], -("maintenance", "nemotron", "openorca").index(source)))
        try:
            yield name, next(streams[name])
        except StopIteration:
            log.warning("%s stream ended after fewer rows than requested", name)
            remaining[name] = 0
        else:
            remaining[name] -= 1
        if not any(remaining.values()):
            return


def _row_counts(total_rows: int, weights: dict[str, float]) -> dict[str, int]:
    if total_rows <= 0:
        raise ValueError("total_rows must be positive; use a bounded corpus for repeatable training.")
    if set(weights) != set(DEFAULT_WEIGHTS) or any(value <= 0 for value in weights.values()):
        raise ValueError("all three source weights must be positive")
    scale = sum(weights.values())
    counts = {name: int(total_rows * weight / scale) for name, weight in weights.items()}
    counts["maintenance"] += total_rows - sum(counts.values())
    return counts


def _is_validation(text: str, validation_ratio: float) -> bool:
    # Stable by content: reruns do not leak a document across train and val.
    digest = int(hashlib.sha256(text.encode("utf-8")).hexdigest()[:8], 16)
    return digest % 10_000 < round(validation_ratio * 10_000)


def build_corpus(
    total_rows: int,
    weights: dict[str, float],
    output_dir: Path = ROOT / "data" / "processed",
    validation_ratio: float = 0.02,
    source_dir: Path | None = None,
) -> dict[str, Any]:
    """Stream the selected rows and write train/validation files and provenance."""
    if not 0 <= validation_ratio < 1:
        raise ValueError("validation_ratio must be in [0, 1).")
    try:
        from datasets import load_dataset
    except ImportError as exc:
        raise RuntimeError("Install dependencies first: pip install -r requirements.txt") from exc

    counts = _row_counts(total_rows, weights)
    output_dir.mkdir(parents=True, exist_ok=True)
    train_path, val_path = output_dir / "blended_train.txt", output_dir / "blended_val.txt"
    train_part, val_part = output_dir / "blended_train.txt.part", output_dir / "blended_val.txt.part"
    metadata_path = ROOT / "data" / "metadata" / "blended_corpus.json"
    metadata_path.parent.mkdir(parents=True, exist_ok=True)

    # Explicit split/config selection is important: NVIDIA's repository has
    # SFT and RL configurations. We use its SFT code rows as the reasoning
    # sample. Unlike datasets' fsspec reader, stream_jsonl does not get stuck
    # retrying Xet HTTP range requests on this huge source file.
    if source_dir is not None:
        streams = local_source_streams(source_dir)
    else:
        streams = {
            "maintenance": iter(load_dataset("Jvachier/industrial-maintenance-synthetic", split="train", streaming=True)),
            "nemotron": stream_jsonl(NEMOTRON_CODE_URL),
            "openorca": iter(load_dataset("Open-Orca/OpenOrca", split="train", streaming=True)),
        }
    formatters = {"maintenance": maintenance_to_text, "nemotron": nemotron_to_text, "openorca": openorca_to_text}
    written = {name: {"train": 0, "val": 0, "skipped": 0} for name in counts}

    try:
        with train_part.open("w", encoding="utf-8") as train_file, val_part.open("w", encoding="utf-8") as val_file:
            for index, (name, row) in enumerate(interleave_weighted(streams, counts), start=1):
                text = formatters[name](row)
                if not text:
                    written[name]["skipped"] += 1
                    continue
                destination = val_file if _is_validation(text, validation_ratio) else train_file
                destination.write(text + "\n")
                written[name]["val" if destination is val_file else "train"] += 1
                if index % 1_000 == 0:
                    log.info("Prepared %d/%d source rows", index, total_rows)
        # Windows refuses os.replace when an editor, Explorer preview, or an
        # earlier training process still has the old output open. The newly
        # written .part file is complete at this point, so preserve it and
        # record its usable path rather than turning a successful 300k-row
        # build into an apparent data failure.
        published_paths = {"train": train_part, "val": val_part}
        for label, part_path, final_path in (
            ("train", train_part, train_path),
            ("val", val_part, val_path),
        ):
            try:
                part_path.replace(final_path)
            except PermissionError:
                log.warning(
                    "Cannot replace locked %s. The completed corpus remains at %s. "
                    "Close the program using %s, then rename it when convenient.",
                    final_path.name, part_path.name, final_path.name,
                )
            else:
                published_paths[label] = final_path
    except BaseException:
        # Retain incomplete data only as clearly-labelled .part files; the
        # next invocation safely overwrites them instead of training on a
        # silently truncated corpus.
        log.error("Corpus preparation stopped; final .txt files were not replaced.")
        raise

    stats = {
        "sources": {
            "maintenance": "Jvachier/industrial-maintenance-synthetic (Apache-2.0)",
            "nemotron": "nvidia/Llama-Nemotron-Post-Training-Dataset/SFT/code (see dataset-card terms)",
            "openorca": "Open-Orca/OpenOrca (see dataset card)",
        },
        "requested_rows": counts,
        "written_rows": written,
        "weights": weights,
        "validation_ratio": validation_ratio,
        "train_file": str(published_paths["train"]),
        "val_file": str(published_paths["val"]),
        "streaming": True,
        "source_mode": "local" if source_dir else "remote",
    }
    metadata_path.write_text(json.dumps(stats, indent=2), encoding="utf-8")
    return stats


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build a streamed three-dataset training blend")
    parser.add_argument("--total-rows", type=int, default=300_000,
                        help="Total source rows to materialise (default: 300000).")
    parser.add_argument("--maintenance-weight", type=float, default=DEFAULT_WEIGHTS["maintenance"])
    parser.add_argument("--nemotron-weight", type=float, default=DEFAULT_WEIGHTS["nemotron"])
    parser.add_argument("--openorca-weight", type=float, default=DEFAULT_WEIGHTS["openorca"])
    parser.add_argument("--validation-ratio", type=float, default=0.02)
    parser.add_argument("--source-dir", type=Path, default=None,
                        help="Directory created by download_blend_sources.py; reads sources offline.")
    return parser.parse_args()


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    args = parse_args()
    stats = build_corpus(
        args.total_rows,
        {"maintenance": args.maintenance_weight, "nemotron": args.nemotron_weight, "openorca": args.openorca_weight},
        validation_ratio=args.validation_ratio,
        source_dir=args.source_dir,
    )
    print(json.dumps(stats, indent=2))


if __name__ == "__main__":
    main()
