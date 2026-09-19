"""
scripts/create_recovery_snapshot.py — Capture system state before modifications.

Creates a JSON manifest with hashes, sizes, and metadata for all critical
artifacts so future changes can be detected and compared.

Usage:
    python scripts/create_recovery_snapshot.py
    python scripts/create_recovery_snapshot.py --output snapshots/pre_change.json
"""
from __future__ import annotations

import hashlib
import json
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def sha256_file(path: Path, max_bytes: int = 50 * 1024 * 1024) -> str:
    """SHA256 of the first max_bytes of a file."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        remaining = max_bytes
        while remaining > 0:
            chunk = f.read(min(8192, remaining))
            if not chunk:
                break
            h.update(chunk)
            remaining -= len(chunk)
    return h.hexdigest()


def git_state() -> dict:
    """Capture current git status."""
    result = {}
    try:
        result["branch"] = subprocess.check_output(
            ["git", "branch", "--show-current"], cwd=str(ROOT), text=True
        ).strip()
        result["commit"] = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=str(ROOT), text=True
        ).strip()
        result["dirty_files"] = subprocess.check_output(
            ["git", "status", "--porcelain"], cwd=str(ROOT), text=True
        ).strip().split("\n")
        result["stash_count"] = len(subprocess.check_output(
            ["git", "stash", "list"], cwd=str(ROOT), text=True
        ).strip().split("\n"))
    except Exception as exc:
        result["error"] = str(exc)
    return result


def inventory_checkpoints() -> list[dict]:
    """List all checkpoint files with sizes and partial hashes."""
    ckpts = []
    ckpt_dir = ROOT / "checkpoints"
    if not ckpt_dir.exists():
        return ckpts
    for path in sorted(ckpt_dir.rglob("*.pt")):
        ckpts.append({
            "path": str(path.relative_to(ROOT)),
            "size_bytes": path.stat().st_size,
            "size_mb": round(path.stat().st_size / (1024 * 1024), 1),
            "modified": time.strftime("%Y-%m-%dT%H:%M:%S", time.localtime(path.stat().st_mtime)),
            "sha256_first_50mb": sha256_file(path),
        })
    return ckpts


def inventory_indexes() -> list[dict]:
    """List all RAG index files."""
    indexes = []
    for name in ["rag_index.json", "rag_index_tfidf.json", "user_knowledge_index.json"]:
        path = ROOT / "data" / name
        if path.exists():
            data = json.loads(path.read_text(encoding="utf-8"))
            records = data.get("records", [])
            dims = set(len(r.get("embedding", [])) for r in records[:10])
            indexes.append({
                "path": str(path.relative_to(ROOT)),
                "size_bytes": path.stat().st_size,
                "record_count": len(records),
                "embedding_dimensions": list(dims),
                "sha256_first_50mb": sha256_file(path),
            })
    return indexes


def inventory_tokenizers() -> list[dict]:
    """List all tokenizer vocabulary files."""
    vocabs = []
    vocab_dir = ROOT / "tokenizer" / "vocab"
    if not vocab_dir.exists():
        return vocabs
    for path in sorted(vocab_dir.glob("*.json")):
        if path.name.startswith("_audit"):
            continue
        data = json.loads(path.read_text(encoding="utf-8"))
        vocab_size = len(data) if isinstance(data, dict) else "unknown"
        vocabs.append({
            "path": str(path.relative_to(ROOT)),
            "size_bytes": path.stat().st_size,
            "vocab_size": vocab_size,
            "sha256": sha256_file(path),
        })
    return vocabs


def inventory_training_data() -> list[dict]:
    """List all training data files."""
    files = []
    processed = ROOT / "data" / "processed"
    if not processed.exists():
        return files
    for path in sorted(processed.iterdir()):
        if path.is_file():
            files.append({
                "path": str(path.relative_to(ROOT)),
                "size_bytes": path.stat().st_size,
                "size_mb": round(path.stat().st_size / (1024 * 1024), 1),
                "sha256_first_50mb": sha256_file(path),
            })
    return files


def inventory_fineweb() -> dict:
    """Check FineWeb knowledge store status."""
    fw_dir = ROOT / "data" / "knowledge" / "fineweb_edu_1p6b_embed"
    if not fw_dir.exists():
        return {"available": False}

    manifest_path = fw_dir / "fineweb_edu_manifest.json"
    if not manifest_path.exists():
        return {"available": False, "directory_exists": True}

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    chunk_files = sorted(fw_dir.glob("chunks-*.jsonl"))
    embed_files = sorted(fw_dir.glob("embeddings-*.f32"))

    return {
        "available": True,
        "manifest": manifest,
        "chunk_shard_count": len(chunk_files),
        "embedding_shard_count": len(embed_files),
        "total_size_gb": round(
            sum(f.stat().st_size for f in chunk_files + embed_files) / (1024**3), 2
        ),
    }


def create_snapshot(output_path: Path | None = None) -> dict:
    """Create a complete system state snapshot."""
    snapshot = {
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "python_version": sys.version,
        "git": git_state(),
        "checkpoints": inventory_checkpoints(),
        "indexes": inventory_indexes(),
        "tokenizers": inventory_tokenizers(),
        "training_data": inventory_training_data(),
        "fineweb": inventory_fineweb(),
    }

    # Summary statistics
    snapshot["summary"] = {
        "checkpoint_count": len(snapshot["checkpoints"]),
        "checkpoint_total_gb": round(
            sum(c["size_bytes"] for c in snapshot["checkpoints"]) / (1024**3), 2
        ),
        "index_count": len(snapshot["indexes"]),
        "tokenizer_count": len(snapshot["tokenizers"]),
        "training_data_count": len(snapshot["training_data"]),
        "training_data_total_gb": round(
            sum(f["size_bytes"] for f in snapshot["training_data"]) / (1024**3), 2
        ),
        "fineweb_available": snapshot["fineweb"].get("available", False),
    }

    if output_path is None:
        snapshots_dir = ROOT / "docs"
        snapshots_dir.mkdir(parents=True, exist_ok=True)
        output_path = snapshots_dir / "SYSTEM_MANIFEST.json"

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(snapshot, indent=2, default=str), encoding="utf-8"
    )
    print(f"Recovery snapshot saved: {output_path}")
    print(f"  Checkpoints: {snapshot['summary']['checkpoint_count']} "
          f"({snapshot['summary']['checkpoint_total_gb']} GB)")
    print(f"  Indexes: {snapshot['summary']['index_count']}")
    print(f"  Tokenizers: {snapshot['summary']['tokenizer_count']}")
    print(f"  Training files: {snapshot['summary']['training_data_count']} "
          f"({snapshot['summary']['training_data_total_gb']} GB)")
    print(f"  FineWeb: {'available' if snapshot['summary']['fineweb_available'] else 'not found'}")
    return snapshot


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Create a recovery snapshot")
    parser.add_argument("--output", type=str, default=None, help="Output JSON path")
    args = parser.parse_args()
    output = Path(args.output) if args.output else None
    create_snapshot(output)
