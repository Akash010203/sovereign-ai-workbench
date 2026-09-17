"""
models/custom_minilm/checkpoint.py — Save, load, resume, and export.

This wraps the raw torch.save/load calls from the Trainer with a clean
standalone API that can be used by inference scripts, the backend, and
Phase 8's model adapter — without importing the whole Trainer class.
"""
from __future__ import annotations

import logging
from dataclasses import asdict
from pathlib import Path
from typing import Any

import torch

from models.custom_minilm.config import MiniLLMConfig
from models.custom_minilm.model import MiniLLM

log = logging.getLogger(__name__)


PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent


def resolve_tokenizer_path(
    tok_path: str | Path | None = None,
    vocab_size: int | None = None,
    root: Path | None = None,
) -> Path:
    """
    Resolve a tokenizer vocabulary path robustly across machines, drive changes,
    and directory moves.

    Args:
        tok_path: Path string or Path object stored in checkpoint or passed as argument.
        vocab_size: Optional vocabulary size of the model (e.g. 8000 or 600) to help
                    select the correct vocab file.
        root: Workspace root directory (defaults to PROJECT_ROOT).

    Returns:
        A valid Path to an existing tokenizer JSON vocabulary file.
    """
    workspace_root = Path(root) if root else PROJECT_ROOT
    vocab_dir = workspace_root / "tokenizer" / "vocab"

    candidates: list[Path] = []

    if tok_path:
        tok_p = Path(tok_path)
        # 1. Direct path exists
        if tok_p.exists() and tok_p.is_file():
            candidates.append(tok_p.resolve())

        # 2. Relative to workspace root
        rel_p = workspace_root / tok_p
        if rel_p.exists() and rel_p.is_file():
            candidates.append(rel_p.resolve())

        # 3. Filename inside tokenizer/vocab/ (handles old drives like D:\...\bpe_8k.json)
        name_p = vocab_dir / tok_p.name
        if name_p.exists() and name_p.is_file():
            candidates.append(name_p.resolve())

        # 4. If path string mentions "tokenizer" or "vocab"
        posix_str = tok_p.as_posix()
        if "tokenizer/vocab/" in posix_str:
            sub = posix_str.split("tokenizer/vocab/")[-1]
            sub_p = vocab_dir / sub
            if sub_p.exists() and sub_p.is_file():
                candidates.append(sub_p.resolve())

    # 5. Check based on model vocab_size
    if vocab_size is not None:
        if vocab_size >= 4000:
            bpe_8k = vocab_dir / "bpe_8k.json"
            if bpe_8k.exists():
                candidates.append(bpe_8k.resolve())
        elif vocab_size <= 1000:
            demo_vocab = vocab_dir / "demo_bpe_vocab.json"
            if demo_vocab.exists():
                candidates.append(demo_vocab.resolve())

    # 6. General fallbacks
    for fallback in [vocab_dir / "bpe_8k.json", vocab_dir / "demo_bpe_vocab.json"]:
        if fallback.exists():
            candidates.append(fallback.resolve())

    # Return first existing candidate
    for c in candidates:
        if c.exists() and c.is_file():
            return c

    # Last resort: any json in vocab_dir
    if vocab_dir.exists():
        jsons = sorted(vocab_dir.glob("*.json"))
        if jsons:
            return jsons[0].resolve()

    raise FileNotFoundError(
        f"Could not resolve tokenizer vocabulary path for '{tok_path}' (vocab_size={vocab_size}) in {workspace_root}"
    )


def load_model_from_checkpoint(
    path: str | Path,
    device: str | torch.device = "cpu",
    strict: bool = True,
) -> tuple[MiniLLM, dict]:
    """
    Load a MiniLLM from a checkpoint produced by the Trainer.

    Args:
        path:   Path to the ``.pt`` checkpoint file.
        device: Where to place the model tensors.
        strict: Passed to ``model.load_state_dict()``.

    Returns:
        Tuple of (model, checkpoint_metadata_dict).
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {path}")

    device = torch.device(device) if isinstance(device, str) else device
    # weights_only=False: we load optimizer state + config dicts, not just weights.
    # This is a trusted local checkpoint, not an untrusted download.
    ckpt: dict[str, Any] = torch.load(path, map_location=device, weights_only=False)

    # Reconstruct the config from the stored dict
    cfg_dict = ckpt.get("model_config", {})
    model_cfg = MiniLLMConfig(**cfg_dict)

    model = MiniLLM(model_cfg)
    # Support both Trainer checkpoints ("model_state") and fine-tune checkpoints ("model_state_dict")
    state_key = "model_state" if "model_state" in ckpt else "model_state_dict"
    model.load_state_dict(ckpt[state_key], strict=strict)
    model = model.to(device).eval()

    raw_tok = ckpt.get("tokenizer_path")
    resolved_tok = resolve_tokenizer_path(
        raw_tok, vocab_size=model_cfg.vocab_size, root=PROJECT_ROOT
    )
    try:
        rel_tok_str = resolved_tok.relative_to(PROJECT_ROOT).as_posix()
    except ValueError:
        rel_tok_str = resolved_tok.as_posix()

    meta = {
        "step":           ckpt.get("step", 0),
        "experiment":     ckpt.get("experiment", "unknown"),
        "train_loss":     ckpt.get("train_loss"),
        "val_loss":       ckpt.get("val_loss"),
        "tokenizer_path": rel_tok_str,
        "tokenizer_resolved_path": str(resolved_tok),
        "timestamp":      ckpt.get("timestamp"),
        "model_params":   model.count_parameters(),
    }
    log.info(
        "Loaded checkpoint '%s'  step=%d  params=%s",
        path.name, meta["step"], f"{meta['model_params']:,}"
    )
    return model, meta


def save_model_for_inference(
    model: MiniLLM,
    path: str | Path,
    tokenizer_path: str = "",
    notes: str = "",
) -> None:
    """
    Export a model in a minimal format suitable for inference only
    (no optimizer / scheduler state — smaller file).
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if tokenizer_path:
        try:
            tok_p = Path(tokenizer_path)
            if tok_p.is_absolute() and tok_p.is_relative_to(PROJECT_ROOT):
                tokenizer_path = tok_p.relative_to(PROJECT_ROOT).as_posix()
            else:
                tokenizer_path = tok_p.as_posix()
        except Exception:
            pass
    payload = {
        "model_state":    model.state_dict(),
        "model_config":   asdict(model.config),
        "tokenizer_path": tokenizer_path,
        "notes":          notes,
    }
    torch.save(payload, path)
    log.info("Inference checkpoint saved → %s", path)

