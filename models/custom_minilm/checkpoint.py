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
    ckpt: dict[str, Any] = torch.load(path, map_location=device)

    # Reconstruct the config from the stored dict
    cfg_dict = ckpt.get("model_config", {})
    model_cfg = MiniLLMConfig(**cfg_dict)

    model = MiniLLM(model_cfg)
    model.load_state_dict(ckpt["model_state"], strict=strict)
    model = model.to(device).eval()

    meta = {
        "step":           ckpt.get("step", 0),
        "experiment":     ckpt.get("experiment", "unknown"),
        "train_loss":     ckpt.get("train_loss"),
        "val_loss":       ckpt.get("val_loss"),
        "tokenizer_path": ckpt.get("tokenizer_path"),
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
    payload = {
        "model_state":    model.state_dict(),
        "model_config":   asdict(model.config),
        "tokenizer_path": tokenizer_path,
        "notes":          notes,
    }
    torch.save(payload, path)
    log.info("Inference checkpoint saved → %s", path)
