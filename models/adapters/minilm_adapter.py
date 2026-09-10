"""
models/adapters/minilm_adapter.py — Adapter for the custom MiniLLM.

This wraps the from-scratch MiniLLM (Phase 5-7) in the ModelProvider
interface so the router and agent can call it the same way they call
any other model.
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Optional

import torch

from models.adapters.base import ModelProvider, GenerationConfig
from models.custom_minilm.checkpoint import load_model_from_checkpoint
from models.custom_minilm.generate import generate as _generate
from tokenizer.tokenizer import ByteLevelBPETokenizer

log = logging.getLogger(__name__)


class CustomMiniLLMAdapter(ModelProvider):
    """
    Adapter wrapping the custom from-scratch MiniLLM.

    This is honestly labelled: it is a tiny educational model
    (~5M parameters, trained on a small domain corpus).
    It demonstrates that every Transformer component was built
    in-house — it is NOT a production LLM.
    """

    def __init__(
        self,
        checkpoint_path: str | Path,
        tokenizer_path: str | Path,
        device: str = "auto",
    ) -> None:
        self._ckpt_path = Path(checkpoint_path)
        self._tok_path  = Path(tokenizer_path)
        self._device    = (
            torch.device("cuda" if torch.cuda.is_available() else "cpu")
            if device == "auto" else torch.device(device)
        )
        self._model     = None
        self._tokenizer = None
        self._meta: dict = {}
        self._loaded    = False

    # ── Lazy loading ───────────────────────────────────────────────────

    def _ensure_loaded(self) -> None:
        if self._loaded:
            return
        if not self._ckpt_path.exists():
            log.warning("MiniLLM checkpoint not found: %s", self._ckpt_path)
            return
        self._model, self._meta = load_model_from_checkpoint(
            self._ckpt_path, device=self._device
        )
        self._tokenizer = ByteLevelBPETokenizer.load(self._tok_path)
        self._loaded = True
        log.info("CustomMiniLLMAdapter loaded — params: %s",
                 f"{self._meta.get('model_params', 0):,}")

    # ── ModelProvider interface ────────────────────────────────────────

    @property
    def name(self) -> str:
        return "custom_minilm_v1"

    @property
    def model_type(self) -> str:
        return "custom_minilm"

    def is_available(self) -> bool:
        self._ensure_loaded()
        return self._loaded and self._model is not None

    def generate(self, prompt: str, config: Optional[GenerationConfig] = None) -> str:
        self._ensure_loaded()
        if not self._loaded:
            return "[MiniLLM not available — checkpoint not found]"

        cfg = config or GenerationConfig()
        eos_id = self._tokenizer.vocab.special_to_id.get("<eos>")
        prompt_ids = self._tokenizer.encode(prompt, add_bos=True)

        new_ids = _generate(
            self._model,
            prompt_ids=prompt_ids,
            max_new_tokens=cfg.max_new_tokens,
            temperature=cfg.temperature,
            top_k=cfg.top_k,
            top_p=cfg.top_p,
            eos_id=eos_id,
            seed=cfg.seed,
            device=self._device,
        )
        return self._tokenizer.decode(new_ids)

    def model_info(self) -> dict[str, Any]:
        self._ensure_loaded()
        return {
            "name":          self.name,
            "model_type":    self.model_type,
            "is_available":  self._loaded,
            "device":        str(self._device),
            "parameters":    self._meta.get("model_params", 0),
            "checkpoint":    str(self._ckpt_path),
            "train_step":    self._meta.get("step", 0),
            "description":   "Custom Educational MiniLLM — ~5M params, trained from scratch",
            "honest_label":  "Tiny domain-specific model; NOT a production LLM",
        }
