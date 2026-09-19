"""
models/registry.py — Central model registry for SovereignAI.

The registry is the single place that:
  1. Knows about every available model adapter.
  2. Can look up a model by name or type.
  3. Reports which models are currently available.
  4. Is used by the router (Phase 9) to select models for tasks.
"""
from __future__ import annotations

import logging
from typing import Optional

from models.adapters.base import ModelProvider, GenerationConfig

log = logging.getLogger(__name__)


class ModelRegistry:
    """
    Central registry of all local model adapters.

    Usage::

        registry = ModelRegistry()
        registry.register(CustomMiniLLMAdapter(...))
        model = registry.get("custom_minilm_v1")
        text  = model.generate("The pump inspection")
    """

    def __init__(self) -> None:
        self._models: dict[str, ModelProvider] = {}

    def register(self, adapter: ModelProvider) -> None:
        """Register a model adapter under its name."""
        self._models[adapter.name] = adapter
        log.info("Registered model: %s  (type=%s)", adapter.name, adapter.model_type)

    def get(self, name: str) -> Optional[ModelProvider]:
        """Return the adapter with the given name, or None if not found."""
        return self._models.get(name)

    def get_by_type(self, model_type: str) -> list[ModelProvider]:
        """Return all adapters of a given type."""
        return [m for m in self._models.values() if m.model_type == model_type]

    def available(self) -> list[ModelProvider]:
        """Return all adapters that are currently available."""
        return [m for m in self._models.values() if m.is_available()]

    def all_info(self) -> list[dict]:
        """Return model_info() for every registered model."""
        return [m.model_info() for m in self._models.values()]

    def __repr__(self) -> str:
        names = list(self._models.keys())
        return f"ModelRegistry({names})"


# ── Default registry factory ───────────────────────────────────────────────────

def build_default_registry(
    minilm_checkpoint: str = "",
    tokenizer_path: str = "tokenizer/vocab/demo_bpe_vocab.json",
) -> ModelRegistry:
    """
    Build the default platform registry.

    Args:
        minilm_checkpoint: Path to the trained MiniLLM checkpoint.
                           If empty, the adapter is registered but marked
                           unavailable.
        tokenizer_path:    Path to the tokenizer vocab JSON.
    Returns:
        A registry containing the project's from-scratch MiniLLM adapter.
    """
    from models.adapters.minilm_adapter import CustomMiniLLMAdapter

    registry = ModelRegistry()

    # The only generative model is the project's own MiniLLM.  It is loaded
    # directly from a locally trained checkpoint and never calls a model API.
    registry.register(
        CustomMiniLLMAdapter(
            checkpoint_path=minilm_checkpoint,
            tokenizer_path=tokenizer_path,
        )
    )

    return registry
