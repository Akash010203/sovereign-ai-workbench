"""
models.custom_minilm — SovereignAI's from-scratch decoder-only Transformer.

Public API (Phase 5 complete):
    MiniLLM       — full assembled model
    MiniLLMConfig — configuration dataclass
"""
from models.custom_minilm.config import MiniLLMConfig
from models.custom_minilm.model import MiniLLM

__all__ = ["MiniLLM", "MiniLLMConfig"]
