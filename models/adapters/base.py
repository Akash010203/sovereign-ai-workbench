"""
models/adapters/base.py — Abstract model interface for SovereignAI.

WHY AN ABSTRACTION LAYER?
--------------------------
The SovereignAI platform supports multiple local models:
  - The custom from-scratch MiniLLM (Phase 5-7)
  - A vision/multimodal model (Phase 14)

Without an abstraction layer, every component (router, agent, backend)
would need to know the internal API of each model.  Adding a new model
would require touching every component.

With an abstraction layer, every consumer speaks one interface
(``ModelProvider``) and model-specific details stay inside each adapter.

INTERFACE CONTRACT
------------------
Every model adapter must implement:
  - generate(prompt, **kwargs) → str
  - is_available() → bool
  - model_info() → dict (name, type, params, device, etc.)

Optional:
  - encode_prompt(text) → model-specific input
  - embed(text) → list[float] (for RAG / Phase 12)
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Optional


@dataclass
class GenerationConfig:
    """Standard generation parameters shared across all model adapters.

    Defaults are tuned for the industrial 80M domain model — conservative
    decoding to reduce hallucination from the mixed training data.
    """
    max_new_tokens: int = 80
    temperature: float = 0.6
    top_k: int = 30
    top_p: float = 0.90
    seed: Optional[int] = None
    stop_sequences: list[str] = field(default_factory=list)


class ModelProvider(ABC):
    """
    Abstract base class for all local model adapters.

    Every model — the custom MiniLLM, open-weight models loaded via
    any future in-process model backend — must implement this
    interface so the router and agent can use them interchangeably.
    """

    @property
    @abstractmethod
    def name(self) -> str:
        """Human-readable model identifier, e.g. 'custom_minilm_v1'."""

    @property
    @abstractmethod
    def model_type(self) -> str:
        """Category: 'custom_minilm' | 'open_weight' | 'vision'."""

    @abstractmethod
    def is_available(self) -> bool:
        """Return True if the model is loaded and ready to generate."""

    @abstractmethod
    def generate(self, prompt: str, config: Optional[GenerationConfig] = None) -> str:
        """
        Generate a text completion for ``prompt``.

        Args:
            prompt: The full input text (system + context + user query).
            config: Sampling parameters.  If None, use model defaults.

        Returns:
            The generated text string (NOT including the prompt).
        """

    @abstractmethod
    def model_info(self) -> dict[str, Any]:
        """
        Return a dict describing this model for the UI and audit log.

        Minimum keys: name, model_type, is_available, device.
        """

    def embed(self, text: str) -> list[float]:
        """
        Embed ``text`` as a vector (for RAG).  Optional — not all
        model adapters need to implement this.  Raises NotImplementedError
        by default.
        """
        raise NotImplementedError(
            f"Model '{self.name}' does not support embedding. "
            "Use a dedicated embedding model for RAG."
        )
