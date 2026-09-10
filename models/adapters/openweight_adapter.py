"""
models/adapters/openweight_adapter.py — Adapter for locally-hosted
open-weight models via Ollama.

WHAT THIS IS
------------
For tasks that require genuine reasoning capability (coding, document
analysis, complex Q&A), the platform routes to a locally-hosted
open-weight model.  This adapter supports models served through Ollama
(a local model server that runs entirely on-premises).

SUPPORTED MODELS (examples — install whichever fits your VRAM):
  - phi3:mini         (3.8B params, ~2.2 GB, good for coding+chat)
  - mistral:7b-q4     (7B quantized, ~4 GB, excellent for chat/docs)
  - deepseek-coder:6.7b (6.7B, ~4 GB, strong coding model)
  - llava:7b          (7B multimodal, for vision tasks in Phase 14)

HONESTY NOTE
-----------
These are pre-trained open-weight models, NOT trained by this project.
They are used as the "production capability" layer of the platform.
The custom MiniLLM (Phase 5-7) proves the from-scratch ML work.
This adapter proves the multi-model platform architecture.

INSTALLATION (one-time, offline-capable after first download):
  1. Install Ollama: https://ollama.ai (Windows MSI)
  2. Pull a model: ollama pull phi3:mini
  3. Ollama runs as a local server on http://localhost:11434
  4. No cloud calls are made at runtime.

AIR-GAP NOTE
-------------
Ollama's inference is 100% local.  The adapter sends HTTP requests to
localhost — never to the internet.  This is verified by the network
monitor in Phase 18.
"""
from __future__ import annotations

import logging
from typing import Any, Optional

from models.adapters.base import ModelProvider, GenerationConfig

log = logging.getLogger(__name__)


class OllamaModelAdapter(ModelProvider):
    """
    Adapter for models served locally by Ollama.

    This uses Ollama's local REST API (http://localhost:11434).
    No internet connection is required at runtime.

    Args:
        model_name:  The Ollama model tag, e.g. 'phi3:mini'.
        base_url:    Ollama server URL (default: localhost).
        context_len: Max context window to use.
    """

    OLLAMA_DEFAULT_URL = "http://localhost:11434"

    def __init__(
        self,
        model_name: str = "phi3:mini",
        base_url: str = OLLAMA_DEFAULT_URL,
        context_len: int = 2048,
    ) -> None:
        self._model_name  = model_name
        self._base_url    = base_url.rstrip("/")
        self._context_len = context_len
        self._available: Optional[bool] = None

    # ── ModelProvider interface ────────────────────────────────────────

    @property
    def name(self) -> str:
        return f"ollama/{self._model_name}"

    @property
    def model_type(self) -> str:
        return "open_weight"

    def is_available(self) -> bool:
        """Check if Ollama is running and the model is pulled."""
        try:
            import urllib.request, json as _json
            url = f"{self._base_url}/api/tags"
            with urllib.request.urlopen(url, timeout=2) as resp:
                data = _json.loads(resp.read())
            models = [m["name"] for m in data.get("models", [])]
            tag = self._model_name
            # Match either exact or base name
            self._available = any(
                m == tag or m.split(":")[0] == tag.split(":")[0]
                for m in models
            )
        except Exception as exc:
            log.debug("Ollama availability check failed: %s", exc)
            self._available = False
        return bool(self._available)

    def generate(self, prompt: str, config: Optional[GenerationConfig] = None) -> str:
        """Send prompt to local Ollama and return generated text."""
        import json as _json
        import urllib.request

        cfg = config or GenerationConfig()
        payload = {
            "model":  self._model_name,
            "prompt": prompt,
            "stream": False,
            "options": {
                "temperature": cfg.temperature,
                "top_k":       cfg.top_k,
                "top_p":       cfg.top_p,
                "num_predict": cfg.max_new_tokens,
                "num_ctx":     self._context_len,
            },
        }
        if cfg.seed is not None:
            payload["options"]["seed"] = cfg.seed

        url = f"{self._base_url}/api/generate"
        data = _json.dumps(payload).encode("utf-8")
        req  = urllib.request.Request(
            url, data=data,
            headers={"Content-Type": "application/json"},
        )
        try:
            with urllib.request.urlopen(req, timeout=120) as resp:
                result = _json.loads(resp.read())
            return result.get("response", "")
        except Exception as exc:
            log.error("Ollama generate failed: %s", exc)
            return f"[Ollama error: {exc}]"

    def model_info(self) -> dict[str, Any]:
        return {
            "name":         self.name,
            "model_type":   self.model_type,
            "is_available": self.is_available(),
            "device":       "local (Ollama)",
            "base_url":     self._base_url,
            "description":  f"Open-weight model '{self._model_name}' via local Ollama server",
            "honest_label": "Pre-trained open-weight model — NOT trained by this project",
        }
