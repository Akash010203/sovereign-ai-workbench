"""
models/adapters/minilm_adapter.py — Adapter for the custom MiniLLM.

This wraps the from-scratch MiniLLM (Phase 5-7) in the ModelProvider
interface so the router and agent can call it the same way they call
any other model.
"""
from __future__ import annotations

import logging
import re
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

    This is honestly labelled: it is a small educational model
    (~5M parameters, trained on OpenOrca instruction data).
    It demonstrates that every Transformer component was built
    in-house — it is NOT a production LLM.
    """

    def __init__(
        self,
        checkpoint_path: str | Path,
        tokenizer_path: str | Path = "",
        device: str = "auto",
    ) -> None:
        self._ckpt_path = Path(checkpoint_path)
        # tokenizer_path may be overridden by what's stored in the checkpoint
        self._tok_path_hint = Path(tokenizer_path) if tokenizer_path else None
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
        # Use the tokenizer path embedded in the checkpoint; fall back to hint
        tok_path_in_ckpt = self._meta.get("tokenizer_path", "")
        if tok_path_in_ckpt and Path(tok_path_in_ckpt).exists():
            tok_path = Path(tok_path_in_ckpt)
        elif self._tok_path_hint and self._tok_path_hint.exists():
            tok_path = self._tok_path_hint
        else:
            # last resort: search for any vocab json next to checkpoint
            vocab_dir = self._ckpt_path.parent.parent / "tokenizer" / "vocab"
            candidates = sorted(vocab_dir.glob("*.json")) if vocab_dir.exists() else []
            tok_path = candidates[0] if candidates else None

        if tok_path is None or not tok_path.exists():
            log.error("No tokenizer found for checkpoint %s", self._ckpt_path)
            return

        self._tokenizer = ByteLevelBPETokenizer.load(tok_path)
        self._loaded = True
        log.info(
            "CustomMiniLLMAdapter loaded — params: %s  tokenizer: %s",
            f"{self._meta.get('model_params', 0):,}",
            tok_path.name,
        )

    # ── Instruction prompt builder ─────────────────────────────────────

    @staticmethod
    def _build_instruction_prompt(user_text: str) -> str:
        """
        Wrap the user query in a minimal Q/A prompt.

        If the text already contains RAG context (has 'Context:' block)
        or is already formatted, pass it through as-is to avoid
        double-wrapping which confuses the model.
        """
        stripped = user_text.strip()
        # Already has context / structure — don't re-wrap
        if "Context:" in stripped or stripped.startswith("Q:"):
            # Just ensure it ends with the generation cue
            if not stripped.endswith("A:"):
                stripped = stripped.rstrip() + "\nA:"
            return stripped
        return f"Q: {stripped}\nA:"

    # ── Output cleaner ─────────────────────────────────────────────────

    @staticmethod
    def _clean_output(raw: str, prompt: str = "") -> str:
        """
        Remove artifacts common in small-model outputs.

        Strategy: only strip the *echoed prompt* from the front,
        then clean up obvious repetition loops. Avoid stripping
        tokens like 'A:' globally — they appear legitimately in output.
        """
        # 1. Remove any BOS/EOS special tokens
        raw = raw.replace("<bos>", "").replace("<eos>", "")

        # 2. If the prompt was echoed at the start, remove it
        if prompt:
            # Try to strip the prompt prefix from the beginning of the output
            prompt_stripped = prompt.replace("<bos>", "").strip()
            if raw.startswith(prompt_stripped):
                raw = raw[len(prompt_stripped):]

        # 3. Remove leading prompt artifacts (Q:/A: at very start only)
        raw = re.sub(r"^\s*(Q:|A:|### Response:|### Instruction:)\s*", "", raw)

        # 4. Strip meta-commentary that small models produce instead of answers
        raw = re.sub(
            r"(?i)(the given task is|here is a step.by.step|step \d+:|determine if|find the appropriate)[^\n]*\n?",
            "",
            raw,
        )

        # 5. Collapse runs of the same word repeated 3+ times (hallucination loop)
        raw = re.sub(r"\b(\w+)(\s+\1){3,}", r"\1 \1", raw)

        # 6. Collapse sequences of the same punctuation (e.g. ".....", "-----")
        raw = re.sub(r"([^\w\s]){5,}", r"\1\1", raw)

        raw = raw.strip()

        # 7. Hard cap at 800 chars — cut at sentence boundary if possible
        if len(raw) > 800:
            cutpoint = raw.rfind(".", 0, 800)
            if cutpoint > 100:
                raw = raw[:cutpoint + 1]
            else:
                raw = raw[:800].rstrip() + "…"

        # 8. Trim to at most 3 paragraphs
        paragraphs = [p.strip() for p in raw.split("\n\n") if p.strip()]
        if paragraphs:
            raw = "\n\n".join(paragraphs[:3])

        cleaned = raw.strip()
        if not cleaned:
            return "[The model is still learning — try rephrasing your question or add more documents to the knowledge base.]"
        return cleaned

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

        # Wrap with instruction template so the model knows what to do
        instruction_prompt = self._build_instruction_prompt(prompt)
        prompt_ids = self._tokenizer.encode(instruction_prompt, add_bos=True)

        # Truncate prompt to leave room for generation (max_seq_len - max_new_tokens)
        max_ctx = getattr(self._model.config, "max_seq_len", 128)
        headroom = max(16, max_ctx - cfg.max_new_tokens)
        if len(prompt_ids) > headroom:
            prompt_ids = prompt_ids[-headroom:]

        new_ids = _generate(
            self._model,
            prompt_ids=prompt_ids,
            max_new_tokens=cfg.max_new_tokens,
            temperature=cfg.temperature,
            top_k=cfg.top_k,
            top_p=cfg.top_p,
            # Do NOT pass eos_id to the generator: the fine-tuned model
            # over-predicts <eos> at low temperature on the very first step,
            # producing zero output. Instead we let it run and strip EOS
            # tokens from the decoded string in _clean_output.
            eos_id=None,
            seed=cfg.seed,
            device=self._device,
        )
        raw = self._tokenizer.decode(new_ids)
        # Pass the instruction_prompt so _clean_output can strip any echoed prefix
        return self._clean_output(raw, prompt=instruction_prompt)

    def model_info(self) -> dict[str, Any]:
        self._ensure_loaded()
        tok_name = ""
        if self._tokenizer is not None:
            try:
                tok_name = str(len(self._tokenizer.vocab.id_to_token)) + " tokens"
            except Exception:
                pass
        return {
            "name":          self.name,
            "model_type":    self.model_type,
            "is_available":  self._loaded,
            "device":        str(self._device),
            "parameters":    self._meta.get("model_params", 0),
            "checkpoint":    str(self._ckpt_path),
            "train_step":    self._meta.get("step", 0),
            "vocab_size":    tok_name,
            "description":   "Custom MiniLLM — ~5M params, trained from scratch on OpenOrca",
            "honest_label":  "Small domain-specific model; NOT a production LLM",
        }
