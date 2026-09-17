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
from models.custom_minilm.checkpoint import load_model_from_checkpoint, resolve_tokenizer_path
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
        model_vocab_size = getattr(self._model.config, "vocab_size", None)

        # Resolve tokenizer path robustly across drives / relocations
        raw_tok_path = (
            self._meta.get("tokenizer_resolved_path")
            or self._meta.get("tokenizer_path")
            or self._tok_path_hint
        )
        tok_path = None
        try:
            tok_path = resolve_tokenizer_path(
                raw_tok_path,
                vocab_size=model_vocab_size,
                root=self._ckpt_path.parent.parent,
            )
        except Exception as err:
            log.warning("resolve_tokenizer_path failed for %s: %s", raw_tok_path, err)

        if tok_path is None or not tok_path.exists():
            log.error("No tokenizer found for checkpoint %s", self._ckpt_path)
            return

        self._tokenizer = ByteLevelBPETokenizer.load(tok_path)

        # Ensure vocab size matches model embedding dimensions
        if model_vocab_size and self._tokenizer.vocab_size != model_vocab_size:
            log.warning(
                "Loaded tokenizer vocab (%d) does not match model vocab (%d); resolving matching vocab...",
                self._tokenizer.vocab_size, model_vocab_size,
            )
            try:
                matching_path = resolve_tokenizer_path(
                    None, vocab_size=model_vocab_size, root=self._ckpt_path.parent.parent
                )
                if matching_path.exists():
                    self._tokenizer = ByteLevelBPETokenizer.load(matching_path)
                    tok_path = matching_path
                    log.info("Switched to matching tokenizer: %s (vocab %d)", tok_path.name, self._tokenizer.vocab_size)
            except Exception as e:
                log.warning("Could not switch to matching tokenizer: %s", e)

        self._loaded = True
        log.info(
            "CustomMiniLLMAdapter loaded — params: %s  tokenizer: %s",
            f"{self._meta.get('model_params', 0):,}",
            tok_path.name,
        )

    # ── Instruction prompt builder ─────────────────────────────────────

    def _uses_chat_format(self) -> bool:
        """Whether this checkpoint's tokenizer supports the trained role tags."""
        return self._tokenizer is not None and {
            "<|user|>", "<|assistant|>"
        }.issubset(self._tokenizer.vocab.special_to_id)

    def _build_instruction_prompt(self, user_text: str) -> str:
        """
        Wrap the user query in a minimal Q/A prompt.

        If the text already contains RAG context (has 'Context:' block)
        or is already formatted, pass it through as-is to avoid
        double-wrapping which confuses the model.
        """
        stripped = user_text.strip()
        if self._uses_chat_format():
            # The industrial 80M model was trained on exactly this format.
            # Using the legacy Q:/A: template makes it treat ordinary chat as
            # an explanation/completion task instead of an assistant reply.
            if "<|assistant|>" in stripped:
                return stripped if stripped.rstrip().endswith("<|assistant|>") else stripped.rstrip() + " <|assistant|>"
            return f"<|user|> {stripped} <|assistant|>"

        # Legacy checkpoints were trained on Q:/A: text.
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
            # The legacy checkpoint over-predicts EOS; the industrial model
            # was trained with explicit role/EOS boundaries and should stop
            # normally instead of filling the entire token budget.
            eos_id=eos_id if self._uses_chat_format() else None,
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
            "description":   "Custom MiniLLM trained locally from scratch",
            "honest_label":  "Domain-specific local model; outputs require human review",
        }
