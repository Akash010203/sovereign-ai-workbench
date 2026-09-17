"""Regression coverage for the industrial model's chat prompt format."""
from __future__ import annotations


class _Tokenizer:
    vocab = type("Vocab", (), {"special_to_id": {"<|user|>": 4, "<|assistant|>": 5}})()


def test_industrial_adapter_uses_trained_role_markers():
    from models.adapters.minilm_adapter import CustomMiniLLMAdapter

    adapter = CustomMiniLLMAdapter("missing.pt")
    adapter._tokenizer = _Tokenizer()
    assert adapter._build_instruction_prompt("hi") == "<|user|> hi <|assistant|>"


def test_legacy_adapter_keeps_legacy_instruction_format():
    from models.adapters.minilm_adapter import CustomMiniLLMAdapter

    adapter = CustomMiniLLMAdapter("missing.pt")
    assert adapter._build_instruction_prompt("hi") == "Q: hi\nA:"
