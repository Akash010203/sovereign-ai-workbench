"""Regression coverage for the industrial model's chat prompt format."""
from __future__ import annotations


class _Tokenizer:
    vocab = type("Vocab", (), {"special_to_id": {"<|user|>": 4, "<|assistant|>": 5}})()


def test_industrial_adapter_uses_trained_role_markers():
    from models.adapters.minilm_adapter import CustomMiniLLMAdapter

    adapter = CustomMiniLLMAdapter("missing.pt")
    adapter._tokenizer = _Tokenizer()
    prompt = adapter._build_instruction_prompt("hi")
    assert prompt.endswith("<|user|> hi <|assistant|>")
    assert "industrial maintenance assistant" in prompt


def test_legacy_adapter_keeps_legacy_instruction_format():
    from models.adapters.minilm_adapter import CustomMiniLLMAdapter

    adapter = CustomMiniLLMAdapter("missing.pt")
    assert adapter._build_instruction_prompt("hi") == "Q: hi\nA:"


def test_chat_scope_allows_general_questions_to_reach_generation():
    from app.backend.app import _chat_response_kind

    assert _chat_response_kind("ihi") == "greeting"
    assert _chat_response_kind("what?") == "question"
    assert _chat_response_kind("Pump P-101 has high vibration") == "question"
    assert _chat_response_kind("Tell me a joke") == "question"
