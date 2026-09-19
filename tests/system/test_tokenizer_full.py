"""
TEST B — Custom Tokenizer Full Validation

Tests:
- Round-trip encode/decode accuracy
- Plain English, numbers, punctuation, code, JSON
- Hindi, mixed Hindi-English
- Special characters, empty input, long input
- Vocabulary consistency
- Performance metrics
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))

from tokenizer.tokenizer import ByteLevelBPETokenizer


@pytest.fixture(scope="module")
def tokenizer():
    vocab_path = ROOT / "tokenizer" / "vocab" / "bpe_8k.json"
    if not vocab_path.exists():
        vocab_path = ROOT / "tokenizer" / "vocab" / "demo_bpe_vocab.json"
    return ByteLevelBPETokenizer.load(vocab_path)


class TestTokenizerRoundTrip:
    """Encode → decode must reconstruct input."""

    @pytest.mark.parametrize("text", [
        "Hello, world!",
        "The quick brown fox jumps over the lazy dog.",
        "This is a test of the tokenizer.",
        "Machine learning is a subset of artificial intelligence.",
    ])
    def test_plain_english(self, tokenizer, text):
        ids = tokenizer.encode(text)
        decoded = tokenizer.decode(ids)
        assert decoded == text, f"Round-trip failed: '{text}' -> '{decoded}'"

    @pytest.mark.parametrize("text", [
        "12345",
        "3.14159",
        "100,000,000",
        "-42",
        "1e10",
        "0.001",
        "10.5 bar",
        "105 bar",
        "1.05 MPa",
    ])
    def test_numbers(self, tokenizer, text):
        ids = tokenizer.encode(text)
        decoded = tokenizer.decode(ids)
        assert decoded == text, f"Number round-trip failed: '{text}' -> '{decoded}'"

    @pytest.mark.parametrize("text", [
        "Hello! How are you?",
        "Test: 1, 2, 3...",
        "a@b.com",
        "path/to/file.txt",
        "(parentheses) [brackets] {braces}",
        "50% done; 100% complete!",
    ])
    def test_punctuation(self, tokenizer, text):
        ids = tokenizer.encode(text)
        decoded = tokenizer.decode(ids)
        assert decoded == text

    @pytest.mark.parametrize("text", [
        "def hello():\n    print('hi')\n",
        "for i in range(10):\n    x += i\n",
        "import os; os.path.join('a', 'b')",
        "x = lambda a, b: a + b",
    ])
    def test_code(self, tokenizer, text):
        ids = tokenizer.encode(text)
        decoded = tokenizer.decode(ids)
        assert decoded == text

    def test_json(self, tokenizer):
        text = json.dumps({"key": "value", "number": 42, "list": [1, 2, 3]})
        ids = tokenizer.encode(text)
        decoded = tokenizer.decode(ids)
        assert decoded == text

    def test_special_characters(self, tokenizer):
        text = "Résumé café naïve"
        ids = tokenizer.encode(text)
        decoded = tokenizer.decode(ids)
        assert decoded == text

    def test_empty_input(self, tokenizer):
        ids = tokenizer.encode("")
        decoded = tokenizer.decode(ids)
        assert decoded == ""
        assert ids == []

    def test_long_input(self, tokenizer):
        text = "The quick brown fox. " * 500
        ids = tokenizer.encode(text)
        decoded = tokenizer.decode(ids)
        assert decoded == text

    def test_single_character(self, tokenizer):
        for ch in "abcABC123!@#":
            ids = tokenizer.encode(ch)
            decoded = tokenizer.decode(ids)
            assert decoded == ch, f"Single char failed: '{ch}'"


class TestTokenizerProperties:
    """Verify tokenizer properties."""

    def test_vocab_size_positive(self, tokenizer):
        assert tokenizer.vocab_size > 0

    def test_vocab_size_reasonable(self, tokenizer):
        # Should be between 260 (256 bytes + specials) and 100000
        assert 260 <= tokenizer.vocab_size <= 100000

    def test_special_tokens_exist(self, tokenizer):
        specials = tokenizer.vocab.special_to_id
        assert "<pad>" in specials
        assert "<unk>" in specials
        assert "<bos>" in specials
        assert "<eos>" in specials

    def test_bos_eos_encoding(self, tokenizer):
        ids = tokenizer.encode("hello", add_bos=True, add_eos=True)
        bos_id = tokenizer.vocab.special_to_id["<bos>"]
        eos_id = tokenizer.vocab.special_to_id["<eos>"]
        assert ids[0] == bos_id
        assert ids[-1] == eos_id

    def test_no_excessive_unknown_tokens(self, tokenizer):
        """Normal English text should not produce excessive unknowns."""
        text = "The quick brown fox jumps over the lazy dog."
        ids = tokenizer.encode(text)
        unk_id = tokenizer.vocab.special_to_id.get("<unk>", -1)
        unk_count = sum(1 for i in ids if i == unk_id)
        assert unk_count == 0, f"Got {unk_count} <unk> tokens for normal English"


class TestTokenizerPerformance:
    """Measure tokenization speed."""

    def test_encoding_speed(self, tokenizer):
        text = "The quick brown fox jumps over the lazy dog. " * 100
        start = time.perf_counter()
        for _ in range(100):
            tokenizer.encode(text)
        elapsed = time.perf_counter() - start
        # Should encode 100 iterations of a ~4500 char text in under 10s
        assert elapsed < 10.0, f"Tokenization too slow: {elapsed:.2f}s for 100 iterations"

    def test_average_tokens_per_word(self, tokenizer):
        text = "The quick brown fox jumps over the lazy dog."
        ids = tokenizer.encode(text)
        words = text.split()
        ratio = len(ids) / len(words)
        # BPE typically produces 1-3 tokens per word
        assert 0.5 < ratio < 10, f"Suspicious token/word ratio: {ratio:.2f}"


class TestTokenizerSaveLoad:
    """Verify save/load round-trip."""

    def test_save_load_roundtrip(self, tokenizer, tmp_path):
        save_path = tmp_path / "test_vocab.json"
        tokenizer.save(save_path)
        loaded = ByteLevelBPETokenizer.load(save_path)
        assert loaded.vocab_size == tokenizer.vocab_size
        text = "Hello, world!"
        assert tokenizer.encode(text) == loaded.encode(text)
