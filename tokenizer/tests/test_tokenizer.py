"""
Tests for the from-scratch byte-level BPE tokenizer.

Run with:
    python -m pytest tokenizer/tests -v
"""
from __future__ import annotations

import pytest

from tokenizer.tokenizer import ByteLevelBPETokenizer, DEFAULT_SPECIAL_TOKENS

TOY_CORPUS = [
    "the quick brown fox jumps over the lazy dog",
    "the dog barked at the fox",
    "a quick fox is a happy fox",
]


def test_training_reaches_at_most_the_requested_vocab_size():
    tok = ByteLevelBPETokenizer.train(TOY_CORPUS, vocab_size=300)
    assert tok.vocab_size <= 300
    # Special tokens + all 256 base bytes must always be present.
    assert tok.vocab_size >= len(DEFAULT_SPECIAL_TOKENS) + 256


def test_special_tokens_are_registered():
    tok = ByteLevelBPETokenizer.train(TOY_CORPUS, vocab_size=300)
    for name in DEFAULT_SPECIAL_TOKENS:
        assert name in tok.vocab.special_to_id


def test_every_raw_byte_value_is_encodable():
    """Byte-level BPE must never have an 'unknown byte' - all 256
    values are in the base vocabulary before any merges are learned."""
    tok = ByteLevelBPETokenizer.train(TOY_CORPUS, vocab_size=300)
    assert len([b for b in tok.vocab.id_to_bytes.values() if len(b) == 1]) == 256


@pytest.mark.parametrize(
    "text",
    [
        "the quick brown fox",
        "Hello, World! 123",
        "unseen words are still encodable",
        "emoji and unicode: caf\u00e9 \u4e16\u754c \U0001F600",
        "",
    ],
)
def test_encode_decode_roundtrip(text):
    tok = ByteLevelBPETokenizer.train(TOY_CORPUS, vocab_size=300)
    ids = tok.encode(text)
    assert tok.decode(ids) == text


def test_common_word_compresses_to_fewer_tokens_than_raw_bytes():
    """A word that repeats often in the corpus (" fox") should end up
    as fewer tokens than its raw byte count, proving merges are
    actually learned and actually used at encode time."""
    tok = ByteLevelBPETokenizer.train(TOY_CORPUS, vocab_size=300)
    ids = tok.encode(" fox")
    raw_byte_count = len(" fox".encode("utf-8"))
    assert len(ids) < raw_byte_count


def test_bos_and_eos_are_added_only_when_requested():
    tok = ByteLevelBPETokenizer.train(TOY_CORPUS, vocab_size=300)
    plain = tok.encode("the fox")
    wrapped = tok.encode("the fox", add_bos=True, add_eos=True)

    assert wrapped[0] == tok.vocab.special_to_id["<bos>"]
    assert wrapped[-1] == tok.vocab.special_to_id["<eos>"]
    assert wrapped[1:-1] == plain


def test_training_is_deterministic():
    tok_a = ByteLevelBPETokenizer.train(TOY_CORPUS, vocab_size=300)
    tok_b = ByteLevelBPETokenizer.train(TOY_CORPUS, vocab_size=300)
    assert tok_a.vocab.merges == tok_b.vocab.merges
    assert tok_a.encode("the quick fox") == tok_b.encode("the quick fox")


def test_save_and_load_preserve_behaviour(tmp_path):
    tok = ByteLevelBPETokenizer.train(TOY_CORPUS, vocab_size=300)
    save_path = tmp_path / "vocab.json"
    tok.save(save_path)

    loaded = ByteLevelBPETokenizer.load(save_path)
    text = "the quick brown fox jumps"

    assert loaded.encode(text) == tok.encode(text)
    assert loaded.decode(tok.encode(text)) == text
    assert loaded.vocab_size == tok.vocab_size
    assert loaded.vocab.merges == tok.vocab.merges


def test_min_pair_frequency_prevents_overfitting_to_the_toy_corpus():
    tok = ByteLevelBPETokenizer.train(
        TOY_CORPUS, vocab_size=100_000, min_pair_frequency=2
    )
    # Training must stop once no pair repeats >= 2 times, so it should
    # never reach the absurdly large requested vocab size on this tiny
    # corpus.
    assert tok.vocab_size < 100_000


def test_encoding_is_consistent_regardless_of_input_order_of_words():
    tok = ByteLevelBPETokenizer.train(TOY_CORPUS, vocab_size=300)
    ids_1 = tok.encode(" fox")
    ids_2 = tok.encode("the fox")[len(tok.encode("the")):]
    assert ids_1 == ids_2
