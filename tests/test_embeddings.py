"""
Tests for the from-scratch MiniLLM's token embedding layer.

Requires torch (not installed in the build sandbox that generated this
repo -- see docs/BUILD_STATUS.md Phase 4 notes). Run on your machine
after `scripts\\setup_windows.ps1`:

    python -m pytest tests/test_embeddings.py -v
"""
import pytest

torch = pytest.importorskip("torch")

from models.custom_minilm.config import MiniLLMConfig
from models.custom_minilm.embeddings import TokenEmbedding


def make_tiny_config() -> MiniLLMConfig:
    return MiniLLMConfig(
        vocab_size=64, max_seq_len=16, d_model=32, n_layers=2, n_heads=4, d_ff=64
    )


def test_embedding_output_shape():
    config = make_tiny_config()
    embed = TokenEmbedding(config)

    token_ids = torch.randint(0, config.vocab_size, (2, 8))
    out = embed(token_ids)

    assert out.shape == (2, 8, config.d_model)


def test_embedding_rejects_wrong_rank_input():
    config = make_tiny_config()
    embed = TokenEmbedding(config)

    bad_input = torch.randint(0, config.vocab_size, (8,))  # missing batch dim
    with pytest.raises(ValueError):
        embed(bad_input)


def test_same_token_id_gives_the_same_vector():
    config = make_tiny_config()
    embed = TokenEmbedding(config)

    token_ids = torch.tensor([[3, 3, 3]])
    out = embed(token_ids)

    assert torch.allclose(out[0, 0], out[0, 1])
    assert torch.allclose(out[0, 1], out[0, 2])


def test_weight_property_matches_the_embedding_table():
    config = make_tiny_config()
    embed = TokenEmbedding(config)

    assert embed.weight is embed.embedding.weight
    assert embed.weight.shape == (config.vocab_size, config.d_model)


def test_embedding_gradients_flow_to_the_weight_table():
    config = make_tiny_config()
    embed = TokenEmbedding(config)

    token_ids = torch.randint(0, config.vocab_size, (2, 5))
    out = embed(token_ids)
    out.sum().backward()

    assert embed.weight.grad is not None
