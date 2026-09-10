"""
Tests for the from-scratch causal multi-head self-attention.

Requires torch (not installed in the build sandbox that generated this
repo -- see docs/BUILD_STATUS.md Phase 4 notes). Run on your machine:

    python -m pytest tests/test_attention.py -v
"""
import pytest

torch = pytest.importorskip("torch")

from models.custom_minilm.attention import CausalSelfAttention
from models.custom_minilm.config import MiniLLMConfig


def make_tiny_config() -> MiniLLMConfig:
    return MiniLLMConfig(
        vocab_size=64,
        max_seq_len=16,
        d_model=32,
        n_layers=2,
        n_heads=4,
        d_ff=64,
        dropout=0.0,
    )


def test_attention_output_shape():
    config = make_tiny_config()
    attn = CausalSelfAttention(config)

    x = torch.randn(2, 8, config.d_model)
    y = attn(x)

    assert y.shape == x.shape


def test_attention_is_causal():
    """Changing a *future* token must not change an *earlier* token's
    output -- this is the entire point of the causal mask, and the
    single most important behavioural guarantee of this module."""
    torch.manual_seed(0)
    config = make_tiny_config()
    attn = CausalSelfAttention(config)
    attn.eval()

    x = torch.randn(1, 8, config.d_model)
    x_modified = x.clone()
    prefix_len = 3
    x_modified[:, prefix_len:] = torch.randn_like(x_modified[:, prefix_len:])

    y = attn(x)
    y_modified = attn(x_modified)

    assert torch.allclose(y[:, :prefix_len], y_modified[:, :prefix_len], atol=1e-6)


def test_attention_weights_sum_to_one_per_query():
    """Recomputes the attention weights the same way forward() does,
    as an internal sanity check on the softmax step itself."""
    config = make_tiny_config()
    attn = CausalSelfAttention(config)
    attn.eval()

    x = torch.randn(1, 5, config.d_model)
    batch_size, seq_len, _ = x.shape

    q = attn._split_heads(attn.q_proj(x), batch_size, seq_len)
    k = attn._split_heads(attn.k_proj(x), batch_size, seq_len)
    q, k = attn.rope(q), attn.rope(k)

    scores = (q @ k.transpose(-2, -1)) / (config.head_dim**0.5)
    mask = attn.causal_mask[:seq_len, :seq_len]
    scores = scores.masked_fill(mask, float("-inf"))
    weights = torch.softmax(scores, dim=-1)

    row_sums = weights.sum(dim=-1)
    assert torch.allclose(row_sums, torch.ones_like(row_sums), atol=1e-5)


def test_attention_gradients_flow():
    config = make_tiny_config()
    attn = CausalSelfAttention(config)

    x = torch.randn(2, 6, config.d_model, requires_grad=True)
    y = attn(x)
    y.sum().backward()

    assert x.grad is not None
    assert attn.q_proj.weight.grad is not None


def test_attention_handles_a_single_token_sequence():
    """T=1 is an edge case worth checking explicitly: there is nothing
    to mask, and the single token can only attend to itself."""
    config = make_tiny_config()
    attn = CausalSelfAttention(config)

    x = torch.randn(1, 1, config.d_model)
    y = attn(x)

    assert y.shape == x.shape
    assert torch.isfinite(y).all()
