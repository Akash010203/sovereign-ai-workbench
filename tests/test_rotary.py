"""
Tests for the from-scratch Rotary Position Embedding (RoPE).

Requires torch (not installed in the build sandbox that generated this
repo -- see docs/BUILD_STATUS.md Phase 4 notes). Run on your machine:

    python -m pytest tests/test_rotary.py -v
"""
import pytest

torch = pytest.importorskip("torch")

from models.custom_minilm.rotary import RotaryEmbedding


def test_rope_preserves_shape():
    rope = RotaryEmbedding(head_dim=32, max_seq_len=64)
    x = torch.randn(2, 4, 10, 32)
    y = rope(x)
    assert y.shape == x.shape


def test_rope_position_zero_is_unchanged():
    """At position 0, cos(0)=1 and sin(0)=0, so rotation is the identity."""
    rope = RotaryEmbedding(head_dim=32, max_seq_len=64)
    x = torch.randn(2, 4, 10, 32)
    y = rope(x)
    assert torch.allclose(y[:, :, 0], x[:, :, 0], atol=1e-6)


def test_rope_preserves_vector_norm():
    """Rotation never changes a vector's length, only its direction --
    this is what lets RoPE inject position information without
    distorting the magnitude of query/key vectors."""
    rope = RotaryEmbedding(head_dim=32, max_seq_len=64)
    x = torch.randn(2, 4, 10, 32)
    y = rope(x)

    assert torch.allclose(x.norm(dim=-1), y.norm(dim=-1), atol=1e-4)


def test_rope_rejects_odd_head_dim():
    with pytest.raises(ValueError):
        RotaryEmbedding(head_dim=31, max_seq_len=64)


def test_rope_gives_different_output_at_different_positions():
    """Two identical vectors placed at different sequence positions
    must come out differently rotated -- otherwise RoPE would be
    injecting no position information at all."""
    rope = RotaryEmbedding(head_dim=16, max_seq_len=64)
    same_vector = torch.randn(1, 1, 1, 16)
    x = same_vector.repeat(1, 1, 5, 1)  # same vector at 5 positions

    y = rope(x)

    assert not torch.allclose(y[:, :, 1], y[:, :, 2])
