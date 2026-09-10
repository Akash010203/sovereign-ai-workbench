"""
Tests for the from-scratch RMSNorm implementation.

Requires torch (not installed in the build sandbox that generated this
repo -- see docs/BUILD_STATUS.md Phase 4 notes). Run on your machine:

    python -m pytest tests/test_normalization.py -v
"""
import pytest

torch = pytest.importorskip("torch")

from models.custom_minilm.normalization import RMSNorm


def test_rmsnorm_preserves_shape():
    norm = RMSNorm(d_model=32)
    x = torch.randn(2, 8, 32)
    y = norm(x)
    assert y.shape == x.shape


def test_rmsnorm_produces_unit_rms_before_weight_scaling():
    norm = RMSNorm(d_model=32)
    # Force the learned scale to 1 so we test the *normalization* math
    # itself, not the learned rescaling layered on top of it.
    with torch.no_grad():
        norm.weight.fill_(1.0)

    x = torch.randn(4, 10, 32) * 5.0  # arbitrary large input scale
    y = norm(x)

    rms = y.pow(2).mean(dim=-1).sqrt()
    assert torch.allclose(rms, torch.ones_like(rms), atol=1e-4)


def test_rmsnorm_is_invariant_to_input_scale_up_to_learned_weight():
    norm = RMSNorm(d_model=16)
    with torch.no_grad():
        norm.weight.fill_(1.0)

    x = torch.randn(2, 5, 16)
    y_small = norm(x)
    y_large = norm(x * 10.0)  # same direction, 10x the magnitude

    assert torch.allclose(y_small, y_large, atol=1e-4)


def test_rmsnorm_gradients_flow_to_input_and_weight():
    norm = RMSNorm(d_model=16)
    x = torch.randn(2, 5, 16, requires_grad=True)

    y = norm(x)
    y.sum().backward()

    assert x.grad is not None
    assert norm.weight.grad is not None


def test_rmsnorm_eps_prevents_division_by_zero_on_all_zero_input():
    norm = RMSNorm(d_model=8, eps=1e-5)
    x = torch.zeros(1, 1, 8)
    y = norm(x)
    assert torch.isfinite(y).all()
