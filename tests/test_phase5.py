"""
Phase 5 tests — Custom Transformer / MiniLLM assembly.

WHAT THESE TESTS VERIFY
------------------------
1. parameter_count   — model reports a sensible total (4-8 M range)
2. parameter_summary — summary string prints without error
3. weight_tying      — lm_head.weight IS the embedding matrix (same
                       object, not just equal values)
4. forward_shape     — forward() returns exactly [B, T, vocab_size]
5. forward_no_nan    — no NaN/Inf in logits at init (numerical sanity)
6. loss_shape        — compute_loss() returns a scalar
7. loss_is_finite    — loss is finite at random init
8. loss_is_roughly_ln_V — at random init, cross-entropy loss should be
                           approximately ln(vocab_size) ≈ 6.4 for
                           vocab_size=600 (model is guessing uniformly).
                           We allow ±2.0 of tolerance.
9. causal_mask       — changing a FUTURE token must NOT change an
                       EARLIER token's logits. This re-validates
                       causality at the assembled-model level.
10. backward_pass    — loss.backward() runs without error, all
                       parameters receive a gradient.
11. seq_len_1        — a single-token input (T=1) works (important for
                       generation where the model processes one token
                       at a time in KV-cache mode later).
12. max_seq_len      — a batch of max_seq_len tokens works.
13. seq_len_overflow — a sequence longer than max_seq_len raises
                       ValueError (model must refuse, not silently
                       truncate or crash with a shape error).
14. ffn_shape        — SwiGLUFFN forward produces [B, T, d_model].
15. ffn_no_nan       — no NaN in FFN output at init.
16. block_shape      — TransformerBlock forward produces [B, T, d_model].
17. block_residual   — zero-input → zero-output (residuals preserve
                       zero when weights are zero; here we test that
                       a non-zero input produces a non-zero output,
                       confirming the non-linear path fires).

All tests are self-contained and use a tiny, fixed config so they run
quickly even on CPU.

Run with:
    python -m pytest tests/test_phase5.py -v
"""
from __future__ import annotations

import math

import pytest

torch = pytest.importorskip("torch", reason="PyTorch required for Phase 5 tests")

from models.custom_minilm.config import MiniLLMConfig
from models.custom_minilm.ffn import SwiGLUFFN
from models.custom_minilm.block import TransformerBlock
from models.custom_minilm.model import MiniLLM


# ─── Shared tiny config ────────────────────────────────────────────────────────
@pytest.fixture(scope="module")
def cfg() -> MiniLLMConfig:
    """
    A deliberately small config so tests run in seconds on CPU.

    vocab_size = 64   (matches minimum BPE special+base, unrealistically
                       small, but fast for tests)
    d_model    = 32
    n_layers   = 2
    n_heads    = 4    (head_dim = 32/4 = 8, even ✓ for RoPE)
    d_ff       = 86   (≈ 2.67 × 32, SwiGLU ratio)
    max_seq_len= 16
    """
    return MiniLLMConfig(
        vocab_size=64,
        d_model=32,
        n_layers=2,
        n_heads=4,
        d_ff=86,
        max_seq_len=16,
        dropout=0.0,
    )


@pytest.fixture(scope="module")
def model(cfg: MiniLLMConfig) -> MiniLLM:
    """Instantiate the full model once per test module."""
    m = MiniLLM(cfg)
    m.eval()
    return m


# ─── Fixtures for sub-modules ──────────────────────────────────────────────────
@pytest.fixture(scope="module")
def ffn(cfg: MiniLLMConfig) -> SwiGLUFFN:
    return SwiGLUFFN(cfg).eval()


@pytest.fixture(scope="module")
def block(cfg: MiniLLMConfig) -> TransformerBlock:
    return TransformerBlock(cfg).eval()


# ═══════════════════════════════════════════════════════════════════════════════
# 1. Parameter count
# ═══════════════════════════════════════════════════════════════════════════════

def test_parameter_count(model: MiniLLM, cfg: MiniLLMConfig) -> None:
    """Total unique trainable parameters should be positive and plausible."""
    total = model.count_parameters()
    assert total > 0, "Model has no parameters — something is very wrong."
    # For tiny cfg: rough lower bound = 2 blocks × (4 × 32² + 3 × 32 × 86) = ~50 K
    # just verify it's > 10 K and < 100 M (sanity bounds)
    assert total > 10_000, f"Parameter count {total:,} is suspiciously low."
    assert total < 100_000_000, f"Parameter count {total:,} exceeds sanity bound."


def test_parameter_summary_runs(model: MiniLLM) -> None:
    """parameter_summary() must return a non-empty string without errors."""
    summary = model.parameter_summary()
    assert isinstance(summary, str)
    assert len(summary) > 0
    assert "TOTAL" in summary


# ═══════════════════════════════════════════════════════════════════════════════
# 3. Weight tying
# ═══════════════════════════════════════════════════════════════════════════════

def test_weight_tying(model: MiniLLM) -> None:
    """
    lm_head.weight must be the *same tensor object* as
    token_embedding.embedding.weight (tied, not copied).
    """
    assert model.lm_head.weight is model.token_embedding.weight, (
        "Weight tying failed: lm_head.weight and embedding.weight "
        "are different tensors."
    )


# ═══════════════════════════════════════════════════════════════════════════════
# 4-5. Forward pass — shape and numerical sanity
# ═══════════════════════════════════════════════════════════════════════════════

def test_forward_shape(model: MiniLLM, cfg: MiniLLMConfig) -> None:
    """forward() must return [B, T, vocab_size]."""
    B, T = 2, 8
    input_ids = torch.randint(0, cfg.vocab_size, (B, T))
    with torch.no_grad():
        logits = model(input_ids)
    assert logits.shape == (B, T, cfg.vocab_size), (
        f"Expected logits shape ({B}, {T}, {cfg.vocab_size}), "
        f"got {tuple(logits.shape)}."
    )


def test_forward_no_nan(model: MiniLLM, cfg: MiniLLMConfig) -> None:
    """No NaN or Inf in logits at random initialization."""
    input_ids = torch.randint(0, cfg.vocab_size, (2, 8))
    with torch.no_grad():
        logits = model(input_ids)
    assert torch.isfinite(logits).all(), "NaN or Inf detected in logits at init."


# ═══════════════════════════════════════════════════════════════════════════════
# 6-8. Loss
# ═══════════════════════════════════════════════════════════════════════════════

def test_loss_shape(model: MiniLLM, cfg: MiniLLMConfig) -> None:
    """compute_loss() must return a scalar (0-dim) tensor."""
    B, T = 2, 8
    input_ids = torch.randint(0, cfg.vocab_size, (B, T))
    targets   = torch.randint(0, cfg.vocab_size, (B, T))
    loss = model.compute_loss(input_ids, targets)
    assert loss.ndim == 0, f"Loss is not a scalar — shape: {tuple(loss.shape)}."


def test_loss_is_finite(model: MiniLLM, cfg: MiniLLMConfig) -> None:
    """Loss must be a finite number at random initialization."""
    B, T = 2, 8
    input_ids = torch.randint(0, cfg.vocab_size, (B, T))
    targets   = torch.randint(0, cfg.vocab_size, (B, T))
    loss = model.compute_loss(input_ids, targets)
    assert torch.isfinite(loss), f"Loss is not finite: {loss.item()}"


def test_loss_approximately_ln_vocab(model: MiniLLM, cfg: MiniLLMConfig) -> None:
    """
    At random init a well-calibrated model guesses uniformly, so the
    expected cross-entropy is ln(vocab_size).  We allow ±2.0 tolerance
    to account for RNG variation and small initialisation asymmetries.
    """
    torch.manual_seed(0)
    B, T = 4, 16
    input_ids = torch.randint(0, cfg.vocab_size, (B, T))
    targets   = torch.randint(0, cfg.vocab_size, (B, T))
    with torch.no_grad():
        loss = model.compute_loss(input_ids, targets)
    expected = math.log(cfg.vocab_size)   # ln(64) ≈ 4.16
    assert abs(loss.item() - expected) < 2.0, (
        f"Loss at init ({loss.item():.3f}) is far from "
        f"ln(vocab_size)={expected:.3f}.  "
        "Check initialization or cross-entropy implementation."
    )


# ═══════════════════════════════════════════════════════════════════════════════
# 9. Causal mask — assembled-model level
# ═══════════════════════════════════════════════════════════════════════════════

def test_causal_mask_assembled_model(model: MiniLLM, cfg: MiniLLMConfig) -> None:
    """
    Changing a FUTURE token must NOT affect an EARLIER position's logits.

    Concretely: for a sequence of length T=8, we take positions 0..3
    as "past" and position 4 as "future".  We run the model on the
    original sequence and then on a version where position 4 is set to a
    different token.  The logits at positions 0..3 must be identical.
    """
    T = 8
    torch.manual_seed(42)
    input_ids = torch.randint(0, cfg.vocab_size, (1, T))

    with torch.no_grad():
        logits_orig = model(input_ids).clone()

    # Change position 4 (future relative to positions 0-3)
    modified = input_ids.clone()
    future_pos = 4
    original_tok = int(modified[0, future_pos].item())
    # Pick a different token
    new_tok = (original_tok + 1) % cfg.vocab_size
    modified[0, future_pos] = new_tok

    with torch.no_grad():
        logits_mod = model(modified).clone()

    # Positions 0..future_pos-1 must be unchanged
    past_logits_orig = logits_orig[0, :future_pos, :]
    past_logits_mod  = logits_mod[0,  :future_pos, :]
    assert torch.allclose(past_logits_orig, past_logits_mod, atol=1e-5), (
        "Causal mask VIOLATED at assembled model level: "
        "changing a future token changed an earlier position's logits."
    )

    # Position future_pos itself CAN change (it sees its own input)
    # — we just assert positions BEFORE it are unaffected.


# ═══════════════════════════════════════════════════════════════════════════════
# 10. Backward pass / gradient flow
# ═══════════════════════════════════════════════════════════════════════════════

def test_backward_pass(cfg: MiniLLMConfig) -> None:
    """
    loss.backward() must complete without error and every trainable
    parameter must receive a non-None gradient.
    """
    m = MiniLLM(cfg)   # fresh model (not eval, we need grad tracking)
    B, T = 2, 8
    input_ids = torch.randint(0, cfg.vocab_size, (B, T))
    targets   = torch.randint(0, cfg.vocab_size, (B, T))
    loss = m.compute_loss(input_ids, targets)
    loss.backward()

    params_without_grad = [
        name for name, p in m.named_parameters() if p.requires_grad and p.grad is None
    ]
    assert not params_without_grad, (
        f"These parameters have no gradient after backward(): "
        f"{params_without_grad}"
    )


# ═══════════════════════════════════════════════════════════════════════════════
# 11-13. Sequence-length edge cases
# ═══════════════════════════════════════════════════════════════════════════════

def test_seq_len_1(model: MiniLLM, cfg: MiniLLMConfig) -> None:
    """A single-token input (T=1) must work — needed for generation."""
    input_ids = torch.randint(0, cfg.vocab_size, (1, 1))
    with torch.no_grad():
        logits = model(input_ids)
    assert logits.shape == (1, 1, cfg.vocab_size)


def test_max_seq_len(model: MiniLLM, cfg: MiniLLMConfig) -> None:
    """A full-length sequence (T = max_seq_len) must work."""
    input_ids = torch.randint(0, cfg.vocab_size, (1, cfg.max_seq_len))
    with torch.no_grad():
        logits = model(input_ids)
    assert logits.shape == (1, cfg.max_seq_len, cfg.vocab_size)


def test_seq_len_overflow(model: MiniLLM, cfg: MiniLLMConfig) -> None:
    """Sequences longer than max_seq_len must raise ValueError."""
    input_ids = torch.randint(0, cfg.vocab_size, (1, cfg.max_seq_len + 1))
    with pytest.raises(ValueError, match="max_seq_len"):
        model(input_ids)


# ═══════════════════════════════════════════════════════════════════════════════
# 14-15. SwiGLUFFN sub-module
# ═══════════════════════════════════════════════════════════════════════════════

def test_ffn_shape(ffn: SwiGLUFFN, cfg: MiniLLMConfig) -> None:
    """FFN forward: [B, T, d_model] → [B, T, d_model]."""
    x = torch.randn(2, 8, cfg.d_model)
    with torch.no_grad():
        out = ffn(x)
    assert out.shape == x.shape, (
        f"FFN output shape {tuple(out.shape)} != input shape {tuple(x.shape)}."
    )


def test_ffn_no_nan(ffn: SwiGLUFFN, cfg: MiniLLMConfig) -> None:
    """No NaN in FFN output at random initialization."""
    x = torch.randn(2, 8, cfg.d_model)
    with torch.no_grad():
        out = ffn(x)
    assert torch.isfinite(out).all(), "NaN or Inf detected in FFN output."


# ═══════════════════════════════════════════════════════════════════════════════
# 16-17. TransformerBlock sub-module
# ═══════════════════════════════════════════════════════════════════════════════

def test_block_shape(block: TransformerBlock, cfg: MiniLLMConfig) -> None:
    """Block forward: [B, T, d_model] → [B, T, d_model]."""
    x = torch.randn(2, 8, cfg.d_model)
    with torch.no_grad():
        out = block(x)
    assert out.shape == x.shape, (
        f"Block output shape {tuple(out.shape)} != input shape {tuple(x.shape)}."
    )


def test_block_non_trivial_output(block: TransformerBlock, cfg: MiniLLMConfig) -> None:
    """
    A non-zero input through a randomly-initialized block must produce
    a non-zero output.  (Verifies the non-linear path is active,
    not collapsed to a trivial identity.)
    """
    x = torch.randn(2, 8, cfg.d_model)
    with torch.no_grad():
        out = block(x)
    # output != input (attention + FFN should transform the signal)
    assert not torch.allclose(out, x, atol=1e-6), (
        "Block output is identical to its input — the non-linear "
        "transformations do not appear to be firing."
    )
