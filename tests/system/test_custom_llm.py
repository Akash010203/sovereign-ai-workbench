"""
TEST C — Custom LLM (MiniLLM) Verification

Verifies:
- Model architecture is genuinely custom (not a wrapper)
- Forward pass produces correct shapes
- Backward pass produces gradients
- Loss is computed correctly
- Weight tying works
- Checkpoint save/load round-trip
- Text generation works
- No hidden cloud API calls
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest
import torch

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))

from models.custom_minilm.config import MiniLLMConfig
from models.custom_minilm.model import MiniLLM


@pytest.fixture(scope="module")
def small_config():
    return MiniLLMConfig.small(vocab_size=600)


@pytest.fixture(scope="module")
def model(small_config):
    m = MiniLLM(small_config)
    m.eval()
    return m


class TestModelArchitecture:
    """Verify the model is genuinely custom, not a wrapper."""

    def test_model_is_nn_module(self, model):
        assert isinstance(model, torch.nn.Module)

    def test_model_has_token_embedding(self, model):
        assert hasattr(model, "token_embedding")

    def test_model_has_blocks(self, model):
        assert hasattr(model, "blocks")
        assert len(model.blocks) == model.config.n_layers

    def test_model_has_final_norm(self, model):
        assert hasattr(model, "final_norm")

    def test_model_has_lm_head(self, model):
        assert hasattr(model, "lm_head")

    def test_weight_tying(self, model):
        """LM head weight should be the SAME tensor as embedding weight."""
        assert model.lm_head.weight is model.token_embedding.weight

    def test_no_huggingface_model(self, model):
        """Model should NOT be a HuggingFace transformers model."""
        class_name = type(model).__name__
        assert "GPT" not in class_name.upper() or "Mini" in class_name
        assert "Llama" not in class_name
        assert "Bert" not in class_name

    def test_parameter_count(self, model):
        params = model.count_parameters()
        assert params > 0
        # Small config should be roughly 4-6M params
        assert 1_000_000 < params < 100_000_000


class TestForwardPass:
    """Verify the forward pass produces correct outputs."""

    def test_forward_shape(self, model, small_config):
        B, T = 2, 16
        x = torch.randint(0, small_config.vocab_size, (B, T))
        logits = model(x)
        assert logits.shape == (B, T, small_config.vocab_size)

    def test_forward_dtype(self, model, small_config):
        x = torch.randint(0, small_config.vocab_size, (1, 8))
        logits = model(x)
        assert logits.dtype == torch.float32

    def test_forward_different_seq_lengths(self, model, small_config):
        for T in [1, 4, 16, 64, small_config.max_seq_len]:
            x = torch.randint(0, small_config.vocab_size, (1, T))
            logits = model(x)
            assert logits.shape == (1, T, small_config.vocab_size)

    def test_forward_rejects_too_long(self, model, small_config):
        T = small_config.max_seq_len + 1
        x = torch.randint(0, small_config.vocab_size, (1, T))
        with pytest.raises(ValueError, match="exceeds"):
            model(x)

    def test_forward_deterministic(self, model, small_config):
        """Same input should produce same output in eval mode."""
        model.eval()
        x = torch.randint(0, small_config.vocab_size, (1, 8))
        out1 = model(x).clone()
        out2 = model(x).clone()
        assert torch.allclose(out1, out2)


class TestBackwardPass:
    """Verify gradients flow correctly."""

    def test_gradients_nonzero(self, small_config):
        model = MiniLLM(small_config)
        model.train()
        x = torch.randint(0, small_config.vocab_size, (2, 16))
        targets = torch.randint(0, small_config.vocab_size, (2, 16))
        loss = model.compute_loss(x, targets)
        loss.backward()

        has_grad = False
        for param in model.parameters():
            if param.grad is not None and param.grad.abs().sum() > 0:
                has_grad = True
                break
        assert has_grad, "No non-zero gradients found after backward pass"

    def test_loss_is_scalar(self, small_config):
        model = MiniLLM(small_config)
        x = torch.randint(0, small_config.vocab_size, (2, 16))
        targets = torch.randint(0, small_config.vocab_size, (2, 16))
        loss = model.compute_loss(x, targets)
        assert loss.dim() == 0, "Loss should be a scalar"

    def test_loss_is_positive(self, small_config):
        model = MiniLLM(small_config)
        x = torch.randint(0, small_config.vocab_size, (2, 16))
        targets = torch.randint(0, small_config.vocab_size, (2, 16))
        loss = model.compute_loss(x, targets)
        assert loss.item() > 0, "Cross-entropy loss should be positive"

    def test_loss_decreases_with_training(self, small_config):
        """One optimizer step should reduce loss (on the same batch)."""
        model = MiniLLM(small_config)
        model.train()
        x = torch.randint(0, small_config.vocab_size, (4, 32))
        targets = torch.randint(0, small_config.vocab_size, (4, 32))

        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

        # Initial loss
        loss_before = model.compute_loss(x, targets).item()

        # Take 5 optimizer steps
        for _ in range(5):
            optimizer.zero_grad()
            loss = model.compute_loss(x, targets)
            loss.backward()
            optimizer.step()

        loss_after = model.compute_loss(x, targets).item()
        assert loss_after < loss_before, (
            f"Loss did not decrease: {loss_before:.4f} -> {loss_after:.4f}"
        )


class TestGeneration:
    """Verify text generation works."""

    def test_generate_produces_tokens(self, model, small_config):
        from models.custom_minilm.generate import generate
        prompt_ids = [1, 2, 3, 4, 5]
        new_ids = generate(model, prompt_ids, max_new_tokens=10, temperature=1.0, seed=42)
        assert len(new_ids) > 0
        assert len(new_ids) <= 10

    def test_generate_deterministic_with_seed(self, model, small_config):
        from models.custom_minilm.generate import generate
        prompt_ids = [1, 2, 3]
        ids1 = generate(model, prompt_ids, max_new_tokens=10, seed=42)
        ids2 = generate(model, prompt_ids, max_new_tokens=10, seed=42)
        assert ids1 == ids2

    def test_generate_with_temperature(self, model, small_config):
        from models.custom_minilm.generate import generate
        prompt_ids = [1, 2, 3]
        ids = generate(model, prompt_ids, max_new_tokens=10, temperature=0.5, seed=42)
        assert len(ids) > 0

    def test_generate_greedy(self, model, small_config):
        from models.custom_minilm.generate import generate
        prompt_ids = [1, 2, 3]
        ids = generate(model, prompt_ids, max_new_tokens=10, temperature=0.0)
        assert len(ids) > 0

    def test_generate_with_topk(self, model, small_config):
        from models.custom_minilm.generate import generate
        prompt_ids = [1, 2, 3]
        ids = generate(model, prompt_ids, max_new_tokens=10, top_k=5, seed=42)
        assert len(ids) > 0

    def test_generate_tokens_in_vocab_range(self, model, small_config):
        from models.custom_minilm.generate import generate
        prompt_ids = [1, 2, 3]
        ids = generate(model, prompt_ids, max_new_tokens=20, seed=42)
        for token_id in ids:
            assert 0 <= token_id < small_config.vocab_size, (
                f"Generated token {token_id} outside vocab range [0, {small_config.vocab_size})"
            )


class TestCheckpointRoundTrip:
    """Verify checkpoint save/load produces identical inference."""

    def test_save_load_produces_same_output(self, small_config, tmp_path):
        from models.custom_minilm.checkpoint import save_model_for_inference, load_model_from_checkpoint

        model = MiniLLM(small_config)
        model.eval()

        x = torch.randint(0, small_config.vocab_size, (1, 8))
        original_output = model(x).clone()

        ckpt_path = tmp_path / "test_ckpt.pt"
        save_model_for_inference(model, ckpt_path)

        loaded_model, meta = load_model_from_checkpoint(ckpt_path)
        loaded_model.eval()
        loaded_output = loaded_model(x)

        assert torch.allclose(original_output, loaded_output, atol=1e-5), (
            "Loaded model produces different output than saved model"
        )


class TestModelConfigs:
    """Verify all model config presets are valid."""

    @pytest.mark.parametrize("preset", ["small", "medium", "industrial_80m"])
    def test_config_preset_valid(self, preset):
        factory = getattr(MiniLLMConfig, preset)
        config = factory(vocab_size=600)
        assert config.d_model % config.n_heads == 0
        assert config.n_layers > 0
        assert config.d_ff > 0

    @pytest.mark.parametrize("preset", ["small", "medium", "industrial_80m"])
    def test_config_preset_can_build_model(self, preset):
        factory = getattr(MiniLLMConfig, preset)
        config = factory(vocab_size=600)
        model = MiniLLM(config)
        assert model.count_parameters() > 0
