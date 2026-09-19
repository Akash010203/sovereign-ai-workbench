"""
TEST D — Training Pipeline Verification

Verifies:
- Tiny training experiment runs end-to-end
- Loss decreases
- Gradients are non-zero
- Weights actually change
- Checkpoint changes after training
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


class TestTrainingLoop:
    """Verify a minimal training loop works correctly."""

    def test_loss_decreases_over_steps(self):
        config = MiniLLMConfig.small(vocab_size=600)
        model = MiniLLM(config)
        model.train()
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

        # Fixed batch
        x = torch.randint(0, 600, (4, 32))
        targets = torch.randint(0, 600, (4, 32))

        losses = []
        for step in range(20):
            optimizer.zero_grad()
            loss = model.compute_loss(x, targets)
            loss.backward()
            optimizer.step()
            losses.append(loss.item())

        # Loss should decrease overall
        assert losses[-1] < losses[0], (
            f"Loss did not decrease: {losses[0]:.4f} -> {losses[-1]:.4f}"
        )

    def test_weights_change_after_training(self):
        config = MiniLLMConfig.small(vocab_size=600)
        model = MiniLLM(config)
        model.train()

        # Snapshot initial weights
        initial_weights = {
            name: param.clone().detach()
            for name, param in model.named_parameters()
        }

        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
        x = torch.randint(0, 600, (4, 32))
        targets = torch.randint(0, 600, (4, 32))

        for _ in range(5):
            optimizer.zero_grad()
            loss = model.compute_loss(x, targets)
            loss.backward()
            optimizer.step()

        # At least some weights should have changed
        changed = False
        for name, param in model.named_parameters():
            if not torch.allclose(param, initial_weights[name]):
                changed = True
                break
        assert changed, "No weights changed after training"

    def test_gradients_are_nonzero(self):
        config = MiniLLMConfig.small(vocab_size=600)
        model = MiniLLM(config)
        model.train()

        x = torch.randint(0, 600, (2, 16))
        targets = torch.randint(0, 600, (2, 16))
        loss = model.compute_loss(x, targets)
        loss.backward()

        nonzero_grads = 0
        total_params = 0
        for param in model.parameters():
            if param.grad is not None:
                total_params += 1
                if param.grad.abs().sum() > 0:
                    nonzero_grads += 1

        assert nonzero_grads > 0, "All gradients are zero"
        # Most parameters should have non-zero gradients
        assert nonzero_grads / total_params > 0.5, (
            f"Only {nonzero_grads}/{total_params} parameters have non-zero gradients"
        )

    def test_checkpoint_changes_after_training(self, tmp_path):
        config = MiniLLMConfig.small(vocab_size=600)
        model = MiniLLM(config)

        # Save before training
        ckpt_before = tmp_path / "before.pt"
        torch.save(model.state_dict(), ckpt_before)

        # Train
        model.train()
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
        x = torch.randint(0, 600, (4, 32))
        targets = torch.randint(0, 600, (4, 32))
        for _ in range(10):
            optimizer.zero_grad()
            loss = model.compute_loss(x, targets)
            loss.backward()
            optimizer.step()

        # Save after training
        ckpt_after = tmp_path / "after.pt"
        torch.save(model.state_dict(), ckpt_after)

        # Checkpoint files should differ
        before_bytes = ckpt_before.read_bytes()
        after_bytes = ckpt_after.read_bytes()
        assert before_bytes != after_bytes, "Checkpoint unchanged after training"

    def test_reload_trained_model_gives_valid_inference(self, tmp_path):
        config = MiniLLMConfig.small(vocab_size=600)
        model = MiniLLM(config)
        model.train()

        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
        x = torch.randint(0, 600, (4, 32))
        targets = torch.randint(0, 600, (4, 32))
        for _ in range(10):
            optimizer.zero_grad()
            loss = model.compute_loss(x, targets)
            loss.backward()
            optimizer.step()

        # Save
        ckpt = tmp_path / "trained.pt"
        torch.save({
            "model_state": model.state_dict(),
            "model_config": {"vocab_size": 600, "d_model": 256, "n_layers": 6,
                             "n_heads": 4, "d_ff": 683, "max_seq_len": 128,
                             "dropout": 0.0, "rms_norm_eps": 1e-5,
                             "rope_theta": 10000.0, "bias": False},
        }, ckpt)

        # Reload
        from models.custom_minilm.checkpoint import load_model_from_checkpoint
        loaded, meta = load_model_from_checkpoint(ckpt)
        loaded.eval()

        # Should produce valid logits
        test_input = torch.randint(0, 600, (1, 8))
        logits = loaded(test_input)
        assert logits.shape == (1, 8, 600)
        assert not torch.isnan(logits).any()
        assert not torch.isinf(logits).any()
