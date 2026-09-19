"""
TEST S — Performance Measurements

Measures:
- Model loading time
- Tokenizer encoding speed
- Generation speed
- RAM usage
- GPU detection
"""
from __future__ import annotations

import os
import sys
import time
from pathlib import Path

import pytest
import torch

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))


class TestGPUDetection:
    """Verify GPU/CUDA detection is accurate."""

    def test_cuda_availability_reported(self):
        available = torch.cuda.is_available()
        # Just verify it doesn't crash and returns a bool
        assert isinstance(available, bool)

    def test_gpu_name_if_available(self):
        if torch.cuda.is_available():
            name = torch.cuda.get_device_name(0)
            assert len(name) > 0
            print(f"GPU: {name}")

    def test_vram_if_available(self):
        if torch.cuda.is_available():
            total = torch.cuda.get_device_properties(0).total_memory / (1024**3)
            assert total > 0
            print(f"VRAM: {total:.1f} GB")


class TestTokenizerPerformance:
    """Measure tokenizer performance."""

    def test_encoding_throughput(self):
        from tokenizer.tokenizer import ByteLevelBPETokenizer
        vocab_path = ROOT / "tokenizer" / "vocab" / "bpe_8k.json"
        if not vocab_path.exists():
            vocab_path = ROOT / "tokenizer" / "vocab" / "demo_bpe_vocab.json"
        tok = ByteLevelBPETokenizer.load(vocab_path)

        text = "The quick brown fox jumps over the lazy dog. " * 50
        start = time.perf_counter()
        n_iterations = 100
        total_tokens = 0
        for _ in range(n_iterations):
            ids = tok.encode(text)
            total_tokens += len(ids)
        elapsed = time.perf_counter() - start

        tokens_per_sec = total_tokens / elapsed
        print(f"Tokenizer: {tokens_per_sec:.0f} tokens/sec")
        assert tokens_per_sec > 100, f"Tokenizer too slow: {tokens_per_sec:.0f} tok/s"


class TestModelPerformance:
    """Measure model performance."""

    def test_forward_pass_speed(self):
        from models.custom_minilm.config import MiniLLMConfig
        from models.custom_minilm.model import MiniLLM

        config = MiniLLMConfig.small(vocab_size=600)
        model = MiniLLM(config)
        model.eval()

        x = torch.randint(0, 600, (1, 64))

        # Warmup
        with torch.no_grad():
            model(x)

        start = time.perf_counter()
        n = 50
        with torch.no_grad():
            for _ in range(n):
                model(x)
        elapsed = time.perf_counter() - start

        ms_per_forward = (elapsed / n) * 1000
        print(f"Forward pass: {ms_per_forward:.1f} ms")
        assert ms_per_forward < 1000, f"Forward pass too slow: {ms_per_forward:.1f} ms"

    def test_generation_speed(self):
        from models.custom_minilm.config import MiniLLMConfig
        from models.custom_minilm.model import MiniLLM
        from models.custom_minilm.generate import generate

        config = MiniLLMConfig.small(vocab_size=600)
        model = MiniLLM(config)
        model.eval()

        prompt = [1, 2, 3, 4, 5]
        start = time.perf_counter()
        tokens = generate(model, prompt, max_new_tokens=32, temperature=1.0, seed=42)
        elapsed = time.perf_counter() - start

        tokens_per_sec = len(tokens) / elapsed
        print(f"Generation: {tokens_per_sec:.1f} tokens/sec ({elapsed:.2f}s for {len(tokens)} tokens)")
        assert tokens_per_sec > 1, f"Generation too slow: {tokens_per_sec:.1f} tok/s"


class TestMemoryUsage:
    """Verify memory usage is reasonable."""

    def test_model_ram_usage(self):
        import psutil
        process = psutil.Process(os.getpid())
        mem_before = process.memory_info().rss / (1024**2)

        from models.custom_minilm.config import MiniLLMConfig
        from models.custom_minilm.model import MiniLLM
        config = MiniLLMConfig.small(vocab_size=600)
        model = MiniLLM(config)

        mem_after = process.memory_info().rss / (1024**2)
        model_mem = mem_after - mem_before
        print(f"Model RAM: ~{model_mem:.0f} MB")
        # Small model should be under 100 MB
        assert model_mem < 500, f"Model uses too much RAM: {model_mem:.0f} MB"
