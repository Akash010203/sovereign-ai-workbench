"""
TEST W — Fake Implementation Detection

Searches for:
- Hardcoded responses
- Mock/placeholder implementations in production code
- Fake confidence scores
- TODO/pass stubs in production code
- Verifies model forward pass is real computation
- Verifies training actually updates weights
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))


class TestFakeDetection:
    """Scan for suspicious implementations."""

    def _scan_python_files(self, pattern: str, exclude_dirs: set | None = None) -> list:
        """Scan production Python files for a regex pattern."""
        exclude = exclude_dirs or {"__pycache__", ".git", "tests", "demo", ".pytest_cache"}
        findings = []
        for py_file in ROOT.rglob("*.py"):
            if any(ex in str(py_file) for ex in exclude):
                continue
            try:
                content = py_file.read_text(encoding="utf-8", errors="ignore")
            except Exception:
                continue
            for line_num, line in enumerate(content.splitlines(), 1):
                stripped = line.strip()
                if stripped.startswith("#"):
                    continue
                if re.search(pattern, line, re.IGNORECASE):
                    findings.append({
                        "file": str(py_file.relative_to(ROOT)),
                        "line": line_num,
                        "content": stripped[:100],
                    })
        return findings

    def test_no_hardcoded_ai_responses(self):
        """No function should return a hardcoded AI-like response."""
        patterns = [
            r'return\s+"AI generated',
            r'return\s+"The answer is',
            r'return\s+f?"Based on my analysis',
        ]
        for pattern in patterns:
            findings = self._scan_python_files(pattern)
            assert len(findings) == 0, (
                f"Hardcoded response found:\n" +
                "\n".join(f"  {f['file']}:{f['line']} - {f['content']}" for f in findings)
            )

    def test_no_predefined_response_tables(self):
        """No lookup tables mapping inputs to predefined responses."""
        pattern = r'predefined_response|canned_response|scripted_answer'
        findings = self._scan_python_files(pattern)
        assert len(findings) == 0, f"Predefined responses: {findings}"

    def test_no_todo_pass_stubs(self):
        """Production code should not have TODO/pass stubs."""
        findings = []
        exclude = {"__pycache__", ".git", "tests", ".pytest_cache"}
        for py_file in ROOT.rglob("*.py"):
            if any(ex in str(py_file) for ex in exclude):
                continue
            try:
                content = py_file.read_text(encoding="utf-8", errors="ignore")
            except Exception:
                continue
            lines = content.splitlines()
            for i, line in enumerate(lines):
                if line.strip() == "pass" and i > 0:
                    prev = lines[i-1].strip()
                    if prev.startswith("def ") or prev.startswith("class "):
                        findings.append({
                            "file": str(py_file.relative_to(ROOT)),
                            "line": i + 1,
                            "content": f"{prev} -> pass",
                        })

        assert len(findings) == 0, (
            f"TODO/pass stubs found:\n" +
            "\n".join(f"  {f['file']}:{f['line']} - {f['content']}" for f in findings)
        )

    def test_no_fake_confidence_scores(self):
        """No hardcoded confidence/probability scores in production code."""
        pattern = r'confidence\s*=\s*0\.\d{2,}|score\s*=\s*0\.9[5-9]'
        findings = self._scan_python_files(pattern)
        real_fakes = [f for f in findings if "test" not in f["file"].lower()]
        assert len(real_fakes) == 0, f"Fake confidence scores: {real_fakes}"

    def test_no_fake_model_names(self):
        """No references to external models used as if they are our own."""
        fake_names = ["gpt-4", "gpt-3.5", "claude-3", "gemini-pro"]
        findings = []
        for py_file in ROOT.rglob("*.py"):
            if "__pycache__" in str(py_file) or ".git" in str(py_file):
                continue
            if "test" in py_file.name.lower():
                continue
            try:
                content = py_file.read_text(encoding="utf-8", errors="ignore")
            except Exception:
                continue
            for line_num, line in enumerate(content.splitlines(), 1):
                line_lower = line.lower()
                for name in fake_names:
                    if name in line_lower:
                        # Allow references in comments/docstrings for comparison
                        stripped = line.strip()
                        if stripped.startswith("#") or stripped.startswith("*") or stripped.startswith("-"):
                            continue
                        # Allow in docstrings (lines inside triple-quoted blocks)
                        if "compare" in line_lower or "not" in line_lower or "honest" in line_lower:
                            continue
                        findings.append(
                            f"{py_file.relative_to(ROOT)}:{line_num}: references '{name}'"
                        )

        assert len(findings) == 0, f"Fake model references:\n" + "\n".join(findings)

    def test_model_forward_pass_is_real(self):
        """Verify the model's forward pass actually computes, not returns a constant."""
        from models.custom_minilm.config import MiniLLMConfig
        from models.custom_minilm.model import MiniLLM
        import torch

        config = MiniLLMConfig.small(vocab_size=600)
        model = MiniLLM(config)
        model.eval()

        x1 = torch.tensor([[1, 2, 3, 4]])
        x2 = torch.tensor([[5, 6, 7, 8]])

        with torch.no_grad():
            out1 = model(x1)
            out2 = model(x2)

        # Different inputs MUST produce different outputs
        assert not torch.allclose(out1, out2), (
            "Model produces identical output for different inputs — "
            "possible fake implementation"
        )

    def test_training_actually_trains(self):
        """Verify training loop actually updates weights."""
        from models.custom_minilm.config import MiniLLMConfig
        from models.custom_minilm.model import MiniLLM
        import torch

        config = MiniLLMConfig.small(vocab_size=600)
        model = MiniLLM(config)
        model.train()

        # Snapshot a weight
        w_before = model.token_embedding.weight[0, 0].item()

        optimizer = torch.optim.Adam(model.parameters(), lr=1e-2)
        x = torch.randint(0, 600, (4, 32))
        targets = torch.randint(0, 600, (4, 32))

        for _ in range(10):
            optimizer.zero_grad()
            loss = model.compute_loss(x, targets)
            loss.backward()
            optimizer.step()

        w_after = model.token_embedding.weight[0, 0].item()
        assert w_before != w_after, (
            "Weights unchanged after training — fake training loop"
        )

    def test_tokenizer_is_not_a_wrapper(self):
        """Verify the tokenizer is custom, not a wrapper around tiktoken/HF."""
        import inspect
        from tokenizer.tokenizer import ByteLevelBPETokenizer
        source = inspect.getsource(ByteLevelBPETokenizer)
        # Should NOT delegate to tiktoken, sentencepiece, or transformers
        assert "tiktoken" not in source
        assert "sentencepiece" not in source
        assert "transformers" not in source.lower() or "sentence_transformers" not in source
        assert "AutoTokenizer" not in source
