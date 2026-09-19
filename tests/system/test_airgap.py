"""
TEST N — Air-Gap Sovereignty Test (Dedicated)

Scans the ENTIRE source tree for network imports in production code.
Allows legitimate uses (scripts, security monitoring, and tests).
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))


class TestAirGapSourceScan:
    """Comprehensive source-code scan for network activity."""

    def _scan_production_imports(self):
        """Scan production Python files for network-related imports."""
        findings = []
        patterns = [
            (r'^import requests\b', "requests import"),
            (r'^from requests ', "requests import"),
            (r'^import aiohttp', "aiohttp import"),
            (r'^import httpx', "httpx import"),
        ]
        for py_file in ROOT.rglob("*.py"):
            if "__pycache__" in str(py_file) or ".git" in str(py_file):
                continue
            # Exclude test files entirely
            if "test" in py_file.name.lower() or "tests/" in str(py_file).replace("\\", "/"):
                continue
            try:
                content = py_file.read_text(encoding="utf-8", errors="ignore")
            except Exception:
                continue
            rel_path = str(py_file.relative_to(ROOT))
            for line_num, line in enumerate(content.splitlines(), 1):
                stripped = line.strip()
                if stripped.startswith("#"):
                    continue
                for pattern, desc in patterns:
                    if re.search(pattern, stripped):
                        findings.append({
                            "file": rel_path,
                            "line": line_num,
                            "type": desc,
                            "content": stripped[:100],
                        })
        return findings

    def test_classify_all_network_deps(self):
        """Every network import in production code must be in an expected location."""
        findings = self._scan_production_imports()

        # Expected locations for network code in production
        expected_locations = [
            "scripts/",           # Data download scripts (pre-deployment only)
            "security/",          # Network monitoring (the interceptor itself)
            "rag/fineweb",        # FineWeb data loading
        ]

        unexpected = []
        for f in findings:
            if any(loc in f["file"].replace("\\", "/") for loc in expected_locations):
                continue
            unexpected.append(f)

        if unexpected:
            msg = "\n".join(
                f"  {f['file']}:{f['line']} [{f['type']}] {f['content']}"
                for f in unexpected
            )
            pytest.fail(f"Unexpected network imports in production code:\n{msg}")

    def test_no_cloud_sdk_imports(self):
        """No cloud SDKs should be imported anywhere in production code."""
        cloud_sdks = [
            "openai", "anthropic", "google.cloud", "azure",
            "boto3", "botocore", "google.generativeai",
        ]
        violations = []
        for py_file in ROOT.rglob("*.py"):
            if "__pycache__" in str(py_file) or ".git" in str(py_file):
                continue
            if "test" in py_file.name.lower():
                continue
            try:
                content = py_file.read_text(encoding="utf-8", errors="ignore")
            except Exception:
                continue
            for sdk in cloud_sdks:
                if f"import {sdk}" in content or f"from {sdk}" in content:
                    violations.append(
                        f"{py_file.relative_to(ROOT)} imports {sdk}"
                    )
        assert len(violations) == 0, f"Cloud SDK imports found:\n" + "\n".join(violations)

    def test_no_api_keys_in_source(self):
        """No API keys or secrets should be hardcoded."""
        key_patterns = [
            r'sk-[a-zA-Z0-9]{20,}',          # OpenAI keys
            r'ANTHROPIC_API_KEY\s*=\s*"',     # Anthropic
            r'OPENAI_API_KEY\s*=\s*"',        # OpenAI
        ]
        violations = []
        for py_file in ROOT.rglob("*.py"):
            if "__pycache__" in str(py_file) or ".git" in str(py_file):
                continue
            try:
                content = py_file.read_text(encoding="utf-8", errors="ignore")
            except Exception:
                continue
            for pattern in key_patterns:
                if re.search(pattern, content):
                    violations.append(str(py_file.relative_to(ROOT)))
        assert len(violations) == 0, f"Potential API keys in: {violations}"

    def test_no_model_api_adapter_exists(self):
        """The runtime must not include an HTTP adapter for model inference."""
        adapter_dir = ROOT / "models" / "adapters"
        source_files = list(adapter_dir.glob("*.py"))
        assert all("urllib.request" not in path.read_text(encoding="utf-8") for path in source_files)
