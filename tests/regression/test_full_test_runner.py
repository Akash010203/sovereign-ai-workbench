"""Regression tests for the master validation runner."""
from pathlib import Path


def test_master_runner_uses_console_safe_status_markers():
    """The runner must not crash on Windows consoles configured as cp1252."""
    runner = Path(__file__).resolve().parents[2] / "scripts" / "run_full_test.py"
    source = runner.read_text(encoding="utf-8")

    assert 'status_icon = "PASS"' in source
    assert '"✓"' not in source
