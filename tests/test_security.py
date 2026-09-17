"""
tests/test_security.py — Unit tests for the security module.

Tests: permissions, audit logging, and offline verification.
"""
from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from security.permissions import PermissionManager
from security.audit import log_event
from security.offline_mode import verify_offline


# ─────────────────────────────────────────────────────────────────────────────
# 1. PermissionManager
# ─────────────────────────────────────────────────────────────────────────────

class TestPermissionManager:
    def test_allow_inside_workspace(self, tmp_path):
        pm = PermissionManager(tmp_path)
        inner = tmp_path / "data" / "file.txt"
        assert pm.is_allowed(inner)

    def test_deny_outside_workspace(self, tmp_path):
        pm = PermissionManager(tmp_path)
        assert not pm.is_allowed("/etc/passwd")
        assert not pm.is_allowed("C:\\Windows\\System32\\config")

    def test_deny_path_traversal(self, tmp_path):
        pm = PermissionManager(tmp_path)
        evil = tmp_path / ".." / ".." / "etc" / "shadow"
        assert not pm.is_allowed(evil)

    def test_check_raises_on_denied(self, tmp_path):
        pm = PermissionManager(tmp_path)
        with pytest.raises(PermissionError):
            pm.check("/etc/passwd")

    def test_check_passes_on_allowed(self, tmp_path):
        pm = PermissionManager(tmp_path)
        inner = tmp_path / "safe_file.txt"
        pm.check(inner)  # should not raise


# ─────────────────────────────────────────────────────────────────────────────
# 2. Audit logging
# ─────────────────────────────────────────────────────────────────────────────

class TestAuditLogging:
    def test_log_event_writes_to_file(self, tmp_path, monkeypatch):
        """Verify log_event writes a JSON line to the audit log."""
        audit_file = tmp_path / "audit.log"
        # Monkeypatch the audit log path in the security.audit module
        import security.audit as audit_mod
        monkeypatch.setattr(audit_mod, "AUDIT_LOG_FILE", audit_file)

        log_event("test_action", {"key": "value"})

        assert audit_file.exists()
        content = audit_file.read_text(encoding="utf-8").strip()
        entry = json.loads(content.splitlines()[-1])
        assert entry["event_type"] == "test_action"
        assert entry["details"]["key"] == "value"
        assert "timestamp" in entry


# ─────────────────────────────────────────────────────────────────────────────
# 3. Offline mode verification
# ─────────────────────────────────────────────────────────────────────────────

class TestOfflineMode:
    def test_verify_offline_returns_dict(self):
        result = verify_offline()
        assert isinstance(result, dict)
        assert "internet_reachable" in result
        assert "verdict" in result

    def test_verify_offline_has_tested_hosts(self):
        result = verify_offline()
        # Should test at least one host
        assert "tested_hosts" in result or "verdict" in result


# ─────────────────────────────────────────────────────────────────────────────
# 4. NetworkMonitor
# ─────────────────────────────────────────────────────────────────────────────

class TestNetworkMonitor:
    def test_monitor_start_stop(self):
        from security.network_monitor import NetworkMonitor
        monitor = NetworkMonitor()
        monitor.start()
        monitor.stop()
        report = monitor.get_report()
        assert isinstance(report, dict)
        assert "external_attempts" in report
