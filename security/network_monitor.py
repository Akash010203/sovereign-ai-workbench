"""
security/network_monitor.py — Detect and log external network connections.

This module provides VISIBLE EVIDENCE that the system is not making
external network calls at runtime. This is a core SIH 117 requirement:
the demo must show proof of offline/air-gap operation.

HOW IT WORKS
------------
1. ``check_internet()``  — performs a passive policy check. It never probes
   external hosts, because attempting a connection would violate the air-gap
   policy it is intended to validate.

2. ``NetworkMonitor``  — patches ``socket.connect`` at import time
   to intercept ALL outgoing TCP connection attempts.  Any connection
   to a non-localhost address is logged as a security alert.

WHAT IT PROVES
--------------
If you run the full application with NetworkMonitor active and the
audit log shows zero external connection attempts → the air-gap claim
is demonstrated.

HONESTY NOTE: ``socket`` patching intercepts Python-level TCP connections.
It does NOT intercept connections made by subprocess, C extensions
making raw OS syscalls, or network activity outside this Python process.
For production air-gap assurance, OS-level network namespaces or
a firewall policy would be needed.  This is sufficient for the SIH demo.
"""
from __future__ import annotations

import logging
import socket
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

log = logging.getLogger(__name__)

NETWORK_LOG_FILE = Path("logs") / "network.log"

# Addresses that are allowed for local application communication.
_ALLOWED_HOSTS = {"127.0.0.1", "localhost", "::1"}


def check_internet(timeout: float = 2.0) -> dict:
    """
    Report the runtime's passive no-egress policy without opening a socket.

    Args:
        timeout: Retained for backward-compatible callers; never used.

    Returns:
        dict with:
            connected (bool): Always False: no external host is contacted.
            tested_hosts (list): Always empty.
            result (str): Human-readable summary.
    """
    return {
        "connected":     False,
        "tested_hosts":  [],
        "result":        "PASSIVE CHECK: no external network probe was made.",
        "timestamp":     datetime.now(timezone.utc).isoformat(),
    }


class NetworkMonitor:
    """
    Intercepts all outgoing Python socket connections and logs
    any attempt to reach a non-localhost address.

    Usage:
        monitor = NetworkMonitor()
        monitor.start()
        # ... run the application ...
        monitor.stop()
        print(monitor.get_report())
    """

    def __init__(self) -> None:
        self._original_connect = None
        self._alerts: list[dict] = []
        self._active = False

    def start(self) -> None:
        """Patch socket.connect to intercept external connection attempts."""
        self._original_connect = socket.socket.connect
        monitor = self

        def patched_connect(sock_self, address):
            host = address[0] if isinstance(address, tuple) else str(address)
            if host not in _ALLOWED_HOSTS:
                alert = {
                    "timestamp":   datetime.now(timezone.utc).isoformat(),
                    "attempted":   str(address),
                    "event":       "EXTERNAL_CONNECTION_ATTEMPT",
                }
                monitor._alerts.append(alert)
                log.warning("NETWORK MONITOR: External connection attempt: %s", address)
                _write_network_log(alert)
            return monitor._original_connect(sock_self, address)

        socket.socket.connect = patched_connect
        self._active = True
        log.info("NetworkMonitor active — intercepting all socket.connect() calls.")

    def stop(self) -> None:
        """Restore the original socket.connect."""
        if self._original_connect and self._active:
            socket.socket.connect = self._original_connect
            self._active = False

    def get_report(self) -> dict:
        return {
            "active":           self._active,
            "external_attempts": len(self._alerts),
            "alerts":           self._alerts,
            "verdict": (
                "CLEAN — No external connections detected."
                if not self._alerts
                else f"WARNING — {len(self._alerts)} external connection attempt(s) detected."
            ),
        }


def _write_network_log(entry: dict) -> None:
    NETWORK_LOG_FILE.parent.mkdir(parents=True, exist_ok=True)
    import json
    with NETWORK_LOG_FILE.open("a", encoding="utf-8") as f:
        f.write(json.dumps(entry) + "\n")
