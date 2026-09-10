"""
security/network_monitor.py — Detect and log external network connections.

This module provides VISIBLE EVIDENCE that the system is not making
external network calls at runtime. This is a core SIH 117 requirement:
the demo must show proof of offline/air-gap operation.

HOW IT WORKS
------------
1. ``check_internet()``  — tries to connect to well-known external hosts
   and reports whether the attempt succeeds.  In normal offline mode,
   it should fail (no internet).

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
from datetime import datetime
from pathlib import Path
from typing import Optional

log = logging.getLogger(__name__)

NETWORK_LOG_FILE = Path("logs") / "network.log"

# Addresses that ARE allowed (the Ollama local server)
_ALLOWED_HOSTS = {"127.0.0.1", "localhost", "::1"}


def check_internet(timeout: float = 2.0) -> dict:
    """
    Attempt to reach external hosts and report connectivity status.

    Args:
        timeout: Seconds to wait before declaring no connection.

    Returns:
        dict with:
            connected (bool): True if any external host was reachable.
            tested_hosts (list): Hosts that were tried.
            result (str): Human-readable summary.
    """
    test_hosts = [
        ("8.8.8.8", 53),       # Google DNS
        ("1.1.1.1", 53),       # Cloudflare DNS
        ("api.openai.com", 443),
    ]
    connected = False
    for host, port in test_hosts:
        try:
            s = socket.create_connection((host, port), timeout=timeout)
            s.close()
            connected = True
            log.warning("NETWORK ALERT: external host reachable: %s:%d", host, port)
            break
        except (OSError, socket.timeout):
            pass

    result = (
        "WARNING: External internet is reachable. "
        "Ensure the machine is air-gapped for production use."
        if connected
        else "OFFLINE CONFIRMED: No external hosts reachable. Air-gap is active."
    )
    return {
        "connected":     connected,
        "tested_hosts":  [f"{h}:{p}" for h, p in test_hosts],
        "result":        result,
        "timestamp":     datetime.utcnow().isoformat(),
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
                    "timestamp":   datetime.utcnow().isoformat(),
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
