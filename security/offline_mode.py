"""
security/offline_mode.py — Offline mode enforcement and verification.
"""
from __future__ import annotations

import logging
from pathlib import Path

log = logging.getLogger(__name__)
OFFLINE_PROOF_LOG = Path("logs") / "offline_proof.log"


def verify_offline() -> dict:
    """
    Run the full offline verification suite and log results.

    Returns a report dict suitable for display in the UI.
    """
    from security.network_monitor import check_internet
    connectivity = check_internet(timeout=1.5)

    import json
    from datetime import datetime, timezone
    is_offline = not connectivity["connected"]
    report = {
        "timestamp":         datetime.now(timezone.utc).isoformat(),
        "offline":           is_offline,
        "internet_reachable": connectivity["connected"],
        "tested_hosts":      connectivity["tested_hosts"],
        "verdict":           "NO OUTBOUND PROBE" if is_offline else "ONLINE",
    }

    # Write proof log
    OFFLINE_PROOF_LOG.parent.mkdir(parents=True, exist_ok=True)
    with OFFLINE_PROOF_LOG.open("a", encoding="utf-8") as f:
        f.write(json.dumps(report) + "\n")

    log.info("Offline verification: %s", report["verdict"])
    return report
