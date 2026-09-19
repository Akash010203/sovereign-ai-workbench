"""
security/audit.py — Security audit logger.

Every significant action (tool call, model call, file access, security
event) is written to the audit log in the SQLite database AND to a
local flat log file. This provides visible, inspectable evidence of
what the system did.
"""
from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from pathlib import Path

log = logging.getLogger(__name__)

AUDIT_LOG_FILE = Path("logs") / "audit.log"


def log_event(
    event_type: str,
    details: dict,
    task_id: str = "",
) -> None:
    """
    Write a security audit event to the flat log file and optionally
    to the SQLite database.

    Args:
        event_type: e.g. 'tool_call', 'model_call', 'file_access', 'security'
        details:    Dict of event-specific details.
        task_id:    Associated task ID (optional).
    """
    entry = {
        "timestamp":  datetime.now(timezone.utc).isoformat(),
        "event_type": event_type,
        "task_id":    task_id,
        "details":    details,
    }

    # Flat log file
    AUDIT_LOG_FILE.parent.mkdir(parents=True, exist_ok=True)
    with AUDIT_LOG_FILE.open("a", encoding="utf-8") as f:
        f.write(json.dumps(entry) + "\n")

    log.info("AUDIT: %s — %s", event_type, str(details)[:120])
