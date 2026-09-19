# Security Architecture, Permissions & Audit Logging

Security in the **Sovereign AI Workbench** is treated as a foundational architectural tier rather than a cosmetic feature. The workbench enforces least-privilege tool execution, comprehensive audit logging, and strict offline guarantees.

---

## 1. Security Subsystem Architecture

The security subsystem lives under `security/`:

```
security/
├── permissions.py      # Role-based tool access control & privilege verification
├── audit.py            # Structured append-only audit trail logger
├── offline_mode.py     # Offline verification routines & air-gap reporting
└── network_monitor.py  # Runtime socket interception & external egress detector
```

---

## 2. Component Breakdown

### A. Permissions Control (`security/permissions.py`)
Enforces strict boundaries around sensitive tools (e.g., `code_sandbox`, `filesystem_delete`, `shell_exec`).
- Execution is governed by explicit permission profiles: `READ_ONLY`, `OPERATOR`, and `ADMIN`.
- Dangerous operations (such as code evaluation or filesystem writes outside designated work folders) require explicit authorization.

### B. Structured Audit Logging (`security/audit.py`)
Every critical event is logged to a persistent SQLite database table and append-only JSONL files:
- Timestamp (UTC ISO-8601)
- User ID / Session ID
- Requested task & routing decision
- Tool invoked with arguments
- Security verdict (ALLOW / DENY)
- Execution outcome & latency

### C. Network Monitor (`security/network_monitor.py`)
At application startup, `NetworkMonitor` monkey-patches `socket.connect` to intercept outgoing TCP attempts.
- Connections to `127.0.0.1`, `localhost`, and `::1` for the local web application are permitted.
- Any attempt to reach an external non-localhost IP or domain triggers an immediate security alert and is recorded in `logs/network.log`.

### D. Offline Mode Verification (`security/offline_mode.py`)
Provides deterministic programmatic verification that the system is operating without internet connectivity.
```powershell
python scripts/verify_offline.py
```
Outputs a signed JSON report proving zero internet reachability and verifying system isolation.
