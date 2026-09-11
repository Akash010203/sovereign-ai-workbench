# Offline & Air-Gap Proof Methodology

The **Sovereign AI Workbench** is built specifically for secure, air-gapped environments (defense establishments, power plants, refineries, public sector undertakings) where data exfiltration is strictly prohibited.

---

## 1. The Core Principle: Evidence Over Assertion

Many AI systems claim "local" operation while secretly making telemetry calls, license validations, or model download requests to cloud servers.

Sovereign AI Workbench provides **visible, verifiable, and cryptographic proof** that no data leaves the machine:

```
Runtime Application Process
            │
            ▼
┌───────────────────────┐
│ Python socket.connect │  <── Monkey-patched by NetworkMonitor
└───────────┬───────────┘
            ├── Dest: 127.0.0.1 (Localhost / Ollama / DB)  ──► [ALLOW]
            └── Dest: External IP / Domain               ──► [LOG ALERT & BLOCK]
```

---

## 2. Verification Protocol for Evaluators & Judges

To prove complete offline operation during evaluation:

### Step 1: Disconnect Internet
Physically disconnect Ethernet and disable Wi-Fi on the host machine.

### Step 2: Run Verification Script
```powershell
python scripts/verify_offline.py
```
Expected output:
```
======================================================================
  SOVEREIGN AI WORKBENCH — OFFLINE & AIR-GAP VERIFICATION SUITE
======================================================================
[1/3] Testing outbound internet connectivity against standard DNS / Web hosts...
  Internet Reachable: False
  Verification Result: [PASS] System is air-gapped. Zero connectivity to external hosts.

[2/3] Checking active socket monitoring hook...
  Socket connect hook installed successfully.
  External non-localhost TCP attempts will be logged and intercepted.

[3/3] Inspecting offline audit trail log...
  Offline proof log verified: logs/offline_proof.log (Active)
======================================================================
```

### Step 3: Run Full End-to-End AI Tasks
Execute inference, document RAG, calculation, and multi-step agent scenarios:
```powershell
python scripts/run_demo.py --all
```

### Step 4: Audit Network Log
Inspect `logs/network.log` and `logs/offline_proof.log`.
Confirm zero external TCP connection attempts occurred during the entire run.
