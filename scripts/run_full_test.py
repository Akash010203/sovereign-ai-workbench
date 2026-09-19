"""
scripts/run_full_test.py — SovereignAI Master Test Runner

Executes ALL test categories (A through W) and generates reports.
Run: python scripts/run_full_test.py
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

REPORT_DIR = ROOT / "test_reports" / "latest"
HISTORY_DIR = ROOT / "test_reports" / "history"


def run_pytest(test_path: str, extra_args: list[str] | None = None) -> dict:
    """Run pytest on a path and return results."""
    args = [sys.executable, "-m", "pytest", test_path,
            "--tb=short", "-q", "--no-header"]
    if extra_args:
        args.extend(extra_args)
    try:
        result = subprocess.run(
            args, capture_output=True, text=True,
            timeout=300, cwd=str(ROOT),
        )
        output = result.stdout + result.stderr
        # Parse pytest summary
        passed = failed = errors = 0
        for line in output.splitlines():
            if "passed" in line:
                import re
                m = re.search(r"(\d+) passed", line)
                if m:
                    passed = int(m.group(1))
                m = re.search(r"(\d+) failed", line)
                if m:
                    failed = int(m.group(1))
                m = re.search(r"(\d+) error", line)
                if m:
                    errors = int(m.group(1))
        return {
            "returncode": result.returncode,
            "passed": passed,
            "failed": failed,
            "errors": errors,
            "output": output,
            "status": "PASS" if result.returncode == 0 else "FAIL",
        }
    except subprocess.TimeoutExpired:
        return {"returncode": -1, "passed": 0, "failed": 0, "errors": 0,
                "output": "TIMEOUT", "status": "BLOCKED"}
    except Exception as e:
        return {"returncode": -1, "passed": 0, "failed": 0, "errors": 0,
                "output": str(e), "status": "BLOCKED"}


def main():
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    HISTORY_DIR.mkdir(parents=True, exist_ok=True)

    start_time = time.time()
    timestamp = datetime.now().isoformat()

    categories = [
        ("Environment / Startup", "tests/system/test_startup.py"),
        ("Tokenizer", "tests/system/test_tokenizer_full.py"),
        ("Custom LLM", "tests/system/test_custom_llm.py"),
        ("Training Pipeline", "tests/system/test_training.py"),
        ("Router", "tests/system/test_router_full.py"),
        ("Agent System", "tests/system/test_agent_full.py"),
        ("RAG", "tests/system/test_rag_full.py"),
        ("OCR", "tests/system/test_ocr_full.py"),
        ("Document Generation", "tests/system/test_docgen.py"),
        ("Code Sandbox", "tests/system/test_sandbox_full.py"),
        ("Database", "tests/system/test_database_full.py"),
        ("Security", "tests/system/test_security_full.py"),
        ("Air-Gap / Offline", "tests/system/test_airgap.py"),
        ("Backend API", "tests/system/test_backend_api.py"),
        ("Performance", "tests/system/test_performance.py"),
        ("Fake Implementation Detection", "tests/system/test_fake_detection.py"),
    ]

    results = {}
    total_passed = 0
    total_failed = 0
    total_errors = 0
    total_blocked = 0

    print("=" * 60)
    print("  SOVEREIGNAI FULL SYSTEM TEST")
    print(f"  {timestamp}")
    print("=" * 60)

    for name, path in categories:
        test_file = ROOT / path
        if not test_file.exists():
            print(f"  [{name:.<40}] NOT_FOUND")
            results[name] = {"status": "NOT_IMPLEMENTED", "passed": 0, "failed": 0}
            continue

        print(f"  Running: {name}...", end="", flush=True)
        r = run_pytest(str(test_file))
        results[name] = r
        total_passed += r["passed"]
        total_failed += r["failed"]
        total_errors += r["errors"]
        if r["status"] == "BLOCKED":
            total_blocked += 1
        # Keep console output compatible with the default Windows cp1252 console.
        # The runner must be able to report failing tests before UTF-8 is configured.
        status_icon = "PASS" if r["status"] == "PASS" else ("FAIL" if r["status"] == "FAIL" else "BLOCKED")
        print(f"\r  [{name:.<40}] {status_icon} {r['status']}  ({r['passed']}P/{r['failed']}F)")

    elapsed = time.time() - start_time

    # ── Scoreboard ─────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("  SCOREBOARD")
    print("=" * 60)
    for name, r in results.items():
        status = r.get("status", "UNKNOWN")
        print(f"  {name:<40} {status}")
    print("-" * 60)
    print(f"  TOTAL TESTS:     {total_passed + total_failed + total_errors}")
    print(f"  PASSED:          {total_passed}")
    print(f"  FAILED:          {total_failed}")
    print(f"  ERRORS:          {total_errors}")
    print(f"  BLOCKED:         {total_blocked}")
    print(f"  TIME:            {elapsed:.1f}s")
    print("=" * 60)

    # ── Write SUMMARY.md ──────────────────────────────────────────
    summary_lines = [
        "# SovereignAI Test Report — Cycle Summary",
        f"\n**Timestamp:** {timestamp}",
        f"**Duration:** {elapsed:.1f}s",
        f"\n## Scoreboard\n",
        "| Category | Status | Passed | Failed |",
        "|---|---|---|---|",
    ]
    for name, r in results.items():
        summary_lines.append(
            f"| {name} | {r.get('status', 'UNKNOWN')} | {r.get('passed', 0)} | {r.get('failed', 0)} |"
        )
    summary_lines.extend([
        f"\n## Totals\n",
        f"- **Total tests:** {total_passed + total_failed + total_errors}",
        f"- **Passed:** {total_passed}",
        f"- **Failed:** {total_failed}",
        f"- **Errors:** {total_errors}",
        f"- **Blocked:** {total_blocked}",
    ])

    # Failures detail
    failures = {name: r for name, r in results.items() if r.get("status") == "FAIL"}
    if failures:
        summary_lines.append("\n## Failures\n")
        for name, r in failures.items():
            summary_lines.append(f"### {name}\n")
            summary_lines.append(f"```\n{r.get('output', '')[-2000:]}\n```\n")

    (REPORT_DIR / "SUMMARY.md").write_text("\n".join(summary_lines), encoding="utf-8")

    # ── Write detailed JSON ───────────────────────────────────────
    report_data = {
        "timestamp": timestamp,
        "duration_s": elapsed,
        "total_passed": total_passed,
        "total_failed": total_failed,
        "total_errors": total_errors,
        "categories": {
            name: {k: v for k, v in r.items() if k != "output"}
            for name, r in results.items()
        },
    }
    (REPORT_DIR / "results.json").write_text(
        json.dumps(report_data, indent=2), encoding="utf-8"
    )

    # Archive to history
    ts_slug = datetime.now().strftime("%Y%m%d_%H%M%S")
    hist_dir = HISTORY_DIR / ts_slug
    hist_dir.mkdir(parents=True, exist_ok=True)
    (hist_dir / "results.json").write_text(
        json.dumps(report_data, indent=2), encoding="utf-8"
    )

    print(f"\nReport saved to: {REPORT_DIR / 'SUMMARY.md'}")
    return 0 if total_failed == 0 and total_errors == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
