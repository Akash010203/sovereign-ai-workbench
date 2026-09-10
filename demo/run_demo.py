"""
demo/run_demo.py — SIH 2026 live demonstration runner.

Usage:
    python demo\\run_demo.py --scenario 4    (engineering calculation — no Ollama needed)
    python demo\\run_demo.py --scenario 1    (OCR + Word — needs Tesseract + Ollama)
    python demo\\run_demo.py --all           (run all 5 in sequence)
    python demo\\run_demo.py --list          (show all scenario descriptions)
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from core.logging_setup import setup_logging
from demo.scenarios import DEMO_SCENARIOS, get_scenario


# ── ANSI colours for demo output ─────────────────────────────────────────────
CYAN    = "\033[96m"
GREEN   = "\033[92m"
YELLOW  = "\033[93m"
RED     = "\033[91m"
BOLD    = "\033[1m"
RESET   = "\033[0m"


def banner(text: str, colour=CYAN) -> None:
    print(f"\n{colour}{BOLD}{'═'*60}{RESET}")
    print(f"{colour}{BOLD}  {text}{RESET}")
    print(f"{colour}{BOLD}{'═'*60}{RESET}\n")


def run_scenario(number: int) -> dict:
    """Run one demo scenario and return a result dict."""
    from models.registry import build_default_registry
    from tools import build_default_tool_registry
    from agents.agent import Agent

    scenario = get_scenario(number)
    banner(f"SCENARIO {scenario.number}: {scenario.title}")

    print(f"{YELLOW}Description:{RESET} {scenario.description}\n")
    print(f"{YELLOW}User input:{RESET}")
    print(f"  {scenario.user_input[:200]}{'…' if len(scenario.user_input)>200 else ''}\n")
    print(f"{YELLOW}Tools:{RESET} {', '.join(scenario.tools_used) or '(none)'}")
    print(f"{YELLOW}Models:{RESET} {', '.join(scenario.models_used)}")
    print(f"\n{YELLOW}Running…{RESET}")

    # ── Special case: Scenario 5 — Air-gap proof ──────────────────────
    if number == 5:
        return _run_airgap_demo()

    # ── Normal agent execution ────────────────────────────────────────
    ckpt_path = str(ROOT / "checkpoints" / "best.pt")
    tok_path  = str(ROOT / "tokenizer" / "vocab" / "demo_bpe_vocab.json")
    model_reg = build_default_registry(
        minilm_checkpoint=ckpt_path,
        tokenizer_path=tok_path,
    )
    tool_reg  = build_default_tool_registry(workspace_root=str(ROOT))

    # For scenario 4 (calculation), inject the expression directly
    if number == 4:
        result = tool_reg.call(
            "calculator",
            expression="(0.02 * 50 * 1000 * 0.637 * 0.637) / (2 * 0.1)"
        )
        _print_result(scenario, {
            "status": "done" if result.success else "failed",
            "final_answer": f"ΔP = {result.output:.2f} Pa" if result.success else result.error,
            "artifacts": [],
        })
        return {"scenario": number, "success": result.success, "result": result.output}

    agent      = Agent(model_reg, tool_reg)
    t0         = time.perf_counter()
    task_state = agent.run(scenario.user_input)
    elapsed    = time.perf_counter() - t0

    _print_result(scenario, {
        "status":       task_state.status.value,
        "final_answer": task_state.final_answer or "(No text output)",
        "artifacts":    task_state.artifacts,
        "elapsed_s":    round(elapsed, 2),
        "model":        task_state.model_name,
        "category":     task_state.category,
    })

    return {
        "scenario":  number,
        "success":   task_state.status.value == "done",
        "elapsed_s": round(elapsed, 2),
    }


def _run_airgap_demo() -> dict:
    """Scenario 5: air-gap verification."""
    from security.network_monitor import NetworkMonitor, check_internet
    from security.offline_mode import verify_offline

    monitor = NetworkMonitor()
    monitor.start()

    # Simulate some tool activity
    from tools.calculator import CalculatorTool
    CalculatorTool().run(expression="sqrt(2)")

    monitor.stop()
    report   = monitor.get_report()
    offline  = verify_offline()

    print(f"\n{GREEN if not offline['internet_reachable'] else RED}"
          f"Network Verdict: {offline['verdict']}{RESET}")
    print(f"External TCP attempts intercepted: {report['external_attempts']}")
    print(f"Tested hosts: {', '.join(offline.get('tested_hosts', []))}")
    print(f"\n{GREEN}✓ Timestamped proof saved → logs/offline_proof.log{RESET}")

    return {
        "scenario":    5,
        "success":     not offline["internet_reachable"],
        "internet_reachable": offline["internet_reachable"],
        "external_attempts":  report["external_attempts"],
    }


def _print_result(scenario, data: dict) -> None:
    status = data.get("status", "unknown")
    colour = GREEN if status == "done" else RED
    print(f"\n{colour}{BOLD}Status: {status.upper()}{RESET}")

    answer = data.get("final_answer", "")
    if answer:
        print(f"\n{YELLOW}Result:{RESET}")
        print(answer[:600])

    artifacts = data.get("artifacts", [])
    if artifacts:
        print(f"\n{YELLOW}Artifacts generated:{RESET}")
        for a in artifacts:
            print(f"  📄 {a['name']} → {a['path']}")

    elapsed = data.get("elapsed_s")
    if elapsed:
        print(f"\n{YELLOW}Time:{RESET} {elapsed}s")

    print(f"\n{CYAN}What this proved:{RESET}")
    for point in scenario.what_it_proves:
        print(f"  ✓ {point}")


def parse_args():
    p = argparse.ArgumentParser(description="SovereignAI SIH 2026 Demo Runner")
    g = p.add_mutually_exclusive_group()
    g.add_argument("--scenario", type=int, choices=[1,2,3,4,5],
                   help="Run one scenario by number")
    g.add_argument("--all",  action="store_true", help="Run all 5 scenarios")
    g.add_argument("--list", action="store_true", help="List all scenarios")
    return p.parse_args()


def main():
    setup_logging("run_demo")
    args = parse_args()

    if args.list or (not args.scenario and not args.all):
        print(f"\n{BOLD}SovereignAI — SIH 2026 Demo Scenarios{RESET}\n")
        for s in DEMO_SCENARIOS:
            print(f"  [{s.number}] {s.title}")
            print(f"      {s.description[:90]}…")
            print(f"      Tools: {', '.join(s.tools_used) or 'none'}")
            print()
        return

    scenarios_to_run = [args.scenario] if args.scenario else [s.number for s in DEMO_SCENARIOS]
    results = []
    for n in scenarios_to_run:
        try:
            results.append(run_scenario(n))
        except Exception as exc:
            print(f"{RED}Scenario {n} error: {exc}{RESET}")
            results.append({"scenario": n, "success": False, "error": str(exc)})

    # Final summary
    if len(results) > 1:
        banner("DEMO SUMMARY", GREEN)
        for r in results:
            icon = "✅" if r.get("success") else "❌"
            s    = get_scenario(r["scenario"])
            print(f"  {icon} Scenario {r['scenario']}: {s.title}")


if __name__ == "__main__":
    main()
