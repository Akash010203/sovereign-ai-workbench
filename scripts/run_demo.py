"""
scripts/run_demo.py — Convenience CLI runner for Sovereign AI Workbench demo scenarios.
Delegates directly to demo/run_demo.py.
"""
import sys
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

if __name__ == "__main__":
    demo_script = ROOT / "demo" / "run_demo.py"
    cmd = [sys.executable, str(demo_script)] + sys.argv[1:]
    sys.exit(subprocess.call(cmd, cwd=str(ROOT)))
