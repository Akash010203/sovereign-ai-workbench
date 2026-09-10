"""scripts/start_backend.py — Start the SovereignAI Flask backend."""
from __future__ import annotations

import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

os.chdir(ROOT)   # ensure relative paths resolve from project root

from app.backend.app import create_app
import logging

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)

app = create_app()
port = int(os.environ.get("SOVEREIGN_PORT", 5000))
print(f"\n{'='*60}")
print(f"  SovereignAI Workbench — Local Backend")
print(f"  URL: http://localhost:{port}")
print(f"  Press Ctrl+C to stop")
print(f"{'='*60}\n")
app.run(host="127.0.0.1", port=port, debug=False, use_reloader=False)
