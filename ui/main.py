"""
ui/main.py
==========
Sole server entry point. Validates the API key and starts the NiceGUI server.

Usage:
    python ui/main.py
"""

import os
import sys

# ── Ensure project root is on sys.path before any ui.* imports ────────────────
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from dotenv import load_dotenv

from ui.config import ROOT

load_dotenv(os.path.join(ROOT, ".env"))

# ── API key guard ─────────────────────────────────────────────────────────────
api_key = os.environ.get("NVIDIA_API_KEY") or os.environ.get("OPENAI_API_KEY") or ""
if not api_key or "your" in api_key.lower():
    print("\n[ERROR] API key missing — add NVIDIA_API_KEY or OPENAI_API_KEY to .env\n")
    sys.exit(1)

# ── Register the page route ───────────────────────────────────────────────────
import ui.app  # noqa: F401, E402  — side-effect: registers @ui.page("/")

from nicegui import ui  # noqa: E402

# ── Start the server ──────────────────────────────────────────────────────────
if __name__ in ("__main__", "__mp_main__"):
    print("\n  IT Helpdesk Agent Benchmark UI")
    print("  http://localhost:8000\n")
    try:
        ui.run(
            title="IT Helpdesk Agent Benchmark",
            favicon="🖥️",
            port=8000,
            reload=False,
            dark=False,
            storage_secret="itbenchmark_storage_secret_change_me_in_production",
            reconnect_timeout=600,  # seconds — keeps the session alive while tab is in background
        )
    except KeyboardInterrupt:
        print("\n  Server stopped.")
