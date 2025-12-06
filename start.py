# run_all.py
import subprocess
import sys
import os
import webbrowser
import time
from pathlib import Path

ROOT = Path(__file__).parent.resolve()
DEPS_MARKER = ROOT / ".deps_installed"
GROBID_DIR = ROOT / "grobid_runtime" / "grobid"


def run():
    print("[1/4] Installing Python deps...")
    if not DEPS_MARKER.exists():
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-r", str(ROOT / "requirements.txt")])
        DEPS_MARKER.touch()
    else:
        print("  - Dependencies already installed (skipping). Delete .deps_installed to force reinstall.")

    print("[2/4] Downloading Grobid + JRE...")
    if not GROBID_DIR.exists():
        subprocess.check_call([sys.executable, str(ROOT / "scripts" / "grobid_bootstrap.py"), "setup"])
    else:
        print("  - Grobid runtime already present (skipping setup).")

    print("[3/4] Starting Grobid...")
    grobid_proc = subprocess.Popen([sys.executable, str(ROOT / "scripts" / "grobid_bootstrap.py"), "start"])
    grobid_first_start = not (GROBID_DIR.exists() and any(GROBID_DIR.iterdir()))
    warmup = 60 if grobid_first_start else 10
    time.sleep(warmup)

    print("[4/4] Starting Application...")
    app_proc = subprocess.Popen([sys.executable, str(ROOT / "app.py")])

    # Give the app a moment to start, then open the browser.
    time.sleep(3)
    try:
        webbrowser.open("http://127.0.0.1:8888")
    except Exception:
        pass

    app_proc.wait()


if __name__ == "__main__":
    run()
