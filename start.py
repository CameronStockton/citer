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
GROBID_SETUP_MARKER = ROOT / ".grobid_setup_done"


def run():
    print("[1/4] Installing Python deps...")
    if not DEPS_MARKER.exists():
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-r", str(ROOT / "requirements.txt")])
        DEPS_MARKER.touch()
    else:
        print("  - Dependencies already installed (skipping). Delete .deps_installed to force reinstall.")

    print("[2/4] Downloading Grobid + JRE...")
    grobid_first_setup = not GROBID_SETUP_MARKER.exists()
    if not GROBID_DIR.exists() or grobid_first_setup:
        subprocess.check_call([sys.executable, str(ROOT / "scripts" / "grobid_bootstrap.py"), "setup"])
        GROBID_SETUP_MARKER.touch()
    else:
        print("  - Grobid runtime already present (skipping setup).")

    print("[3/4] Starting Grobid...")
    grobid_available = True
    try:
        grobid_proc = subprocess.Popen([sys.executable, str(ROOT / "scripts" / "grobid_bootstrap.py"), "start"])
        warmup = 120 if grobid_first_setup else 20
        print(f"  - Waiting ~{warmup} seconds for Grobid warm-up...")
        time.sleep(warmup)
        if grobid_proc.poll() is not None:
            print("  ! Grobid exited unexpectedly during warm-up; continuing without Grobid.")
            grobid_available = False
    except Exception as exc:  # noqa: BLE001
        print(f"  ! Grobid failed to start ({exc}); continuing without Grobid.")
        grobid_available = False

    print("[4/4] Starting Application...")
    app_env = os.environ.copy()
    if not grobid_available:
        # Signal the app to disable Grobid usage and fall back to local parsing.
        app_env["GROBID_DISABLED"] = "1"
    app_proc = subprocess.Popen([sys.executable, str(ROOT / "app.py")], env=app_env)

    # Give the app a moment to start, then open the browser.
    time.sleep(6)
    try:
        webbrowser.open("http://127.0.0.1:8888")
    except Exception:
        pass

    app_proc.wait()


if __name__ == "__main__":
    run()
