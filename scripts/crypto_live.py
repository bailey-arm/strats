#!/usr/bin/env python3
"""
Crypto live launcher.
Opens two Terminal windows (prices + charts) and starts the data daemon.

Usage:  python3 scripts/crypto_live.py
        python3 scripts/crypto_live.py  # from any dir if added to $PATH
"""

import os
import subprocess
import sys
import time
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent

# Use the venv python if available, otherwise the current interpreter
_venv = SCRIPT_DIR.parent / "venv" / "bin" / "python3"
PYTHON = str(_venv) if _venv.exists() else sys.executable


def _open_window(script: str, title: str, x: int, y: int, w: int, h: int) -> None:
    """Open a new macOS Terminal window running a script."""
    cmd_str = f"{PYTHON} {SCRIPT_DIR / script}"
    apple = f"""
    tell application "Terminal"
        set w to do script "{cmd_str}"
        set custom title of front window to "{title}"
        set bounds of front window to {{{x}, {y}, {x + w}, {y + h}}}
    end tell
    """
    subprocess.run(["osascript", "-e", apple], check=False)
    time.sleep(0.6)   # let Terminal finish opening before next window


def main() -> None:
    print("Starting crypto data daemon …")
    daemon = subprocess.Popen(
        [PYTHON, str(SCRIPT_DIR / "crypto_data.py")],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    time.sleep(1.2)   # give daemon time to connect and write first state

    # Side-by-side windows — tweak x/y/w/h to taste
    WIN_W, WIN_H = 760, 820
    TOP          = 50

    _open_window("crypto_prices.py", "Crypto  ·  Prices",  0,       TOP, WIN_W, WIN_H)
    _open_window("crypto_charts.py", "Crypto  ·  Charts",  WIN_W,   TOP, WIN_W, WIN_H)

    print(f"Launched. Data daemon PID: {daemon.pid}")
    print("Close the Terminal windows and press Ctrl+C here to stop the daemon.\n")

    try:
        daemon.wait()
    except KeyboardInterrupt:
        daemon.terminate()
        print("Stopped.")


if __name__ == "__main__":
    main()
