#!/usr/bin/env python3
"""Terminal homepage — Bailey's tool suite."""

import os
import subprocess
import sys
from pathlib import Path

R    = "\033[0m"
BOLD = "\033[1m"
DIM  = "\033[2m"
CYAN = "\033[96m"
BLUE = "\033[94m"
WHITE = "\033[97m"

ROOT   = Path(__file__).resolve().parent
_venv  = ROOT / "venv" / "bin" / "python3"
PYTHON = str(_venv) if _venv.exists() else sys.executable

TOOLS = [
    ("Mental Math",        ROOT / "mental_math.py"),
    ("Trading Math Drill", ROOT / "scripts" / "trading_math_drill.py"),
    ("Quick Calc",         ROOT / "scripts" / "quick_calc.py"),
    ("Live Ticker Tape",   ROOT / "scripts" / "ticker_tape.py"),
    ("Vol Dashboard",      ROOT / "scripts" / "vol_dashboard.py"),
    ("Options Chain",      ROOT / "scripts" / "options_chain.py"),
    ("Sector Heatmap",     ROOT / "scripts" / "sector_heatmap.py"),
    ("News Tape",          ROOT / "scripts" / "news_tape.py"),
    ("Macro Dashboard",    ROOT / "scripts" / "macro_dashboard.py"),
    ("Crypto Live",        ROOT / "scripts" / "crypto_live.py"),
    ("Equities Snapshot",  ROOT / "scripts" / "equities_snapshot.py"),
    ("Market Brief",       ROOT / "scripts" / "market_brief_terminal.py"),
]


def clr():
    os.system("clear")


def box(text, color=BLUE, width=50):
    line = "─" * width
    pad  = " " * ((width - len(text)) // 2)
    print(f"{color}┌{line}┐{R}")
    print(f"{color}│{pad}{BOLD}{WHITE}{text}{R}{color}{' ' * (width - len(pad) - len(text))}│{R}")
    print(f"{color}└{line}┘{R}")


def menu():
    clr()
    box("  Bailey's Terminal  ")
    print()
    for i, (name, _) in enumerate(TOOLS, 1):
        print(f"  {CYAN}{BOLD}{i}{R}  {name}")
    print()
    print(f"  {CYAN}{BOLD}q{R}  {DIM}Quit{R}")
    print()
    while True:
        c = input("  > ").strip().lower()
        if c == "q":
            return "q"
        if c.isdigit() and 1 <= int(c) <= len(TOOLS):
            return c


def main():
    while True:
        choice = menu()
        if choice == "q":
            print(f"\n  {DIM}Bye!{R}\n")
            break
        _, script = TOOLS[int(choice) - 1]
        subprocess.run([PYTHON, str(script)])


if __name__ == "__main__":
    main()
