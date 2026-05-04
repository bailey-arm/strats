#!/usr/bin/env python3
"""Mental Math Trainer — no external dependencies."""

import json
import os
import random
import time
from datetime import datetime
from pathlib import Path

LOG_FILE = Path.home() / ".mental_math_scores.json"
QUESTIONS = 10

# ── ANSI helpers ──────────────────────────────────────────────────────────────

R = "\033[0m"
BOLD = "\033[1m"
DIM = "\033[2m"
GREEN = "\033[92m"
YELLOW = "\033[93m"
RED = "\033[91m"
CYAN = "\033[96m"
WHITE = "\033[97m"
BLUE = "\033[94m"

def clr(): os.system("clear")

def box(text, color=BLUE, width=50):
    line = "─" * width
    pad = " " * ((width - len(text)) // 2)
    print(f"{color}┌{line}┐{R}")
    print(f"{color}│{pad}{BOLD}{WHITE}{text}{R}{color}{' ' * (width - len(pad) - len(text))}│{R}")
    print(f"{color}└{line}┘{R}")

def hline(color=DIM, width=52):
    print(f"{color}{'─' * width}{R}")

# ── difficulties ──────────────────────────────────────────────────────────────

DIFFICULTIES = {
    "1": {"name": "Easy",   "color": GREEN,  "ops": ["+", "-"],
          "ranges": {"+": (1, 20),  "-": (1, 20),  "×": (2, 9),  "÷": (2, 9)}},
    "2": {"name": "Medium", "color": YELLOW, "ops": ["+", "-", "×"],
          "ranges": {"+": (10, 100),"−": (10, 100),"×": (2, 12), "÷": (2, 12)}},
    "3": {"name": "Hard",   "color": RED,    "ops": ["+", "-", "×", "÷"],
          "ranges": {"+": (50, 500),"-": (50, 500),"×": (12, 50),"÷": (3, 20)}},
}

def make_question(cfg):
    op = random.choice(cfg["ops"])
    lo, hi = list(cfg["ranges"].values())[cfg["ops"].index(op) if op in cfg["ops"] else 0]
    # look up by symbol
    sym_map = {"+": "+", "-": "-", "×": "×", "÷": "÷"}
    range_key = op
    # find range from ranges dict (keys may use − or -)
    for k in cfg["ranges"]:
        if k.strip() in (op, op.replace("-","−")):
            lo, hi = cfg["ranges"][k]
            break

    if op == "+":
        a, b = random.randint(lo, hi), random.randint(lo, hi)
        return f"{a} + {b}", a + b
    elif op == "-":
        a, b = random.randint(lo, hi), random.randint(lo, hi)
        a, b = max(a, b), min(a, b)
        return f"{a} - {b}", a - b
    elif op == "×":
        a, b = random.randint(lo, hi), random.randint(lo, hi)
        return f"{a} × {b}", a * b
    else:
        b = random.randint(lo, hi)
        ans = random.randint(lo, hi)
        return f"{b * ans} ÷ {b}", ans

# ── persistence ───────────────────────────────────────────────────────────────

def load_log():
    if LOG_FILE.exists():
        try:
            return json.loads(LOG_FILE.read_text())
        except Exception:
            pass
    return []

def save_session(s):
    log = load_log()
    log.append(s)
    LOG_FILE.write_text(json.dumps(log, indent=2))

# ── screens ───────────────────────────────────────────────────────────────────

def main_menu():
    clr()
    box("  🧮  Mental Math Trainer  ")
    print()
    print(f"  {CYAN}{BOLD}1{R}  Start session")
    print(f"  {CYAN}{BOLD}2{R}  View history")
    print(f"  {CYAN}{BOLD}q{R}  Quit")
    print()
    while True:
        c = input("  Choose: ").strip().lower()
        if c in ("1", "2", "q"):
            return c

def difficulty_menu():
    clr()
    box("  Choose difficulty  ")
    print()
    for k, cfg in DIFFICULTIES.items():
        ops = "  ".join(cfg["ops"])
        print(f"  {CYAN}{BOLD}{k}{R}  {cfg['color']}{cfg['name']:8}{R}  {DIM}{ops}{R}")
    print(f"  {CYAN}{BOLD}b{R}  Back")
    print()
    while True:
        c = input("  Choose: ").strip().lower()
        if c == "b":
            return None
        if c in DIFFICULTIES:
            return DIFFICULTIES[c]

def show_header(cfg, q_num, correct, streak):
    c = cfg["color"]
    streak_str = f"🔥 {streak}" if streak >= 3 else str(streak)
    print(f"{c}{BOLD}  Mental Math  {R}"
          f"{DIM}│  Q {q_num}/{QUESTIONS}  │  ✓ {correct}  │  streak {streak_str}{R}")
    hline(c)

def show_history():
    clr()
    box("  Session History  ")
    log = load_log()
    if not log:
        print(f"\n  {DIM}No sessions yet.{R}\n")
        input("  Press Enter to continue...")
        return
    print()
    print(f"  {DIM}{'Date':<18} {'Diff':<8} {'Score':<8} {'Acc':<7} {'AvgTime'}{R}")
    hline()
    for s in log[-15:]:
        acc = f"{s['correct']/s['total']*100:.0f}%"
        col = GREEN if s['correct']/s['total'] >= 0.8 else YELLOW if s['correct']/s['total'] >= 0.5 else RED
        print(f"  {DIM}{s['date'][:16]:<18}{R}"
              f"{s['difficulty']:<8}"
              f"{col}{s['correct']}/{s['total']:<6}{R}"
              f"{col}{acc:<7}{R}"
              f"{DIM}{s['avg_time']:.1f}s{R}")
    print()
    input("  Press Enter to continue...")

def run_session(cfg):
    correct = 0
    streak = 0
    best_streak = 0
    times = []

    for q_num in range(1, QUESTIONS + 1):
        question, answer = make_question(cfg)

        clr()
        show_header(cfg, q_num, correct, streak)
        print()
        q_color = cfg["color"]
        print(f"  {q_color}{BOLD}  {question}  {R}")
        print()

        t0 = time.perf_counter()
        raw = input(f"  {WHITE}Answer (s=skip):{R} ").strip()
        elapsed = time.perf_counter() - t0
        times.append(elapsed)

        if raw.lower() == "s":
            print(f"\n  {DIM}Skipped — answer was {answer}{R}")
            streak = 0
            time.sleep(0.8)
            continue

        try:
            user_ans = int(raw)
        except ValueError:
            print(f"\n  {RED}Invalid — answer was {answer}{R}")
            streak = 0
            time.sleep(0.8)
            continue

        if user_ans == answer:
            correct += 1
            streak += 1
            best_streak = max(best_streak, streak)
            print(f"\n  {GREEN}{BOLD}✓  Correct!{R}  {DIM}({elapsed:.1f}s){R}")
        else:
            streak = 0
            print(f"\n  {RED}{BOLD}✗  Wrong — answer was {answer}{R}  {DIM}({elapsed:.1f}s){R}")

        time.sleep(0.9)

    # ── summary ───────────────────────────────────────────────────────────────
    clr()
    accuracy = correct / QUESTIONS * 100
    avg_time = sum(times) / len(times)
    grade_color = GREEN if accuracy >= 80 else YELLOW if accuracy >= 50 else RED

    box("  Session Complete  ", grade_color)
    print()
    print(f"  {'Difficulty':<16} {cfg['color']}{cfg['name']}{R}")
    print(f"  {'Score':<16} {grade_color}{BOLD}{correct} / {QUESTIONS}{R}")
    print(f"  {'Accuracy':<16} {grade_color}{accuracy:.0f}%{R}")
    print(f"  {'Best streak':<16} {'🔥 ' if best_streak >= 5 else ''}{best_streak}")
    print(f"  {'Avg time':<16} {DIM}{avg_time:.1f}s{R}")
    print()

    save_session({
        "date": datetime.now().isoformat(),
        "difficulty": cfg["name"],
        "correct": correct,
        "total": QUESTIONS,
        "best_streak": best_streak,
        "avg_time": avg_time,
    })
    print(f"  {DIM}Saved to {LOG_FILE}{R}\n")
    input("  Press Enter to continue...")

# ── main ──────────────────────────────────────────────────────────────────────

def main():
    while True:
        choice = main_menu()
        if choice == "q":
            print(f"\n  {DIM}Bye!{R}\n")
            break
        elif choice == "2":
            show_history()
        elif choice == "1":
            cfg = difficulty_menu()
            if cfg:
                run_session(cfg)

if __name__ == "__main__":
    main()
