#!/usr/bin/env python3
"""Trading Math Drill — bps, P&L, position sizing, Kelly, vol, Sharpe."""

import json
import math
import os
import random
import time
from datetime import datetime
from pathlib import Path

LOG_FILE = Path.home() / ".trading_math_scores.json"
QUESTIONS = 10

R      = "\033[0m"
BOLD   = "\033[1m"
DIM    = "\033[2m"
GREEN  = "\033[92m"
YELLOW = "\033[93m"
RED    = "\033[91m"
CYAN   = "\033[96m"
WHITE  = "\033[97m"
BLUE   = "\033[94m"
MAGENTA = "\033[95m"

def clr(): os.system("clear")

def box(text, color=BLUE, width=54):
    line = "─" * width
    pad  = " " * ((width - len(text)) // 2)
    print(f"{color}┌{line}┐{R}")
    print(f"{color}│{pad}{BOLD}{WHITE}{text}{R}{color}{' ' * (width - len(pad) - len(text))}│{R}")
    print(f"{color}└{line}┘{R}")

def hline(color=DIM, width=56):
    print(f"{color}{'─' * width}{R}")

# ── question generators ────────────────────────────────────────────────────────

def q_bps():
    """Dollar value of basis points on a notional."""
    notional_m = random.choice([1, 2, 5, 10, 25, 50, 100])
    bps = random.choice([1, 5, 10, 25, 50, 100, 200])
    answer = notional_m * 1_000_000 * bps / 10_000
    label = f"What is {bps}bps on ${notional_m}M?"
    hint  = f"notional × bps / 10,000 = ${notional_m}M × {bps} / 10,000"
    return label, answer, max(1, answer * 0.01), hint

def q_pct_move():
    """New price after a % move."""
    price = random.choice([50, 80, 100, 120, 150, 200, 250, 400, 500])
    pct   = round(random.uniform(1.0, 8.0) * random.choice([-1, 1]), 1)
    new_p = round(price * (1 + pct / 100), 2)
    dir_s = f"up {abs(pct)}%" if pct > 0 else f"down {abs(pct)}%"
    label = f"Stock at ${price}, moves {dir_s}. New price ($)?"
    hint  = f"{price} × {1 + pct/100:.4f} = {new_p}"
    return label, new_p, 0.5, hint

def q_pnl():
    """P&L on a simple directional trade."""
    shares = random.choice([100, 200, 500, 1000, 2000])
    entry  = round(random.uniform(20, 200), 2)
    move   = round(random.uniform(0.5, 8.0), 2)
    side   = random.choice(["Long", "Short"])
    exit_p = round(entry + move if side == "Long" else entry - move, 2)
    pnl    = round(shares * move, 2)
    label  = f"{side} {shares:,} @ ${entry:.2f}, exit @ ${exit_p:.2f}. P&L ($)?"
    hint   = f"{shares} × ${move:.2f} = ${pnl:,.2f}"
    return label, pnl, max(1, pnl * 0.02), hint

def q_pos_size():
    """Max shares given % risk, entry, and stop distance."""
    acct     = random.choice([25_000, 50_000, 100_000, 200_000, 500_000])
    risk_pct = random.choice([0.5, 1.0, 1.5, 2.0])
    entry    = random.randint(20, 150)
    stop_pct = round(random.uniform(0.5, 2.5), 1)
    risk_d   = acct * risk_pct / 100
    stop_d   = entry * stop_pct / 100
    shares   = int(risk_d / stop_d)
    label    = (f"Acct ${acct:,}, risk {risk_pct}%, entry ${entry}, "
                f"stop {stop_pct}% below. Max shares?")
    hint     = f"risk ${risk_d:,.0f} ÷ stop ${stop_d:.2f} = {shares}"
    return label, shares, max(5, shares * 0.05), hint

def q_kelly():
    """Kelly criterion percentage."""
    wr = random.choice([0.45, 0.50, 0.55, 0.60, 0.65, 0.70])
    rr = random.choice([1.0, 1.5, 2.0, 2.5, 3.0])
    kelly_pct = round((wr - (1 - wr) / rr) * 100, 1)
    label = (f"Win rate {wr*100:.0f}%, avg win/loss ratio {rr:.1f}:1.\n"
             f"  Kelly fraction (%)?")
    hint  = f"W − (1−W)/R = {wr} − {1-wr:.2f}/{rr} = {kelly_pct:.1f}%"
    return label, kelly_pct, 1.0, hint

def q_daily_sigma():
    """Daily 1-sigma move from annual IV (÷16 shortcut)."""
    iv    = random.choice([12, 16, 20, 24, 30, 32, 40, 48])
    daily = round(iv / math.sqrt(252), 2)
    label = (f"Annual IV = {iv}%. Approx daily 1-sigma move (%)?\n"
             f"  {DIM}Hint: use the ÷16 shortcut{R}")
    hint  = f"{iv} ÷ √252 ≈ {iv}/16 ≈ {daily:.2f}%"
    return label, daily, 0.15, hint

def q_sharpe():
    """Sharpe ratio."""
    ret = random.choice([8, 10, 12, 15, 18, 20, 24, 30])
    vol = random.choice([8, 10, 12, 15, 18, 20, 25])
    rfr = random.choice([2, 3, 4, 5])
    sharpe = round((ret - rfr) / vol, 2)
    label  = f"Return {ret}%, vol {vol}%, RFR {rfr}%. Sharpe ratio?"
    hint   = f"({ret}−{rfr}) / {vol} = {sharpe:.2f}"
    return label, sharpe, 0.05, hint

def q_implied_move():
    """Annual 1-sigma dollar move from IV."""
    price = random.choice([50, 100, 150, 200, 500])
    iv    = random.choice([15, 20, 25, 30, 40])
    move  = round(price * iv / 100)
    label = (f"Stock ${price}, IV {iv}%, 1 year.\n"
             f"  1-sigma expected move ($)?")
    hint  = f"${price} × {iv}% = ${move}"
    return label, float(move), max(1, move * 0.05), hint

QUESTIONS_POOL = [
    ("BPS Value",       CYAN,    q_bps),
    ("% Move",          YELLOW,  q_pct_move),
    ("P&L",             GREEN,   q_pnl),
    ("Position Size",   MAGENTA, q_pos_size),
    ("Kelly %",         RED,     q_kelly),
    ("Daily Sigma",     BLUE,    q_daily_sigma),
    ("Sharpe Ratio",    CYAN,    q_sharpe),
    ("Implied Move $",  YELLOW,  q_implied_move),
]

# ── persistence ────────────────────────────────────────────────────────────────

def load_log():
    if LOG_FILE.exists():
        try: return json.loads(LOG_FILE.read_text())
        except Exception: pass
    return []

def save_session(s):
    log = load_log()
    log.append(s)
    LOG_FILE.write_text(json.dumps(log, indent=2))

# ── screens ────────────────────────────────────────────────────────────────────

def show_history():
    clr()
    box("  Session History  ")
    log = load_log()
    if not log:
        print(f"\n  {DIM}No sessions yet.{R}\n")
        input("  Press Enter...")
        return
    print()
    print(f"  {DIM}{'Date':<18} {'Score':<8} {'Acc':<7} {'AvgTime'}{R}")
    hline()
    for s in log[-15:]:
        acc = s['correct'] / s['total']
        col = GREEN if acc >= 0.8 else YELLOW if acc >= 0.5 else RED
        print(f"  {DIM}{s['date'][:16]:<18}{R}"
              f"{col}{s['correct']}/{s['total']:<6}{R}"
              f"{col}{acc*100:.0f}%{'':<4}{R}"
              f"{DIM}{s['avg_time']:.1f}s{R}")
    print()
    input("  Press Enter...")

def run_session():
    correct = 0
    streak  = 0
    best_streak = 0
    times   = []
    pool    = QUESTIONS_POOL * 2
    random.shuffle(pool)
    selected = pool[:QUESTIONS]

    for q_num, (cat, color, gen) in enumerate(selected, 1):
        label, answer, tol, hint = gen()

        clr()
        streak_s = f"  🔥 {streak}" if streak >= 3 else ""
        print(f"{color}{BOLD}  Trading Math{R}  "
              f"{DIM}│  Q {q_num}/{QUESTIONS}  │  ✓ {correct}{streak_s}{R}")
        hline(color)
        print()
        print(f"  {color}{BOLD}[{cat}]{R}")
        print()
        for line in label.split("\n"):
            print(f"  {WHITE}{BOLD}{line}{R}")
        print()

        t0  = time.perf_counter()
        raw = input(f"  {WHITE}Answer (s=skip):{R} ").strip()
        elapsed = time.perf_counter() - t0
        times.append(elapsed)

        if raw.lower() == "s":
            print(f"\n  {DIM}Skipped — {hint}{R}")
            streak = 0
            time.sleep(1.2)
            continue

        try:
            user_ans = float(raw.replace(",", "").replace("$", "").replace("%", ""))
        except ValueError:
            print(f"\n  {RED}Invalid — {hint}{R}")
            streak = 0
            time.sleep(1.2)
            continue

        if abs(user_ans - answer) <= tol:
            correct += 1
            streak  += 1
            best_streak = max(best_streak, streak)
            print(f"\n  {GREEN}{BOLD}✓  Correct!{R}  {DIM}({elapsed:.1f}s)  {hint}{R}")
        else:
            streak = 0
            print(f"\n  {RED}{BOLD}✗  Off — answer: {answer:g}{R}  {DIM}{hint}{R}")

        time.sleep(1.2)

    # summary
    clr()
    accuracy  = correct / QUESTIONS * 100
    avg_time  = sum(times) / len(times)
    grade_col = GREEN if accuracy >= 80 else YELLOW if accuracy >= 50 else RED

    box("  Session Complete  ", grade_col)
    print()
    print(f"  {'Score':<16} {grade_col}{BOLD}{correct} / {QUESTIONS}{R}")
    print(f"  {'Accuracy':<16} {grade_col}{accuracy:.0f}%{R}")
    print(f"  {'Best streak':<16} {'🔥 ' if best_streak >= 5 else ''}{best_streak}")
    print(f"  {'Avg time':<16} {DIM}{avg_time:.1f}s{R}")
    print()

    save_session({
        "date": datetime.now().isoformat(),
        "correct": correct,
        "total": QUESTIONS,
        "best_streak": best_streak,
        "avg_time": avg_time,
    })
    input("  Press Enter...")

def main():
    while True:
        clr()
        box("  Trading Math Drill  ")
        print()
        print(f"  {DIM}Categories: BPS · P&L · Position Size · Kelly · Vol · Sharpe{R}")
        print()
        print(f"  {CYAN}{BOLD}1{R}  Start session")
        print(f"  {CYAN}{BOLD}2{R}  View history")
        print(f"  {CYAN}{BOLD}q{R}  Quit")
        print()
        c = input("  > ").strip().lower()
        if c == "q":
            break
        elif c == "1":
            run_session()
        elif c == "2":
            show_history()

if __name__ == "__main__":
    main()
