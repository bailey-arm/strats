#!/usr/bin/env python3
"""Quick Calc — trading calculator REPL. No external deps."""

import math
import os
import re
import sys

R       = "\033[0m"
BOLD    = "\033[1m"
DIM     = "\033[2m"
GREEN   = "\033[92m"
YELLOW  = "\033[93m"
RED     = "\033[91m"
CYAN    = "\033[96m"
WHITE   = "\033[97m"
BLUE    = "\033[94m"

def clr(): os.system("clear")

def parse_num(s: str) -> float:
    """Parse '2.5m', '100k', '$50', '25%' etc."""
    s = s.strip().lstrip("$").replace(",", "")
    m = re.match(r"^([\d.]+)([kmb%]?)$", s, re.I)
    if not m:
        raise ValueError(s)
    n, suf = float(m.group(1)), m.group(2).lower()
    if suf == "k": n *= 1_000
    elif suf == "m": n *= 1_000_000
    elif suf == "b": n *= 1_000_000_000
    elif suf == "%": n /= 100
    return n

def fmt(n: float) -> str:
    if abs(n) >= 1_000_000:
        return f"${n:,.2f}  ({n/1_000_000:.3f}M)"
    if abs(n) >= 1_000:
        return f"${n:,.2f}  ({n/1_000:.2f}k)"
    if abs(n) < 0.01:
        return f"{n:.6f}"
    return f"{n:,.4f}".rstrip("0").rstrip(".")

# ── commands ──────────────────────────────────────────────────────────────────

def cmd_bps(args):
    """bps <bps> <notional>   → dollar value"""
    b, n = parse_num(args[0]), parse_num(args[1])
    val = n * b / 10_000
    print(f"  {GREEN}{b:.0f}bps on {fmt(n)} = {BOLD}{fmt(val)}{R}")

def cmd_kelly(args):
    """kelly <win_rate> <win_loss_ratio>   → Kelly fraction"""
    wr = parse_num(args[0])
    rr = parse_num(args[1])
    if wr > 1: wr /= 100   # handle "55" → 0.55
    k = wr - (1 - wr) / rr
    half_k = k / 2
    print(f"  {GREEN}Kelly:  {BOLD}{k*100:.1f}%{R}  {DIM}(half-Kelly: {half_k*100:.1f}%){R}")
    if k <= 0:
        print(f"  {RED}Negative edge — do not trade this setup.{R}")

def cmd_pos(args):
    """pos <acct> <risk%> <entry> <stop>   → shares, notional, $ risk"""
    acct  = parse_num(args[0])
    risk  = parse_num(args[1])
    if risk > 1: risk /= 100
    entry = parse_num(args[2])
    stop  = parse_num(args[3])
    if stop > 1: stop /= 100   # treat as % if ≤ 1 else $ distance
    risk_d = acct * risk
    stop_d = entry * stop if stop <= 0.5 else stop
    shares = int(risk_d / stop_d)
    notional = shares * entry
    print(f"  {GREEN}Risk:     {BOLD}{fmt(risk_d)}{R}")
    print(f"  {GREEN}Stop $:   {BOLD}{fmt(stop_d)}/share{R}")
    print(f"  {GREEN}Shares:   {BOLD}{shares:,}{R}")
    print(f"  {GREEN}Notional: {BOLD}{fmt(notional)}{R}  {DIM}({notional/acct*100:.1f}% of acct){R}")

def cmd_pnl(args):
    """pnl <L|S> <qty> <entry> <exit>   → P&L"""
    side  = args[0].upper()
    qty   = parse_num(args[1])
    entry = parse_num(args[2])
    exit_ = parse_num(args[3])
    gross = qty * (exit_ - entry) * (1 if side == "L" else -1)
    pct   = (exit_ / entry - 1) * 100 * (1 if side == "L" else -1)
    col   = GREEN if gross >= 0 else RED
    print(f"  {col}{BOLD}P&L: {fmt(gross)}{R}  {DIM}({pct:+.2f}%){R}")

def cmd_move(args):
    """move <price> <iv%> [dte_days]   → expected 1-sigma move"""
    price = parse_num(args[0])
    iv    = parse_num(args[1])
    if iv > 1: iv /= 100
    dte   = float(args[2]) if len(args) > 2 else 365.0
    sigma = price * iv * math.sqrt(dte / 252)
    lo, hi = price - sigma, price + sigma
    print(f"  {GREEN}1-sigma move: ±{BOLD}{fmt(sigma)}{R}  {DIM}({iv*100:.0f}% IV, {dte:.0f}d){R}")
    print(f"  {DIM}Range: {lo:,.2f} — {hi:,.2f}{R}")

def cmd_sharpe(args):
    """sharpe <ret%> <vol%> <rfr%>   → Sharpe ratio"""
    ret = parse_num(args[0])
    vol = parse_num(args[1])
    rfr = parse_num(args[2])
    if ret > 1: ret /= 100
    if vol > 1: vol /= 100
    if rfr > 1: rfr /= 100
    s = (ret - rfr) / vol
    col = GREEN if s >= 1 else YELLOW if s >= 0.5 else RED
    print(f"  {col}{BOLD}Sharpe: {s:.2f}{R}  {DIM}(ret {ret*100:.1f}%, vol {vol*100:.1f}%, rfr {rfr*100:.1f}%){R}")

def cmd_fib(args):
    """fib <high> <low>   → Fibonacci retracement levels"""
    high = parse_num(args[0])
    low  = parse_num(args[1])
    rng  = high - low
    levels = [0.0, 0.236, 0.382, 0.5, 0.618, 0.786, 1.0]
    ext    = [1.272, 1.414, 1.618, 2.0, 2.618]
    print(f"  {CYAN}Retracements  (high {high:,.2f} → low {low:,.2f}){R}")
    for lvl in levels:
        price = high - rng * lvl
        marker = f"  {YELLOW}◄ 50{R}" if lvl == 0.5 else (f"  {GREEN}◄ key{R}" if lvl in (0.382, 0.618) else "")
        print(f"  {DIM}{lvl*100:>5.1f}%{R}  {WHITE}{price:>10,.2f}{R}{marker}")
    print(f"\n  {CYAN}Extensions{R}")
    for lvl in ext:
        price = low + rng * lvl
        print(f"  {DIM}{lvl*100:>5.1f}%{R}  {WHITE}{price:>10,.2f}{R}")

def cmd_compound(args):
    """compound <rate%> <years> [pv]   → future value / total growth"""
    rate = parse_num(args[0])
    if rate > 1: rate /= 100
    years = float(args[1])
    pv    = parse_num(args[2]) if len(args) > 2 else 1.0
    fv    = pv * (1 + rate) ** years
    cagr_mult = (1 + rate) ** years
    print(f"  {GREEN}FV: {BOLD}{fmt(fv)}{R}  {DIM}(×{cagr_mult:.2f} in {years:.0f}y @ {rate*100:.1f}%/yr){R}")
    print()
    milestones = [1, 2, 5, 10, 15, 20, 30]
    for y in [y for y in milestones if y <= years * 1.01]:
        v = pv * (1 + rate) ** y
        print(f"  {DIM}y{y:<4}{R}  {fmt(v)}")

def cmd_dv01(args):
    """dv01 <face> <duration> [price%]   → DV01 estimate"""
    face     = parse_num(args[0])
    dur      = float(args[1])
    price_pct = parse_num(args[2]) / 100 if len(args) > 2 else 1.0
    if price_pct > 2: price_pct /= 100
    mv    = face * price_pct
    dv01  = mv * dur * 0.0001
    print(f"  {GREEN}Market Value: {BOLD}{fmt(mv)}{R}")
    print(f"  {GREEN}DV01:         {BOLD}{fmt(dv01)}{R}  {DIM}per 1bp{R}")
    print(f"  {DIM}DV10: {fmt(dv01*10)}  │  DV100: {fmt(dv01*100)}{R}")

def cmd_carry(args):
    """carry <spot> <fwd> <days>   → annualised carry (bps)"""
    spot = parse_num(args[0])
    fwd  = parse_num(args[1])
    days = float(args[2])
    roll = (spot - fwd) / spot
    ann  = roll * 365 / days
    bps  = ann * 10_000
    col  = GREEN if bps > 0 else RED
    print(f"  {col}Carry: {BOLD}{bps:.1f}bps/yr{R}  {DIM}({roll*100:.3f}% over {days:.0f}d){R}")

def cmd_zscore(args):
    """zscore <value> <mean> <std>   → z-score"""
    val, mu, sigma = parse_num(args[0]), parse_num(args[1]), parse_num(args[2])
    z = (val - mu) / sigma
    col = RED if abs(z) > 2 else YELLOW if abs(z) > 1 else GREEN
    print(f"  {col}Z-score: {BOLD}{z:.2f}σ{R}  {DIM}(val {val:g}, μ {mu:g}, σ {sigma:g}){R}")

def cmd_rr(args):
    """rr <entry> <stop> <target>   → risk/reward ratio"""
    entry  = parse_num(args[0])
    stop   = parse_num(args[1])
    target = parse_num(args[2])
    risk   = abs(entry - stop)
    reward = abs(target - entry)
    ratio  = reward / risk
    be_wr  = 1 / (1 + ratio)
    col    = GREEN if ratio >= 2 else YELLOW if ratio >= 1 else RED
    print(f"  {col}R/R: {BOLD}{ratio:.2f}:1{R}  {DIM}(break-even win rate: {be_wr*100:.1f}%){R}")
    print(f"  {DIM}Risk: {fmt(risk)} │ Reward: {fmt(reward)}{R}")

# ── dispatch ──────────────────────────────────────────────────────────────────

COMMANDS = {
    "bps":      (cmd_bps,      "bps <bps> <notional>",         "dollar value of bps"),
    "kelly":    (cmd_kelly,    "kelly <win%> <rr>",             "Kelly fraction"),
    "pos":      (cmd_pos,      "pos <acct> <risk%> <entry> <stop%>", "position size"),
    "pnl":      (cmd_pnl,      "pnl <L|S> <qty> <entry> <exit>","P&L"),
    "move":     (cmd_move,     "move <price> <iv%> [dte]",      "expected 1-sigma move"),
    "sharpe":   (cmd_sharpe,   "sharpe <ret%> <vol%> <rfr%>",   "Sharpe ratio"),
    "fib":      (cmd_fib,      "fib <high> <low>",              "Fibonacci levels"),
    "compound": (cmd_compound, "compound <rate%> <years> [pv]", "compound growth"),
    "dv01":     (cmd_dv01,     "dv01 <face> <dur> [price%]",    "bond DV01"),
    "carry":    (cmd_carry,    "carry <spot> <fwd> <days>",     "annualised carry (bps)"),
    "zscore":   (cmd_zscore,   "zscore <val> <mean> <std>",     "z-score"),
    "rr":       (cmd_rr,       "rr <entry> <stop> <target>",    "risk/reward ratio"),
}

HELP_WIDTH = 46

def show_help():
    print(f"\n  {CYAN}{BOLD}Commands{R}")
    print(f"  {DIM}{'─' * HELP_WIDTH}{R}")
    for name, (_, usage, desc) in COMMANDS.items():
        print(f"  {CYAN}{name:<10}{R} {DIM}{usage:<32}{R}  {desc}")
    print(f"\n  {DIM}Numbers: 2.5m, 100k, 50%, $42.80 all parsed automatically.{R}")
    print(f"  {DIM}Type 'q' or Ctrl-C to exit.{R}\n")

def main():
    clr()
    print(f"\n  {BLUE}{BOLD}╔{'═'*50}╗{R}")
    print(f"  {BLUE}{BOLD}║{'  Quick Calc — Trading Calculator':^50}║{R}")
    print(f"  {BLUE}{BOLD}╚{'═'*50}╝{R}")
    show_help()

    while True:
        try:
            raw = input(f"  {CYAN}calc>{R} ").strip()
        except (KeyboardInterrupt, EOFError):
            print(f"\n  {DIM}Bye!{R}\n")
            break

        if not raw:
            continue
        if raw.lower() in ("q", "quit", "exit"):
            print(f"\n  {DIM}Bye!{R}\n")
            break
        if raw.lower() in ("h", "help", "?"):
            show_help()
            continue

        parts = raw.split()
        cmd   = parts[0].lower()
        args  = parts[1:]

        if cmd not in COMMANDS:
            print(f"  {RED}Unknown command '{cmd}'. Type 'help' for list.{R}")
            continue

        fn, usage, _ = COMMANDS[cmd]
        try:
            fn(args)
        except (IndexError, ValueError) as e:
            print(f"  {RED}Usage: {usage}{R}")
        print()

if __name__ == "__main__":
    main()
