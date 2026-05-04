#!/usr/bin/env python3
"""Volatility Dashboard — VIX term structure, RV vs IV, regime, 45s refresh."""

import math, os, sys, time
from datetime import datetime

try:
    import yfinance as yf
    import pandas as pd
    import numpy as np
except ImportError:
    print("yfinance/pandas/numpy not installed.")
    sys.exit(1)

R = "\033[0m"; BOLD = "\033[1m"; DIM = "\033[2m"
GREEN = "\033[92m"; RED = "\033[91m"; YELLOW = "\033[93m"
CYAN = "\033[96m"; WHITE = "\033[97m"; BLUE = "\033[94m"; MAGENTA = "\033[95m"

REFRESH = 45

VIX_TERM = [("9D", "^VIX9D"), ("1M", "^VIX"), ("3M", "^VIX3M"), ("6M", "^VIX6M")]
RV_TICKER = "^GSPC"


def fetch_vix_term():
    tickers = [t for _, t in VIX_TERM]
    try:
        raw = yf.download(tickers, period="2d", interval="1d",
                          auto_adjust=True, progress=False)
        closes = raw["Close"]
        if isinstance(closes, pd.Series):
            closes = closes.to_frame(tickers[0])
        result = {}
        for label, ticker in VIX_TERM:
            if ticker in closes.columns:
                col = closes[ticker].dropna()
                if len(col):
                    result[label] = float(col.iloc[-1])
        return result
    except Exception:
        return {}


def fetch_rv():
    try:
        raw = yf.download(RV_TICKER, period="6mo", interval="1d",
                          auto_adjust=True, progress=False)
        cl = raw["Close"]
        if isinstance(cl, pd.DataFrame):
            cl = cl.iloc[:, 0]
        rets  = cl.pct_change().dropna()
        rv_5  = float(rets.rolling(5).std().iloc[-1])  * math.sqrt(252) * 100
        rv_20 = float(rets.rolling(20).std().iloc[-1]) * math.sqrt(252) * 100
        rv_60 = float(rets.rolling(60).std().iloc[-1]) * math.sqrt(252) * 100
        return rv_5, rv_20, rv_60
    except Exception:
        return None, None, None


def regime(vix_1m, is_backwardation):
    if vix_1m is None:
        return YELLOW, "Unknown"
    if is_backwardation:
        return RED, "⚡ Backwardation — Stress Signal"
    if vix_1m < 12:  return CYAN,    "Suppressed Vol"
    if vix_1m < 16:  return GREEN,   "Low Vol"
    if vix_1m < 22:  return GREEN,   "Normal"
    if vix_1m < 30:  return YELLOW,  "Elevated"
    if vix_1m < 40:  return RED,     "High Fear"
    return RED + BOLD, "Extreme Fear (VIX > 40)"


def draw_term_chart(levels):
    """Horizontal dot-plot, one row per tenor, shared scale."""
    if len(levels) < 2:
        return
    labels = list(levels.keys())
    vals   = list(levels.values())
    lo     = min(vals) * 0.97
    hi     = max(vals) * 1.03
    rng    = hi - lo
    bar_w  = 34

    print(f"\n  {CYAN}Term Structure{R}  "
          f"{DIM}{lo:.1f} {'─' * (bar_w - 10)} {hi:.1f}{R}")

    prev = None
    for lbl, val in zip(labels, vals):
        pos = int((val - lo) / rng * (bar_w - 1))
        bar = DIM + "─" * pos + R + GREEN + "●" + R
        delta = ""
        if prev is not None:
            d = val - prev
            col = GREEN if d > 0 else RED
            delta = f"  {col}{d:+.2f}{R}"
        print(f"  {WHITE}{lbl:<4}{R}  {bar:<{bar_w + 20}}  {BOLD}{val:.2f}{R}{delta}")
        prev = val

    # contango/backwardation
    slope = vals[-1] - vals[0]
    sc = GREEN if slope > 0 else RED
    struct = "Contango" if slope > 0 else "Backwardation"
    print(f"\n  {sc}{BOLD}{struct}{R}  "
          f"{DIM}(front {vals[0]:.1f} → back {vals[-1]:.1f}, Δ{slope:+.2f}){R}")


def draw_rv_history():
    """Show 20d rolling RV sparkline for SPX."""
    try:
        raw = yf.download(RV_TICKER, period="6mo", interval="1d",
                          auto_adjust=True, progress=False)
        cl = raw["Close"]
        if isinstance(cl, pd.DataFrame):
            cl = cl.iloc[:, 0]
        rets = cl.pct_change().dropna()
        rv_series = rets.rolling(20).std().dropna() * math.sqrt(252) * 100
        vals = rv_series.values[-30:]

        lo, hi = vals.min(), vals.max()
        rng = hi - lo or 1
        SPARK = "▁▂▃▄▅▆▇█"
        spark = "".join(SPARK[min(7, int((v - lo) / rng * 8))] for v in vals)
        print(f"\n  {CYAN}SPX 20d RV (30d history){R}")
        print(f"  {DIM}{lo:.1f}% {R}{spark}{DIM} {hi:.1f}%{R}")
    except Exception:
        pass


def render(term, rv_5, rv_20, rv_60):
    os.system("clear")
    now = datetime.now().strftime("%H:%M:%S")
    print(f"\n  {BLUE}{BOLD}Volatility Dashboard{R}  "
          f"{DIM}│  {now}  │  refresh {REFRESH}s  │  Ctrl+C exit{R}\n")

    vix_1m  = term.get("1M")
    vix_9d  = term.get("9D")
    is_back = (vix_9d is not None and vix_1m is not None and vix_9d > vix_1m + 1.0)

    # ── VIX table ──
    print(f"  {CYAN}VIX Levels{R}")
    print(f"  {DIM}{'─' * 46}{R}")
    for label, _ in VIX_TERM:
        val = term.get(label)
        if val is None:
            print(f"  {DIM}{label:<6} ─{R}")
        else:
            col = BOLD if label == "1M" else ""
            print(f"  {col}{WHITE}{label:<6}{R}  {BOLD}{val:.2f}{R}")

    # ── term structure chart ──
    if len(term) >= 2:
        draw_term_chart(term)

    # ── RV vs IV ──
    print(f"\n  {CYAN}Realized vs Implied  (SPX){R}")
    print(f"  {DIM}{'─' * 46}{R}")
    iv = vix_1m
    for label, val in [("RV  5d", rv_5), ("RV 20d", rv_20), ("RV 60d", rv_60)]:
        if val is None:
            print(f"  {DIM}{label:<10} ─{R}")
        else:
            print(f"  {WHITE}{label:<10}{R}  {BOLD}{val:.2f}%{R}")
    if iv and rv_20:
        spread = iv - rv_20
        sc = GREEN if spread > 0 else RED
        print(f"\n  {WHITE}{'IV-RV spread':<14}{R}  {sc}{BOLD}{spread:+.2f}%{R}  "
              f"{DIM}({'IV premium' if spread > 0 else 'RV exceeds IV'}){R}")

    # ── regime ──
    rc, rl = regime(vix_1m, is_back)
    print(f"\n  {CYAN}Regime{R}")
    print(f"  {DIM}{'─' * 46}{R}")
    print(f"  {rc}{BOLD}{rl}{R}")

    # ── RV sparkline ──
    draw_rv_history()

    print(f"\n  {DIM}Ctrl+C to return to menu{R}")


def main():
    try:
        while True:
            print(f"\n  {DIM}Fetching vol data...{R}", end="", flush=True)
            term           = fetch_vix_term()
            rv_5, rv_20, rv_60 = fetch_rv()
            render(term, rv_5, rv_20, rv_60)
            deadline = time.time() + REFRESH
            while time.time() < deadline:
                time.sleep(1)
    except KeyboardInterrupt:
        print(f"\n  {DIM}Exiting.{R}\n")


if __name__ == "__main__":
    main()
