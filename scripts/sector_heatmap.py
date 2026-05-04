#!/usr/bin/env python3
"""Sector Heatmap — GICS sector performance across 1D / 1W / 1M / 3M."""

import os, sys, time
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
CYAN = "\033[96m"; WHITE = "\033[97m"; BLUE = "\033[94m"

REFRESH = 60

SECTORS = [
    ("Technology",   "XLK"),
    ("Healthcare",   "XLV"),
    ("Financials",   "XLF"),
    ("Energy",       "XLE"),
    ("Industrials",  "XLI"),
    ("Cons. Disc.",  "XLY"),
    ("Staples",      "XLP"),
    ("Utilities",    "XLU"),
    ("Materials",    "XLB"),
    ("Real Estate",  "XLRE"),
    ("Comm. Svcs",   "XLC"),
]


def fetch():
    tickers = [t for _, t in SECTORS] + ["SPY"]
    raw = yf.download(tickers, period="4mo", interval="1d",
                      auto_adjust=True, progress=False)
    closes = raw["Close"]
    if isinstance(closes, pd.Series):
        closes = closes.to_frame(tickers[0])

    spy_col = closes.get("SPY") if isinstance(closes, pd.DataFrame) else None

    result = {}
    for label, ticker in SECTORS:
        if ticker not in closes.columns:
            result[label] = {}
            continue
        col = closes[ticker].dropna()
        if len(col) < 22:
            result[label] = {}
            continue
        p = col.values

        r1d = (p[-1] / p[-2]  - 1) * 100 if len(p) >= 2  else None
        r1w = (p[-1] / p[-6]  - 1) * 100 if len(p) >= 6  else None
        r1m = (p[-1] / p[-22] - 1) * 100 if len(p) >= 22 else None
        r3m = (p[-1] / p[0]   - 1) * 100

        # vs SPY (1M)
        rel = None
        if spy_col is not None and len(spy_col.dropna()) >= 22:
            spy = spy_col.dropna().values
            spy_1m = (spy[-1] / spy[-22] - 1) * 100
            rel = r1m - spy_1m if r1m is not None else None

        result[label] = {"1D": r1d, "1W": r1w, "1M": r1m, "3M": r3m,
                         "rel": rel, "price": float(p[-1])}
    return result


def col_for(pct):
    if pct is None:  return DIM,   "─"
    s = f"{'+' if pct > 0 else ''}{pct:.1f}%"
    if pct >  2.5:   return GREEN + BOLD, s
    if pct >  0.5:   return GREEN,        s
    if pct > -0.5:   return YELLOW,       s
    if pct > -2.5:   return RED,          s
    return RED + BOLD,  s


def heat_block(pct):
    if pct is None:   return DIM + "░░" + R
    if pct >  2.5:    return GREEN  + BOLD + "██" + R
    if pct >  0.5:    return GREEN  + "██" + R
    if pct > -0.5:    return YELLOW + "██" + R
    if pct > -2.5:    return RED    + "██" + R
    return RED + BOLD + "██" + R


def render(data, sort_col="1D"):
    os.system("clear")
    now = datetime.now().strftime("%H:%M:%S")
    print(f"\n  {BLUE}{BOLD}Sector Heatmap{R}  "
          f"{DIM}│  {now}  │  refresh {REFRESH}s  │  Ctrl+C exit{R}\n")

    ordered = sorted(
        [(lbl, data.get(lbl, {})) for lbl, _ in SECTORS],
        key=lambda x: (x[1].get(sort_col) or -999),
        reverse=True
    )

    print(f"  {DIM}{'Sector':<16}  {'Price':>7}  {'1D':>8}  {'1W':>8}  "
          f"{'1M':>8}  {'3M':>8}  {'vs SPY':>8}{R}")
    print(f"  {DIM}{'─' * 72}{R}")

    for lbl, d in ordered:
        if not d:
            print(f"  {DIM}{lbl:<16}  {'─':>7}{R}")
            continue
        price = d["price"]
        c1d, s1d = col_for(d.get("1D"))
        c1w, s1w = col_for(d.get("1W"))
        c1m, s1m = col_for(d.get("1M"))
        c3m, s3m = col_for(d.get("3M"))
        crl, srl = col_for(d.get("rel"))

        print(f"  {WHITE}{lbl:<16}{R}  "
              f"{DIM}{price:>7.2f}{R}  "
              f"{c1d}{s1d:>8}{R}  "
              f"{c1w}{s1w:>8}{R}  "
              f"{c1m}{s1m:>8}{R}  "
              f"{c3m}{s3m:>8}{R}  "
              f"{crl}{srl:>8}{R}")

    # heat strip
    print(f"\n  {CYAN}Heat Strip  {DIM}1D sort  (top performer → bottom){R}")
    print("  ", end="")
    for lbl, d in ordered:
        print(heat_block(d.get("1D") if d else None), end="")
    print()
    print("  " + "".join(f"{DIM}{lbl[:3]:<5}{R}" for lbl, _ in ordered))

    # 1M heat strip
    by_1m = sorted(ordered, key=lambda x: (x[1].get("1M") or -999), reverse=True)
    print(f"\n  {DIM}1M sort:{R}  ", end="")
    for lbl, d in by_1m:
        print(heat_block(d.get("1M") if d else None), end="")
    print()
    print("  " + " " * 10 + "".join(f"{DIM}{lbl[:3]:<5}{R}" for lbl, _ in by_1m))

    print(f"\n  {DIM}Ctrl+C to return to menu{R}")


def main():
    try:
        while True:
            print(f"\n  {DIM}Fetching sector data...{R}", end="", flush=True)
            data = fetch()
            render(data)
            deadline = time.time() + REFRESH
            while time.time() < deadline:
                time.sleep(1)
    except KeyboardInterrupt:
        print(f"\n  {DIM}Exiting.{R}\n")


if __name__ == "__main__":
    main()
