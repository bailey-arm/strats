#!/usr/bin/env python3
"""Macro Dashboard — live multi-asset snapshot with auto-refresh."""

import os
import sys
import time
from datetime import datetime

try:
    import yfinance as yf
except ImportError:
    print("yfinance not installed. Run: pip install yfinance")
    sys.exit(1)

R       = "\033[0m"
BOLD    = "\033[1m"
DIM     = "\033[2m"
GREEN   = "\033[92m"
YELLOW  = "\033[93m"
RED     = "\033[91m"
CYAN    = "\033[96m"
WHITE   = "\033[97m"
BLUE    = "\033[94m"
MAGENTA = "\033[95m"

REFRESH = 30  # seconds

PANELS = [
    # label,        ticker,     fmt,      unit
    ("EQUITIES",    None,       None,     None),
    ("SPX",         "^GSPC",    ",.0f",   ""),
    ("NDX",         "^NDX",     ",.0f",   ""),
    ("SX5E",        "^STOXX50E",",.0f",   ""),
    ("Russell",     "^RUT",     ",.0f",   ""),
    ("",            None,       None,     None),
    ("VOLATILITY",  None,       None,     None),
    ("VIX",         "^VIX",     ".2f",    ""),
    ("VVIX",        "^VVIX",    ".2f",    ""),
    ("",            None,       None,     None),
    ("RATES",       None,       None,     None),
    ("US 2Y",       "^IRX",     ".3f",    "%"),
    ("US 10Y",      "^TNX",     ".3f",    "%"),
    ("US 30Y",      "^TYX",     ".3f",    "%"),
    ("",            None,       None,     None),
    ("FX / MACRO",  None,       None,     None),
    ("DXY",         "DX-Y.NYB", ".2f",    ""),
    ("EURUSD",      "EURUSD=X", ".4f",    ""),
    ("GBPUSD",      "GBPUSD=X", ".4f",    ""),
    ("USDJPY",      "JPY=X",    ".2f",    ""),
    ("",            None,       None,     None),
    ("COMMODITIES", None,       None,     None),
    ("Gold",        "GC=F",     ",.2f",   ""),
    ("Oil (WTI)",   "CL=F",     ".2f",    ""),
    ("Nat Gas",     "NG=F",     ".3f",    ""),
    ("",            None,       None,     None),
    ("CRYPTO",      None,       None,     None),
    ("BTC",         "BTC-USD",  ",.0f",   ""),
    ("ETH",         "ETH-USD",  ",.2f",   ""),
]

TICKERS = [p[1] for p in PANELS if p[1]]


def fetch():
    try:
        data = yf.download(TICKERS, period="2d", interval="1d",
                           auto_adjust=True, progress=False)
        closes = data["Close"]
        prices, changes = {}, {}
        for t in TICKERS:
            if t in closes.columns:
                col = closes[t].dropna()
                if len(col) >= 2:
                    prices[t]  = float(col.iloc[-1])
                    changes[t] = (float(col.iloc[-1]) / float(col.iloc[-2]) - 1) * 100
                elif len(col) == 1:
                    prices[t]  = float(col.iloc[-1])
                    changes[t] = 0.0
        return prices, changes
    except Exception as e:
        return {}, {}


def chg_color(chg):
    if chg > 0.5:   return GREEN
    if chg < -0.5:  return RED
    return YELLOW


def bar(chg, width=8):
    """Tiny ASCII bar for the change magnitude."""
    filled = min(width, int(abs(chg) / 0.25))
    ch = "█" if chg >= 0 else "▓"
    col = chg_color(chg)
    return f"{col}{ch * filled}{DIM}{'░' * (width - filled)}{R}"


def render(prices, changes, last_updated):
    os.system("clear")
    now = datetime.now().strftime("%H:%M:%S")
    width = 58

    # header
    line = "─" * width
    print(f"{BLUE}┌{line}┐{R}")
    title = f"  Macro Dashboard  {DIM}│{R}{BLUE}  {now}  {DIM}│{R}{BLUE}  refresh {REFRESH}s  "
    pad   = width - len(f"  Macro Dashboard    {now}   refresh {REFRESH}s  ") + 2
    print(f"{BLUE}│{R}  {BOLD}{WHITE}Macro Dashboard{R}  "
          f"{DIM}│  {now}  │  next refresh {REFRESH}s  {' ' * max(0,pad)}{BLUE}│{R}")
    print(f"{BLUE}└{line}┘{R}")
    print()

    for label, ticker, fmt, unit in PANELS:
        if ticker is None:
            if label == "":
                print()
            else:
                print(f"  {CYAN}{BOLD}{label}{R}")
                print(f"  {DIM}{'─' * 50}{R}")
            continue

        if ticker not in prices:
            print(f"  {DIM}{label:<14} —{R}")
            continue

        price = prices[ticker]
        chg   = changes.get(ticker, 0.0)
        col   = chg_color(chg)
        chg_s = f"{'+' if chg >= 0 else ''}{chg:.2f}%"
        p_s   = format(price, fmt) + unit

        b = bar(chg)
        print(f"  {WHITE}{label:<14}{R}  {BOLD}{p_s:<12}{R}  "
              f"{col}{chg_s:>8}{R}  {b}")

    print()
    if not prices:
        print(f"  {RED}Failed to fetch data — check connection.{R}\n")
    print(f"  {DIM}Press Ctrl+C to exit{R}")


def main():
    try:
        prices, changes = fetch()
        render(prices, changes, datetime.now())
        last = time.time()

        while True:
            time.sleep(1)
            if time.time() - last >= REFRESH:
                prices, changes = fetch()
                last = time.time()
                render(prices, changes, datetime.now())
    except KeyboardInterrupt:
        print(f"\n  {DIM}Exiting dashboard.{R}\n")


if __name__ == "__main__":
    main()
