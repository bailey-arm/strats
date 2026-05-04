#!/usr/bin/env python3
"""Live Ticker Tape — multi-asset price stream with sparklines, 8s refresh."""

import os, sys, time, threading
from collections import deque
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime

try:
    import yfinance as yf
    import pandas as pd
except ImportError:
    print("yfinance/pandas not installed.")
    sys.exit(1)

R = "\033[0m"; BOLD = "\033[1m"; DIM = "\033[2m"
GREEN = "\033[92m"; RED = "\033[91m"; YELLOW = "\033[93m"
CYAN = "\033[96m"; WHITE = "\033[97m"; BLUE = "\033[94m"

SPARK_CHARS = "▁▂▃▄▅▆▇█"
REFRESH = 8  # seconds between fetches

WATCHLIST = [
    ("SPX",   "^GSPC"),
    ("NDX",   "^NDX"),
    ("SX5E",  "^STOXX50E"),
    ("VIX",   "^VIX"),
    ("10Y",   "^TNX"),
    ("DXY",   "DX-Y.NYB"),
    ("Gold",  "GC=F"),
    ("Oil",   "CL=F"),
    ("BTC",   "BTC-USD"),
    ("ETH",   "ETH-USD"),
    ("NVDA",  "NVDA"),
    ("AAPL",  "AAPL"),
]

# Group separators — drawn above these labels
GROUP_BEFORE = {"VIX", "10Y", "DXY", "Gold", "BTC", "NVDA"}

_lock   = threading.Lock()
_prices = {}   # ticker -> {price, prev, hist: deque}


def sparkline(hist):
    vals = list(hist)
    if len(vals) < 2:
        return DIM + "─" * 12 + R
    lo, hi = min(vals), max(vals)
    rng = hi - lo
    if rng == 0:
        return DIM + "─" * min(12, len(vals)) + R
    return "".join(SPARK_CHARS[min(7, int((v - lo) / rng * 8))] for v in vals[-12:])


def velocity(hist):
    vals = list(hist)
    if len(vals) < 5:
        return " "
    d = (vals[-1] - vals[-4]) / vals[-4] * 100 if vals[-4] else 0
    if d >  0.1: return GREEN + "▲" + R
    if d < -0.1: return RED   + "▼" + R
    return YELLOW + "─" + R


def _fetch_one(item):
    _, ticker = item
    try:
        fi = yf.Ticker(ticker).fast_info
        p  = getattr(fi, "last_price", None)
        pr = getattr(fi, "previous_close", None)
        return ticker, p, pr
    except Exception:
        return ticker, None, None


def fetch_all():
    with ThreadPoolExecutor(max_workers=6) as ex:
        results = list(ex.map(_fetch_one, WATCHLIST))
    with _lock:
        for ticker, price, prev in results:
            if price is None:
                continue
            if ticker not in _prices:
                _prices[ticker] = {"price": price, "prev": prev or price,
                                   "hist": deque(maxlen=30)}
            _prices[ticker]["price"] = price
            if prev:
                _prices[ticker]["prev"] = prev
            _prices[ticker]["hist"].append(price)


def seed_history():
    """Seed sparklines from 1d of 5-min bars."""
    tickers = [t for _, t in WATCHLIST]
    try:
        raw = yf.download(tickers, period="1d", interval="5m",
                          auto_adjust=True, progress=False)
        closes = raw["Close"]
        if isinstance(closes, pd.Series):
            closes = closes.to_frame(tickers[0])
        with _lock:
            for _, ticker in WATCHLIST:
                if ticker not in closes.columns:
                    continue
                col = closes[ticker].dropna()
                if ticker not in _prices:
                    _prices[ticker] = {"price": 0.0, "prev": 0.0,
                                       "hist": deque(maxlen=30)}
                for v in col.values[-20:]:
                    if v == v:  # NaN check
                        _prices[ticker]["hist"].append(float(v))
    except Exception:
        pass


def render(countdown):
    os.system("clear")
    now = datetime.now().strftime("%H:%M:%S")
    print(f"\n  {BLUE}{BOLD}Live Ticker Tape{R}  "
          f"{DIM}│  {now}  │  refresh in {countdown:>2}s  │  Ctrl+C exit{R}\n")
    print(f"  {DIM}{'Asset':<8}  {'Price':>12}  {'Day Chg':>8}  {'Sparkline':^14}  V{R}")

    with _lock:
        for label, ticker in WATCHLIST:
            if label in GROUP_BEFORE:
                print(f"  {DIM}{'─' * 56}{R}")

            if ticker not in _prices:
                print(f"  {DIM}{label:<8}  {'loading...':>12}{R}")
                continue

            d     = _prices[ticker]
            price = d["price"]
            prev  = d["prev"] or price
            chg   = (price / prev - 1) * 100 if prev else 0
            col   = GREEN if chg >= 0 else RED
            spark = sparkline(d["hist"])
            vel   = velocity(d["hist"])
            p_str = f"{price:>12,.2f}" if price >= 10 else f"{price:>12,.4f}"
            c_str = f"{'+' if chg >= 0 else ''}{chg:.2f}%"

            print(f"  {WHITE}{BOLD}{label:<8}{R}  "
                  f"{col}{p_str}{R}  "
                  f"{col}{c_str:>8}{R}  "
                  f"{spark}  {vel}")

    print(f"\n  {DIM}Ctrl+C to return to menu{R}")


def main():
    print(f"\n  {DIM}Seeding history from 5m bars...{R}", end="", flush=True)
    seed_history()
    print(" done.")
    print(f"  {DIM}Fetching live prices...{R}", end="", flush=True)
    fetch_all()
    print(" done.")
    try:
        while True:
            deadline = time.time() + REFRESH
            while time.time() < deadline:
                render(int(deadline - time.time()))
                time.sleep(1)
            fetch_all()
    except KeyboardInterrupt:
        print(f"\n  {DIM}Exiting.{R}\n")


if __name__ == "__main__":
    main()
