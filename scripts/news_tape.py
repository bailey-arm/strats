#!/usr/bin/env python3
"""News Tape — live yfinance headlines, 2-min refresh with age tracking."""

import os, sys, time
from datetime import datetime

try:
    import yfinance as yf
except ImportError:
    print("yfinance not installed.")
    sys.exit(1)

R = "\033[0m"; BOLD = "\033[1m"; DIM = "\033[2m"
GREEN = "\033[92m"; RED = "\033[91m"; YELLOW = "\033[93m"
CYAN = "\033[96m"; WHITE = "\033[97m"; BLUE = "\033[94m"; MAGENTA = "\033[95m"

REFRESH = 120  # seconds

WATCHLIST = [
    "SPY", "QQQ", "^GSPC", "^VIX", "^TNX",
    "GLD", "USO", "BTC-USD",
    "NVDA", "AAPL", "META", "TSLA", "MSFT",
]

TICKER_COLORS = {
    "SPY":     BLUE,  "QQQ":    BLUE,   "^GSPC": BLUE,
    "^VIX":    RED,   "^TNX":   CYAN,
    "GLD":     YELLOW,"USO":    MAGENTA,
    "BTC-USD": YELLOW,
    "NVDA":    GREEN, "AAPL":   GREEN,  "META":  GREEN,
    "TSLA":    RED,   "MSFT":   CYAN,
}


def _parse_item(item, ticker):
    """Normalise old and new yfinance news formats → (title, timestamp)."""
    content = item.get("content", {}) if isinstance(item.get("content"), dict) else {}
    title = (content.get("title") or item.get("title") or "").strip()

    raw_ts = (content.get("pubDate") or item.get("providerPublishTime") or 0)
    if isinstance(raw_ts, str):
        try:
            dt = datetime.fromisoformat(raw_ts.replace("Z", "+00:00"))
            ts = dt.timestamp()
        except Exception:
            ts = 0.0
    else:
        ts = float(raw_ts) if raw_ts else 0.0

    return title, ts


def fetch_news():
    all_items = []
    seen = set()
    for ticker in WATCHLIST:
        try:
            raw = yf.Ticker(ticker).news or []
            for item in raw[:8]:
                title, ts = _parse_item(item, ticker)
                if not title or title in seen:
                    continue
                seen.add(title)
                all_items.append({"ticker": ticker, "title": title, "ts": ts})
        except Exception:
            pass
    return sorted(all_items, key=lambda x: x["ts"], reverse=True)[:45]


def age_str(ts):
    if not ts:
        return "─"
    ago = time.time() - ts
    if ago < 60:      return f"{int(ago)}s"
    if ago < 3600:    return f"{int(ago / 60)}m"
    if ago < 86400:   return f"{int(ago / 3600)}h"
    return f"{int(ago / 86400)}d"


def freshness_col(ts):
    if not ts:
        return DIM
    ago = time.time() - ts
    if ago < 1800:   return GREEN
    if ago < 7200:   return YELLOW
    if ago < 86400:  return WHITE
    return DIM


def render(news, next_ref):
    os.system("clear")
    now       = datetime.now().strftime("%H:%M:%S")
    remaining = max(0, int(next_ref - time.time()))
    print(f"\n  {BLUE}{BOLD}News Tape{R}  "
          f"{DIM}│  {now}  │  refresh in {remaining}s  │  Ctrl+C exit{R}\n")
    print(f"  {DIM}{'Age':>4}  {'Ticker':<10}  Headline{R}")
    print(f"  {DIM}{'─' * 72}{R}")

    if not news:
        print(f"\n  {DIM}No news found — check connection.{R}\n")
        return

    prev_bucket = None
    buckets = {"recent": f"─── Recent ({'<'}30m) ────────────────────────",
               "today":  "─── Earlier Today ──────────────────────────────",
               "older":  "─── Older ──────────────────────────────────────"}

    for item in news:
        ago = time.time() - item["ts"]
        if   ago < 1800:  bucket = "recent"
        elif ago < 86400: bucket = "today"
        else:             bucket = "older"

        if bucket != prev_bucket:
            print(f"\n  {DIM}{buckets[bucket]}{R}")
            prev_bucket = bucket

        age  = age_str(item["ts"])
        sym  = item["ticker"].lstrip("^").replace("-USD", "")[:10]
        col  = TICKER_COLORS.get(item["ticker"], CYAN)
        fc   = freshness_col(item["ts"])
        title = item["title"][:78]

        print(f"  {fc}{age:>4}{R}  {col}{BOLD}{sym:<10}{R}  {title}")

    print(f"\n  {DIM}Ctrl+C to return to menu{R}")


def main():
    news    = []
    next_ref = 0.0
    try:
        while True:
            if time.time() >= next_ref:
                print(f"\n  {DIM}Fetching news...{R}", end="", flush=True)
                news     = fetch_news()
                next_ref = time.time() + REFRESH
            render(news, next_ref)
            time.sleep(10)
    except KeyboardInterrupt:
        print(f"\n  {DIM}Exiting.{R}\n")


if __name__ == "__main__":
    main()
