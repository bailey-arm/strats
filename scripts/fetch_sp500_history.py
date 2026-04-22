"""
Build a survivorship-bias-free US-equity ticker universe.

Seeds the universe from the two Wikipedia tables:
  1. Current S&P 500 constituents
  2. Historical S&P 500 additions / removals

The union of (current) ∪ (every ticker ever added OR removed) gives every
symbol that was a member at any point in the window Wikipedia covers
(roughly 1990-present). Using this union to drive OHLCV downloads
mitigates survivorship bias: you include names that dropped out.

Output
------
data/universe/sp500_pit.parquet   — one row per ticker, first/last-seen dates
data/universe/sp500_tickers.txt   — flat list of tickers (one per line)

Caveats
-------
- Wikipedia is a volunteer-maintained source; it's the best *free* proxy but
  not a vendor-grade PIT file. Spot-checks against known index changes are
  advisable before production use.
- Ticker symbols occasionally get reassigned to a different company after a
  delisting (e.g. "GM"). The downloader in fetch_ohlcv.py fetches whatever
  yfinance currently maps the symbol to — acceptable for a first pass but a
  known caveat.
"""

from __future__ import annotations

import io
from pathlib import Path

import pandas as pd
import requests

HERE = Path(__file__).resolve().parents[1]
OUT_DIR = HERE / "data" / "universe"
OUT_DIR.mkdir(parents=True, exist_ok=True)

WIKI_URL = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
HEADERS = {"User-Agent": "Mozilla/5.0 (strats-research data fetcher)"}


def _fetch_wiki_tables() -> list[pd.DataFrame]:
    resp = requests.get(WIKI_URL, headers=HEADERS, timeout=30)
    resp.raise_for_status()
    return pd.read_html(io.StringIO(resp.text))


def _clean_ticker(t: str) -> str:
    # Wikipedia uses "BRK.B"; yfinance expects "BRK-B".
    return str(t).strip().replace(".", "-")


def build_universe() -> pd.DataFrame:
    tables = _fetch_wiki_tables()
    current = tables[0].copy()
    changes = tables[1].copy()

    # --- current constituents -------------------------------------------------
    # Columns have varied historically; locate by case-insensitive match.
    sym_col = next(c for c in current.columns if "symbol" in str(c).lower())
    current = current.rename(columns={sym_col: "ticker"})
    current["ticker"] = current["ticker"].map(_clean_ticker)
    current_tickers = set(current["ticker"])

    # --- historical changes ---------------------------------------------------
    # Columns look like: ("Effective Date", "Effective Date"), ("Added", "Ticker"),
    # ("Added", "Security"), ("Removed", "Ticker"), ("Removed", "Security"),
    # ("Reason", "Reason"). Flatten and locate by substring.
    changes.columns = [
        "|".join(str(x) for x in col) if isinstance(col, tuple) else str(col)
        for col in changes.columns
    ]
    date_col = next(c for c in changes.columns if "date" in c.lower())
    added_col = next(
        c for c in changes.columns if c.lower().startswith("added") and "ticker" in c.lower()
    )
    removed_col = next(
        c for c in changes.columns if c.lower().startswith("removed") and "ticker" in c.lower()
    )

    # pandas 2.x leaves the MultiIndex header-row values inside the body when the
    # levels duplicate ("Effective Date" appears at both levels, etc.) — drop any
    # row whose date cell is literally the header text.
    changes = changes[changes[date_col].astype(str) != date_col.split("|")[-1]]
    changes["event_date"] = pd.to_datetime(changes[date_col], errors="coerce")

    added = changes[[added_col, "event_date"]].rename(columns={added_col: "ticker"})
    added["event"] = "added"
    removed = changes[[removed_col, "event_date"]].rename(columns={removed_col: "ticker"})
    removed["event"] = "removed"
    events = pd.concat([added, removed], ignore_index=True)
    events = events.dropna(subset=["ticker"])
    events["ticker"] = events["ticker"].map(_clean_ticker)

    # --- union ---------------------------------------------------------------
    all_tickers = sorted(current_tickers | set(events["ticker"]))

    # first_seen / last_seen give a crude PIT window; use together with price
    # history availability to filter later.
    first_seen = events.groupby("ticker")["event_date"].min()
    last_removed = (
        events[events["event"] == "removed"].groupby("ticker")["event_date"].max()
    )

    universe = pd.DataFrame({"ticker": all_tickers})
    universe["currently_in_index"] = universe["ticker"].isin(current_tickers)
    universe["first_event_date"] = universe["ticker"].map(first_seen)
    universe["last_removed_date"] = universe["ticker"].map(last_removed)
    # If still in the index, last_removed is NaT — flag accordingly.
    universe["ever_removed"] = universe["last_removed_date"].notna()

    return universe.sort_values("ticker").reset_index(drop=True)


def main() -> None:
    uni = build_universe()
    out_parquet = OUT_DIR / "sp500_pit.parquet"
    out_txt = OUT_DIR / "sp500_tickers.txt"
    uni.to_parquet(out_parquet, index=False)
    out_txt.write_text("\n".join(uni["ticker"].tolist()) + "\n")
    print(f"Wrote {len(uni):,} tickers → {out_parquet}")
    print(
        f"  currently in index: {uni['currently_in_index'].sum()} | "
        f"ever removed: {uni['ever_removed'].sum()}"
    )


if __name__ == "__main__":
    main()
