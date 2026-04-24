"""
Build per-ticker metadata for alpha research: liquidity, volatility, beta, and
(optionally) fundamentals / classification fetched from yfinance.

Panel-derived columns come from ``data/processed/ohlcv_long.parquet`` — no
network required. The optional ``--enrich-yfinance`` flag additionally hits
``yfinance.Ticker(t).info`` for each currently-listed ticker to pull sector /
industry / country / exchange / market cap / shares outstanding. That call is
rate-limited and spotty for delisted names, so it's gated behind a flag.

Output
------
data/universe/metadata.parquet  — one row per ticker with:

  ticker, n_bars, first_bar, last_bar, last_close,
  adv_20d_usd, adv_60d_usd, med_close,
  vol_60d, vol_252d, beta_252d,
  ret_12m_ex_1m,
  (optional) yf_sector, yf_industry, yf_country, yf_exchange,
             market_cap, shares_outstanding, yf_beta

Notes
-----
- ADV = dollar volume = raw close × volume (shares-volume is advisory per the
  yfinance caveats documented in scripts/fetch_ohlcv.py).
- Beta is computed vs the equal-weighted panel mean return — a crude market
  proxy, but avoids an extra network fetch for SPY. Good enough for
  cross-sectional bucketing; not a substitute for a factor-model beta.
- 12-1 momentum (``ret_12m_ex_1m``) is the classic: cumulative return from
  t-252 to t-21, skipping the most recent month to avoid reversal
  contamination. Standard in cross-sectional momentum research.
"""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from tqdm import tqdm

HERE = Path(__file__).resolve().parents[1]
PANEL_PATH = HERE / "data" / "processed" / "ohlcv_long.parquet"
UNIVERSE_PATH = HERE / "data" / "universe" / "sp500_pit.parquet"
OUT_PATH = HERE / "data" / "universe" / "metadata.parquet"


# ---------------------------------------------------------------------------
# Panel-derived metrics
# ---------------------------------------------------------------------------

def _compute_panel_metrics(df: pd.DataFrame) -> pd.DataFrame:
    """Compute liquidity, volatility, beta, and momentum per ticker."""
    df = df.sort_values(["ticker", "date"]).copy()

    # Dollar volume (raw close × volume — adj_close is retroactively restated)
    df["dollar_vol"] = df["close"] * df["volume"]

    # Daily total return from adj_close (splits + divs folded in)
    df["ret"] = df.groupby("ticker")["adj_close"].pct_change()

    # Market proxy = cross-sectional mean return across all tickers on each date
    market = df.groupby("date")["ret"].mean().rename("mkt_ret")
    df = df.merge(market, left_on="date", right_index=True, how="left")

    # Per-ticker aggregates
    rows: list[dict[str, Any]] = []
    for ticker, g in tqdm(df.groupby("ticker", sort=False), desc="metrics", total=df["ticker"].nunique()):
        valid = g.dropna(subset=["adj_close"])
        if len(valid) < 5:
            continue

        last = valid.iloc[-1]
        tail_20 = valid.tail(20)
        tail_60 = valid.tail(60)
        tail_252 = valid.tail(252)

        ret = valid["ret"]
        mkt = valid["mkt_ret"]

        # Beta vs market proxy, 252d window on pair-complete observations
        beta_252d = np.nan
        recent = valid[["ret", "mkt_ret"]].dropna().tail(252)
        if len(recent) >= 60 and recent["mkt_ret"].var() > 0:
            cov = recent["ret"].cov(recent["mkt_ret"])
            beta_252d = cov / recent["mkt_ret"].var()

        # 12-1 momentum: cumulative adj-return from t-252 to t-21
        ret_12m_ex_1m = np.nan
        adj = valid["adj_close"].to_numpy()
        if len(adj) >= 252:
            # skip last ~21 bars; use bar at t-252 as base, bar at t-21 as end
            start, end = adj[-252], adj[-21]
            if np.isfinite(start) and start > 0 and np.isfinite(end):
                ret_12m_ex_1m = float(end / start - 1.0)

        rows.append({
            "ticker": ticker,
            "n_bars": int(len(valid)),
            "first_bar": valid["date"].iloc[0],
            "last_bar": valid["date"].iloc[-1],
            "last_close": float(last["close"]) if pd.notna(last["close"]) else np.nan,
            "med_close": float(valid["close"].median()),
            "adv_20d_usd": float(tail_20["dollar_vol"].median()) if len(tail_20) else np.nan,
            "adv_60d_usd": float(tail_60["dollar_vol"].median()) if len(tail_60) else np.nan,
            "vol_60d": float(tail_60["ret"].std() * np.sqrt(252)) if tail_60["ret"].notna().sum() >= 20 else np.nan,
            "vol_252d": float(tail_252["ret"].std() * np.sqrt(252)) if tail_252["ret"].notna().sum() >= 60 else np.nan,
            "beta_252d": float(beta_252d) if np.isfinite(beta_252d) else np.nan,
            "ret_12m_ex_1m": ret_12m_ex_1m,
        })

    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# yfinance .info enrichment (optional)
# ---------------------------------------------------------------------------

_INFO_FIELDS: dict[str, str] = {
    # yfinance field                       → output column
    "sector": "yf_sector",
    "industry": "yf_industry",
    "country": "yf_country",
    "exchange": "yf_exchange",
    "marketCap": "market_cap",
    "sharesOutstanding": "shares_outstanding",
    "beta": "yf_beta",
    "fullTimeEmployees": "employees",
}


def _fetch_one_info(ticker: str) -> dict[str, Any]:
    import yfinance as yf

    row: dict[str, Any] = {"ticker": ticker}
    try:
        info = yf.Ticker(ticker).info or {}
    except Exception:  # noqa: BLE001 — yf.info raises a zoo of things
        return row
    for src, dst in _INFO_FIELDS.items():
        val = info.get(src)
        if val is not None:
            row[dst] = val
    return row


def _enrich_with_yfinance(tickers: list[str], workers: int) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    with ThreadPoolExecutor(max_workers=workers) as ex:
        futs = {ex.submit(_fetch_one_info, t): t for t in tickers}
        for fut in tqdm(as_completed(futs), total=len(futs), desc="yf.info"):
            rows.append(fut.result())
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--panel",
        type=Path,
        default=PANEL_PATH,
        help="long-format OHLCV parquet",
    )
    ap.add_argument(
        "--universe",
        type=Path,
        default=UNIVERSE_PATH,
        help="PIT universe parquet (for join / survivor filter)",
    )
    ap.add_argument(
        "--out",
        type=Path,
        default=OUT_PATH,
        help="output parquet path",
    )
    ap.add_argument(
        "--enrich-yfinance",
        action="store_true",
        help="additionally fetch yfinance .info per ticker (slow, rate-limited)",
    )
    ap.add_argument(
        "--yf-workers",
        type=int,
        default=6,
        help="parallel workers for yfinance .info calls",
    )
    ap.add_argument(
        "--yf-survivors-only",
        action="store_true",
        help="only query yfinance for tickers currently in the index",
    )
    args = ap.parse_args()

    if not args.panel.exists():
        raise SystemExit(f"Panel not found: {args.panel}. Run fetch_ohlcv.py first.")

    print(f"Loading panel: {args.panel}")
    df = pd.read_parquet(args.panel)
    df["date"] = pd.to_datetime(df["date"])

    print(f"Computing panel-derived metrics for {df['ticker'].nunique()} tickers...")
    meta = _compute_panel_metrics(df)

    if args.universe.exists():
        uni = pd.read_parquet(args.universe)
        join_cols = ["ticker"] + [
            c for c in [
                "currently_in_index", "ever_removed", "name", "gics_sector",
                "gics_sub_industry", "headquarters", "date_added", "cik", "founded",
            ] if c in uni.columns
        ]
        meta = meta.merge(uni[join_cols], on="ticker", how="left")

    if args.enrich_yfinance:
        if args.yf_survivors_only and "currently_in_index" in meta.columns:
            targets = meta.loc[meta["currently_in_index"].fillna(False), "ticker"].tolist()
        else:
            targets = meta["ticker"].tolist()
        print(f"Enriching {len(targets)} tickers via yfinance.info...")
        info = _enrich_with_yfinance(targets, workers=args.yf_workers)
        meta = meta.merge(info, on="ticker", how="left")

    meta = meta.sort_values("ticker").reset_index(drop=True)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    meta.to_parquet(args.out, index=False)
    print(f"\nWrote {len(meta):,} rows, {len(meta.columns)} cols → {args.out}")
    print(f"Columns: {list(meta.columns)}")


if __name__ == "__main__":
    main()
