"""
Bulk-download daily OHLCV for every ticker in data/universe/sp500_tickers.txt.

Downloads both raw OHLC and split/dividend-adjusted fields via yfinance.
Stores one Parquet per ticker under data/raw/ohlcv/ plus a single consolidated
long-format Parquet at data/processed/ohlcv_long.parquet for easy loading.

Usage
-----
    python scripts/fetch_ohlcv.py                       # default: since 2000-01-01
    python scripts/fetch_ohlcv.py --start 1990-01-01
    python scripts/fetch_ohlcv.py --universe my_list.txt
    python scripts/fetch_ohlcv.py --workers 8 --batch 50

Notes on lookahead bias
-----------------------
- `Adj Close` returned by yfinance applies all splits + cash dividends
  retroactively. That's correct for computing RETURNS, but the adjusted price
  at a past date is NOT the price a trader would have seen — so don't use it
  as a dollar level (e.g. for position-sizing in $ terms, use raw Close).
- Volume is NOT adjusted by yfinance for splits in older data in all cases;
  treat share-volume as advisory and prefer dollar-volume = raw Close × Volume.
- The script writes BOTH raw and adjusted fields so downstream code can pick.
"""

from __future__ import annotations

import argparse
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import pandas as pd
import yfinance as yf
from tqdm import tqdm

HERE = Path(__file__).resolve().parents[1]
RAW_DIR = HERE / "data" / "raw" / "ohlcv"
PROCESSED_DIR = HERE / "data" / "processed"
DEFAULT_UNIVERSE = HERE / "data" / "universe" / "sp500_tickers.txt"

RAW_DIR.mkdir(parents=True, exist_ok=True)
PROCESSED_DIR.mkdir(parents=True, exist_ok=True)


def _download_one(ticker: str, start: str, end: str | None) -> pd.DataFrame | None:
    """Single-ticker download. Returns a long-format frame or None on failure."""
    try:
        # auto_adjust=False → keep raw OHLC; Adj Close comes in its own column.
        df = yf.download(
            ticker,
            start=start,
            end=end,
            auto_adjust=False,
            progress=False,
            threads=False,
            timeout=30,
        )
    except Exception as exc:  # noqa: BLE001 — yf raises a zoo of things
        return pd.DataFrame({"_error": [str(exc)]}).assign(ticker=ticker)

    if df is None or df.empty:
        return None

    # yfinance sometimes returns a column MultiIndex with the ticker on level 1.
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)

    df = df.rename(
        columns={
            "Open": "open",
            "High": "high",
            "Low": "low",
            "Close": "close",
            "Adj Close": "adj_close",
            "Volume": "volume",
        }
    )
    expected = ["open", "high", "low", "close", "adj_close", "volume"]
    for col in expected:
        if col not in df.columns:
            df[col] = pd.NA
    df = df[expected].copy()
    df.index.name = "date"
    df = df.reset_index()
    df["ticker"] = ticker
    return df


def fetch_all(
    tickers: list[str],
    start: str,
    end: str | None,
    workers: int,
) -> tuple[list[str], list[str]]:
    """Download each ticker in parallel. Returns (succeeded, failed)."""
    succeeded: list[str] = []
    failed: list[str] = []

    def _job(t: str) -> tuple[str, pd.DataFrame | None]:
        return t, _download_one(t, start, end)

    with ThreadPoolExecutor(max_workers=workers) as ex:
        futures = {ex.submit(_job, t): t for t in tickers}
        for fut in tqdm(as_completed(futures), total=len(futures), desc="OHLCV"):
            ticker, df = fut.result()
            if df is None or "_error" in df.columns:
                failed.append(ticker)
                continue
            out = RAW_DIR / f"{ticker}.parquet"
            df.to_parquet(out, index=False)
            succeeded.append(ticker)
    return succeeded, failed


def consolidate(tickers: list[str]) -> Path:
    """Merge all per-ticker files into one long-format parquet."""
    frames = []
    for t in tickers:
        p = RAW_DIR / f"{t}.parquet"
        if p.exists():
            frames.append(pd.read_parquet(p))
    if not frames:
        raise RuntimeError("No per-ticker parquet files found to consolidate.")
    big = pd.concat(frames, ignore_index=True)
    big = big.sort_values(["ticker", "date"]).reset_index(drop=True)
    out = PROCESSED_DIR / "ohlcv_long.parquet"
    big.to_parquet(out, index=False)
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--universe", type=Path, default=DEFAULT_UNIVERSE)
    ap.add_argument("--start", default="2000-01-01")
    ap.add_argument("--end", default=None, help="inclusive upper bound, YYYY-MM-DD")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument(
        "--limit",
        type=int,
        default=None,
        help="debug: only fetch the first N tickers",
    )
    args = ap.parse_args()

    if not args.universe.exists():
        raise SystemExit(
            f"Universe file not found: {args.universe}\n"
            "Run scripts/fetch_sp500_history.py first."
        )

    tickers = [t.strip() for t in args.universe.read_text().splitlines() if t.strip()]
    if args.limit:
        tickers = tickers[: args.limit]

    t0 = time.time()
    ok, bad = fetch_all(tickers, args.start, args.end, args.workers)
    print(f"\nDownloaded {len(ok)} / {len(tickers)} tickers in {time.time()-t0:.1f}s")
    if bad:
        fail_path = PROCESSED_DIR / "failed_tickers.txt"
        fail_path.write_text("\n".join(bad) + "\n")
        print(f"  {len(bad)} failed (symbol delisted / renamed / no data) → {fail_path}")

    out = consolidate(ok)
    print(f"Consolidated long frame → {out}")


if __name__ == "__main__":
    main()
