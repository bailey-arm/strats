# strats

Scratchpad for building and backtesting systematic strategies on US equities.

## Setup

```bash
cd strats
python3.11 -m venv venv
source venv/bin/activate
pip install -U pip
pip install -r requirements.txt

# register the venv as a Jupyter kernel
python -m ipykernel install --user --name strats --display-name "Python (strats)"

jupyter lab          # or: jupyter notebook
```

## Getting the data

Two steps. The first builds a survivorship-bias-free universe from Wikipedia's
historical S&P 500 constituent list; the second bulk-downloads daily OHLCV via
yfinance.

```bash
python scripts/fetch_sp500_history.py           # → data/universe/sp500_*.{parquet,txt}
python scripts/fetch_ohlcv.py --start 2000-01-01 --workers 8
```

Output layout:

```
data/
├── universe/
│   ├── sp500_pit.parquet        # ticker + first/last-seen dates, current-index flag
│   └── sp500_tickers.txt        # flat list used by the OHLCV downloader
├── raw/ohlcv/
│   └── <TICKER>.parquet         # per-ticker daily bars (date, O/H/L/C, adj_close, volume)
└── processed/
    └── ohlcv_long.parquet       # all tickers in one long-format frame
```

Expect ~1000 tickers and a few hundred MB for a 25-year window.

## Lookahead-bias notes

The data pipeline is **as bias-free as free sources allow**, but not perfect.
Points to know:

- **Survivorship.** The universe is the union of (current S&P 500) ∪ (every
  ticker ever added OR removed per Wikipedia). That gets you most delisted
  names that ever mattered in large-cap US equities. It does *not* cover the
  long tail of small-caps that never made the index.
- **Adjusted prices.** yfinance's `adj_close` is back-adjusted for splits +
  cash dividends. Use it for **returns**; do not use it as a dollar price
  level — compute notional with raw `close × volume`.
- **Volume.** Share-volume adjustment for splits is inconsistent in free
  feeds. Prefer dollar-volume for liquidity filters.
- **Point-in-time index membership.** `sp500_pit.parquet` has first/last-seen
  event dates, so you can reconstruct approximate membership on any date. It
  is not vendor-grade PIT and should be spot-checked against known index
  changes before anything goes live.
- **Ticker reassignment.** Symbols like `GM` have mapped to different
  companies after a delisting. yfinance returns whatever the symbol currently
  points at; a known caveat of any free source.

If you outgrow this (e.g. need intraday, fundamentals, or true
delisted-ticker coverage across the broader market), paid options are
Polygon, Norgate, or a CRSP academic licence.

## Layout

```
strats/
├── requirements.txt
├── scripts/
│   ├── fetch_sp500_history.py   # builds survivorship-bias-free universe
│   └── fetch_ohlcv.py           # bulk OHLCV downloader
├── notebooks/
│   └── 01_data_overview.ipynb   # sanity-check the panel
├── src/                         # strategy code goes here
└── data/                        # git-ignored
```
