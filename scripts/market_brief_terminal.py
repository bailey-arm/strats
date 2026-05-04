#!/usr/bin/env python3
"""Terminal market brief — indices, FX, yields, commodities."""

import sys
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

try:
    import pandas as pd
    from rich.console import Console
    from rich.table import Table
    from rich import box as rbox
    from rich.text import Text
    from rich.rule import Rule
    from market_brief import fetch_closes, fetch_yields, compute_row, YIELDS
except ImportError as e:
    print(f"Missing dependency: {e}")
    sys.exit(1)

console = Console()

SECTIONS: dict[str, tuple[list, dict]] = {
    "Indices": (
        ["^GSPC", "^NDX", "^STOXX50E", "^FTSE", "^N225", "^HSI"],
        {"^GSPC": "S&P 500", "^NDX": "NASDAQ 100", "^STOXX50E": "Euro Stoxx 50",
         "^FTSE": "FTSE 100", "^N225": "Nikkei 225", "^HSI": "Hang Seng"},
    ),
    "FX": (
        ["EURUSD=X", "GBPUSD=X", "USDJPY=X", "EURGBP=X", "DX-Y.NYB"],
        {"EURUSD=X": "EUR/USD", "GBPUSD=X": "GBP/USD", "USDJPY=X": "USD/JPY",
         "EURGBP=X": "EUR/GBP", "DX-Y.NYB": "DXY"},
    ),
    "Commodities": (
        ["CL=F", "GC=F", "HG=F", "NG=F"],
        {"CL=F": "WTI Crude", "GC=F": "Gold", "HG=F": "Copper", "NG=F": "Nat Gas"},
    ),
}


def styled_pct(val, bps: bool = False):
    if val is None:
        return Text("n/a", style="dim")
    s = f"{val:+.1f} bps" if bps else f"{val:+.2f}%"
    if val > 0:
        return Text(s, style="bold green")
    if val < 0:
        return Text(s, style="bold red")
    return Text(s, style="dim")


def price_table(title: str, closes: pd.DataFrame, tickers: list, labels: dict):
    t = Table(box=rbox.SIMPLE, header_style="dim")
    t.add_column(title,   style="white",   min_width=16)
    t.add_column("Last",  justify="right", min_width=12)
    t.add_column("Daily", justify="right", min_width=10)
    for ticker in tickers:
        label = labels.get(ticker, ticker)
        if ticker not in closes.columns:
            t.add_row(label, Text("n/a", style="dim"), Text("n/a", style="dim"))
            continue
        row = compute_row(closes[ticker], include_wtd=False, mode="pct")
        last_str = f"{row['last']:,.4f}" if row["last"] is not None else "n/a"
        t.add_row(label, last_str, styled_pct(row["daily"]))
    return t


def main():
    now = datetime.now().strftime("%A %d %b %Y  %H:%M")
    console.print()
    console.print(Rule(f"[bold cyan]Market Brief[/bold cyan]  [dim]{now}[/dim]"))
    console.print()

    all_tickers = [t for tickers, _ in SECTIONS.values() for t in tickers]

    with console.status("[dim]Fetching market data…[/dim]"):
        closes = fetch_closes(all_tickers)

    for section, (tickers, labels) in SECTIONS.items():
        console.print(price_table(section, closes, tickers, labels))

    with console.status("[dim]Fetching yields…[/dim]"):
        try:
            yield_df = fetch_yields()
        except Exception:
            yield_df = pd.DataFrame()

    if not yield_df.empty:
        yt = Table(box=rbox.SIMPLE, header_style="dim")
        yt.add_column("Yields",      style="white",   min_width=12)
        yt.add_column("Last (%)",    justify="right", min_width=10)
        yt.add_column("Daily (bps)", justify="right", min_width=14)
        for label, _, _ in YIELDS:
            if label in yield_df.columns:
                row = compute_row(yield_df[label], include_wtd=False, mode="bps")
                last_str = f"{row['last']:.3f}" if row["last"] is not None else "n/a"
                yt.add_row(label, last_str, styled_pct(row["daily"], bps=True))
        console.print(yt)

    console.print()
    input("  Press Enter to return to menu…")


if __name__ == "__main__":
    main()
