#!/usr/bin/env python3
"""SX5E equities snapshot — global indices + top movers."""

import sys
from datetime import datetime

try:
    import pandas as pd
    import yfinance as yf
    from rich.console import Console
    from rich.table import Table
    from rich import box as rbox
    from rich.text import Text
    from rich.rule import Rule
except ImportError as e:
    print(f"Missing dependency: {e}")
    sys.exit(1)

console = Console()

INDICES = [
    ("Euro Stoxx 50", "^STOXX50E"),
    ("S&P 500",       "^GSPC"),
    ("NASDAQ 100",    "^NDX"),
    ("DAX",           "^GDAXI"),
    ("FTSE 100",      "^FTSE"),
    ("Nikkei 225",    "^N225"),
]

SX5E_TICKERS = [
    "ADS.DE", "AIR.DE", "ALV.DE", "BAS.DE", "BAYN.DE", "BMW.DE", "DBK.DE", "DB1.DE",
    "DTE.DE", "DHL.DE", "EOAN.DE", "HEN3.DE", "IFX.DE", "MBG.DE", "MRK.DE", "MUV2.DE",
    "SAP.DE", "SIE.DE", "VOW3.DE",
    "AI.PA", "CS.PA", "BNP.PA", "EN.PA", "CAP.PA", "SU.PA", "GLE.PA", "BN.PA", "EL.PA",
    "KER.PA", "OR.PA", "MC.PA", "ORA.PA", "RI.PA", "SAN.PA", "SAF.PA", "SGO.PA", "DG.PA",
    "VIV.PA", "TTE.PA",
    "ASML.AS", "INGA.AS", "PHIA.AS",
    "ENEL.MI", "ENI.MI", "ISP.MI", "UCG.MI",
    "SAN.MC", "BBVA.MC", "IBE.MC", "ITX.MC",
    "CRH.L", "NOKIA.HE", "ABI.BR",
]


def fetch_closes(tickers: list, period: str = "5d") -> pd.DataFrame:
    data = yf.download(
        tickers, period=period, interval="1d", progress=False,
        auto_adjust=False, group_by="ticker", threads=True,
    )
    if data.empty:
        return pd.DataFrame()
    if isinstance(data.columns, pd.MultiIndex):
        return pd.DataFrame({
            t: data[t]["Close"]
            for t in tickers
            if t in data.columns.get_level_values(0)
        }).dropna(how="all")
    return data[["Close"]].rename(columns={"Close": tickers[0]}).dropna(how="all")


def pct_change(series: pd.Series):
    s = series.dropna()
    return (s.iloc[-1] / s.iloc[-2] - 1) * 100 if len(s) >= 2 else None


def styled_pct(val):
    if val is None:
        return Text("n/a", style="dim")
    s = f"{val:+.2f}%"
    if val > 0:
        return Text(s, style="bold green")
    if val < 0:
        return Text(s, style="bold red")
    return Text(s, style="dim")


def main():
    now = datetime.now().strftime("%A %d %b %Y  %H:%M")
    console.print()
    console.print(Rule(f"[bold cyan]Equities Snapshot[/bold cyan]  [dim]{now}[/dim]"))
    console.print()

    with console.status("[dim]Fetching indices…[/dim]"):
        idx_tickers = [t for _, t in INDICES]
        closes = fetch_closes(idx_tickers)

    t = Table(box=rbox.SIMPLE, header_style="dim")
    t.add_column("Index",  style="white",   min_width=16)
    t.add_column("Last",   justify="right", min_width=12)
    t.add_column("Daily",  justify="right", min_width=10)

    for name, ticker in INDICES:
        if ticker not in closes.columns:
            t.add_row(name, Text("n/a", style="dim"), Text("n/a", style="dim"))
            continue
        s = closes[ticker].dropna()
        last = f"{s.iloc[-1]:,.2f}" if len(s) >= 1 else "n/a"
        t.add_row(name, last, styled_pct(pct_change(s)))

    console.print("[bold]Global Indices[/bold]")
    console.print(t)

    with console.status("[dim]Fetching SX5E constituents…[/dim]"):
        stock_closes = fetch_closes(SX5E_TICKERS)

    changes = {
        tkr: chg
        for tkr in SX5E_TICKERS
        if tkr in stock_closes.columns
        for chg in [pct_change(stock_closes[tkr])]
        if chg is not None
    }
    ranked  = sorted(changes.items(), key=lambda x: x[1])
    losers  = ranked[:5]
    gainers = list(reversed(ranked[-5:]))

    m = Table(box=rbox.SIMPLE, header_style="dim")
    m.add_column("Top Gainers", style="bold green", min_width=14)
    m.add_column("",            justify="right",    min_width=8)
    m.add_column("  ",          min_width=2)
    m.add_column("Top Losers",  style="bold red",   min_width=14)
    m.add_column("",            justify="right",    min_width=8)

    for i in range(5):
        g = gainers[i] if i < len(gainers) else ("—", None)
        lo = losers[i]  if i < len(losers)  else ("—", None)
        m.add_row(g[0], styled_pct(g[1]), "", lo[0], styled_pct(lo[1]))

    console.print("[bold]SX5E Top Movers[/bold]")
    console.print(m)
    console.print()
    input("  Press Enter to return to menu…")


if __name__ == "__main__":
    main()
