#!/usr/bin/env python3
"""
Crypto prices window — Bid / Ask / Spread / Volume
Reads from /tmp/crypto_live.json written by crypto_data.py
"""

import json
import time
from datetime import datetime
from pathlib import Path

from rich.console import Console
from rich.layout import Layout
from rich.live import Live
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

STATE_FILE = Path("/tmp/crypto_live.json")
COINS = [
    "BTCUSDT", "ETHUSDT", "SOLUSDT", "BNBUSDT", "XRPUSDT",
    "DOGEUSDT", "ADAUSDT", "AVAXUSDT", "LINKUSDT", "DOTUSDT",
]


def _load() -> dict | None:
    try:
        return json.loads(STATE_FILE.read_text())
    except Exception:
        return None


def _table(coins: dict) -> Table:
    t = Table(
        show_header=True,
        header_style="bold cyan",
        border_style="grey23",
        show_lines=True,
        expand=True,
        padding=(0, 1),
    )
    t.add_column("Coin",    style="bold white",  width=8)
    t.add_column("Price",   justify="right",     width=14)
    t.add_column("Bid",     justify="right",     width=14)
    t.add_column("Ask",     justify="right",     width=14)
    t.add_column("Spread",  justify="right",     width=10)
    t.add_column("24 h %",  justify="right",     width=10)
    t.add_column("Vol USDT", justify="right",    width=14)

    for coin in COINS:
        d = coins.get(coin, {})
        if not d or d.get("price", 0) == 0:
            t.add_row(coin.replace("USDT", ""), *["·"] * 6)
            continue

        price  = d["price"]
        bid    = d["bid"]
        ask    = d["ask"]
        spread = (ask - bid) / price * 10_000 if price else 0   # bps
        pct    = d["change"]
        vol    = d["volume"]

        pct_col  = "bright_green" if pct >= 0 else "bright_red"
        sign     = "+" if pct >= 0 else ""
        vol_str  = f"{vol/1e9:.2f} B" if vol >= 1e9 else f"{vol/1e6:.1f} M"

        t.add_row(
            coin.replace("USDT", ""),
            f"{price:,.4f}",
            f"[green]{bid:,.4f}[/]",
            f"[red]{ask:,.4f}[/]",
            f"{spread:.1f} bps",
            f"[{pct_col}]{sign}{pct:.2f}%[/]",
            vol_str,
        )
    return t


def _frame(state: dict | None) -> Panel:
    ts = datetime.now().strftime("%Y-%m-%d  %H:%M:%S")
    header = Text(f"  PRICES  ·  Bid / Ask / Spread / Volume   {ts}", style="bold bright_white")

    if state is None:
        body = Text("\n  Waiting for data daemon …", style="dim")
    else:
        body = _table(state.get("coins", {}))

    layout = Layout()
    layout.split_column(
        Layout(header, size=1),
        Layout(name="gap", size=1),
        Layout(body),
    )
    return Panel(layout, border_style="grey23", title="[bold cyan]● PRICES[/]")


def main() -> None:
    console = Console()
    with Live(console=console, screen=True, refresh_per_second=5) as live:
        while True:
            live.update(_frame(_load()))
            time.sleep(0.2)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        pass
