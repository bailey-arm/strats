#!/usr/bin/env python3
"""
Crypto charts window — live price sparklines for all tracked coins.
Reads from /tmp/crypto_live.json written by crypto_data.py
"""

import json
import time
from datetime import datetime
from pathlib import Path

import plotext as plt
from rich.console import Console
from rich.layout import Layout
from rich.live import Live
from rich.panel import Panel
from rich.text import Text

STATE_FILE = Path("/tmp/crypto_live.json")
COINS = [
    "BTCUSDT", "ETHUSDT", "SOLUSDT", "BNBUSDT", "XRPUSDT",
    "DOGEUSDT", "ADAUSDT", "AVAXUSDT", "LINKUSDT", "DOTUSDT",
]
SHOW   = 5   # coins to chart (top N by index order)
MIN_CH = 7   # minimum chart height


def _load() -> dict | None:
    try:
        return json.loads(STATE_FILE.read_text())
    except Exception:
        return None


def _sparkline(data: list[float], label: str, price: float, pct: float,
               w: int, h: int) -> str:
    plt.clf()
    plt.plot_size(w, h)
    plt.theme("dark")
    sign  = "+" if pct >= 0 else ""
    color = "bright_green" if pct >= 0 else "bright_red"
    plt.title(f"  {label}   ${price:,.4f}   {sign}{pct:.2f}%")
    if len(data) >= 2:
        lo, hi = min(data), max(data)
        plt.ylim(lo * 0.9995, hi * 1.0005)
        plt.plot(data, color=color)
    else:
        plt.plot([price or 0], color=color)
    plt.xticks([])
    return plt.build()


def _charts(state: dict | None, w: int, h: int) -> Text:
    header_h  = 3                              # panel header + gap
    per_chart = max((h - header_h) // SHOW, MIN_CH)
    chart_w   = w - 4

    if state is None:
        return Text("\n  Waiting for data daemon …", style="dim")

    coins  = state.get("coins", {})
    output = ""
    for coin in COINS[:SHOW]:
        d      = coins.get(coin, {})
        data   = d.get("history", [])
        price  = d.get("price", 0)
        pct    = d.get("change", 0)
        label  = coin.replace("USDT", "")
        output += _sparkline(data, label, price, pct, chart_w, per_chart)

    return Text.from_ansi(output)


def _frame(state: dict | None, w: int, h: int) -> Panel:
    ts     = datetime.now().strftime("%Y-%m-%d  %H:%M:%S")
    header = Text(f"  CHARTS  ·  Price History   {ts}", style="bold bright_white")
    charts = _charts(state, w, h)

    layout = Layout()
    layout.split_column(
        Layout(header, size=1),
        Layout(name="gap", size=1),
        Layout(charts),
    )
    return Panel(layout, border_style="grey23", title="[bold cyan]● CHARTS[/]")


def main() -> None:
    console = Console()
    with Live(console=console, screen=True, refresh_per_second=5) as live:
        while True:
            w, h = console.size
            live.update(_frame(_load(), w, h))
            time.sleep(0.2)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        pass
