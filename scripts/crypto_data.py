#!/usr/bin/env python3
"""
Crypto data daemon — Binance WebSocket → /tmp/crypto_live.json
Run by the launcher; do not run directly.
"""

import asyncio
import json
import time
from collections import deque
from pathlib import Path

import websockets

STATE_FILE = Path("/tmp/crypto_live.json")
COINS = [
    "BTCUSDT", "ETHUSDT", "SOLUSDT", "BNBUSDT", "XRPUSDT",
    "DOGEUSDT", "ADAUSDT", "AVAXUSDT", "LINKUSDT", "DOTUSDT",
]
HISTORY = 200

STREAM_URL = (
    "wss://stream.binance.com:9443/stream?streams="
    + "/".join(f"{c.lower()}@ticker" for c in COINS)
)

_state: dict[str, dict] = {
    c: {"price": 0, "bid": 0, "ask": 0, "change": 0, "volume": 0, "history": []}
    for c in COINS
}
_history: dict[str, deque] = {c: deque(maxlen=HISTORY) for c in COINS}


async def ws_stream() -> None:
    while True:
        try:
            async with websockets.connect(STREAM_URL, ping_interval=20) as ws:
                async for raw in ws:
                    msg  = json.loads(raw)
                    data = msg.get("data", {})
                    sym  = data.get("s", "")
                    if sym not in COINS:
                        continue
                    price = float(data["c"])
                    _history[sym].append(price)
                    _state[sym] = {
                        "price":   price,
                        "bid":     float(data["b"]),
                        "ask":     float(data["a"]),
                        "change":  float(data["P"]),
                        "volume":  float(data["q"]),
                        "history": list(_history[sym]),
                    }
        except Exception:
            await asyncio.sleep(2)


async def write_loop() -> None:
    while True:
        payload = {"updated": time.time(), "coins": _state}
        STATE_FILE.write_text(json.dumps(payload))
        await asyncio.sleep(0.15)


async def main() -> None:
    await asyncio.gather(ws_stream(), write_loop())


if __name__ == "__main__":
    asyncio.run(main())
