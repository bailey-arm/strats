"""Send market-brief email via Resend. Called from GitHub Actions cron."""
from __future__ import annotations

import argparse
import datetime as dt
import os
import sys

import pandas as pd
import requests
import yfinance as yf


BUCKETS = {
    "Indices": ["^GSPC", "^NDX", "^STOXX50E", "^FTSE", "^N225", "^HSI"],
    "Bonds": ["^TNX", "IEF", "IBTM.L"],
    "Commodities": ["CL=F", "GC=F", "HG=F", "NG=F"],
    "Single Stocks": ["NVDA", "AAPL", "MSFT", "TSLA", "ASML.AS", "MC.PA"],
}
MAG7 = ["AAPL", "MSFT", "GOOGL", "AMZN", "NVDA", "META", "TSLA"]

SLOT_LABELS = {"am": "AM", "midday": "Midday", "pm": "PM"}


def fetch_closes(tickers: list[str]) -> pd.DataFrame:
    data = yf.download(
        tickers, period="10d", interval="1d", progress=False, auto_adjust=False,
        group_by="ticker", threads=True,
    )
    if isinstance(data.columns, pd.MultiIndex):
        closes = pd.DataFrame({t: data[t]["Close"] for t in tickers if t in data.columns.levels[0]})
    else:
        closes = data[["Close"]].rename(columns={"Close": tickers[0]})
    return closes.dropna(how="all")


def compute_row(series: pd.Series, include_wtd: bool) -> dict:
    s = series.dropna()
    if len(s) < 2:
        return {"last": None, "daily": None, "wtd": None}
    last = float(s.iloc[-1])
    prior = float(s.iloc[-2])
    daily = (last / prior - 1) * 100
    wtd = None
    if include_wtd:
        last_date = s.index[-1].date()
        days_since_mon = last_date.weekday()
        prior_fri = last_date - dt.timedelta(days=days_since_mon + 3)
        mask = s.index.date <= prior_fri
        base = float(s[mask].iloc[-1]) if mask.any() else float(s.iloc[0])
        wtd = (last / base - 1) * 100
    return {"last": last, "daily": daily, "wtd": wtd}


def fmt_pct(v: float | None) -> str:
    if v is None:
        return '<span style="color:#888">n/a</span>'
    color = "#c00" if v < 0 else "#080"
    weight = "bold" if abs(v) > 1 else "normal"
    return f'<span style="color:{color};font-weight:{weight}">{v:+.2f}%</span>'


def fmt_last(v: float | None) -> str:
    if v is None:
        return '<span style="color:#888">n/a</span>'
    return f"{v:,.2f}"


def build_table(title: str, rows: list[tuple[str, dict]], include_wtd: bool) -> str:
    cols = ["Ticker", "Last", "Daily %"] + (["WTD %"] if include_wtd else [])
    head = "".join(f"<th style='text-align:left;padding:4px 10px;border-bottom:1px solid #ccc'>{c}</th>" for c in cols)
    body_rows = []
    for ticker, r in rows:
        tds = [
            f"<td style='padding:4px 10px'>{ticker}</td>",
            f"<td style='padding:4px 10px'>{fmt_last(r['last'])}</td>",
            f"<td style='padding:4px 10px'>{fmt_pct(r['daily'])}</td>",
        ]
        if include_wtd:
            tds.append(f"<td style='padding:4px 10px'>{fmt_pct(r['wtd'])}</td>")
        body_rows.append(f"<tr>{''.join(tds)}</tr>")
    return (
        f"<h3 style='font-family:sans-serif;margin:18px 0 6px'>{title}</h3>"
        f"<table style='font-family:sans-serif;font-size:14px;border-collapse:collapse'>"
        f"<thead><tr>{head}</tr></thead><tbody>{''.join(body_rows)}</tbody></table>"
    )


def biggest_mover(all_rows: list[tuple[str, dict]]) -> tuple[str, float] | None:
    valid = [(t, r["daily"]) for t, r in all_rows if r["daily"] is not None]
    if not valid:
        return None
    return max(valid, key=lambda x: abs(x[1]))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--slot", choices=["am", "midday", "pm"], required=True)
    args = ap.parse_args()

    include_wtd = args.slot == "pm"
    label = SLOT_LABELS[args.slot]

    all_tickers = sorted({t for b in BUCKETS.values() for t in b} | set(MAG7))
    closes = fetch_closes(all_tickers)

    computed: dict[str, dict] = {}
    failures: list[str] = []
    for t in all_tickers:
        if t not in closes.columns:
            failures.append(t)
            computed[t] = {"last": None, "daily": None, "wtd": None}
            continue
        row = compute_row(closes[t], include_wtd)
        if row["last"] is None:
            failures.append(t)
        computed[t] = row

    tables = []
    all_flat = []
    for bucket, tickers in BUCKETS.items():
        rows = [(t, computed[t]) for t in tickers]
        all_flat.extend(rows)
        tables.append(build_table(bucket, rows, include_wtd))

    mag7_daily = [computed[t]["daily"] for t in MAG7 if computed[t]["daily"] is not None]
    mag7_row = {
        "last": None,
        "daily": sum(mag7_daily) / len(mag7_daily) if mag7_daily else None,
        "wtd": None,
    }
    if include_wtd:
        mag7_wtd = [computed[t]["wtd"] for t in MAG7 if computed[t]["wtd"] is not None]
        mag7_row["wtd"] = sum(mag7_wtd) / len(mag7_wtd) if mag7_wtd else None
    tables.append(build_table("Custom Factors", [("Mag7 (eq-wt)", mag7_row)], include_wtd))

    mover = biggest_mover(all_flat)
    commentary = (
        f"Biggest mover: <b>{mover[0]}</b> at {mover[1]:+.2f}%."
        if mover else "No valid prices retrieved."
    )

    today = dt.date.today().isoformat()
    subject = f"Market brief — {label} — {today}"
    footer = (
        f"<p style='font-family:sans-serif;font-size:12px;color:#888;margin-top:20px'>"
        f"Failed tickers: {', '.join(failures)}</p>" if failures else ""
    )
    html = (
        f"<div style='font-family:sans-serif;font-size:14px'>"
        f"<p>{commentary}</p>"
        f"{''.join(tables)}"
        f"{footer}"
        f"</div>"
    )

    api_key = os.environ["RESEND_API_KEY"]
    r = requests.post(
        "https://api.resend.com/emails",
        headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
        json={
            "from": "onboarding@resend.dev",
            "to": ["bailey.arm.business@gmail.com"],
            "subject": subject,
            "html": html,
        },
        timeout=30,
    )
    if r.status_code != 200:
        print(f"Resend error {r.status_code}: {r.text}", file=sys.stderr)
        return 1
    print(f"sent: {r.json().get('id')}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
