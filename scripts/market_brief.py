"""Send market-brief email via Resend. Called from GitHub Actions cron."""
from __future__ import annotations

import argparse
import base64
import datetime as dt
import io
import os
import sys
import urllib.request

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import requests
import yfinance as yf


BUCKETS = {
    "Indices": ["^GSPC", "^NDX", "^STOXX50E", "^FTSE", "^N225", "^HSI"],
    "Commodities": ["CL=F", "GC=F", "HG=F", "NG=F"],
    "Single Stocks": ["NVDA", "AAPL", "MSFT", "TSLA", "ASML.AS", "MC.PA"],
}

# (display label, source, source-specific code)
YIELDS: list[tuple[str, str, str]] = [
    ("US 3M",  "yahoo",      "^IRX"),
    ("US 5Y",  "yahoo",      "^FVX"),
    ("US 10Y", "yahoo",      "^TNX"),
    ("US 30Y", "yahoo",      "^TYX"),
    ("UK 10Y", "boe",        "IUDMNZC"),
    ("DE 10Y", "bundesbank", "BBSIS/D.I.ZST.ZI.EUR.S1311.B.A604.R10XX.R.A.A._Z._Z.A"),
]

MAG7 = ["AAPL", "MSFT", "GOOGL", "AMZN", "NVDA", "META", "TSLA"]
SLOT_LABELS = {"am": "AM", "midday": "Midday", "pm": "PM"}

FONT = "Consolas, Menlo, 'Courier New', monospace"
AMBER = "#FA8C00"
BG = "#000000"
FG = "#E8E8E8"
DIM = "#888888"
POS = "#4CFF4C"
NEG = "#FF4C4C"
GRID = "#333333"
LINE_COLORS = [AMBER, "#00D9FF", POS, "#BE85FF", "#FFE14C", "#FF9EC6", NEG]


def fetch_closes(tickers: list[str]) -> pd.DataFrame:
    data = yf.download(
        tickers, period="15d", interval="1d", progress=False, auto_adjust=False,
        group_by="ticker", threads=True,
    )
    if isinstance(data.columns, pd.MultiIndex):
        closes = pd.DataFrame({t: data[t]["Close"] for t in tickers if t in data.columns.levels[0]})
    else:
        closes = data[["Close"]].rename(columns={"Close": tickers[0]})
    return closes.dropna(how="all")


def fetch_boe(code: str) -> pd.Series:
    today = dt.date.today()
    df_from = (today - dt.timedelta(days=30)).strftime("%d/%b/%Y")
    df_to = today.strftime("%d/%b/%Y")
    url = (
        f"https://www.bankofengland.co.uk/boeapps/iadb/fromshowcolumns.asp"
        f"?csv.x=yes&Datefrom={df_from}&Dateto={df_to}&SeriesCodes={code}"
        f"&CSVF=TN&UsingCodes=Y&VPD=Y&VFD=N"
    )
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    txt = urllib.request.urlopen(req, timeout=15).read().decode()
    rows: list[tuple[pd.Timestamp, float]] = []
    for line in txt.splitlines()[1:]:
        parts = line.split(",")
        if len(parts) < 2:
            continue
        try:
            d = pd.Timestamp(dt.datetime.strptime(parts[0].strip(), "%d %b %Y"))
            rows.append((d, float(parts[1])))
        except ValueError:
            continue
    return pd.Series({d: v for d, v in rows}).sort_index() if rows else pd.Series(dtype=float)


def fetch_bundesbank(key: str) -> pd.Series:
    url = f"https://api.statistiken.bundesbank.de/rest/data/{key}?format=csv&lastNObservations=20"
    txt = urllib.request.urlopen(url, timeout=15).read().decode()
    rows: list[tuple[pd.Timestamp, float]] = []
    for line in txt.splitlines()[2:]:
        parts = line.split(";")
        if len(parts) < 2:
            continue
        try:
            d = pd.Timestamp(parts[0].strip())
            v = float(parts[1].replace(",", "."))
            rows.append((d, v))
        except ValueError:
            continue
    return pd.Series({d: v for d, v in rows}).sort_index() if rows else pd.Series(dtype=float)


def fetch_yields() -> pd.DataFrame:
    out: dict[str, pd.Series] = {}
    yahoo_codes = [c for _, s, c in YIELDS if s == "yahoo"]
    if yahoo_codes:
        closes = fetch_closes(yahoo_codes)
        for label, src, code in YIELDS:
            if src == "yahoo" and code in closes.columns:
                out[label] = closes[code]
    for label, src, code in YIELDS:
        if src == "boe":
            try:
                s = fetch_boe(code)
                if not s.empty:
                    out[label] = s
            except Exception as e:
                print(f"BoE fetch failed for {code}: {e}", file=sys.stderr)
        elif src == "bundesbank":
            try:
                s = fetch_bundesbank(code)
                if not s.empty:
                    out[label] = s
            except Exception as e:
                print(f"Bundesbank fetch failed for {code}: {e}", file=sys.stderr)
    return pd.DataFrame(out).sort_index()


def compute_row(series: pd.Series, include_wtd: bool, mode: str = "pct") -> dict:
    s = series.dropna()
    if len(s) < 2:
        return {"last": None, "daily": None, "wtd": None}
    last = float(s.iloc[-1])
    prior = float(s.iloc[-2])
    daily = (last - prior) * 100 if mode == "bps" else (last / prior - 1) * 100
    wtd = None
    if include_wtd:
        last_date = s.index[-1].date()
        days_since_mon = last_date.weekday()
        prior_fri = last_date - dt.timedelta(days=days_since_mon + 3)
        mask = s.index.date <= prior_fri
        base = float(s[mask].iloc[-1]) if mask.any() else float(s.iloc[0])
        wtd = (last - base) * 100 if mode == "bps" else (last / base - 1) * 100
    return {"last": last, "daily": daily, "wtd": wtd}


def fmt_pct(v: float | None) -> str:
    if v is None:
        return f'<span style="color:{DIM}">n/a</span>'
    color = NEG if v < 0 else POS
    weight = "bold" if abs(v) > 1 else "normal"
    return f'<span style="color:{color};font-weight:{weight}">{v:+.2f}%</span>'


def fmt_bps(v: float | None) -> str:
    if v is None:
        return f'<span style="color:{DIM}">n/a</span>'
    color = NEG if v < 0 else POS
    weight = "bold" if abs(v) > 10 else "normal"
    return f'<span style="color:{color};font-weight:{weight}">{v:+.1f} bps</span>'


def fmt_last(v: float | None, mode: str = "pct") -> str:
    if v is None:
        return f'<span style="color:{DIM}">n/a</span>'
    if mode == "bps":
        return f"{v:.2f}%"
    return f"{v:,.2f}"


def build_table(title: str, rows: list[tuple[str, dict]], include_wtd: bool, mode: str = "pct") -> str:
    change_head = "&#916; BPS" if mode == "bps" else "DAILY %"
    wtd_head = "WTD &#916;" if mode == "bps" else "WTD %"
    cols = ["TICKER", "LAST", change_head] + ([wtd_head] if include_wtd else [])
    th_style = (
        f"text-align:right;padding:4px 10px;color:{AMBER};"
        f"border-bottom:1px solid {AMBER};font-weight:normal;letter-spacing:0.5px"
    )
    th_left = th_style.replace("text-align:right", "text-align:left")
    head = "".join(
        f"<th style=\"{th_left if i == 0 else th_style}\">{c}</th>"
        for i, c in enumerate(cols)
    )
    fmt_change = fmt_bps if mode == "bps" else fmt_pct
    body_rows = []
    for ticker, r in rows:
        td_num = f"padding:2px 10px;text-align:right;border-bottom:1px solid {GRID}"
        td_txt = f"padding:2px 10px;text-align:left;border-bottom:1px solid {GRID};color:{FG}"
        tds = [
            f'<td style="{td_txt}">{ticker}</td>',
            f'<td style="{td_num};color:{FG}">{fmt_last(r["last"], mode)}</td>',
            f'<td style="{td_num}">{fmt_change(r["daily"])}</td>',
        ]
        if include_wtd:
            tds.append(f'<td style="{td_num}">{fmt_change(r["wtd"])}</td>')
        body_rows.append(f"<tr>{''.join(tds)}</tr>")
    return (
        f'<div style="color:{AMBER};font-family:{FONT};font-size:13px;'
        f'margin:20px 0 4px;letter-spacing:1px">&#9632; {title.upper()}</div>'
        f'<table cellspacing="0" cellpadding="0" style="font-family:{FONT};'
        f'font-size:13px;border-collapse:collapse;color:{FG};width:100%">'
        f"<thead><tr>{head}</tr></thead><tbody>{''.join(body_rows)}</tbody></table>"
    )


def build_chart(series_by_name: dict[str, pd.Series], mode: str = "pct", days: int = 6) -> str:
    """Return inline <img> tag with dark/amber multi-line chart of last `days` observations."""
    fig, ax = plt.subplots(figsize=(6.4, 2.1), facecolor=BG)
    ax.set_facecolor(BG)
    for spine in ax.spines.values():
        spine.set_color(AMBER)
        spine.set_linewidth(0.5)
    ax.tick_params(colors=FG, labelsize=7, length=0)
    ax.grid(color=GRID, linestyle="-", linewidth=0.3, alpha=0.6)
    ax.axhline(0, color=FG, linewidth=0.3, alpha=0.5)

    drawn = 0
    max_len = 0
    for i, (name, s) in enumerate(series_by_name.items()):
        s = s.dropna().tail(days)
        if len(s) < 2:
            continue
        y = (s - s.iloc[0]) * 100 if mode == "bps" else (s / s.iloc[0] - 1) * 100
        color = LINE_COLORS[drawn % len(LINE_COLORS)]
        x = list(range(len(y)))
        ax.plot(x, y.values, color=color, linewidth=1.2)
        ax.text(x[-1] + 0.15, float(y.iloc[-1]), name,
                color=color, fontsize=7, va="center", family="monospace")
        drawn += 1
        max_len = max(max_len, len(y))

    if drawn == 0:
        plt.close(fig)
        return ""

    ax.set_xticks([])
    ax.set_xlim(-0.3, max_len - 1 + 1.8)
    ax.set_ylabel("bps" if mode == "bps" else "%", color=FG, fontsize=7)

    fig.tight_layout(pad=0.2)
    buf = io.BytesIO()
    fig.savefig(buf, format="png", facecolor=BG, bbox_inches="tight", dpi=80)
    plt.close(fig)

    # Re-encode as indexed (8-bit) PNG — the chart uses a tiny palette so this
    # cuts size ~4-5x with no visible quality loss. Keeps us under Gmail's
    # ~102KB clip threshold.
    try:
        from PIL import Image
        im = Image.open(buf).convert("RGB").quantize(colors=32, method=Image.Quantize.MEDIANCUT)
        out = io.BytesIO()
        im.save(out, format="PNG", optimize=True)
        png_bytes = out.getvalue()
    except Exception:
        png_bytes = buf.getvalue()

    b64 = base64.b64encode(png_bytes).decode()
    return (
        f'<img src="data:image/png;base64,{b64}" alt="" '
        f'style="width:100%;max-width:640px;display:block;margin:6px 0 0">'
    )


def biggest_mover(all_rows: list[tuple[str, dict]]) -> tuple[str, float] | None:
    valid = [(t, r["daily"]) for t, r in all_rows if r["daily"] is not None]
    if not valid:
        return None
    return max(valid, key=lambda x: abs(x[1]))


def build_email(slot: str) -> tuple[str, str]:
    """Fetch data and build the email. Returns (subject, html). No network send."""
    include_wtd = slot == "pm"
    label = SLOT_LABELS[slot]

    price_tickers = sorted({t for b in BUCKETS.values() for t in b} | set(MAG7))
    closes = fetch_closes(price_tickers)
    yields_df = fetch_yields()

    price_rows: dict[str, dict] = {}
    failures: list[str] = []
    for t in price_tickers:
        if t not in closes.columns:
            failures.append(t)
            price_rows[t] = {"last": None, "daily": None, "wtd": None}
            continue
        row = compute_row(closes[t], include_wtd, "pct")
        if row["last"] is None:
            failures.append(t)
        price_rows[t] = row

    yield_rows: dict[str, dict] = {}
    for ylabel, _, _ in YIELDS:
        if ylabel in yields_df.columns:
            row = compute_row(yields_df[ylabel], include_wtd, "bps")
        else:
            row = {"last": None, "daily": None, "wtd": None}
        if row["last"] is None:
            failures.append(ylabel)
        yield_rows[ylabel] = row

    sections: list[str] = []
    price_flat: list[tuple[str, dict]] = []

    # Indices
    idx_tickers = BUCKETS["Indices"]
    rows = [(t, price_rows[t]) for t in idx_tickers]
    price_flat.extend(rows)
    sections.append(
        build_table("Indices", rows, include_wtd, "pct")
        + build_chart({t: closes[t] for t in idx_tickers if t in closes.columns}, "pct")
    )

    # Yields
    y_rows = [(lbl, yield_rows[lbl]) for lbl, _, _ in YIELDS]
    sections.append(
        build_table("Yields", y_rows, include_wtd, "bps")
        + build_chart({lbl: yields_df[lbl] for lbl, _, _ in YIELDS if lbl in yields_df.columns}, "bps")
    )

    # Commodities
    comm_tickers = BUCKETS["Commodities"]
    rows = [(t, price_rows[t]) for t in comm_tickers]
    price_flat.extend(rows)
    sections.append(
        build_table("Commodities", rows, include_wtd, "pct")
        + build_chart({t: closes[t] for t in comm_tickers if t in closes.columns}, "pct")
    )

    # Single Stocks
    ss_tickers = BUCKETS["Single Stocks"]
    rows = [(t, price_rows[t]) for t in ss_tickers]
    price_flat.extend(rows)
    sections.append(
        build_table("Single Stocks", rows, include_wtd, "pct")
        + build_chart({t: closes[t] for t in ss_tickers if t in closes.columns}, "pct")
    )

    # Custom factors: Mag7 equal-weight
    mag7_daily = [price_rows[t]["daily"] for t in MAG7 if price_rows[t]["daily"] is not None]
    mag7_row = {
        "last": None,
        "daily": sum(mag7_daily) / len(mag7_daily) if mag7_daily else None,
        "wtd": None,
    }
    if include_wtd:
        mag7_wtd = [price_rows[t]["wtd"] for t in MAG7 if price_rows[t]["wtd"] is not None]
        mag7_row["wtd"] = sum(mag7_wtd) / len(mag7_wtd) if mag7_wtd else None
    sections.append(build_table("Custom Factors", [("Mag7 (eq-wt)", mag7_row)], include_wtd, "pct"))

    mover = biggest_mover(price_flat)
    if mover:
        mover_color = NEG if mover[1] < 0 else POS
        commentary = (
            f'BIGGEST MOVER: <span style="color:{AMBER};font-weight:bold">{mover[0]}</span> '
            f'<span style="color:{mover_color};font-weight:bold">{mover[1]:+.2f}%</span>'
        )
    else:
        commentary = "NO VALID PRICES RETRIEVED"

    today = dt.date.today().isoformat()
    subject = f"Market brief — {label} — {today}"

    header_bar = (
        f'<div style="background:{BG};border-bottom:2px solid {AMBER};'
        f'padding:10px 14px;font-family:{FONT};font-size:13px;color:{AMBER};'
        f'letter-spacing:2px;font-weight:bold">'
        f'MARKET BRIEF &nbsp;&#9474;&nbsp; {label.upper()} &nbsp;&#9474;&nbsp; {today}'
        f'</div>'
    )
    commentary_bar = (
        f'<div style="padding:10px 14px;font-family:{FONT};font-size:13px;'
        f'color:{FG};border-bottom:1px solid {GRID}">{commentary}</div>'
    )
    footer = (
        f'<div style="padding:12px 14px;margin-top:18px;border-top:1px solid {GRID};'
        f'font-family:{FONT};font-size:11px;color:{DIM};letter-spacing:0.5px">'
        f'FAILED TICKERS: {", ".join(failures)}</div>'
        if failures else ""
    )
    html = (
        f'<table cellspacing="0" cellpadding="0" style="width:100%;background:{BG};'
        f'border-collapse:collapse"><tr><td bgcolor="{BG}" style="background:{BG};'
        f'padding:0">'
        f'<div style="background:{BG};padding:0 14px 16px">'
        f"{header_bar}{commentary_bar}"
        f'<div style="padding:0 4px">{"".join(sections)}</div>'
        f"{footer}"
        f"</div>"
        f"</td></tr></table>"
    )
    return subject, html


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--slot", choices=["am", "midday", "pm"], required=True)
    args = ap.parse_args()

    subject, html = build_email(args.slot)

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
