"""
move_rate_vol.py — Rate Volatility: MOVE Index and FOMC Event Risk
7-page PDF for trading interview prep.

Usage:
    python scripts/move_rate_vol.py [--out move_rate_vol.pdf]
"""

import argparse
import io
import warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.patches import Patch
import requests
import yfinance as yf

warnings.filterwarnings("ignore")

# ── Style ─────────────────────────────────────────────────────────────────────
plt.rcParams.update({
    'figure.facecolor': '#0d1117', 'axes.facecolor': '#161b22',
    'axes.edgecolor': '#30363d', 'text.color': '#e6edf3',
    'axes.labelcolor': '#e6edf3', 'xtick.color': '#8b949e',
    'ytick.color': '#8b949e', 'grid.color': '#21262d',
    'grid.linestyle': '--', 'grid.alpha': 0.5,
    'axes.titlecolor': '#e6edf3', 'legend.facecolor': '#161b22',
    'legend.edgecolor': '#30363d', 'font.size': 10, 'figure.dpi': 120,
})

START = '2022-01-01'
END   = '2026-05-03'

FOMC = pd.to_datetime([
    '2022-03-16','2022-05-04','2022-06-15','2022-07-27',
    '2022-09-21','2022-11-02','2022-12-14','2023-02-01','2023-03-22',
    '2023-05-03','2023-06-14','2023-07-26','2023-09-20','2023-11-01',
    '2023-12-13','2024-01-31','2024-03-20','2024-05-01','2024-06-12',
    '2024-07-31','2024-09-18','2024-11-07','2024-12-18','2025-01-29',
    '2025-03-19','2025-05-07','2025-06-18','2025-09-17','2026-01-28',
    '2026-03-18','2026-04-29',
])


# ── Data fetchers ─────────────────────────────────────────────────────────────
def _fred(series_id, start=START, end=END):
    url = (f'https://fred.stlouisfed.org/graph/fredgraph.csv'
           f'?id={series_id}&vintage_date={end}')
    r = requests.get(url, timeout=30)
    r.raise_for_status()
    df = pd.read_csv(io.StringIO(r.text), index_col='observation_date')
    df.index = pd.to_datetime(df.index)
    s = pd.to_numeric(df.iloc[:, 0], errors='coerce')
    return s.loc[start:end]


def _yf(ticker, start=START, end=END):
    raw = yf.download(ticker, start=start, end=end, auto_adjust=True, progress=False)
    if raw.empty:
        return None
    s = raw['Close'].squeeze()
    s.index = pd.to_datetime(s.index).tz_localize(None)
    return pd.to_numeric(s, errors='coerce').dropna()


def add_fomc_lines(ax, alpha=0.35, color='#f0883e'):
    ylim = ax.get_ylim()
    for dt in FOMC:
        ax.axvline(dt, color=color, linewidth=0.6, alpha=alpha, zorder=1)
    ax.set_ylim(ylim)


# ── Load all data ─────────────────────────────────────────────────────────────
def load_data():
    print("Fetching FRED yields…")
    yields = {
        '2Y':  _fred('DGS2'),
        '5Y':  _fred('DGS5'),
        '10Y': _fred('DGS10'),
        '30Y': _fred('DGS30'),
    }
    dff = _fred('DFF')

    print("Fetching yfinance series…")
    vix = _yf('^VIX')
    tnx = _yf('^TNX')
    eurusd = _yf('EURUSD=X')

    move_raw = _yf('^MOVE')
    if move_raw is None or move_raw.dropna().empty:
        print("MOVE index unavailable, using computed proxy")
        move_raw = None
    else:
        print(f"^MOVE available — {len(move_raw)} observations")

    # Daily changes in bps
    chg = {tenor: yields[tenor].diff() * 100 for tenor in yields}

    # Realised vol (rolling std)
    rv21 = {t: chg[t].rolling(21).std() for t in chg}
    rv5  = {t: chg[t].rolling(5).std()  for t in chg}

    # MOVE proxy (annualised)
    weights = {'2Y': 0.2, '5Y': 0.3, '10Y': 0.3, '30Y': 0.2}
    proxy_daily = sum(weights[t] * rv21[t] for t in weights)
    move_proxy = proxy_daily * np.sqrt(252)

    return dict(
        yields=yields, dff=dff, vix=vix, tnx=tnx, eurusd=eurusd,
        move_raw=move_raw, chg=chg, rv21=rv21, rv5=rv5,
        move_proxy=move_proxy,
    )


# ── Page helpers ──────────────────────────────────────────────────────────────
def _fig():
    return plt.figure(figsize=(13, 9))


def _shade_levels(ax, series, lo=120, hi=150):
    """Shade MOVE elevated / stressed regions."""
    ax.fill_between(series.index, series, lo,
                    where=(series >= lo) & (series < hi),
                    alpha=0.18, color='#f0883e', label=f'Elevated (>{lo})')
    ax.fill_between(series.index, series, hi,
                    where=(series >= hi),
                    alpha=0.28, color='#f85149', label=f'Stressed (>{hi})')


# ── Page 1 — Cover ────────────────────────────────────────────────────────────
def page_cover(d, pdf):
    fig = _fig()
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_facecolor('#0d1117')
    ax.axis('off')

    move_series = d['move_raw'] if d['move_raw'] is not None else d['move_proxy']
    cur_move   = move_series.dropna().iloc[-1]
    cur_vix    = d['vix'].dropna().iloc[-1]
    ratio      = cur_move / cur_vix
    move_label = "MOVE" if d['move_raw'] is not None else "MOVE Proxy"
    as_of      = move_series.dropna().index[-1].strftime('%Y-%m-%d')

    fig.text(0.5, 0.82, "Rate Volatility", ha='center', fontsize=30,
             color='#e6edf3', fontweight='bold')
    fig.text(0.5, 0.74, "MOVE Index and FOMC Event Risk",
             ha='center', fontsize=20, color='#8b949e')
    fig.text(0.5, 0.66, f"As of {as_of}",
             ha='center', fontsize=13, color='#8b949e')

    # Key stats
    stats = [
        (f"Current {move_label}", f"{cur_move:.1f}"),
        ("Current VIX",           f"{cur_vix:.1f}"),
        (f"{move_label} / VIX",   f"{ratio:.2f}x"),
    ]
    for i, (lbl, val) in enumerate(stats):
        x = 0.22 + i * 0.28
        fig.text(x, 0.55, val,   ha='center', fontsize=24,
                 color='#58a6ff', fontweight='bold')
        fig.text(x, 0.50, lbl,   ha='center', fontsize=11, color='#8b949e')

    blurb = (
        "The MOVE Index (Merrill Lynch Option Volatility Estimate) measures implied volatility\n"
        "in the US Treasury market via 1-month options on 2Y, 5Y, 10Y and 30Y Treasuries.\n"
        "It is the rates-market analogue of the VIX — a fear gauge for bonds. Readings below 80\n"
        "signal calm; 100–120 reflects moderate uncertainty; above 150 indicates acute stress.\n"
        "FOMC decisions are the dominant driver: vol typically elevates in the days before a\n"
        "meeting (uncertainty premium) and collapses after (resolution of event risk). Tracking\n"
        "the MOVE/VIX ratio reveals whether rate or equity vol is in the driving seat."
    )
    fig.text(0.5, 0.26, blurb, ha='center', va='top', fontsize=11,
             color='#c9d1d9', linespacing=1.7)

    fig.text(0.5, 0.07, "Sources: ICE / FRED / Yahoo Finance",
             ha='center', fontsize=9, color='#484f58')
    pdf.savefig(fig, bbox_inches='tight')
    plt.close(fig)


# ── Page 2 — MOVE history ─────────────────────────────────────────────────────
def page_move_history(d, pdf):
    fig, ax = plt.subplots(figsize=(13, 7))
    fig.patch.set_facecolor('#0d1117')

    proxy = d['move_proxy'].dropna()
    src_note = ""
    if d['move_raw'] is not None:
        mv = d['move_raw'].dropna()
        ax.plot(mv.index, mv, color='#58a6ff', lw=1.6, label='MOVE (^MOVE)', zorder=3)
        src_note = " — ICE data + computed proxy overlay"
    else:
        src_note = " — computed proxy (realised vol weighted average)"

    ax.plot(proxy.index, proxy, color='#f0883e', lw=1.2,
            alpha=0.75, linestyle='--', label='MOVE Proxy (computed)', zorder=2)

    display = d['move_raw'].dropna() if d['move_raw'] is not None else proxy
    _shade_levels(ax, display)
    add_fomc_lines(ax)
    ax.axhline(120, color='#f0883e', lw=0.5, linestyle=':', alpha=0.5)
    ax.axhline(150, color='#f85149', lw=0.5, linestyle=':', alpha=0.5)

    ax.set_title(f'MOVE Index — 3-Year History{src_note}', pad=12)
    ax.set_ylabel('MOVE Level')
    ax.grid(True)
    ax.legend(loc='upper right')
    fomc_line = plt.Line2D([0], [0], color='#f0883e', lw=0.8, linestyle='-',
                           alpha=0.7, label='FOMC date')
    handles, labels = ax.get_legend_handles_labels()
    ax.legend(handles + [fomc_line], labels + ['FOMC date'], loc='upper right')
    fig.tight_layout()
    pdf.savefig(fig, bbox_inches='tight')
    plt.close(fig)


# ── Page 3 — MOVE vs VIX ──────────────────────────────────────────────────────
def page_move_vs_vix(d, pdf):
    fig = plt.figure(figsize=(13, 9))
    fig.patch.set_facecolor('#0d1117')
    gs = gridspec.GridSpec(2, 1, height_ratios=[2, 1], hspace=0.35)

    move_s = (d['move_raw'].dropna() if d['move_raw'] is not None
              else d['move_proxy'].dropna())
    vix_s  = d['vix'].dropna()
    common = move_s.index.intersection(vix_s.index)
    move_c = move_s.loc[common]
    vix_c  = vix_s.loc[common]
    ratio  = move_c / vix_c

    ax1 = fig.add_subplot(gs[0])
    ax2 = ax1.twinx()
    move_label = "MOVE" if d['move_raw'] is not None else "MOVE Proxy"
    ax1.plot(move_c.index, move_c, color='#58a6ff', lw=1.5, label=move_label)
    ax2.plot(vix_c.index,  vix_c,  color='#3fb950', lw=1.5, label='VIX', alpha=0.8)
    add_fomc_lines(ax1)
    ax1.set_ylabel(move_label, color='#58a6ff')
    ax2.set_ylabel('VIX', color='#3fb950')
    ax1.set_title(f'{move_label} vs VIX — Dual Axis', pad=10)
    ax1.grid(True)
    lines1, lbl1 = ax1.get_legend_handles_labels()
    lines2, lbl2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, lbl1 + lbl2, loc='upper left')

    ax3 = fig.add_subplot(gs[1])
    ax3.plot(ratio.index, ratio, color='#d2a8ff', lw=1.3)
    ax3.axhline(ratio.mean(), color='#8b949e', lw=0.8, linestyle='--',
                label=f'Mean {ratio.mean():.2f}x')
    ax3.fill_between(ratio.index, ratio, ratio.mean(),
                     where=(ratio > ratio.mean()), alpha=0.2, color='#d2a8ff')
    add_fomc_lines(ax3)
    ax3.set_ylabel(f'{move_label}/VIX Ratio')
    ax3.set_title('Rate Vol Premium over Equity Vol', pad=8)
    ax3.legend(loc='upper right')
    ax3.grid(True)

    fig.tight_layout()
    pdf.savefig(fig, bbox_inches='tight')
    plt.close(fig)


# ── Page 4 — Realised vol by tenor ───────────────────────────────────────────
def page_tenor_rv(d, pdf):
    fig, ax = plt.subplots(figsize=(13, 7))
    fig.patch.set_facecolor('#0d1117')
    colors = {'2Y': '#58a6ff', '5Y': '#3fb950', '10Y': '#f0883e', '30Y': '#d2a8ff'}
    for tenor, col in colors.items():
        s = d['rv21'][tenor].dropna()
        ax.plot(s.index, s, color=col, lw=1.4, label=f'{tenor} 21d RV')
    add_fomc_lines(ax)
    ax.set_title('21-Day Realised Vol by Tenor (bps/day) — Term Structure of Rate Vol', pad=12)
    ax.set_ylabel('Realised Vol (bps/day)')
    ax.legend(loc='upper right')
    ax.grid(True)
    fig.tight_layout()
    pdf.savefig(fig, bbox_inches='tight')
    plt.close(fig)


# ── Page 5 — Vol around FOMC ──────────────────────────────────────────────────
def page_fomc_event(d, pdf):
    rv = d['rv21']['10Y'].dropna()
    results = []
    fomc_past = [f for f in FOMC if f <= rv.index[-1]][-15:]
    for dt in fomc_past:
        pre  = rv.loc[(rv.index >= dt - pd.Timedelta(days=14)) &
                      (rv.index <  dt)].tail(10)
        post = rv.loc[(rv.index >  dt) &
                      (rv.index <= dt + pd.Timedelta(days=14))].head(10)
        if len(pre) >= 3 and len(post) >= 3:
            results.append({'date': dt.strftime('%b %y'),
                            'pre': pre.mean(), 'post': post.mean()})

    if not results:
        return

    df = pd.DataFrame(results)
    x  = np.arange(len(df))
    w  = 0.38

    fig, axes = plt.subplots(2, 1, figsize=(13, 9),
                             gridspec_kw={'height_ratios': [3, 1]}, hspace=0.4)
    fig.patch.set_facecolor('#0d1117')
    ax = axes[0]
    ax.bar(x - w/2, df['pre'],  w, color='#58a6ff', label='Pre-FOMC (10d avg)')
    ax.bar(x + w/2, df['post'], w, color='#f0883e', label='Post-FOMC (10d avg)')
    ax.set_xticks(x)
    ax.set_xticklabels(df['date'], rotation=45, ha='right', fontsize=8)
    ax.set_ylabel('10Y Realised Vol (bps/day)')
    ax.set_title('10Y Rate Realised Vol: 10 Days Pre vs Post FOMC (last 15 meetings)', pad=12)
    ax.legend()
    ax.grid(True, axis='y')

    ax2 = axes[1]
    premium = df['pre'] - df['post']
    colors_bar = ['#3fb950' if v > 0 else '#f85149' for v in premium]
    ax2.bar(x, premium, color=colors_bar, alpha=0.85)
    ax2.axhline(0, color='#8b949e', lw=0.7)
    avg_pre = premium[premium > 0].mean()
    ax2.axhline(avg_pre, color='#f0883e', lw=0.8, linestyle='--',
                label=f'Avg pre-FOMC premium: {avg_pre:.2f} bps/day')
    ax2.set_xticks(x)
    ax2.set_xticklabels(df['date'], rotation=45, ha='right', fontsize=8)
    ax2.set_ylabel('Pre − Post (bps/day)')
    ax2.set_title('Pre-FOMC Vol Premium', pad=8)
    ax2.legend(loc='upper right', fontsize=9)
    ax2.grid(True, axis='y')

    fig.tight_layout()
    pdf.savefig(fig, bbox_inches='tight')
    plt.close(fig)


# ── Page 6 — Vol term structure snapshots ────────────────────────────────────
def page_term_structure(d, pdf):
    today    = pd.Timestamp(END)
    snap_dates = {
        '12m ago':          today - pd.DateOffset(months=12),
        '6m ago':           today - pd.DateOffset(months=6),
        'Warsh nom (13-Nov-25)': pd.Timestamp('2025-11-13'),
        'Today':            today,
    }
    tenors = ['2Y', '5Y', '10Y', '30Y']
    colors = ['#58a6ff', '#3fb950', '#f0883e', '#d2a8ff']

    fig, ax = plt.subplots(figsize=(11, 7))
    fig.patch.set_facecolor('#0d1117')

    for (label, snap_dt), col in zip(snap_dates.items(), colors):
        vals = []
        for t in tenors:
            s = d['rv21'][t].dropna()
            idx = s.index.get_indexer([snap_dt], method='nearest')[0]
            vals.append(s.iloc[idx] if idx >= 0 else np.nan)
        ax.plot(tenors, vals, marker='o', lw=1.8, color=col, label=label)

    ax.set_title('Realised Vol Term Structure — Key Snapshots (21d RV, bps/day)', pad=12)
    ax.set_ylabel('21d Realised Vol (bps/day)')
    ax.set_xlabel('Tenor')
    ax.legend(loc='upper right')
    ax.grid(True)
    fig.tight_layout()
    pdf.savefig(fig, bbox_inches='tight')
    plt.close(fig)


# ── Page 7 — Rate vol vs FX vol ───────────────────────────────────────────────
def page_rate_vs_fx(d, pdf):
    if d['eurusd'] is None or d['eurusd'].dropna().empty:
        print("EURUSD=X unavailable — skipping page 7")
        return

    fx_chg  = d['eurusd'].pct_change() * 100
    fx_rv   = fx_chg.rolling(21).std() * np.sqrt(252)

    proxy   = d['move_proxy'].dropna()
    common  = proxy.index.intersection(fx_rv.dropna().index)
    p_c     = proxy.loc[common]
    f_c     = fx_rv.loc[common]
    corr_r  = p_c.rolling(63).corr(f_c)

    fig = plt.figure(figsize=(13, 9))
    fig.patch.set_facecolor('#0d1117')
    gs = gridspec.GridSpec(2, 1, height_ratios=[2, 1], hspace=0.38)

    ax1 = fig.add_subplot(gs[0])
    ax2 = ax1.twinx()
    ax1.plot(p_c.index, p_c, color='#58a6ff', lw=1.5, label='MOVE Proxy (ann.)')
    ax2.plot(f_c.index, f_c, color='#3fb950', lw=1.3, label='EUR/USD 21d RV (ann. %)', alpha=0.8)
    add_fomc_lines(ax1)
    ax1.set_ylabel('MOVE Proxy', color='#58a6ff')
    ax2.set_ylabel('EUR/USD RV (%)', color='#3fb950')
    ax1.set_title('Rate Vol (MOVE Proxy) vs EUR/USD FX Vol', pad=12)
    ax1.grid(True)
    l1, b1 = ax1.get_legend_handles_labels()
    l2, b2 = ax2.get_legend_handles_labels()
    ax1.legend(l1 + l2, b1 + b2, loc='upper left')

    ax3 = fig.add_subplot(gs[1])
    ax3.plot(corr_r.index, corr_r, color='#d2a8ff', lw=1.3)
    ax3.axhline(0, color='#8b949e', lw=0.7)
    ax3.axhline(corr_r.mean(), color='#8b949e', lw=0.8, linestyle='--',
                label=f'Mean corr: {corr_r.mean():.2f}')
    ax3.fill_between(corr_r.index, corr_r, 0,
                     where=(corr_r < 0), alpha=0.2, color='#f85149',
                     label='Idiosyncratic rate shock (neg. corr)')
    ax3.set_ylabel('63d Rolling Corr')
    ax3.set_title('MOVE vs FX Vol: 63d Rolling Correlation — negative = rate-idiosyncratic shock',
                  pad=8)
    ax3.set_ylim(-1.1, 1.1)
    ax3.legend(loc='lower right', fontsize=9)
    ax3.grid(True)

    fig.tight_layout()
    pdf.savefig(fig, bbox_inches='tight')
    plt.close(fig)


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description='Rate vol PDF — MOVE & FOMC')
    parser.add_argument('--out', default='move_rate_vol.pdf',
                        help='Output PDF path (default: move_rate_vol.pdf)')
    args = parser.parse_args()

    d = load_data()

    print(f"Writing PDF → {args.out}")
    with PdfPages(args.out) as pdf:
        page_cover(d, pdf)
        page_move_history(d, pdf)
        page_move_vs_vix(d, pdf)
        page_tenor_rv(d, pdf)
        page_fomc_event(d, pdf)
        page_term_structure(d, pdf)
        page_rate_vs_fx(d, pdf)

        meta = pdf.infodict()
        meta['Title']   = 'Rate Volatility: MOVE Index and FOMC Event Risk'
        meta['Author']  = 'move_rate_vol.py'
        meta['Subject'] = 'Fixed income vol, MOVE, FOMC event study'

    print("Done.")


if __name__ == '__main__':
    main()
