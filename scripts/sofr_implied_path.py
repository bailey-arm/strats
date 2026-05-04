"""
SOFR & Fed Rate Path Dynamics — multi-page PDF.

Implied rate path is proxied via Treasury bill / short-term note yields
(3M, 6M, 1Y, 2Y T-bills and notes) — the cleanest free daily forward-rate
signal. The 3M T-bill prices the expected Fed Funds rate over the next quarter;
1Y prices the expected average over the next year.  No CME subscription needed.

Usage:
    python scripts/sofr_implied_path.py [--out sofr_implied_path.pdf]

Dependencies: pip install pandas requests matplotlib yfinance
"""

import argparse, io, warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import matplotlib.patches as mpatches
from matplotlib.backends.backend_pdf import PdfPages
import requests

# ── Config ────────────────────────────────────────────────────────────────────
TODAY        = pd.Timestamp('2026-05-04')
WINDOW_START = TODAY - pd.DateOffset(months=18)
FETCH_START  = TODAY - pd.DateOffset(months=36)   # 3Y for context

FOMC_DATES = pd.to_datetime([
    '2023-02-01','2023-03-22','2023-05-03','2023-06-14',
    '2023-07-26','2023-09-20','2023-11-01','2023-12-13',
    '2024-01-31','2024-03-20','2024-05-01','2024-06-12',
    '2024-07-31','2024-09-18','2024-11-07','2024-12-18',
    '2025-01-29','2025-03-19','2025-05-07','2025-06-18',
    '2025-09-17','2026-01-28','2026-03-18','2026-04-29',
])

# Snapshot dates for the "forward curve evolution" chart
SNAPSHOTS = {
    '12m ago':         TODAY - pd.DateOffset(months=12),
    '6m ago':          TODAY - pd.DateOffset(months=6),
    'Warsh nom':       pd.Timestamp('2025-11-13'),
    'Today':           TODAY,
}

# ── Style ─────────────────────────────────────────────────────────────────────
plt.rcParams.update({
    'figure.facecolor':'#0d1117', 'axes.facecolor':'#161b22',
    'axes.edgecolor':  '#30363d', 'text.color':    '#e6edf3',
    'axes.labelcolor': '#e6edf3', 'xtick.color':   '#8b949e',
    'ytick.color':     '#8b949e', 'grid.color':    '#21262d',
    'grid.linestyle':  '--',      'grid.alpha':    0.5,
    'axes.titlecolor': '#e6edf3', 'legend.facecolor':'#161b22',
    'legend.edgecolor':'#30363d', 'font.size':     10,
    'figure.dpi':      120,
})

C = {
    '3M':  '#58a6ff', '6M':  '#3fb950', '1Y':  '#d29922', '2Y':  '#f78166',
    'dff': '#e6edf3', 'lo':  '#3fb950', 'hi':  '#f78166',
    'be':  '#bc8cff', 'real':'#56d364',
}

# ── Data ──────────────────────────────────────────────────────────────────────
def _fred(sid: str) -> pd.Series:
    url = (f'https://fred.stlouisfed.org/graph/fredgraph.csv'
           f'?id={sid}&vintage_date={TODAY.strftime("%Y-%m-%d")}')
    r = requests.get(url, timeout=30)
    r.raise_for_status()
    df = pd.read_csv(io.StringIO(r.text), index_col='observation_date')
    df.index = pd.to_datetime(df.index)
    return pd.to_numeric(df.iloc[:, 0], errors='coerce').loc[FETCH_START:]


def fetch_all() -> dict:
    print('Fetching FRED data ...')
    series = {
        # Implied path proxies (T-bill / note yields)
        '3M':  'DGS3MO',
        '6M':  'DGS6MO',
        '1Y':  'DGS1',
        '2Y':  'DGS2',
        # Actual policy
        'dff':  'DFF',
        'lo':   'DFEDTARL',
        'hi':   'DFEDTARU',
        # Inflation
        'be10': 'T10YIE',
        'be5':  'T5YIE',
        'fwd':  'T5YIFR',   # 5Y5Y forward inflation
    }
    data = {}
    for label, sid in series.items():
        print(f'  {sid} ({label})', end='', flush=True)
        try:
            data[label] = _fred(sid)
            print(' ✓')
        except Exception as e:
            print(f' ✗  ({e})')
    return data


# ── Helpers ───────────────────────────────────────────────────────────────────
def _fomc_lines(ax, alpha=0.25):
    for dt in FOMC_DATES:
        if WINDOW_START <= dt <= TODAY:
            ax.axvline(dt, color='#8b949e', lw=0.6, ls=':', alpha=alpha)


def _datefmt(ax):
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%b\n%Y'))
    ax.xaxis.set_major_locator(mdates.MonthLocator(interval=2))


def _xlim(ax, start=None):
    ax.set_xlim((start or WINDOW_START) - pd.Timedelta(days=5),
                TODAY + pd.Timedelta(days=10))


def _nom_line(ax):
    ax.axvline(pd.Timestamp('2025-11-13'), color='#d29922',
               lw=1.3, ls='--', alpha=0.9, label='Warsh nom.')


def _nearest(s: pd.Series, dt: pd.Timestamp):
    idx = s.dropna().index
    pos = idx.get_indexer([dt], method='nearest')[0]
    return s.dropna().iloc[pos]


# ── Pages ─────────────────────────────────────────────────────────────────────
def page_cover(pdf, data):
    fig, ax = plt.subplots(figsize=(11, 8.5))
    ax.axis('off')

    dff_now   = data['dff'].dropna().iloc[-1]
    lo_now    = data['lo'].dropna().iloc[-1]
    hi_now    = data['hi'].dropna().iloc[-1]
    y1_now    = data['1Y'].dropna().iloc[-1]
    implied_cuts = (y1_now - dff_now) / 0.25
    be10_now  = data.get('be10', pd.Series()).dropna()
    be10_val  = be10_now.iloc[-1] if not be10_now.empty else float('nan')

    ax.text(0.5, 0.87, 'SOFR & Fed Rate Path Dynamics',
            ha='center', fontsize=28, fontweight='bold',
            color='#e6edf3', transform=ax.transAxes)
    ax.text(0.5, 0.79, 'Implied path from Treasury bill yields  ·  Event-study around FOMC',
            ha='center', fontsize=14, color='#8b949e', transform=ax.transAxes)

    stats = [
        ('Effective Fed Funds',   f'{dff_now:.2f}%'),
        ('Target range',          f'{lo_now:.2f}% – {hi_now:.2f}%'),
        ('1Y T-bill (implied avg)',f'{y1_now:.2f}%'),
        ('Implied cuts (1Y)',      f'{implied_cuts:+.1f}  ×25bp'),
        ('10Y breakeven',          f'{be10_val:.2f}%' if not np.isnan(be10_val) else 'n/a'),
    ]
    xs = [0.12, 0.38, 0.62, 0.88]
    for i, (label, val) in enumerate(stats):
        x = 0.1 + i * 0.21
        ax.text(x, 0.60, val,   ha='center', fontsize=18, fontweight='bold',
                color='#58a6ff', transform=ax.transAxes)
        ax.text(x, 0.54, label, ha='center', fontsize=9, color='#8b949e',
                transform=ax.transAxes)

    body = (
        'Methodology\n'
        '  Treasury bill and short-term note yields are the cleanest free proxy\n'
        '  for the market-implied rate path:\n'
        '    · 3M T-bill  ≈  expected Fed Funds over the next quarter\n'
        '    · 6M T-bill  ≈  expected average over 6 months\n'
        '    · 1Y T-note  ≈  expected average Fed Funds over 1 year\n'
        '    · 2Y T-note  ≈  2-year expectation + small term premium\n\n'
        '  Implied cuts = (1Y yield − current DFF) ÷ 0.25\n'
        '  A negative number means hikes are priced.\n\n'
        'Warsh context\n'
        '  Nomination date (2025-11-13) is annotated on all time-series charts.\n'
        '  The key question: did the implied rate path re-price upward (hawkish\n'
        '  Warsh premium) or downward (growth/recession concerns)?'
    )
    ax.text(0.08, 0.47, body, ha='left', va='top', fontsize=10,
            color='#e6edf3', transform=ax.transAxes, linespacing=1.65,
            fontfamily='monospace')

    pdf.savefig(fig, bbox_inches='tight')
    plt.close(fig)


def page_implied_path_evolution(pdf, data):
    """3M / 6M / 1Y / 2Y yield history — the implied path through time."""
    fig, axes = plt.subplots(2, 1, figsize=(14, 10),
                             gridspec_kw={'height_ratios': [2, 1]})

    ax = axes[0]
    for key in ['3M', '6M', '1Y', '2Y']:
        if key not in data:
            continue
        s = data[key].loc[WINDOW_START:]
        ax.plot(s.index, s.values, color=C[key], lw=1.8, label=f'{key} T-bill/note')

    # Actual fed funds corridor
    if 'lo' in data and 'hi' in data:
        lo = data['lo'].loc[WINDOW_START:]
        hi = data['hi'].loc[WINDOW_START:]
        ax.fill_between(lo.index, lo.values, hi.values,
                        color='#e6edf3', alpha=0.06, label='FOMC target range')
    if 'dff' in data:
        s = data['dff'].loc[WINDOW_START:]
        ax.plot(s.index, s.values, color=C['dff'], lw=1.2, ls=':', label='Effective DFF')

    _fomc_lines(ax)
    _nom_line(ax)
    _xlim(ax)
    _datefmt(ax)
    ax.set_title('Implied rate path via T-bill/note yields  ·  FOMC dates (dotted)')
    ax.set_ylabel('Yield (%)')
    ax.legend(fontsize=8, ncol=3)
    ax.grid(True)

    # Sub-panel: implied cuts
    ax2 = axes[1]
    if '1Y' in data and 'dff' in data:
        dff_last = data['dff'].dropna().iloc[-1]
        impl = (data['1Y'].loc[WINDOW_START:] - dff_last) / 0.25
        pos  = impl.clip(lower=0)
        neg  = impl.clip(upper=0)
        ax2.fill_between(impl.index, 0, pos.values, color='#3fb950', alpha=0.4, label='Cuts priced')
        ax2.fill_between(impl.index, 0, neg.values, color='#f78166', alpha=0.4, label='Hikes priced')
        ax2.plot(impl.index, impl.values, color='#e6edf3', lw=1.2)
        ax2.axhline(0, color='#8b949e', lw=0.7, ls=':')
    _fomc_lines(ax2)
    _nom_line(ax2)
    _xlim(ax2)
    _datefmt(ax2)
    ax2.set_title('Implied 25bp cuts priced by 1Y T-bill  (vs current DFF)')
    ax2.set_ylabel('# cuts')
    ax2.legend(fontsize=8)
    ax2.grid(True)

    fig.suptitle('Implied Rate Path Evolution', fontsize=14, y=1.01)
    plt.tight_layout()
    pdf.savefig(fig, bbox_inches='tight')
    plt.close(fig)


def page_forward_curve_snapshots(pdf, data):
    """Forward curve (3M/6M/1Y/2Y yield) at 4 key dates."""
    tenors = ['3M', '6M', '1Y', '2Y']
    tenor_x = [0.25, 0.5, 1, 2]   # x-axis in years

    snap_colors = ['#58a6ff', '#3fb950', '#d29922', '#e6edf3']
    fig, ax = plt.subplots(figsize=(12, 6))

    for (label, dt), color in zip(SNAPSHOTS.items(), snap_colors):
        vals = []
        for t in tenors:
            if t in data:
                try:
                    vals.append(_nearest(data[t], dt))
                except Exception:
                    vals.append(np.nan)
            else:
                vals.append(np.nan)
        if any(not np.isnan(v) for v in vals):
            ax.plot(tenor_x, vals, color=color, marker='o', ms=6, lw=2.2, label=label)

    # Actual current rate
    if 'lo' in data:
        lo = data['lo'].dropna().iloc[-1]
        ax.axhline(lo, color='#8b949e', lw=1, ls='--', alpha=0.6, label=f'Target floor ({lo:.2f}%)')

    ax.set_xticks(tenor_x)
    ax.set_xticklabels(['3M', '6M', '1Y', '2Y'])
    ax.set_xlabel('Tenor')
    ax.set_ylabel('Yield (%)')
    ax.set_title('Implied forward curve at key dates\n(shape = slope of expected rate path)')
    ax.legend(fontsize=9)
    ax.grid(True)

    fig.suptitle('Forward Curve Snapshots — How the Rate Path Re-Priced', fontsize=14, y=1.02)
    plt.tight_layout()
    pdf.savefig(fig, bbox_inches='tight')
    plt.close(fig)


def page_actual_vs_expected(pdf, data):
    """Actual DFF vs where the 1Y T-bill was pricing 12 months prior."""
    if '1Y' not in data or 'dff' not in data:
        return

    actual = data['dff'].dropna()
    fwd_1y = data['1Y'].dropna()

    # Shift forward curve back 252 days to align with actual realisation
    fwd_shifted = fwd_1y.shift(-252)
    aligned     = pd.concat([actual.rename('actual'),
                             fwd_shifted.rename('expected_1y_ago')], axis=1).dropna()
    aligned     = aligned.loc[WINDOW_START:]
    surprise    = aligned['actual'] - aligned['expected_1y_ago']

    fig, axes = plt.subplots(2, 1, figsize=(14, 9),
                             gridspec_kw={'height_ratios': [2, 1]})

    ax = axes[0]
    ax.plot(aligned.index, aligned['actual'],        color='#e6edf3', lw=1.8, label='Actual DFF')
    ax.plot(aligned.index, aligned['expected_1y_ago'],color='#58a6ff', lw=1.5, ls='--',
            label='1Y T-bill priced 12m earlier')
    ax.fill_between(aligned.index, aligned['expected_1y_ago'], aligned['actual'],
                    color='#d29922', alpha=0.15, label='Surprise gap')
    _fomc_lines(ax)
    _nom_line(ax)
    _xlim(ax)
    _datefmt(ax)
    ax.set_title('Actual Fed Funds vs 1Y forward priced 12 months prior')
    ax.set_ylabel('Rate (%)')
    ax.legend(fontsize=8)
    ax.grid(True)

    ax2 = axes[1]
    pos = surprise.clip(lower=0)
    neg = surprise.clip(upper=0)
    ax2.fill_between(surprise.index, 0, pos.values, color='#f78166', alpha=0.5,
                     label='Rates came in higher than expected')
    ax2.fill_between(surprise.index, 0, neg.values, color='#3fb950', alpha=0.5,
                     label='Rates came in lower than expected')
    ax2.plot(surprise.index, surprise.values, color='#e6edf3', lw=1)
    ax2.axhline(0, color='#8b949e', lw=0.7, ls=':')
    _fomc_lines(ax2)
    _nom_line(ax2)
    _xlim(ax2)
    _datefmt(ax2)
    ax2.set_title('Surprise: actual − expected (pp)')
    ax2.set_ylabel('pp')
    ax2.legend(fontsize=8)
    ax2.grid(True)

    fig.suptitle('Actual Path vs Market Expectations — Forecast Errors', fontsize=14, y=1.01)
    plt.tight_layout()
    pdf.savefig(fig, bbox_inches='tight')
    plt.close(fig)


def page_fomc_event_study(pdf, data):
    """
    For each FOMC meeting: 10Y yield change in 5 days before vs 5 days after.
    Shows whether meetings were 'hawkish surprises' or 'dovish surprises'.
    """
    if '1Y' not in data:
        return

    y1 = data['1Y'].dropna()
    past_fomc = [d for d in FOMC_DATES if d >= WINDOW_START and d <= TODAY]

    pre_chg, post_chg, labels = [], [], []
    for dt in past_fomc:
        try:
            window = y1.loc[dt - pd.Timedelta(days=10): dt + pd.Timedelta(days=10)]
            pre  = window.loc[:dt].iloc[-6:-1]
            post = window.loc[dt:].iloc[1:6]
            if len(pre) >= 3 and len(post) >= 3:
                pre_chg.append((pre.iloc[-1] - pre.iloc[0]) * 100)
                post_chg.append((post.iloc[-1] - post.iloc[0]) * 100)
                labels.append(dt.strftime('%b\n%Y'))
        except Exception:
            pass

    if not pre_chg:
        return

    fig, axes = plt.subplots(2, 1, figsize=(14, 9))
    x = np.arange(len(labels))
    w = 0.35

    ax = axes[0]
    bars_pre  = ax.bar(x - w/2, pre_chg,  w, label='5d pre-meeting',  color='#58a6ff', alpha=0.85)
    bars_post = ax.bar(x + w/2, post_chg, w, label='5d post-meeting', color='#d29922', alpha=0.85)
    ax.axhline(0, color='#8b949e', lw=0.7, ls=':')
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=8)
    ax.set_title('1Y T-bill yield change around FOMC meetings (bps)')
    ax.set_ylabel('Change (bps)')
    ax.legend(fontsize=8)
    ax.grid(True, axis='y')

    # Surprise = post - pre (net directional move from meeting)
    ax2 = axes[1]
    surprise = [p - r for p, r in zip(post_chg, pre_chg)]
    colors   = ['#3fb950' if s < 0 else '#f78166' for s in surprise]
    ax2.bar(x, surprise, color=colors, alpha=0.85)
    ax2.axhline(0, color='#8b949e', lw=0.7, ls=':')
    ax2.axhline(np.mean(surprise), color='#bc8cff', lw=1.2, ls='--',
                label=f'Avg surprise: {np.mean(surprise):+.1f}bps')
    ax2.set_xticks(x)
    ax2.set_xticklabels(labels, fontsize=8)
    ax2.set_title('Net hawkish/dovish surprise (post − pre, bps)  ·  green = dovish')
    ax2.set_ylabel('bps')
    ax2.legend(fontsize=8)
    ax2.grid(True, axis='y')

    fig.suptitle('FOMC Event Study — Rate Surprises', fontsize=14, y=1.01)
    plt.tight_layout()
    pdf.savefig(fig, bbox_inches='tight')
    plt.close(fig)


def page_breakeven_real(pdf, data):
    """Breakeven inflation, nominal 10Y, and implied real rate."""
    if 'be10' not in data or '2Y' not in data:
        return

    dgs10_key = None
    # Try to fetch 10Y if not in data (fetch on the fly)
    try:
        dgs10 = _fred('DGS10').loc[WINDOW_START:]
    except Exception:
        return

    be   = data['be10'].loc[WINDOW_START:]
    fwd  = data.get('fwd', pd.Series()).loc[WINDOW_START:]
    real = (dgs10 - be).dropna()

    fig, axes = plt.subplots(2, 1, figsize=(14, 9),
                             gridspec_kw={'height_ratios': [2, 1]})

    ax = axes[0]
    ax.plot(dgs10.index, dgs10.values, color='#58a6ff', lw=1.8, label='10Y nominal')
    ax.plot(be.index,    be.values,    color='#f78166', lw=1.8, label='10Y breakeven')
    ax.plot(real.index,  real.values,  color='#3fb950', lw=1.8, label='10Y real (nominal − BE)')
    ax.axhline(0, color='#8b949e', lw=0.7, ls=':')
    if not fwd.empty:
        ax.plot(fwd.index, fwd.values, color='#bc8cff', lw=1.4, ls='--',
                label='5Y5Y fwd inflation')
    _fomc_lines(ax)
    _nom_line(ax)
    _xlim(ax)
    _datefmt(ax)
    ax.set_title('Nominal 10Y, Breakeven Inflation, and Real Rate')
    ax.set_ylabel('Rate (%)')
    ax.legend(fontsize=8)
    ax.grid(True)

    # Ratio: real / nominal — how much of yield is real vs inflation
    ax2 = axes[1]
    share_real = (real / dgs10.reindex(real.index).ffill() * 100).dropna()
    ax2.fill_between(share_real.index, 50, share_real.values,
                     where=share_real.values > 50, color='#3fb950', alpha=0.4,
                     label='Real rate > 50% of nominal')
    ax2.fill_between(share_real.index, 50, share_real.values,
                     where=share_real.values <= 50, color='#f78166', alpha=0.4,
                     label='Inflation BE > 50% of nominal')
    ax2.plot(share_real.index, share_real.values, color='#e6edf3', lw=1)
    ax2.axhline(50, color='#8b949e', lw=0.8, ls='--')
    _fomc_lines(ax2)
    _nom_line(ax2)
    _xlim(ax2)
    _datefmt(ax2)
    ax2.set_title('Real rate as % of nominal 10Y yield')
    ax2.set_ylabel('%')
    ax2.legend(fontsize=8)
    ax2.grid(True)

    fig.suptitle('Nominal vs Real Rates and Breakeven Inflation', fontsize=14, y=1.01)
    plt.tight_layout()
    pdf.savefig(fig, bbox_inches='tight')
    plt.close(fig)


def page_summary(pdf, data):
    fig, ax = plt.subplots(figsize=(11, 8.5))
    ax.axis('off')
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)

    ax.text(0.5, 0.95, 'Rate Path Summary', ha='center', fontsize=20,
            fontweight='bold', color='#e6edf3', transform=ax.transAxes)

    # Build stats
    dff  = data['dff'].dropna().iloc[-1]
    lo   = data['lo'].dropna().iloc[-1]
    hi   = data['hi'].dropna().iloc[-1]
    rows = [['Metric', 'Current', '6m ago', '12m ago', 'Δ 12m']]

    for label, key in [('3M T-bill', '3M'), ('6M T-bill', '6M'),
                        ('1Y T-note', '1Y'), ('2Y T-note', '2Y')]:
        if key not in data:
            continue
        s      = data[key].dropna()
        cur    = s.iloc[-1]
        ago6   = _nearest(s, TODAY - pd.DateOffset(months=6))
        ago12  = _nearest(s, TODAY - pd.DateOffset(months=12))
        delta  = cur - ago12
        rows.append([label, f'{cur:.2f}%', f'{ago6:.2f}%', f'{ago12:.2f}%',
                     f'{delta:+.2f}pp'])

    rows.append(['─' * 20, '─' * 8, '─' * 8, '─' * 8, '─' * 8])
    rows.append(['Effective DFF', f'{dff:.2f}%', '—', '—', '—'])
    rows.append(['Target range', f'{lo:.2f}–{hi:.2f}%', '—', '—', '—'])

    if '1Y' in data:
        s  = data['1Y'].dropna()
        y1 = s.iloc[-1]
        implied = (y1 - dff) / 0.25
        rows.append(['Implied cuts (1Y)', f'{implied:+.1f} × 25bp', '—', '—', '—'])

    col_x  = [0.04, 0.32, 0.50, 0.67, 0.84]
    row_h  = 0.065
    y_top  = 0.82
    hdr_c  = '#58a6ff'

    for ri, row in enumerate(rows):
        y = y_top - ri * row_h
        bg = '#161b22' if ri % 2 == 0 else '#0d1117'
        if ri > 0:
            ax.add_patch(mpatches.FancyBboxPatch(
                (0.01, y - row_h * 0.7), 0.98, row_h * 0.85,
                boxstyle='round,pad=0.01', fc=bg, ec='none',
                transform=ax.transAxes, clip_on=False))
        for ci, cell in enumerate(row):
            color = hdr_c if ri == 0 else ('#f78166' if cell.startswith('+') and ri > 0
                                           else '#3fb950' if cell.startswith('-') and ri > 0
                                           else '#e6edf3')
            ax.text(col_x[ci], y, cell, ha='left', va='top', fontsize=10,
                    color=color, transform=ax.transAxes)

    pdf.savefig(fig, bbox_inches='tight')
    plt.close(fig)


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--out', default='sofr_implied_path.pdf')
    args = parser.parse_args()

    data = fetch_all()
    print(f'\nWriting {args.out} ...')

    with PdfPages(args.out) as pdf:
        print('  [1] Cover')
        page_cover(pdf, data)
        print('  [2] Implied path evolution')
        page_implied_path_evolution(pdf, data)
        print('  [3] Forward curve snapshots')
        page_forward_curve_snapshots(pdf, data)
        print('  [4] Actual vs expected')
        page_actual_vs_expected(pdf, data)
        print('  [5] FOMC event study')
        page_fomc_event_study(pdf, data)
        print('  [6] Breakevens and real rates')
        page_breakeven_real(pdf, data)
        print('  [7] Summary table')
        page_summary(pdf, data)

        meta = pdf.infodict()
        meta['Title']   = 'SOFR & Fed Rate Path Dynamics'
        meta['Subject'] = 'Implied path, FOMC event study, breakevens'

    print(f'\nDone → {args.out}')


if __name__ == '__main__':
    main()
