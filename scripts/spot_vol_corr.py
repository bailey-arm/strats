"""
spot_vol_corr.py — Spot-Vol Correlation Across Asset Classes
Breaks down rolling spot-vol correlation by Trump presidential terms vs. other periods.
Covers equities (SPY, QQQ), US bonds (TLT), and FX (EUR/USD, GBP/USD).
History from 2004 captures GFC, Euro debt crisis, VIXplosion, COVID, Ukraine, Iran.

Usage:
    python scripts/spot_vol_corr.py [--out spot_vol_corr.pdf]
"""

import argparse
import io
import warnings
from datetime import date

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.gridspec import GridSpec
import numpy as np
import pandas as pd
import requests
import yfinance as yf

warnings.filterwarnings('ignore')

# ── Constants ─────────────────────────────────────────────────────────────────
START = '2004-01-01'
END   = str(date.today())

TRUMP_TERMS = [
    (pd.Timestamp('2017-01-20'), pd.Timestamp('2021-01-20'), 'Trump Term 1'),
    (pd.Timestamp('2025-01-20'), pd.Timestamp(END),          'Trump Term 2'),
]

# Major market shock events for annotation
MAJOR_EVENTS = {
    'GFC\nLehman':      pd.Timestamp('2008-09-15'),
    'Euro\nDebt Crisis': pd.Timestamp('2011-08-05'),
    'China\nShock':     pd.Timestamp('2015-08-24'),
    'VIX-\nplosion':    pd.Timestamp('2018-02-05'),
    'COVID\nCrash':     pd.Timestamp('2020-03-16'),
    'Ukraine\nInvasion':pd.Timestamp('2022-02-24'),
    'SVB\nCollapse':    pd.Timestamp('2023-03-10'),
    'Iran\nStrike':     pd.Timestamp('2024-04-14'),
    'Liberation\nDay':  pd.Timestamp('2025-04-02'),
}

PALETTE = {
    'spy':    '#58a6ff',
    'qqq':    '#3fb950',
    'tlt':    '#d29922',
    'eurusd': '#bc8cff',
    'gbpusd': '#f78166',
    'oil':    '#e3b341',
    'natgas': '#39d353',
    'vix':    '#ff7b72',
    'neutral':'#8b949e',
    'trump1': '#f0883e',
    'trump2': '#d2a8ff',
    'event':  '#f85149',
}

plt.rcParams.update({
    'figure.facecolor': '#0d1117',
    'axes.facecolor':   '#161b22',
    'axes.edgecolor':   '#30363d',
    'text.color':       '#e6edf3',
    'axes.labelcolor':  '#e6edf3',
    'xtick.color':      '#8b949e',
    'ytick.color':      '#8b949e',
    'grid.color':       '#21262d',
    'grid.linestyle':   '--',
    'grid.alpha':       0.5,
    'axes.titlecolor':  '#e6edf3',
    'legend.facecolor': '#161b22',
    'legend.edgecolor': '#30363d',
    'font.size':        10,
    'figure.dpi':       120,
})


# ── Data fetchers ─────────────────────────────────────────────────────────────
def _yf(ticker: str) -> pd.Series:
    raw = yf.download(ticker, start=START, end=END, auto_adjust=True, progress=False)
    if raw.empty:
        return pd.Series(dtype=float, name=ticker)
    s = raw['Close']
    if isinstance(s, pd.DataFrame):
        s = s.iloc[:, 0]
    s.index = pd.to_datetime(s.index).tz_localize(None)
    s.name = ticker
    return s.dropna()


def _fred(series_id: str) -> pd.Series:
    url = (f'https://fred.stlouisfed.org/graph/fredgraph.csv'
           f'?id={series_id}&vintage_date={END}')
    r = requests.get(url, timeout=30)
    r.raise_for_status()
    df = pd.read_csv(io.StringIO(r.text), index_col='observation_date')
    df.index = pd.to_datetime(df.index)
    s = pd.to_numeric(df.iloc[:, 0], errors='coerce')
    return s.loc[START:END].dropna()


def load_data() -> dict:
    print('Fetching yfinance: SPY, QQQ, TLT, VIX, MOVE, EURUSD, GBPUSD, CL=F, NG=F…')
    spy    = _yf('SPY')
    qqq    = _yf('QQQ')
    tlt    = _yf('TLT')
    vix    = _yf('^VIX')
    move   = _yf('^MOVE')
    eurusd = _yf('EURUSD=X')
    gbpusd = _yf('GBPUSD=X')
    oil    = _yf('CL=F')    # WTI crude front-month futures
    natgas = _yf('NG=F')    # Henry Hub natural gas front-month futures

    print('Fetching FRED: Treasury yields for MOVE proxy…')
    try:
        y2  = _fred('DGS2');  y5  = _fred('DGS5')
        y10 = _fred('DGS10'); y30 = _fred('DGS30')
        chg = {t: s.diff() * 100 for t, s in
               [('2Y', y2), ('5Y', y5), ('10Y', y10), ('30Y', y30)]}
        move_proxy = (0.2 * chg['2Y'].rolling(21).std() +
                      0.3 * chg['5Y'].rolling(21).std() +
                      0.3 * chg['10Y'].rolling(21).std() +
                      0.2 * chg['30Y'].rolling(21).std()) * np.sqrt(252)
    except Exception:
        move_proxy = pd.Series(dtype=float)

    bond_vol       = move if not move.dropna().empty else move_proxy
    bond_vol_label = 'MOVE' if not move.dropna().empty else 'MOVE Proxy'

    dvix      = vix.diff()
    dbond_vol = bond_vol.diff()

    returns = {k: v.pct_change() for k, v in
               [('SPY', spy), ('QQQ', qqq), ('TLT', tlt),
                ('EURUSD', eurusd), ('GBPUSD', gbpusd),
                ('OIL', oil), ('NATGAS', natgas)]}

    def _rv21(s):
        return s.pct_change().rolling(21).std() * np.sqrt(252) * 100

    print('Done.\n')
    return dict(
        spy=spy, qqq=qqq, tlt=tlt, eurusd=eurusd, gbpusd=gbpusd,
        oil=oil, natgas=natgas,
        vix=vix, bond_vol=bond_vol, bond_vol_label=bond_vol_label,
        returns=returns, dvix=dvix, dbond_vol=dbond_vol,
        eur_rv=_rv21(eurusd), gbp_rv=_rv21(gbpusd),
        oil_rv=_rv21(oil),    gas_rv=_rv21(natgas),
    )


# ── Helpers ───────────────────────────────────────────────────────────────────
def _trump_mask(index: pd.DatetimeIndex) -> pd.Series:
    mask = pd.Series(False, index=index)
    for start, end, _ in TRUMP_TERMS:
        mask |= (index >= start) & (index <= end)
    return mask


def _shade_trump(ax, alpha=0.13):
    """Shade Trump terms; preserve y-limits; return legend patches."""
    ylim = ax.get_ylim()
    patches = []
    for i, (t0, t1, label) in enumerate(TRUMP_TERMS):
        col = PALETTE['trump1'] if i == 0 else PALETTE['trump2']
        ax.axvspan(t0, t1, alpha=alpha, color=col, zorder=0)
        patches.append(mpatches.Patch(facecolor=col, alpha=0.5, label=label))
    ax.set_ylim(ylim)
    return patches


def _annotate_events(ax, events: dict | None = None, y_frac=0.97):
    """
    Draw vertical event lines with rotated labels.
    Uses get_xaxis_transform() so y is in axes-fraction and x is data coords —
    labels land at a fixed height regardless of the y-axis scale.
    """
    if events is None:
        events = MAJOR_EVENTS
    tr = ax.get_xaxis_transform()
    xlim_left  = ax.get_xlim()[0]
    xlim_right = ax.get_xlim()[1]
    for label, dt in events.items():
        x = matplotlib.dates.date2num(dt.to_pydatetime())
        if not (xlim_left <= x <= xlim_right):
            continue
        ax.axvline(dt, color=PALETTE['event'], lw=0.8, linestyle=':', alpha=0.75, zorder=3)
        ax.text(dt, y_frac, label, transform=tr,
                color=PALETTE['event'], fontsize=6, ha='left', va='top',
                rotation=90, clip_on=True,
                bbox=dict(facecolor='none', edgecolor='none', pad=0))


def _rolling_corr(ret: pd.Series, vol: pd.Series, window: int = 63) -> pd.Series:
    df = pd.concat([ret.rename('r'), vol.rename('v')], axis=1).dropna()
    return df['r'].rolling(window).corr(df['v'])


def _regime_stats(corr: pd.Series) -> dict:
    c = corr.dropna()

    def _s(mask):
        s = c[mask]
        if s.empty:
            return dict(mean=np.nan, pct_neg=np.nan, n=0)
        return dict(mean=s.mean(), pct_neg=(s < 0).mean() * 100, n=len(s))

    base_mask = _trump_mask(c.index)
    t1 = (c.index >= TRUMP_TERMS[0][0]) & (c.index <= TRUMP_TERMS[0][1])
    t2 = (c.index >= TRUMP_TERMS[1][0]) & (c.index <= TRUMP_TERMS[1][1])
    return dict(
        trump1=_s(t1), trump2=_s(t2),
        other=_s(~base_mask),
    )


# ── Cover ─────────────────────────────────────────────────────────────────────
def page_cover(d: dict, pdf: PdfPages):
    fig = plt.figure(figsize=(13, 9))
    fig.patch.set_facecolor('#0d1117')
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_facecolor('#0d1117')
    ax.axis('off')

    fig.text(0.5, 0.84, 'Spot-Vol Correlation Across Asset Classes',
             ha='center', fontsize=26, color='#e6edf3', fontweight='bold')
    fig.text(0.5, 0.77, 'Trump Administrations vs. Other Periods  —  Iran War Context',
             ha='center', fontsize=17, color='#58a6ff')
    fig.text(0.5, 0.71, f'Data: {START}–{END}  |  63-day rolling window',
             ha='center', fontsize=11, color='#8b949e')

    # Term boxes
    box_data = [
        (0.25, PALETTE['trump1'], 'Trump Term 1', 'Jan 2017 – Jan 2021'),
        (0.75, PALETTE['trump2'], 'Trump Term 2', 'Jan 2025 – present'),
    ]
    for x, col, title, dates in box_data:
        fig.text(x, 0.64, title,  ha='center', fontsize=13, color=col, fontweight='bold')
        fig.text(x, 0.60, dates,  ha='center', fontsize=10, color='#8b949e')

    blurb = (
        'Spot-vol correlation quantifies how asset prices and volatility co-move.'
        ' For equities the relationship is robustly negative: sell-offs spike the VIX'
        ' while rallies compress it (leverage effect + vol-feedback loop).'
        ' Fixed income shows the same sign: bond prices fall as rates move, and rate'
        ' vol (MOVE) peaks during rapid rate moves. FX correlations are weaker and'
        ' more regime-sensitive.\n\n'
        'Geopolitical shocks such as the ongoing Iran conflict challenge this baseline'
        ' by elevating vol independently of spot direction. This report tracks the'
        ' 63-day rolling spot-vol correlation for SPY, QQQ, TLT, EUR/USD, and GBP/USD'
        ' from 2004 to present, disaggregated by Trump vs. non-Trump administrations,'
        ' with major market events annotated throughout.'
    )
    fig.text(0.5, 0.54, blurb, ha='center', va='top', fontsize=10.5,
             color='#c9d1d9', linespacing=1.7, wrap=True)

    # Asset tiles
    tiles = [
        ('SPY',     'S&P 500 ETF',       'vs VIX',    PALETTE['spy']),
        ('QQQ',     'Nasdaq 100 ETF',     'vs VIX',    PALETTE['qqq']),
        ('TLT',     '20Y Treasury ETF',   'vs MOVE',   PALETTE['tlt']),
        ('EUR/USD', 'FX',                 'vs 21d RV', PALETTE['eurusd']),
        ('GBP/USD', 'FX',                 'vs 21d RV', PALETTE['gbpusd']),
        ('WTI Oil', 'CL=F Futures',       'vs 21d RV', PALETTE['oil']),
        ('Nat Gas', 'NG=F Futures',        'vs 21d RV', PALETTE['natgas']),
    ]
    for i, (ticker, desc, vol_note, col) in enumerate(tiles):
        x = 0.07 + i * 0.13
        fig.text(x, 0.22, ticker,    ha='center', fontsize=11, color=col, fontweight='bold')
        fig.text(x, 0.18, desc,      ha='center', fontsize=7.5, color='#8b949e')
        fig.text(x, 0.15, vol_note,  ha='center', fontsize=7,   color='#484f58')

    # Event legend row
    shocks = ['GFC 2008', 'Euro Debt 2011', 'China 2015',
              'VIXplosion 2018', 'COVID 2020', 'Ukraine 2022', 'SVB 2023', 'Iran 2024–']
    fig.text(0.5, 0.09, 'Annotated shocks:  ' + '  ·  '.join(shocks),
             ha='center', fontsize=8.5, color=PALETTE['event'])
    fig.text(0.5, 0.04,
             'Sources: Yahoo Finance · FRED (Federal Reserve)',
             ha='center', fontsize=8.5, color='#484f58')

    pdf.savefig(fig, bbox_inches='tight')
    plt.close(fig)


# ── Equity overview ───────────────────────────────────────────────────────────
def page_equity_overview(d: dict, pdf: PdfPages):
    """SPY+QQQ vs VIX time series with rolling corr — full history."""
    fig = plt.figure(figsize=(13, 10))
    fig.patch.set_facecolor('#0d1117')
    gs = GridSpec(3, 1, figure=fig, hspace=0.40,
                  top=0.92, bottom=0.05, height_ratios=[2, 2, 1.8])
    fig.suptitle('Equity Spot vs VIX — Full History (2004–present)',
                 fontsize=14, fontweight='bold', color='#e6edf3')

    for row, (ticker, series, col) in enumerate([('SPY', d['spy'], PALETTE['spy']),
                                                  ('QQQ', d['qqq'], PALETTE['qqq'])]):
        ax  = fig.add_subplot(gs[row])
        axb = ax.twinx()
        ax.plot(series.index, series, color=col, lw=1.3, label=ticker)
        axb.plot(d['vix'].index, d['vix'], color=PALETTE['vix'], lw=0.9,
                 alpha=0.75, label='VIX')
        ax.fill_between(series.index, series.min(), series,
                        color=col, alpha=0.07)
        _shade_trump(ax)
        ax.set_ylabel(f'{ticker} Price', color=col, fontsize=9)
        axb.set_ylabel('VIX', color=PALETTE['vix'], fontsize=9)
        ax.set_title(f'{ticker} price (left) and VIX (right) — orange = Trump T1, purple = Trump T2',
                     fontsize=9, pad=4)
        ax.grid(True)
        l1, b1 = ax.get_legend_handles_labels()
        l2, b2 = axb.get_legend_handles_labels()
        ax.legend(l1 + l2, b1 + b2, loc='upper left', fontsize=8, framealpha=0.6)
        _annotate_events(ax)

    # Rolling corr panel
    corr_spy = _rolling_corr(d['returns']['SPY'], d['dvix'])
    corr_qqq = _rolling_corr(d['returns']['QQQ'], d['dvix'])
    ax3 = fig.add_subplot(gs[2])
    ax3.plot(corr_spy.index, corr_spy, color=PALETTE['spy'], lw=1.2,
             label='SPY–VIX')
    ax3.plot(corr_qqq.index, corr_qqq, color=PALETTE['qqq'], lw=1.2,
             alpha=0.85, linestyle='--', label='QQQ–VIX')
    ax3.axhline(0, color='#8b949e', lw=0.8)
    ax3.fill_between(corr_spy.index, corr_spy, 0,
                     where=(corr_spy < 0), alpha=0.15, color=PALETTE['spy'])
    _shade_trump(ax3)
    _annotate_events(ax3, y_frac=0.97)
    ax3.set_ylim(-1.1, 1.1)
    ax3.set_ylabel('63d Rolling Corr', fontsize=9)
    ax3.set_title('63-Day Rolling Spot-Return / VIXΔ Correlation'
                  '  (negative = normal equity regime)', fontsize=9, pad=4)
    ax3.legend(loc='lower left', fontsize=8, ncol=2, framealpha=0.6)
    ax3.grid(True)

    pdf.savefig(fig, bbox_inches='tight')
    plt.close(fig)


# ── Per-asset deep dive ───────────────────────────────────────────────────────
def page_asset_deep_dive(d: dict, pdf: PdfPages,
                          asset: str, spot: pd.Series, vol: pd.Series,
                          vol_label: str, spot_color: str, vol_color: str,
                          use_vol_diff: bool = True):
    """
    3-panel layout:
      Top   : spot vs vol time series (dual axis)
      Middle: 63d rolling corr with Trump shading + event lines
      Bottom: regime bar charts (mean corr | % negative)
    """
    ret  = spot.pct_change()
    dvol = vol.diff() if use_vol_diff else vol
    corr = _rolling_corr(ret, dvol)
    st   = _regime_stats(corr)
    full_mean = corr.dropna().mean()

    fig = plt.figure(figsize=(13, 10))
    fig.patch.set_facecolor('#0d1117')
    gs = GridSpec(3, 2, figure=fig, hspace=0.45, wspace=0.32,
                  top=0.91, bottom=0.05,
                  height_ratios=[1.9, 1.9, 1.5])
    fig.suptitle(f'{asset} — Spot-Vol Correlation by Regime',
                 fontsize=14, fontweight='bold', color='#e6edf3')

    # ── Top row: spot vs vol ──
    ax_t = fig.add_subplot(gs[0, :])
    axb  = ax_t.twinx()
    ax_t.plot(spot.index, spot, color=spot_color, lw=1.3, label=asset)
    axb.plot(vol.index, vol, color=vol_color, lw=0.9, alpha=0.75, label=vol_label)
    _shade_trump(ax_t)
    _annotate_events(ax_t)
    ax_t.set_ylabel(f'{asset}', color=spot_color, fontsize=9)
    axb.set_ylabel(vol_label, color=vol_color, fontsize=9)
    ax_t.set_title(f'{asset} (left, {spot_color.upper()[:3]}) vs {vol_label} (right)'
                   '  —  orange shading = Trump T1, purple = Trump T2',
                   fontsize=8.5, pad=4)
    ax_t.grid(True)
    l1, b1 = ax_t.get_legend_handles_labels()
    l2, b2 = axb.get_legend_handles_labels()
    ax_t.legend(l1 + l2, b1 + b2, loc='upper left', fontsize=8, framealpha=0.6)

    # ── Middle row: rolling correlation ──
    ax_c = fig.add_subplot(gs[1, :])
    ax_c.plot(corr.index, corr, color=spot_color, lw=1.2,
              label=f'63d rolling corr')
    ax_c.axhline(0, color='#8b949e', lw=0.9)
    ax_c.axhline(full_mean, color='#8b949e', lw=0.7, linestyle=':',
                 label=f'Full-period mean: {full_mean:.2f}')
    ax_c.fill_between(corr.index, corr, 0,
                      where=(corr < 0), alpha=0.18, color=spot_color)
    ax_c.fill_between(corr.index, corr, 0,
                      where=(corr > 0), alpha=0.18, color=vol_color)
    _shade_trump(ax_c)
    _annotate_events(ax_c, y_frac=0.97)
    ax_c.set_ylim(-1.1, 1.1)
    ax_c.set_yticks([-1, -0.5, 0, 0.5, 1])
    ax_c.set_ylabel('63d Rolling Corr', fontsize=9)
    ax_c.set_title(f'63-Day Rolling Spot-Return / {vol_label} Correlation'
                   '  (negative fill = normal regime, positive fill = breakdown)',
                   fontsize=8.5, pad=4)
    ax_c.legend(loc='lower left', fontsize=8, ncol=2, framealpha=0.6)
    ax_c.grid(True)

    # ── Bottom-left: mean corr by regime ──
    ax_bar = fig.add_subplot(gs[2, 0])
    regimes = ['Trump T1\n2017–21', 'Non-Trump', 'Trump T2\n2025–']
    colors  = [PALETTE['trump1'], PALETTE['neutral'], PALETTE['trump2']]
    keys    = ['trump1', 'other', 'trump2']
    means   = [st[k]['mean'] for k in keys]

    x = np.arange(len(regimes))
    bars = ax_bar.bar(x, means, color=colors, alpha=0.82, width=0.55, zorder=2)
    ax_bar.axhline(0, color='#8b949e', lw=0.9, zorder=3)
    ax_bar.set_xticks(x)
    ax_bar.set_xticklabels(regimes, fontsize=8.5)
    ax_bar.set_ylabel('Mean 63d Corr', fontsize=9)
    ax_bar.set_title('Avg Spot-Vol Corr by Regime', fontsize=9, pad=4)
    ax_bar.set_ylim(-1.1, 0.6)
    ax_bar.grid(True, axis='y', zorder=0)
    for bar, val in zip(bars, means):
        if np.isnan(val):
            continue
        ypos = val + 0.03 if val >= 0 else val - 0.08
        ax_bar.text(bar.get_x() + bar.get_width() / 2, ypos,
                    f'{val:.2f}', ha='center', va='bottom', fontsize=9,
                    color='#e6edf3', fontweight='bold')

    # ── Bottom-right: % negative ──
    ax_pct = fig.add_subplot(gs[2, 1])
    pcts = [st[k]['pct_neg'] for k in keys]
    bars2 = ax_pct.bar(x, pcts, color=colors, alpha=0.82, width=0.55, zorder=2)
    ax_pct.axhline(50, color='#8b949e', lw=0.8, linestyle='--',
                   label='50% baseline', zorder=3)
    ax_pct.set_xticks(x)
    ax_pct.set_xticklabels(regimes, fontsize=8.5)
    ax_pct.set_ylabel('% Days', fontsize=9)
    ax_pct.set_title('% of Days with Negative Corr', fontsize=9, pad=4)
    ax_pct.set_ylim(0, 108)
    ax_pct.grid(True, axis='y', zorder=0)
    ax_pct.legend(fontsize=8, framealpha=0.6)
    for bar, val in zip(bars2, pcts):
        if np.isnan(val):
            continue
        ax_pct.text(bar.get_x() + bar.get_width() / 2, val + 1.5,
                    f'{val:.0f}%', ha='center', va='bottom', fontsize=9,
                    color='#e6edf3', fontweight='bold')

    pdf.savefig(fig, bbox_inches='tight')
    plt.close(fig)


# ── Cross-asset summary ───────────────────────────────────────────────────────
def page_cross_asset_summary(d: dict, pdf: PdfPages):
    asset_defs = [
        ('SPY',     d['returns']['SPY'],    d['dvix'],         True,  PALETTE['spy']),
        ('QQQ',     d['returns']['QQQ'],    d['dvix'],         True,  PALETTE['qqq']),
        ('TLT',     d['returns']['TLT'],    d['dbond_vol'],    True,  PALETTE['tlt']),
        ('EUR/USD', d['returns']['EURUSD'], d['eur_rv'],       False, PALETTE['eurusd']),
        ('GBP/USD', d['returns']['GBPUSD'], d['gbp_rv'],       False, PALETTE['gbpusd']),
        ('Oil',     d['returns']['OIL'],    d['oil_rv'],       False, PALETTE['oil']),
        ('Nat Gas', d['returns']['NATGAS'], d['gas_rv'],       False, PALETTE['natgas']),
    ]

    all_stats = {}
    for name, ret, vol, use_diff, col in asset_defs:
        dvol = vol.diff() if use_diff else vol
        corr = _rolling_corr(ret, dvol)
        all_stats[name] = (_regime_stats(corr), col)

    fig = plt.figure(figsize=(13, 10))
    fig.patch.set_facecolor('#0d1117')
    gs = GridSpec(2, 1, figure=fig, hspace=0.45,
                  top=0.91, bottom=0.07)
    fig.suptitle('Cross-Asset Spot-Vol Correlation — Regime Comparison',
                 fontsize=14, fontweight='bold', color='#e6edf3')

    regime_keys   = ['trump1', 'other', 'trump2']
    regime_labels = ['Trump T1 (2017–21)', 'Non-Trump', 'Trump T2 (2025–)']
    regime_colors = [PALETTE['trump1'], PALETTE['neutral'], PALETTE['trump2']]

    names = list(all_stats.keys())
    x = np.arange(len(names))
    w = 0.22

    for ax_idx, (metric, ylabel, title, ylim) in enumerate([
        ('mean',    'Mean 63d Spot-Vol Corr',       'Average Spot-Vol Correlation by Regime',    (-1.1, 0.7)),
        ('pct_neg', '% Days with Negative Corr',    'Frequency of Negative Correlation by Regime', (0, 115)),
    ]):
        ax = fig.add_subplot(gs[ax_idx])
        for i, (rk, rl, rc) in enumerate(zip(regime_keys, regime_labels, regime_colors)):
            vals = [all_stats[n][0][rk][metric] for n in names]
            offset = (i - 1) * w
            bars = ax.bar(x + offset, vals, w, color=rc, alpha=0.82,
                          label=rl, zorder=2)
            for bar, val in zip(bars, vals):
                if np.isnan(val):
                    continue
                suffix = '%' if metric == 'pct_neg' else ''
                ypos = val + (1.5 if metric == 'pct_neg' else 0.025)
                ax.text(bar.get_x() + bar.get_width() / 2, ypos,
                        f'{val:.0f}{suffix}' if metric == 'pct_neg' else f'{val:.2f}',
                        ha='center', va='bottom', fontsize=7.5,
                        color='#e6edf3', fontweight='bold')

        ref_val = 0 if metric == 'mean' else 50
        ref_label = 'zero' if metric == 'mean' else '50% baseline'
        ax.axhline(ref_val, color='#8b949e', lw=0.85,
                   linestyle='--' if metric == 'pct_neg' else '-',
                   label=ref_label, zorder=3)
        ax.set_xticks(x)
        ax.set_xticklabels(names, fontsize=10)
        ax.set_ylabel(ylabel, fontsize=9)
        ax.set_title(title, fontsize=10, pad=5)
        ax.set_ylim(*ylim)
        ax.legend(loc='lower right', fontsize=8, framealpha=0.6)
        ax.grid(True, axis='y', zorder=0)

    pdf.savefig(fig, bbox_inches='tight')
    plt.close(fig)


# ── Iran / recent zoom ────────────────────────────────────────────────────────
def page_recent_zoom(d: dict, pdf: PdfPages):
    """2024–present: all corr series on one panel, Iran events annotated."""
    ZOOM = pd.Timestamp('2024-01-01')

    asset_defs = [
        ('SPY',     d['returns']['SPY'],    d['dvix'],      True,  PALETTE['spy']),
        ('QQQ',     d['returns']['QQQ'],    d['dvix'],      True,  PALETTE['qqq']),
        ('TLT',     d['returns']['TLT'],    d['dbond_vol'], True,  PALETTE['tlt']),
        ('EUR/USD', d['returns']['EURUSD'], d['eur_rv'],    False, PALETTE['eurusd']),
        ('GBP/USD', d['returns']['GBPUSD'], d['gbp_rv'],    False, PALETTE['gbpusd']),
        ('Oil',     d['returns']['OIL'],    d['oil_rv'],    False, PALETTE['oil']),
        ('Nat Gas', d['returns']['NATGAS'], d['gas_rv'],    False, PALETTE['natgas']),
    ]

    recent_events = {k: v for k, v in MAJOR_EVENTS.items() if v >= ZOOM}

    fig = plt.figure(figsize=(13, 10))
    fig.patch.set_facecolor('#0d1117')
    gs = GridSpec(2, 1, figure=fig, hspace=0.40,
                  top=0.91, bottom=0.05)
    fig.suptitle('Recent Period Zoom: 2024–present  —  Iran War & Trump T2 Impact',
                 fontsize=13, fontweight='bold', color='#e6edf3')

    # Top: all corr series
    ax1 = fig.add_subplot(gs[0])
    for name, ret, vol, use_diff, col in asset_defs:
        dvol = vol.diff() if use_diff else vol
        corr = _rolling_corr(ret, dvol).loc[ZOOM:]
        ax1.plot(corr.index, corr, color=col, lw=1.3, label=name)
    ax1.axhline(0, color='#8b949e', lw=0.9)
    _shade_trump(ax1)
    _annotate_events(ax1, events=recent_events, y_frac=0.97)
    ax1.set_ylim(-1.1, 1.1)
    ax1.set_yticks([-1, -0.5, 0, 0.5, 1])
    ax1.set_ylabel('63d Rolling Corr', fontsize=9)
    ax1.set_title('All Assets — 63d Rolling Spot-Vol Correlation'
                  '  (red dotted = major events, purple shading = Trump T2)',
                  fontsize=9, pad=4)
    ax1.legend(loc='lower left', fontsize=8, ncol=7, framealpha=0.6)
    ax1.grid(True)

    # Bottom: SPY and VIX
    ax2  = fig.add_subplot(gs[1])
    ax2b = ax2.twinx()
    spy_z = d['spy'].loc[ZOOM:]
    vix_z = d['vix'].loc[ZOOM:]
    ax2.plot(spy_z.index, spy_z, color=PALETTE['spy'], lw=1.4, label='SPY')
    ax2b.plot(vix_z.index, vix_z, color=PALETTE['vix'], lw=1.0, alpha=0.85, label='VIX')
    _shade_trump(ax2)
    _annotate_events(ax2, events=recent_events)
    ax2.set_ylabel('SPY Price', color=PALETTE['spy'], fontsize=9)
    ax2b.set_ylabel('VIX', color=PALETTE['vix'], fontsize=9)
    ax2.set_title('SPY vs VIX  —  major events annotated', fontsize=9, pad=4)
    ax2.grid(True)
    l1, b1 = ax2.get_legend_handles_labels()
    l2, b2 = ax2b.get_legend_handles_labels()
    ax2.legend(l1 + l2, b1 + b2, loc='upper left', fontsize=8, framealpha=0.6)

    pdf.savefig(fig, bbox_inches='tight')
    plt.close(fig)


# ── Narrative ─────────────────────────────────────────────────────────────────
def page_narrative(d: dict, pdf: PdfPages):
    spy_corr  = _rolling_corr(d['returns']['SPY'], d['dvix'])
    eur_corr  = _rolling_corr(d['returns']['EURUSD'], d['eur_rv'])
    tlt_corr  = _rolling_corr(d['returns']['TLT'], d['dbond_vol'])
    spy_st    = _regime_stats(spy_corr)
    eur_st    = _regime_stats(eur_corr)
    tlt_st    = _regime_stats(tlt_corr)

    def f(v):
        return f'{v:.2f}' if not np.isnan(v) else 'N/A'

    fig = plt.figure(figsize=(13, 9))
    fig.patch.set_facecolor('#0d1117')
    fig.suptitle('Key Takeaways — Spot-Vol Correlation & Presidential Regime',
                 fontsize=14, fontweight='bold', color='#e6edf3', y=0.97)

    ax = fig.add_axes([0.06, 0.04, 0.88, 0.88])
    ax.set_facecolor('#0d1117')
    ax.axis('off')

    sections = [
        {
            'heading': '1.  Equity Spot-Vol Correlation (SPY / QQQ)',
            'color':   PALETTE['spy'],
            'body': (
                f'Robustly negative across all regimes — the leverage effect is structural.'
                f' SPY–VIX averaged {f(spy_st["trump1"]["mean"])} in Trump T1,'
                f' {f(spy_st["other"]["mean"])} in non-Trump periods,'
                f' and {f(spy_st["trump2"]["mean"])} in Trump T2 (to date).'
                f' The correlation is negative more than {spy_st["other"]["pct_neg"]:.0f}% of non-Trump'
                f' days and {spy_st["trump1"]["pct_neg"]:.0f}% of Trump T1 days.'
            ),
        },
        {
            'heading': '2.  Why Trump Terms Can Look Different',
            'color':   PALETTE['trump1'],
            'body': (
                'Policy unpredictability (tariff announcements, Fed pressure, geopolitical brinkmanship)'
                ' tends to keep vol floors elevated even on up days, compressing the magnitude of the'
                ' negative correlation without flipping its sign. Acute shock days'
                ' (spot ↓, VIX ↑) actually reinforce negative corr; it is the sustained'
                ' “vol floor” on quiet up-days that dilutes the average.'
            ),
        },
        {
            'heading': '3.  Bond Spot-Vol Correlation (TLT / MOVE)',
            'color':   PALETTE['tlt'],
            'body': (
                f'TLT–MOVE averaged {f(tlt_st["trump1"]["mean"])} (T1),'
                f' {f(tlt_st["other"]["mean"])} (non-Trump),'
                f' {f(tlt_st["trump2"]["mean"])} (T2).'
                ' The sign is negative for the same reason as equities: bond prices fall fastest'
                ' when rates are moving rapidly, and rate vol peaks in exactly those windows.'
                ' The 2022–23 hiking cycle held this relationship tightly; tariff-driven'
                ' rate uncertainty in 2025 has reinforced it.'
            ),
        },
        {
            'heading': '4.  FX Spot-Vol Correlation (EUR/USD, GBP/USD)',
            'color':   PALETTE['eurusd'],
            'body': (
                f'FX spot-vol correlation is weaker and more volatile.'
                f' EUR/USD–RV averaged {f(eur_st["trump1"]["mean"])} in Trump T1'
                f' vs {f(eur_st["other"]["mean"])} elsewhere.'
                ' Risk-off episodes bid the USD (EUR/USD falls) while FX vol spikes, which'
                ' produces a negative reading in EUR/USD terms. However, sustained USD weakness'
                ' (e.g. dollar smile mid-regime) can decouple the pair from vol, producing'
                ' near-zero or transiently positive correlation.'
            ),
        },
        {
            'heading': '5.  Iran War & Geopolitical Vol Shocks',
            'color':   PALETTE['event'],
            'body': (
                'The Iran escalation is a test case for “correlation breakdown.”'
                ' Classical geopolitical vol shocks raise implied vol independently of spot'
                ' direction — if equities are cushioned by defence spending or energy'
                ' hedges while VIX spikes, the rolling correlation compresses toward zero or'
                ' turns transiently positive. The Liberation Day tariff shock (Apr 2025) showed'
                ' the same dynamic. Watch the 63d rolling correlation level as a real-time'
                ' regime indicator: a sustained move above −0.2 signals a structural break'
                ' from the normal negative-corr regime.'
            ),
        },
    ]

    y = 0.96
    for sec in sections:
        ax.text(0.0, y, sec['heading'], transform=ax.transAxes,
                fontsize=10.5, color=sec['color'], va='top', fontweight='bold')
        y -= 0.055
        # wrap body text manually at ~115 chars per line
        words   = sec['body'].split()
        lines   = []
        current = ''
        for w in words:
            if len(current) + len(w) + 1 > 115:
                lines.append(current.rstrip())
                current = w + ' '
            else:
                current += w + ' '
        if current.strip():
            lines.append(current.rstrip())
        body_text = '\n'.join(lines)
        ax.text(0.02, y, body_text, transform=ax.transAxes,
                fontsize=9.5, color='#c9d1d9', va='top', linespacing=1.55)
        n = body_text.count('\n') + 1
        y -= 0.052 + (n - 1) * 0.038 + 0.018   # section gap

    ax.text(0.0, 0.01,
            f'Data: {START}–{END}  ·  63-day rolling window'
            '  ·  Sources: Yahoo Finance, FRED',
            transform=ax.transAxes, fontsize=8, color='#484f58')

    pdf.savefig(fig, bbox_inches='tight')
    plt.close(fig)


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description='Spot-Vol Correlation PDF')
    parser.add_argument('--out', default='spot_vol_corr.pdf')
    args = parser.parse_args()

    d = load_data()

    print(f'Writing PDF → {args.out}')
    with PdfPages(args.out) as pdf:
        page_cover(d, pdf)                          # P1

        page_equity_overview(d, pdf)                # P2 — SPY + QQQ full history

        page_asset_deep_dive(                       # P3 — SPY
            d, pdf, 'SPY', d['spy'], d['vix'],
            vol_label='VIX', spot_color=PALETTE['spy'],
            vol_color=PALETTE['vix'], use_vol_diff=True)

        page_asset_deep_dive(                       # P4 — QQQ
            d, pdf, 'QQQ', d['qqq'], d['vix'],
            vol_label='VIX', spot_color=PALETTE['qqq'],
            vol_color=PALETTE['vix'], use_vol_diff=True)

        page_asset_deep_dive(                       # P5 — TLT
            d, pdf, 'TLT (20Y Treasury)', d['tlt'], d['bond_vol'],
            vol_label=d['bond_vol_label'], spot_color=PALETTE['tlt'],
            vol_color=PALETTE['vix'], use_vol_diff=True)

        page_asset_deep_dive(                       # P6 — EUR/USD
            d, pdf, 'EUR/USD', d['eurusd'], d['eur_rv'],
            vol_label='EUR/USD 21d RV', spot_color=PALETTE['eurusd'],
            vol_color=PALETTE['vix'], use_vol_diff=False)

        page_asset_deep_dive(                       # P7 — GBP/USD
            d, pdf, 'GBP/USD', d['gbpusd'], d['gbp_rv'],
            vol_label='GBP/USD 21d RV', spot_color=PALETTE['gbpusd'],
            vol_color=PALETTE['vix'], use_vol_diff=False)

        page_asset_deep_dive(                       # P8 — WTI crude
            d, pdf, 'WTI Crude Oil (CL=F)', d['oil'], d['oil_rv'],
            vol_label='Oil 21d RV', spot_color=PALETTE['oil'],
            vol_color=PALETTE['vix'], use_vol_diff=False)

        page_asset_deep_dive(                       # P9 — Natural gas
            d, pdf, 'Natural Gas (NG=F)', d['natgas'], d['gas_rv'],
            vol_label='Nat Gas 21d RV', spot_color=PALETTE['natgas'],
            vol_color=PALETTE['vix'], use_vol_diff=False)

        page_cross_asset_summary(d, pdf)            # P10 — regime comparison

        page_recent_zoom(d, pdf)                    # P11 — 2024–present zoom

        page_narrative(d, pdf)                      # P12 — takeaways

        meta = pdf.infodict()
        meta['Title']   = 'Spot-Vol Correlation: Trump Terms vs Other Regimes'
        meta['Author']  = 'spot_vol_corr.py'
        meta['Subject'] = 'Spot-vol correlation, regime analysis, Iran geopolitical risk'

    print('Done.')


if __name__ == '__main__':
    main()
