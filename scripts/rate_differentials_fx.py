"""
rate_differentials_fx.py
------------------------
Generates a multi-page PDF on rate differentials and FX dynamics.
Core thesis: US-Euro 2Y spread is the primary driver of EUR/USD.

Usage:
    python scripts/rate_differentials_fx.py [--out rate_differentials_fx.pdf]
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
import matplotlib.cm as cm
import numpy as np
import pandas as pd
import requests
import yfinance as yf

warnings.filterwarnings('ignore')

# ── Constants ────────────────────────────────────────────────────────────────

START = '2022-01-01'
END   = str(date.today())

FOMC = pd.to_datetime([
    '2022-03-16','2022-05-04','2022-06-15','2022-07-27',
    '2022-09-21','2022-11-02','2022-12-14','2023-02-01','2023-03-22',
    '2023-05-03','2023-06-14','2023-07-26','2023-09-20','2023-11-01',
    '2023-12-13','2024-01-31','2024-03-20','2024-05-01','2024-06-12',
    '2024-07-31','2024-09-18','2024-11-07','2024-12-18','2025-01-29',
    '2025-03-19','2025-05-07','2025-06-18','2025-09-17','2026-01-28',
    '2026-03-18','2026-04-29',
])

PALETTE = ['#58a6ff', '#3fb950', '#d29922', '#f78166', '#bc8cff']

plt.rcParams.update({
    'figure.facecolor':   '#0d1117',
    'axes.facecolor':     '#161b22',
    'axes.edgecolor':     '#30363d',
    'text.color':         '#e6edf3',
    'axes.labelcolor':    '#e6edf3',
    'xtick.color':        '#8b949e',
    'ytick.color':        '#8b949e',
    'grid.color':         '#21262d',
    'grid.linestyle':     '--',
    'grid.alpha':         0.5,
    'axes.titlecolor':    '#e6edf3',
    'legend.facecolor':   '#161b22',
    'legend.edgecolor':   '#30363d',
    'font.size':          10,
    'figure.dpi':         120,
})


# ── Data fetchers ─────────────────────────────────────────────────────────────

def _fred(series_id: str, start: str = START, end: str = END) -> pd.Series:
    url = (f'https://fred.stlouisfed.org/graph/fredgraph.csv'
           f'?id={series_id}&vintage_date={end}')
    r = requests.get(url, timeout=30)
    r.raise_for_status()
    df = pd.read_csv(io.StringIO(r.text), index_col='observation_date')
    df.index = pd.to_datetime(df.index)
    s = pd.to_numeric(df.iloc[:, 0], errors='coerce')
    return s.loc[start:end].dropna()


def _ecb(tenor_years: int, start: str = START, end: str = END) -> pd.Series:
    key = f'B.U2.EUR.4F.G_N_A.SV_C_YM.SR_{tenor_years}Y'
    url = (f'https://data-api.ecb.europa.eu/service/data/YC/{key}'
           f'?format=csvdata&startPeriod={start}&endPeriod={end}')
    r = requests.get(url, timeout=30)
    r.raise_for_status()
    df = pd.read_csv(io.StringIO(r.text))
    out = df[['TIME_PERIOD', 'OBS_VALUE']].copy()
    out['TIME_PERIOD'] = pd.to_datetime(out['TIME_PERIOD'])
    out = out.dropna().set_index('TIME_PERIOD')['OBS_VALUE']
    out.name = f'{tenor_years}Y'
    return out.loc[start:end]


def _yf(ticker: str, start: str = START, end: str = END) -> pd.Series:
    raw = yf.download(ticker, start=start, end=end, progress=False, auto_adjust=True)
    if raw.empty:
        return pd.Series(dtype=float, name=ticker)
    close = raw['Close']
    if isinstance(close, pd.DataFrame):
        close = close.iloc[:, 0]
    close.index = pd.to_datetime(close.index)
    close.name = ticker
    return close.loc[start:end].dropna()


# ── Helpers ───────────────────────────────────────────────────────────────────

def align(*series) -> pd.DataFrame:
    """Inner-join a collection of Series on their date indices."""
    return pd.concat(series, axis=1).dropna()


def add_fomc(ax, ymin=None, ymax=None):
    """Add subtle FOMC vertical lines to an axes."""
    xlim = ax.get_xlim()
    for d in FOMC:
        x = matplotlib.dates.date2num(d.to_pydatetime())
        if xlim[0] <= x <= xlim[1]:
            ax.axvline(d, color='#8b949e', alpha=0.3, lw=0.8, zorder=0)


def rolling_corr(a: pd.Series, b: pd.Series, window: int = 60) -> pd.Series:
    df = align(a, b)
    return df.iloc[:, 0].rolling(window).corr(df.iloc[:, 1])


def shade_divergence(ax, spread: pd.Series, fx: pd.Series,
                     spread_norm: pd.Series, fx_norm: pd.Series):
    """Shade regions where spread and FX move in opposite directions."""
    df = align(spread_norm, fx_norm)
    diff = (df.iloc[:, 0] - df.iloc[:, 1]).abs()
    threshold = diff.quantile(0.75)
    in_div = diff > threshold
    idx = df.index
    for i in range(1, len(idx)):
        if in_div.iloc[i]:
            ax.axvspan(idx[i - 1], idx[i], alpha=0.12, color='#f78166', zorder=0)


def _normalize(s: pd.Series) -> pd.Series:
    rng = s.max() - s.min()
    if rng == 0:
        return s - s.mean()
    return (s - s.min()) / rng


# ── Page builders ─────────────────────────────────────────────────────────────

def page_cover(pdf: PdfPages):
    fig = plt.figure(figsize=(11, 8.5))
    fig.patch.set_facecolor('#0d1117')
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_facecolor('#0d1117')
    ax.axis('off')

    ax.text(0.5, 0.72, 'Rate Differentials and FX',
            ha='center', va='center', fontsize=32, fontweight='bold',
            color='#e6edf3', transform=ax.transAxes)

    ax.text(0.5, 0.60,
            'How 2-Year Yield Spreads Drive EUR/USD and GBP/USD',
            ha='center', va='center', fontsize=18, color='#58a6ff',
            transform=ax.transAxes)

    thesis = (
        'Core Thesis\n\n'
        'The US–Euro 2-year yield spread is the single most reliable driver\n'
        'of EUR/USD in the post-COVID rate cycle. When the Fed tightens faster\n'
        'than the ECB, the spread widens (USD pays more), attracting capital\n'
        'flows into dollar assets and depressing EUR/USD. The relationship\n'
        'weakens — and divergences open — around risk-off episodes (dollar\n'
        'smile) and when term-premium or growth differentials dominate at\n'
        'the long end.'
    )
    ax.text(0.5, 0.36, thesis,
            ha='center', va='center', fontsize=13, color='#c9d1d9',
            transform=ax.transAxes, linespacing=1.7)

    ax.text(0.5, 0.08, f'Data through {END}  |  FRED · ECB · Yahoo Finance',
            ha='center', va='center', fontsize=10, color='#8b949e',
            transform=ax.transAxes)

    for x, label, col in [(0.25, 'US–Euro 2Y Spread', PALETTE[0]),
                           (0.5,  'EUR/USD',           PALETTE[1]),
                           (0.75, 'DXY',               PALETTE[2])]:
        ax.add_patch(mpatches.FancyBboxPatch((x - 0.07, 0.14), 0.14, 0.06,
            boxstyle='round,pad=0.01', facecolor='#161b22',
            edgecolor=col, lw=1.5, transform=ax.transAxes))
        ax.text(x, 0.17, label, ha='center', va='center',
                fontsize=10, color=col, transform=ax.transAxes)

    pdf.savefig(fig, bbox_inches='tight')
    plt.close(fig)


def page_us_euro_2y(pdf: PdfPages, us2y: pd.Series, eu2y: pd.Series):
    fig = plt.figure(figsize=(11, 8.5))
    gs = GridSpec(2, 1, figure=fig, hspace=0.35, top=0.90, bottom=0.08)
    fig.suptitle('US 2Y vs Euro 2Y Yields', fontsize=16, fontweight='bold',
                 color='#e6edf3')

    ax1 = fig.add_subplot(gs[0])
    ax1.plot(us2y.index, us2y.values, color=PALETTE[0], lw=1.5, label='US 2Y')
    ax1.plot(eu2y.index, eu2y.values, color=PALETTE[1], lw=1.5, label='Euro AAA 2Y')
    ax1.set_ylabel('Yield (%)')
    ax1.set_title('2-Year Yields', fontsize=12)
    ax1.legend(loc='upper left')
    ax1.grid(True)
    add_fomc(ax1)

    spread = (us2y - eu2y).dropna() * 100  # bps
    ax2 = fig.add_subplot(gs[1])
    ax2.fill_between(spread.index, spread.values, 0,
                     where=(spread > 0), alpha=0.3, color=PALETTE[0], label='Positive (USD premium)')
    ax2.fill_between(spread.index, spread.values, 0,
                     where=(spread < 0), alpha=0.3, color=PALETTE[3], label='Negative (Euro premium)')
    ax2.plot(spread.index, spread.values, color='#e6edf3', lw=1.0)
    ax2.axhline(0, color='#8b949e', lw=0.8, ls='--')
    ax2.set_ylabel('Spread (bps)')
    ax2.set_title('US–Euro 2Y Spread (US minus Euro)', fontsize=12)
    ax2.legend(loc='upper left')
    ax2.grid(True)
    add_fomc(ax2)

    ax2.text(0.01, 0.04,
             'Vertical lines = FOMC meeting dates',
             transform=ax2.transAxes, fontsize=8, color='#8b949e')

    pdf.savefig(fig, bbox_inches='tight')
    plt.close(fig)


def page_spread_vs_fx(pdf: PdfPages, spread_bps: pd.Series,
                      fx: pd.Series, label_spread: str, label_fx: str,
                      title: str):
    fig = plt.figure(figsize=(11, 8.5))
    gs = GridSpec(2, 1, figure=fig, hspace=0.4, top=0.90, bottom=0.08,
                  height_ratios=[3, 1])
    fig.suptitle(title, fontsize=16, fontweight='bold', color='#e6edf3')

    df = align(spread_bps, fx)
    sp = df.iloc[:, 0]
    fx_s = df.iloc[:, 1]

    ax1 = fig.add_subplot(gs[0])
    ax1.set_facecolor('#161b22')

    # Shade divergence
    shade_divergence(ax1, sp, fx_s, _normalize(sp), _normalize(fx_s))

    color_spread = PALETTE[0]
    color_fx     = PALETTE[1]

    l1, = ax1.plot(sp.index, sp.values, color=color_spread, lw=1.6,
                   label=label_spread)
    ax1.set_ylabel(f'{label_spread} (bps)', color=color_spread)
    ax1.tick_params(axis='y', colors=color_spread)

    ax1b = ax1.twinx()
    ax1b.set_facecolor('#161b22')
    l2, = ax1b.plot(fx_s.index, fx_s.values, color=color_fx, lw=1.6,
                    label=label_fx)
    ax1b.set_ylabel(label_fx, color=color_fx)
    ax1b.tick_params(axis='y', colors=color_fx)

    ax1.legend(handles=[l1, l2], loc='upper left')
    ax1.grid(True)
    add_fomc(ax1)

    ax1.text(0.01, 0.04,
             'Red shading = spread/FX divergence (top quartile)',
             transform=ax1.transAxes, fontsize=8, color='#f78166')

    # Rolling correlation sub-panel
    rc = rolling_corr(sp, fx_s, window=60)
    ax2 = fig.add_subplot(gs[1])
    ax2.fill_between(rc.index, rc.values, 0,
                     where=(rc > 0), alpha=0.4, color=PALETTE[1])
    ax2.fill_between(rc.index, rc.values, 0,
                     where=(rc < 0), alpha=0.4, color=PALETTE[3])
    ax2.plot(rc.index, rc.values, color='#e6edf3', lw=0.9)
    ax2.axhline(0, color='#8b949e', lw=0.8, ls='--')
    ax2.set_ylim(-1.1, 1.1)
    ax2.set_ylabel('60d Corr')
    ax2.set_title('60-Day Rolling Correlation', fontsize=10)
    ax2.grid(True)

    pdf.savefig(fig, bbox_inches='tight')
    plt.close(fig)


def page_dxy_vs_2y(pdf: PdfPages, us2y: pd.Series, dxy: pd.Series):
    fig = plt.figure(figsize=(11, 8.5))
    gs = GridSpec(2, 1, figure=fig, hspace=0.4, top=0.90, bottom=0.08,
                  height_ratios=[3, 1])
    fig.suptitle('DXY vs US 2-Year Yield', fontsize=16, fontweight='bold',
                 color='#e6edf3')

    df = align(us2y, dxy)
    y2 = df.iloc[:, 0]
    dx = df.iloc[:, 1]

    ax1 = fig.add_subplot(gs[0])
    shade_divergence(ax1, y2, dx, _normalize(y2), _normalize(dx))

    l1, = ax1.plot(y2.index, y2.values, color=PALETTE[0], lw=1.6, label='US 2Y Yield (%)')
    ax1.set_ylabel('US 2Y Yield (%)', color=PALETTE[0])
    ax1.tick_params(axis='y', colors=PALETTE[0])

    ax1b = ax1.twinx()
    ax1b.set_facecolor('#161b22')
    l2, = ax1b.plot(dx.index, dx.values, color=PALETTE[2], lw=1.6, label='DXY')
    ax1b.set_ylabel('DXY', color=PALETTE[2])
    ax1b.tick_params(axis='y', colors=PALETTE[2])

    ax1.legend(handles=[l1, l2], loc='upper left')
    ax1.grid(True)
    add_fomc(ax1)

    rc = rolling_corr(y2, dx, window=60)
    ax2 = fig.add_subplot(gs[1])
    ax2.fill_between(rc.index, rc.values, 0,
                     where=(rc > 0), alpha=0.4, color=PALETTE[2])
    ax2.fill_between(rc.index, rc.values, 0,
                     where=(rc < 0), alpha=0.4, color=PALETTE[3])
    ax2.plot(rc.index, rc.values, color='#e6edf3', lw=0.9)
    ax2.axhline(0, color='#8b949e', lw=0.8, ls='--')
    ax2.set_ylim(-1.1, 1.1)
    ax2.set_ylabel('60d Corr')
    ax2.set_title('60-Day Rolling Correlation', fontsize=10)
    ax2.grid(True)

    pdf.savefig(fig, bbox_inches='tight')
    plt.close(fig)


def page_dollar_smile(pdf: PdfPages, dxy: pd.Series, vix: pd.Series,
                      us2y: pd.Series):
    fig, ax = plt.subplots(figsize=(11, 8.5))
    fig.suptitle('Dollar Smile Analysis', fontsize=16, fontweight='bold',
                 color='#e6edf3')

    df = align(vix.rename('vix'), dxy.rename('dxy'), us2y.rename('us2y'))
    df['dxy_1m_chg'] = df['dxy'].pct_change(21) * 100

    df = df.dropna()
    if df.empty:
        ax.text(0.5, 0.5, 'Insufficient data', ha='center', va='center',
                transform=ax.transAxes, fontsize=14, color='#f78166')
        pdf.savefig(fig, bbox_inches='tight')
        plt.close(fig)
        return

    sc = ax.scatter(df['vix'], df['dxy_1m_chg'],
                    c=df['us2y'], cmap='plasma', alpha=0.55, s=15,
                    vmin=df['us2y'].quantile(0.05),
                    vmax=df['us2y'].quantile(0.95))

    cbar = plt.colorbar(sc, ax=ax, pad=0.01)
    cbar.set_label('US 2Y Yield (%)', color='#e6edf3')
    cbar.ax.yaxis.set_tick_params(color='#8b949e')
    plt.setp(plt.getp(cbar.ax.axes, 'yticklabels'), color='#8b949e')

    ax.axhline(0, color='#8b949e', lw=0.8, ls='--')

    # Annotate quadrants
    xmax = df['vix'].quantile(0.95)
    ymax = df['dxy_1m_chg'].abs().quantile(0.90)
    ax.text(0.97, 0.97, 'High VIX → Risk-off USD bid',
            ha='right', va='top', transform=ax.transAxes,
            fontsize=9, color='#f78166')
    ax.text(0.03, 0.97, 'Low VIX, high rates → Carry USD bid',
            ha='left', va='top', transform=ax.transAxes,
            fontsize=9, color=PALETTE[0])

    # Polynomial trendline (LOWESS-style via np.polyfit bins)
    vix_bins = np.linspace(df['vix'].min(), df['vix'].max(), 50)
    try:
        poly = np.polyfit(df['vix'], df['dxy_1m_chg'], 2)
        trend = np.polyval(poly, vix_bins)
        ax.plot(vix_bins, trend, color=PALETTE[3], lw=2.0, ls='--',
                label='Quadratic fit (smile shape)')
        ax.legend(loc='lower center')
    except Exception:
        pass

    ax.set_xlabel('VIX Level')
    ax.set_ylabel('1-Month DXY Return (%)')
    ax.set_title(
        'Dollar Smile: USD strengthens in both high-VIX (risk-off) '
        'and high-rate (carry) regimes',
        fontsize=11, color='#c9d1d9')
    ax.grid(True)

    pdf.savefig(fig, bbox_inches='tight')
    plt.close(fig)


def page_carry_vol_regime(pdf: PdfPages, us2y: pd.Series, eu2y: pd.Series,
                          eurusd: pd.Series):
    fig = plt.figure(figsize=(11, 8.5))
    gs = GridSpec(2, 1, figure=fig, hspace=0.4, top=0.90, bottom=0.08,
                  height_ratios=[1, 2])
    fig.suptitle('Carry vs Vol Regime — EUR/USD Performance',
                 fontsize=16, fontweight='bold', color='#e6edf3')

    df = align(us2y.rename('us2y'), eu2y.rename('eu2y'),
               eurusd.rename('eurusd'))
    df['carry']    = df['us2y'] - df['eu2y']
    df['eurusd_vol'] = df['eurusd'].pct_change().rolling(21).std() * np.sqrt(252) * 100
    df = df.dropna()

    carry_med = df['carry'].median()
    high_carry = df['carry'] >= carry_med
    low_carry  = ~high_carry

    ax_carry = fig.add_subplot(gs[0])
    ax_carry.fill_between(df.index, df['carry'], carry_med,
                          where=high_carry, alpha=0.4, color=PALETTE[0],
                          label=f'High carry (≥ {carry_med:.2f}%)')
    ax_carry.fill_between(df.index, df['carry'], carry_med,
                          where=low_carry, alpha=0.4, color=PALETTE[3],
                          label=f'Low carry (< {carry_med:.2f}%)')
    ax_carry.plot(df.index, df['carry'], color='#e6edf3', lw=0.9)
    ax_carry.axhline(carry_med, color='#8b949e', lw=0.8, ls='--')
    ax_carry.set_ylabel('US–Euro 2Y (%)')
    ax_carry.set_title('Carry (US 2Y – Euro 2Y)', fontsize=11)
    ax_carry.legend(loc='upper left', fontsize=8)
    ax_carry.grid(True)

    ax_fx = fig.add_subplot(gs[1])

    # Shade background by carry regime
    for i in range(1, len(df)):
        col = PALETTE[0] if high_carry.iloc[i] else PALETTE[3]
        ax_fx.axvspan(df.index[i - 1], df.index[i], alpha=0.06,
                      color=col, zorder=0)

    ax_fx.plot(df.index, df['eurusd'], color='#e6edf3', lw=1.4,
               label='EUR/USD', zorder=2)

    ax_fxb = ax_fx.twinx()
    ax_fxb.set_facecolor('#161b22')
    ax_fxb.plot(df.index, df['eurusd_vol'], color=PALETTE[2], lw=1.0,
                alpha=0.7, label='21d Realised Vol (%)')
    ax_fxb.set_ylabel('EUR/USD Annualised Vol (%)', color=PALETTE[2])
    ax_fxb.tick_params(axis='y', colors=PALETTE[2])

    ax_fx.set_ylabel('EUR/USD')
    ax_fx.set_title('EUR/USD with Carry Regime Overlay', fontsize=11)
    ax_fx.grid(True)

    lines1, labels1 = ax_fx.get_legend_handles_labels()
    lines2, labels2 = ax_fxb.get_legend_handles_labels()
    ax_fx.legend(lines1 + lines2, labels1 + labels2, loc='lower left', fontsize=8)

    # Summary stats annotation
    hc_ret = df.loc[high_carry, 'eurusd'].pct_change().mean() * 252 * 100
    lc_ret = df.loc[low_carry,  'eurusd'].pct_change().mean() * 252 * 100
    ax_fx.text(0.99, 0.97,
               f'Ann. EUR/USD drift\nHigh carry: {hc_ret:+.1f}%\nLow carry:  {lc_ret:+.1f}%',
               ha='right', va='top', transform=ax_fx.transAxes,
               fontsize=9, color='#c9d1d9',
               bbox=dict(facecolor='#161b22', edgecolor='#30363d', pad=4))

    pdf.savefig(fig, bbox_inches='tight')
    plt.close(fig)


def page_dislocation_table(pdf: PdfPages, us2y: pd.Series, eu2y: pd.Series,
                           us10y: pd.Series, eu10y: pd.Series,
                           eurusd: pd.Series, dxy: pd.Series):
    fig = plt.figure(figsize=(11, 8.5))
    fig.patch.set_facecolor('#0d1117')
    ax = fig.add_axes([0.05, 0.05, 0.90, 0.90])
    ax.set_facecolor('#0d1117')
    ax.axis('off')

    fig.suptitle('Current Dislocation — Rate Differential Snapshot',
                 fontsize=16, fontweight='bold', color='#e6edf3', y=0.97)

    def safe_last(s):
        return float(s.dropna().iloc[-1]) if not s.dropna().empty else np.nan

    def safe_mean(s, days):
        trimmed = s.dropna().iloc[-days:]
        return float(trimmed.mean()) if len(trimmed) >= 5 else np.nan

    sp2y   = (us2y - eu2y) * 100
    sp10y  = (us10y - eu10y) * 100

    cur_sp2y   = safe_last(sp2y)
    cur_sp10y  = safe_last(sp10y)
    cur_eurusd = safe_last(eurusd)
    cur_dxy    = safe_last(dxy)
    cur_us2y   = safe_last(us2y)
    cur_eu2y   = safe_last(eu2y)

    avg1y_sp2y  = safe_mean(sp2y,   252)
    avg2y_sp2y  = safe_mean(sp2y,   504)
    avg1y_sp10y = safe_mean(sp10y,  252)
    avg2y_sp10y = safe_mean(sp10y,  504)

    # OLS fair value for EUR/USD from 2Y spread
    df_reg = align(sp2y.rename('sp'), eurusd.rename('fx')).dropna()
    fair_value = np.nan
    if len(df_reg) >= 30:
        coeffs = np.polyfit(df_reg['sp'], df_reg['fx'], 1)
        fair_value = np.polyval(coeffs, cur_sp2y) if not np.isnan(cur_sp2y) else np.nan

    def fmt(v, dec=2):
        return f'{v:.{dec}f}' if not np.isnan(v) else 'N/A'

    rows = [
        ('Metric',                       'Current',              '1Y Avg',             '2Y Avg'),
        ('US 2Y Yield (%)',               fmt(cur_us2y),          fmt(avg1y_sp2y - cur_sp2y + cur_us2y), '—'),
        ('Euro AAA 2Y Yield (%)',         fmt(cur_eu2y),          '—',                  '—'),
        ('US–Euro 2Y Spread (bps)',       fmt(cur_sp2y, 1),       fmt(avg1y_sp2y, 1),   fmt(avg2y_sp2y, 1)),
        ('US–Euro 10Y Spread (bps)',      fmt(cur_sp10y, 1),      fmt(avg1y_sp10y, 1),  fmt(avg2y_sp10y, 1)),
        ('EUR/USD',                       fmt(cur_eurusd, 4),     '—',                  '—'),
        ('EUR/USD Fair Value (2Y reg.)',  fmt(fair_value, 4),     '—',                  '—'),
        ('EUR/USD Dislocation (pips)',    fmt((cur_eurusd - fair_value) * 10000, 0) if not np.isnan(fair_value) else 'N/A', '—', '—'),
        ('DXY',                           fmt(cur_dxy, 2),        '—',                  '—'),
        ('UK 2Y',                         'N/A',                  'N/A (unavailable)',  '—'),
    ]

    col_x = [0.02, 0.38, 0.60, 0.80]
    row_h = 0.075
    y0    = 0.88

    header_cols = ['#58a6ff', '#3fb950', '#d29922', '#bc8cff']
    for ci, (txt, col) in enumerate(zip(rows[0], header_cols)):
        ax.text(col_x[ci], y0, txt, transform=ax.transAxes,
                fontsize=11, fontweight='bold', color=col, va='top')

    ax.plot([0.02, 0.98], [y0 - 0.015, y0 - 0.015], color='#30363d',
            lw=1.0, transform=ax.transAxes, clip_on=False)

    for ri, row in enumerate(rows[1:], 1):
        y = y0 - ri * row_h
        bg = '#161b22' if ri % 2 == 0 else '#0d1117'
        ax.add_patch(mpatches.FancyBboxPatch(
            (0.01, y - 0.005), 0.98, row_h - 0.005,
            boxstyle='square,pad=0', facecolor=bg,
            edgecolor='none', transform=ax.transAxes, zorder=0))
        for ci, cell in enumerate(row):
            color = '#e6edf3' if ci == 0 else '#c9d1d9'
            ax.text(col_x[ci], y, cell, transform=ax.transAxes,
                    fontsize=10, color=color, va='top')

    # Narrative block
    y_narr = y0 - (len(rows)) * row_h - 0.02
    disloc = (cur_eurusd - fair_value) * 10000 if not np.isnan(fair_value) else np.nan
    if not np.isnan(disloc):
        direction = 'above' if disloc > 0 else 'below'
        mag = abs(disloc)
        narrative = (
            f'Current EUR/USD is {mag:.0f} pips {direction} the fair value implied by the '
            f'2Y spread regression. The US–Euro 2Y spread stands at {fmt(cur_sp2y, 1)} bps '
            f'vs a 1-year average of {fmt(avg1y_sp2y, 1)} bps. '
        )
        if cur_sp2y > avg1y_sp2y:
            narrative += ('The spread is above its historical norm, consistent with continued '
                          'USD strength via rate carry. Watch for ECB/Fed pivot signals '
                          'to drive mean reversion.')
        else:
            narrative += ('The spread is below its historical average, suggesting USD rate '
                          'advantage is fading. EUR/USD may find support as carry differentials '
                          'compress.')
    else:
        narrative = 'Insufficient data for regression-based fair value estimate.'

    ax.text(0.01, y_narr,
            'Key Takeaway\n' + narrative,
            transform=ax.transAxes, fontsize=10, color='#c9d1d9',
            va='top', wrap=True,
            bbox=dict(facecolor='#161b22', edgecolor='#30363d', pad=8))

    pdf.savefig(fig, bbox_inches='tight')
    plt.close(fig)


# ── Main ──────────────────────────────────────────────────────────────────────

def fetch_all():
    print('Fetching data...')
    print('  FRED: US 2Y, 5Y, 10Y, DFF')
    us2y  = _fred('DGS2')
    us5y  = _fred('DGS5')
    us10y = _fred('DGS10')

    print('  ECB: Euro 2Y, 5Y, 10Y')
    eu2y  = _ecb(2)
    eu10y = _ecb(10)

    print('  yfinance: EURUSD, GBPUSD, DXY, VIX')
    eurusd = _yf('EURUSD=X')
    gbpusd = _yf('GBPUSD=X')
    dxy    = _yf('DX-Y.NYB')
    vix    = _yf('^VIX')

    print('  Done.\n')
    return us2y, us5y, us10y, eu2y, eu10y, eurusd, gbpusd, dxy, vix


def main():
    parser = argparse.ArgumentParser(description='Rate Differentials and FX PDF')
    parser.add_argument('--out', default='rate_differentials_fx.pdf',
                        help='Output PDF path')
    args = parser.parse_args()

    us2y, us5y, us10y, eu2y, eu10y, eurusd, gbpusd, dxy, vix = fetch_all()

    sp2y_bps  = (us2y - eu2y).dropna() * 100
    sp10y_bps = (us10y - eu10y).dropna() * 100

    print(f'Writing PDF: {args.out}')
    with PdfPages(args.out) as pdf:
        # P1 — Cover
        page_cover(pdf)

        # P2 — US 2Y vs Euro 2Y
        page_us_euro_2y(pdf, us2y, eu2y)

        # P3 — US-Euro 2Y spread vs EUR/USD  (KEY chart)
        page_spread_vs_fx(pdf, sp2y_bps, eurusd,
                          label_spread='US–Euro 2Y Spread',
                          label_fx='EUR/USD',
                          title='US–Euro 2Y Spread vs EUR/USD  [Key Driver Chart]')

        # P4 — US-Euro 10Y spread vs EUR/USD
        page_spread_vs_fx(pdf, sp10y_bps, eurusd,
                          label_spread='US–Euro 10Y Spread',
                          label_fx='EUR/USD',
                          title='US–Euro 10Y Spread vs EUR/USD  [Growth & Term Premium]')

        # P5 — DXY vs US 2Y
        page_dxy_vs_2y(pdf, us2y, dxy)

        # P6 — Dollar Smile
        page_dollar_smile(pdf, dxy, vix, us2y)

        # P7 — Carry vs Vol Regime
        page_carry_vol_regime(pdf, us2y, eu2y, eurusd)

        # P8 — Current Dislocation Table
        page_dislocation_table(pdf, us2y, eu2y, us10y, eu10y, eurusd, dxy)

    print(f'Done. PDF saved to: {args.out}')


if __name__ == '__main__':
    main()
