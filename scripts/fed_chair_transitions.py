"""
Fed Chair Transition Dynamics
Generates a multi-page PDF: spot yields, spreads, realised vol, PCA.

Usage:
    python scripts/fed_chair_transitions.py [--out output.pdf]

Dependencies:
    pip install pandas scikit-learn matplotlib requests  (no pandas_datareader needed)
"""

import argparse
import io
import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.backends.backend_pdf import PdfPages
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import requests

# ── Style ─────────────────────────────────────────────────────────────────────
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

PALETTE = [
    '#58a6ff', '#3fb950', '#d29922', '#f78166',
    '#bc8cff', '#79c0ff', '#56d364', '#e3b341',
]

# ── Transition dates ──────────────────────────────────────────────────────────
TRANSITIONS = [
    dict(label='Bernanke',     chair='Ben Bernanke',
         nomination='2005-10-24', confirm='2006-01-31', start='2006-02-01'),
    dict(label='Yellen',       chair='Janet Yellen',
         nomination='2013-10-09', confirm='2014-01-06', start='2014-02-03'),
    dict(label='Powell (1st)', chair='Jerome Powell',
         nomination='2017-11-02', confirm='2018-01-23', start='2018-02-05'),
    dict(label='Powell (2nd)', chair='Jerome Powell (renominated)',
         nomination='2021-11-22', confirm='2022-05-23', start='2022-05-23'),
    # Add current 2026 successor here once confirmed:
    # dict(label='[Successor]', chair='???',
    #      nomination='2025-??-??', confirm='2026-??-??', start='2026-05-15'),
]

for t in TRANSITIONS:
    for k in ('nomination', 'confirm', 'start'):
        t[k] = pd.Timestamp(t[k])
    t['limbo_days'] = (t['start'] - t['nomination']).days

# ── FRED series ───────────────────────────────────────────────────────────────
FRED_SERIES = {
    '1M': 'DGS1MO', '3M': 'DGS3MO', '6M': 'DGS6MO',
    '1Y': 'DGS1',   '2Y': 'DGS2',   '3Y': 'DGS3',
    '5Y': 'DGS5',   '7Y': 'DGS7',   '10Y': 'DGS10',
    '20Y': 'DGS20', '30Y': 'DGS30',
}
TENORS      = list(FRED_SERIES.keys())
TENOR_MATS  = [1/12, 3/12, 6/12, 1, 2, 3, 5, 7, 10, 20, 30]
PCA_TENORS  = ['3M', '2Y', '5Y', '10Y', '30Y']
PCA_MATS    = [3/12, 2, 5, 10, 30]

PRE_DAYS   = 90
POST_DAYS  = 90
ROLL_WIN   = 250


# FRED FX series: DEXUSEU = USD per EUR (EURUSD), DEXUSUK = USD per GBP (GBPUSD)
FX_SERIES = {'EURUSD': 'DEXUSEU', 'GBPUSD': 'DEXUSUK'}


# ── Data fetching — direct FRED CSV (no API key, no pandas_datareader) ────────
def _fred_series(series_id: str, start: str, end: str) -> pd.Series:
    url = (
        f'https://fred.stlouisfed.org/graph/fredgraph.csv'
        f'?id={series_id}&vintage_date={end}'
    )
    r = requests.get(url, timeout=30)
    r.raise_for_status()
    df = pd.read_csv(io.StringIO(r.text), index_col='observation_date')
    df.index = pd.to_datetime(df.index)
    s  = pd.to_numeric(df.iloc[:, 0], errors='coerce')
    s.name = series_id
    return s.loc[start:end]


def fetch_series_dict(series_dict: dict, label: str,
                      start='2004-01-01', end='2026-05-03') -> pd.DataFrame:
    print(f'Fetching {label} from FRED ({start} → {end}) ...')
    frames = {}
    for col, sid in series_dict.items():
        print(f'  {sid} ({col})', end='', flush=True)
        frames[col] = _fred_series(sid, start, end)
        print(' ✓')
    df = pd.DataFrame(frames).ffill()
    # drop rows where ALL are NaN (weekends stored as '.' on some series)
    df = df[df.notna().any(axis=1)]
    print(f'  {len(df)} days loaded.')
    return df


def fetch_yields(start='2004-01-01', end='2026-05-03') -> pd.DataFrame:
    return fetch_series_dict(FRED_SERIES, 'Treasury yields', start, end)


def fetch_fx(start='2004-01-01', end='2026-05-03') -> pd.DataFrame:
    return fetch_series_dict(FX_SERIES, 'FX rates', start, end)


# ── Helpers ───────────────────────────────────────────────────────────────────
def window(df, t, pre=PRE_DAYS, post=POST_DAYS):
    lo = t['nomination'] - pd.Timedelta(days=pre)
    hi = t['start']      + pd.Timedelta(days=post)
    return df.loc[lo:hi].copy()

def xdays(df, nom):
    return (df.index - nom).days

def add_event_lines(ax, t):
    limbo = t['limbo_days']
    ax.axvspan(0, limbo, alpha=0.12, color='#d29922')
    ax.axvline(0,     color='#d29922', lw=1.2, ls='--', alpha=0.9, label='Nomination')
    ax.axvline(limbo, color='#3fb950', lw=1.2, ls='--', alpha=0.9, label='Start date')
    ax.axhline(0, color='#8b949e', lw=0.7, ls=':')


# ── Page 1 — Cover / summary table ───────────────────────────────────────────
def page_cover(pdf):
    fig, ax = plt.subplots(figsize=(11, 8.5))
    ax.axis('off')
    title = 'Fed Chair Transition Dynamics\nSpot Yields · Realised Vol · PCA'
    ax.text(0.5, 0.75, title, ha='center', va='center', fontsize=22,
            color='#e6edf3', transform=ax.transAxes, fontweight='bold')

    cols = ['Label', 'Chair', 'Nomination', 'Start', 'Limbo (days)']
    rows = [[t['label'], t['chair'],
             t['nomination'].strftime('%Y-%m-%d'),
             t['start'].strftime('%Y-%m-%d'),
             str(t['limbo_days'])] for t in TRANSITIONS]

    tbl = ax.table(cellText=rows, colLabels=cols, loc='center',
                   bbox=[0.05, 0.25, 0.9, 0.35])
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(10)
    for (r, c), cell in tbl.get_celld().items():
        cell.set_facecolor('#21262d' if r == 0 else '#161b22')
        cell.set_edgecolor('#30363d')
        cell.set_text_props(color='#e6edf3')

    ax.text(0.5, 0.15,
            'Event windows:  Pre = 90 days before nomination  |  '
            'Limbo = nomination → start  |  Post = 90 days after start',
            ha='center', va='center', fontsize=9, color='#8b949e',
            transform=ax.transAxes)

    pdf.savefig(fig, bbox_inches='tight')
    plt.close(fig)


# ── Page 2 — Spot yield levels ────────────────────────────────────────────────
def page_spot_yields(pdf, raw):
    PLOT_TENORS = ['2Y', '5Y', '10Y', '30Y']
    fig, axes = plt.subplots(2, 2, figsize=(16, 10))
    axes = axes.flatten()

    for ax, t in zip(axes, TRANSITIONS):
        wd   = window(raw, t)
        nom  = t['nomination']
        base = raw.loc[:nom].iloc[-1]
        idx  = wd[PLOT_TENORS] - base[PLOT_TENORS]
        x    = xdays(idx, nom)

        for i, tenor in enumerate(PLOT_TENORS):
            ax.plot(x, idx[tenor] * 100, color=PALETTE[i], lw=1.6, label=tenor)

        add_event_lines(ax, t)
        ax.set_title(f'{t["label"]}  ({t["nomination"].strftime("%b %Y")} → {t["start"].strftime("%b %Y")})')
        ax.set_xlabel('Calendar days from nomination')
        ax.set_ylabel('Δ yield (bps, indexed to nomination)')
        ax.legend(fontsize=8, loc='upper left')
        ax.grid(True)

    fig.suptitle('Spot yield changes around Fed Chair transitions\n'
                 'Indexed to 0 at nomination day  |  shaded = limbo period', y=1.01)
    plt.tight_layout()
    pdf.savefig(fig, bbox_inches='tight')
    plt.close(fig)


# ── Page 3 — Curve spreads ────────────────────────────────────────────────────
def page_spreads(pdf, raw):
    raw = raw.copy()
    raw['2s10s'] = raw['10Y'] - raw['2Y']
    raw['3m10y'] = raw['10Y'] - raw['3M']
    SPREADS = ['2s10s', '3m10y']

    fig, axes = plt.subplots(2, 2, figsize=(16, 10))
    axes = axes.flatten()

    for ax, t in zip(axes, TRANSITIONS):
        wd = window(raw, t)[SPREADS]
        x  = xdays(wd, t['nomination'])

        ax.plot(x, wd['2s10s'] * 100, color='#58a6ff', lw=1.6, label='2s10s')
        ax.plot(x, wd['3m10y'] * 100, color='#f78166', lw=1.6, label='3m10y')
        add_event_lines(ax, t)
        ax.set_title(t['label'])
        ax.set_xlabel('Days from nomination')
        ax.set_ylabel('Spread (bps)')
        ax.legend(fontsize=8)
        ax.grid(True)

    fig.suptitle('Curve spreads (2s10s and 3m10y) around Fed Chair transitions', y=1.01)
    plt.tight_layout()
    pdf.savefig(fig, bbox_inches='tight')
    plt.close(fig)


# ── Page 4 — Realised vol ─────────────────────────────────────────────────────
def page_rvol(pdf, raw):
    RVOL_TENORS = ['2Y', '5Y', '10Y', '30Y']
    rvol = (raw[TENORS].diff() * 100).rolling(21).std()

    fig, axes = plt.subplots(2, 2, figsize=(16, 10))
    axes = axes.flatten()

    for ax, t in zip(axes, TRANSITIONS):
        wd = window(rvol, t)[RVOL_TENORS]
        x  = xdays(wd, t['nomination'])

        for i, tenor in enumerate(RVOL_TENORS):
            ax.plot(x, wd[tenor], color=PALETTE[i], lw=1.5, label=tenor)

        add_event_lines(ax, t)
        ax.set_title(t['label'])
        ax.set_xlabel('Days from nomination')
        ax.set_ylabel('21d realised vol (bps/day)')
        ax.legend(fontsize=8)
        ax.grid(True)

    fig.suptitle('Realised vol — 21d rolling σ of daily Δyield', y=1.01)
    plt.tight_layout()
    pdf.savefig(fig, bbox_inches='tight')
    plt.close(fig)

    return rvol


# ── Pages 5–8 — PCA: loadings + scores per transition ─────────────────────────
def page_pca_per_transition(pdf, raw):
    for t in TRANSITIONS:
        nom = t['nomination']
        fit_start = nom - pd.Timedelta(days=730)
        fit_data  = raw.loc[fit_start:nom, PCA_TENORS].dropna()
        scaler    = StandardScaler()
        fit_sc    = scaler.fit_transform(fit_data)
        pca       = PCA(n_components=3)
        pca.fit(fit_sc)
        evr = pca.explained_variance_ratio_

        wd   = window(raw, t)[PCA_TENORS].dropna()
        proj = pca.transform(scaler.transform(wd))
        x    = xdays(wd, nom)

        pc_labels = [
            f'PC1 level ({evr[0]*100:.1f}%)',
            f'PC2 slope ({evr[1]*100:.1f}%)',
            f'PC3 curvature ({evr[2]*100:.1f}%)',
        ]
        colors = ['#58a6ff', '#3fb950', '#d29922']

        fig, axes = plt.subplots(1, 4, figsize=(20, 5))

        # Loadings
        ax = axes[0]
        for j in range(3):
            ax.plot(PCA_MATS, pca.components_[j],
                    color=colors[j], marker='o', ms=4, label=pc_labels[j])
        ax.set_title('PC loadings')
        ax.set_xlabel('Tenor (years)')
        ax.set_ylabel('Loading')
        ax.axhline(0, color='#8b949e', lw=0.7, ls=':')
        ax.legend(fontsize=7)
        ax.grid(True)

        # PC1, PC2, PC3 scores
        for col, (pc_idx, label) in enumerate(zip([0, 1, 2], pc_labels), start=1):
            ax = axes[col]
            ax.plot(x, proj[:, pc_idx], color=colors[pc_idx], lw=1.5)
            add_event_lines(ax, t)
            ax.set_title(label)
            ax.set_xlabel('Days from nomination')
            ax.set_ylabel('Score')
            ax.grid(True)

        fig.suptitle(
            f'PCA — {t["label"]}  ({t["nomination"].strftime("%b %Y")} → {t["start"].strftime("%b %Y")})\n'
            'PCA fit on 2 years pre-nomination  |  shaded = limbo  |  green dash = start',
            y=1.02
        )
        plt.tight_layout()
        pdf.savefig(fig, bbox_inches='tight')
        plt.close(fig)


# ── Page — Rolling PCA ────────────────────────────────────────────────────────
def rolling_pca_scores(df, tenors, window_size):
    arr = df[tenors].values
    idx = df.index
    records = []
    for i in range(window_size - 1, len(arr)):
        block = arr[i - window_size + 1 : i + 1]
        if np.isnan(block).any():
            records.append([np.nan] * 3)
            continue
        sc    = StandardScaler().fit_transform(block)
        score = PCA(n_components=3).fit(sc).transform(sc[-1:])[0]
        records.append(score)
    return pd.DataFrame(records, index=idx[window_size - 1:], columns=['PC1', 'PC2', 'PC3'])

def page_rolling_pca(pdf, raw):
    print(f'  Computing rolling PCA ({ROLL_WIN}-day window) ...')
    scores = rolling_pca_scores(raw, PCA_TENORS, ROLL_WIN)

    fig, axes = plt.subplots(2, 2, figsize=(16, 10))
    axes = axes.flatten()
    pc_colors = {'PC1': '#58a6ff', 'PC2': '#3fb950', 'PC3': '#d29922'}

    for ax, t in zip(axes, TRANSITIONS):
        wd = window(scores, t).dropna()
        x  = xdays(wd, t['nomination'])

        for pc, color in pc_colors.items():
            ax.plot(x, wd[pc], color=color, lw=1.5, label=pc)

        add_event_lines(ax, t)
        ax.set_title(f'{t["label"]} — rolling PCA ({ROLL_WIN}d)')
        ax.set_xlabel('Days from nomination')
        ax.set_ylabel('Score')
        ax.legend(fontsize=8)
        ax.grid(True)

    fig.suptitle(
        f'Rolling PCA scores ({ROLL_WIN}-day window)  |  PC1=level · PC2=slope · PC3=curvature\n'
        'shaded = limbo  |  green dash = start date',
        y=1.01
    )
    plt.tight_layout()
    pdf.savefig(fig, bbox_inches='tight')
    plt.close(fig)

    return scores


# ── Page — Cross-transition overlay ──────────────────────────────────────────
def page_overlay(pdf, raw, rvol, roll_scores):
    OVERLAY_TENOR = '10Y'
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))

    series_cfg = [
        (axes[0], raw[OVERLAY_TENOR],        f'{OVERLAY_TENOR} yield change (bps)', True),
        (axes[1], rvol[OVERLAY_TENOR],        f'{OVERLAY_TENOR} 21d realised vol (bps/day)', False),
        (axes[2], roll_scores['PC2'],          f'Rolling PC2 slope ({ROLL_WIN}d window)', False),
    ]

    for ax, series, ylabel, index_to_zero in series_cfg:
        for i, t in enumerate(TRANSITIONS):
            nom = t['nomination']
            lo  = nom - pd.Timedelta(days=PRE_DAYS)
            hi  = t['start'] + pd.Timedelta(days=POST_DAYS)
            wd  = series.loc[lo:hi].dropna()
            x   = xdays(wd, nom)
            y   = wd.values.copy()
            if index_to_zero:
                base = series.loc[:nom].iloc[-1]
                y    = (y - base) * 100
            ax.plot(x, y, color=PALETTE[i], lw=1.5, label=t['label'], alpha=0.85)

        ax.axvline(0, color='#d29922', lw=1.5, ls='--', alpha=0.9, label='Nomination day')
        ax.axhline(0, color='#8b949e', lw=0.7, ls=':')
        ax.set_xlabel('Days from nomination')
        ax.set_ylabel(ylabel)
        ax.set_title(ylabel.split('(')[0].strip())
        ax.legend(fontsize=7)
        ax.grid(True)

    fig.suptitle(f'Cross-transition overlay — {OVERLAY_TENOR}  |  all transitions aligned to nomination = day 0', y=1.02)
    plt.tight_layout()
    pdf.savefig(fig, bbox_inches='tight')
    plt.close(fig)


# ── Page — FX performance ────────────────────────────────────────────────────
def page_fx(pdf, fx):
    PAIRS = ['EURUSD', 'GBPUSD']
    pair_colors = {'EURUSD': '#58a6ff', 'GBPUSD': '#3fb950'}
    fx_rvol = (fx[PAIRS].pct_change() * 100).rolling(21).std()

    # Layout: 2 rows (indexed level, realised vol) × 4 cols (transitions)
    fig, axes = plt.subplots(2, len(TRANSITIONS), figsize=(5 * len(TRANSITIONS), 10))

    for col, t in enumerate(TRANSITIONS):
        nom = t['nomination']

        # Row 0: indexed level (% from nomination)
        ax0 = axes[0, col]
        wd  = window(fx, t)[PAIRS].dropna()
        x   = xdays(wd, nom)
        for pair in PAIRS:
            base = fx.loc[:nom, pair].iloc[-1]
            indexed_pct = (wd[pair] / base - 1) * 100
            ax0.plot(x, indexed_pct, color=pair_colors[pair], lw=1.6, label=pair)
        add_event_lines(ax0, t)
        ax0.set_title(t['label'])
        ax0.set_ylabel('% change from nomination' if col == 0 else '')
        ax0.set_xlabel('Days from nomination')
        ax0.legend(fontsize=8)
        ax0.grid(True)

        # Row 1: realised vol
        ax1 = axes[1, col]
        wd_vol = window(fx_rvol, t)[PAIRS].dropna()
        xv     = xdays(wd_vol, nom)
        for pair in PAIRS:
            ax1.plot(xv, wd_vol[pair], color=pair_colors[pair], lw=1.6, label=pair)
        add_event_lines(ax1, t)
        ax1.set_title(t['label'])
        ax1.set_ylabel('21d realised vol (%/day)' if col == 0 else '')
        ax1.set_xlabel('Days from nomination')
        ax1.legend(fontsize=8)
        ax1.grid(True)

    axes[0, 0].set_ylabel('% change from nomination')
    axes[1, 0].set_ylabel('21d realised vol (%/day)')

    fig.suptitle(
        'EURUSD & GBPUSD around Fed Chair transitions\n'
        'Top: price indexed to 0 at nomination  |  Bottom: 21d realised vol\n'
        'shaded = limbo  |  green dash = start date',
        y=1.02
    )
    plt.tight_layout()
    pdf.savefig(fig, bbox_inches='tight')
    plt.close(fig)

    # Overlay: all transitions on same axis per pair
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    for ax, pair in zip(axes, PAIRS):
        for i, t in enumerate(TRANSITIONS):
            nom  = t['nomination']
            lo   = nom - pd.Timedelta(days=PRE_DAYS)
            hi   = t['start'] + pd.Timedelta(days=POST_DAYS)
            wd   = fx.loc[lo:hi, pair].dropna()
            x    = xdays(wd, nom)
            base = fx.loc[:nom, pair].iloc[-1]
            y    = (wd / base - 1) * 100
            ax.plot(x, y, color=PALETTE[i], lw=1.5, label=t['label'], alpha=0.85)
        ax.axvline(0, color='#d29922', lw=1.5, ls='--', alpha=0.9, label='Nomination')
        ax.axhline(0, color='#8b949e', lw=0.7, ls=':')
        ax.set_title(pair)
        ax.set_xlabel('Days from nomination')
        ax.set_ylabel('% change from nomination')
        ax.legend(fontsize=7)
        ax.grid(True)
    fig.suptitle('EURUSD & GBPUSD — cross-transition overlay  |  nomination = day 0', y=1.02)
    plt.tight_layout()
    pdf.savefig(fig, bbox_inches='tight')
    plt.close(fig)


# ── Page — Summary heatmaps ───────────────────────────────────────────────────
def page_heatmaps(pdf, raw, rvol):
    def phase_rows(tenors):
        rows = []
        for t in TRANSITIONS:
            nom, start = t['nomination'], t['start']
            phases = {
                'pre':   (nom - pd.Timedelta(days=PRE_DAYS), nom - pd.Timedelta(days=1)),
                'limbo': (nom, start - pd.Timedelta(days=1)),
                'post':  (start, start + pd.Timedelta(days=POST_DAYS)),
            }
            for phase, (lo, hi) in phases.items():
                y  = raw.loc[lo:hi, tenors]
                rv = rvol.loc[lo:hi, tenors]
                chg     = (y.iloc[-1] - y.iloc[0]) * 100
                avg_vol = rv.mean()
                for tenor in tenors:
                    rows.append({'transition': t['label'], 'phase': phase,
                                 'tenor': tenor,
                                 'yield_chg_bps': round(chg[tenor], 1),
                                 'avg_rvol_bpd':  round(avg_vol[tenor], 2)})
        return rows

    HEAT_TENORS = ['2Y', '5Y', '10Y', '30Y']
    df = pd.DataFrame(phase_rows(HEAT_TENORS))

    pivot_chg = df.pivot_table(index=['transition', 'tenor'], columns='phase',
                                values='yield_chg_bps')[['pre', 'limbo', 'post']]
    pivot_vol = df.pivot_table(index=['transition', 'tenor'], columns='phase',
                                values='avg_rvol_bpd')[['pre', 'limbo', 'post']]

    def draw_heatmap(ax, pivot, title, cmap):
        data = pivot.values.astype(float)
        vmax = np.nanpercentile(np.abs(data), 95)
        im = ax.imshow(data, cmap=cmap, aspect='auto',
                       vmin=-vmax if 'RdYlGn' in cmap else 0, vmax=vmax)
        ax.set_xticks(range(len(pivot.columns)))
        ax.set_xticklabels(pivot.columns)
        ax.set_yticks(range(len(pivot.index)))
        ax.set_yticklabels([' / '.join(map(str, idx)) for idx in pivot.index], fontsize=8)
        for i in range(data.shape[0]):
            for j in range(data.shape[1]):
                v = data[i, j]
                if not np.isnan(v):
                    ax.text(j, i, f'{v:.1f}', ha='center', va='center', fontsize=7,
                            color='black' if abs(v) < vmax * 0.5 else 'white')
        plt.colorbar(im, ax=ax, shrink=0.6)
        ax.set_title(title)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))
    draw_heatmap(ax1, pivot_chg, 'Yield change (bps) by phase', 'RdYlGn')
    draw_heatmap(ax2, pivot_vol, 'Avg realised vol (bps/day) by phase', 'YlOrRd')
    fig.suptitle('Summary heatmaps by transition, tenor, and event phase', y=1.02)
    plt.tight_layout()
    pdf.savefig(fig, bbox_inches='tight')
    plt.close(fig)


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--out', default='fed_chair_transitions.pdf',
                        help='Output PDF path')
    args = parser.parse_args()

    raw   = fetch_yields()
    fx    = fetch_fx()
    rvol  = (raw[TENORS].diff() * 100).rolling(21).std()

    print(f'Writing {args.out} ...')
    with PdfPages(args.out) as pdf:
        page_cover(pdf)
        print('  [1/8] Spot yields')
        page_spot_yields(pdf, raw)
        print('  [2/8] Spreads')
        page_spreads(pdf, raw)
        print('  [3/8] Realised vol (rates)')
        page_rvol(pdf, raw)
        print('  [4/8] PCA per transition')
        page_pca_per_transition(pdf, raw)
        print('  [5/8] Rolling PCA')
        roll_scores = page_rolling_pca(pdf, raw)
        print('  [6/8] Cross-transition overlay (rates)')
        page_overlay(pdf, raw, rvol, roll_scores)
        print('  [7/8] FX (EURUSD, GBPUSD)')
        page_fx(pdf, fx)
        print('  [8/8] Summary heatmaps')
        page_heatmaps(pdf, raw, rvol)

        meta = pdf.infodict()
        meta['Title']   = 'Fed Chair Transition Dynamics'
        meta['Author']  = 'agentic-equity-research'
        meta['Subject'] = 'Yield curve, realised vol, PCA, FX around Fed Chair transitions'

    print(f'\nDone → {args.out}')

if __name__ == '__main__':
    main()
