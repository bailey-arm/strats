"""
Warsh Nomination Window — Fixed Income PCA Dynamics
Rolling PCA on US Treasuries and Euro AAA bonds, plus UK and FX context.

Data sources  (all free, no API key required):
  · US yields   →  FRED  (DGS2/5/10/30)
  · Euro yields →  ECB SDW API  (AAA Euro area 2/5/10/30Y)
  · UK context  →  yfinance  IGLT.L  (iShares UK Gilts ETF, price proxy)
  · Country 10Y →  FRED  (where daily series exist; Germany, France, Italy)
  · FX          →  yfinance  (EURUSD, GBPUSD, EURGBP, DXY)

Usage:
    python scripts/warsh_pca_dynamics.py [--out warsh_pca.pdf]

Dependencies:
    pip install pandas scikit-learn matplotlib requests yfinance
"""

import argparse, io, warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.backends.backend_pdf import PdfPages
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import requests

try:
    import yfinance as yf
except ImportError:
    raise SystemExit('pip install yfinance')

# ─── CONFIGURATION  ────────────────────────────────────────────────────────────
# !! UPDATE: set the confirmed Warsh nomination date !!
NOMINATION  = pd.Timestamp('2025-11-13')   # <─ SET ACTUAL DATE
TODAY       = pd.Timestamp('2026-05-03')

PRE_DAYS    = 120   # days before nomination to include in charts
ROLL_WIN    = 40    # rolling PCA window in trading days (~2 months)

# Fetch extra warmup so rolling PCA is warm by the start of the chart window
FETCH_START = NOMINATION - pd.Timedelta(days=PRE_DAYS + ROLL_WIN + 15)
FETCH_END   = TODAY

# ─── STYLE ─────────────────────────────────────────────────────────────────────
plt.rcParams.update({
    'figure.facecolor': '#0d1117', 'axes.facecolor':   '#161b22',
    'axes.edgecolor':   '#30363d', 'text.color':       '#e6edf3',
    'axes.labelcolor':  '#e6edf3', 'xtick.color':      '#8b949e',
    'ytick.color':      '#8b949e', 'grid.color':       '#21262d',
    'grid.linestyle':   '--',      'grid.alpha':       0.5,
    'axes.titlecolor':  '#e6edf3', 'legend.facecolor': '#161b22',
    'legend.edgecolor': '#30363d', 'font.size':        10,
    'figure.dpi':       120,
})

C_NOM   = '#d29922'   # nomination vline
C_TODAY = '#3fb950'   # today vline
PALETTE = ['#58a6ff', '#3fb950', '#d29922', '#f78166',
           '#bc8cff', '#79c0ff', '#56d364', '#e3b341']
PC_COL  = ['#58a6ff', '#3fb950', '#d29922']
PC_LBL  = ['PC1 (level)', 'PC2 (slope)', 'PC3 (curvature)']


# ─── DATA  ─────────────────────────────────────────────────────────────────────
def _fred(series_id: str) -> pd.Series:
    d2  = FETCH_END.strftime('%Y-%m-%d')
    url = f'https://fred.stlouisfed.org/graph/fredgraph.csv?id={series_id}&vintage_date={d2}'
    r   = requests.get(url, timeout=30)
    r.raise_for_status()
    df  = pd.read_csv(io.StringIO(r.text), index_col='observation_date')
    df.index = pd.to_datetime(df.index)
    s   = pd.to_numeric(df.iloc[:, 0], errors='coerce')
    return s.loc[FETCH_START:FETCH_END]


def _ecb(tenor_years: int) -> pd.Series:
    key = f'B.U2.EUR.4F.G_N_A.SV_C_YM.SR_{tenor_years}Y'
    url = (f'https://data-api.ecb.europa.eu/service/data/YC/{key}'
           f'?format=csvdata'
           f'&startPeriod={FETCH_START.strftime("%Y-%m-%d")}'
           f'&endPeriod={FETCH_END.strftime("%Y-%m-%d")}')
    r   = requests.get(url, timeout=30)
    r.raise_for_status()
    df  = pd.read_csv(io.StringIO(r.text))
    out = df[['TIME_PERIOD', 'OBS_VALUE']].copy()
    out['TIME_PERIOD'] = pd.to_datetime(out['TIME_PERIOD'])
    out = out.dropna().set_index('TIME_PERIOD')['OBS_VALUE']
    out.name = f'{tenor_years}Y'
    return out


def _yf(ticker: str) -> pd.Series | None:
    try:
        raw = yf.download(ticker,
                          start=FETCH_START.strftime('%Y-%m-%d'),
                          end=(FETCH_END + pd.Timedelta(days=1)).strftime('%Y-%m-%d'),
                          auto_adjust=True, progress=False)
        if raw.empty:
            return None
        s = raw['Close'].squeeze()
        s.index = pd.to_datetime(s.index).tz_localize(None)
        s = pd.to_numeric(s, errors='coerce').dropna()
        return s if len(s) > 5 else None
    except Exception:
        return None


def fetch_us() -> pd.DataFrame:
    print('Fetching US yields (FRED) ...')
    mapping = {'2Y': 'DGS2', '5Y': 'DGS5', '10Y': 'DGS10', '30Y': 'DGS30'}
    out = {}
    for tenor, sid in mapping.items():
        print(f'  {sid}', end='', flush=True)
        out[tenor] = _fred(sid)
        print(' ✓')
    df = pd.DataFrame(out).ffill()
    print(f'  {len(df.dropna())} trading days.')
    return df


def fetch_euro() -> pd.DataFrame:
    print('Fetching Euro AAA yields (ECB) ...')
    out = {}
    for t in [2, 5, 10, 30]:
        label = f'{t}Y'
        print(f'  ECB {label}', end='', flush=True)
        try:
            out[label] = _ecb(t)
            print(' ✓')
        except Exception as e:
            print(f' ✗  ({e})')
    if not out:
        return pd.DataFrame()
    df = pd.DataFrame(out).ffill()
    print(f'  {len(df.dropna())} trading days.')
    return df


def fetch_country_10y() -> pd.DataFrame:
    """
    Daily 10Y benchmark yields for Germany, France, Italy via FRED.
    FRED series names: IRDE10YR (Germany), IRFR10YR (France), IRIT10YR (Italy) — if they exist.
    Falls back gracefully if series are not found.
    """
    print('Fetching country 10Y benchmarks (FRED) ...')
    # FRED daily series for Euro area sovereign 10Y yields
    candidates = {
        'Germany 10Y': 'IRDE10T2YR',   # DE 10Y-2Y spread from FRED — fallback
        'France 10Y':  'IRFR10T2YR',
        'Italy 10Y':   'IRIT10T2YR',
    }
    # Try the correct FRED series IDs for sovereign 10Y yields
    # These are the daily long-term government bond yields from BIS/OECD via FRED
    series_map = {
        'Germany 10Y': 'IRLTLT01DEM156N',  # monthly from OECD
        'France 10Y':  'IRLTLT01FRM156N',  # monthly
        'Italy 10Y':   'IRLTLT01ITM156N',  # monthly
    }
    # Prefer daily series if available
    daily_series = {
        'Germany 10Y': 'BDTHS10YM',   # may not exist
        'France 10Y':  None,
        'Italy 10Y':   None,
    }
    out = {}
    for label, sid in series_map.items():
        print(f'  {label} ({sid})', end='', flush=True)
        try:
            s = _fred(sid)
            if s.dropna().empty:
                raise ValueError('empty')
            out[label] = s
            print(f' ✓  (freq: {pd.infer_freq(s.dropna().index) or "irregular"})')
        except Exception:
            print(' ✗')
    if not out:
        return pd.DataFrame()
    return pd.DataFrame(out).ffill()


def fetch_uk_etf() -> pd.Series | None:
    """iShares UK Gilts ETF — price proxy for Gilt direction."""
    print('Fetching UK Gilt ETF (IGLT.L via yfinance) ...', end='', flush=True)
    s = _yf('IGLT.L')
    if s is not None:
        print(f' ✓  ({len(s)} days)')
    else:
        print(' ✗')
    return s


def fetch_fx() -> pd.DataFrame:
    print('Fetching FX (yfinance) ...')
    tickers = {'EURUSD': 'EURUSD=X', 'GBPUSD': 'GBPUSD=X',
               'EURGBP': 'EURGBP=X', 'DXY':    'DX-Y.NYB'}
    out = {}
    for label, ticker in tickers.items():
        print(f'  {ticker}', end='', flush=True)
        s = _yf(ticker)
        if s is not None:
            out[label] = s
            print(' ✓')
        else:
            print(' ✗')
    return pd.DataFrame(out).ffill() if out else pd.DataFrame()


# ─── PCA HELPERS  ──────────────────────────────────────────────────────────────
TENOR_ORDER = ['2Y', '5Y', '10Y', '30Y']


def _sort(df: pd.DataFrame) -> pd.DataFrame:
    cols = [c for c in TENOR_ORDER if c in df.columns] + \
           [c for c in df.columns if c not in TENOR_ORDER]
    return df[cols]


def fit_pca(df: pd.DataFrame, fit_through: pd.Timestamp):
    """Fit on data up to fit_through. Returns (pca, scaler, cols, n)."""
    df_s   = _sort(df)
    cols   = df_s.columns.tolist()
    data   = df_s.loc[:fit_through].dropna()
    n      = min(3, len(cols))
    scaler = StandardScaler()
    pca    = PCA(n_components=n)
    pca.fit(scaler.fit_transform(data))
    # anchor: PC1 has positive loading at longest tenor
    for i in range(n):
        if pca.components_[i, -1] < 0:
            pca.components_[i] *= -1
    return pca, scaler, cols, n


def project(df: pd.DataFrame, pca, scaler, cols) -> pd.DataFrame:
    valid  = df[cols].dropna()
    scores = pca.transform(scaler.transform(valid))
    return pd.DataFrame(scores, index=valid.index,
                        columns=[f'PC{i+1}' for i in range(scores.shape[1])])


def rolling_pca(df: pd.DataFrame) -> pd.DataFrame:
    """40-day rolling PCA; sign-corrected; returns PC1/2/3 score at each day."""
    df_s = _sort(df).dropna()
    arr  = df_s.values
    idx  = df_s.index
    n    = min(3, arr.shape[1])
    records = []
    for i in range(ROLL_WIN - 1, len(arr)):
        block = arr[i - ROLL_WIN + 1 : i + 1]
        sc    = StandardScaler().fit_transform(block)
        p     = PCA(n_components=n).fit(sc)
        for j in range(n):
            if p.components_[j, -1] < 0:
                p.components_[j] *= -1
        score = list(p.transform(sc[-1:])[0]) + [np.nan] * (3 - n)
        records.append(score)
    return pd.DataFrame(records, index=idx[ROLL_WIN - 1:],
                        columns=['PC1', 'PC2', 'PC3'])


# ─── CHART HELPERS  ────────────────────────────────────────────────────────────
WIN_LO = NOMINATION - pd.Timedelta(days=PRE_DAYS)


def _vlines(ax):
    ax.axvline(NOMINATION, color=C_NOM,   lw=1.4, ls='--', alpha=0.9, label='Nomination')
    ax.axvline(TODAY,      color=C_TODAY, lw=1.2, ls=':',  alpha=0.8, label='Today')


def _xlim(ax):
    ax.set_xlim(WIN_LO - pd.Timedelta(days=3), TODAY + pd.Timedelta(days=5))


def _datefmt(ax):
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%b\n%Y'))
    ax.xaxis.set_major_locator(mdates.MonthLocator(interval=1))


# ─── PAGES  ────────────────────────────────────────────────────────────────────
def page_cover(pdf):
    fig, ax = plt.subplots(figsize=(11, 8.5))
    ax.axis('off')
    ax.text(0.5, 0.83, 'Fixed Income PCA Dynamics',
            ha='center', fontsize=30, fontweight='bold',
            color='#e6edf3', transform=ax.transAxes)
    ax.text(0.5, 0.74, 'The Warsh Nomination Window',
            ha='center', fontsize=20, color='#8b949e', transform=ax.transAxes)

    body = [
        f'Nomination:       {NOMINATION.strftime("%d %b %Y")}',
        f'Chart window:     {WIN_LO.strftime("%d %b %Y")}  →  {TODAY.strftime("%d %b %Y")}',
        f'Rolling PCA:      {ROLL_WIN}-day trailing window',
        '',
        'Markets covered',
        '  ·  US Treasuries    2Y · 5Y · 10Y · 30Y         FRED',
        '  ·  Euro AAA bonds   2Y · 5Y · 10Y · 30Y         ECB SDW',
        '  ·  UK Gilts         ETF price proxy (IGLT.L)     Yahoo Finance',
        '  ·  FX               EURUSD · GBPUSD · EURGBP · DXY',
        '',
        'PCA methodology',
        '  · Fit on pre-nomination window; projected forward onto fixed loadings',
        '  · Rolling 40-day refit also shown (dashed) for intra-window shifts',
        '  · Signs anchored: PC1 always positive at the longest available tenor',
        '  · Scores z-normalised to pre-nomination period in cross-market comparisons',
        '',
        '  PC1 = level   ·   PC2 = slope   ·   PC3 = curvature',
    ]
    ax.text(0.12, 0.65, '\n'.join(body), ha='left', va='top',
            fontsize=10.5, color='#e6edf3', transform=ax.transAxes,
            linespacing=1.75, fontfamily='monospace')

    pdf.savefig(fig, bbox_inches='tight')
    plt.close(fig)


def page_curve_snapshots(pdf, df_us: pd.DataFrame, df_euro: pd.DataFrame):
    """Side-by-side yield curve at −120d, nomination, today."""
    snap_dates = [
        (f'−{PRE_DAYS}d', WIN_LO,      '#58a6ff'),
        ('Nomination',     NOMINATION,  '#d29922'),
        ('Today',          TODAY,       '#3fb950'),
    ]
    markets = [
        ('US Treasuries', df_us),
        ('Euro AAA (ECB)', df_euro),
    ]
    n = sum(1 for _, df in markets if not df.empty)
    if n == 0:
        return

    fig, axes = plt.subplots(1, n, figsize=(7 * n, 6))
    if n == 1:
        axes = [axes]

    col = 0
    for name, df in markets:
        if df.empty:
            continue
        ax   = axes[col]
        df_s = _sort(df)
        cols = df_s.columns.tolist()
        xs   = range(len(cols))
        for label, snap_dt, color in snap_dates:
            row = df_s.loc[:snap_dt].dropna(how='all').iloc[-1]
            if row.isna().all():
                continue
            ax.plot(xs, row.values, color=color, marker='o', ms=5, lw=2.2, label=label)
            ax.fill_between(xs, row.values.min() - 0.2, row.values,
                            color=color, alpha=0.05)
        ax.set_xticks(list(xs))
        ax.set_xticklabels(cols)
        ax.set_title(name, fontsize=13)
        ax.set_ylabel('Yield (%)')
        ax.legend(fontsize=9)
        ax.grid(True)
        col += 1

    fig.suptitle('Yield curve snapshots: pre-nomination → today', fontsize=14, y=1.02)
    plt.tight_layout()
    pdf.savefig(fig, bbox_inches='tight')
    plt.close(fig)


def page_rates_pca(pdf, df: pd.DataFrame, market: str):
    """One page: PC loadings + PC1/PC2/PC3 time series (fixed + rolling)."""
    if df.empty or len(df.columns) < 2:
        print(f'    Skipping {market} PCA — not enough tenors.')
        return

    df_s        = _sort(df)
    pca, sc, cols, n = fit_pca(df_s, NOMINATION)
    evr         = pca.explained_variance_ratio_
    scores_fix  = project(df_s, pca, sc, cols)
    scores_roll = rolling_pca(df_s)

    fig, axes = plt.subplots(1, 4, figsize=(22, 5))

    # ── Panel 0: loadings ────────────────────────────────────────────────────
    ax = axes[0]
    xs = range(len(cols))
    for i in range(n):
        ax.plot(xs, pca.components_[i],
                color=PC_COL[i], marker='o', ms=5, lw=2.0,
                label=f'{PC_LBL[i]} ({evr[i]*100:.0f}%)')
    ax.axhline(0, color='#8b949e', lw=0.7, ls=':')
    ax.set_xticks(list(xs))
    ax.set_xticklabels(cols)
    ax.set_title('PC loadings\n(fit: pre-nomination window)')
    ax.set_ylabel('Loading')
    ax.legend(fontsize=8)
    ax.grid(True)

    # ── Panels 1–3: PC scores ────────────────────────────────────────────────
    for panel, pc in enumerate(['PC1', 'PC2', 'PC3'][:n], start=1):
        ax  = axes[panel]
        sf  = scores_fix.loc[WIN_LO:, pc] if pc in scores_fix else pd.Series()
        sr  = scores_roll.loc[WIN_LO:, pc] if pc in scores_roll else pd.Series()

        if not sf.empty:
            ax.plot(sf.index, sf.values, color=PC_COL[panel-1], lw=1.8,
                    label='Fixed basis')
        if not sr.empty:
            ax.plot(sr.index, sr.values, color=PC_COL[panel-1], lw=1.2,
                    ls='--', alpha=0.55, label=f'{ROLL_WIN}d rolling')

        ax.axhline(0, color='#8b949e', lw=0.7, ls=':')
        _vlines(ax)
        _xlim(ax)
        _datefmt(ax)
        ax.set_title(PC_LBL[panel-1])
        ax.set_ylabel('Score (σ)')
        ax.legend(fontsize=7)
        ax.grid(True)

    fig.suptitle(f'{market} — PCA Dynamics  ·  {NOMINATION.strftime("%b %Y")} nomination window',
                 fontsize=14, y=1.02)
    plt.tight_layout()
    pdf.savefig(fig, bbox_inches='tight')
    plt.close(fig)


def page_uk_context(pdf, uk_etf: pd.Series | None, df_us: pd.DataFrame, df_euro: pd.DataFrame):
    """UK Gilt ETF + 10Y yields from US and Euro for context."""
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    # ── UK ETF indexed ────────────────────────────────────────────────────────
    ax = axes[0]
    if uk_etf is not None:
        wd   = uk_etf.loc[WIN_LO:]
        base = uk_etf.loc[:NOMINATION].iloc[-1]
        idx  = (wd / base - 1) * 100
        ax.plot(idx.index, idx.values, color='#3fb950', lw=1.8, label='IGLT.L')
        ax.fill_between(idx.index, 0, idx.values, color='#3fb950', alpha=0.07)
        ax.axhline(0, color='#8b949e', lw=0.7, ls=':')
        _vlines(ax)
        _xlim(ax)
        _datefmt(ax)
        ax.set_title('UK Gilt ETF (IGLT.L)\nprice % change from nomination')
        ax.set_ylabel('% change')
        ax.legend(fontsize=8)
    else:
        ax.text(0.5, 0.5, 'UK Gilt data\nunavailable\n(no free daily yield curve API)',
                ha='center', va='center', transform=ax.transAxes, fontsize=10,
                color='#8b949e')
        ax.set_title('UK Gilts')
    ax.grid(True)

    # ── US 10Y vs Euro 10Y absolute level ────────────────────────────────────
    ax = axes[1]
    for name, df, color in [
        ('US 10Y', df_us, '#58a6ff'),
        ('Euro 10Y', df_euro, '#d29922'),
    ]:
        if not df.empty and '10Y' in df.columns:
            s = df['10Y'].loc[WIN_LO:]
            ax.plot(s.index, s.values, color=color, lw=1.8, label=name)
    _vlines(ax)
    _xlim(ax)
    _datefmt(ax)
    ax.set_title('US vs Euro 10Y yield levels')
    ax.set_ylabel('Yield (%)')
    ax.legend(fontsize=8)
    ax.grid(True)

    # ── US–Euro 10Y spread ────────────────────────────────────────────────────
    ax = axes[2]
    if not df_us.empty and not df_euro.empty and '10Y' in df_us.columns and '10Y' in df_euro.columns:
        spread = (df_us['10Y'] - df_euro['10Y']).loc[WIN_LO:] * 100
        spread = spread.dropna()
        ax.plot(spread.index, spread.values, color='#bc8cff', lw=1.8, label='US−Euro 10Y (bps)')
        ax.fill_between(spread.index, 0, spread.values, color='#bc8cff', alpha=0.07)
        ax.axhline(0, color='#8b949e', lw=0.7, ls=':')
        _vlines(ax)
        _xlim(ax)
        _datefmt(ax)
        ax.set_title('US vs Euro 10Y spread (bps)')
        ax.set_ylabel('Spread (bps)')
        ax.legend(fontsize=8)
    ax.grid(True)

    fig.suptitle('UK and Cross-Atlantic Rate Context', fontsize=14, y=1.02)
    plt.tight_layout()
    pdf.savefig(fig, bbox_inches='tight')
    plt.close(fig)


def page_fx(pdf, df_fx: pd.DataFrame):
    if df_fx.empty:
        return

    pairs = [c for c in ['EURUSD', 'GBPUSD', 'EURGBP', 'DXY'] if c in df_fx]
    rvol  = df_fx[pairs].pct_change().rolling(21).std() * 100

    fig, axes = plt.subplots(2, 2, figsize=(16, 10))
    ax_idx, ax_vol, ax_level, ax_dxy = axes.flatten()

    # indexed level
    for i, pair in enumerate(pairs):
        wd   = df_fx[pair].loc[WIN_LO:]
        base = df_fx[pair].loc[:NOMINATION].iloc[-1]
        idx  = (wd / base - 1) * 100
        ax_idx.plot(idx.index, idx.values, color=PALETTE[i], lw=1.8, label=pair)
    ax_idx.axhline(0, color='#8b949e', lw=0.7, ls=':')
    _vlines(ax_idx); _xlim(ax_idx); _datefmt(ax_idx)
    ax_idx.set_title('FX performance (% from nomination)')
    ax_idx.set_ylabel('% change')
    ax_idx.legend(fontsize=8)
    ax_idx.grid(True)

    # realised vol
    for i, pair in enumerate(pairs):
        rv = rvol[pair].loc[WIN_LO:]
        ax_vol.plot(rv.index, rv.values, color=PALETTE[i], lw=1.6, label=pair)
    _vlines(ax_vol); _xlim(ax_vol); _datefmt(ax_vol)
    ax_vol.set_title('21d realised FX vol (%/day)')
    ax_vol.set_ylabel('%/day')
    ax_vol.legend(fontsize=8)
    ax_vol.grid(True)

    # EURUSD + GBPUSD levels
    for pair, color in [('EURUSD', '#58a6ff'), ('GBPUSD', '#3fb950')]:
        if pair in df_fx:
            s = df_fx[pair].loc[WIN_LO:]
            ax_level.plot(s.index, s.values, color=color, lw=1.8, label=pair)
    _vlines(ax_level); _xlim(ax_level); _datefmt(ax_level)
    ax_level.set_title('EURUSD and GBPUSD levels')
    ax_level.set_ylabel('Rate')
    ax_level.legend(fontsize=8)
    ax_level.grid(True)

    # DXY
    if 'DXY' in df_fx:
        dxy = df_fx['DXY'].loc[WIN_LO:]
        ax_dxy.plot(dxy.index, dxy.values, color='#d29922', lw=1.8)
        ax_dxy.fill_between(dxy.index, dxy.min(), dxy.values,
                            color='#d29922', alpha=0.08)
    _vlines(ax_dxy); _xlim(ax_dxy); _datefmt(ax_dxy)
    ax_dxy.set_title('DXY (US Dollar Index)')
    ax_dxy.set_ylabel('Index')
    ax_dxy.legend(fontsize=8)
    ax_dxy.grid(True)

    fig.suptitle('FX Dynamics — Warsh Nomination Window', fontsize=14, y=1.01)
    plt.tight_layout()
    pdf.savefig(fig, bbox_inches='tight')
    plt.close(fig)


def page_cross_market(pdf, df_us: pd.DataFrame, df_euro: pd.DataFrame):
    """
    US vs Euro: PC1/PC2/PC3 scores on the same axes.
    Scores are z-normalised to the pre-nomination window for comparability.
    """
    datasets = [
        ('US Treasuries', df_us,   '#58a6ff'),
        ('Euro AAA',      df_euro, '#d29922'),
    ]

    fig, axes = plt.subplots(1, 3, figsize=(18, 6))

    for pc_idx, (ax, title) in enumerate(
            zip(axes, ['PC1 — Level', 'PC2 — Slope', 'PC3 — Curvature'])):

        for name, df, color in datasets:
            if df.empty or len(df.columns) < 2:
                continue
            pca, scaler, cols, n = fit_pca(_sort(df), NOMINATION)
            if pc_idx >= n:
                continue
            scores = project(df, pca, scaler, cols)
            pc     = f'PC{pc_idx + 1}'
            if pc not in scores:
                continue
            pre    = scores.loc[:NOMINATION, pc]
            znorm  = (scores[pc] - pre.mean()) / (pre.std() + 1e-8)
            s      = znorm.loc[WIN_LO:]
            ax.plot(s.index, s.values, color=color, lw=1.8, label=name)

        ax.axhline(0, color='#8b949e', lw=0.7, ls=':')
        _vlines(ax); _xlim(ax); _datefmt(ax)
        ax.set_title(title)
        ax.set_ylabel('Score (σ, pre-nom normalised)')
        ax.legend(fontsize=8)
        ax.grid(True)

    fig.suptitle(
        'Cross-Market PCA Comparison — US Treasuries vs Euro AAA\n'
        'Z-normalised to pre-nomination window  ·  divergence = regimes decoupling',
        fontsize=13, y=1.02
    )
    plt.tight_layout()
    pdf.savefig(fig, bbox_inches='tight')
    plt.close(fig)


# ─── MAIN  ─────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--out', default='warsh_pca.pdf')
    args = parser.parse_args()

    print(f'Nomination: {NOMINATION.strftime("%d %b %Y")}')
    print(f'Window:     {WIN_LO.strftime("%d %b %Y")} → {TODAY.strftime("%d %b %Y")}\n')

    df_us   = fetch_us()
    df_euro = fetch_euro()
    uk_etf  = fetch_uk_etf()
    df_fx   = fetch_fx()

    print(f'\nWriting {args.out} ...')
    with PdfPages(args.out) as pdf:
        print('  [1] Cover')
        page_cover(pdf)

        print('  [2] Yield curve snapshots')
        page_curve_snapshots(pdf, df_us, df_euro)

        print('  [3] US Treasuries PCA')
        page_rates_pca(pdf, df_us, 'US Treasuries')

        print('  [4] Euro AAA PCA')
        page_rates_pca(pdf, df_euro, 'Euro AAA Bonds (ECB)')

        print('  [5] UK + cross-Atlantic context')
        page_uk_context(pdf, uk_etf, df_us, df_euro)

        print('  [6] FX')
        page_fx(pdf, df_fx)

        print('  [7] Cross-market PCA comparison')
        page_cross_market(pdf, df_us, df_euro)

        meta = pdf.infodict()
        meta['Title']   = 'Fixed Income PCA Dynamics — Warsh Nomination Window'
        meta['Subject'] = 'US Treasuries, Euro AAA bonds, FX — rolling PCA'

    print(f'\nDone → {args.out}')


if __name__ == '__main__':
    main()
