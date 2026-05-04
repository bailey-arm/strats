# Data sourced from Bank of England MPC minutes — verify member-level votes for precision.
"""
boe_mpc_votes.py — BoE MPC Voting History (multi-page PDF)

Usage:
    python scripts/boe_mpc_votes.py [--out boe_mpc_votes.pdf]
"""

import argparse
import warnings
from collections import defaultdict
from datetime import datetime

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.ticker as mticker
import numpy as np
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.colors import LinearSegmentedColormap

warnings.filterwarnings("ignore")

# ---------------------------------------------------------------------------
# Style
# ---------------------------------------------------------------------------

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

VOTE_COLORS = {'hike': '#f78166', 'hold': '#8b949e', 'cut': '#58a6ff'}

# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------

# Each meeting: date, decision ('hike'/'hold'/'cut'), new_rate,
# votes_for_decision (int), total_votes (9),
# and member_votes dict mapping member name to their vote ('hike'/'hold'/'cut')

MPC_DATA = [
    # 2023
    {'date': '2023-06-22', 'decision': 'hike', 'rate': 5.00,
     'for': 7, 'against': 2,
     'members': {'Bailey': 'hike', 'Broadbent': 'hike', 'Ramsden': 'hike', 'Pill': 'hike',
                 'Greene': 'hike', 'Haskel': 'hike', 'Mann': 'hike',
                 'Dhingra': 'hold', 'Tenreyro': 'hold'}},
    {'date': '2023-08-03', 'decision': 'hike', 'rate': 5.25,
     'for': 6, 'against': 3,
     'members': {'Bailey': 'hike', 'Broadbent': 'hike', 'Ramsden': 'hike', 'Pill': 'hike',
                 'Greene': 'hike', 'Haskel': 'hike_larger',
                 'Mann': 'hike_larger', 'Dhingra': 'hold', 'Tenreyro': 'hold'}},
    {'date': '2023-09-21', 'decision': 'hold', 'rate': 5.25,
     'for': 5, 'against': 4,
     'members': {'Bailey': 'hold', 'Broadbent': 'hold', 'Ramsden': 'hold', 'Pill': 'hold',
                 'Greene': 'hold', 'Haskel': 'hike', 'Mann': 'hike',
                 'Dhingra': 'cut', 'Tenreyro': 'cut'}},
    {'date': '2023-11-02', 'decision': 'hold', 'rate': 5.25,
     'for': 6, 'against': 3,
     'members': {'Bailey': 'hold', 'Broadbent': 'hold', 'Ramsden': 'hold', 'Pill': 'hold',
                 'Greene': 'hold', 'Haskel': 'hike', 'Mann': 'hike',
                 'Dhingra': 'cut', 'Breeden': 'hold'}},
    {'date': '2023-12-14', 'decision': 'hold', 'rate': 5.25,
     'for': 6, 'against': 3,
     'members': {'Bailey': 'hold', 'Broadbent': 'hold', 'Ramsden': 'hold', 'Pill': 'hold',
                 'Greene': 'hike', 'Haskel': 'hike', 'Mann': 'hike',
                 'Dhingra': 'cut', 'Breeden': 'hold'}},
    # 2024
    {'date': '2024-02-01', 'decision': 'hold', 'rate': 5.25,
     'for': 6, 'against': 3,
     'members': {'Bailey': 'hold', 'Broadbent': 'hold', 'Ramsden': 'hold', 'Pill': 'hold',
                 'Greene': 'hike', 'Haskel': 'hike', 'Mann': 'hike',
                 'Dhingra': 'cut', 'Breeden': 'hold'}},
    {'date': '2024-03-21', 'decision': 'hold', 'rate': 5.25,
     'for': 8, 'against': 1,
     'members': {'Bailey': 'hold', 'Broadbent': 'hold', 'Ramsden': 'hold', 'Pill': 'hold',
                 'Greene': 'hold', 'Haskel': 'hold', 'Mann': 'hold',
                 'Dhingra': 'cut', 'Breeden': 'hold'}},
    {'date': '2024-05-09', 'decision': 'hold', 'rate': 5.25,
     'for': 7, 'against': 2,
     'members': {'Bailey': 'hold', 'Broadbent': 'hold', 'Ramsden': 'cut', 'Pill': 'hold',
                 'Greene': 'hold', 'Haskel': 'hold', 'Mann': 'hold',
                 'Dhingra': 'cut', 'Breeden': 'hold'}},
    {'date': '2024-06-20', 'decision': 'hold', 'rate': 5.25,
     'for': 7, 'against': 2,
     'members': {'Bailey': 'hold', 'Broadbent': 'hold', 'Ramsden': 'cut', 'Pill': 'hold',
                 'Greene': 'hold', 'Haskel': 'hold', 'Mann': 'hold',
                 'Dhingra': 'cut', 'Breeden': 'hold'}},
    {'date': '2024-08-01', 'decision': 'cut', 'rate': 5.00,
     'for': 5, 'against': 4,
     'members': {'Bailey': 'cut', 'Lombardelli': 'cut', 'Ramsden': 'cut', 'Pill': 'hold',
                 'Greene': 'hold', 'Haskel': 'hold', 'Mann': 'hold',
                 'Dhingra': 'cut', 'Breeden': 'cut'}},
    {'date': '2024-09-19', 'decision': 'hold', 'rate': 5.00,
     'for': 8, 'against': 1,
     'members': {'Bailey': 'hold', 'Lombardelli': 'hold', 'Ramsden': 'hold', 'Pill': 'hold',
                 'Greene': 'hold', 'Taylor': 'hold', 'Mann': 'hold',
                 'Dhingra': 'cut', 'Breeden': 'hold'}},
    {'date': '2024-11-07', 'decision': 'cut', 'rate': 4.75,
     'for': 8, 'against': 1,
     'members': {'Bailey': 'cut', 'Lombardelli': 'cut', 'Ramsden': 'cut', 'Pill': 'cut',
                 'Greene': 'cut', 'Taylor': 'cut', 'Mann': 'hold',
                 'Dhingra': 'cut', 'Breeden': 'cut'}},
    {'date': '2024-12-19', 'decision': 'hold', 'rate': 4.75,
     'for': 6, 'against': 3,
     'members': {'Bailey': 'hold', 'Lombardelli': 'hold', 'Ramsden': 'cut', 'Pill': 'hold',
                 'Greene': 'cut', 'Taylor': 'hold', 'Mann': 'hold',
                 'Dhingra': 'cut', 'Breeden': 'hold'}},
    # 2025
    {'date': '2025-02-06', 'decision': 'cut', 'rate': 4.50,
     'for': 7, 'against': 2,
     'members': {'Bailey': 'cut', 'Lombardelli': 'cut', 'Ramsden': 'cut', 'Pill': 'cut',
                 'Greene': 'cut', 'Taylor': 'cut', 'Mann': 'hold',
                 'Dhingra': 'cut_larger', 'Breeden': 'cut'}},  # Dhingra voted 50bps
    {'date': '2025-03-20', 'decision': 'hold', 'rate': 4.50,
     'for': 8, 'against': 1,
     'members': {'Bailey': 'hold', 'Lombardelli': 'hold', 'Ramsden': 'hold', 'Pill': 'hold',
                 'Greene': 'hold', 'Taylor': 'hold', 'Mann': 'hold',
                 'Dhingra': 'cut', 'Breeden': 'hold'}},
    {'date': '2025-05-08', 'decision': 'cut', 'rate': 4.25,
     'for': 5, 'against': 4,
     'members': {'Bailey': 'cut', 'Lombardelli': 'cut', 'Ramsden': 'cut', 'Pill': 'cut',
                 'Greene': 'hold', 'Taylor': 'hold', 'Mann': 'hold',
                 'Dhingra': 'cut', 'Breeden': 'cut'}},  # approximate
]

MEMBER_META = {
    'Bailey':      {'role': 'Governor',        'type': 'internal', 'tenure_end': None},
    'Broadbent':   {'role': 'DG Monetary',     'type': 'internal', 'tenure_end': '2024-06'},
    'Ramsden':     {'role': 'DG Markets',      'type': 'internal', 'tenure_end': '2025-06'},
    'Pill':        {'role': 'Chief Economist', 'type': 'internal', 'tenure_end': None},
    'Breeden':     {'role': 'DG Fin Stab',     'type': 'internal', 'tenure_end': None},
    'Lombardelli': {'role': 'DG Monetary',     'type': 'internal', 'tenure_end': None},
    'Tenreyro':    {'role': 'External',        'type': 'external', 'tenure_end': '2023-07'},
    'Haskel':      {'role': 'External',        'type': 'external', 'tenure_end': '2024-08'},
    'Greene':      {'role': 'External',        'type': 'external', 'tenure_end': None},
    'Mann':        {'role': 'External',        'type': 'external', 'tenure_end': None},
    'Dhingra':     {'role': 'External',        'type': 'external', 'tenure_end': None},
    'Taylor':      {'role': 'External',        'type': 'external', 'tenure_end': None},
}

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def normalize_vote(v: str) -> str:
    """Map hike_larger / cut_larger to canonical hike / cut."""
    if v == 'hike_larger':
        return 'hike'
    if v == 'cut_larger':
        return 'cut'
    return v


def is_larger_vote(v: str) -> bool:
    return v in ('hike_larger', 'cut_larger')


def parse_meetings():
    """Return list of enriched meeting dicts with parsed dates and normalised votes."""
    meetings = []
    for m in MPC_DATA:
        d = m.copy()
        d['dt'] = datetime.strptime(m['date'], '%Y-%m-%d')
        d['norm_members'] = {k: normalize_vote(v) for k, v in m['members'].items()}
        d['larger_voters'] = [k for k, v in m['members'].items() if is_larger_vote(v)]
        meetings.append(d)
    return meetings


def hawk_score(member: str, meetings: list) -> float:
    """Fraction of attended meetings where member voted hike, or voted hold when decision was cut."""
    hawkish, total = 0, 0
    for m in meetings:
        if member not in m['norm_members']:
            continue
        total += 1
        v = m['norm_members'][member]
        if v == 'hike':
            hawkish += 1
        elif v == 'hold' and m['decision'] == 'cut':
            hawkish += 1
    return hawkish / total if total else 0.0


def vote_profile(member: str, meetings: list) -> dict:
    counts = defaultdict(int)
    for m in meetings:
        if member not in m['norm_members']:
            continue
        counts[m['norm_members'][member]] += 1
    total = sum(counts.values())
    return {k: counts[k] / total * 100 for k in ('hike', 'hold', 'cut')}, total


# ---------------------------------------------------------------------------
# Page 1: Cover
# ---------------------------------------------------------------------------

def page_cover(pdf: PdfPages, meetings: list):
    fig = plt.figure(figsize=(11, 8.5))
    fig.patch.set_facecolor('#0d1117')
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_facecolor('#0d1117')
    ax.axis('off')

    latest = meetings[-1]
    prev = meetings[-2]

    # Title block
    ax.text(0.5, 0.88, 'Bank of England', ha='center', va='center',
            fontsize=14, color='#8b949e', transform=ax.transAxes)
    ax.text(0.5, 0.78, 'MPC Vote Analysis', ha='center', va='center',
            fontsize=36, fontweight='bold', color='#e6edf3', transform=ax.transAxes)
    ax.text(0.5, 0.70, '2023 – 2025  |  June 2023 Hiking Cycle to May 2025',
            ha='center', va='center', fontsize=13, color='#8b949e', transform=ax.transAxes)

    # Divider
    ax.axhline(0.64, color='#30363d', linewidth=1, xmin=0.1, xmax=0.9)

    # Stats row
    stats = [
        ('Current Bank Rate', f"{latest['rate']:.2f}%"),
        ('Latest Vote', f"{latest['for']}–{latest['against']}"),
        ('Latest Decision', latest['decision'].upper()),
        ('Total Meetings', str(len(meetings))),
    ]
    for i, (label, val) in enumerate(stats):
        x = 0.15 + i * 0.22
        ax.text(x, 0.56, val, ha='center', va='center', fontsize=22,
                fontweight='bold', color=VOTE_COLORS.get(latest['decision'], '#e6edf3'),
                transform=ax.transAxes)
        ax.text(x, 0.50, label, ha='center', va='center', fontsize=9,
                color='#8b949e', transform=ax.transAxes)

    ax.axhline(0.44, color='#30363d', linewidth=1, xmin=0.1, xmax=0.9)

    # MPC structure description
    body = (
        "The Monetary Policy Committee (MPC) has nine members: the Governor, three Deputy Governors\n"
        "(Monetary Policy, Financial Stability, Markets & Banking), the Chief Economist, and four\n"
        "external members appointed by the Chancellor. The committee meets eight times per year.\n\n"
        "Decisions require a simple majority; in a tie the Governor has the casting vote. Each member's\n"
        "individual vote is published two weeks after the meeting alongside the minutes. Votes may differ\n"
        "in magnitude — e.g. a member may vote for a 50 bps move while the majority votes 25 bps.\n\n"
        f"Most recent meeting:  {latest['date']}  —  Decision: {latest['decision'].upper()} "
        f"to {latest['rate']:.2f}%  ({latest['for']}–{latest['against']})\n"
        f"Previous meeting:     {prev['date']}  —  Decision: {prev['decision'].upper()} "
        f"to {prev['rate']:.2f}%  ({prev['for']}–{prev['against']})"
    )
    ax.text(0.5, 0.28, body, ha='center', va='center', fontsize=10.5,
            color='#c9d1d9', transform=ax.transAxes,
            linespacing=1.7, family='monospace')

    ax.axhline(0.10, color='#30363d', linewidth=1, xmin=0.1, xmax=0.9)
    ax.text(0.5, 0.06, 'Data sourced from Bank of England MPC minutes — verify member-level votes for precision.',
            ha='center', va='center', fontsize=8, color='#484f58', transform=ax.transAxes)

    pdf.savefig(fig, bbox_inches='tight')
    plt.close(fig)


# ---------------------------------------------------------------------------
# Page 2: Bank Rate timeline with vote margin annotations
# ---------------------------------------------------------------------------

def page_rate_timeline(pdf: PdfPages, meetings: list):
    fig, ax = plt.subplots(figsize=(14, 6))
    fig.patch.set_facecolor('#0d1117')
    ax.set_facecolor('#161b22')

    dates = [m['dt'] for m in meetings]
    rates = [m['rate'] for m in meetings]

    # Step function — extend backwards from first meeting
    step_dates = []
    step_rates = []
    for i, m in enumerate(meetings):
        if i == 0:
            step_dates.append(m['dt'])
        else:
            step_dates.append(m['dt'])
        step_rates.append(m['rate'])

    ax.step(step_dates, step_rates, where='post', color='#58a6ff',
            linewidth=2.5, zorder=2, label='Bank Rate')
    ax.fill_between(step_dates, step_rates, step=('post'),
                    color='#58a6ff', alpha=0.07, zorder=1)

    # Annotate each meeting point
    for m in meetings:
        margin = m['for']
        total = m['for'] + m['against']
        label = f"{m['for']}–{m['against']}"

        # Colour by how close the vote was
        if m['for'] <= 5:          # 5-4
            mc = '#f78166'
        elif m['for'] <= 7:        # 6-3 or 7-2
            mc = '#e3b341'
        else:                       # 8-1 or 9-0
            mc = '#56d364'

        ax.scatter(m['dt'], m['rate'], color=mc, s=80, zorder=5, edgecolors='#0d1117', linewidths=0.8)
        ax.annotate(label, xy=(m['dt'], m['rate']),
                    xytext=(0, 12), textcoords='offset points',
                    ha='center', fontsize=7.5, color=mc,
                    arrowprops=None)

    ax.set_title('Bank Rate Path with Vote Margins (Jun 2023 – May 2025)',
                 fontsize=13, pad=12)
    ax.set_ylabel('Bank Rate (%)', fontsize=11)
    ax.set_xlabel('')
    ax.yaxis.set_major_formatter(mticker.FormatStrFormatter('%.2f'))
    ax.grid(True, axis='y')
    ax.set_ylim(3.8, 5.7)

    # Legend for vote margin colours
    patches = [
        mpatches.Patch(color='#f78166', label='Tight vote (≤ 5–4)'),
        mpatches.Patch(color='#e3b341', label='Divided (6–3 or 7–2)'),
        mpatches.Patch(color='#56d364', label='Consensus (≥ 8–1)'),
    ]
    ax.legend(handles=patches, loc='lower left', fontsize=9)

    fig.autofmt_xdate()
    fig.tight_layout()
    pdf.savefig(fig, bbox_inches='tight')
    plt.close(fig)


# ---------------------------------------------------------------------------
# Page 3: Vote split stacked bar chart
# ---------------------------------------------------------------------------

def page_vote_stacks(pdf: PdfPages, meetings: list):
    fig, ax = plt.subplots(figsize=(14, 6))
    fig.patch.set_facecolor('#0d1117')
    ax.set_facecolor('#161b22')

    x = np.arange(len(meetings))
    labels = [m['date'][2:7] for m in meetings]  # YY-MM

    cut_counts, hold_counts, hike_counts = [], [], []
    for m in meetings:
        votes = list(m['norm_members'].values())
        cut_counts.append(votes.count('cut'))
        hold_counts.append(votes.count('hold'))
        hike_counts.append(votes.count('hike'))

    cut_arr  = np.array(cut_counts)
    hold_arr = np.array(hold_counts)
    hike_arr = np.array(hike_counts)

    bar_w = 0.65
    b1 = ax.bar(x, cut_arr,  bar_w, label='Cut',  color=VOTE_COLORS['cut'],  alpha=0.85)
    b2 = ax.bar(x, hold_arr, bar_w, bottom=cut_arr, label='Hold', color=VOTE_COLORS['hold'], alpha=0.85)
    b3 = ax.bar(x, hike_arr, bar_w, bottom=cut_arr + hold_arr, label='Hike', color=VOTE_COLORS['hike'], alpha=0.85)

    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=45, ha='right', fontsize=8)
    ax.set_yticks(range(10))
    ax.set_ylabel('Votes', fontsize=11)
    ax.set_title('MPC Vote Composition per Meeting — Drift from Hawkish to Dovish',
                 fontsize=13, pad=12)
    ax.axhline(4.5, color='#30363d', linewidth=1, linestyle=':', label='Majority threshold')
    ax.legend(loc='upper right', fontsize=9)
    ax.set_ylim(0, 9.8)
    ax.grid(True, axis='y')

    # Annotate decision on each bar
    for i, m in enumerate(meetings):
        sym = {'hike': '▲', 'hold': '●', 'cut': '▼'}[m['decision']]
        col = VOTE_COLORS[m['decision']]
        ax.text(i, 9.4, sym, ha='center', va='center', fontsize=9, color=col)

    fig.tight_layout()
    pdf.savefig(fig, bbox_inches='tight')
    plt.close(fig)


# ---------------------------------------------------------------------------
# Page 4: Individual member heatmap
# ---------------------------------------------------------------------------

def page_heatmap(pdf: PdfPages, meetings: list):
    all_members = list(MEMBER_META.keys())
    # Sort by hawk score descending
    scores = {m: hawk_score(m, meetings) for m in all_members}
    all_members.sort(key=lambda m: scores[m], reverse=True)

    n_members = len(all_members)
    n_meetings = len(meetings)

    # Build matrix: 0 = not on MPC, 1 = cut, 2 = hold, 3 = hike
    mat = np.zeros((n_members, n_meetings))
    for j, m in enumerate(meetings):
        for i, member in enumerate(all_members):
            if member in m['norm_members']:
                v = m['norm_members'][member]
                mat[i, j] = {'cut': 1, 'hold': 2, 'hike': 3}[v]

    fig, ax = plt.subplots(figsize=(16, 7))
    fig.patch.set_facecolor('#0d1117')
    ax.set_facecolor('#161b22')

    # Custom colormap: white (absent) → blue (cut) → grey (hold) → red (hike)
    cmap = LinearSegmentedColormap.from_list(
        'mpc', ['#0d1117', '#58a6ff', '#8b949e', '#f78166'], N=4)

    im = ax.imshow(mat, aspect='auto', cmap=cmap, vmin=0, vmax=3,
                   interpolation='nearest')

    # Gridlines
    ax.set_xticks(np.arange(-0.5, n_meetings, 1), minor=True)
    ax.set_yticks(np.arange(-0.5, n_members, 1), minor=True)
    ax.grid(which='minor', color='#0d1117', linewidth=1.2)
    ax.tick_params(which='minor', bottom=False, left=False)

    # Axis labels
    ax.set_xticks(range(n_meetings))
    ax.set_xticklabels([m['date'][2:7] for m in meetings],
                       rotation=45, ha='right', fontsize=8)
    ax.set_yticks(range(n_members))
    ax.set_yticklabels(all_members, fontsize=9)

    # Annotate larger-move votes
    for j, m in enumerate(meetings):
        for voter in m['larger_voters']:
            if voter in all_members:
                i = all_members.index(voter)
                raw = m['members'][voter]
                sym = '+' if 'hike' in raw else '−'
                ax.text(j, i, sym, ha='center', va='center',
                        fontsize=9, color='#e6edf3', fontweight='bold')

    # Hawk score column on right
    ax2 = ax.twinx()
    ax2.set_ylim(ax.get_ylim())
    ax2.set_yticks(range(n_members))
    ax2.set_yticklabels(
        [f"{scores[m]*100:.0f}%" for m in all_members], fontsize=8, color='#e3b341')
    ax2.set_ylabel('Hawk Score', fontsize=9, color='#e3b341')
    ax2.tick_params(axis='y', colors='#e3b341')

    ax.set_title('MPC Member Vote Heatmap  |  Blue = Cut  ·  Grey = Hold  ·  Red = Hike  ·  White = Absent\n'
                 '(+/− = voted for larger move)  |  Hawk score = % meetings voted hike or against cut',
                 fontsize=10, pad=10)

    # Legend patches
    patches = [
        mpatches.Patch(color='#58a6ff', label='Cut'),
        mpatches.Patch(color='#8b949e', label='Hold'),
        mpatches.Patch(color='#f78166', label='Hike'),
        mpatches.Patch(color='#0d1117', label='Not on MPC'),
    ]
    ax.legend(handles=patches, loc='upper left', fontsize=8,
              bbox_to_anchor=(0, -0.18), ncol=4)

    fig.tight_layout()
    pdf.savefig(fig, bbox_inches='tight')
    plt.close(fig)


# ---------------------------------------------------------------------------
# Page 5: Member voting profiles (horizontal bar)
# ---------------------------------------------------------------------------

def page_profiles(pdf: PdfPages, meetings: list):
    all_members = list(MEMBER_META.keys())
    scores = {m: hawk_score(m, meetings) for m in all_members}
    all_members.sort(key=lambda m: scores[m], reverse=True)

    profiles = []
    for member in all_members:
        pct, total = vote_profile(member, meetings)
        profiles.append((member, pct, total))

    fig, ax = plt.subplots(figsize=(11, 7))
    fig.patch.set_facecolor('#0d1117')
    ax.set_facecolor('#161b22')

    y = np.arange(len(profiles))
    bar_h = 0.6

    for i, (member, pct, total) in enumerate(profiles):
        left = 0
        for vote_type in ('hike', 'hold', 'cut'):
            val = pct.get(vote_type, 0)
            if val > 0:
                ax.barh(i, val, bar_h, left=left,
                        color=VOTE_COLORS[vote_type], alpha=0.85)
                if val > 8:
                    ax.text(left + val / 2, i, f"{val:.0f}%",
                            ha='center', va='center', fontsize=8,
                            color='#0d1117', fontweight='bold')
            left += val

    ax.set_yticks(y)
    yticklabels = []
    for member, pct, total in profiles:
        meta = MEMBER_META.get(member, {})
        role = meta.get('role', '')
        mtype = 'INT' if meta.get('type') == 'internal' else 'EXT'
        hs = scores[member]
        yticklabels.append(f"{member}  [{mtype}] {role}  HS:{hs*100:.0f}%")
    ax.set_yticklabels(yticklabels, fontsize=8.5)

    ax.set_xlabel('% of attended meetings', fontsize=10)
    ax.set_xlim(0, 100)
    ax.set_title('MPC Member Voting Profiles — Sorted by Hawk Score (most hawkish at top)',
                 fontsize=12, pad=12)
    ax.axvline(50, color='#30363d', linewidth=1, linestyle=':')
    ax.grid(True, axis='x')

    patches = [mpatches.Patch(color=VOTE_COLORS[v], label=v.capitalize()) for v in ('hike', 'hold', 'cut')]
    ax.legend(handles=patches, loc='lower right', fontsize=9)

    fig.tight_layout()
    pdf.savefig(fig, bbox_inches='tight')
    plt.close(fig)


# ---------------------------------------------------------------------------
# Page 6: Dissent analysis
# ---------------------------------------------------------------------------

def page_dissent(pdf: PdfPages, meetings: list):
    fig = plt.figure(figsize=(14, 8))
    fig.patch.set_facecolor('#0d1117')
    gs = fig.add_gridspec(2, 2, hspace=0.45, wspace=0.35,
                          left=0.07, right=0.97, top=0.90, bottom=0.08)

    ax_main   = fig.add_subplot(gs[0, :])   # top: dissent count over time
    ax_pie1   = fig.add_subplot(gs[1, 0])   # bottom left: what dissenters wanted (all)
    ax_pie2   = fig.add_subplot(gs[1, 1])   # bottom right: recent (2024–2025)

    # --- Main: dissent count over time ---
    ax_main.set_facecolor('#161b22')
    dates = [m['dt'] for m in meetings]

    # Count dissenters and categorise their direction
    n_dissent, dissent_hawkish, dissent_dovish = [], [], []
    for m in meetings:
        decision = m['decision']
        dissenters = [v for v in m['norm_members'].values() if v != decision]
        n_dissent.append(len(dissenters))
        # hawkish dissent = voted hike when majority held/cut, or held when majority cut
        hawk_d = sum(1 for v in dissenters if
                     (v == 'hike') or (v == 'hold' and decision == 'cut'))
        dove_d = sum(1 for v in dissenters if
                     (v == 'cut') or (v == 'hold' and decision == 'hike'))
        dissent_hawkish.append(hawk_d)
        dissent_dovish.append(dove_d)

    x = np.arange(len(meetings))
    bar_w = 0.65
    ax_main.bar(x, dissent_hawkish, bar_w, label='Hawkish dissent', color=VOTE_COLORS['hike'], alpha=0.8)
    ax_main.bar(x, dissent_dovish, bar_w, bottom=dissent_hawkish,
                label='Dovish dissent', color=VOTE_COLORS['cut'], alpha=0.8)
    ax_main.set_xticks(x)
    ax_main.set_xticklabels([m['date'][2:7] for m in meetings],
                             rotation=45, ha='right', fontsize=7.5)
    ax_main.set_yticks(range(6))
    ax_main.set_ylabel('Dissenters', fontsize=10)
    ax_main.set_title('Number of Dissenters per Meeting — Red = Hawkish, Blue = Dovish',
                      fontsize=11, pad=8)
    ax_main.legend(fontsize=9, loc='upper right')
    ax_main.grid(True, axis='y')
    ax_main.set_ylim(0, 5.5)

    # Annotate majority decision above bars
    for i, m in enumerate(meetings):
        sym = {'hike': '▲', 'hold': '●', 'cut': '▼'}[m['decision']]
        ax_main.text(i, 5.1, sym, ha='center', fontsize=7.5, color=VOTE_COLORS[m['decision']])

    # --- Pie 1: all dissent ---
    all_hawk_d = sum(dissent_hawkish)
    all_dove_d = sum(dissent_dovish)
    _draw_dissent_pie(ax_pie1, all_hawk_d, all_dove_d,
                      'All Meetings (Jun 2023 – May 2025)')

    # --- Pie 2: recent 2024–2025 ---
    recent_idx = [i for i, m in enumerate(meetings) if m['date'] >= '2024-01-01']
    r_hawk = sum(dissent_hawkish[i] for i in recent_idx)
    r_dove = sum(dissent_dovish[i] for i in recent_idx)
    _draw_dissent_pie(ax_pie2, r_hawk, r_dove, 'Recent (Jan 2024 – May 2025)')

    fig.suptitle('MPC Dissent Analysis — Dissenters Split Between Faster & Slower Pace',
                 fontsize=13, color='#e6edf3', y=0.97)

    pdf.savefig(fig, bbox_inches='tight')
    plt.close(fig)


def _draw_dissent_pie(ax, hawk: int, dove: int, title: str):
    ax.set_facecolor('#161b22')
    if hawk + dove == 0:
        ax.text(0.5, 0.5, 'No dissent', ha='center', va='center',
                transform=ax.transAxes, color='#8b949e')
        ax.axis('off')
        ax.set_title(title, fontsize=9, pad=6)
        return

    sizes = [hawk, dove]
    colors = [VOTE_COLORS['hike'], VOTE_COLORS['cut']]
    labels = [f'Hawkish\n({hawk})', f'Dovish\n({dove})']
    wedges, texts, autotexts = ax.pie(
        sizes, labels=labels, colors=colors,
        autopct='%1.0f%%', startangle=90,
        wedgeprops={'edgecolor': '#0d1117', 'linewidth': 1.5},
        textprops={'color': '#e6edf3', 'fontsize': 9},
    )
    for at in autotexts:
        at.set_color('#0d1117')
        at.set_fontweight('bold')
    ax.set_title(title, fontsize=9, pad=6)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description='BoE MPC Voting History PDF')
    parser.add_argument('--out', default='boe_mpc_votes.pdf',
                        help='Output PDF path (default: boe_mpc_votes.pdf)')
    args = parser.parse_args()

    meetings = parse_meetings()

    print(f"Generating BoE MPC vote analysis PDF → {args.out}")
    print(f"  Meetings loaded: {len(meetings)}")
    print(f"  Date range: {meetings[0]['date']} to {meetings[-1]['date']}")

    with PdfPages(args.out) as pdf:
        print("  [1/6] Cover page...")
        page_cover(pdf, meetings)

        print("  [2/6] Bank Rate timeline...")
        page_rate_timeline(pdf, meetings)

        print("  [3/6] Vote split stacked bars...")
        page_vote_stacks(pdf, meetings)

        print("  [4/6] Member heatmap...")
        page_heatmap(pdf, meetings)

        print("  [5/6] Member profiles...")
        page_profiles(pdf, meetings)

        print("  [6/6] Dissent analysis...")
        page_dissent(pdf, meetings)

        # PDF metadata
        d = pdf.infodict()
        d['Title'] = 'BoE MPC Vote Analysis'
        d['Author'] = 'boe_mpc_votes.py'
        d['Subject'] = 'Bank of England Monetary Policy Committee voting history 2023-2025'
        d['Keywords'] = 'BoE MPC Bank Rate monetary policy votes hawk dove'

    print(f"\nDone. PDF saved to: {args.out}")


if __name__ == '__main__':
    main()
