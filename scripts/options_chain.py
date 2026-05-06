#!/usr/bin/env python3
"""Options Chain Viewer — IV table, delta, smile for any ticker."""

import math, os, sys, time
from datetime import datetime

try:
    import yfinance as yf
    import pandas as pd
except ImportError:
    print("yfinance/pandas not installed.")
    sys.exit(1)

R = "\033[0m"; BOLD = "\033[1m"; DIM = "\033[2m"
GREEN = "\033[92m"; RED = "\033[91m"; YELLOW = "\033[93m"
CYAN = "\033[96m"; WHITE = "\033[97m"; BLUE = "\033[94m"

RFR = 0.045  # approx risk-free rate
MAX_IV = 150000

def _int(v, default=0):
    """NaN-safe int: NaN is truthy so `v or 0` doesn't protect against it."""
    return default if (v is None or v != v) else int(v)

def norm_cdf(x):
    return 0.5 * (1 + math.erf(x / math.sqrt(2)))


def bs_delta(S, K, T, sigma, is_call):
    if T <= 0 or sigma <= 0:
        if is_call:  return 1.0 if S > K else 0.0
        else:        return -1.0 if S < K else 0.0
    d1 = (math.log(S / K) + (RFR + 0.5 * sigma ** 2) * T) / (sigma * math.sqrt(T))
    return norm_cdf(d1) if is_call else norm_cdf(d1) - 1


def dte(expiry_str):
    exp = datetime.strptime(expiry_str, "%Y-%m-%d")
    return max(0, (exp - datetime.now()).days)


def draw_smile(strikes, call_ivs, put_ivs, spot, width=40):
    """ASCII IV smile — calls green, puts red, shared vertical scale."""
    triples = [(s, c, p) for s, c, p in zip(strikes, call_ivs, put_ivs)
               if c == c and p == p and c > 0 and p > 0]
    if len(triples) < 3:
        return
    sv, cv, pv = zip(*triples)
    all_iv = cv + pv
    lo  = min(all_iv) * 0.95
    hi  = max(all_iv) * 1.05
    rng = hi - lo
    h   = 7

    print(f"\n  {CYAN}IV Smile{R}  {DIM}calls={GREEN}█{DIM}  puts={RED}█{R}\n")
    for row in range(h - 1, -1, -1):
        threshold = lo + rng * row / (h - 1)
        c_row = "".join(GREEN + "█" + R if v >= threshold else " " for v in cv[-width:])
        p_row = "".join(RED   + "█" + R if v >= threshold else " " for v in pv[-width:])
        label = f"{lo + rng * row / (h - 1):.0f}%"
        print(f"  {DIM}{label:>5}{R} │{c_row}   {p_row}")

    n = min(width, len(sv))
    print(f"  {DIM}{'':>6}└{'─' * n}   {'─' * n}{R}")
    # ATM marker
    if sv:
        atm_idx = min(range(len(sv)), key=lambda i: abs(sv[i] - spot))
        atm_idx = min(atm_idx, n - 1)
        atm_line = " " * atm_idx + "↑"
        print(f"  {DIM}{'':>7}{YELLOW}{atm_line}{R}  ATM = {spot:.2f}")


def draw_rr_plot(deltas, rr_vals, h=11, width=70):
    """ASCII dot plot of call IV − put IV vs call delta (risk reversal skew)."""
    pairs = [(d, r) for d, r in zip(deltas, rr_vals)
             if d == d and r == r and 0.02 < d < 0.98]
    if len(pairs) < 3:
        return
    dv, rv = zip(*pairs)
    lo = min(rv); hi = max(rv)
    pad = max(abs(lo), abs(hi)) * 0.12
    lo -= pad; hi += pad
    rng = hi - lo or 1.0

    def val_to_row(v):
        return max(0, min(h - 1, round((1 - (v - lo) / rng) * (h - 1))))

    def delta_to_col(d):
        return max(0, min(width - 1, round(d * (width - 1))))

    zero_row = val_to_row(0)
    atm_col  = delta_to_col(0.5)

    # build sparse grid — on collision keep whichever is further from zero
    grid = [[None] * width for _ in range(h)]
    for d, v in pairs:
        col = delta_to_col(d)
        row = val_to_row(v)
        prev = grid[row][col]
        if prev is None or abs(v) > abs(prev):
            grid[row][col] = v

    print(f"\n  {CYAN}Risk Reversal  {DIM}(call IV − put IV) vs call Δ{R}\n")
    for row in range(h):
        label_v = hi - rng * row / (h - 1)
        label   = f"{label_v:+.1f}%"
        is_zero = (row == zero_row)
        line    = ""
        for col in range(width):
            v = grid[row][col]
            if v is not None:
                color = GREEN if v > 0 else (RED if v < 0 else YELLOW)
                line += color + BOLD + "●" + R
            elif is_zero:
                line += DIM + ("┼" if col == atm_col else "─") + R
            elif col == atm_col:
                line += DIM + "│" + R
            else:
                line += " "
        axis = f"{YELLOW}──{R}" if is_zero else "  "
        print(f"  {DIM}{label:>7}{R} {axis}│{line}")

    print(f"  {DIM}{'':>9}└{'─' * width}{R}")

    # tick marks at standard delta levels
    ticks = [0.10, 0.25, 0.50, 0.75, 0.90]
    tick_row  = [" "] * width
    label_row = list(" " * width)
    for t in ticks:
        col = delta_to_col(t)
        tick_row[col] = "↑"
        s = f"Δ{t:.2f}"
        start = max(0, col - len(s) // 2)
        for i, ch in enumerate(s):
            if start + i < width:
                label_row[start + i] = ch
    print(f"  {DIM}{'':>10}{''.join(tick_row)}{R}")
    print(f"  {DIM}{'':>10}{''.join(label_row)}{R}")


def draw_oi_tables(calls_d, puts_d, all_k, spot, top_n=8):
    """Two ranked OI tables — calls and puts — printed side by side."""
    def build_rows(side_d, top3):
        ranked = sorted(
            [(k, _int(side_d[k].get("openInterest"))) for k in all_k if k in side_d],
            key=lambda x: -x[1]
        )[:top_n]
        rows = []
        for rank, (k, oi) in enumerate(ranked, 1):
            d    = side_d[k]
            vol  = _int(d.get("volume"))
            mid  = ((d.get("bid") or 0) + (d.get("ask") or 0)) / 2
            dlt  = d.get("delta", float("nan"))
            pct  = (k - spot) / spot * 100
            sign = "+" if pct >= 0 else ""
            star = "★" if k in top3 else " "
            rows.append((rank, k, oi, vol, mid, dlt, pct, sign, star))
        return rows

    call_rows = build_rows(calls_d, {k for k, _ in sorted(
        [(k, _int(calls_d[k].get("openInterest"))) for k in all_k if k in calls_d],
        key=lambda x: -x[1])[:3]})
    put_rows  = build_rows(puts_d,  {k for k, _ in sorted(
        [(k, _int(puts_d[k].get("openInterest")))  for k in all_k if k in puts_d],
        key=lambda x: -x[1])[:3]})

    def fmt_row(rank, k, oi, vol, mid, dlt, pct, sign, star, color):
        dlt_s = f"{dlt:.2f}" if dlt == dlt else "  ─ "
        return (f"  {color}{rank:>2}{R}  {color}{BOLD}{k:>8.2f}{R}  "
                f"{DIM}{sign}{pct:.1f}%{R}  "
                f"{color}{oi:>8,}{YELLOW}{star}{R}  "
                f"{DIM}{vol:>7,}{R}  "
                f"{DIM}{mid:>6.2f}{R}  "
                f"{DIM}{dlt_s:>5}{R}")

    col_hdr = (f"  {DIM}{'#':>2}  {'Strike':>8}  {'vs spot':>6}  "
               f"{'OI':>9}  {'Vol':>7}  {'Mid':>6}  {'Δ':>5}{R}")
    col_sep = f"  {DIM}{'─' * 52}{R}"

    print(f"\n  {GREEN}{BOLD}Call OI Leaders{R}")
    print(col_hdr)
    print(col_sep)
    for r in call_rows:
        print(fmt_row(*r, GREEN))

    print(f"\n  {RED}{BOLD}Put OI Leaders{R}")
    print(col_hdr)
    print(col_sep)
    for r in put_rows:
        print(fmt_row(*r, RED))


def show_chain(sym, expiry):
    os.system("clear")
    print(f"\n  {DIM}Loading {sym} {expiry} chain...{R}", end="", flush=True)
    try:
        t     = yf.Ticker(sym)
        spot  = t.fast_info.last_price
        chain = t.option_chain(expiry)
        calls = chain.calls.copy()
        puts  = chain.puts.copy()
    except Exception as e:
        print(f"\n  {RED}Error: {e}{R}")
        return

    T = dte(expiry) / 365.0

    # filter to ±18% around spot
    lo_k = spot * 0.9
    hi_k = spot * 1.1
    calls = calls[(calls["strike"] >= lo_k) & (calls["strike"] <= hi_k)]
    puts  = puts [(puts["strike"]  >= lo_k) & (puts["strike"]  <= hi_k)]

    def delta_col(row, is_call):
        iv = row.get("impliedVolatility", 0.0)
        if not iv or not (iv == iv):
            return float("nan")
        return bs_delta(spot, row["strike"], T, iv, is_call)

    calls = calls.assign(delta=calls.apply(lambda r: delta_col(r, True),  axis=1))
    puts  = puts.assign( delta=puts.apply( lambda r: delta_col(r, False), axis=1))

    calls_d = calls.set_index("strike").to_dict("index")
    puts_d  = puts.set_index("strike").to_dict("index")
    all_k   = sorted(set(calls["strike"].tolist() + puts["strike"].tolist()))

    # top-3 OI strikes per side
    call_oi_ranked = sorted(
        [(k, _int(calls_d[k].get("openInterest"))) for k in all_k if k in calls_d],
        key=lambda x: -x[1]
    )
    put_oi_ranked = sorted(
        [(k, _int(puts_d[k].get("openInterest")))  for k in all_k if k in puts_d],
        key=lambda x: -x[1]
    )
    top3_call_oi = {k for k, _ in call_oi_ranked[:3]}
    top3_put_oi  = {k for k, _ in put_oi_ranked[:3]}

    # max pain: expiry price minimising total intrinsic value to option buyers
    def calc_max_pain(calls_d, puts_d, strikes):
        best_k, best_pain = None, float("inf")
        for p in strikes:
            call_pain = sum(max(0, p - k) * _int(calls_d[k].get("openInterest"))
                            for k in strikes if k in calls_d)
            put_pain  = sum(max(0, k - p) * _int(puts_d[k].get("openInterest"))
                            for k in strikes if k in puts_d)
            total = call_pain + put_pain
            if total < best_pain:
                best_pain, best_k = total, p
        return best_k

    max_pain_strike = calc_max_pain(calls_d, puts_d, all_k)

    # put/call OI ratio
    total_call_oi = sum(_int(calls_d[k].get("openInterest")) for k in all_k if k in calls_d)
    total_put_oi  = sum(_int(puts_d[k].get("openInterest"))  for k in all_k if k in puts_d)
    pc_ratio = total_put_oi / total_call_oi if total_call_oi else float("nan")

    os.system("clear")
    print(f"\n  {BLUE}{BOLD}{sym}  {expiry}{R}  "
          f"{DIM}│  spot {spot:.2f}  │  {dte(expiry)}d  │  Ctrl+C exit{R}\n")

    hdr = (f"  {DIM}{'─CALLS─':>34}  {'Strike':^8}  {'─PUTS─':<34}{R}")
    sub = (f"  {DIM}{'Δ':>5}  {'IV':>6}  {'Mid':>6}  {'Vol':>6}  {'OI':>7}"
           f"  {'Strike':^8}"
           f"  {'Δ':>5}  {'IV':>6}  {'Mid':>6}  {'Vol':>6}  {'OI':>7}{R}")
    sep = f"  {DIM}{'─' * 86}{R}"
    print(hdr)
    print(sub)
    print(sep)

    smile_k, smile_c_iv, smile_p_iv = [], [], []

    def fmt_f(x, fmt=".2f"):
        return format(x, fmt) if x == x and x is not None else "─"

    def fmt_oi(val, k, top3_set):
        if not val:
            return f"{'─':>7}"
        s = f"{val:,}"
        star = f"{YELLOW}★{R}" if k in top3_set else " "
        return f"{s:>6}{star}"

    for k in all_k:
        atm = abs(k - spot) <= spot * 0.005

        c = calls_d.get(k, {})
        p = puts_d.get(k, {})

        c_iv_raw = (c.get("impliedVolatility") or 0) * 100
        p_iv_raw = (p.get("impliedVolatility") or 0) * 100

        c_d   = fmt_f(c.get("delta", float("nan")), ".2f")
        c_iv  = fmt_f(c_iv_raw, ".1f") + ("%" if c else "")
        c_mid = fmt_f(((c.get("bid") or 0) + (c.get("ask") or 0)) / 2, ".2f") if c else "─"
        c_vol = f"{_int(c.get('volume')):,}" if c else "─"
        c_oi  = fmt_oi(_int(c.get("openInterest")) if c else 0, k, top3_call_oi)

        p_d   = fmt_f(p.get("delta", float("nan")), ".2f")
        p_iv  = fmt_f(p_iv_raw, ".1f") + ("%" if p else "")
        p_mid = fmt_f(((p.get("bid") or 0) + (p.get("ask") or 0)) / 2, ".2f") if p else "─"
        p_vol = f"{_int(p.get('volume')):,}" if p else "─"
        p_oi  = fmt_oi(_int(p.get("openInterest")) if p else 0, k, top3_put_oi)

        atm_tag = f" {YELLOW}◄{R}" if atm else ""
        mp_tag  = f" {CYAN}↔{R}" if k == max_pain_strike else ""

        row_c = f"{GREEN}{c_d:>5}  {c_iv:>6}  {c_mid:>6}  {c_vol:>6}  {c_oi}{R}"
        row_p = f"{RED}{p_d:>5}  {p_iv:>6}  {p_mid:>6}  {p_vol:>6}  {p_oi}{R}"
        k_col = f"{YELLOW if atm else WHITE}{BOLD}{k:>8.2f}{R}{atm_tag}{mp_tag}"

        if c_iv_raw < MAX_IV / 100 and p_iv_raw < MAX_IV / 100:
            print(f"  {row_c}  {k_col}  {row_p}")

        if c and p and c_iv_raw < MAX_IV / 100 and p_iv_raw < MAX_IV / 100:
            smile_k.append(k)
            smile_c_iv.append(c_iv_raw)
            smile_p_iv.append(p_iv_raw)

    sep2 = f"  {DIM}{'─' * 86}{R}"
    print(sep2)
    print(f"  {DIM}{YELLOW}★{R}{DIM} = top-3 OI   {CYAN}↔{R}{DIM} = max pain   {YELLOW}◄{R}{DIM} = ATM{R}")

    draw_oi_tables(calls_d, puts_d, all_k, spot)
    draw_smile(smile_k, smile_c_iv, smile_p_iv, spot)
    rr_vals     = [c - p for c, p in zip(smile_c_iv, smile_p_iv)]
    smile_deltas = [calls_d[k].get("delta", float("nan")) for k in smile_k]
    draw_rr_plot(smile_deltas, rr_vals)

    # summary stats
    atm_call = min(calls_d, key=lambda k: abs(k - spot), default=None)
    if atm_call and atm_call in calls_d:
        c = calls_d[atm_call]
        iv_atm = (c.get("impliedVolatility") or 0) * 100
        exp_move = spot * iv_atm / 100 * math.sqrt(dte(expiry) / 365)
        print(f"\n  {CYAN}ATM stats{R}  {DIM}strike {atm_call:.2f}{R}")
        print(f"  ATM IV:       {BOLD}{iv_atm:.1f}%{R}")
        print(f"  Expected move {dte(expiry)}d:  {BOLD}±${exp_move:.2f}  "
              f"(±{exp_move/spot*100:.1f}%){R}")

    # put/call OI and max pain
    pc_color = RED if pc_ratio > 1.2 else (GREEN if pc_ratio < 0.8 else WHITE)
    print(f"\n  {CYAN}Flow stats{R}")
    print(f"  P/C OI ratio:   {pc_color}{BOLD}{pc_ratio:.2f}{R}  "
          f"{DIM}(calls {total_call_oi:,}  puts {total_put_oi:,}){R}")
    if max_pain_strike is not None:
        mp_dist = max_pain_strike - spot
        mp_pct  = mp_dist / spot * 100
        sign    = "+" if mp_dist >= 0 else ""
        print(f"  Max pain:       {BOLD}{max_pain_strike:.2f}{R}  "
              f"{DIM}({sign}{mp_dist:.2f} / {sign}{mp_pct:.1f}% from spot){R}")


def main():
    while True:
        os.system("clear")
        print(f"\n  {BLUE}{BOLD}Options Chain Viewer{R}\n")
        print(f"  Enter ticker (SPY, AAPL, NVDA ...) or {CYAN}q{R} to quit")
        print()
        sym = input(f"  {CYAN}Ticker:{R} ").strip().upper()
        if sym in ("Q", "QUIT", ""):
            break

        try:
            t        = yf.Ticker(sym)
            expiries = t.options
            if not expiries:
                print(f"\n  {RED}No options data for {sym}.{R}")
                time.sleep(2)
                continue
        except Exception as e:
            print(f"\n  {RED}Error: {e}{R}")
            time.sleep(2)
            continue

        os.system("clear")
        print(f"\n  {BLUE}{BOLD}Options — {sym}{R}  {DIM}Available expiries:{R}\n")
        for i, exp in enumerate(expiries[:14], 1):
            d = dte(exp)
            marker = f"  {YELLOW}← nearest{R}" if i == 1 else ""
            print(f"  {CYAN}{i:>2}{R}  {exp}  {DIM}({d}d){R}{marker}")
        print()

        raw = input(f"  {CYAN}Expiry # (Enter = nearest, q = back):{R} ").strip().lower()
        if raw == "q":
            continue
        try:
            idx = int(raw) - 1 if raw else 0
            expiry = expiries[max(0, min(idx, len(expiries) - 1))]
        except ValueError:
            expiry = expiries[0]

        show_chain(sym, expiry)
        print(f"\n  {DIM}Press Enter to continue...{R}")
        input()


if __name__ == "__main__":
    main()
