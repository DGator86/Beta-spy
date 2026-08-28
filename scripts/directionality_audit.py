#!/usr/bin/env python3
"""Does the tape support a directional bet at all?

Gate 1 of the long-directional plan. The 56.9% directional hit rate measured
from ``paper.sqlite`` came from trades the system chose to take, so it is
selection-biased: the sample is conditioned on the gate having fired. This
script re-measures direction as a *forecast* over every session in the captured
tape, whether or not anything was traded, at three levels of increasing
strictness.

Level 1  Day-level realised move for a 10:00 entry, so the magnitudes can be
         read against the option break-even table.
Level 2  Bar-level hit rate for each raw feature, with a day-clustered
         bootstrap (overlapping forward windows inside a session are not
         independent, so the naive z is meaningless) and a split by side.
         A signal that only works long is the period's drift, not a forecast.
Level 3  Walk-forward logistic over the whole feature vector, expanding window,
         trained strictly on prior sessions. Scored against the always-long
         baseline with a *paired* bootstrap over the same resampled days,
         because beating a coin flip in a rising market is not an edge.
Level 4  Untrained day-level rules at the horizon the strategy actually bets
         on. No fitting means all sessions are scoreable, and the exact
         binomial p-value is honest about a 25-session sample.

Input is the flattened tape panel produced by ``--extract`` from the raw
``ticks_YYYY-MM-DD.jsonl`` capture files.

Usage:
    python scripts/directionality_audit.py --extract --tape DIR --work DIR
    python scripts/directionality_audit.py --work DIR
"""

from __future__ import annotations

import argparse
import csv
import glob
import json
import os
import random
import statistics as st
from collections import defaultdict
from math import comb

import numpy as np

FEATURES = (
    "spot", "net_gex", "gamma_flip", "call_wall", "put_wall", "gex_pct_rank",
    "vix9d", "vix", "vix3m", "vvix", "straddle_breakeven", "expected_range",
    "adx", "rsi", "bb_width", "bb_width_baseline", "vwap", "tick_abs_mean",
    "cvd_slope", "pcr_volume", "rsp_spy_div", "sector_align", "top10_pressure",
    "volume_oi_ratio", "vwap_reversion_count", "has_catalyst",
)

# Signals derived from the raw features, all computable at decision time.
SIGNALS = ("mom_open", "mom_30m", "vwap_dev", "rsi_50", "cvd_slope", "net_gex",
           "top10", "rsp_div", "pcr", "wall_pos", "gflip")

ENTRY = "10:00"
EXIT_EARLY = "13:30"
EXIT_LATE = "15:30"

# Accuracy an ATM debit spread has to clear to break even, measured off this
# same tape in the break-even study: the exit time is worth ~7 points.
BREAKEVEN = {EXIT_EARLY: 0.428, EXIT_LATE: 0.501}


# --------------------------------------------------------------------------- io

def extract(tape_dir: str, work_dir: str) -> None:
    """Flatten raw capture files into a per-tick panel plus a union of bars."""
    paths = sorted(glob.glob(os.path.join(tape_dir, "ticks_*.jsonl")))
    if not paths:
        raise SystemExit(f"no ticks_*.jsonl under {tape_dir}")
    os.makedirs(work_dir, exist_ok=True)

    bars: dict[str, list] = {}
    settles: dict[str, float] = {}
    rows: list[dict] = []

    for path in paths:
        day = os.path.basename(path)[6:16]
        kept = 0
        with open(path) as fh:
            for line in fh:
                try:
                    record = json.loads(line)
                except ValueError:
                    continue
                if record.get("t") == "settle":
                    settles[record["date"]] = record["price"]
                    continue
                if record.get("t") != "tick":
                    continue
                market = record.get("market") or {}
                if market.get("spot") is None:
                    continue
                row = {"date": day, "ts": record["ts"], "seq": record.get("seq"),
                       "has_chain": 1 if record.get("option_rows") else 0}
                row.update({name: market.get(name) for name in FEATURES})
                rows.append(row)
                kept += 1
                # Every snapshot carries a few days of trailing minute bars; the
                # union covers sessions the tick capture itself missed.
                for bar in record.get("bars") or []:
                    bars.setdefault(bar[0], bar[1:])
        print(f"{day} {kept} ticks", flush=True)

    _write(os.path.join(work_dir, "panel.csv"), list(rows[0]), rows)
    _write(os.path.join(work_dir, "bars.csv"), ["ts", "open", "high", "low", "close", "volume"],
           [dict(zip(["ts", "open", "high", "low", "close", "volume"], [ts, *bars[ts]]))
            for ts in sorted(bars)])
    _write(os.path.join(work_dir, "settles.csv"), ["date", "settle"],
           [{"date": d, "settle": settles[d]} for d in sorted(settles)])
    print(f"ticks={len(rows)} bars={len(bars)} settles={len(settles)}")


def _write(path: str, fields: list[str], rows: list[dict]) -> None:
    with open(path, "w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _num(value):
    if value in (None, "", "None"):
        return None
    try:
        return float(value)
    except ValueError:
        return None


def load_panel(work_dir: str) -> dict[str, list[dict]]:
    by_day: dict[str, list[dict]] = defaultdict(list)
    with open(os.path.join(work_dir, "panel.csv")) as fh:
        for raw in csv.DictReader(fh):
            row = {k: _num(v) for k, v in raw.items() if k not in ("date", "ts")}
            row["date"] = raw["date"]
            row["ts"] = raw["ts"]
            row["hhmm"] = raw["ts"][11:16]
            by_day[raw["date"]].append(row)
    for day in by_day:
        by_day[day].sort(key=lambda r: r["ts"])
    return by_day


def minutes(hhmm: str) -> int:
    return int(hhmm[:2]) * 60 + int(hhmm[3:5])


def spot_at(rows: list[dict], hhmm: str) -> dict | None:
    """Last snapshot at or before hhmm — no lookahead."""
    found = None
    for row in rows:
        if row["hhmm"] > hhmm:
            break
        if row["spot"]:
            found = row
    return found


def signals(row: dict, session_open: float, thirty_ago: dict | None) -> dict:
    return {
        "mom_open": row["spot"] - session_open,
        "mom_30m": (row["spot"] - thirty_ago["spot"]) if thirty_ago else None,
        "vwap_dev": (row["spot"] - row["vwap"]) if row.get("vwap") else None,
        "rsi_50": (row["rsi"] - 50) if row.get("rsi") is not None else None,
        "cvd_slope": row.get("cvd_slope"),
        "net_gex": row.get("net_gex"),
        "top10": row.get("top10_pressure"),
        "rsp_div": row.get("rsp_spy_div"),
        "pcr": -(row["pcr_volume"] - 1) if row.get("pcr_volume") else None,
        "wall_pos": ((row["call_wall"] + row["put_wall"]) / 2 - row["spot"])
                    if row.get("call_wall") and row.get("put_wall") else None,
        "gflip": (row["spot"] - row["gamma_flip"]) if row.get("gamma_flip") else None,
    }


def build_samples(by_day, horizon: int, first="09:45", last="14:30") -> list[tuple]:
    """(date, feature vector, up flag, forward move) for every scoreable bar."""
    out = []
    for day in sorted(by_day):
        rows = [r for r in by_day[day] if r["spot"]]
        if len(rows) < 50:
            continue
        session_open = rows[0]["spot"]
        index = {minutes(r["hhmm"]): r for r in rows}
        for row in rows:
            now = minutes(row["hhmm"])
            if not minutes(first) <= now <= minutes(last):
                continue
            ahead = next((index[now + k] for k in range(horizon, horizon + 6)
                          if now + k in index), None)
            if ahead is None:
                continue
            vector = signals(row, session_open, index.get(now - 30))
            values = [vector[name] for name in SIGNALS]
            if any(v is None for v in values):
                continue
            move = ahead["spot"] - row["spot"]
            out.append((day, values, 1 if move > 0 else 0, move))
    return out


# ------------------------------------------------------------------ inference

def clustered_ci(pairs, reps=2000, rng=None):
    """Bootstrap a hit rate by resampling whole sessions, not bars."""
    rng = rng or random
    bucket = defaultdict(list)
    for day, hit in pairs:
        bucket[day].append(hit)
    keys = list(bucket)
    draws = []
    for _ in range(reps):
        pool = []
        for _ in keys:
            pool.extend(bucket[rng.choice(keys)])
        draws.append(sum(pool) / len(pool))
    draws.sort()
    return draws[int(0.025 * reps)], draws[int(0.975 * reps)]


def paired_vs_long(scored, reps=4000, rng=None):
    """Model hit minus always-long hit on the *same* resampled sessions."""
    rng = rng or random
    bucket = defaultdict(list)
    for day, prob, truth, _ in scored:
        bucket[day].append((1 if (prob > 0.5) == (truth == 1) else 0, truth))
    keys = list(bucket)
    diffs = []
    for _ in range(reps):
        pool = []
        for _ in keys:
            pool.extend(bucket[rng.choice(keys)])
        model = sum(x[0] for x in pool) / len(pool)
        always = sum(x[1] for x in pool) / len(pool)
        diffs.append(model - always)
    diffs.sort()
    return (diffs[int(0.025 * reps)], diffs[int(0.975 * reps)],
            sum(1 for d in diffs if d <= 0) / reps)


def exact_binomial_p(k: int, n: int, p: float = 0.5) -> float:
    """Two-sided exact test — the sample is far too small for a normal approx."""
    pmf = [comb(n, i) * p ** i * (1 - p) ** (n - i) for i in range(n + 1)]
    return min(1.0, sum(x for x in pmf if x <= pmf[k] * (1 + 1e-9)))


def fit_logistic(X, y, iters=600, lr=0.3, l2=1e-3):
    X = np.asarray(X, float)
    y = np.asarray(y, float)
    mean = X.mean(0)
    scale = X.std(0)
    scale[scale == 0] = 1.0
    Z = (X - mean) / scale
    weights = np.zeros(Z.shape[1])
    bias = 0.0
    for _ in range(iters):
        error = 1 / (1 + np.exp(-np.clip(Z @ weights + bias, -30, 30))) - y
        weights -= lr * (Z.T @ error / len(Z) + l2 * weights)
        bias -= lr * error.mean()
    return weights, bias, mean, scale


def predict(model, x) -> float:
    weights, bias, mean, scale = model
    z = (np.asarray(x, float) - mean) / scale
    return float(1 / (1 + np.exp(-np.clip(z @ weights + bias, -30, 30))))


def walk_forward(samples, warmup_days=8):
    """Train on every prior session, predict the next. Never trains on itself."""
    days = sorted({s[0] for s in samples})
    scored = []
    for i in range(warmup_days, len(days)):
        train = [s for s in samples if s[0] < days[i]]
        test = [s for s in samples if s[0] == days[i]]
        if not test:
            continue
        model = fit_logistic([s[1] for s in train], [s[2] for s in train])
        scored.extend((s[0], predict(model, s[1]), s[2], s[3]) for s in test)
    return scored


# --------------------------------------------------------------------- levels

def level1(by_day) -> None:
    print("=" * 78)
    print(f"LEVEL 1 - day-level realised move, {ENTRY} entry")
    print("=" * 78)
    print(f"{'date':11} {ENTRY:>8} {EXIT_EARLY:>8} {EXIT_LATE:>8} "
          f"{'early':>7} {'late':>7} {'brkevn':>7} {'|d|/be':>7}")
    moves, ratios, ups = [], [], 0
    for day in sorted(by_day):
        rows = by_day[day]
        entry = spot_at(rows, ENTRY)
        early = spot_at(rows, EXIT_EARLY)
        late = spot_at(rows, EXIT_LATE)
        if not (entry and early):
            continue
        d_early = early["spot"] - entry["spot"]
        d_late = (late["spot"] - entry["spot"]) if late else float("nan")
        breakeven = entry.get("straddle_breakeven") or float("nan")
        moves.append(abs(d_early))
        ups += d_early > 0
        if breakeven == breakeven:
            ratios.append(abs(d_early) / breakeven)
        print(f"{day:11} {entry['spot']:8.2f} {early['spot']:8.2f} "
              f"{(late['spot'] if late else float('nan')):8.2f} "
              f"{d_early:+7.2f} {d_late:+7.2f} {breakeven:7.2f} "
              f"{abs(d_early) / breakeven:7.2f}")
    n = len(moves)
    print(f"\nsessions={n}  up={ups} ({ups / n:.1%})  down={n - ups}")
    print(f"|{ENTRY}->{EXIT_EARLY}| move: median ${st.median(moves):.2f}  "
          f"mean ${st.mean(moves):.2f}  max ${max(moves):.2f}")
    print(f"|move| / straddle breakeven: median {st.median(ratios):.2f}  "
          f"fraction clearing 1.0: {sum(r > 1 for r in ratios) / len(ratios):.1%}")


def level2(samples, horizon: int, rng) -> None:
    print()
    print("=" * 78)
    print(f"LEVEL 2 - per-signal hit rate, forward {horizon}m, split by side")
    print("=" * 78)
    baseline = sum(1 for s in samples if s[2] == 1) / len(samples)
    print(f"bars={len(samples)}  sessions={len({s[0] for s in samples})}")
    print(f"always-long baseline {baseline:.1%}  "
          f"mean forward move {st.mean([s[3] for s in samples]):+.4f} $/bar")
    print(f"\n{'signal':10} {'long n':>7} {'long hit':>9} {'vs base':>8} | "
          f"{'short n':>7} {'short hit':>10} {'vs base':>8} | {'both':>5}")
    print("-" * 78)
    survivors = []
    for position, name in enumerate(SIGNALS):
        longs, shorts = [], []
        for day, values, up, _ in samples:
            value = values[position]
            if value == 0:
                continue
            (longs if value > 0 else shorts).append((day, up))
        if len(longs) < 200 or len(shorts) < 200:
            print(f"{name:10} one-sided in this sample (long={len(longs)} "
                  f"short={len(shorts)}) - not a signal here")
            continue
        long_hit = sum(u for _, u in longs) / len(longs)
        short_hit = 1 - sum(u for _, u in shorts) / len(shorts)
        beats_long = long_hit - baseline
        beats_short = short_hit - (1 - baseline)
        both = beats_long > 0 and beats_short > 0
        if both:
            survivors.append(name)
        print(f"{name:10} {len(longs):7d} {long_hit:9.1%} {beats_long:+8.1%} | "
              f"{len(shorts):7d} {short_hit:10.1%} {beats_short:+8.1%} | "
              f"{'yes' if both else 'no':>5}")
    print(f"\nbeat the drift on both sides: {', '.join(survivors) or 'NONE'}")
    print("A signal that only wins long is reproducing the period's drift.")


def level3(by_day, rng) -> None:
    print()
    print("=" * 78)
    print("LEVEL 3 - walk-forward logistic, expanding window, OOS by session")
    print("=" * 78)
    print(f"{'horizon':>8} {'oos n':>7} {'sess':>5} {'model':>7} {'long':>7} "
          f"{'diff':>7} {'95% CI on diff':>20} {'p(diff<=0)':>11}")
    for horizon in (30, 60, 120, 210):
        samples = build_samples(by_day, horizon)
        if not samples:
            continue
        scored = walk_forward(samples)
        if not scored:
            continue
        hit = sum(1 for _, p, y, _ in scored if (p > 0.5) == (y == 1)) / len(scored)
        always = sum(1 for _, _, y, _ in scored if y == 1) / len(scored)
        low, high, pvalue = paired_vs_long(scored, rng=rng)
        print(f"{horizon:7d}m {len(scored):7d} {len({s[0] for s in scored}):5d} "
              f"{hit:7.1%} {always:7.1%} {hit - always:+7.1%} "
              f"  [{low:+6.1%}, {high:+6.1%}]{'':4} {pvalue:11.3f}")

    print(f"\nThe decision the strategy actually makes: one call at {ENTRY}, "
          f"exit {EXIT_EARLY}")
    samples = build_samples(by_day, 210, first="09:55", last="10:05")
    scored = walk_forward(samples)
    by_session = defaultdict(list)
    for day, prob, truth, move in scored:
        by_session[day].append((prob, truth, move))
    right, correct_moves, wrong_moves = 0, [], []
    print(f"{'date':11} {'p(up)':>7} {'call':>5} {'real':>5} {'move$':>7} {'ok':>3}")
    for day in sorted(by_session):
        prob = st.mean(x[0] for x in by_session[day])
        truth, move = by_session[day][0][1], by_session[day][0][2]
        called_up = prob > 0.5
        ok = called_up == (truth == 1)
        right += ok
        (correct_moves if ok else wrong_moves).append(abs(move))
        print(f"{day:11} {prob:7.3f} {'UP' if called_up else 'DN':>5} "
              f"{'UP' if truth else 'DN':>5} {move:+7.2f} {'Y' if ok else '.':>3}")
    total = len(by_session)
    always = sum(1 for d in by_session if by_session[d][0][1] == 1)
    print(f"\nOOS sessions={total}  model {right}/{total} = {right / total:.1%}"
          f"   always-long {always}/{total} = {always / total:.1%}"
          f"   break-even needed {BREAKEVEN[EXIT_EARLY]:.1%}")
    if correct_moves and wrong_moves:
        print(f"avg |move| when right ${st.mean(correct_moves):.2f}, "
              f"when wrong ${st.mean(wrong_moves):.2f}  "
              f"(a debit spread needs right >> wrong, not right == wrong)")


def level4(by_day) -> None:
    sessions = []
    for day in sorted(by_day):
        rows = by_day[day]
        entry = spot_at(rows, ENTRY)
        early = spot_at(rows, EXIT_EARLY)
        late = spot_at(rows, EXIT_LATE)
        opening = spot_at(rows, "09:30") or rows[0]
        if not (entry and early):
            continue
        sessions.append({
            "date": day, "open": opening["spot"], "entry": entry,
            EXIT_EARLY: early["spot"] - entry["spot"],
            EXIT_LATE: (late["spot"] - entry["spot"]) if late else None,
        })

    rules = {
        "always long": lambda s: 1,
        "always short": lambda s: -1,
        "open momentum": lambda s: 1 if s["entry"]["spot"] > s["open"] else -1,
        "fade open momentum": lambda s: -1 if s["entry"]["spot"] > s["open"] else 1,
        "above VWAP": lambda s: 1 if s["entry"]["vwap"] and s["entry"]["spot"] > s["entry"]["vwap"] else -1,
        "RSI > 50": lambda s: 1 if s["entry"]["rsi"] and s["entry"]["rsi"] > 50 else -1,
        "CVD slope": lambda s: 1 if (s["entry"]["cvd_slope"] or 0) > 0 else -1,
        "above gamma flip": lambda s: 1 if s["entry"]["gamma_flip"] and s["entry"]["spot"] > s["entry"]["gamma_flip"] else -1,
        "toward wall mid": lambda s: 1 if s["entry"]["call_wall"] and s["entry"]["put_wall"]
                                    and (s["entry"]["call_wall"] + s["entry"]["put_wall"]) / 2 > s["entry"]["spot"] else -1,
        "top10 pressure": lambda s: 1 if (s["entry"]["top10_pressure"] or 0) > 0 else -1,
        "sector align": lambda s: 1 if (s["entry"]["sector_align"] or 0) > 0 else -1,
    }

    for exit_time in (EXIT_EARLY, EXIT_LATE):
        scoreable = [s for s in sessions if s[exit_time] is not None]
        print()
        print("=" * 78)
        print(f"LEVEL 4 - untrained day rules, {ENTRY} -> {exit_time}, "
              f"n={len(scoreable)} sessions")
        print(f"          option break-even accuracy at this exit: "
              f"{BREAKEVEN[exit_time]:.1%}")
        print("=" * 78)
        print(f"{'rule':20} {'hit':>6} {'k/n':>8} {'exact p':>8} "
              f"{'right $':>8} {'wrong $':>8} {'$/day':>8}")
        for name, rule in rules.items():
            correct, total, right_moves, wrong_moves, pnl = 0, 0, [], [], []
            for session in scoreable:
                move = session[exit_time]
                if move == 0:
                    continue
                side = rule(session)
                total += 1
                ok = side * move > 0
                correct += ok
                (right_moves if ok else wrong_moves).append(abs(move))
                pnl.append(side * move)
            if not total:
                continue
            print(f"{name:20} {correct / total:6.1%} {correct:3d}/{total:<4d} "
                  f"{exact_binomial_p(correct, total):8.3f} "
                  f"{(st.mean(right_moves) if right_moves else 0):8.2f} "
                  f"{(st.mean(wrong_moves) if wrong_moves else 0):8.2f} "
                  f"{st.mean(pnl):+8.2f}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--tape", default="data/tape",
                        help="directory of raw ticks_YYYY-MM-DD.jsonl capture files")
    parser.add_argument("--work", default="data/direction",
                        help="directory for the flattened panel")
    parser.add_argument("--extract", action="store_true",
                        help="rebuild the panel from the raw tape first")
    parser.add_argument("--horizon", type=int, default=60,
                        help="forward horizon in minutes for the level 2 scan")
    parser.add_argument("--seed", type=int, default=7)
    args = parser.parse_args()

    if args.extract:
        extract(args.tape, args.work)

    rng = random.Random(args.seed)
    by_day = load_panel(args.work)
    level1(by_day)
    level2(build_samples(by_day, args.horizon), args.horizon, rng)
    level3(by_day, rng)
    level4(by_day)

    print()
    print("=" * 78)
    print("Read the paired diff in level 3 and the exact p in level 4, not the")
    print("headline hit rates. Overlapping windows inside a session inflate any")
    print("naive test, and a period that rose on 68% of sessions will make every")
    print("long-biased rule look like a forecast.")
    print("=" * 78)


if __name__ == "__main__":
    main()
