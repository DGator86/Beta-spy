"""Compare the production decision config with and without the session-window gate.

Runs the same trade mask as production over the dumped signals, then reports
full-sample stats and CPCV-style partition robustness (median mean-bps and the
fraction of 70 half-and-half day-block partitions that end negative) for both
variants. The gate should raise mean bps and reduce negative partitions.
"""

from __future__ import annotations

import argparse
import itertools

import numpy as np
import pandas as pd

BREADTH_COLS = [
    "f_trend_ew",
    "f_trend_weighted",
    "f_momentum_ew",
    "f_momentum_weighted",
    "f_participation",
]
BLOCKED_WINDOWS = ((150, 180), (300, 330), (360, 390))
MIN_P = 0.58
HORIZON = 15


def _sign(series: pd.Series) -> pd.Series:
    return np.sign(series.fillna(0.0)).astype(int)


def build(df: pd.DataFrame, *, session_gate: bool) -> tuple[pd.Series, pd.Series]:
    p = df[f"p_up_{HORIZON}"]
    direction = pd.Series(0, index=df.index, dtype=int)
    direction[p >= MIN_P] = 1
    direction[p <= 1.0 - MIN_P] = -1

    signs = {c: _sign(df[c]) for c in BREADTH_COLS}
    agree = sum((signs[c] == direction) & (direction != 0) for c in BREADTH_COLS)
    known = sum((signs[c] != 0).astype(int) for c in BREADTH_COLS)
    breadth_ok = (agree >= np.maximum(2, known // 2 + 1)) & (known > 0)

    dirs = {h: np.where(df[f"p_up_{h}"] >= 0.5, 1, -1) for h in (5, 15, 30)}
    confident = {h: df[f"conf_{h}"] > 0.05 for h in (5, 15, 30)}
    votes = sum(((dirs[h] == direction) & confident[h]).astype(int) for h in (5, 15, 30))
    agree_ok = votes >= 2

    liquidity_ok = df["f_spy_spread_bps"].isna() | (df["f_spy_spread_bps"] <= 4.0)
    coverage_ok = (df["f_coverage_ratio"] >= 0.90) & (df["f_covered_weight"] >= 0.85)

    fwd = df[f"fwd_bps_{HORIZON}"]
    trades = (direction != 0) & breadth_ok & agree_ok & liquidity_ok & coverage_ok & fwd.notna()

    if session_gate:
        eastern = df["timestamp"].dt.tz_convert("America/New_York")
        from_open = eastern.dt.hour * 60 + eastern.dt.minute - (9 * 60 + 30)
        blocked = pd.Series(False, index=df.index)
        for start, end in BLOCKED_WINDOWS:
            blocked |= (from_open >= start) & (from_open < end)
        trades &= ~blocked

    return trades, fwd * direction


def report(name: str, df: pd.DataFrame, trades: pd.Series, realized: pd.Series, blocks: int) -> None:
    pnl = realized[trades]
    dev = float(pnl.std(ddof=1)) if len(pnl) > 1 else 0.0
    t_stat = float(pnl.mean() / (dev / np.sqrt(len(pnl)))) if dev > 0 else 0.0

    dates = sorted(df["date"].unique())
    chunks = [list(c) for c in np.array_split(np.array(dates), blocks)]
    combos = list(itertools.combinations(range(len(chunks)), len(chunks) // 2))
    means = []
    for combo in combos:
        keep = {d for i in combo for d in chunks[i]}
        mask = trades & df["date"].isin(keep)
        if mask.sum() >= 10:
            means.append(float(realized[mask].mean()))
    arr = np.array(means) if means else np.array([np.nan])
    print(
        f"{name:12s} trades={int(trades.sum()):4d} mean_bps={pnl.mean():+.3f} "
        f"total_bps={pnl.sum():+.1f} t_stat={t_stat:+.2f} "
        f"partition_median_bps={np.nanmedian(arr):+.3f} pct_partitions_neg={float((arr < 0).mean()):.2f}"
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--signals", required=True)
    parser.add_argument("--blocks", type=int, default=8)
    args = parser.parse_args()

    df = pd.read_csv(args.signals, parse_dates=["timestamp"])
    df["date"] = df["timestamp"].dt.date.astype(str)

    for session_gate in (False, True):
        trades, realized = build(df, session_gate=session_gate)
        report("gated" if session_gate else "ungated", df, trades, realized, args.blocks)


if __name__ == "__main__":
    main()
