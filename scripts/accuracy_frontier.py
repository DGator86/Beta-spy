"""Map the selectivity/accuracy frontier of the decision layer, robustly.

For every candidate gate configuration this evaluates 15-minute directional
accuracy over the dumped signals, then re-evaluates it on 70 half-and-half
day-block partitions (CPCV style). A config only counts as achieving an
accuracy level if its *worst-decile* partition still clears it — full-sample
accuracy alone is how you fool yourself on a few weeks of tape.

Output: the frontier (best robust accuracy at each trade-frequency tier) and
the full ranked table.
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
FLOW_COLS = ["f_flow_ew", "f_flow_weighted", "f_spy_flow"]
BLOCKED_WINDOWS = ((150, 180), (300, 330), (360, 390))
HORIZON = 15


def _sign(series: pd.Series) -> pd.Series:
    return np.sign(series.fillna(0.0)).astype(int)


def prepare(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["date"] = df["timestamp"].dt.date.astype(str)
    eastern = df["timestamp"].dt.tz_convert("America/New_York")
    df["from_open"] = eastern.dt.hour * 60 + eastern.dt.minute - (9 * 60 + 30)
    blocked = pd.Series(False, index=df.index)
    for start, end in BLOCKED_WINDOWS:
        blocked |= (df["from_open"] >= start) & (df["from_open"] < end)
    df["session_ok"] = ~blocked
    return df


def trade_mask(
    df: pd.DataFrame,
    *,
    min_p: float,
    votes_needed: int,
    vote_conf: float,
    breadth_mode: str,
    flow_gate: bool,
    min_exp_bps: float,
) -> tuple[pd.Series, pd.Series]:
    p = df[f"p_up_{HORIZON}"]
    direction = pd.Series(0, index=df.index, dtype=int)
    direction[p >= min_p] = 1
    direction[p <= 1.0 - min_p] = -1

    signs = {c: _sign(df[c]) for c in BREADTH_COLS}
    agree = sum((signs[c] == direction) & (direction != 0) for c in BREADTH_COLS)
    known = sum((signs[c] != 0).astype(int) for c in BREADTH_COLS)
    if breadth_mode == "strict":
        breadth_ok = (agree >= np.maximum(2, known // 2 + 1)) & (known > 0)
    elif breadth_mode == "super":
        breadth_ok = (agree >= np.maximum(3, known - 1)) & (known >= 3)
    else:
        breadth_ok = pd.Series(True, index=df.index)

    dirs = {h: np.where(df[f"p_up_{h}"] >= 0.5, 1, -1) for h in (5, 15, 30)}
    confident = {h: df[f"conf_{h}"] > vote_conf for h in (5, 15, 30)}
    votes = sum(((dirs[h] == direction) & confident[h]).astype(int) for h in (5, 15, 30))
    agree_ok = votes >= votes_needed

    flow_signs = {c: _sign(df[c]) for c in FLOW_COLS}
    flow_known = sum((flow_signs[c] != 0).astype(int) for c in FLOW_COLS)
    flow_agree = sum((flow_signs[c] == direction).astype(int) for c in FLOW_COLS)
    flow_ok = (
        (flow_known == 0) | (flow_agree >= 2)
        if flow_gate
        else pd.Series(True, index=df.index)
    )

    exp_ok = (
        df[f"exp_bps_{HORIZON}"].abs() >= min_exp_bps
        if min_exp_bps > 0
        else pd.Series(True, index=df.index)
    )

    liquidity_ok = df["f_spy_spread_bps"].isna() | (df["f_spy_spread_bps"] <= 4.0)
    coverage_ok = (df["f_coverage_ratio"] >= 0.90) & (df["f_covered_weight"] >= 0.85)

    fwd = df[f"fwd_bps_{HORIZON}"]
    trades = (
        (direction != 0)
        & breadth_ok
        & agree_ok
        & flow_ok
        & exp_ok
        & liquidity_ok
        & coverage_ok
        & df["session_ok"]
        & fwd.notna()
    )
    return trades, fwd * direction


def cpcv_accuracy(df: pd.DataFrame, trades: pd.Series, realized: pd.Series, blocks: int) -> dict:
    dates = sorted(df["date"].unique())
    chunks = [list(c) for c in np.array_split(np.array(dates), blocks)]
    combos = list(itertools.combinations(range(len(chunks)), len(chunks) // 2))
    accuracies = []
    for combo in combos:
        keep = {d for i in combo for d in chunks[i]}
        mask = trades & df["date"].isin(keep)
        if mask.sum() >= 8:
            accuracies.append(float((realized[mask] > 0).mean()))
    if not accuracies:
        return {"partitions": 0}
    arr = np.array(accuracies)
    return {
        "partitions": len(arr),
        "median": float(np.median(arr)),
        "p10": float(np.percentile(arr, 10)),
        "pct_ge_65": float((arr >= 0.65).mean()),
        "pct_ge_75": float((arr >= 0.75).mean()),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--signals", required=True)
    parser.add_argument("--blocks", type=int, default=8)
    args = parser.parse_args()

    df = prepare(pd.read_csv(args.signals, parse_dates=["timestamp"]))
    sessions = df["date"].nunique()
    print(f"tape: {sessions} sessions, {len(df)} minutes\n")

    rows = []
    for min_p in (0.55, 0.58, 0.60, 0.62, 0.65, 0.68, 0.70, 0.72, 0.75):
        for votes_needed in (2, 3):
            for vote_conf in (0.05, 0.30):
                for breadth_mode in ("strict", "super"):
                    for flow_gate in (False, True):
                        for min_exp_bps in (0.0, 2.0):
                            trades, realized = trade_mask(
                                df,
                                min_p=min_p,
                                votes_needed=votes_needed,
                                vote_conf=vote_conf,
                                breadth_mode=breadth_mode,
                                flow_gate=flow_gate,
                                min_exp_bps=min_exp_bps,
                            )
                            n = int(trades.sum())
                            if n < max(sessions, 10):  # ~1/day minimum to mean anything
                                continue
                            hits = realized[trades] > 0
                            robust = cpcv_accuracy(df, trades, realized, args.blocks)
                            if robust.get("partitions", 0) < 30:
                                continue
                            rows.append(
                                {
                                    "min_p": min_p,
                                    "votes": votes_needed,
                                    "conf": vote_conf,
                                    "breadth": breadth_mode,
                                    "flow": flow_gate,
                                    "exp": min_exp_bps,
                                    "trades": n,
                                    "per_day": round(n / sessions, 1),
                                    "acc": round(float(hits.mean()), 3),
                                    "mean_bps": round(float(realized[trades].mean()), 2),
                                    "cv_med": round(robust["median"], 3),
                                    "cv_p10": round(robust["p10"], 3),
                                    "ge65": round(robust["pct_ge_65"], 2),
                                    "ge75": round(robust["pct_ge_75"], 2),
                                }
                            )

    table = pd.DataFrame(rows).sort_values(["cv_p10", "cv_med"], ascending=False)
    print("=== Top 20 by worst-decile (p10) partition accuracy ===")
    print(table.head(20).to_string(index=False))

    print("\n=== Frontier: best robust accuracy by trade frequency ===")
    for lo, hi, label in ((20, 999, ">=20/day"), (8, 20, "8-20/day"), (3, 8, "3-8/day"), (1, 3, "1-3/day")):
        tier = table[(table["per_day"] >= lo) & (table["per_day"] < hi)]
        if len(tier):
            print(f"\n-- {label} --")
            print(tier.head(3).to_string(index=False))


if __name__ == "__main__":
    main()
