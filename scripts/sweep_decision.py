"""Offline sweep of decision-layer and sizing variants over dumped signals.

Train/test split by session date guards against picking noise: configs are
ranked on the training sessions and the winner is reported on held-out
sessions it never saw.
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
DEADBAND = 0.05


def _sign(series: pd.Series) -> pd.Series:
    out = pd.Series(0, index=series.index, dtype=int)
    out[series > DEADBAND] = 1
    out[series < -DEADBAND] = -1
    out[series.isna()] = 0
    return out


def evaluate(df: pd.DataFrame, *, horizon: int, min_p: float, breadth_mode: str,
             agree_mode: str, flow_gate: bool, conf_sizing: bool) -> dict:
    p = df[f"p_up_{horizon}"]
    direction = pd.Series(0, index=df.index, dtype=int)
    direction[p >= min_p] = 1
    direction[p <= 1.0 - min_p] = -1

    signs = {c: _sign(df[c]) for c in BREADTH_COLS}
    agree_count = sum((signs[c] == direction) & (direction != 0) for c in BREADTH_COLS)
    known = sum((signs[c] != 0).astype(int) for c in BREADTH_COLS)
    if breadth_mode == "strict":
        breadth_ok = (agree_count >= np.maximum(2, known // 2 + 1)) & (known > 0)
    elif breadth_mode == "two":
        breadth_ok = agree_count >= 2
    else:
        breadth_ok = pd.Series(True, index=df.index)

    dirs = {h: np.where(df[f"p_up_{h}"] >= 0.5, 1, -1) for h in (5, 15, 30)}
    confident = {h: df[f"conf_{h}"] > 0.05 for h in (5, 15, 30)}
    if agree_mode == "two_of_three":
        votes = sum(((dirs[h] == direction) & confident[h]).astype(int) for h in (5, 15, 30))
        agree_ok = votes >= 2
    elif agree_mode == "fast_pair":
        agree_ok = ((dirs[5] == direction) & confident[5]) & ((dirs[15] == direction) & confident[15])
    else:
        agree_ok = pd.Series(True, index=df.index)

    flow_signs = {c: _sign(df[c]) for c in FLOW_COLS}
    flow_known = sum((flow_signs[c] != 0).astype(int) for c in FLOW_COLS)
    flow_agree = sum((flow_signs[c] == direction).astype(int) for c in FLOW_COLS)
    flow_ok = (flow_known == 0) | (flow_agree >= 2) if flow_gate else pd.Series(True, index=df.index)

    liquidity_ok = df["f_spy_spread_bps"].isna() | (df["f_spy_spread_bps"] <= 4.0)
    coverage_ok = (df["f_coverage_ratio"] >= 0.90) & (df["f_covered_weight"] >= 0.85)

    trades = (direction != 0) & breadth_ok & agree_ok & flow_ok & liquidity_ok & coverage_ok
    fwd = df[f"fwd_bps_{horizon}"]
    mask = trades & fwd.notna()
    if mask.sum() == 0:
        return {"trades": 0}
    realized = fwd[mask] * direction[mask]
    edge = (p[mask] - 0.5).abs()
    size = np.clip(edge / 0.10, 0.5, 2.0) if conf_sizing else pd.Series(1.0, index=realized.index)
    pnl = realized * size
    deviation = float(pnl.std(ddof=1)) if len(pnl) > 1 else 0.0
    t_stat = float(pnl.mean() / (deviation / np.sqrt(len(pnl)))) if deviation > 0 else 0.0
    return {
        "trades": int(mask.sum()),
        "accuracy": float((realized > 0).mean()),
        "mean_bps": float(realized.mean()),
        "sized_total_bps": float(pnl.sum()),
        "sized_mean_bps": float(pnl.mean()),
        "flat_total_bps": float(realized.sum()),
        "t_stat": t_stat,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--signals", required=True)
    parser.add_argument("--split-date", default="2026-08-08")
    args = parser.parse_args()

    df = pd.read_csv(args.signals, parse_dates=["timestamp"])
    df["date"] = df["timestamp"].dt.date.astype(str)
    train = df[df["date"] < args.split_date].reset_index(drop=True)
    test = df[df["date"] >= args.split_date].reset_index(drop=True)
    print(f"train rows={len(train)} sessions={train['date'].nunique()}  "
          f"test rows={len(test)} sessions={test['date'].nunique()}")

    grid = list(itertools.product(
        (5, 15, 30),                      # horizon
        (0.55, 0.58, 0.60, 0.62, 0.65),   # min_probability
        ("strict", "two", "off"),         # breadth gate
        ("two_of_three", "fast_pair", "off"),  # multi-horizon gate
        (True, False),                    # flow gate
    ))
    results = []
    for horizon, min_p, breadth, agree, flow in grid:
        row = evaluate(train, horizon=horizon, min_p=min_p, breadth_mode=breadth,
                       agree_mode=agree, flow_gate=flow, conf_sizing=True)
        if row["trades"] < 150:  # too few trades to trust
            continue
        row.update(horizon=horizon, min_p=min_p, breadth=breadth, agree=agree, flow=flow)
        results.append(row)

    frame = pd.DataFrame(results).sort_values("sized_total_bps", ascending=False)
    print("\n=== top 12 configs on TRAIN (sized total bps) ===")
    print(frame.head(12).to_string(index=False))

    print("\n=== held-out TEST performance of top 5 ===")
    for _, row in frame.head(5).iterrows():
        out = evaluate(test, horizon=int(row["horizon"]), min_p=row["min_p"],
                       breadth_mode=row["breadth"], agree_mode=row["agree"],
                       flow_gate=row["flow"], conf_sizing=True)
        print(f"h={int(row['horizon'])} p>={row['min_p']} breadth={row['breadth']} "
              f"agree={row['agree']} flow={row['flow']} -> {out}")

    print("\n=== baseline (current production config) train/test ===")
    for name, part in (("train", train), ("test", test)):
        out = evaluate(part, horizon=15, min_p=0.58, breadth_mode="strict",
                       agree_mode="two_of_three", flow_gate=True, conf_sizing=False)
        print(name, out)


if __name__ == "__main__":
    main()
