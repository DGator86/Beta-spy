"""Time-of-day robustness analysis for the production config.

Buckets every production-config trade by session half-hour and reports the
per-bucket per-trade quality on each of the CPCV day-block partitions. A
bucket only deserves a hard gate if it is bad in (nearly) every partition;
one bad week in one bucket is noise, not signal.
"""
from __future__ import annotations

import argparse
import itertools
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from cscv_eval import PRODUCTION  # noqa: E402
from sweep_decision import BREADTH_COLS, _sign  # noqa: E402


def trade_mask(df: pd.DataFrame) -> tuple[pd.Series, pd.Series]:
    cfg = PRODUCTION
    horizon = cfg["horizon"]
    p = df[f"p_up_{horizon}"]
    direction = pd.Series(0, index=df.index, dtype=int)
    direction[p >= cfg["min_p"]] = 1
    direction[p <= 1.0 - cfg["min_p"]] = -1
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
    fwd = df[f"fwd_bps_{horizon}"]
    trades = (direction != 0) & breadth_ok & agree_ok & liquidity_ok & coverage_ok & fwd.notna()
    return trades, fwd * direction


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--signals", required=True)
    parser.add_argument("--blocks", type=int, default=8)
    args = parser.parse_args()

    df = pd.read_csv(args.signals, parse_dates=["timestamp"])
    df["date"] = df["timestamp"].dt.date.astype(str)
    eastern = df["timestamp"].dt.tz_convert("America/New_York")
    minutes = eastern.dt.hour * 60 + eastern.dt.minute - (9 * 60 + 30)
    df["bucket"] = (minutes // 30).clip(0, 12)

    trades, realized = trade_mask(df)
    dates = sorted(df["date"].unique())
    blocks = [list(chunk) for chunk in np.array_split(np.array(dates), args.blocks)]
    combos = list(itertools.combinations(range(len(blocks)), len(blocks) // 2))

    rows = []
    for bucket in sorted(df.loc[trades, "bucket"].unique()):
        in_bucket = trades & (df["bucket"] == bucket)
        per_partition = []
        for combo in combos:
            keep_dates = {d for i in combo for d in blocks[i]}
            mask = in_bucket & df["date"].isin(keep_dates)
            if mask.sum() >= 5:
                per_partition.append(float(realized[mask].mean()))
        if not per_partition:
            continue
        arr = np.array(per_partition)
        start = 9 * 60 + 30 + int(bucket) * 30
        rows.append(
            {
                "window_et": f"{start // 60:02d}:{start % 60:02d}",
                "trades": int(in_bucket.sum()),
                "mean_bps": float(realized[in_bucket].mean()),
                "partition_median_bps": float(np.median(arr)),
                "pct_partitions_negative": float((arr < 0).mean()),
            }
        )
    table = pd.DataFrame(rows)
    print(table.to_string(index=False))


if __name__ == "__main__":
    main()
