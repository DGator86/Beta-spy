"""Combinatorial purged cross-validation (CPCV) with a PBO estimate.

One train/test split can be lucky; parameters chosen on it can be fitted
noise. This harness groups session days into contiguous blocks and forms
every balanced combination of blocks into a train/test partition, scoring
every candidate decision config on both sides of each partition.

Two outputs guard against overfitting:

- PBO (probability of backtest overfitting, Lopez de Prado): across all
  partitions, how often does the in-sample winner rank below the median
  config out-of-sample? Near 0.5 means selection is picking noise; near 0
  means the edge generalizes.
- Robust selection table: each config's median and 10th-percentile
  out-of-sample t-statistic across all partitions. The recommended config
  maximizes the median while never collapsing in its worst windows —
  stability across every window beats a peak in one.

Leakage control: forward returns in the signal dump never cross a session
boundary, so splitting on whole days purges the 5/15/30-minute outcome
overlap between adjacent train and test data.
"""
from __future__ import annotations

import argparse
import itertools
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from sweep_decision import evaluate  # noqa: E402


def config_grid() -> list[dict]:
    grid = []
    for horizon, min_p, breadth, agree, flow in itertools.product(
        (5, 15),
        (0.55, 0.58, 0.60, 0.62, 0.65),
        ("strict", "two"),
        ("two_of_three", "fast_pair"),
        (True, False),
    ):
        grid.append(
            {
                "horizon": horizon,
                "min_p": min_p,
                "breadth_mode": breadth,
                "agree_mode": agree,
                "flow_gate": flow,
                "conf_sizing": True,
            }
        )
    return grid


PRODUCTION = {
    "horizon": 15,
    "min_p": 0.60,
    "breadth_mode": "strict",
    "agree_mode": "fast_pair",
    "flow_gate": True,
    "conf_sizing": True,
}


def label(cfg: dict) -> str:
    return (
        f"h={cfg['horizon']} p>={cfg['min_p']:.2f} breadth={cfg['breadth_mode']} "
        f"agree={cfg['agree_mode']} flow={'on' if cfg['flow_gate'] else 'off'}"
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--signals", required=True)
    parser.add_argument("--blocks", type=int, default=8)
    parser.add_argument("--min-train-trades", type=int, default=40)
    parser.add_argument("--min-test-trades", type=int, default=10)
    args = parser.parse_args()

    df = pd.read_csv(args.signals, parse_dates=["timestamp"])
    df["date"] = df["timestamp"].dt.date.astype(str)
    dates = sorted(df["date"].unique())
    blocks = [list(chunk) for chunk in np.array_split(np.array(dates), args.blocks)]
    print(f"sessions={len(dates)} blocks={len(blocks)} sizes={[len(b) for b in blocks]}")

    grid = config_grid()
    if not any(cfg == PRODUCTION for cfg in grid):
        grid.append(dict(PRODUCTION))
    combos = list(itertools.combinations(range(len(blocks)), len(blocks) // 2))
    print(f"configs={len(grid)} partitions={len(combos)}")

    frames = {date: df[df["date"] == date] for date in dates}

    def subset(block_ids: tuple[int, ...]) -> pd.DataFrame:
        keep = [date for index in block_ids for date in blocks[index]]
        return pd.concat([frames[date] for date in keep], ignore_index=True)

    oos_scores: dict[int, list[float]] = {index: [] for index in range(len(grid))}
    oos_means: dict[int, list[float]] = {index: [] for index in range(len(grid))}
    oos_trades: dict[int, list[int]] = {index: [] for index in range(len(grid))}
    below_median = 0
    usable_partitions = 0

    for combo in combos:
        rest = tuple(sorted(set(range(len(blocks))) - set(combo)))
        train, test = subset(combo), subset(rest)
        train_scores: list[float] = []
        test_scores: list[float] = []
        for index, cfg in enumerate(grid):
            row_train = evaluate(train, **cfg)
            row_test = evaluate(test, **cfg)
            ok_train = row_train["trades"] >= args.min_train_trades
            ok_test = row_test["trades"] >= args.min_test_trades
            train_scores.append(row_train.get("t_stat", 0.0) if ok_train else -np.inf)
            score = row_test.get("t_stat", 0.0) if ok_test else 0.0
            test_scores.append(score)
            oos_scores[index].append(score)
            oos_means[index].append(row_test.get("sized_mean_bps", 0.0) if ok_test else 0.0)
            oos_trades[index].append(row_test.get("trades", 0))
        if np.all(np.isneginf(train_scores)):
            continue
        usable_partitions += 1
        winner = int(np.argmax(train_scores))
        rank = float(np.mean(np.array(test_scores) < test_scores[winner]))
        if rank < 0.5:
            below_median += 1

    pbo = below_median / usable_partitions if usable_partitions else float("nan")
    print(f"\nPBO (in-sample winner below OOS median): {pbo:.2%} "
          f"over {usable_partitions} partitions")

    rows = []
    for index, cfg in enumerate(grid):
        scores = np.array(oos_scores[index])
        rows.append(
            {
                "config": label(cfg),
                "median_oos_t": float(np.median(scores)),
                "p10_oos_t": float(np.percentile(scores, 10)),
                "median_oos_mean_bps": float(np.median(oos_means[index])),
                "median_oos_trades": float(np.median(oos_trades[index])),
                "production": cfg == PRODUCTION,
            }
        )
    table = pd.DataFrame(rows).sort_values("median_oos_t", ascending=False)
    print("\n=== configs ranked by MEDIAN out-of-sample t-stat across partitions ===")
    print(table.head(15).to_string(index=False))
    print("\n=== production config ===")
    print(table[table["production"]].to_string(index=False))


if __name__ == "__main__":
    main()
