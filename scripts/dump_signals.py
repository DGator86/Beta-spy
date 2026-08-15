"""Replay the stored tape once and dump per-minute signals to CSV.

Each row carries the factor vector, all horizon forecasts, and the realized
same-session forward SPY return for each horizon, so decision-layer and
sizing variants can be evaluated offline without re-running the engine.
"""
from __future__ import annotations

import argparse
import csv
from datetime import timedelta

from beta_spy.engine import Tape500Engine
from beta_spy.forecast import FEATURE_NAMES
from beta_spy.replay import HistoricalReplay
from beta_spy.storage import Tape500Store
from beta_spy.universe import load_universe_csv

HORIZONS = (5, 15, 30)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--db", required=True)
    parser.add_argument("--universe", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    store = Tape500Store(args.db)
    holdings = load_universe_csv(args.universe)
    engine = Tape500Engine(tuple(holdings), store=None)
    replay = HistoricalReplay(store, engine)

    rows = []
    for index, snapshot in enumerate(replay.run(), start=1):
        spy = next((item for item in snapshot.symbols if item.symbol == "SPY"), None)
        if spy is None or spy.close <= 0:
            continue
        record: dict[str, object] = {
            "timestamp": snapshot.timestamp.isoformat(),
            "spy_price": float(spy.close),
        }
        features = snapshot.factors.feature_dict()
        for name in FEATURE_NAMES:
            record[f"f_{name}"] = features.get(name)
        for forecast in snapshot.forecasts:
            h = forecast.horizon_minutes
            record[f"p_up_{h}"] = forecast.probability_up
            record[f"exp_bps_{h}"] = forecast.expected_return_bps
            record[f"conf_{h}"] = forecast.confidence
            record[f"ready_{h}"] = int(forecast.model_ready)
        record["_ts"] = snapshot.timestamp
        rows.append(record)
        if index % 500 == 0:
            print(f"replayed {index} snapshots", flush=True)

    by_time = {record["_ts"]: record["spy_price"] for record in rows}
    for record in rows:
        ts = record.pop("_ts")
        for h in HORIZONS:
            target = ts + timedelta(minutes=h)
            price = by_time.get(target) if target.date() == ts.date() else None
            record[f"fwd_bps_{h}"] = (
                (price / record["spy_price"] - 1.0) * 10_000.0 if price else None
            )

    fieldnames = list(rows[0].keys())
    with open(args.output, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    print(f"wrote {len(rows)} rows to {args.output}")


if __name__ == "__main__":
    main()
