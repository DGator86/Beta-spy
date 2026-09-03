#!/usr/bin/env python3
from __future__ import annotations

import argparse
from dataclasses import asdict
from datetime import datetime
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss, log_loss
from sklearn.model_selection import TimeSeriesSplit
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from beta_spy.mechanics import MechanicsEstimator
from beta_spy.models import FlowFeatures
from beta_spy.storage import Tape500Store


BASELINE_FEATURES = [
    "velocity_bps",
    "acceleration_bps",
    "force",
    "force_ofi",
    "force_quote",
]
MECHANICS_FEATURES = [
    "upside_inertia",
    "downside_inertia",
    "inertial_bias",
    "momentum",
    "impulse",
]


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Walk-forward kill test for the SPY Market Mechanics MVP."
    )
    parser.add_argument("--database", required=True, type=Path, help="Beta-spy SQLite database")
    parser.add_argument("--horizon", type=int, default=5, help="Future SPY return horizon in minutes")
    parser.add_argument(
        "--windows",
        type=int,
        nargs="+",
        default=[60, 120, 240],
        help="Rolling response-estimation windows to test",
    )
    parser.add_argument("--min-samples", type=int, default=30)
    parser.add_argument("--output", type=Path, default=None, help="Optional JSON report path")
    return parser.parse_args()


def _load_spy_rows(store: Tape500Store) -> list[tuple[datetime, float, FlowFeatures]]:
    sql = """
        SELECT b.timestamp, b.close,
               f.buy_volume, f.sell_volume, f.neutral_volume,
               f.order_flow_imbalance, f.quote_imbalance, f.average_spread_bps,
               f.trade_intensity, f.average_trade_size, f.price_impact_bps_per_10k,
               f.absorption, f.quote_updates, f.trades, f.payload
        FROM minute_bars b
        LEFT JOIN minute_flow f
          ON f.timestamp = b.timestamp AND f.symbol = b.symbol
        WHERE b.symbol = 'SPY'
        ORDER BY b.timestamp
    """
    rows = store.connection.execute(sql).fetchall()
    output: list[tuple[datetime, float, FlowFeatures]] = []
    for row in rows:
        timestamp = datetime.fromisoformat(str(row[0]).replace("Z", "+00:00"))
        close = float(row[1])
        if row[2] is None:
            flow = FlowFeatures()
        else:
            extras: dict[str, object] = {}
            if row[14]:
                try:
                    parsed = json.loads(str(row[14]))
                except json.JSONDecodeError:
                    parsed = {}
                if isinstance(parsed, dict):
                    extras = {
                        key: value
                        for key, value in parsed.items()
                        if key in FlowFeatures.__dataclass_fields__
                    }
            flow = FlowFeatures(
                buy_volume=float(row[2]),
                sell_volume=float(row[3]),
                neutral_volume=float(row[4]),
                order_flow_imbalance=float(row[5]) if row[5] is not None else None,
                quote_imbalance=float(row[6]) if row[6] is not None else None,
                average_spread_bps=float(row[7]) if row[7] is not None else None,
                trade_intensity=float(row[8] or 0.0),
                average_trade_size=float(row[9]) if row[9] is not None else None,
                price_impact_bps_per_10k=float(row[10]) if row[10] is not None else None,
                absorption=float(row[11]) if row[11] is not None else None,
                quote_updates=int(row[12] or 0),
                trades=int(row[13] or 0),
                **{
                    key: value
                    for key, value in extras.items()
                    if key
                    not in {
                        "buy_volume",
                        "sell_volume",
                        "neutral_volume",
                        "order_flow_imbalance",
                        "quote_imbalance",
                        "average_spread_bps",
                        "trade_intensity",
                        "average_trade_size",
                        "price_impact_bps_per_10k",
                        "absorption",
                        "quote_updates",
                        "trades",
                    }
                },
            )
        output.append((timestamp, close, flow))
    return output


def _build_frame(
    rows: list[tuple[datetime, float, FlowFeatures]],
    *,
    window: int,
    min_samples: int,
    horizon: int,
) -> pd.DataFrame:
    estimator = MechanicsEstimator(
        window=window,
        min_samples=min(min_samples, window),
    )
    records: list[dict[str, object]] = []
    closes = np.asarray([row[1] for row in rows], dtype=float)

    for i, (timestamp, close, flow) in enumerate(rows):
        state = estimator.step(timestamp, close, flow)
        if i + horizon >= len(rows):
            continue
        future_close = closes[i + horizon]
        future_return_bps = float(np.log(future_close / close) * 10_000.0)
        record = asdict(state)
        record["close"] = close
        record["future_return_bps"] = future_return_bps
        record["future_up"] = int(future_return_bps > 0.0)
        record["session"] = timestamp.date().isoformat()
        records.append(record)

    frame = pd.DataFrame.from_records(records)
    if frame.empty:
        return frame
    return frame.replace([np.inf, -np.inf], np.nan)


def _safe_spearman(series: pd.Series, other: pd.Series) -> float | None:
    value = series.corr(other, method="spearman")
    return None if pd.isna(value) else float(value)


def _walk_forward(frame: pd.DataFrame, horizon: int) -> dict[str, object]:
    required = BASELINE_FEATURES + MECHANICS_FEATURES + ["future_up"]
    ready = frame.loc[frame["model_ready"] == True, required].dropna().copy()  # noqa: E712
    if len(ready) < 300 or ready["future_up"].nunique() < 2:
        return {
            "status": "INSUFFICIENT_READY_ROWS",
            "ready_rows": int(len(ready)),
        }

    n_splits = 5
    splitter = TimeSeriesSplit(n_splits=n_splits, gap=max(horizon, 1))
    fold_rows: list[dict[str, float]] = []

    for train_idx, test_idx in splitter.split(ready):
        train = ready.iloc[train_idx]
        test = ready.iloc[test_idx]
        if train["future_up"].nunique() < 2 or test["future_up"].nunique() < 2:
            continue

        y_train = train["future_up"].to_numpy(dtype=int)
        y_test = test["future_up"].to_numpy(dtype=int)

        baseline = make_pipeline(
            StandardScaler(),
            LogisticRegression(max_iter=2000, C=0.5),
        )
        augmented = make_pipeline(
            StandardScaler(),
            LogisticRegression(max_iter=2000, C=0.5),
        )

        baseline.fit(train[BASELINE_FEATURES], y_train)
        augmented.fit(train[BASELINE_FEATURES + MECHANICS_FEATURES], y_train)

        p_base = baseline.predict_proba(test[BASELINE_FEATURES])[:, 1]
        p_aug = augmented.predict_proba(test[BASELINE_FEATURES + MECHANICS_FEATURES])[:, 1]

        base_ll = float(log_loss(y_test, p_base, labels=[0, 1]))
        aug_ll = float(log_loss(y_test, p_aug, labels=[0, 1]))
        fold_rows.append(
            {
                "baseline_log_loss": base_ll,
                "mechanics_log_loss": aug_ll,
                "relative_log_loss_improvement": (base_ll - aug_ll) / base_ll,
                "baseline_brier": float(brier_score_loss(y_test, p_base)),
                "mechanics_brier": float(brier_score_loss(y_test, p_aug)),
            }
        )

    if not fold_rows:
        return {
            "status": "INSUFFICIENT_VALID_FOLDS",
            "ready_rows": int(len(ready)),
        }

    improvement = np.asarray(
        [item["relative_log_loss_improvement"] for item in fold_rows],
        dtype=float,
    )
    brier_delta = np.asarray(
        [item["baseline_brier"] - item["mechanics_brier"] for item in fold_rows],
        dtype=float,
    )
    candidate_pass = bool(
        np.median(improvement) >= 0.01
        and np.sum(improvement > 0.0) >= max(1, len(improvement) - 1)
        and np.median(brier_delta) > 0.0
    )
    return {
        "status": "OK",
        "ready_rows": int(len(ready)),
        "folds": fold_rows,
        "median_relative_log_loss_improvement": float(np.median(improvement)),
        "positive_log_loss_folds": int(np.sum(improvement > 0.0)),
        "median_brier_improvement": float(np.median(brier_delta)),
        "candidate_pass": candidate_pass,
    }


def _evaluate_window(
    rows: list[tuple[datetime, float, FlowFeatures]],
    *,
    window: int,
    min_samples: int,
    horizon: int,
) -> dict[str, object]:
    frame = _build_frame(
        rows,
        window=window,
        min_samples=min_samples,
        horizon=horizon,
    )
    if frame.empty:
        return {"window": window, "status": "NO_DATA"}

    eligible = frame.iloc[min(min_samples, len(frame)) :]
    ready = frame.loc[frame["model_ready"] == True].copy()  # noqa: E712
    sessions = int(frame["session"].nunique())

    persistence = None
    if len(ready) > 2:
        persistence = {
            "upside_inertia_lag1_spearman": _safe_spearman(
                ready["upside_inertia"].iloc[1:].reset_index(drop=True),
                ready["upside_inertia"].shift(1).iloc[1:].reset_index(drop=True),
            ),
            "downside_inertia_lag1_spearman": _safe_spearman(
                ready["downside_inertia"].iloc[1:].reset_index(drop=True),
                ready["downside_inertia"].shift(1).iloc[1:].reset_index(drop=True),
            ),
        }

    diagnostics = {
        "window": window,
        "rows": int(len(frame)),
        "sessions": sessions,
        "model_ready_rows": int(len(ready)),
        "model_ready_coverage_after_warmup": float(len(ready) / max(len(eligible), 1)),
        "inertial_bias_future_return_spearman": (
            _safe_spearman(ready["inertial_bias"], ready["future_return_bps"])
            if len(ready) > 2
            else None
        ),
        "persistence": persistence,
        "walk_forward": _walk_forward(frame, horizon),
    }
    return diagnostics


def main() -> int:
    args = _parse_args()
    if args.horizon <= 0:
        raise SystemExit("--horizon must be > 0")

    store = Tape500Store(args.database)
    try:
        rows = _load_spy_rows(store)
    finally:
        store.close()

    report: dict[str, object] = {
        "protocol": "MARKET_MECHANICS_MVP_V1",
        "database": str(args.database),
        "horizon_minutes": args.horizon,
        "spy_minutes": len(rows),
        "windows": [],
    }
    for window in args.windows:
        report["windows"].append(
            _evaluate_window(
                rows,
                window=window,
                min_samples=args.min_samples,
                horizon=args.horizon,
            )
        )

    successful = [
        item
        for item in report["windows"]
        if isinstance(item, dict)
        and isinstance(item.get("walk_forward"), dict)
        and item["walk_forward"].get("status") == "OK"
    ]
    passes = [
        bool(item["walk_forward"].get("candidate_pass"))
        for item in successful
    ]
    report["robustness_summary"] = {
        "successful_windows": len(successful),
        "candidate_pass_windows": int(sum(passes)),
        "robust_candidate": bool(len(successful) >= 2 and sum(passes) >= 2),
        "interpretation": (
            "Candidate only. A robust_candidate=true result authorizes deeper research, "
            "not trading or production promotion."
        ),
    }

    text = json.dumps(report, indent=2, sort_keys=True)
    print(text)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
