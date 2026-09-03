#!/usr/bin/env python3
from __future__ import annotations

import argparse
from dataclasses import asdict
from datetime import UTC, datetime, time, timedelta
import json
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss, log_loss
from sklearn.model_selection import TimeSeriesSplit
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from beta_spy.flow import FlowAccumulator
from beta_spy.mechanics import MechanicsEstimator
from beta_spy.models import FlowFeatures, QuoteTop, TradePrint
from beta_spy.storage import Tape500Store


ET = ZoneInfo("America/New_York")
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
_CORE_FLOW_FIELDS = {
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


def _stamp(value: object) -> datetime:
    parsed = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
    return parsed if parsed.tzinfo is not None else parsed.replace(tzinfo=UTC)


def _minute(value: datetime) -> datetime:
    return value.astimezone(UTC).replace(second=0, microsecond=0)


def _is_rth(value: datetime) -> bool:
    local = value.astimezone(ET)
    clock = local.time().replace(tzinfo=None)
    return time(9, 30) <= clock < time(16, 0)


def _flow_from_joined_row(row: tuple) -> FlowFeatures:
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
                if key in FlowFeatures.__dataclass_fields__ and key not in _CORE_FLOW_FIELDS
            }
    return FlowFeatures(
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
        **extras,
    )


def _saved_flow_rows(store: Tape500Store) -> list[tuple[datetime, float, FlowFeatures]]:
    sql = """
        SELECT b.timestamp, b.close,
               f.buy_volume, f.sell_volume, f.neutral_volume,
               f.order_flow_imbalance, f.quote_imbalance, f.average_spread_bps,
               f.trade_intensity, f.average_trade_size, f.price_impact_bps_per_10k,
               f.absorption, f.quote_updates, f.trades, f.payload
        FROM minute_bars b
        JOIN minute_flow f
          ON f.timestamp = b.timestamp AND f.symbol = b.symbol
        WHERE b.symbol = 'SPY' AND f.symbol = 'SPY'
        ORDER BY b.timestamp
    """
    output: list[tuple[datetime, float, FlowFeatures]] = []
    for row in store.connection.execute(sql).fetchall():
        timestamp = _stamp(row[0])
        if _is_rth(timestamp):
            output.append((timestamp, float(row[1]), _flow_from_joined_row(row)))
    return output


def _raw_flow_rows(store: Tape500Store) -> list[tuple[datetime, float, FlowFeatures]]:
    """Reconstruct minute SPY flow causally from stored raw quotes and prints.

    Beta historically retained raw SPY tape even when a minute_flow row was not
    written. This fallback deliberately reuses the production FlowAccumulator so
    the research force is derived from the same aggressor/quote logic.
    """

    bar_rows = store.connection.execute(
        "SELECT timestamp,close FROM minute_bars WHERE symbol='SPY' ORDER BY timestamp"
    ).fetchall()
    bars = {
        _minute(timestamp): float(close)
        for raw, close in bar_rows
        if _is_rth(timestamp := _stamp(raw))
    }
    if not bars:
        return []

    events = store.connection.execute(
        """
        SELECT timestamp, kind, price, size, bid, ask, bid_size, ask_size, sequence
        FROM (
            SELECT timestamp, 0 AS kind, NULL AS price, NULL AS size,
                   bid, ask, bid_size, ask_size, NULL AS sequence
            FROM spy_quotes
            UNION ALL
            SELECT timestamp, 1 AS kind, price, size,
                   bid, ask, NULL AS bid_size, NULL AS ask_size, sequence
            FROM spy_trades
        )
        ORDER BY timestamp, kind, sequence
        """
    ).fetchall()

    flow_by_minute: dict[datetime, FlowFeatures] = {}
    accumulator: FlowAccumulator | None = None
    current_minute: datetime | None = None
    current_day = None

    def flush() -> None:
        if accumulator is not None and current_minute is not None and current_minute in bars:
            flow = accumulator.snapshot(window_seconds=60.0, now=current_minute + timedelta(minutes=1))
            if flow.trades > 0 or flow.quote_updates > 0:
                flow_by_minute[current_minute] = flow

    for raw in events:
        timestamp = _stamp(raw[0])
        if not _is_rth(timestamp):
            continue
        minute = _minute(timestamp)
        if minute not in bars:
            # Still process off-grid events only when they are within an RTH
            # minute represented by the price substrate. Missing price minutes
            # are treated as missing observations, not synthetic rows.
            continue
        local_day = timestamp.astimezone(ET).date()
        if accumulator is None or current_day != local_day:
            flush()
            accumulator = FlowAccumulator()
            current_minute = minute
            current_day = local_day
        elif current_minute != minute:
            flush()
            accumulator.reset()
            current_minute = minute

        if int(raw[1]) == 0:
            bid = float(raw[4] or 0.0)
            ask = float(raw[5] or 0.0)
            if bid > 0 and ask >= bid:
                accumulator.on_quote(
                    QuoteTop(
                        symbol="SPY",
                        timestamp=timestamp,
                        bid=bid,
                        ask=ask,
                        bid_size=float(raw[6]) if raw[6] is not None else None,
                        ask_size=float(raw[7]) if raw[7] is not None else None,
                    )
                )
        else:
            price = float(raw[2] or 0.0)
            size = float(raw[3] or 0.0)
            if price > 0 and size > 0:
                accumulator.on_trade(
                    TradePrint(
                        symbol="SPY",
                        timestamp=timestamp,
                        price=price,
                        size=size,
                        bid=float(raw[4]) if raw[4] is not None else None,
                        ask=float(raw[5]) if raw[5] is not None else None,
                        sequence=int(raw[8]) if raw[8] is not None else None,
                    )
                )
    flush()

    return [
        (timestamp, bars[timestamp], flow_by_minute[timestamp])
        for timestamp in sorted(flow_by_minute)
        if timestamp in bars
    ]


def _load_spy_rows(
    store: Tape500Store,
) -> tuple[list[tuple[datetime, float, FlowFeatures]], dict[str, object]]:
    bar_count = int(
        store.connection.execute("SELECT COUNT(*) FROM minute_bars WHERE symbol='SPY'").fetchone()[0]
    )
    saved_flow_count = int(
        store.connection.execute("SELECT COUNT(*) FROM minute_flow WHERE symbol='SPY'").fetchone()[0]
    )
    raw_trade_count = int(store.connection.execute("SELECT COUNT(*) FROM spy_trades").fetchone()[0])
    raw_quote_count = int(store.connection.execute("SELECT COUNT(*) FROM spy_quotes").fetchone()[0])

    saved = _saved_flow_rows(store) if saved_flow_count else []
    if saved:
        rows = saved
        source = "minute_flow"
    elif raw_trade_count or raw_quote_count:
        rows = _raw_flow_rows(store)
        source = "raw_spy_trades_quotes"
    else:
        rows = []
        source = "none"

    diagnostics = {
        "flow_source": source,
        "spy_bar_rows_all_sessions": bar_count,
        "saved_spy_minute_flow_rows": saved_flow_count,
        "raw_spy_trade_rows": raw_trade_count,
        "raw_spy_quote_rows": raw_quote_count,
        "usable_rth_price_flow_minutes": len(rows),
        "usable_sessions": len({stamp.astimezone(ET).date() for stamp, _, _ in rows}),
    }
    return rows, diagnostics


def _build_frame(
    rows: list[tuple[datetime, float, FlowFeatures]],
    *,
    window: int,
    min_samples: int,
    horizon: int,
) -> pd.DataFrame:
    estimator = MechanicsEstimator(window=window, min_samples=min(min_samples, window))
    records: list[dict[str, object]] = []
    price_by_time = {timestamp: close for timestamp, close, _ in rows}

    for timestamp, close, flow in rows:
        state = estimator.step(timestamp, close, flow)
        future_close = price_by_time.get(timestamp + timedelta(minutes=horizon))
        future_return_bps: float | None = None
        future_up: int | None = None
        if future_close is not None and timestamp.date() == (timestamp + timedelta(minutes=horizon)).date():
            future_return_bps = float(np.log(future_close / close) * 10_000.0)
            future_up = int(future_return_bps > 0.0)

        record = asdict(state)
        record["close"] = close
        record["future_return_bps"] = future_return_bps
        record["future_up"] = future_up
        record["session"] = timestamp.astimezone(ET).date().isoformat()
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
        return {"status": "INSUFFICIENT_READY_ROWS", "ready_rows": int(len(ready))}

    splitter = TimeSeriesSplit(n_splits=5, gap=max(horizon, 1))
    fold_rows: list[dict[str, float]] = []

    for train_idx, test_idx in splitter.split(ready):
        train = ready.iloc[train_idx]
        test = ready.iloc[test_idx]
        if train["future_up"].nunique() < 2 or test["future_up"].nunique() < 2:
            continue
        y_train = train["future_up"].to_numpy(dtype=int)
        y_test = test["future_up"].to_numpy(dtype=int)
        baseline = make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000, C=0.5))
        augmented = make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000, C=0.5))
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
        return {"status": "INSUFFICIENT_VALID_FOLDS", "ready_rows": int(len(ready))}

    improvement = np.asarray([item["relative_log_loss_improvement"] for item in fold_rows], dtype=float)
    brier_delta = np.asarray(
        [item["baseline_brier"] - item["mechanics_brier"] for item in fold_rows], dtype=float
    )
    candidate_pass = bool(
        np.median(improvement) >= 0.01
        and len(fold_rows) == 5
        and np.sum(improvement > 0.0) >= 4
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
    frame = _build_frame(rows, window=window, min_samples=min_samples, horizon=horizon)
    if frame.empty:
        return {"window": window, "status": "NO_DATA"}

    eligible = frame.loc[frame["sample_count"] >= min_samples].copy()
    ready = eligible.loc[eligible["model_ready"] == True].copy()  # noqa: E712
    valid_up = eligible["upside_inertia"].notna()
    valid_down = eligible["downside_inertia"].notna()
    labeled_ready = ready.dropna(subset=["future_return_bps"])

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

    return {
        "window": window,
        "rows": int(len(frame)),
        "sessions": int(frame["session"].nunique()),
        "post_warmup_rows": int(len(eligible)),
        "valid_upside_inertia_rows": int(valid_up.sum()),
        "valid_downside_inertia_rows": int(valid_down.sum()),
        "model_ready_rows": int(len(ready)),
        "model_ready_labeled_rows": int(len(labeled_ready)),
        "upside_estimability_after_warmup": float(valid_up.mean()) if len(eligible) else 0.0,
        "downside_estimability_after_warmup": float(valid_down.mean()) if len(eligible) else 0.0,
        "two_sided_estimability_after_warmup": float(len(ready) / max(len(eligible), 1)),
        "inertial_bias_future_return_spearman": (
            _safe_spearman(labeled_ready["inertial_bias"], labeled_ready["future_return_bps"])
            if len(labeled_ready) > 2
            else None
        ),
        "persistence": persistence,
        "walk_forward": _walk_forward(frame, horizon),
    }


def main() -> int:
    args = _parse_args()
    if args.horizon <= 0:
        raise SystemExit("--horizon must be > 0")

    store = Tape500Store(args.database)
    try:
        rows, input_diagnostics = _load_spy_rows(store)
    finally:
        store.close()

    report: dict[str, object] = {
        "protocol": "MARKET_MECHANICS_MVP_V1",
        "database": str(args.database),
        "horizon_minutes": args.horizon,
        "input": input_diagnostics,
        "windows": [],
    }
    for window in args.windows:
        report["windows"].append(
            _evaluate_window(rows, window=window, min_samples=args.min_samples, horizon=args.horizon)
        )

    successful = [
        item
        for item in report["windows"]
        if isinstance(item, dict)
        and isinstance(item.get("walk_forward"), dict)
        and item["walk_forward"].get("status") == "OK"
    ]
    passes = [bool(item["walk_forward"].get("candidate_pass")) for item in successful]
    report["robustness_summary"] = {
        "successful_windows": len(successful),
        "candidate_pass_windows": int(sum(passes)),
        "robust_candidate": bool(len(successful) >= 2 and sum(passes) >= 2),
        "interpretation": (
            "Candidate only. robust_candidate=true authorizes deeper research, not trading or production promotion."
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
