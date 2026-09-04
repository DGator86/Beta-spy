#!/usr/bin/env python3
from __future__ import annotations

import argparse
from dataclasses import asdict
from datetime import timedelta
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss, log_loss
from sklearn.model_selection import TimeSeriesSplit
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from beta_spy.mechanics_v2 import MechanicsV2Estimator
from beta_spy.models import FlowFeatures
from beta_spy.storage import Tape500Store
from market_mechanics_research import ET, _load_spy_rows


BASELINE_FEATURES = [
    "abs_velocity_bps",
    "aligned_acceleration_bps",
    "aligned_force",
    "opposing_force_magnitude",
    "aligned_ofi",
    "aligned_quote",
]
PRIMARY_V2_FEATURES = ["active_braking_inertia"]
SECONDARY_V2_FEATURES = ["active_launch_inertia", "brake_launch_ratio"]


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Frozen walk-forward kill test for Market Mechanics V2 braking inertia."
    )
    parser.add_argument("--database", required=True, type=Path, help="Beta-spy SQLite database")
    parser.add_argument("--horizon", type=int, default=5)
    parser.add_argument("--windows", type=int, nargs="+", default=[60, 120, 240])
    parser.add_argument("--min-quadrant-samples", type=int, default=12)
    parser.add_argument("--output", type=Path, default=None)
    return parser.parse_args()


def _aligned_features(state: object) -> dict[str, float | None]:
    velocity = getattr(state, "velocity_bps")
    acceleration = getattr(state, "acceleration_bps")
    force = float(getattr(state, "force"))
    force_ofi = float(getattr(state, "force_ofi"))
    force_quote = float(getattr(state, "force_quote"))

    if velocity is None or velocity == 0.0:
        return {
            "current_direction": None,
            "abs_velocity_bps": None,
            "aligned_acceleration_bps": None,
            "aligned_force": None,
            "opposing_force_magnitude": None,
            "aligned_ofi": None,
            "aligned_quote": None,
        }

    direction = 1.0 if velocity > 0.0 else -1.0
    aligned_force = force * direction
    return {
        "current_direction": direction,
        "abs_velocity_bps": abs(float(velocity)),
        "aligned_acceleration_bps": (
            None if acceleration is None else float(acceleration) * direction
        ),
        "aligned_force": aligned_force,
        "opposing_force_magnitude": max(-aligned_force, 0.0),
        "aligned_ofi": force_ofi * direction,
        "aligned_quote": force_quote * direction,
    }


def _build_frame(
    rows: list[tuple[object, float, FlowFeatures]],
    *,
    window: int,
    min_quadrant_samples: int,
    horizon: int,
) -> pd.DataFrame:
    estimator = MechanicsV2Estimator(
        window=window,
        min_quadrant_samples=min(min_quadrant_samples, window),
    )
    price_by_time = {timestamp: close for timestamp, close, _ in rows}
    records: list[dict[str, object]] = []

    for timestamp, close, flow in rows:
        state = estimator.step(timestamp, close, flow)
        record = asdict(state)
        aligned = _aligned_features(state)
        record.update(aligned)
        record["close"] = close
        record["session"] = timestamp.astimezone(ET).date().isoformat()

        future_close = price_by_time.get(timestamp + timedelta(minutes=horizon))
        future_return_bps: float | None = None
        continuation: int | None = None
        direction = aligned["current_direction"]
        if future_close is not None and direction is not None:
            future_return_bps = float(np.log(future_close / close) * 10_000.0)
            continuation = int(future_return_bps * float(direction) > 0.0)

        record["future_return_bps"] = future_return_bps
        record["continuation"] = continuation
        records.append(record)

    frame = pd.DataFrame.from_records(records)
    if frame.empty:
        return frame
    return frame.replace([np.inf, -np.inf], np.nan)


def _fit_walk_forward(
    ready: pd.DataFrame,
    *,
    features: list[str],
    horizon: int,
) -> dict[str, object]:
    required = BASELINE_FEATURES + features + ["continuation"]
    sample = ready[required].dropna().copy()
    if len(sample) < 300 or sample["continuation"].nunique() < 2:
        return {"status": "INSUFFICIENT_READY_ROWS", "ready_rows": int(len(sample))}

    splitter = TimeSeriesSplit(n_splits=5, gap=max(horizon, 1))
    folds: list[dict[str, float]] = []

    for train_idx, test_idx in splitter.split(sample):
        train = sample.iloc[train_idx]
        test = sample.iloc[test_idx]
        if train["continuation"].nunique() < 2 or test["continuation"].nunique() < 2:
            continue

        y_train = train["continuation"].to_numpy(dtype=int)
        y_test = test["continuation"].to_numpy(dtype=int)
        baseline = make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000, C=0.5))
        augmented = make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000, C=0.5))
        baseline.fit(train[BASELINE_FEATURES], y_train)
        augmented.fit(train[BASELINE_FEATURES + features], y_train)

        p_base = baseline.predict_proba(test[BASELINE_FEATURES])[:, 1]
        p_aug = augmented.predict_proba(test[BASELINE_FEATURES + features])[:, 1]
        base_ll = float(log_loss(y_test, p_base, labels=[0, 1]))
        aug_ll = float(log_loss(y_test, p_aug, labels=[0, 1]))
        folds.append(
            {
                "baseline_log_loss": base_ll,
                "v2_log_loss": aug_ll,
                "relative_log_loss_improvement": (base_ll - aug_ll) / base_ll,
                "baseline_brier": float(brier_score_loss(y_test, p_base)),
                "v2_brier": float(brier_score_loss(y_test, p_aug)),
            }
        )

    if not folds:
        return {"status": "INSUFFICIENT_VALID_FOLDS", "ready_rows": int(len(sample))}

    improvement = np.asarray([fold["relative_log_loss_improvement"] for fold in folds], dtype=float)
    brier_delta = np.asarray(
        [fold["baseline_brier"] - fold["v2_brier"] for fold in folds],
        dtype=float,
    )
    candidate_pass = bool(
        len(folds) == 5
        and np.median(improvement) >= 0.01
        and np.sum(improvement > 0.0) >= 4
        and np.median(brier_delta) > 0.0
    )
    return {
        "status": "OK",
        "ready_rows": int(len(sample)),
        "folds": folds,
        "median_relative_log_loss_improvement": float(np.median(improvement)),
        "positive_log_loss_folds": int(np.sum(improvement > 0.0)),
        "median_brier_improvement": float(np.median(brier_delta)),
        "candidate_pass": candidate_pass,
    }


def _tercile_diagnostic(frame: pd.DataFrame) -> dict[str, object]:
    sample = frame[["active_braking_inertia", "continuation"]].dropna().copy()
    if len(sample) < 30:
        return {"status": "INSUFFICIENT_ROWS", "rows": int(len(sample))}
    try:
        sample["tercile"] = pd.qcut(
            sample["active_braking_inertia"],
            q=3,
            labels=["low", "mid", "high"],
            duplicates="drop",
        )
    except ValueError:
        return {"status": "INSUFFICIENT_VARIATION", "rows": int(len(sample))}

    rates = sample.groupby("tercile", observed=True)["continuation"].mean()
    return {
        "status": "OK",
        "rows": int(len(sample)),
        "continuation_rate_by_tercile": {
            str(index): float(value) for index, value in rates.items()
        },
    }


def _coverage(frame: pd.DataFrame) -> dict[str, float | int]:
    if frame.empty:
        return {"eligible_rows": 0}

    total_samples = (
        frame["pp_samples"]
        + frame["pm_samples"]
        + frame["mp_samples"]
        + frame["mm_samples"]
    )
    eligible = frame.loc[total_samples >= 30].copy()
    n = len(eligible)
    if n == 0:
        return {"eligible_rows": 0}

    def fraction(column: str) -> float:
        return float(eligible[column].notna().mean())

    return {
        "eligible_rows": int(n),
        "beta_pp_valid_fraction": fraction("beta_pp"),
        "beta_pm_valid_fraction": fraction("beta_pm"),
        "beta_mp_valid_fraction": fraction("beta_mp"),
        "beta_mm_valid_fraction": fraction("beta_mm"),
        "uptrend_braking_inertia_fraction": fraction("braking_inertia_up"),
        "downtrend_braking_inertia_fraction": fraction("braking_inertia_down"),
        "active_braking_inertia_fraction": fraction("active_braking_inertia"),
        "full_quadrant_ready_fraction": float(eligible["full_quadrant_ready"].mean()),
    }


def _evaluate_window(
    rows: list[tuple[object, float, FlowFeatures]],
    *,
    window: int,
    min_quadrant_samples: int,
    horizon: int,
) -> dict[str, object]:
    frame = _build_frame(
        rows,
        window=window,
        min_quadrant_samples=min_quadrant_samples,
        horizon=horizon,
    )
    if frame.empty:
        return {"window": window, "status": "NO_DATA"}

    primary_ready = frame.loc[frame["active_braking_ready"] == True].copy()  # noqa: E712
    secondary_ready = frame.loc[
        frame["active_braking_ready"] == True,  # noqa: E712
    ].copy()

    return {
        "window": window,
        "status": "OK",
        "rows": int(len(frame)),
        "coverage": _coverage(frame),
        "braking_terciles": _tercile_diagnostic(primary_ready),
        "primary_braking_only": _fit_walk_forward(
            primary_ready,
            features=PRIMARY_V2_FEATURES,
            horizon=horizon,
        ),
        "secondary_launch_brake": _fit_walk_forward(
            secondary_ready,
            features=PRIMARY_V2_FEATURES + SECONDARY_V2_FEATURES,
            horizon=horizon,
        ),
    }


def main() -> int:
    args = _parse_args()
    if args.horizon <= 0:
        raise SystemExit("--horizon must be > 0")
    if args.min_quadrant_samples < 6:
        raise SystemExit("--min-quadrant-samples must be >= 6")

    store = Tape500Store(args.database)
    try:
        rows, input_diagnostics = _load_spy_rows(store)
    finally:
        store.close()

    report: dict[str, object] = {
        "protocol": "MARKET_MECHANICS_V2_BRAKING_FROZEN",
        "parent": "MARKET_MECHANICS_MVP_V1_REJECTED",
        "database": str(args.database),
        "horizon_minutes": args.horizon,
        "min_quadrant_samples": args.min_quadrant_samples,
        "input": input_diagnostics,
        "windows": [],
    }

    for window in args.windows:
        report["windows"].append(
            _evaluate_window(
                rows,
                window=window,
                min_quadrant_samples=args.min_quadrant_samples,
                horizon=args.horizon,
            )
        )

    successful = []
    for item in report["windows"]:
        if not isinstance(item, dict):
            continue
        primary = item.get("primary_braking_only")
        if isinstance(primary, dict) and primary.get("status") == "OK":
            successful.append(primary)

    passes = [bool(item.get("candidate_pass")) for item in successful]
    report["robustness_summary"] = {
        "successful_windows": len(successful),
        "candidate_pass_windows": int(sum(passes)),
        "robust_candidate": bool(len(successful) >= 2 and sum(passes) >= 2),
        "interpretation": (
            "robust_candidate=true authorizes deeper braking-inertia research only; "
            "it does not authorize Delta integration or execution."
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
