from __future__ import annotations

import math
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Any

import numpy as np
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.preprocessing import StandardScaler

HGB_DIRECTION_MODEL_VERSION = "beta-spy-v2-hgb-trailing-1"
HORIZON_MINUTES = 15
SIGNAL_GRID_MINUTES = 5
MIN_TRAINING_SESSIONS = 5
MIN_TRAINING_SAMPLES = 200
MIN_SIGNAL_STRENGTH = 0.35

_BREADTH_HORIZONS = (1, 2, 5, 10, 15, 30)
_SPY_RETURN_HORIZONS = (1, 2, 3, 5, 10, 15, 20, 30, 60)
_SPY_RV_HORIZONS = (5, 10, 15, 30, 60)


def _bar_map(state: Any, day) -> dict[datetime, float]:
    return {
        bar.timestamp.replace(second=0, microsecond=0): float(bar.close)
        for bar in state.bars
        if bar.timestamp.date() == day and float(bar.close) > 0
    }


def _price_maps(states: dict[str, Any], day) -> dict[str, dict[datetime, float]]:
    """Materialize each symbol once per signal timestamp.

    The first prototype rebuilt 500 price dictionaries once for every breadth
    horizon. That was exact but wasteful. This preserves identical features while
    reducing the hot-path work by roughly the number of breadth horizons.
    """
    return {symbol: _bar_map(state, day) for symbol, state in states.items()}


def _exact_return(prices: dict[datetime, float], reference: datetime, periods: int) -> float | None:
    current = prices.get(reference)
    previous = prices.get(reference - timedelta(minutes=periods))
    if current is None or previous is None or previous <= 0:
        return None
    return float(current / previous - 1.0)


def _realized_vol(prices: dict[datetime, float], reference: datetime, minutes: int) -> float | None:
    values: list[float] = []
    for offset in range(minutes, -1, -1):
        value = prices.get(reference - timedelta(minutes=offset))
        if value is None or value <= 0:
            return None
        values.append(value)
    returns = np.diff(np.log(np.asarray(values, dtype=float)))
    if len(returns) < 2:
        return 0.0
    return float(np.std(returns, ddof=1) * math.sqrt(max(minutes - 1, 1)))


def _breadth_stats(
    price_maps: dict[str, dict[datetime, float]], reference: datetime, horizon: int
) -> dict[str, float] | None:
    values: list[float] = []
    for symbol, prices in price_maps.items():
        if symbol == "SPY":
            continue
        value = _exact_return(prices, reference, horizon)
        if value is not None and math.isfinite(value):
            values.append(value)
    if len(values) < 100:
        return None
    array = np.asarray(values, dtype=float)
    return {
        "pos": float(np.mean(array > 0.0)),
        "mean": float(np.mean(array)),
        "med": float(np.median(array)),
        "disp": float(np.std(array, ddof=0)),
        "p10": float(np.quantile(array, 0.10)),
        "p90": float(np.quantile(array, 0.90)),
    }


def trailing_feature_vectors(states: dict[str, Any], timestamp: datetime) -> tuple[np.ndarray, np.ndarray] | None:
    """Build the exact leakage-resistant feature family validated in research.

    `Tape500Engine.build_snapshot()` flushes bars strictly before `timestamp`, so
    the feature reference is timestamp-1 minute. Returns are matched by exact
    minute timestamp: a missing constituent minute is excluded instead of being
    silently treated as a longer return interval.
    """
    reference = timestamp.replace(second=0, microsecond=0) - timedelta(minutes=1)
    day = reference.date()
    price_maps = _price_maps(states, day)
    spy = price_maps.get("SPY") or {}
    if reference not in spy:
        return None

    spy_returns: dict[int, float] = {}
    for horizon in _SPY_RETURN_HORIZONS:
        value = _exact_return(spy, reference, horizon)
        if value is None:
            return None
        spy_returns[horizon] = value

    spy_rv: dict[int, float] = {}
    for horizon in _SPY_RV_HORIZONS:
        value = _realized_vol(spy, reference, horizon)
        if value is None:
            return None
        spy_rv[horizon] = max(value, 1e-8)

    breadth: dict[int, dict[str, float]] = {}
    for horizon in _BREADTH_HORIZONS:
        stats = _breadth_stats(price_maps, reference, horizon)
        if stats is None:
            return None
        breadth[horizon] = stats

    mom_accel_1v5 = spy_returns[1] - spy_returns[5] / 5.0
    mom_accel_5v15 = spy_returns[5] / 5.0 - spy_returns[15] / 15.0
    breadth_accel_1_5 = breadth[1]["pos"] - breadth[5]["pos"]
    breadth_accel_5_15 = breadth[5]["pos"] - breadth[15]["pos"]
    spy_vs_breadth_5 = spy_returns[5] - breadth[5]["med"]
    spy_vs_breadth_15 = spy_returns[15] - breadth[15]["med"]
    vol_ratio_5_30 = spy_rv[5] / max(spy_rv[30], 1e-8)
    vol_ratio_15_60 = spy_rv[15] / max(spy_rv[60], 1e-8)
    breadth_trend_gap_5_30 = breadth[5]["pos"] - breadth[30]["pos"]

    core: list[float] = []
    core.extend(spy_returns[h] for h in _SPY_RETURN_HORIZONS)
    core.extend(spy_rv[h] for h in _SPY_RV_HORIZONS)
    core.extend((mom_accel_1v5, mom_accel_5v15))
    for horizon in _BREADTH_HORIZONS:
        stats = breadth[horizon]
        core.extend(stats[key] for key in ("pos", "mean", "med", "disp", "p10", "p90"))
    core.extend(
        (
            breadth_accel_1_5,
            breadth_accel_5_15,
            vol_ratio_5_30,
            vol_ratio_15_60,
            breadth_trend_gap_5_30,
        )
    )

    breadth_only: list[float] = []
    for horizon in _BREADTH_HORIZONS:
        stats = breadth[horizon]
        breadth_only.extend(stats[key] for key in ("pos", "mean", "med", "disp", "p10", "p90"))
    breadth_only.extend(
        (
            breadth_accel_1_5,
            breadth_accel_5_15,
            spy_vs_breadth_5,
            spy_vs_breadth_15,
            vol_ratio_5_30,
            vol_ratio_15_60,
            breadth_trend_gap_5_30,
        )
    )
    return np.asarray(core, dtype=float), np.asarray(breadth_only, dtype=float)


@dataclass
class _Pending:
    target_time: datetime
    session_date: object
    core: np.ndarray
    breadth: np.ndarray
    start_price: float


@dataclass(frozen=True)
class HGBDirectionSignal:
    timestamp: datetime
    ready: bool
    eligible: bool
    direction: str
    expected_return_bps: float
    core_prediction_bps: float
    breadth_prediction_bps: float
    core_residual_sigma_bps: float
    breadth_residual_sigma_bps: float
    strength: float
    probability_up: float
    training_sessions: int
    training_samples: int
    model_version: str = HGB_DIRECTION_MODEL_VERSION

    def as_dict(self) -> dict[str, Any]:
        return {
            "timestamp": self.timestamp.isoformat().replace("+00:00", "Z"),
            "ready": self.ready,
            "eligible": self.eligible,
            "direction": self.direction,
            "expected_return_bps": self.expected_return_bps,
            "core_prediction_bps": self.core_prediction_bps,
            "breadth_prediction_bps": self.breadth_prediction_bps,
            "core_residual_sigma_bps": self.core_residual_sigma_bps,
            "breadth_residual_sigma_bps": self.breadth_residual_sigma_bps,
            "strength": self.strength,
            "probability_up": self.probability_up,
            "training_sessions": self.training_sessions,
            "training_samples": self.training_samples,
            "model_version": self.model_version,
        }


@dataclass
class CausalHGBDirectionStack:
    """Daily-refit 15m HGB ensemble matched to the profitable walk-forward protocol.

    Every five-minute observation is queued whether or not a model is warm. Its
    outcome is released only after +15 minutes. Models refit only when the session
    date advances, so no current-session label can alter that session's model.
    """

    pending: deque[_Pending] = field(default_factory=deque)
    core_x: list[np.ndarray] = field(default_factory=list)
    breadth_x: list[np.ndarray] = field(default_factory=list)
    y_bps: list[float] = field(default_factory=list)
    sample_dates: list[object] = field(default_factory=list)
    current_session: object | None = None
    core_scaler: StandardScaler | None = None
    breadth_scaler: StandardScaler | None = None
    core_model: HistGradientBoostingRegressor | None = None
    breadth_model: HistGradientBoostingRegressor | None = None
    core_sigma: float = math.inf
    breadth_sigma: float = math.inf

    @staticmethod
    def _new_model() -> HistGradientBoostingRegressor:
        return HistGradientBoostingRegressor(
            max_iter=80,
            max_leaf_nodes=7,
            min_samples_leaf=35,
            l2_regularization=3.0,
            learning_rate=0.04,
            random_state=42,
        )

    def _mature(self, timestamp: datetime, spy_price: float) -> None:
        while self.pending and self.pending[0].target_time <= timestamp:
            item = self.pending.popleft()
            if item.session_date != timestamp.date() or item.start_price <= 0 or spy_price <= 0:
                continue
            realized_bps = (spy_price / item.start_price - 1.0) * 10_000.0
            if not math.isfinite(realized_bps):
                continue
            self.core_x.append(item.core)
            self.breadth_x.append(item.breadth)
            self.y_bps.append(float(np.clip(realized_bps, -40.0, 40.0)))
            self.sample_dates.append(item.session_date)

    @staticmethod
    def _robust_sigma(residuals: np.ndarray) -> float:
        median = float(np.median(residuals))
        mad = 1.4826 * float(np.median(np.abs(residuals - median)))
        return max(mad, 3.0)

    def _refit_for_session(self, session_date) -> None:
        self.current_session = session_date
        sessions = sorted(set(self.sample_dates))
        if len(sessions) < MIN_TRAINING_SESSIONS or len(self.y_bps) < MIN_TRAINING_SAMPLES:
            self.core_model = None
            self.breadth_model = None
            return
        core = np.vstack(self.core_x)
        breadth = np.vstack(self.breadth_x)
        y = np.asarray(self.y_bps, dtype=float)
        self.core_scaler = StandardScaler().fit(core)
        self.breadth_scaler = StandardScaler().fit(breadth)
        z_core = self.core_scaler.transform(core)
        z_breadth = self.breadth_scaler.transform(breadth)
        self.core_model = self._new_model().fit(z_core, y)
        self.breadth_model = self._new_model().fit(z_breadth, y)
        self.core_sigma = self._robust_sigma(y - self.core_model.predict(z_core))
        self.breadth_sigma = self._robust_sigma(y - self.breadth_model.predict(z_breadth))

    @staticmethod
    def _probability_up(expected_bps: float, strength: float) -> float:
        directional = 0.5 * (1.0 + math.erf(max(strength, 0.0) / math.sqrt(2.0)))
        return directional if expected_bps >= 0 else 1.0 - directional

    def step(self, timestamp: datetime, states: dict[str, Any], spy_price: float) -> HGBDirectionSignal:
        self._mature(timestamp, spy_price)
        if self.current_session != timestamp.date():
            self._refit_for_session(timestamp.date())

        on_grid = timestamp.minute % SIGNAL_GRID_MINUTES == 0
        vectors = trailing_feature_vectors(states, timestamp) if on_grid else None
        ready = bool(
            vectors is not None
            and self.core_model is not None
            and self.breadth_model is not None
            and self.core_scaler is not None
            and self.breadth_scaler is not None
        )
        core_pred = breadth_pred = expected = 0.0
        strength = 0.0
        eligible = False
        direction = "FLAT"
        probability_up = 0.5
        core: np.ndarray | None = None
        breadth: np.ndarray | None = None
        if vectors is not None:
            core, breadth = vectors
        if ready and core is not None and breadth is not None:
            core_pred = float(self.core_model.predict(self.core_scaler.transform(core.reshape(1, -1)))[0])
            breadth_pred = float(
                self.breadth_model.predict(self.breadth_scaler.transform(breadth.reshape(1, -1)))[0]
            )
            expected = 0.5 * (core_pred + breadth_pred)
            same_direction = np.sign(core_pred) == np.sign(breadth_pred) and np.sign(expected) != 0
            strength = min(
                abs(core_pred) / max(self.core_sigma, 1e-9),
                abs(breadth_pred) / max(self.breadth_sigma, 1e-9),
            )
            eligible = bool(same_direction and strength >= MIN_SIGNAL_STRENGTH)
            direction = "BULLISH" if expected > 0 else "BEARISH"
            probability_up = self._probability_up(expected, strength)

        # Queue all valid five-minute observations, including the cold-start
        # sessions. Otherwise the stack could never accumulate enough samples to
        # become ready.
        if core is not None and breadth is not None and spy_price > 0:
            self.pending.append(
                _Pending(
                    target_time=timestamp + timedelta(minutes=HORIZON_MINUTES),
                    session_date=timestamp.date(),
                    core=core.copy(),
                    breadth=breadth.copy(),
                    start_price=spy_price,
                )
            )

        sessions = len(set(self.sample_dates))
        return HGBDirectionSignal(
            timestamp=timestamp,
            ready=ready,
            eligible=eligible,
            direction=direction,
            expected_return_bps=expected,
            core_prediction_bps=core_pred,
            breadth_prediction_bps=breadth_pred,
            core_residual_sigma_bps=float(self.core_sigma if math.isfinite(self.core_sigma) else 0.0),
            breadth_residual_sigma_bps=float(
                self.breadth_sigma if math.isfinite(self.breadth_sigma) else 0.0
            ),
            strength=float(strength),
            probability_up=float(np.clip(probability_up, 0.01, 0.99)),
            training_sessions=sessions,
            training_samples=len(self.y_bps),
        )
