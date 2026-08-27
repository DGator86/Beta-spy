from __future__ import annotations

import math
from collections import deque
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta
from typing import Any

import numpy as np
from sklearn.ensemble import ExtraTreesRegressor

STATE_MODEL_VERSION = "beta-spy-v2-predictive-state-1"
HORIZONS = (5, 15, 30)
SIGNAL_GRID_MINUTES = 5
MIN_TRAINING_SESSIONS = 5
MIN_TRAINING_SAMPLES = 250
ANALOG_K = 60
N_ESTIMATORS = 48
MAX_DEPTH = 6
MIN_SAMPLES_LEAF = 12
MAX_FEATURES = 0.65
SAME_REGIME_BONUS = 1.25

_RETURN_HORIZONS = (1, 2, 3, 5, 10, 15, 20, 30, 60)
_RV_HORIZONS = (5, 10, 15, 30, 60)
_BREADTH_HORIZONS = (1, 2, 5, 10, 15, 30)


def _minute(value: datetime) -> datetime:
    return value.astimezone(UTC).replace(second=0, microsecond=0)


def _bar_map(state: Any, day) -> dict[datetime, Any]:
    return {
        _minute(bar.timestamp): bar
        for bar in state.bars
        if _minute(bar.timestamp).date() == day and float(bar.close) > 0
    }


def _close_maps(states: dict[str, Any], day) -> dict[str, dict[datetime, float]]:
    return {
        symbol: {stamp: float(bar.close) for stamp, bar in _bar_map(state, day).items()}
        for symbol, state in states.items()
    }


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
    close_maps: dict[str, dict[datetime, float]], reference: datetime, horizon: int
) -> dict[str, float] | None:
    values: list[float] = []
    for symbol, prices in close_maps.items():
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


def _relative_volume_stats(states: dict[str, Any], day, reference: datetime) -> tuple[float, float]:
    values: list[float] = []
    for symbol, state in states.items():
        if symbol == "SPY":
            continue
        bars = sorted(
            [bar for bar in state.bars if _minute(bar.timestamp) <= reference],
            key=lambda bar: _minute(bar.timestamp),
        )
        if len(bars) < 2 or _minute(bars[-1].timestamp) != reference:
            continue
        lookback = bars[-21:-1] if len(bars) >= 21 else bars[:-1]
        baseline = float(np.mean([max(float(bar.volume), 0.0) for bar in lookback])) if lookback else 0.0
        if baseline > 0:
            values.append(max(float(bars[-1].volume), 0.0) / baseline)
    if not values:
        return 0.0, 0.0
    array = np.asarray(values, dtype=float)
    return float(np.median(array)), float(np.quantile(array, 0.90))


def state_feature_vector(states: dict[str, Any], timestamp: datetime) -> np.ndarray | None:
    """Leakage-resistant state vector matched to the validated research feature family.

    The engine flushes completed bars before calling this stack, so `timestamp - 1m`
    is the most recent fully known minute. No future bar or end-of-session statistic is used.
    """
    reference = _minute(timestamp) - timedelta(minutes=1)
    day = reference.date()
    spy_state = states.get("SPY")
    if spy_state is None:
        return None
    close_maps = _close_maps(states, day)
    spy_prices = close_maps.get("SPY") or {}
    if reference not in spy_prices:
        return None
    spy_bars = _bar_map(spy_state, day)

    returns: dict[int, float] = {}
    for horizon in _RETURN_HORIZONS:
        value = _exact_return(spy_prices, reference, horizon)
        if value is None:
            return None
        returns[horizon] = value

    realized: dict[int, float] = {}
    for horizon in _RV_HORIZONS:
        value = _realized_vol(spy_prices, reference, horizon)
        if value is None:
            return None
        realized[horizon] = max(value, 1e-8)

    breadth: dict[int, dict[str, float]] = {}
    for horizon in _BREADTH_HORIZONS:
        stats = _breadth_stats(close_maps, reference, horizon)
        if stats is None:
            return None
        breadth[horizon] = stats

    session = [bar for stamp, bar in sorted(spy_bars.items()) if stamp <= reference]
    if not session:
        return None
    open_price = float(session[0].open)
    spot = float(session[-1].close)
    high = max(float(bar.high) for bar in session)
    low = min(float(bar.low) for bar in session)
    session_range = max(high - low, 0.0)
    range_pos = (spot - low) / session_range if session_range > 0 else 0.5
    range_pos = float(np.clip(range_pos, 0.0, 1.0))
    range_edge_distance = min(range_pos, 1.0 - range_pos)
    session_range_bps = session_range / max(open_price, 1e-9) * 10_000.0
    from_open_bps = (spot / max(open_price, 1e-9) - 1.0) * 10_000.0

    pv = 0.0
    volume = 0.0
    for bar in session:
        bar_volume = max(float(bar.volume), 0.0)
        typical = float(bar.vwap) if bar.vwap is not None else (
            float(bar.high) + float(bar.low) + float(bar.close)
        ) / 3.0
        pv += typical * bar_volume
        volume += bar_volume
    session_vwap = pv / volume if volume > 0 else spot
    spy_vwap_bps = (spot / max(session_vwap, 1e-9) - 1.0) * 10_000.0

    relvol_med, relvol_p90 = _relative_volume_stats(states, day, reference)
    from zoneinfo import ZoneInfo

    local = reference.astimezone(ZoneInfo("America/New_York"))
    fraction = float(np.clip((local.hour * 60 + local.minute - 570) / 390.0, 0.0, 1.0))
    tod_sin = math.sin(2.0 * math.pi * fraction)
    tod_cos = math.cos(2.0 * math.pi * fraction)

    mom_accel_1v5 = returns[1] - returns[5] / 5.0
    mom_accel_5v15 = returns[5] / 5.0 - returns[15] / 15.0
    breadth_accel_1_5 = breadth[1]["pos"] - breadth[5]["pos"]
    breadth_accel_5_15 = breadth[5]["pos"] - breadth[15]["pos"]
    spy_vs_breadth_5 = returns[5] - breadth[5]["med"]
    spy_vs_breadth_15 = returns[15] - breadth[15]["med"]
    vol_ratio_5_30 = realized[5] / max(realized[30], 1e-8)
    vol_ratio_15_60 = realized[15] / max(realized[60], 1e-8)
    breadth_trend_gap_5_30 = breadth[5]["pos"] - breadth[30]["pos"]

    values: list[float] = []
    values.extend(returns[h] for h in _RETURN_HORIZONS)
    values.extend(realized[h] for h in _RV_HORIZONS)
    values.extend((spy_vwap_bps, session_range_bps, range_pos, from_open_bps, mom_accel_1v5, mom_accel_5v15))
    for horizon in _BREADTH_HORIZONS:
        stats = breadth[horizon]
        values.extend(stats[key] for key in ("pos", "mean", "med", "disp", "p10", "p90"))
    values.extend(
        (
            breadth_accel_1_5,
            breadth_accel_5_15,
            spy_vs_breadth_5,
            spy_vs_breadth_15,
            relvol_med,
            relvol_p90,
            tod_sin,
            tod_cos,
            vol_ratio_5_30,
            vol_ratio_15_60,
            breadth_trend_gap_5_30,
            range_edge_distance,
        )
    )
    vector = np.asarray(values, dtype=float)
    return vector if np.all(np.isfinite(vector)) else None


@dataclass
class _Pending:
    timestamp: datetime
    session_date: object
    vector: np.ndarray
    start_price: float
    y5: float | None = None
    y15: float | None = None
    forecast_mean15: float | None = None
    forecast_sigma15: float | None = None


@dataclass(frozen=True)
class PredictiveStateDistribution:
    timestamp: datetime
    ready: bool
    regime: str
    analog_count: int
    effective_analogs: float
    mean_proximity: float
    direct_pred_5: float
    direct_pred_15: float
    direct_pred_30: float
    direct_pred_abs15: float
    conformal_scale: float
    mean_5: float
    mean_15: float
    mean_30: float
    sigma_5: float
    sigma_15: float
    sigma_30: float
    p_up_5: float
    p_up_15: float
    p_up_30: float
    p_big_5: float
    p_big_15: float
    p_big_30: float
    quantiles_5: dict[str, float]
    quantiles_15: dict[str, float]
    quantiles_30: dict[str, float]
    p_persistent_30: float
    p_reversal_15: float
    p_reversal_30: float
    p_acceleration: float
    analog_y15_bps: tuple[float, ...]
    analog_weights: tuple[float, ...]
    training_sessions: int
    training_samples: int
    model_version: str = STATE_MODEL_VERSION
    strategy_authority: bool = False

    def as_dict(self) -> dict[str, Any]:
        return {
            "timestamp": self.timestamp.isoformat().replace("+00:00", "Z"),
            **{key: value for key, value in self.__dict__.items() if key != "timestamp"},
        }


def _weighted_quantile(values: np.ndarray, weights: np.ndarray, q: float) -> float:
    order = np.argsort(values)
    values = values[order]
    weights = weights[order]
    cumulative = np.cumsum(weights)
    return float(values[min(np.searchsorted(cumulative, q * cumulative[-1]), len(values) - 1)])


def _effective_n(weights: np.ndarray) -> float:
    return float(weights.sum() ** 2 / max(float(np.sum(weights * weights)), 1e-12))


@dataclass
class CausalPredictiveStateStack:
    pending: deque[_Pending] = field(default_factory=deque)
    x: list[np.ndarray] = field(default_factory=list)
    y5: list[float] = field(default_factory=list)
    y15: list[float] = field(default_factory=list)
    y30: list[float] = field(default_factory=list)
    sample_dates: list[object] = field(default_factory=list)
    validation_dates: list[object] = field(default_factory=list)
    validation_z15: list[float] = field(default_factory=list)
    current_session: object | None = None
    forest: ExtraTreesRegressor | None = None
    fit_x: np.ndarray | None = None
    fit_y5: np.ndarray | None = None
    fit_y15: np.ndarray | None = None
    fit_y30: np.ndarray | None = None
    fit_leaves: np.ndarray | None = None
    fit_regimes: np.ndarray | None = None
    abs_lo: float = 0.0
    abs_hi: float = 0.0
    rv_median: float = 0.0
    session_conformal_scale: float = 1.10

    @staticmethod
    def _new_model() -> ExtraTreesRegressor:
        return ExtraTreesRegressor(
            n_estimators=N_ESTIMATORS,
            max_depth=MAX_DEPTH,
            min_samples_leaf=MIN_SAMPLES_LEAF,
            max_features=MAX_FEATURES,
            bootstrap=False,
            n_jobs=-1,
            random_state=42,
        )

    def _mature(self, timestamp: datetime, spy_price: float) -> None:
        kept: deque[_Pending] = deque()
        for item in self.pending:
            if item.session_date != timestamp.date():
                continue
            elapsed = (timestamp - item.timestamp).total_seconds() / 60.0
            if item.start_price <= 0 or spy_price <= 0:
                continue
            realized = (spy_price / item.start_price - 1.0) * 10_000.0
            if item.y5 is None and elapsed >= 5:
                item.y5 = realized
            if item.y15 is None and elapsed >= 15:
                item.y15 = realized
                if item.forecast_mean15 is not None and item.forecast_sigma15 not in (None, 0.0):
                    z = abs(realized - item.forecast_mean15) / max(float(item.forecast_sigma15), 0.50)
                    self.validation_dates.append(item.session_date)
                    self.validation_z15.append(float(z))
            if elapsed >= 30:
                if item.y5 is not None and item.y15 is not None and math.isfinite(realized):
                    self.x.append(item.vector)
                    self.y5.append(float(item.y5))
                    self.y15.append(float(item.y15))
                    self.y30.append(float(realized))
                    self.sample_dates.append(item.session_date)
                continue
            kept.append(item)
        self.pending = kept

    @staticmethod
    def _regimes(pred: np.ndarray, rv15: np.ndarray, abs_lo: float, abs_hi: float, rv_median: float) -> np.ndarray:
        p5, p15, p30, pabs = pred.T
        aligned = (np.sign(p5) == np.sign(p15)) & (np.sign(p15) == np.sign(p30)) & (np.sign(p15) != 0)
        directional = aligned & (np.abs(p15) >= 0.35 * np.maximum(pabs, 1e-6))
        out = np.full(len(pred), "TRANSITION", dtype=object)
        out[(pabs <= abs_lo) & (rv15 <= rv_median)] = "QUIET"
        out[(pabs >= abs_hi) & ~directional] = "EXPANSION"
        out[directional & (p15 > 0)] = "DIRECTIONAL_UP"
        out[directional & (p15 < 0)] = "DIRECTIONAL_DOWN"
        return out

    def _refit_for_session(self, session_date) -> None:
        self.current_session = session_date
        sessions = sorted(set(self.sample_dates))
        prior_z = [
            value
            for day, value in zip(self.validation_dates, self.validation_z15, strict=False)
            if day < session_date
        ]
        if len(prior_z) >= 250:
            self.session_conformal_scale = float(np.clip(np.quantile(prior_z, 0.90) / 1.645, 0.85, 1.50))
        else:
            self.session_conformal_scale = 1.10
        if len(sessions) < MIN_TRAINING_SESSIONS or len(self.y15) < MIN_TRAINING_SAMPLES:
            self.forest = None
            return
        self.fit_x = np.vstack(self.x)
        self.fit_y5 = np.asarray(self.y5, dtype=float)
        self.fit_y15 = np.asarray(self.y15, dtype=float)
        self.fit_y30 = np.asarray(self.y30, dtype=float)
        targets = np.column_stack((self.fit_y5, self.fit_y15, self.fit_y30, np.abs(self.fit_y15)))
        self.forest = self._new_model().fit(self.fit_x, targets)
        self.fit_leaves = self.forest.apply(self.fit_x)
        pred = self.forest.predict(self.fit_x)
        self.abs_lo, self.abs_hi = [float(x) for x in np.quantile(pred[:, 3], (0.33, 0.67))]
        rv15 = self.fit_x[:, 11]
        self.rv_median = float(np.median(rv15))
        self.fit_regimes = self._regimes(pred, rv15, self.abs_lo, self.abs_hi, self.rv_median)

    def _distribution(self, timestamp: datetime, vector: np.ndarray) -> PredictiveStateDistribution:
        assert self.forest is not None and self.fit_x is not None and self.fit_leaves is not None
        assert self.fit_y5 is not None and self.fit_y15 is not None and self.fit_y30 is not None
        assert self.fit_regimes is not None
        pred = self.forest.predict(vector.reshape(1, -1))[0]
        leaf = self.forest.apply(vector.reshape(1, -1))[0]
        proximity = np.mean(self.fit_leaves == leaf, axis=1)
        regime = str(
            self._regimes(
                pred.reshape(1, -1),
                np.asarray([vector[11]]),
                self.abs_lo,
                self.abs_hi,
                self.rv_median,
            )[0]
        )
        score = proximity.copy()
        score[self.fit_regimes == regime] *= SAME_REGIME_BONUS
        k = min(ANALOG_K, len(score))
        indices = np.argpartition(score, -k)[-k:] if k < len(score) else np.arange(len(score))
        weighted_score = np.maximum(score[indices], 1e-4)
        weights = weighted_score * weighted_score
        weights /= weights.sum()

        def stats(y: np.ndarray, threshold: float) -> tuple[float, float, float, float, dict[str, float]]:
            values = y[indices]
            mean = float(np.dot(weights, values))
            sigma = math.sqrt(max(float(np.dot(weights, (values - mean) ** 2)), 1e-9))
            p_up = float(np.dot(weights, (values > 0).astype(float)))
            p_big = float(np.dot(weights, (np.abs(values) >= threshold).astype(float)))
            quantiles = {
                str(int(q * 100)): _weighted_quantile(values, weights, q)
                for q in (0.05, 0.10, 0.25, 0.50, 0.75, 0.90, 0.95)
            }
            return mean, sigma, p_up, p_big, quantiles

        mean5, sigma5, pup5, pbig5, q5 = stats(self.fit_y5, 4.5)
        mean15, sigma15, pup15, pbig15, q15 = stats(self.fit_y15, 7.5)
        mean30, sigma30, pup30, pbig30, q30 = stats(self.fit_y30, 10.5)
        y5 = self.fit_y5[indices]
        y15 = self.fit_y15[indices]
        y30 = self.fit_y30[indices]
        persistent = float(
            np.dot(
                weights,
                ((np.sign(y5) == np.sign(y15)) & (np.sign(y15) == np.sign(y30))).astype(float),
            )
        )
        reversal15 = float(np.dot(weights, (np.sign(y5) != np.sign(y15)).astype(float)))
        reversal30 = float(np.dot(weights, (np.sign(y15) != np.sign(y30)).astype(float)))
        acceleration = float(np.dot(weights, (np.abs(y30) > 1.35 * np.abs(y15)).astype(float)))
        return PredictiveStateDistribution(
            timestamp=timestamp,
            ready=True,
            regime=regime,
            analog_count=len(indices),
            effective_analogs=_effective_n(weights),
            mean_proximity=float(np.dot(weights, proximity[indices])),
            direct_pred_5=float(pred[0]),
            direct_pred_15=float(pred[1]),
            direct_pred_30=float(pred[2]),
            direct_pred_abs15=max(float(pred[3]), 0.0),
            conformal_scale=self.session_conformal_scale,
            mean_5=mean5,
            mean_15=mean15,
            mean_30=mean30,
            sigma_5=sigma5,
            sigma_15=sigma15,
            sigma_30=sigma30,
            p_up_5=pup5,
            p_up_15=pup15,
            p_up_30=pup30,
            p_big_5=pbig5,
            p_big_15=pbig15,
            p_big_30=pbig30,
            quantiles_5=q5,
            quantiles_15=q15,
            quantiles_30=q30,
            p_persistent_30=persistent,
            p_reversal_15=reversal15,
            p_reversal_30=reversal30,
            p_acceleration=acceleration,
            analog_y15_bps=tuple(float(x) for x in y15),
            analog_weights=tuple(float(x) for x in weights),
            training_sessions=len(set(self.sample_dates)),
            training_samples=len(self.y15),
        )

    def step(self, timestamp: datetime, states: dict[str, Any], spy_price: float) -> PredictiveStateDistribution:
        self._mature(timestamp, spy_price)
        if self.current_session != timestamp.date():
            self._refit_for_session(timestamp.date())
        on_grid = timestamp.minute % SIGNAL_GRID_MINUTES == 0
        vector = state_feature_vector(states, timestamp) if on_grid else None
        ready = bool(on_grid and vector is not None and self.forest is not None)
        if ready and vector is not None:
            distribution = self._distribution(timestamp, vector)
            forecast_mean = distribution.mean_15
            forecast_sigma = distribution.sigma_15
        else:
            distribution = PredictiveStateDistribution(
                timestamp=timestamp,
                ready=False,
                regime="WARMING",
                analog_count=0,
                effective_analogs=0.0,
                mean_proximity=0.0,
                direct_pred_5=0.0,
                direct_pred_15=0.0,
                direct_pred_30=0.0,
                direct_pred_abs15=0.0,
                conformal_scale=self.session_conformal_scale,
                mean_5=0.0,
                mean_15=0.0,
                mean_30=0.0,
                sigma_5=0.0,
                sigma_15=0.0,
                sigma_30=0.0,
                p_up_5=0.5,
                p_up_15=0.5,
                p_up_30=0.5,
                p_big_5=0.0,
                p_big_15=0.0,
                p_big_30=0.0,
                quantiles_5={},
                quantiles_15={},
                quantiles_30={},
                p_persistent_30=0.0,
                p_reversal_15=0.0,
                p_reversal_30=0.0,
                p_acceleration=0.0,
                analog_y15_bps=(),
                analog_weights=(),
                training_sessions=len(set(self.sample_dates)),
                training_samples=len(self.y15),
            )
            forecast_mean = None
            forecast_sigma = None

        if on_grid and vector is not None and spy_price > 0:
            self.pending.append(
                _Pending(
                    timestamp=timestamp,
                    session_date=timestamp.date(),
                    vector=vector.copy(),
                    start_price=spy_price,
                    forecast_mean15=forecast_mean,
                    forecast_sigma15=forecast_sigma,
                )
            )
        return distribution
