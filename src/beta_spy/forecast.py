from __future__ import annotations

import pickle
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
from sklearn.linear_model import SGDClassifier, SGDRegressor
from sklearn.preprocessing import StandardScaler

from .models import HorizonForecast, MarketFactors


FEATURE_NAMES = (
    "coverage_ratio",
    "covered_weight",
    "trend_ew",
    "trend_weighted",
    "momentum_ew",
    "momentum_weighted",
    "volume_ew",
    "volume_weighted",
    "flow_ew",
    "flow_weighted",
    "volatility_ew",
    "volatility_weighted",
    "pct_above_vwap",
    "pct_ema_bullish",
    "pct_positive_5m",
    "pct_buy_flow",
    "participation",
    "concentration",
    "breadth_acceleration",
    "spy_return_1m",
    "spy_return_5m",
    "spy_vwap_distance_bps",
    "spy_flow",
    "spy_quote_imbalance",
    "spy_spread_bps",
)


SESSION_OPEN_MINUTES = 13 * 60 + 30  # 13:30 UTC regular-session open
SESSION_LENGTH_MINUTES = 390.0


def session_fraction(timestamp: datetime) -> float:
    minutes = timestamp.hour * 60 + timestamp.minute - SESSION_OPEN_MINUTES
    return float(min(max(minutes / SESSION_LENGTH_MINUTES, 0.0), 1.0))


def vectorize(factors: MarketFactors) -> np.ndarray:
    values = factors.feature_dict()
    base = [values[name] for name in FEATURE_NAMES]
    # Intraday seasonality: opens and closes behave differently from lunch.
    fraction = session_fraction(factors.timestamp)
    base.append(fraction)
    base.append(fraction * fraction)
    return np.asarray(base, dtype=float)


@dataclass
class _OnlineCalibrator:
    """Online Platt scaling: sigmoid(a * logit(p_raw) + b) fitted by SGD.

    The raw SGD classifier probabilities are badly overconfident (backtest
    Brier ~0.43 versus 0.25 for always predicting 0.5). Every consumer of
    probability_up -- the trade threshold, edge sizing, and the option EV
    model -- assumes calibrated probabilities, so miscalibration leaks into
    every layer. Starts as the identity and is only applied once enough
    matured labels have been seen.
    """

    a: float = 1.0
    b: float = 0.0
    learning_rate: float = 0.02
    min_samples: int = 100
    sample_count: int = 0

    @staticmethod
    def _logit(p: float) -> float:
        p = min(max(p, 1e-6), 1.0 - 1e-6)
        raw = float(np.log(p / (1.0 - p)))
        return min(max(raw, -6.0), 6.0)

    def update(self, raw_probability: float, outcome_up: bool) -> None:
        logit = self._logit(raw_probability)
        z = self.a * logit + self.b
        q = 1.0 / (1.0 + np.exp(-z))
        gradient = q - (1.0 if outcome_up else 0.0)
        self.a -= self.learning_rate * gradient * logit
        self.b -= self.learning_rate * gradient
        self.a = min(max(self.a, 0.05), 5.0)
        self.b = min(max(self.b, -2.0), 2.0)
        self.sample_count += 1

    def calibrate(self, raw_probability: float) -> float:
        if self.sample_count < self.min_samples:
            return raw_probability
        z = self.a * self._logit(raw_probability) + self.b
        return float(1.0 / (1.0 + np.exp(-z)))


def _fallback(factors: MarketFactors, horizon: int) -> tuple[float, float]:
    terms: list[float] = []
    for value, weight in [
        (factors.trend_ew, 0.20),
        (factors.trend_weighted, 0.20),
        (factors.momentum_ew, 0.15),
        (factors.momentum_weighted, 0.15),
        (factors.flow_ew, 0.10),
        (factors.flow_weighted, 0.10),
        (factors.participation, 0.05),
        (factors.spy_flow, 0.05),
    ]:
        if value is not None:
            terms.append(float(value) * weight)
    score = float(sum(terms))
    probability = 1.0 / (1.0 + np.exp(-2.2 * score))
    scale = max(horizon / 15.0, 0.25) ** 0.5
    expected_bps = score * 12.0 * scale
    return float(probability), float(expected_bps)


@dataclass
class _Pending:
    target_time: datetime
    vector: np.ndarray
    start_price: float
    raw_probability: float = 0.5
    model_ready: bool = False


@dataclass
class OnlineHorizonModel:
    horizon_minutes: int
    min_samples: int = 200
    max_pending: int = 5000
    # Direction labels inside the deadband are noise; the classifier only
    # trains on moves large enough to mean something at this horizon.
    deadband_bps: float = 2.0
    scaler: StandardScaler = field(default_factory=StandardScaler)
    classifier: SGDClassifier = field(
        default_factory=lambda: SGDClassifier(
            loss="log_loss",
            penalty="l2",
            alpha=0.0005,
            learning_rate="optimal",
            random_state=17,
        )
    )
    regressor: SGDRegressor = field(
        default_factory=lambda: SGDRegressor(
            penalty="l2",
            alpha=0.0005,
            learning_rate="invscaling",
            eta0=0.005,
            random_state=17,
        )
    )
    pending: deque[_Pending] = field(default_factory=deque)
    calibrator: _OnlineCalibrator = field(default_factory=_OnlineCalibrator)
    sample_count: int = 0
    _classifier_initialized: bool = False

    def mature(self, timestamp: datetime, spy_price: float) -> list[float]:
        realized: list[float] = []
        while self.pending and self.pending[0].target_time <= timestamp:
            item = self.pending.popleft()
            if item.target_time.date() != timestamp.date():
                continue
            if item.start_price <= 0 or spy_price <= 0:
                continue
            target_return = spy_price / item.start_price - 1.0
            target_bps = target_return * 10_000.0
            x = item.vector.reshape(1, -1)
            self.scaler.partial_fit(x)
            z = self.scaler.transform(x)
            if abs(target_bps) >= self.deadband_bps or not self._classifier_initialized:
                y_class = np.asarray([1 if target_return > 0 else 0], dtype=int)
                if not self._classifier_initialized:
                    self.classifier.partial_fit(z, y_class, classes=np.asarray([0, 1], dtype=int))
                    self._classifier_initialized = True
                else:
                    self.classifier.partial_fit(z, y_class)
            self.regressor.partial_fit(z, np.asarray([target_bps], dtype=float))
            if item.model_ready:
                self.calibrator.update(item.raw_probability, target_return > 0)
            self.sample_count += 1
            realized.append(target_return)
        return realized

    def queue(self, timestamp: datetime, vector: np.ndarray, spy_price: float) -> None:
        raw_probability = 0.5
        ready = self.sample_count >= self.min_samples and self._classifier_initialized
        if ready:
            z = self.scaler.transform(vector.reshape(1, -1))
            raw_probability = float(self.classifier.predict_proba(z)[0, 1])
        self.pending.append(
            _Pending(
                target_time=timestamp + timedelta(minutes=self.horizon_minutes),
                vector=vector.copy(),
                start_price=spy_price,
                raw_probability=raw_probability,
                model_ready=ready,
            )
        )
        while len(self.pending) > self.max_pending:
            self.pending.popleft()

    def predict(self, factors: MarketFactors, vector: np.ndarray) -> HorizonForecast:
        fallback_probability, fallback_return = _fallback(factors, self.horizon_minutes)
        ready = self.sample_count >= self.min_samples and self._classifier_initialized
        if ready:
            z = self.scaler.transform(vector.reshape(1, -1))
            probability = self.calibrator.calibrate(float(self.classifier.predict_proba(z)[0, 1]))
            expected_bps = float(self.regressor.predict(z)[0])
        else:
            probability = fallback_probability
            expected_bps = fallback_return
        directional_strength = abs(probability - 0.5) * 2.0
        evidence = min(self.sample_count / max(self.min_samples, 1), 1.0)
        confidence = directional_strength * (0.6 + 0.4 * evidence) * min(factors.coverage_ratio / 0.9, 1.0)
        return HorizonForecast(
            horizon_minutes=self.horizon_minutes,
            probability_up=max(0.01, min(0.99, probability)),
            expected_return_bps=expected_bps,
            confidence=max(0.0, min(1.0, confidence)),
            model_ready=ready,
            sample_count=self.sample_count,
        )


class OnlineForecastStack:
    def __init__(self, horizons: tuple[int, ...] = (5, 15, 30), min_samples: int = 200):
        self.models = {
            horizon: OnlineHorizonModel(
                horizon_minutes=horizon,
                min_samples=min_samples,
                deadband_bps=2.0 * (horizon / 15.0) ** 0.5,
            )
            for horizon in horizons
        }

    def step(self, timestamp: datetime, factors: MarketFactors, spy_price: float) -> tuple[HorizonForecast, ...]:
        x = vectorize(factors)
        for model in self.models.values():
            model.mature(timestamp, spy_price)
        forecasts = tuple(self.models[horizon].predict(factors, x) for horizon in sorted(self.models))
        for model in self.models.values():
            model.queue(timestamp, x, spy_price)
        return forecasts

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("wb") as handle:
            pickle.dump(self, handle, protocol=pickle.HIGHEST_PROTOCOL)

    @classmethod
    def load(cls, path: Path) -> "OnlineForecastStack":
        with path.open("rb") as handle:
            model = pickle.load(handle)
        if not isinstance(model, cls):
            raise TypeError("Saved object is not an OnlineForecastStack")
        return model


@dataclass
class _PendingMeta:
    target_time: datetime
    vector: np.ndarray
    direction: int
    start_price: float


def meta_vector(
    base_vector: np.ndarray, forecasts: tuple[HorizonForecast, ...], direction: int
) -> np.ndarray:
    """Meta-model features: market state plus what the primary stack believes."""
    extras = [float(direction)]
    for forecast in sorted(forecasts, key=lambda item: item.horizon_minutes):
        extras.append(forecast.probability_up)
        extras.append(forecast.expected_return_bps / 10.0)
        extras.append(forecast.confidence)
    return np.concatenate([base_vector, np.asarray(extras, dtype=float)])


@dataclass
class OnlineMetaGate:
    """Meta-labeling: predicts whether a gated trade signal will be profitable.

    Trains online on every signal that passes the primary gates (including
    ones it later vetoes, so it never suffers selection bias) and, once warm,
    blocks trades whose predicted win probability is below threshold.
    """

    horizon_minutes: int = 15
    min_samples: int = 150
    threshold: float = 0.50
    max_pending: int = 2000
    scaler: StandardScaler = field(default_factory=StandardScaler)
    classifier: SGDClassifier = field(
        default_factory=lambda: SGDClassifier(
            loss="log_loss",
            penalty="l2",
            alpha=0.001,
            learning_rate="optimal",
            random_state=29,
        )
    )
    pending: deque[_PendingMeta] = field(default_factory=deque)
    sample_count: int = 0
    _initialized: bool = False

    def mature(self, timestamp: datetime, spy_price: float) -> None:
        while self.pending and self.pending[0].target_time <= timestamp:
            item = self.pending.popleft()
            if item.target_time.date() != timestamp.date():
                continue
            if item.start_price <= 0 or spy_price <= 0:
                continue
            realized = (spy_price / item.start_price - 1.0) * item.direction
            x = item.vector.reshape(1, -1)
            self.scaler.partial_fit(x)
            z = self.scaler.transform(x)
            y = np.asarray([1 if realized > 0 else 0], dtype=int)
            if not self._initialized:
                self.classifier.partial_fit(z, y, classes=np.asarray([0, 1], dtype=int))
                self._initialized = True
            else:
                self.classifier.partial_fit(z, y)
            self.sample_count += 1

    def queue(self, timestamp: datetime, vector: np.ndarray, direction: int, spy_price: float) -> None:
        self.pending.append(
            _PendingMeta(
                target_time=timestamp + timedelta(minutes=self.horizon_minutes),
                vector=vector.copy(),
                direction=direction,
                start_price=spy_price,
            )
        )
        while len(self.pending) > self.max_pending:
            self.pending.popleft()

    def win_probability(self, vector: np.ndarray) -> float | None:
        if not self._initialized or self.sample_count < self.min_samples:
            return None
        z = self.scaler.transform(vector.reshape(1, -1))
        return float(self.classifier.predict_proba(z)[0, 1])
