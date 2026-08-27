from __future__ import annotations

import math
from collections import deque
from dataclasses import asdict, dataclass, field
from datetime import datetime, timedelta

import numpy as np
from sklearn.linear_model import SGDClassifier, SGDRegressor
from sklearn.preprocessing import StandardScaler

from .forecast import vectorize
from .models import MarketFactors


@dataclass(frozen=True)
class V2HorizonState:
    horizon_minutes: int
    big_move_threshold_bps: float
    probability_big_move: float
    probability_up_given_big_move: float
    expected_abs_move_bps: float
    expected_signed_move_bps: float
    magnitude_trust: float
    direction_trust: float
    signed_alignment: float
    matured_samples: int
    matured_big_moves: int
    model_ready: bool


@dataclass(frozen=True)
class V2MarketState:
    timestamp: datetime
    regime: str
    probability_big_move: float
    probability_up_given_big_move: float
    expected_abs_move_bps: float
    validated_direction_edge: float
    magnitude_trust: float
    direction_trust: float
    overall_trust: float
    horizons: tuple[V2HorizonState, ...]
    strategy_authority: bool = False
    version: str = "beta-spy-v2.0"

    def as_dict(self) -> dict:
        return asdict(self)


@dataclass
class _ValidationRecord:
    raw_big_probability: float
    raw_up_probability: float
    realized_big: bool
    realized_up: bool | None
    realized_abs_bps: float


@dataclass
class _Pending:
    target_time: datetime
    vector: np.ndarray
    start_price: float
    raw_big_probability: float
    raw_up_probability: float
    expected_abs_bps: float
    expected_signed_bps: float
    model_ready: bool


@dataclass
class V2HorizonModel:
    horizon_minutes: int
    min_samples: int = 200
    validation_window: int = 400
    max_pending: int = 5000
    scaler: StandardScaler = field(default_factory=StandardScaler)
    magnitude_classifier: SGDClassifier = field(
        default_factory=lambda: SGDClassifier(
            loss="log_loss",
            penalty="l2",
            alpha=0.0007,
            learning_rate="optimal",
            random_state=41,
        )
    )
    direction_classifier: SGDClassifier = field(
        default_factory=lambda: SGDClassifier(
            loss="log_loss",
            penalty="l2",
            alpha=0.0007,
            learning_rate="optimal",
            random_state=43,
        )
    )
    abs_regressor: SGDRegressor = field(
        default_factory=lambda: SGDRegressor(
            penalty="l2",
            alpha=0.0007,
            learning_rate="invscaling",
            eta0=0.004,
            random_state=47,
        )
    )
    signed_regressor: SGDRegressor = field(
        default_factory=lambda: SGDRegressor(
            penalty="l2",
            alpha=0.0007,
            learning_rate="invscaling",
            eta0=0.004,
            random_state=53,
        )
    )
    pending: deque[_Pending] = field(default_factory=deque)
    validations: deque[_ValidationRecord] = field(default_factory=deque)
    sample_count: int = 0
    big_move_count: int = 0
    _magnitude_initialized: bool = False
    _direction_initialized: bool = False

    @property
    def big_move_threshold_bps(self) -> float:
        return 7.5 * math.sqrt(max(self.horizon_minutes, 1) / 15.0)

    def _reset_models(self) -> None:
        self.scaler = StandardScaler()
        self.magnitude_classifier = SGDClassifier(
            loss="log_loss",
            penalty="l2",
            alpha=0.0007,
            learning_rate="optimal",
            random_state=41,
        )
        self.direction_classifier = SGDClassifier(
            loss="log_loss",
            penalty="l2",
            alpha=0.0007,
            learning_rate="optimal",
            random_state=43,
        )
        self.abs_regressor = SGDRegressor(
            penalty="l2",
            alpha=0.0007,
            learning_rate="invscaling",
            eta0=0.004,
            random_state=47,
        )
        self.signed_regressor = SGDRegressor(
            penalty="l2",
            alpha=0.0007,
            learning_rate="invscaling",
            eta0=0.004,
            random_state=53,
        )
        self.pending.clear()
        self.validations.clear()
        self.sample_count = 0
        self.big_move_count = 0
        self._magnitude_initialized = False
        self._direction_initialized = False

    def align_features(self, n_features: int) -> None:
        expected = getattr(self.scaler, "n_features_in_", None)
        if expected is not None and int(expected) != int(n_features):
            self._reset_models()

    def _magnitude_base_rate(self) -> float:
        """Causal Beta-prior base rate that cannot collapse at 0% or 100%."""
        if not self.validations:
            return 0.5
        big = sum(1 for row in self.validations if row.realized_big)
        return float((big + 10.0) / (len(self.validations) + 20.0))

    def _validation_metrics(self) -> tuple[float, float, float]:
        if not self.validations:
            return 0.0, 0.0, 0.0
        rows = list(self.validations)
        outcomes = np.asarray(
            [1.0 if row.realized_big else 0.0 for row in rows],
            dtype=float,
        )
        probabilities = np.asarray(
            [row.raw_big_probability for row in rows],
            dtype=float,
        )
        base = self._magnitude_base_rate()
        model_brier = float(np.mean((probabilities - outcomes) ** 2))
        baseline_brier = max(float(np.mean((base - outcomes) ** 2)), 1e-6)
        brier_skill = 1.0 - model_brier / baseline_brier
        evidence = min(len(rows) / 100.0, 1.0)
        magnitude_trust = float(
            np.clip(max(brier_skill, 0.0) * evidence, 0.0, 1.0)
        )

        directional = [
            row
            for row in rows
            if row.realized_big and row.realized_up is not None
        ]
        if not directional:
            return magnitude_trust, 0.0, 0.0
        correct = sum(
            (row.raw_up_probability >= 0.5) == bool(row.realized_up)
            for row in directional
        )
        shrunk_accuracy = (correct + 10.0) / (len(directional) + 20.0)
        signed_alignment = float(
            np.clip(2.0 * shrunk_accuracy - 1.0, -1.0, 1.0)
        )
        direction_trust = float(
            np.clip(
                abs(signed_alignment) * min(len(directional) / 60.0, 1.0),
                0.0,
                1.0,
            )
        )
        return magnitude_trust, direction_trust, signed_alignment

    def mature(self, timestamp: datetime, spy_price: float) -> None:
        while self.pending and self.pending[0].target_time <= timestamp:
            item = self.pending.popleft()
            if (
                item.target_time.date() != timestamp.date()
                or item.start_price <= 0
                or spy_price <= 0
            ):
                continue
            target_bps = (spy_price / item.start_price - 1.0) * 10_000.0
            realized_big = abs(target_bps) >= self.big_move_threshold_bps
            x = item.vector.reshape(1, -1)
            self.scaler.partial_fit(x)
            z = self.scaler.transform(x)
            y_big = np.asarray([1 if realized_big else 0], dtype=int)
            if not self._magnitude_initialized:
                self.magnitude_classifier.partial_fit(
                    z,
                    y_big,
                    classes=np.asarray([0, 1], dtype=int),
                )
                self._magnitude_initialized = True
            else:
                self.magnitude_classifier.partial_fit(z, y_big)
            self.abs_regressor.partial_fit(
                z,
                np.asarray([abs(target_bps)], dtype=float),
            )
            self.signed_regressor.partial_fit(
                z,
                np.asarray([target_bps], dtype=float),
            )
            if realized_big:
                y_dir = np.asarray([1 if target_bps > 0 else 0], dtype=int)
                if not self._direction_initialized:
                    self.direction_classifier.partial_fit(
                        z,
                        y_dir,
                        classes=np.asarray([0, 1], dtype=int),
                    )
                    self._direction_initialized = True
                else:
                    self.direction_classifier.partial_fit(z, y_dir)
                self.big_move_count += 1
            if item.model_ready:
                self.validations.append(
                    _ValidationRecord(
                        raw_big_probability=item.raw_big_probability,
                        raw_up_probability=item.raw_up_probability,
                        realized_big=realized_big,
                        realized_up=(target_bps > 0) if realized_big else None,
                        realized_abs_bps=abs(target_bps),
                    )
                )
                while len(self.validations) > self.validation_window:
                    self.validations.popleft()
            self.sample_count += 1

    def raw_predict(
        self,
        vector: np.ndarray,
    ) -> tuple[float, float, float, float, bool]:
        ready = self.sample_count >= self.min_samples and self._magnitude_initialized
        if not ready:
            return 0.5, 0.5, self.big_move_threshold_bps, 0.0, False
        z = self.scaler.transform(vector.reshape(1, -1))
        probability_big = float(self.magnitude_classifier.predict_proba(z)[0, 1])
        probability_up = (
            float(self.direction_classifier.predict_proba(z)[0, 1])
            if self._direction_initialized
            else 0.5
        )
        expected_abs = max(0.0, float(self.abs_regressor.predict(z)[0]))
        expected_signed = float(self.signed_regressor.predict(z)[0])
        return probability_big, probability_up, expected_abs, expected_signed, True

    def queue(
        self,
        timestamp: datetime,
        vector: np.ndarray,
        spy_price: float,
        raw: tuple[float, float, float, float, bool],
    ) -> None:
        probability_big, probability_up, expected_abs, expected_signed, ready = raw
        self.pending.append(
            _Pending(
                target_time=timestamp + timedelta(minutes=self.horizon_minutes),
                vector=vector.copy(),
                start_price=spy_price,
                raw_big_probability=probability_big,
                raw_up_probability=probability_up,
                expected_abs_bps=expected_abs,
                expected_signed_bps=expected_signed,
                model_ready=ready,
            )
        )
        while len(self.pending) > self.max_pending:
            self.pending.popleft()

    def state(
        self,
        raw: tuple[float, float, float, float, bool],
    ) -> V2HorizonState:
        probability_big, probability_up, expected_abs, expected_signed, ready = raw
        magnitude_trust, direction_trust, alignment = self._validation_metrics()
        base_rate = self._magnitude_base_rate()
        compressed_big = base_rate + magnitude_trust * (probability_big - base_rate)
        raw_edge = (probability_up - 0.5) * 2.0
        compressed_edge = raw_edge * alignment * direction_trust
        compressed_up = 0.5 + 0.5 * compressed_edge
        return V2HorizonState(
            horizon_minutes=self.horizon_minutes,
            big_move_threshold_bps=self.big_move_threshold_bps,
            probability_big_move=float(np.clip(compressed_big, 0.01, 0.99)),
            probability_up_given_big_move=float(np.clip(compressed_up, 0.01, 0.99)),
            expected_abs_move_bps=expected_abs,
            expected_signed_move_bps=expected_signed,
            magnitude_trust=magnitude_trust,
            direction_trust=direction_trust,
            signed_alignment=alignment,
            matured_samples=self.sample_count,
            matured_big_moves=self.big_move_count,
            model_ready=ready,
        )


class V2ValidationStack:
    """Maturity-delayed 5/15/30m market-state model."""

    def __init__(
        self,
        horizons: tuple[int, ...] = (5, 15, 30),
        min_samples: int = 200,
    ) -> None:
        self.models = {
            horizon: V2HorizonModel(
                horizon_minutes=horizon,
                min_samples=min_samples,
            )
            for horizon in horizons
        }

    def step(
        self,
        timestamp: datetime,
        factors: MarketFactors,
        spy_price: float,
    ) -> V2MarketState:
        x = vectorize(factors, feature_set="full_v1")
        raw_predictions: dict[int, tuple[float, float, float, float, bool]] = {}
        for model in self.models.values():
            model.align_features(x.size)
            model.mature(timestamp, spy_price)
            raw_predictions[model.horizon_minutes] = model.raw_predict(x)
        states = tuple(
            self.models[horizon].state(raw_predictions[horizon])
            for horizon in sorted(self.models)
        )
        for model in self.models.values():
            model.queue(
                timestamp,
                x,
                spy_price,
                raw_predictions[model.horizon_minutes],
            )

        role_weights = {5: 0.20, 15: 0.55, 30: 0.25}
        magnitude_numerator = 0.0
        magnitude_denominator = 0.0
        direction_numerator = 0.0
        direction_denominator = 0.0
        absolute_numerator = 0.0
        absolute_denominator = 0.0
        for state in states:
            role = role_weights.get(state.horizon_minutes, 0.10)
            magnitude_weight = role * max(state.magnitude_trust, 0.05)
            direction_weight = role * state.direction_trust
            magnitude_numerator += magnitude_weight * state.probability_big_move
            magnitude_denominator += magnitude_weight
            direction_edge = (state.probability_up_given_big_move - 0.5) * 2.0
            direction_numerator += direction_weight * direction_edge
            direction_denominator += direction_weight
            absolute_numerator += magnitude_weight * state.expected_abs_move_bps
            absolute_denominator += magnitude_weight

        probability_big = (
            magnitude_numerator / magnitude_denominator
            if magnitude_denominator
            else 0.5
        )
        validated_direction_edge = (
            direction_numerator / direction_denominator
            if direction_denominator
            else 0.0
        )
        probability_up = 0.5 + 0.5 * validated_direction_edge
        expected_abs = (
            absolute_numerator / absolute_denominator
            if absolute_denominator
            else 0.0
        )
        weights = [role_weights.get(state.horizon_minutes, 0.10) for state in states]
        magnitude_trust = float(
            np.average([state.magnitude_trust for state in states], weights=weights)
        )
        direction_trust = float(
            np.average([state.direction_trust for state in states], weights=weights)
        )
        overall_trust = math.sqrt(
            max(magnitude_trust, 0.0) * max(direction_trust, 0.0)
        )

        if magnitude_trust < 0.15:
            regime = "UNTRUSTED"
        elif (
            probability_big >= 0.65
            and abs(validated_direction_edge) >= 0.20
            and direction_trust >= 0.20
        ):
            regime = "DIRECTIONAL_EXPANSION"
        elif probability_big >= 0.65:
            regime = "EXPANSION_UNCERTAIN_DIRECTION"
        elif probability_big <= 0.35:
            regime = "QUIET"
        else:
            regime = "NORMAL"

        return V2MarketState(
            timestamp=timestamp,
            regime=regime,
            probability_big_move=float(np.clip(probability_big, 0.01, 0.99)),
            probability_up_given_big_move=float(np.clip(probability_up, 0.01, 0.99)),
            expected_abs_move_bps=max(0.0, expected_abs),
            validated_direction_edge=float(
                np.clip(validated_direction_edge, -1.0, 1.0)
            ),
            magnitude_trust=float(np.clip(magnitude_trust, 0.0, 1.0)),
            direction_trust=float(np.clip(direction_trust, 0.0, 1.0)),
            overall_trust=float(np.clip(overall_trust, 0.0, 1.0)),
            horizons=states,
        )
