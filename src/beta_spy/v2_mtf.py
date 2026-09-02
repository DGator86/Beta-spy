from __future__ import annotations

import hashlib
import json
import math
from collections import deque
from dataclasses import asdict, dataclass, field
from datetime import datetime, timedelta
from typing import Any

import numpy as np
from sklearn.linear_model import SGDClassifier, SGDRegressor
from sklearn.preprocessing import StandardScaler

from .forecast import vectorize
from .models import MarketFactors


V2_MODEL_VERSION = "beta-spy-v2-mtf-validated-1"


@dataclass(frozen=True)
class V2Config:
    horizons: tuple[int, ...] = (5, 15, 30)
    big_move_threshold_bps: dict[int, float] = field(
        default_factory=lambda: {5: 4.5, 15: 7.5, 30: 10.5}
    )
    base_weights: dict[int, float] = field(
        default_factory=lambda: {5: 0.25, 15: 0.50, 30: 0.25}
    )
    min_samples: int = 200
    min_big_direction_samples: int = 40
    validation_decay: float = 0.97
    alignment_decay: float = 0.95
    min_matured_for_full_trust: int = 40
    min_validated_big_probability: float = 0.55
    quiet_max_validated_big_probability: float = 0.30
    min_validated_direction_edge: float = 0.12
    min_composite_trust: float = 0.25
    min_agreement: float = 0.35
    feature_set: str = "full_v1"

    def fingerprint(self) -> str:
        raw = json.dumps(asdict(self), sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(raw.encode()).hexdigest()


@dataclass
class _ValidationState:
    magnitude_brier: float = 0.25
    direction_brier: float = 0.25
    abs_mae_ratio: float = 1.0
    direction_alignment: float = 0.0
    magnitude_count: int = 0
    direction_count: int = 0
    abs_count: int = 0

    def update(
        self,
        *,
        probability_big: float,
        probability_up: float,
        expected_abs_bps: float,
        realized_bps: float,
        threshold_bps: float,
        decay: float,
        alignment_decay: float,
    ) -> None:
        is_big = float(abs(realized_bps) >= threshold_bps)
        mag_error = (probability_big - is_big) ** 2
        self.magnitude_brier = decay * self.magnitude_brier + (1.0 - decay) * mag_error
        self.magnitude_count += 1

        if is_big:
            is_up = float(realized_bps > 0.0)
            dir_error = (probability_up - is_up) ** 2
            self.direction_brier = decay * self.direction_brier + (1.0 - decay) * dir_error
            raw_edge = 2.0 * probability_up - 1.0
            if abs(raw_edge) >= 0.02:
                hit = 1.0 if raw_edge * realized_bps > 0 else -1.0
                self.direction_alignment = (
                    alignment_decay * self.direction_alignment + (1.0 - alignment_decay) * hit
                )
            self.direction_count += 1

        scale = max(abs(realized_bps), threshold_bps, 1.0)
        ratio = min(abs(expected_abs_bps - abs(realized_bps)) / scale, 3.0)
        self.abs_mae_ratio = decay * self.abs_mae_ratio + (1.0 - decay) * ratio
        self.abs_count += 1

    def scores(self, config: V2Config) -> dict[str, float]:
        mag_skill = max(0.0, 1.0 - self.magnitude_brier / 0.25)
        dir_skill = max(0.0, 1.0 - self.direction_brier / 0.25)
        abs_skill = max(0.0, 1.0 - self.abs_mae_ratio)
        mag_ramp = min(
            1.0,
            math.sqrt(self.magnitude_count / max(config.min_matured_for_full_trust, 1)),
        )
        dir_ramp = min(
            1.0,
            math.sqrt(self.direction_count / max(config.min_big_direction_samples, 1)),
        )
        magnitude_trust = mag_skill * mag_ramp
        direction_trust = dir_skill * dir_ramp
        overall = 0.60 * magnitude_trust + 0.25 * direction_trust + 0.15 * abs_skill
        signed_alignment = self.direction_alignment * dir_ramp
        return {
            "magnitude_trust": float(np.clip(magnitude_trust, 0.0, 1.0)),
            "direction_trust": float(np.clip(direction_trust, 0.0, 1.0)),
            "abs_move_trust": float(np.clip(abs_skill, 0.0, 1.0)),
            "overall_trust": float(np.clip(overall, 0.0, 1.0)),
            "signed_alignment": float(np.clip(signed_alignment, -1.0, 1.0)),
            "magnitude_brier": self.magnitude_brier,
            "direction_brier": self.direction_brier,
            "matured": self.magnitude_count,
            "matured_big": self.direction_count,
        }


@dataclass
class _Pending:
    target_time: datetime
    vector: np.ndarray
    start_price: float
    probability_big: float
    probability_up: float
    expected_abs_bps: float


@dataclass
class _HorizonHead:
    horizon_minutes: int
    config: V2Config
    scaler: StandardScaler = field(default_factory=StandardScaler)
    magnitude_model: SGDClassifier = field(
        default_factory=lambda: SGDClassifier(
            loss="log_loss", penalty="l2", alpha=0.001, random_state=101
        )
    )
    direction_model: SGDClassifier = field(
        default_factory=lambda: SGDClassifier(
            loss="log_loss", penalty="l2", alpha=0.0015, random_state=103
        )
    )
    abs_model: SGDRegressor = field(
        default_factory=lambda: SGDRegressor(
            penalty="l2", alpha=0.001, learning_rate="invscaling", eta0=0.003, random_state=107
        )
    )
    pending: deque[_Pending] = field(default_factory=deque)
    validation: _ValidationState = field(default_factory=_ValidationState)
    sample_count: int = 0
    big_sample_count: int = 0
    _mag_initialized: bool = False
    _dir_initialized: bool = False
    _abs_initialized: bool = False

    @property
    def threshold_bps(self) -> float:
        return float(self.config.big_move_threshold_bps[self.horizon_minutes])

    def mature(self, timestamp: datetime, spy_price: float) -> None:
        while self.pending and self.pending[0].target_time <= timestamp:
            item = self.pending.popleft()
            if item.target_time.date() != timestamp.date() or item.start_price <= 0 or spy_price <= 0:
                continue
            realized_bps = (spy_price / item.start_price - 1.0) * 10_000.0
            x = item.vector.reshape(1, -1)
            self.scaler.partial_fit(x)
            z = self.scaler.transform(x)

            is_big = int(abs(realized_bps) >= self.threshold_bps)
            y_big = np.asarray([is_big], dtype=int)
            if not self._mag_initialized:
                self.magnitude_model.partial_fit(z, y_big, classes=np.asarray([0, 1], dtype=int))
                self._mag_initialized = True
            else:
                self.magnitude_model.partial_fit(z, y_big)

            if is_big:
                y_dir = np.asarray([1 if realized_bps > 0 else 0], dtype=int)
                if not self._dir_initialized:
                    self.direction_model.partial_fit(z, y_dir, classes=np.asarray([0, 1], dtype=int))
                    self._dir_initialized = True
                else:
                    self.direction_model.partial_fit(z, y_dir)
                self.big_sample_count += 1

            self.abs_model.partial_fit(z, np.asarray([abs(realized_bps)], dtype=float))
            self._abs_initialized = True
            self.sample_count += 1

            self.validation.update(
                probability_big=item.probability_big,
                probability_up=item.probability_up,
                expected_abs_bps=item.expected_abs_bps,
                realized_bps=realized_bps,
                threshold_bps=self.threshold_bps,
                decay=self.config.validation_decay,
                alignment_decay=self.config.alignment_decay,
            )

    def predict(self, vector: np.ndarray) -> dict[str, float | bool | int]:
        ready = self.sample_count >= self.config.min_samples and self._mag_initialized
        if ready:
            z = self.scaler.transform(vector.reshape(1, -1))
            probability_big = float(self.magnitude_model.predict_proba(z)[0, 1])
            probability_up = (
                float(self.direction_model.predict_proba(z)[0, 1])
                if self._dir_initialized and self.big_sample_count >= self.config.min_big_direction_samples
                else 0.5
            )
            expected_abs_bps = (
                max(0.0, float(self.abs_model.predict(z)[0])) if self._abs_initialized else 0.0
            )
        else:
            probability_big = 0.5
            probability_up = 0.5
            expected_abs_bps = 0.0

        scores = self.validation.scores(self.config)
        raw_edge = 2.0 * probability_up - 1.0
        validated_edge = raw_edge * scores["signed_alignment"]
        return {
            "horizon_minutes": self.horizon_minutes,
            "threshold_bps": self.threshold_bps,
            "probability_big": probability_big,
            "probability_up": probability_up,
            "expected_abs_bps": expected_abs_bps,
            "raw_direction_edge": raw_edge,
            "validated_direction_edge": validated_edge,
            "model_ready": ready,
            "sample_count": self.sample_count,
            **scores,
        }

    def queue(
        self,
        timestamp: datetime,
        vector: np.ndarray,
        spy_price: float,
        prediction: dict[str, float | bool | int],
    ) -> None:
        self.pending.append(
            _Pending(
                target_time=timestamp + timedelta(minutes=self.horizon_minutes),
                vector=vector.copy(),
                start_price=spy_price,
                probability_big=float(prediction["probability_big"]),
                probability_up=float(prediction["probability_up"]),
                expected_abs_bps=float(prediction["expected_abs_bps"]),
            )
        )
        while len(self.pending) > 5000:
            self.pending.popleft()


@dataclass(frozen=True)
class V2Opportunity:
    timestamp: datetime
    state: str
    eligible: bool
    probability_big_move: float
    probability_up: float
    expected_abs_bps: float
    validated_direction_edge: float
    trust: float
    agreement: float
    horizons: dict[int, dict[str, Any]]
    reasons: tuple[str, ...]
    model_version: str
    config_sha256: str
    strategy_authority: bool = False

    def as_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["timestamp"] = self.timestamp.isoformat().replace("+00:00", "Z")
        return data


class V2MTFStack:
    """Causal multi-timeframe predictor plus maturity-delayed validation compressor.

    The stack never uses an outcome until that horizon has actually matured. Direction is
    trained only on moves large enough to matter economically; magnitude and direction are
    separate heads. Signed alignment may suppress or invert a chronically backward horizon.
    """

    def __init__(self, config: V2Config | None = None):
        self.config = config or V2Config()
        self.heads = {
            horizon: _HorizonHead(horizon_minutes=horizon, config=self.config)
            for horizon in self.config.horizons
        }

    @staticmethod
    def _agreement(edges: list[float], weights: list[float]) -> float:
        if not edges:
            return 0.0
        w = np.asarray(weights, dtype=float)
        e = np.asarray(edges, dtype=float)
        if float(np.sum(w)) <= 0:
            return 0.0
        return float(abs(np.sum(w * np.sign(e))) / np.sum(w))

    def step(self, timestamp: datetime, factors: MarketFactors, spy_price: float) -> V2Opportunity:
        vector = vectorize(factors, feature_set=self.config.feature_set)
        for head in self.heads.values():
            head.mature(timestamp, spy_price)

        outputs: dict[int, dict[str, Any]] = {}
        rows: list[tuple[float, float, float, float, float]] = []
        edges: list[float] = []
        agreement_weights: list[float] = []
        for horizon in self.config.horizons:
            head = self.heads[horizon]
            pred = head.predict(vector)
            base_weight = float(self.config.base_weights[horizon])
            trust = float(pred["overall_trust"])
            weight = base_weight * (0.35 + 0.65 * trust)
            p_big = float(pred["probability_big"])
            edge = float(pred["validated_direction_edge"])
            expected_abs = float(pred["expected_abs_bps"])
            pred["weight"] = weight
            outputs[horizon] = pred
            rows.append((weight, p_big, edge, expected_abs, trust))
            edges.append(edge)
            agreement_weights.append(weight * p_big)

        total_weight = sum(row[0] for row in rows) or 1.0
        raw_big = sum(w * p for w, p, _, _, _ in rows) / total_weight
        total_direction_weight = sum(w * p for w, p, _, _, _ in rows)
        raw_edge = (
            sum(w * p * edge for w, p, edge, _, _ in rows) / total_direction_weight
            if total_direction_weight > 0
            else 0.0
        )
        expected_abs_bps = sum(w * move for w, _, _, move, _ in rows) / total_weight
        trust = sum(w * t for w, _, _, _, t in rows) / total_weight
        agreement = self._agreement(edges, agreement_weights)

        validated_big = 0.5 + (raw_big - 0.5) * trust
        validated_edge = raw_edge * (0.5 + 0.5 * agreement)
        probability_up = float(np.clip(0.5 + 0.5 * validated_edge, 0.0, 1.0))

        state = "NO_TRADE"
        eligible = False
        reasons: list[str] = []
        if trust < self.config.min_composite_trust:
            reasons.append("mtf_trust_below_threshold")
        elif validated_big <= self.config.quiet_max_validated_big_probability:
            state = "QUIET"
            eligible = True
            reasons.append("validated_quiet_state")
        elif validated_big >= self.config.min_validated_big_probability:
            if (
                abs(validated_edge) >= self.config.min_validated_direction_edge
                and agreement >= self.config.min_agreement
            ):
                state = "DIRECTIONAL_UP" if validated_edge > 0 else "DIRECTIONAL_DOWN"
                eligible = True
                reasons.append("validated_directional_state")
            else:
                state = "EXPANSION_UNCERTAIN"
                eligible = True
                reasons.append("validated_expansion_without_direction")
        else:
            reasons.append("no_validated_tradeable_state")

        opportunity = V2Opportunity(
            timestamp=timestamp,
            state=state,
            eligible=eligible,
            probability_big_move=float(np.clip(validated_big, 0.0, 1.0)),
            probability_up=probability_up,
            expected_abs_bps=max(0.0, expected_abs_bps),
            validated_direction_edge=float(np.clip(validated_edge, -1.0, 1.0)),
            trust=float(np.clip(trust, 0.0, 1.0)),
            agreement=float(np.clip(agreement, 0.0, 1.0)),
            horizons=outputs,
            reasons=tuple(reasons),
            model_version=V2_MODEL_VERSION,
            config_sha256=self.config.fingerprint(),
        )

        for horizon, head in self.heads.items():
            head.queue(timestamp, vector, spy_price, outputs[horizon])
        return opportunity
