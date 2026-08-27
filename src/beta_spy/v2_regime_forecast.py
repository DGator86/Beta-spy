from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

REGIMES = (
    "QUIET",
    "DIRECTIONAL_UP",
    "DIRECTIONAL_DOWN",
    "EXPANSION",
    "TRANSITION",
)


@dataclass(frozen=True)
class RegimeForecast:
    definable: bool
    current_regime: str
    confidence: float
    persistence_15: float
    persistence_30: float
    expected_duration_minutes: float
    successor_probabilities: dict[str, float]
    most_likely_successor: str
    successor_confidence: float
    reasons: tuple[str, ...]

    def as_dict(self) -> dict[str, Any]:
        return {
            "definable": self.definable,
            "current_regime": self.current_regime,
            "confidence": self.confidence,
            "persistence_15": self.persistence_15,
            "persistence_30": self.persistence_30,
            "expected_duration_minutes": self.expected_duration_minutes,
            "successor_probabilities": self.successor_probabilities,
            "most_likely_successor": self.most_likely_successor,
            "successor_confidence": self.successor_confidence,
            "reasons": list(self.reasons),
        }


def _clip(value: Any, default: float = 0.0) -> float:
    try:
        return float(np.clip(float(value), 0.0, 1.0))
    except (TypeError, ValueError):
        return default


def _normalize(raw: dict[str, float]) -> dict[str, float]:
    values = {name: max(0.0, float(raw.get(name, 0.0))) for name in REGIMES}
    total = sum(values.values())
    if total <= 0.0:
        return {name: (1.0 if name == "TRANSITION" else 0.0) for name in REGIMES}
    return {name: value / total for name, value in values.items()}


def _confidence(state: dict[str, Any]) -> float:
    effective = min(1.0, max(0.0, float(state.get("effective_analogs") or 0.0) / 40.0))
    proximity = min(1.0, max(0.0, float(state.get("mean_proximity") or 0.0) / 0.18))
    conformal = float(state.get("conformal_scale") or 1.10)
    calibration = 1.0 - min(1.0, abs(conformal - 1.0) / 0.60)
    samples = min(1.0, max(0.0, float(state.get("training_samples") or 0.0) / 600.0))
    return float(np.clip(0.35 * effective + 0.30 * proximity + 0.20 * calibration + 0.15 * samples, 0.0, 1.0))


def forecast_regime(state: dict[str, Any] | None) -> RegimeForecast:
    """Turn the predictive-state distribution into an explicit regime forecast.

    The underlying state model remains causal and analog-based. This layer does not
    choose an option strategy; it expresses Steps 1-4 of the trading decision chain:
    whether the regime is definable, how long it is likely to persist, and the
    probability distribution over successor regimes if it does not persist.
    """
    if not isinstance(state, dict) or not bool(state.get("ready")):
        return RegimeForecast(
            definable=False,
            current_regime="UNDEFINED",
            confidence=0.0,
            persistence_15=0.0,
            persistence_30=0.0,
            expected_duration_minutes=0.0,
            successor_probabilities={name: 0.0 for name in REGIMES},
            most_likely_successor="UNDEFINED",
            successor_confidence=0.0,
            reasons=("predictive_state_not_ready",),
        )

    regime = str(state.get("regime") or "TRANSITION")
    if regime not in REGIMES:
        regime = "TRANSITION"
    confidence = _confidence(state)
    pbig15 = _clip(state.get("p_big_15"))
    pbig30 = _clip(state.get("p_big_30"))
    pup15 = _clip(state.get("p_up_15"), 0.5)
    pup30 = _clip(state.get("p_up_30"), 0.5)
    reversal15 = _clip(state.get("p_reversal_15"))
    reversal30 = _clip(state.get("p_reversal_30"))
    sign_persistence = _clip(state.get("p_persistent_30"))
    acceleration = _clip(state.get("p_acceleration"))

    if regime == "QUIET":
        persist15 = 1.0 - pbig15
        persist30 = 1.0 - pbig30
        directional_mass = pbig30 * (0.45 + 0.35 * sign_persistence)
        expansion_mass = pbig30 * (1.0 - 0.55 * sign_persistence)
        raw = {
            "QUIET": persist30,
            "DIRECTIONAL_UP": directional_mass * pup30,
            "DIRECTIONAL_DOWN": directional_mass * (1.0 - pup30),
            "EXPANSION": expansion_mass,
            "TRANSITION": 0.10 + 0.20 * reversal30,
        }
    elif regime == "DIRECTIONAL_UP":
        persist15 = (1.0 - reversal15) * max(0.50, pup15)
        persist30 = sign_persistence * max(0.50, pup30)
        raw = {
            "DIRECTIONAL_UP": persist30,
            "DIRECTIONAL_DOWN": reversal30 * (1.0 - pup30),
            "EXPANSION": pbig30 * acceleration * max(0.25, pup30),
            "QUIET": (1.0 - pbig30) * (1.0 - persist30),
            "TRANSITION": max(0.05, 1.0 - persist30) * 0.60,
        }
    elif regime == "DIRECTIONAL_DOWN":
        persist15 = (1.0 - reversal15) * max(0.50, 1.0 - pup15)
        persist30 = sign_persistence * max(0.50, 1.0 - pup30)
        raw = {
            "DIRECTIONAL_DOWN": persist30,
            "DIRECTIONAL_UP": reversal30 * pup30,
            "EXPANSION": pbig30 * acceleration * max(0.25, 1.0 - pup30),
            "QUIET": (1.0 - pbig30) * (1.0 - persist30),
            "TRANSITION": max(0.05, 1.0 - persist30) * 0.60,
        }
    elif regime == "EXPANSION":
        persist15 = pbig15 * (0.55 + 0.30 * acceleration)
        persist30 = pbig30 * (0.45 + 0.35 * acceleration)
        directional_mass = pbig30 * sign_persistence
        raw = {
            "EXPANSION": persist30,
            "DIRECTIONAL_UP": directional_mass * pup30,
            "DIRECTIONAL_DOWN": directional_mass * (1.0 - pup30),
            "QUIET": 1.0 - pbig30,
            "TRANSITION": 0.15 + 0.35 * reversal30,
        }
    else:
        directional_strength = abs(pup30 - 0.5) * 2.0
        persist15 = max(0.10, 1.0 - pbig15 - 0.35 * directional_strength)
        persist30 = max(0.05, 1.0 - pbig30 - 0.45 * directional_strength)
        directional_mass = pbig30 * directional_strength
        raw = {
            "TRANSITION": persist30,
            "DIRECTIONAL_UP": directional_mass * pup30,
            "DIRECTIONAL_DOWN": directional_mass * (1.0 - pup30),
            "EXPANSION": pbig30 * (1.0 - directional_strength),
            "QUIET": 1.0 - pbig30,
        }

    persist15 = float(np.clip(persist15, 0.0, 1.0))
    persist30 = float(np.clip(persist30, 0.0, persist15 if persist15 > 0 else 1.0))
    successors = _normalize(raw)
    # Approximate expected survival time over the actionable 0-30 minute horizon.
    duration = float(np.clip(5.0 + 10.0 * persist15 + 15.0 * persist30, 5.0, 30.0))
    successor_candidates = {k: v for k, v in successors.items() if k != regime}
    successor = max(successor_candidates, key=successor_candidates.get, default="TRANSITION")
    successor_conf = float(successor_candidates.get(successor, 0.0))

    reasons: list[str] = []
    definable = bool(
        confidence >= 0.40
        and int(state.get("analog_count") or 0) >= 25
        and float(state.get("effective_analogs") or 0.0) >= 18.0
    )
    if not definable:
        reasons.append("regime_confidence_or_analog_support_insufficient")
    else:
        reasons.append("causal_predictive_state_supported")
    if duration < 10.0:
        reasons.append("short_expected_regime_duration")
    if successor_conf < 0.30:
        reasons.append("successor_regime_uncertain")

    return RegimeForecast(
        definable=definable,
        current_regime=regime,
        confidence=confidence,
        persistence_15=persist15,
        persistence_30=persist30,
        expected_duration_minutes=duration,
        successor_probabilities=successors,
        most_likely_successor=successor,
        successor_confidence=successor_conf,
        reasons=tuple(reasons),
    )
