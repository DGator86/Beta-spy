from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from datetime import datetime
import math

import numpy as np

from .models import FlowFeatures


@dataclass(frozen=True)
class MechanicsState:
    """Causal SPY Market Mechanics state.

    All values use information available at or before ``timestamp``. ``force`` is
    dimensionless. Velocity and acceleration are log-return basis points per
    minute and basis points per minute squared, respectively. Effective inertia
    is an inverse response coefficient, not physical mass.
    """

    timestamp: datetime
    log_price: float
    velocity_bps: float | None
    acceleration_bps: float | None
    force: float
    force_ofi: float
    force_quote: float
    force_liquidity: float
    upside_response: float | None
    downside_response: float | None
    upside_inertia: float | None
    downside_inertia: float | None
    inertial_bias: float | None
    momentum: float | None
    impulse: float
    sample_count: int
    model_ready: bool


class MechanicsEstimator:
    """Minimum viable effective-inertia estimator for SPY.

    Model estimated causally from prior-minute pressure and subsequent
    acceleration::

        a_t = alpha
            + beta_up * max(F_{t-1}, 0)
            - beta_down * max(-F_{t-1}, 0)
            - gamma * v_{t-1}
            + error_t

    ``M_up = 1 / beta_up`` and ``M_down = 1 / beta_down`` when the fitted
    response has the expected sign and enough samples exist.

    The estimator is explicitly intraday. A new UTC date resets the rolling
    response state, and irregular timestamp gaps are never converted into a
    fake one-minute response observation.

    The force model is deliberately small. It uses only tape/top-of-book
    quantities already available in Beta-spy and does not use future prices,
    option outcomes, or broker/execution information.
    """

    def __init__(
        self,
        *,
        window: int = 120,
        min_samples: int = 30,
        ridge: float = 0.25,
        impulse_decay: float = 0.90,
        min_response: float = 1e-3,
        regular_interval_tolerance: float = 0.50,
    ) -> None:
        if window < 10:
            raise ValueError("window must be >= 10")
        if min_samples < 8 or min_samples > window:
            raise ValueError("min_samples must be between 8 and window")
        if not 0.0 <= impulse_decay < 1.0:
            raise ValueError("impulse_decay must be in [0, 1)")
        if not 0.0 <= regular_interval_tolerance < 1.0:
            raise ValueError("regular_interval_tolerance must be in [0, 1)")
        self.window = int(window)
        self.min_samples = int(min_samples)
        self.ridge = float(ridge)
        self.impulse_decay = float(impulse_decay)
        self.min_response = float(min_response)
        self.regular_interval_tolerance = float(regular_interval_tolerance)
        self._rows: deque[tuple[float, float, float]] = deque(maxlen=self.window)
        self._last_log_price: float | None = None
        self._last_velocity: float | None = None
        self._last_force: float | None = None
        self._last_timestamp: datetime | None = None
        self._impulse = 0.0

    def reset(self) -> None:
        """Reset all intraday dynamics without changing configuration."""

        self._rows.clear()
        self._last_log_price = None
        self._last_velocity = None
        self._last_force = None
        self._last_timestamp = None
        self._impulse = 0.0

    @staticmethod
    def force_from_flow(flow: FlowFeatures) -> tuple[float, float, float, float]:
        """Return composite force plus its three bounded components.

        Components are aggressor order-flow imbalance, average top-of-book quote
        imbalance, and liquidity-support asymmetry from replenishment/withdrawal.
        Fixed MVP weights are hypotheses to validate, not calibrated truths.
        """

        ofi = _clip(flow.order_flow_imbalance)
        quote = _clip(flow.quote_imbalance)

        bid_support = _nz(flow.best_bid_replenishment) + _nz(flow.best_ask_withdrawal_rate)
        ask_support = _nz(flow.best_ask_replenishment) + _nz(flow.best_bid_withdrawal_rate)
        liquidity = _clip((bid_support - ask_support) / 2.0)

        force = _clip(0.60 * ofi + 0.25 * quote + 0.15 * liquidity)
        return force, ofi, quote, liquidity

    def step(self, timestamp: datetime, price: float, flow: FlowFeatures) -> MechanicsState:
        if price <= 0 or not math.isfinite(price):
            raise ValueError("price must be finite and positive")
        if self._last_timestamp is not None and timestamp <= self._last_timestamp:
            raise ValueError("timestamps must be strictly increasing")
        if self._last_timestamp is not None and timestamp.date() != self._last_timestamp.date():
            self.reset()

        log_price = math.log(price)
        force, force_ofi, force_quote, force_liquidity = self.force_from_flow(flow)
        self._impulse = self.impulse_decay * self._impulse + force

        velocity: float | None = None
        acceleration: float | None = None
        dt_minutes: float | None = None
        regular_interval = False

        if self._last_timestamp is not None:
            dt_minutes = (timestamp - self._last_timestamp).total_seconds() / 60.0
            regular_interval = abs(dt_minutes - 1.0) <= self.regular_interval_tolerance

        if self._last_log_price is not None and dt_minutes is not None and dt_minutes > 0:
            velocity = (log_price - self._last_log_price) * 10_000.0 / dt_minutes

        # Acceleration is only defined for the fitted MVP on a regular minute
        # step. A multi-minute gap is a missing-observation interval, not one
        # enormous one-minute acceleration sample.
        if regular_interval and velocity is not None and self._last_velocity is not None:
            acceleration = (velocity - self._last_velocity) / max(dt_minutes or 1.0, 1e-9)
            if self._last_force is not None:
                # F_{t-1} and v_{t-1} explain a_t. No t+1 observation enters.
                self._rows.append((self._last_force, self._last_velocity, acceleration))

        beta_up, beta_down = self._fit_responses()
        upside_inertia = _inertia(beta_up, self.min_response)
        downside_inertia = _inertia(beta_down, self.min_response)

        inertial_bias = None
        if upside_inertia is not None and downside_inertia is not None:
            denom = upside_inertia + downside_inertia
            if denom > 0:
                inertial_bias = (downside_inertia - upside_inertia) / denom

        momentum = None
        if velocity is not None:
            active_inertia = upside_inertia if velocity >= 0 else downside_inertia
            if active_inertia is not None:
                momentum = active_inertia * velocity

        sample_count = len(self._rows)
        state = MechanicsState(
            timestamp=timestamp,
            log_price=log_price,
            velocity_bps=velocity,
            acceleration_bps=acceleration,
            force=force,
            force_ofi=force_ofi,
            force_quote=force_quote,
            force_liquidity=force_liquidity,
            upside_response=beta_up,
            downside_response=beta_down,
            upside_inertia=upside_inertia,
            downside_inertia=downside_inertia,
            inertial_bias=inertial_bias,
            momentum=momentum,
            impulse=self._impulse,
            sample_count=sample_count,
            model_ready=sample_count >= self.min_samples
            and upside_inertia is not None
            and downside_inertia is not None,
        )

        self._last_log_price = log_price
        if velocity is not None:
            self._last_velocity = velocity
        self._last_force = force
        self._last_timestamp = timestamp
        return state

    def _fit_responses(self) -> tuple[float | None, float | None]:
        if len(self._rows) < self.min_samples:
            return None, None

        rows = np.asarray(self._rows, dtype=float)
        force = rows[:, 0]
        previous_velocity = rows[:, 1]
        acceleration = rows[:, 2]

        # Coefficients correspond to [intercept, beta_up, beta_down, gamma].
        x = np.column_stack(
            (
                np.ones(len(rows)),
                np.maximum(force, 0.0),
                -np.maximum(-force, 0.0),
                -previous_velocity,
            )
        )
        penalty = np.eye(x.shape[1]) * self.ridge
        penalty[0, 0] = 0.0
        try:
            coef = np.linalg.solve(x.T @ x + penalty, x.T @ acceleration)
        except np.linalg.LinAlgError:
            coef = np.linalg.pinv(x.T @ x + penalty) @ (x.T @ acceleration)

        beta_up = float(coef[1])
        beta_down = float(coef[2])

        # A negative fitted response violates the current hypothesis in this
        # window. Report no inertia rather than taking abs() and inventing mass.
        if not math.isfinite(beta_up) or beta_up <= self.min_response:
            beta_up = None
        if not math.isfinite(beta_down) or beta_down <= self.min_response:
            beta_down = None
        return beta_up, beta_down


def _inertia(response: float | None, floor: float) -> float | None:
    if response is None or response <= floor:
        return None
    return 1.0 / response


def _nz(value: float | None) -> float:
    if value is None or not math.isfinite(value):
        return 0.0
    return float(value)


def _clip(value: float | None) -> float:
    if value is None or not math.isfinite(value):
        return 0.0
    return float(max(-1.0, min(1.0, value)))
