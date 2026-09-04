from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from datetime import datetime
import math

import numpy as np

from .mechanics import MechanicsEstimator
from .models import FlowFeatures


@dataclass(frozen=True)
class QuadrantFit:
    response: float | None
    inertia: float | None
    samples: int
    r_squared: float | None


@dataclass(frozen=True)
class MechanicsV2State:
    """Causal four-quadrant Market Mechanics state.

    V2 keeps the frozen V1 force definition and changes only the response
    hypothesis. Response is conditioned jointly on the sign of existing
    velocity and the sign of applied force.

    Quadrants:
      pp: v >= 0, F > 0   (uptrend launch / continuation)
      pm: v >= 0, F < 0   (uptrend braking)
      mp: v < 0,  F > 0   (downtrend braking)
      mm: v < 0,  F < 0   (downtrend launch / continuation)

    Force remains signed in every fit. A valid beta is therefore expected to be
    positive in every quadrant. No absolute-value or sign-flip rescue is used.
    """

    timestamp: datetime
    log_price: float
    velocity_bps: float | None
    acceleration_bps: float | None
    force: float
    force_ofi: float
    force_quote: float
    force_liquidity: float
    beta_pp: float | None
    beta_pm: float | None
    beta_mp: float | None
    beta_mm: float | None
    launch_inertia_up: float | None
    braking_inertia_up: float | None
    braking_inertia_down: float | None
    launch_inertia_down: float | None
    pp_samples: int
    pm_samples: int
    mp_samples: int
    mm_samples: int
    braking_ready: bool
    full_quadrant_ready: bool


class MechanicsV2Estimator:
    """Market Mechanics V2: launch inertia is distinct from braking inertia.

    For each rolling window, response is estimated separately inside the four
    velocity-sign x force-sign quadrants using only causal rows::

        a_t = alpha_q + beta_q * F_{t-1} - gamma_q * v_{t-1} + error_t

    The signed force is retained. Thus a physically-consistent response under
    the V2 hypothesis has ``beta_q > 0`` in every quadrant.

    Derived inertias are:

        M_launch_up   = 1 / beta_pp
        M_brake_up    = 1 / beta_pm
        M_brake_down  = 1 / beta_mp
        M_launch_down = 1 / beta_mm

    This module is research-only and has no execution authority.
    """

    def __init__(
        self,
        *,
        window: int = 120,
        min_quadrant_samples: int = 12,
        ridge: float = 0.25,
        min_response: float = 1e-3,
        regular_interval_tolerance: float = 0.50,
    ) -> None:
        if window < 20:
            raise ValueError("window must be >= 20")
        if min_quadrant_samples < 6 or min_quadrant_samples > window:
            raise ValueError("min_quadrant_samples must be between 6 and window")
        if ridge < 0.0:
            raise ValueError("ridge must be nonnegative")
        if min_response < 0.0:
            raise ValueError("min_response must be nonnegative")
        if not 0.0 <= regular_interval_tolerance < 1.0:
            raise ValueError("regular_interval_tolerance must be in [0, 1)")

        self.window = int(window)
        self.min_quadrant_samples = int(min_quadrant_samples)
        self.ridge = float(ridge)
        self.min_response = float(min_response)
        self.regular_interval_tolerance = float(regular_interval_tolerance)

        self._rows: deque[tuple[float, float, float]] = deque(maxlen=self.window)
        self._last_log_price: float | None = None
        self._last_velocity: float | None = None
        self._last_force: float | None = None
        self._last_timestamp: datetime | None = None

    def reset(self) -> None:
        self._rows.clear()
        self._last_log_price = None
        self._last_velocity = None
        self._last_force = None
        self._last_timestamp = None

    @staticmethod
    def force_from_flow(flow: FlowFeatures) -> tuple[float, float, float, float]:
        """Reuse the frozen V1 force exactly; V2 tests only response structure."""

        return MechanicsEstimator.force_from_flow(flow)

    def step(self, timestamp: datetime, price: float, flow: FlowFeatures) -> MechanicsV2State:
        if price <= 0.0 or not math.isfinite(price):
            raise ValueError("price must be finite and positive")
        if self._last_timestamp is not None and timestamp <= self._last_timestamp:
            raise ValueError("timestamps must be strictly increasing")
        if self._last_timestamp is not None and timestamp.date() != self._last_timestamp.date():
            self.reset()

        log_price = math.log(price)
        force, force_ofi, force_quote, force_liquidity = self.force_from_flow(flow)

        velocity: float | None = None
        acceleration: float | None = None
        dt_minutes: float | None = None
        regular_interval = False

        if self._last_timestamp is not None:
            dt_minutes = (timestamp - self._last_timestamp).total_seconds() / 60.0
            regular_interval = abs(dt_minutes - 1.0) <= self.regular_interval_tolerance

        if self._last_log_price is not None and dt_minutes is not None and dt_minutes > 0.0:
            velocity = (log_price - self._last_log_price) * 10_000.0 / dt_minutes

        if regular_interval and velocity is not None and self._last_velocity is not None:
            acceleration = (velocity - self._last_velocity) / max(dt_minutes or 1.0, 1e-9)
            if self._last_force is not None:
                self._rows.append((self._last_force, self._last_velocity, acceleration))

        fits = self._fit_quadrants()

        state = MechanicsV2State(
            timestamp=timestamp,
            log_price=log_price,
            velocity_bps=velocity,
            acceleration_bps=acceleration,
            force=force,
            force_ofi=force_ofi,
            force_quote=force_quote,
            force_liquidity=force_liquidity,
            beta_pp=fits["pp"].response,
            beta_pm=fits["pm"].response,
            beta_mp=fits["mp"].response,
            beta_mm=fits["mm"].response,
            launch_inertia_up=fits["pp"].inertia,
            braking_inertia_up=fits["pm"].inertia,
            braking_inertia_down=fits["mp"].inertia,
            launch_inertia_down=fits["mm"].inertia,
            pp_samples=fits["pp"].samples,
            pm_samples=fits["pm"].samples,
            mp_samples=fits["mp"].samples,
            mm_samples=fits["mm"].samples,
            braking_ready=fits["pm"].inertia is not None and fits["mp"].inertia is not None,
            full_quadrant_ready=all(fit.inertia is not None for fit in fits.values()),
        )

        self._last_log_price = log_price
        if velocity is not None:
            self._last_velocity = velocity
        self._last_force = force
        self._last_timestamp = timestamp
        return state

    def _fit_quadrants(self) -> dict[str, QuadrantFit]:
        rows = list(self._rows)
        quadrants = {
            "pp": [row for row in rows if row[1] >= 0.0 and row[0] > 0.0],
            "pm": [row for row in rows if row[1] >= 0.0 and row[0] < 0.0],
            "mp": [row for row in rows if row[1] < 0.0 and row[0] > 0.0],
            "mm": [row for row in rows if row[1] < 0.0 and row[0] < 0.0],
        }
        return {name: self._fit_one(subset) for name, subset in quadrants.items()}

    def _fit_one(self, rows: list[tuple[float, float, float]]) -> QuadrantFit:
        n = len(rows)
        if n < self.min_quadrant_samples:
            return QuadrantFit(response=None, inertia=None, samples=n, r_squared=None)

        data = np.asarray(rows, dtype=float)
        force = data[:, 0]
        velocity = data[:, 1]
        acceleration = data[:, 2]

        if float(np.std(force, ddof=1)) <= 1e-12:
            return QuadrantFit(response=None, inertia=None, samples=n, r_squared=None)

        x = np.column_stack((np.ones(n), force, -velocity))
        penalty = np.eye(x.shape[1]) * self.ridge
        penalty[0, 0] = 0.0
        try:
            coef = np.linalg.solve(x.T @ x + penalty, x.T @ acceleration)
        except np.linalg.LinAlgError:
            coef = np.linalg.pinv(x.T @ x + penalty) @ (x.T @ acceleration)

        beta = float(coef[1])
        prediction = x @ coef
        target_mean = float(np.mean(acceleration))
        ss_tot = float(np.sum((acceleration - target_mean) ** 2))
        ss_res = float(np.sum((acceleration - prediction) ** 2))
        r_squared = None if ss_tot <= 1e-15 else 1.0 - ss_res / ss_tot

        if not math.isfinite(beta) or beta <= self.min_response:
            return QuadrantFit(response=None, inertia=None, samples=n, r_squared=r_squared)
        return QuadrantFit(
            response=beta,
            inertia=1.0 / beta,
            samples=n,
            r_squared=r_squared,
        )
