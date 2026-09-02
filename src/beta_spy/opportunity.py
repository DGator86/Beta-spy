from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import time
from zoneinfo import ZoneInfo

from .models import HorizonForecast, MarketFactors

ET = ZoneInfo("America/New_York")
BLIND_V1_CONFIG_SHA256 = "ac67ca346e7fd069035e043b3e0e220b780732c01260a09299cdba6bc0a9ed56"


@dataclass(frozen=True)
class OpportunitySignal:
    """Strategy-agnostic handoff from Beta-spy to Alpha-Spy.

    Beta-spy is allowed to describe whether a 15-minute opportunity is strong
    enough to inspect and to expose its probabilistic state.  It deliberately
    does *not* choose an option structure or force Alpha-Spy to obey the
    directional prior.
    """

    timestamp: str
    eligible: bool
    direction_prior: str
    probability_up: float
    expected_return_bps: float
    supporting_horizons: int
    breadth_5: float | None
    reasons: tuple[str, ...]
    config_sha256: str = BLIND_V1_CONFIG_SHA256

    def as_dict(self) -> dict[str, object]:
        return asdict(self)


class OpportunityGate:
    """Frozen high-selectivity gate used by the first blind Beta->Alpha test.

    These values are intentionally constants for this research branch.  Do not
    retune them against Aug 18-25 outcomes: changing them creates a new model
    version and invalidates the blind-test lineage.
    """

    def __init__(
        self,
        *,
        primary_horizon: int = 15,
        min_probability: float = 0.65,
        min_expected_move_bps: float = 5.0,
        bull_breadth_5: float = 0.52,
        bear_breadth_5: float = 0.48,
        min_coverage: float = 0.90,
        min_covered_weight: float = 0.85,
        max_spy_spread_bps: float = 4.0,
        entry_start: time = time(10, 0),
        entry_stop: time = time(15, 40),
    ) -> None:
        self.primary_horizon = primary_horizon
        self.min_probability = min_probability
        self.min_expected_move_bps = min_expected_move_bps
        self.bull_breadth_5 = bull_breadth_5
        self.bear_breadth_5 = bear_breadth_5
        self.min_coverage = min_coverage
        self.min_covered_weight = min_covered_weight
        self.max_spy_spread_bps = max_spy_spread_bps
        self.entry_start = entry_start
        self.entry_stop = entry_stop

    def evaluate(
        self,
        timestamp,
        factors: MarketFactors,
        forecasts: tuple[HorizonForecast, ...],
    ) -> OpportunitySignal:
        by_horizon = {item.horizon_minutes: item for item in forecasts}
        primary = by_horizon.get(self.primary_horizon)
        if primary is None:
            return OpportunitySignal(
                timestamp=timestamp.isoformat(),
                eligible=False,
                direction_prior="FLAT",
                probability_up=0.5,
                expected_return_bps=0.0,
                supporting_horizons=0,
                breadth_5=factors.pct_positive_5m,
                reasons=("primary_forecast_missing",),
            )

        probability_up = float(primary.probability_up)
        if probability_up >= self.min_probability:
            direction = 1
            prior = "UP"
        elif probability_up <= 1.0 - self.min_probability:
            direction = -1
            prior = "DOWN"
        else:
            direction = 0
            prior = "FLAT"

        supporting = 0
        if direction:
            for horizon in (5, 30):
                item = by_horizon.get(horizon)
                if item is not None and item.direction == direction:
                    supporting += 1

        breadth = factors.pct_positive_5m
        if direction > 0:
            breadth_ok = breadth is not None and float(breadth) >= self.bull_breadth_5
        elif direction < 0:
            breadth_ok = breadth is not None and float(breadth) <= self.bear_breadth_5
        else:
            breadth_ok = False

        eastern = timestamp.astimezone(ET)
        clock = eastern.time().replace(tzinfo=None)
        window_ok = eastern.weekday() < 5 and self.entry_start <= clock < self.entry_stop
        spread_ok = factors.spy_spread_bps is None or factors.spy_spread_bps <= self.max_spy_spread_bps

        checks = {
            "coverage": factors.coverage_ratio >= self.min_coverage,
            "covered_weight": factors.covered_weight >= self.min_covered_weight,
            "directional_conviction": direction != 0,
            "forecast_magnitude": abs(float(primary.expected_return_bps)) >= self.min_expected_move_bps,
            "horizon_support": supporting >= 1,
            "breadth_5": breadth_ok,
            "spy_liquidity": spread_ok,
            "entry_window": window_ok,
        }
        reasons = tuple(name for name, passed in checks.items() if not passed)
        return OpportunitySignal(
            timestamp=timestamp.isoformat(),
            eligible=all(checks.values()),
            direction_prior=prior,
            probability_up=probability_up,
            expected_return_bps=float(primary.expected_return_bps),
            supporting_horizons=supporting,
            breadth_5=float(breadth) if breadth is not None else None,
            reasons=reasons,
        )
