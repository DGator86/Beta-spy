from __future__ import annotations

from collections import deque
from datetime import datetime

from .models import Decision, HorizonForecast, MarketFactors


def _sign(value: float | None, deadband: float = 0.05) -> int:
    if value is None or abs(value) < deadband:
        return 0
    return 1 if value > 0 else -1


class DecisionEngine:
    def __init__(
        self,
        *,
        primary_horizon: int = 15,
        # Calibrated probabilities: held-out accuracy and per-trade mean both
        # rise with the threshold while total sized P&L stays flat, so fewer,
        # better trades win once options friction is counted.
        min_probability: float = 0.60,
        min_coverage: float = 0.90,
        min_covered_weight: float = 0.85,
        max_spy_spread_bps: float = 4.0,
        neutral_premium_enabled: bool = True,
        # Calibrated-probability scale: edges are compressed, so "no edge"
        # must be much tighter than it was on raw probabilities.
        neutral_max_edge: float = 0.05,
        quiet_return_threshold: float = 0.00012,
        quiet_volatility_threshold: float = 0.20,
        quiet_window_minutes: int = 15,
        # Reference tape activity for regime sizing; deliberately separate
        # from the neutral-gate threshold so tuning one does not move the other.
        regime_reference_return: float = 0.00016,
    ) -> None:
        self.primary_horizon = primary_horizon
        self.min_probability = min_probability
        self.min_coverage = min_coverage
        self.min_covered_weight = min_covered_weight
        self.max_spy_spread_bps = max_spy_spread_bps
        self.neutral_premium_enabled = neutral_premium_enabled
        self.neutral_max_edge = neutral_max_edge
        self.quiet_return_threshold = quiet_return_threshold
        self.quiet_volatility_threshold = quiet_volatility_threshold
        self.quiet_window_minutes = quiet_window_minutes
        self.regime_reference_return = regime_reference_return
        self._recent_returns: deque[float] = deque(maxlen=quiet_window_minutes)
        self._last_session: object = None

    def decide(
        self,
        timestamp: datetime,
        factors: MarketFactors,
        forecasts: tuple[HorizonForecast, ...],
    ) -> Decision:
        if timestamp.date() != self._last_session:
            self._recent_returns.clear()
            self._last_session = timestamp.date()
        if factors.spy_return_1m is not None:
            self._recent_returns.append(abs(float(factors.spy_return_1m)))
        by_horizon = {forecast.horizon_minutes: forecast for forecast in forecasts}
        primary = by_horizon.get(self.primary_horizon)
        if primary is None:
            return Decision(
                timestamp=timestamp,
                action="NO_TRADE",
                direction="FLAT",
                confidence=0.0,
                score=0.0,
                primary_horizon=self.primary_horizon,
                gates={"primary_forecast": False},
                reasons=("Primary forecast is unavailable",),
            )
        bullish = primary.probability_up >= self.min_probability
        bearish = primary.probability_up <= 1.0 - self.min_probability
        direction = 1 if bullish else -1 if bearish else 0
        # Only the fast horizons vote: the 30m model tested at coin-flip
        # accuracy and must not confirm or veto a trade.
        fast_horizons = [
            forecast
            for forecast in forecasts
            if forecast.horizon_minutes <= self.primary_horizon
        ]
        agreement = (
            bool(direction)
            and len(fast_horizons) >= 2
            and all(
                forecast.confidence > 0.05 and forecast.direction == direction
                for forecast in fast_horizons
            )
        )

        breadth_signals = [
            _sign(factors.trend_ew),
            _sign(factors.trend_weighted),
            _sign(factors.momentum_ew),
            _sign(factors.momentum_weighted),
            _sign(factors.participation),
        ]
        known_breadth = [item for item in breadth_signals if item != 0]
        breadth_confirm = (
            sum(item == direction for item in known_breadth) >= max(2, len(known_breadth) // 2 + 1)
            if direction and known_breadth
            else False
        )
        flow_signals = [_sign(factors.flow_ew), _sign(factors.flow_weighted), _sign(factors.spy_flow)]
        known_flow = [item for item in flow_signals if item != 0]
        flow_confirm = True if not known_flow else sum(item == direction for item in known_flow) >= 2
        liquidity_ok = factors.spy_spread_bps is None or factors.spy_spread_bps <= self.max_spy_spread_bps
        gates = {
            "coverage": factors.coverage_ratio >= self.min_coverage,
            "covered_weight": factors.covered_weight >= self.min_covered_weight,
            "directional_edge": direction != 0,
            "multi_horizon": agreement,
            "breadth_confirmation": breadth_confirm,
            "flow_confirmation": flow_confirm,
            "spy_liquidity": liquidity_ok,
        }
        reasons: list[str] = []
        labels = {
            "coverage": "Universe coverage below threshold",
            "covered_weight": "Covered SPY weight below threshold",
            "directional_edge": "15-minute forecast lacks directional edge",
            "multi_horizon": "5/15/30-minute forecasts do not agree",
            "breadth_confirmation": "Constituent breadth contradicts the forecast",
            "flow_confirmation": "Tape/order-flow breadth contradicts the forecast",
            "spy_liquidity": "SPY spread is too wide",
        }
        for key, passed in gates.items():
            if not passed:
                reasons.append(labels[key])

        score_components = [
            (abs(primary.probability_up - 0.5) * 2.0, 0.35),
            (primary.confidence, 0.20),
            (abs(factors.trend_weighted or 0.0), 0.15),
            (abs(factors.momentum_weighted or 0.0), 0.10),
            (abs(factors.flow_weighted or 0.0), 0.10),
            (abs(factors.participation or 0.0), 0.10),
        ]
        score = sum(value * weight for value, weight in score_components)
        if all(gates.values()):
            side = "BULLISH" if direction > 0 else "BEARISH"
            structure = "CALL_DEBIT_SPREAD" if direction > 0 else "PUT_DEBIT_SPREAD"
            edge = abs(primary.probability_up - 0.5)
            edge_multiplier = min(max(edge / 0.10, 0.5), 2.0)
            # The directional edge historically scales with tape activity
            # (roughly 3x larger in volatile weeks than quiet ones), so the
            # risk budget leans into active tape and shrinks in dead tape.
            # Validated out-of-sample on top of edge sizing.
            regime_multiplier = 1.0
            if self._recent_returns:
                recent_mean = sum(self._recent_returns) / len(self._recent_returns)
                regime_multiplier = min(max(recent_mean / self.regime_reference_return, 0.75), 1.5)
            risk_multiplier = float(min(max(edge_multiplier * regime_multiplier, 0.5), 2.5))
            return Decision(
                timestamp=timestamp,
                action="TRADE",
                direction=side,
                confidence=primary.confidence,
                score=score,
                primary_horizon=self.primary_horizon,
                gates=gates,
                reasons=("Constituent breadth, flow, and forecast stack are aligned",),
                structure=structure,
                risk_multiplier=risk_multiplier,
            )
        neutral = self._neutral_premium_decision(timestamp, factors, forecasts, gates)
        if neutral is not None:
            return neutral
        return Decision(
            timestamp=timestamp,
            action="NO_TRADE",
            direction="BULLISH" if direction > 0 else "BEARISH" if direction < 0 else "FLAT",
            confidence=primary.confidence,
            score=score,
            primary_horizon=self.primary_horizon,
            gates=gates,
            reasons=tuple(reasons),
        )

    def _neutral_premium_decision(
        self,
        timestamp: datetime,
        factors: MarketFactors,
        forecasts: tuple[HorizonForecast, ...],
        directional_gates: dict[str, bool],
    ) -> Decision | None:
        """Sell defined-risk premium (iron condor) only on a genuinely quiet tape.

        Model neutrality alone is anti-predictive: when the forecasts sit near
        0.5, realized moves are historically about twice normal size. Premium
        is therefore sold only when there is no directional edge AND the tape
        itself has been quiet — small recent SPY returns and low weighted
        constituent volatility — which did select small forward moves in and
        out of sample.
        """
        if not self.neutral_premium_enabled:
            return None
        # Only the fast horizons define "no edge"; the 30m model's spurious
        # edges would otherwise block nearly every quiet window.
        max_edge = max(
            (
                abs(f.probability_up - 0.5)
                for f in forecasts
                if f.horizon_minutes <= self.primary_horizon
            ),
            default=1.0,
        )
        window_full = len(self._recent_returns) >= max(self.quiet_window_minutes - 5, 5)
        recent_mean = (
            sum(self._recent_returns) / len(self._recent_returns) if self._recent_returns else None
        )
        gates = {
            "coverage": directional_gates.get("coverage", False),
            "covered_weight": directional_gates.get("covered_weight", False),
            "spy_liquidity": directional_gates.get("spy_liquidity", False),
            "no_directional_edge": max_edge < self.neutral_max_edge,
            "quiet_tape": bool(
                window_full
                and recent_mean is not None
                and recent_mean <= self.quiet_return_threshold
                and factors.volatility_weighted is not None
                and factors.volatility_weighted <= self.quiet_volatility_threshold
            ),
        }
        if not all(gates.values()):
            return None
        return Decision(
            timestamp=timestamp,
            action="TRADE_NEUTRAL",
            direction="NEUTRAL",
            confidence=1.0 - max_edge * 2.0,
            score=0.0,
            primary_horizon=self.primary_horizon,
            gates=gates,
            reasons=("Quiet tape with no directional edge: sell defined-risk premium",),
            structure="IRON_CONDOR",
            risk_multiplier=0.5,
        )
