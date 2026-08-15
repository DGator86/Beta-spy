from __future__ import annotations

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
        min_probability: float = 0.58,
        min_coverage: float = 0.90,
        min_covered_weight: float = 0.85,
        max_spy_spread_bps: float = 4.0,
    ) -> None:
        self.primary_horizon = primary_horizon
        self.min_probability = min_probability
        self.min_coverage = min_coverage
        self.min_covered_weight = min_covered_weight
        self.max_spy_spread_bps = max_spy_spread_bps

    def decide(
        self,
        timestamp: datetime,
        factors: MarketFactors,
        forecasts: tuple[HorizonForecast, ...],
    ) -> Decision:
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
            risk_multiplier = float(min(max(edge / 0.10, 0.5), 2.0))
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
