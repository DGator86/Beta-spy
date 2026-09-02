from __future__ import annotations

from datetime import UTC, datetime

from beta_spy.models import HorizonForecast, MarketFactors
from beta_spy.opportunity import BLIND_V1_CONFIG_SHA256, OpportunityGate


def _factors(timestamp: datetime, breadth: float) -> MarketFactors:
    return MarketFactors(
        timestamp=timestamp,
        symbol_count=500,
        expected_symbol_count=500,
        coverage_ratio=0.99,
        covered_weight=0.99,
        trend_ew=0.0,
        trend_weighted=0.0,
        momentum_ew=0.0,
        momentum_weighted=0.0,
        volume_ew=0.0,
        volume_weighted=0.0,
        flow_ew=0.0,
        flow_weighted=0.0,
        volatility_ew=0.10,
        volatility_weighted=0.10,
        pct_above_vwap=0.50,
        pct_ema_bullish=0.50,
        pct_positive_5m=breadth,
        pct_buy_flow=0.50,
        participation=0.50,
        concentration=0.10,
        breadth_acceleration=0.0,
        spy_return_1m=0.0,
        spy_return_5m=0.0,
        spy_vwap_distance_bps=0.0,
        spy_flow=None,
        spy_quote_imbalance=None,
        spy_spread_bps=1.0,
    )


def _forecast(horizon: int, probability: float, expected_bps: float) -> HorizonForecast:
    return HorizonForecast(
        horizon_minutes=horizon,
        probability_up=probability,
        expected_return_bps=expected_bps,
        confidence=abs(probability - 0.5) * 2.0,
        model_ready=True,
        sample_count=500,
    )


def test_blind_v1_gate_emits_opportunity_without_strategy() -> None:
    # 11:00 ET.  The 15m prior is strongly bearish and 30m agrees; 5m may
    # disagree.  Beta should emit a strategy-agnostic opportunity for Alpha.
    timestamp = datetime(2026, 8, 18, 15, 0, tzinfo=UTC)
    signal = OpportunityGate().evaluate(
        timestamp,
        _factors(timestamp, breadth=0.40),
        (
            _forecast(5, 0.55, 1.0),
            _forecast(15, 0.30, -6.0),
            _forecast(30, 0.25, -9.0),
        ),
    )
    assert signal.eligible is True
    assert signal.direction_prior == "DOWN"
    assert signal.supporting_horizons == 1
    assert signal.config_sha256 == BLIND_V1_CONFIG_SHA256
    assert not hasattr(signal, "structure")


def test_blind_v1_gate_rejects_small_move_even_with_direction() -> None:
    timestamp = datetime(2026, 8, 18, 15, 0, tzinfo=UTC)
    signal = OpportunityGate().evaluate(
        timestamp,
        _factors(timestamp, breadth=0.60),
        (
            _forecast(5, 0.70, 3.0),
            _forecast(15, 0.75, 4.9),
            _forecast(30, 0.70, 7.0),
        ),
    )
    assert signal.eligible is False
    assert "forecast_magnitude" in signal.reasons
