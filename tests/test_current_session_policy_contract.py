from datetime import UTC, datetime

from beta_spy.decision import DecisionEngine
from beta_spy.models import HorizonForecast, MarketFactors


def _forecasts(probability_up: float, expected_bps: float):
    return tuple(
        HorizonForecast(
            horizon_minutes=horizon,
            probability_up=probability_up,
            expected_return_bps=expected_bps,
            confidence=0.5,
            model_ready=True,
            sample_count=500,
        )
        for horizon in (5, 15, 30)
    )


def _factors(timestamp: datetime, trend: float):
    return MarketFactors(
        timestamp=timestamp,
        symbol_count=500,
        expected_symbol_count=500,
        coverage_ratio=0.99,
        covered_weight=0.99,
        trend_ew=trend,
        trend_weighted=trend,
        momentum_ew=trend,
        momentum_weighted=trend,
        volume_ew=0.0,
        volume_weighted=0.0,
        flow_ew=trend,
        flow_weighted=trend,
        volatility_ew=0.10,
        volatility_weighted=0.10,
        pct_above_vwap=0.2 if trend < 0 else 0.8,
        pct_ema_bullish=0.2 if trend < 0 else 0.8,
        pct_positive_5m=0.2 if trend < 0 else 0.8,
        pct_buy_flow=0.2 if trend < 0 else 0.8,
        participation=trend,
        concentration=0.1,
        breadth_acceleration=0.0,
        spy_return_1m=-0.0002 if trend < 0 else 0.0002,
        spy_return_5m=-0.001 if trend < 0 else 0.001,
        spy_vwap_distance_bps=-8.0 if trend < 0 else 8.0,
        spy_flow=trend,
        spy_quote_imbalance=-0.2 if trend < 0 else 0.2,
        spy_spread_bps=1.0,
    )


def test_early_bearish_forecast_waits_without_tradeable_impulse():
    engine = DecisionEngine()
    open_ts = datetime(2026, 8, 17, 13, 30, tzinfo=UTC)
    later = datetime(2026, 8, 17, 14, 10, tzinfo=UTC)  # 40 minutes after open

    engine.decide(
        open_ts,
        _factors(open_ts, -0.6),
        _forecasts(0.34, -8.0),
        spy_price=776.18,
    )
    decision = engine.decide(
        later,
        _factors(later, -0.6),
        _forecasts(0.34, -8.0),
        spy_price=775.40,
    )

    assert decision.action == "NO_TRADE"
    assert decision.gates["session_bias"] is True
    assert decision.gates["situation"] is False
    assert "Situation WATCH" in decision.reasons[0]
