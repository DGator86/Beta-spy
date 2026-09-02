from datetime import UTC, datetime, timedelta

from beta_spy.models import MarketFactors
from beta_spy.v2_mtf import V2MTFStack, _ValidationState


def factors(ts: datetime) -> MarketFactors:
    return MarketFactors(
        timestamp=ts,
        symbol_count=500,
        expected_symbol_count=500,
        coverage_ratio=0.99,
        covered_weight=0.99,
        trend_ew=0.2,
        trend_weighted=0.2,
        momentum_ew=0.2,
        momentum_weighted=0.2,
        volume_ew=0.1,
        volume_weighted=0.1,
        flow_ew=0.1,
        flow_weighted=0.1,
        volatility_ew=0.1,
        volatility_weighted=0.1,
        pct_above_vwap=0.6,
        pct_ema_bullish=0.6,
        pct_positive_5m=0.6,
        pct_buy_flow=0.6,
        participation=0.1,
        concentration=0.1,
        breadth_acceleration=0.0,
        spy_return_1m=0.0001,
        spy_return_5m=0.0003,
        spy_vwap_distance_bps=2.0,
        spy_flow=0.1,
        spy_quote_imbalance=0.1,
        spy_spread_bps=0.2,
    )


def test_maturity_delays_validation_until_horizon_is_observable():
    stack = V2MTFStack()
    t0 = datetime(2026, 8, 27, 14, 0, tzinfo=UTC)
    stack.step(t0, factors(t0), 700.0)
    assert stack.heads[15].validation.magnitude_count == 0

    t14 = t0 + timedelta(minutes=14)
    stack.step(t14, factors(t14), 700.2)
    assert stack.heads[15].validation.magnitude_count == 0

    t15 = t0 + timedelta(minutes=15)
    stack.step(t15, factors(t15), 700.7)
    assert stack.heads[15].validation.magnitude_count == 1


def test_backward_direction_can_acquire_negative_signed_alignment():
    state = _ValidationState()
    for _ in range(40):
        state.update(
            probability_big=0.8,
            probability_up=0.8,
            expected_abs_bps=10.0,
            realized_bps=-12.0,
            threshold_bps=7.5,
            decay=0.97,
            alignment_decay=0.95,
        )
    assert state.direction_alignment < 0.0


def test_beta_v2_never_claims_strategy_authority():
    stack = V2MTFStack()
    t0 = datetime(2026, 8, 27, 14, 0, tzinfo=UTC)
    opportunity = stack.step(t0, factors(t0), 700.0)
    assert opportunity.strategy_authority is False
