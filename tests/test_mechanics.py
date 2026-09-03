from __future__ import annotations

from datetime import UTC, datetime, timedelta
import math

import numpy as np
import pytest

from beta_spy.mechanics import MechanicsEstimator
from beta_spy.models import FlowFeatures


def _flow(force_input: float) -> FlowFeatures:
    # With the other components at zero, composite force = 0.60 * OFI.
    return FlowFeatures(order_flow_imbalance=force_input)


def _run_synthetic(beta_up: float, beta_down: float, *, seed: int = 7):
    rng = np.random.default_rng(seed)
    ofi_values = rng.uniform(-0.95, 0.95, 220)
    estimator = MechanicsEstimator(window=120, min_samples=40, ridge=0.05)

    timestamp = datetime(2026, 1, 5, 14, 30, tzinfo=UTC)
    log_price = math.log(600.0)
    velocity = 0.0

    # Prime the estimator at t=0.
    previous_composite = 0.60 * float(ofi_values[0])
    state = estimator.step(timestamp, math.exp(log_price), _flow(float(ofi_values[0])))

    for i in range(1, len(ofi_values)):
        acceleration = (
            beta_up * max(previous_composite, 0.0)
            - beta_down * max(-previous_composite, 0.0)
            - 0.08 * velocity
        )
        velocity += acceleration
        log_price += velocity / 10_000.0
        timestamp += timedelta(minutes=1)
        state = estimator.step(timestamp, math.exp(log_price), _flow(float(ofi_values[i])))
        previous_composite = 0.60 * float(ofi_values[i])

    return state


def test_force_is_directional_and_bounded() -> None:
    positive = FlowFeatures(
        order_flow_imbalance=1.0,
        quote_imbalance=1.0,
        best_bid_replenishment=1.0,
        best_ask_withdrawal_rate=1.0,
    )
    negative = FlowFeatures(
        order_flow_imbalance=-1.0,
        quote_imbalance=-1.0,
        best_ask_replenishment=1.0,
        best_bid_withdrawal_rate=1.0,
    )

    pos_force, *_ = MechanicsEstimator.force_from_flow(positive)
    neg_force, *_ = MechanicsEstimator.force_from_flow(negative)

    assert 0.0 < pos_force <= 1.0
    assert -1.0 <= neg_force < 0.0


def test_recovers_directional_response_ordering() -> None:
    state = _run_synthetic(beta_up=3.0, beta_down=1.0)

    assert state.model_ready
    assert state.upside_response is not None
    assert state.downside_response is not None
    assert state.upside_inertia is not None
    assert state.downside_inertia is not None

    # More acceleration per unit bullish force means lower upside inertia.
    assert state.upside_response > state.downside_response
    assert state.upside_inertia < state.downside_inertia
    assert state.inertial_bias > 0.0


def test_higher_resistance_produces_higher_inertia() -> None:
    responsive = _run_synthetic(beta_up=4.0, beta_down=2.0, seed=11)
    resistant = _run_synthetic(beta_up=1.0, beta_down=0.5, seed=11)

    assert responsive.upside_inertia is not None
    assert resistant.upside_inertia is not None
    assert responsive.downside_inertia is not None
    assert resistant.downside_inertia is not None

    assert resistant.upside_inertia > responsive.upside_inertia
    assert resistant.downside_inertia > responsive.downside_inertia


def test_future_path_cannot_change_prefix_state() -> None:
    prefix = [0.4, -0.3, 0.7, -0.2] * 20
    left = MechanicsEstimator(window=60, min_samples=20)
    right = MechanicsEstimator(window=60, min_samples=20)

    timestamp = datetime(2026, 1, 5, 14, 30, tzinfo=UTC)
    price = 600.0
    state_left = state_right = None

    for i, ofi in enumerate(prefix):
        price *= math.exp((0.4 * ofi) / 10_000.0)
        stamp = timestamp + timedelta(minutes=i)
        state_left = left.step(stamp, price, _flow(ofi))
        state_right = right.step(stamp, price, _flow(ofi))

    assert state_left == state_right

    # Divergent future observations occur only after the prefix.
    left.step(timestamp + timedelta(minutes=len(prefix)), price * 1.002, _flow(0.9))
    right.step(timestamp + timedelta(minutes=len(prefix)), price * 0.998, _flow(-0.9))

    # The previously produced state is immutable and identical.
    assert state_left == state_right


def test_irregular_gap_does_not_create_fake_response_row() -> None:
    estimator = MechanicsEstimator(window=20, min_samples=8)
    stamp = datetime(2026, 1, 5, 14, 30, tzinfo=UTC)
    price = 600.0

    estimator.step(stamp, price, _flow(0.5))
    price *= math.exp(0.5 / 10_000.0)
    estimator.step(stamp + timedelta(minutes=1), price, _flow(-0.4))
    price *= math.exp(-0.2 / 10_000.0)
    before_gap = estimator.step(stamp + timedelta(minutes=2), price, _flow(0.3))
    assert before_gap.sample_count == 1

    price *= math.exp(1.0 / 10_000.0)
    gap_state = estimator.step(stamp + timedelta(minutes=7), price, _flow(0.2))
    assert gap_state.sample_count == before_gap.sample_count
    assert gap_state.acceleration_bps is None

    price *= math.exp(0.2 / 10_000.0)
    after_gap = estimator.step(stamp + timedelta(minutes=8), price, _flow(-0.1))
    assert after_gap.sample_count == before_gap.sample_count + 1


def test_new_session_resets_rolling_response_state() -> None:
    estimator = MechanicsEstimator(window=20, min_samples=8)
    stamp = datetime(2026, 1, 5, 14, 30, tzinfo=UTC)
    price = 600.0

    for i in range(12):
        price *= math.exp((0.2 if i % 2 == 0 else -0.1) / 10_000.0)
        state = estimator.step(stamp + timedelta(minutes=i), price, _flow(0.5 if i % 2 == 0 else -0.4))
    assert state.sample_count > 0

    next_day = estimator.step(stamp + timedelta(days=1), price, _flow(0.2))
    assert next_day.sample_count == 0
    assert next_day.velocity_bps is None
    assert next_day.acceleration_bps is None


def test_timestamps_must_increase() -> None:
    estimator = MechanicsEstimator(window=20, min_samples=8)
    stamp = datetime(2026, 1, 5, 14, 30, tzinfo=UTC)
    estimator.step(stamp, 600.0, _flow(0.1))
    with pytest.raises(ValueError, match="strictly increasing"):
        estimator.step(stamp, 600.1, _flow(0.1))
