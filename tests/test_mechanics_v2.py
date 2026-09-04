from __future__ import annotations

from datetime import UTC, datetime, timedelta
import math

import numpy as np

from beta_spy.mechanics_v2 import MechanicsV2Estimator
from beta_spy.models import FlowFeatures


def _flow_from_force(force: float) -> FlowFeatures:
    # V2 deliberately reuses frozen V1 force. With all other components zero,
    # force = 0.60 * OFI.
    return FlowFeatures(order_flow_imbalance=force / 0.60)


def _run_four_quadrant(
    *,
    beta_pp: float,
    beta_pm: float,
    beta_mp: float,
    beta_mm: float,
    seed: int = 7,
):
    rng = np.random.default_rng(seed)
    forces = rng.uniform(-0.55, 0.55, 900)
    estimator = MechanicsV2Estimator(
        window=360,
        min_quadrant_samples=20,
        ridge=1e-9,
        min_response=1e-6,
    )

    timestamp = datetime(2026, 1, 5, 14, 30, tzinfo=UTC)
    log_price = math.log(600.0)
    velocity = 0.0
    state = estimator.step(timestamp, math.exp(log_price), _flow_from_force(float(forces[0])))

    for index in range(1, len(forces)):
        previous_force = float(forces[index - 1])
        if velocity >= 0.0 and previous_force > 0.0:
            beta = beta_pp
        elif velocity >= 0.0 and previous_force < 0.0:
            beta = beta_pm
        elif velocity < 0.0 and previous_force > 0.0:
            beta = beta_mp
        else:
            beta = beta_mm

        acceleration = beta * previous_force - 0.25 * velocity
        velocity += acceleration
        log_price += velocity / 10_000.0
        timestamp += timedelta(minutes=1)
        state = estimator.step(
            timestamp,
            math.exp(log_price),
            _flow_from_force(float(forces[index])),
        )

    return state


def test_v2_recovers_four_distinct_response_coefficients() -> None:
    state = _run_four_quadrant(
        beta_pp=3.0,
        beta_pm=0.50,
        beta_mp=0.75,
        beta_mm=2.50,
    )

    assert state.full_quadrant_ready
    assert state.beta_pp is not None
    assert state.beta_pm is not None
    assert state.beta_mp is not None
    assert state.beta_mm is not None

    assert math.isclose(state.beta_pp, 3.0, rel_tol=0.03)
    assert math.isclose(state.beta_pm, 0.50, rel_tol=0.05)
    assert math.isclose(state.beta_mp, 0.75, rel_tol=0.05)
    assert math.isclose(state.beta_mm, 2.50, rel_tol=0.03)


def test_v2_separates_launch_and_braking_inertia() -> None:
    state = _run_four_quadrant(
        beta_pp=3.0,
        beta_pm=0.40,
        beta_mp=0.50,
        beta_mm=2.0,
        seed=11,
    )

    assert state.launch_inertia_up is not None
    assert state.braking_inertia_up is not None
    assert state.braking_inertia_down is not None
    assert state.launch_inertia_down is not None

    # Same market can be easy to continue in a direction yet difficult to stop.
    assert state.braking_inertia_up > state.launch_inertia_up
    assert state.braking_inertia_down > state.launch_inertia_down


def test_lower_opposing_response_means_higher_braking_inertia() -> None:
    heavy = _run_four_quadrant(
        beta_pp=2.0,
        beta_pm=0.30,
        beta_mp=0.35,
        beta_mm=2.0,
        seed=19,
    )
    fragile = _run_four_quadrant(
        beta_pp=2.0,
        beta_pm=1.20,
        beta_mp=1.40,
        beta_mm=2.0,
        seed=19,
    )

    assert heavy.braking_inertia_up is not None
    assert fragile.braking_inertia_up is not None
    assert heavy.braking_inertia_down is not None
    assert fragile.braking_inertia_down is not None

    assert heavy.braking_inertia_up > fragile.braking_inertia_up
    assert heavy.braking_inertia_down > fragile.braking_inertia_down


def test_wrong_sign_braking_response_is_rejected_not_absolute_valued() -> None:
    state = _run_four_quadrant(
        beta_pp=2.0,
        beta_pm=-0.80,
        beta_mp=0.70,
        beta_mm=2.0,
        seed=23,
    )

    assert state.beta_pm is None
    assert state.braking_inertia_up is None
    assert not state.braking_ready


def test_v2_prefix_state_cannot_be_changed_by_future_path() -> None:
    prefix = [0.25, -0.20, 0.45, -0.35] * 40
    left = MechanicsV2Estimator(window=120, min_quadrant_samples=8)
    right = MechanicsV2Estimator(window=120, min_quadrant_samples=8)
    stamp = datetime(2026, 1, 5, 14, 30, tzinfo=UTC)
    price = 600.0
    state_left = state_right = None

    for index, force in enumerate(prefix):
        price *= math.exp((0.3 * force) / 10_000.0)
        timestamp = stamp + timedelta(minutes=index)
        state_left = left.step(timestamp, price, _flow_from_force(force))
        state_right = right.step(timestamp, price, _flow_from_force(force))

    assert state_left == state_right

    left.step(stamp + timedelta(minutes=len(prefix)), price * 1.002, _flow_from_force(0.4))
    right.step(stamp + timedelta(minutes=len(prefix)), price * 0.998, _flow_from_force(-0.4))
    assert state_left == state_right


def test_v2_irregular_gap_does_not_create_fake_response() -> None:
    estimator = MechanicsV2Estimator(window=60, min_quadrant_samples=6)
    stamp = datetime(2026, 1, 5, 14, 30, tzinfo=UTC)
    price = 600.0

    estimator.step(stamp, price, _flow_from_force(0.3))
    price *= math.exp(0.3 / 10_000.0)
    estimator.step(stamp + timedelta(minutes=1), price, _flow_from_force(-0.2))
    price *= math.exp(-0.1 / 10_000.0)
    before = estimator.step(stamp + timedelta(minutes=2), price, _flow_from_force(0.2))
    before_count = before.pp_samples + before.pm_samples + before.mp_samples + before.mm_samples

    price *= math.exp(1.0 / 10_000.0)
    gap = estimator.step(stamp + timedelta(minutes=7), price, _flow_from_force(-0.1))
    gap_count = gap.pp_samples + gap.pm_samples + gap.mp_samples + gap.mm_samples

    assert gap.acceleration_bps is None
    assert gap_count == before_count


def test_v2_new_session_resets_quadrant_history() -> None:
    estimator = MechanicsV2Estimator(window=60, min_quadrant_samples=6)
    stamp = datetime(2026, 1, 5, 14, 30, tzinfo=UTC)
    price = 600.0

    for index in range(20):
        force = 0.3 if index % 2 == 0 else -0.25
        price *= math.exp(force / 10_000.0)
        state = estimator.step(stamp + timedelta(minutes=index), price, _flow_from_force(force))

    assert state.pp_samples + state.pm_samples + state.mp_samples + state.mm_samples > 0

    next_day = estimator.step(stamp + timedelta(days=1), price, _flow_from_force(0.2))
    assert next_day.pp_samples == 0
    assert next_day.pm_samples == 0
    assert next_day.mp_samples == 0
    assert next_day.mm_samples == 0
    assert next_day.velocity_bps is None
    assert next_day.acceleration_bps is None
