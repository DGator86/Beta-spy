from datetime import UTC, datetime, timedelta

import numpy as np

from beta_spy.v2_validation import (
    V2HorizonModel,
    V2MarketState,
    _ValidationRecord,
)


def test_v2_does_not_mature_before_target() -> None:
    model = V2HorizonModel(horizon_minutes=15, min_samples=1)
    now = datetime(2026, 8, 27, 14, 0, tzinfo=UTC)
    vector = np.asarray([0.1, -0.2, 0.3], dtype=float)
    raw = model.raw_predict(vector)
    model.queue(now, vector, 100.0, raw)

    model.mature(now + timedelta(minutes=14), 101.0)
    assert model.sample_count == 0

    model.mature(now + timedelta(minutes=15), 101.0)
    assert model.sample_count == 1


def test_direction_training_ignores_small_moves() -> None:
    model = V2HorizonModel(horizon_minutes=15, min_samples=1)
    now = datetime(2026, 8, 27, 14, 0, tzinfo=UTC)
    vector = np.asarray([0.1, 0.2], dtype=float)

    model.queue(now, vector, 100.0, model.raw_predict(vector))
    model.mature(now + timedelta(minutes=15), 100.02)

    assert model.sample_count == 1
    assert model.big_move_count == 0
    assert model._magnitude_initialized is True
    assert model._direction_initialized is False


def test_direction_alignment_can_be_negative() -> None:
    model = V2HorizonModel(horizon_minutes=15, min_samples=1)
    for _ in range(80):
        model.validations.append(
            _ValidationRecord(
                raw_big_probability=0.8,
                raw_up_probability=0.9,
                realized_big=True,
                realized_up=False,
                realized_abs_bps=12.0,
            )
        )

    _magnitude_trust, direction_trust, signed_alignment = model._validation_metrics()

    assert direction_trust > 0
    assert signed_alignment < 0


def test_magnitude_trust_survives_persistent_expansion_regime() -> None:
    model = V2HorizonModel(horizon_minutes=15, min_samples=1)
    for _ in range(80):
        model.validations.append(
            _ValidationRecord(
                raw_big_probability=0.95,
                raw_up_probability=0.5,
                realized_big=True,
                realized_up=True,
                realized_abs_bps=14.0,
            )
        )

    magnitude_trust, _direction_trust, _signed_alignment = model._validation_metrics()

    assert 0.0 < model._magnitude_base_rate() < 1.0
    assert magnitude_trust > 0


def test_beta_v2_has_no_strategy_authority() -> None:
    now = datetime(2026, 8, 27, 14, 0, tzinfo=UTC)
    state = V2MarketState(
        timestamp=now,
        regime="NORMAL",
        probability_big_move=0.5,
        probability_up_given_big_move=0.5,
        expected_abs_move_bps=5.0,
        validated_direction_edge=0.0,
        magnitude_trust=0.3,
        direction_trust=0.2,
        overall_trust=0.24,
        horizons=(),
    )
    assert state.strategy_authority is False
    assert "structure" not in state.as_dict()
