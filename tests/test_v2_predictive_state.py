from datetime import UTC, datetime, timedelta

import numpy as np

from beta_spy.v2_predictive_state import (
    CausalPredictiveStateStack,
    PredictiveStateDistribution,
    _Pending,
)


def test_state_training_sample_is_not_released_before_30_minutes():
    stack = CausalPredictiveStateStack()
    start = datetime(2026, 8, 27, 14, 0, tzinfo=UTC)
    stack.pending.append(
        _Pending(
            timestamp=start,
            session_date=start.date(),
            vector=np.ones(63, dtype=float),
            start_price=700.0,
            forecast_mean15=2.0,
            forecast_sigma15=8.0,
        )
    )

    stack._mature(start + timedelta(minutes=15), 700.7)
    assert len(stack.y15) == 0
    assert len(stack.validation_z15) == 1

    stack._mature(start + timedelta(minutes=30), 701.4)
    assert len(stack.y5) == 1
    assert len(stack.y15) == 1
    assert len(stack.y30) == 1
    assert stack.sample_dates == [start.date()]


def test_predictive_regimes_are_state_labels_not_strategy_commands():
    pred = np.asarray(
        [
            [2.0, 5.0, 8.0, 7.0],
            [-2.0, -5.0, -8.0, 7.0],
            [0.2, 0.1, -0.1, 2.0],
            [1.0, -0.5, 1.0, 12.0],
        ]
    )
    regimes = CausalPredictiveStateStack._regimes(
        pred,
        np.asarray([0.001, 0.001, 0.0002, 0.002]),
        abs_lo=3.0,
        abs_hi=10.0,
        rv_median=0.001,
    )
    assert regimes.tolist() == [
        "DIRECTIONAL_UP",
        "DIRECTIONAL_DOWN",
        "QUIET",
        "EXPANSION",
    ]


def test_state_distribution_never_has_strategy_authority():
    result = PredictiveStateDistribution(
        timestamp=datetime(2026, 8, 27, 14, 0, tzinfo=UTC),
        ready=True,
        regime="QUIET",
        analog_count=60,
        effective_analogs=40.0,
        mean_proximity=0.25,
        direct_pred_5=0.0,
        direct_pred_15=0.0,
        direct_pred_30=0.0,
        direct_pred_abs15=5.0,
        conformal_scale=1.1,
        mean_5=0.0,
        mean_15=0.0,
        mean_30=0.0,
        sigma_5=2.0,
        sigma_15=5.0,
        sigma_30=8.0,
        p_up_5=0.5,
        p_up_15=0.5,
        p_up_30=0.5,
        p_big_5=0.2,
        p_big_15=0.2,
        p_big_30=0.2,
        quantiles_5={},
        quantiles_15={},
        quantiles_30={},
        p_persistent_30=0.4,
        p_reversal_15=0.3,
        p_reversal_30=0.3,
        p_acceleration=0.2,
        analog_y15_bps=(),
        analog_weights=(),
        training_sessions=10,
        training_samples=500,
    )
    assert result.strategy_authority is False
    assert result.as_dict()["strategy_authority"] is False
