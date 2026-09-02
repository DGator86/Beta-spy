from __future__ import annotations

from datetime import UTC, datetime, timedelta

import numpy as np
import pytest

from beta_spy.v2_causal_repaired import (
    CausalHGBDirectionStack,
    CausalPredictiveStateStack,
    V2Config,
    _ExactTargetHorizonHead,
)
from beta_spy.v2_hgb_direction import _Pending as HGBPending
from beta_spy.v2_predictive_state import _Pending as StatePending


def test_hgb_delayed_maturity_is_discarded_not_stretched():
    stack = CausalHGBDirectionStack()
    start = datetime(2026, 8, 10, 14, 0, tzinfo=UTC)
    stack.pending.append(
        HGBPending(
            target_time=start + timedelta(minutes=15),
            session_date=start.date(),
            core=np.ones(57),
            breadth=np.ones(43),
            start_price=100.0,
        )
    )
    stack._mature(start + timedelta(minutes=20), 102.0)
    assert stack.y_bps == []
    assert not stack.pending


def test_hgb_exact_maturity_is_accepted():
    stack = CausalHGBDirectionStack()
    start = datetime(2026, 8, 10, 14, 0, tzinfo=UTC)
    stack.pending.append(
        HGBPending(
            target_time=start + timedelta(minutes=15),
            session_date=start.date(),
            core=np.ones(57),
            breadth=np.ones(43),
            start_price=100.0,
        )
    )
    stack._mature(start + timedelta(minutes=15), 101.0)
    assert stack.y_bps == [40.0]  # clipped by the existing HGB training contract


def test_predictive_state_requires_all_three_exact_horizons():
    stack = CausalPredictiveStateStack()
    start = datetime(2026, 8, 10, 14, 0, tzinfo=UTC)
    item = StatePending(
        timestamp=start,
        session_date=start.date(),
        vector=np.ones(64),
        start_price=100.0,
    )
    stack.pending.append(item)
    stack._mature(start + timedelta(minutes=5), 100.1)
    stack._mature(start + timedelta(minutes=15), 100.2)
    stack._mature(start + timedelta(minutes=30), 100.3)
    assert len(stack.x) == 1
    assert stack.y5[0] == pytest.approx(10.0)
    assert stack.y15[0] == pytest.approx(20.0)
    assert stack.y30[0] == pytest.approx(30.0)


def test_predictive_state_drops_row_if_a_horizon_was_missed():
    stack = CausalPredictiveStateStack()
    start = datetime(2026, 8, 10, 14, 0, tzinfo=UTC)
    stack.pending.append(
        StatePending(
            timestamp=start,
            session_date=start.date(),
            vector=np.ones(64),
            start_price=100.0,
        )
    )
    # The 5-minute target is missed. Later prices must not backfill it.
    stack._mature(start + timedelta(minutes=6), 100.1)
    stack._mature(start + timedelta(minutes=15), 100.2)
    stack._mature(start + timedelta(minutes=30), 100.3)
    assert stack.x == []
    assert stack.y5 == []
    assert stack.y15 == []
    assert stack.y30 == []


def test_mtf_delayed_horizon_is_discarded():
    cfg = V2Config(horizons=(5,))
    head = _ExactTargetHorizonHead(horizon_minutes=5, config=cfg)
    start = datetime(2026, 8, 10, 14, 0, tzinfo=UTC)
    head.queue(
        start,
        np.ones(10),
        100.0,
        {"probability_big": 0.6, "probability_up": 0.6, "expected_abs_bps": 8.0},
    )
    head.mature(start + timedelta(minutes=6), 101.0)
    assert head.sample_count == 0
    assert not head.pending


def test_mtf_exact_horizon_trains():
    cfg = V2Config(horizons=(5,))
    head = _ExactTargetHorizonHead(horizon_minutes=5, config=cfg)
    start = datetime(2026, 8, 10, 14, 0, tzinfo=UTC)
    head.queue(
        start,
        np.ones(10),
        100.0,
        {"probability_big": 0.6, "probability_up": 0.6, "expected_abs_bps": 8.0},
    )
    head.mature(start + timedelta(minutes=5), 100.1)
    assert head.sample_count == 1
