from __future__ import annotations

from datetime import UTC, datetime, timedelta

import numpy as np

import beta_spy.v2_hgb_direction as hgb


def test_cold_stack_queues_before_model_is_ready(monkeypatch):
    stack = hgb.CausalHGBDirectionStack()
    core = np.arange(57, dtype=float)
    breadth = np.arange(43, dtype=float)
    monkeypatch.setattr(hgb, "trailing_feature_vectors", lambda states, timestamp: (core, breadth))

    start = datetime(2026, 8, 3, 14, 0, tzinfo=UTC)
    signal = stack.step(start, {}, 700.0)
    assert signal.ready is False
    assert signal.eligible is False
    assert len(stack.pending) == 1

    # At +15 minutes the first cold-start sample is allowed to mature and train.
    stack.step(start + timedelta(minutes=15), {}, 700.7)
    assert len(stack.y_bps) == 1
    assert stack.sample_dates == [start.date()]


def test_model_does_not_refit_inside_current_session(monkeypatch):
    stack = hgb.CausalHGBDirectionStack()
    core = np.ones(57, dtype=float)
    breadth = np.ones(43, dtype=float)
    monkeypatch.setattr(hgb, "trailing_feature_vectors", lambda states, timestamp: (core, breadth))

    # Seed five completed prior sessions with enough matured observations.
    for day_offset in range(5):
        day = datetime(2026, 8, 3 + day_offset, 14, 0, tzinfo=UTC).date()
        for i in range(45):
            stack.core_x.append(core + i * 0.001)
            stack.breadth_x.append(breadth + i * 0.001)
            stack.y_bps.append(float((i % 7) - 3))
            stack.sample_dates.append(day)

    session = datetime(2026, 8, 10, 14, 0, tzinfo=UTC)
    stack.step(session, {}, 700.0)
    assert stack.core_model is not None
    model_id = id(stack.core_model)

    # Same-session matured observations may enter the training store but cannot
    # trigger a refit until the date changes.
    stack.step(session + timedelta(minutes=15), {}, 700.5)
    assert id(stack.core_model) == model_id


def test_signal_requires_both_hgb_views_to_agree(monkeypatch):
    stack = hgb.CausalHGBDirectionStack()
    core = np.ones(57, dtype=float)
    breadth = np.ones(43, dtype=float)
    monkeypatch.setattr(hgb, "trailing_feature_vectors", lambda states, timestamp: (core, breadth))

    class Scaler:
        def transform(self, x):
            return x

    class Model:
        def __init__(self, value):
            self.value = value

        def predict(self, x):
            return np.asarray([self.value], dtype=float)

    stack.current_session = datetime(2026, 8, 10, tzinfo=UTC).date()
    stack.core_scaler = Scaler()
    stack.breadth_scaler = Scaler()
    stack.core_model = Model(6.0)
    stack.breadth_model = Model(-6.0)
    stack.core_sigma = 6.0
    stack.breadth_sigma = 6.0
    signal = stack.step(datetime(2026, 8, 10, 14, 0, tzinfo=UTC), {}, 700.0)
    assert signal.ready is True
    assert signal.eligible is False
