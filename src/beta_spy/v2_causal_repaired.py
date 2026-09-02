from __future__ import annotations

import math
from collections import deque
from dataclasses import replace
from datetime import datetime, timedelta

import numpy as np

from .v2_hgb_direction import CausalHGBDirectionStack as _BaseHGBDirectionStack
from .v2_mtf import V2Config, V2MTFStack as _BaseMTFStack, _HorizonHead
from .v2_predictive_state import CausalPredictiveStateStack as _BasePredictiveStateStack

HGB_MODEL_VERSION = "beta-spy-v2-hgb-trailing-2-exact-target"
STATE_MODEL_VERSION = "beta-spy-v2-predictive-state-2-exact-target"
MTF_MODEL_VERSION = "beta-spy-v2-mtf-validated-2-exact-target"


def _minute(value: datetime) -> datetime:
    return value.replace(second=0, microsecond=0)


def _same_minute(left: datetime, right: datetime) -> bool:
    return _minute(left) == _minute(right)


class CausalHGBDirectionStack(_BaseHGBDirectionStack):
    """HGB witness that never stretches a missed 15-minute target.

    A delayed process tick, reconnect, or replay gap must not turn a 15-minute
    training target into a 17-, 20-, or 30-minute return. Mature labels are used
    only on the exact scheduled target minute; missed labels are discarded.
    """

    def _mature(self, timestamp: datetime, spy_price: float) -> None:
        while self.pending and self.pending[0].target_time <= timestamp:
            item = self.pending.popleft()
            exact_target = _same_minute(item.target_time, timestamp)
            if (
                not exact_target
                or item.session_date != timestamp.date()
                or item.start_price <= 0
                or spy_price <= 0
            ):
                continue
            realized_bps = (spy_price / item.start_price - 1.0) * 10_000.0
            if not math.isfinite(realized_bps):
                continue
            self.core_x.append(item.core)
            self.breadth_x.append(item.breadth)
            self.y_bps.append(float(np.clip(realized_bps, -40.0, 40.0)))
            self.sample_dates.append(item.session_date)

    def step(self, timestamp: datetime, states, spy_price: float):
        result = super().step(timestamp, states, spy_price)
        return replace(result, model_version=HGB_MODEL_VERSION)


class CausalPredictiveStateStack(_BasePredictiveStateStack):
    """Predictive-state stack with exact 5/15/30-minute outcome labels."""

    def _mature(self, timestamp: datetime, spy_price: float) -> None:
        kept = deque()
        current = _minute(timestamp)
        for item in self.pending:
            if item.session_date != timestamp.date() or item.start_price <= 0 or spy_price <= 0:
                continue

            target5 = _minute(item.timestamp + timedelta(minutes=5))
            target15 = _minute(item.timestamp + timedelta(minutes=15))
            target30 = _minute(item.timestamp + timedelta(minutes=30))
            if current > target30:
                # A missed horizon invalidates the complete 5/15/30 training row.
                continue

            realized = (spy_price / item.start_price - 1.0) * 10_000.0
            if not math.isfinite(realized):
                continue
            if current == target5 and item.y5 is None:
                item.y5 = realized
            if current == target15 and item.y15 is None:
                item.y15 = realized
                if item.forecast_mean15 is not None and item.forecast_sigma15 not in (None, 0.0):
                    z = abs(realized - item.forecast_mean15) / max(float(item.forecast_sigma15), 0.50)
                    self.validation_dates.append(item.session_date)
                    self.validation_z15.append(float(z))
            if current == target30:
                if item.y5 is not None and item.y15 is not None:
                    self.x.append(item.vector)
                    self.y5.append(float(item.y5))
                    self.y15.append(float(item.y15))
                    self.y30.append(float(realized))
                    self.sample_dates.append(item.session_date)
                continue
            kept.append(item)
        self.pending = kept

    def step(self, timestamp: datetime, states, spy_price: float):
        result = super().step(timestamp, states, spy_price)
        return replace(result, model_version=STATE_MODEL_VERSION)


class _ExactTargetHorizonHead(_HorizonHead):
    """One MTF head that learns only at its exact configured horizon."""

    def mature(self, timestamp: datetime, spy_price: float) -> None:
        while self.pending and self.pending[0].target_time <= timestamp:
            item = self.pending.popleft()
            if (
                not _same_minute(item.target_time, timestamp)
                or item.target_time.date() != timestamp.date()
                or item.start_price <= 0
                or spy_price <= 0
            ):
                continue
            realized_bps = (spy_price / item.start_price - 1.0) * 10_000.0
            if not math.isfinite(realized_bps):
                continue
            x = item.vector.reshape(1, -1)
            self.scaler.partial_fit(x)
            z = self.scaler.transform(x)

            is_big = int(abs(realized_bps) >= self.threshold_bps)
            y_big = np.asarray([is_big], dtype=int)
            if not self._mag_initialized:
                self.magnitude_model.partial_fit(z, y_big, classes=np.asarray([0, 1], dtype=int))
                self._mag_initialized = True
            else:
                self.magnitude_model.partial_fit(z, y_big)

            if is_big:
                y_dir = np.asarray([1 if realized_bps > 0 else 0], dtype=int)
                if not self._dir_initialized:
                    self.direction_model.partial_fit(z, y_dir, classes=np.asarray([0, 1], dtype=int))
                    self._dir_initialized = True
                else:
                    self.direction_model.partial_fit(z, y_dir)
                self.big_sample_count += 1

            self.abs_model.partial_fit(z, np.asarray([abs(realized_bps)], dtype=float))
            self._abs_initialized = True
            self.sample_count += 1
            self.validation.update(
                probability_big=item.probability_big,
                probability_up=item.probability_up,
                expected_abs_bps=item.expected_abs_bps,
                realized_bps=realized_bps,
                threshold_bps=self.threshold_bps,
                decay=self.config.validation_decay,
                alignment_decay=self.config.alignment_decay,
            )


class V2MTFStack(_BaseMTFStack):
    """MTF stack whose maturity compressor fails closed across data gaps."""

    def __init__(self, config: V2Config | None = None):
        self.config = config or V2Config()
        self.heads = {
            horizon: _ExactTargetHorizonHead(horizon_minutes=horizon, config=self.config)
            for horizon in self.config.horizons
        }

    def step(self, timestamp: datetime, factors, spy_price: float):
        result = super().step(timestamp, factors, spy_price)
        return replace(result, model_version=MTF_MODEL_VERSION)
