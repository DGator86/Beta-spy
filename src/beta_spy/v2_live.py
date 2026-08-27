from __future__ import annotations

from dataclasses import asdict
from datetime import UTC, datetime
from typing import Any

from .live import TradierMarketStream
from .v2_hgb_direction import CausalHGBDirectionStack
from .v2_mtf import V2MTFStack
from .v2_predictive_state import CausalPredictiveStateStack
from .v2_regime_forecast import forecast_regime


class V2TradierMarketStream(TradierMarketStream):
    """Beta V2 live stream: publish causal intelligence, never an option strategy.

    HGB owns the validated directional witness. Predictive-state compression owns
    regime/analog/path-distribution evidence. The explicit regime forecast turns
    that state into duration and successor probabilities. None of these components
    may choose an option family or place a trade.
    """

    def __init__(
        self,
        *args: Any,
        v2_stack: V2MTFStack | None = None,
        direction_stack: CausalHGBDirectionStack | None = None,
        state_stack: CausalPredictiveStateStack | None = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(*args, **kwargs)
        self.v2_stack = v2_stack or V2MTFStack()
        self.direction_stack = direction_stack or CausalHGBDirectionStack()
        self.state_stack = state_stack or CausalPredictiveStateStack()

    def _publish_snapshot(self, timestamp: datetime) -> None:
        alpha_signal = self._record_alpha_signal(timestamp)
        if alpha_signal is not None:
            self.hub.update(alpha_signal={**alpha_signal, "recorded_at": timestamp.isoformat()})

        snapshot = self.engine.build_snapshot(timestamp)
        if snapshot is None:
            ledger_stats = self.ledger.stats(timestamp) if self.ledger is not None else None
            self.hub.update(timestamp=timestamp.isoformat(), status="WARMING", ledger=ledger_stats)
            return

        spy = next((item for item in snapshot.symbols if item.symbol == "SPY"), None)
        if spy is None or spy.close <= 0:
            self.hub.update(
                timestamp=timestamp.isoformat(),
                snapshot=asdict(snapshot),
                v2_opportunity=None,
                option_plan=None,
                status="WARMING",
            )
            return

        spot = float(spy.close)
        mtf = self.v2_stack.step(timestamp, snapshot.factors, spot)
        direction = self.direction_stack.step(timestamp, self.engine.states, spot)
        state = self.state_stack.step(timestamp, self.engine.states, spot)
        regime_forecast = forecast_regime(state.as_dict())

        payload = mtf.as_dict()
        payload["mtf_context"] = mtf.as_dict()
        payload["hgb_direction"] = direction.as_dict()
        payload["predictive_state"] = state.as_dict()
        payload["regime_forecast"] = regime_forecast.as_dict()
        payload["regime_definable"] = regime_forecast.definable
        payload["regime_confidence"] = regime_forecast.confidence
        payload["regime_persistence_15"] = regime_forecast.persistence_15
        payload["regime_persistence_30"] = regime_forecast.persistence_30
        payload["expected_regime_duration_minutes"] = regime_forecast.expected_duration_minutes
        payload["successor_regimes"] = regime_forecast.successor_probabilities
        payload["most_likely_successor_regime"] = regime_forecast.most_likely_successor
        payload["successor_regime_confidence"] = regime_forecast.successor_confidence
        payload["strategy_authority"] = False
        payload["role"] = "regime_duration_transition_and_distribution_intelligence"

        payload["eligible"] = bool(direction.eligible)
        payload["state"] = (
            "DIRECTIONAL_UP"
            if direction.eligible and direction.direction == "BULLISH"
            else "DIRECTIONAL_DOWN"
            if direction.eligible and direction.direction == "BEARISH"
            else state.regime
            if state.ready
            else "NO_TRADE"
        )
        payload["probability_up"] = direction.probability_up
        payload["expected_return_bps"] = direction.expected_return_bps
        payload["expected_abs_bps"] = max(
            abs(direction.expected_return_bps),
            float(payload.get("expected_abs_bps") or 0.0),
            state.direct_pred_abs15 if state.ready else 0.0,
        )
        payload["validated_direction_edge"] = 2.0 * direction.probability_up - 1.0
        payload["trust"] = max(
            float(payload.get("trust") or 0.0),
            min(1.0, max(0.0, direction.strength)),
            regime_forecast.confidence if regime_forecast.definable else 0.0,
        )
        payload["agreement"] = 1.0 if direction.eligible else float(payload.get("agreement") or 0.0)
        payload["model_version"] = direction.model_version
        payload["reasons"] = [
            "hgb_daily_refit_direction_signal"
            if direction.eligible
            else "regime_defined_waiting_for_monetizable_edge"
            if regime_forecast.definable
            else "state_distribution_ready_regime_uncertain"
            if state.ready
            else "v2_models_warming"
        ]

        if self.engine.store is not None:
            self.engine.store.save_alpha_signal(
                timestamp,
                {
                    "record_type": "beta_v2_opportunity",
                    "recorded_at": datetime.now(UTC).isoformat(),
                    "opportunity": payload,
                },
            )

        ledger_stats = self.ledger.stats(timestamp) if self.ledger is not None else None
        self.hub.update(
            timestamp=timestamp.isoformat(),
            snapshot=asdict(snapshot),
            v2_opportunity=payload,
            opportunity=payload,
            option_plan=None,
            ledger=ledger_stats,
            status="LIVE",
        )
