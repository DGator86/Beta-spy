from __future__ import annotations

from dataclasses import asdict
from datetime import UTC, datetime
from typing import Any

from .live import TradierMarketStream
from .v2_hgb_direction import CausalHGBDirectionStack
from .v2_mtf import V2MTFStack


class V2TradierMarketStream(TradierMarketStream):
    """Beta V2 live stream: publish causal intelligence, never an option strategy.

    The original V2 MTF stack remains as magnitude/regime context. The authoritative
    directional trigger is the daily-refit HGB ensemble that survived the blocked
    walk-forward and leakage-ablation tests. It trains only on completed prior
    sessions and runs on a five-minute grid.
    """

    def __init__(
        self,
        *args: Any,
        v2_stack: V2MTFStack | None = None,
        direction_stack: CausalHGBDirectionStack | None = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(*args, **kwargs)
        self.v2_stack = v2_stack or V2MTFStack()
        self.direction_stack = direction_stack or CausalHGBDirectionStack()

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

        mtf = self.v2_stack.step(timestamp, snapshot.factors, float(spy.close))
        direction = self.direction_stack.step(timestamp, self.engine.states, float(spy.close))
        payload = mtf.as_dict()
        payload["mtf_context"] = mtf.as_dict()
        payload["hgb_direction"] = direction.as_dict()
        payload["strategy_authority"] = False
        payload["role"] = "validated_market_state_only"

        # HGB is the directional trigger. MTF remains context and can inform Alpha,
        # but it does not veto a direction signal that has survived the blocked
        # daily-refit validation protocol.
        payload["eligible"] = bool(direction.eligible)
        payload["state"] = (
            "DIRECTIONAL_UP"
            if direction.eligible and direction.direction == "BULLISH"
            else "DIRECTIONAL_DOWN"
            if direction.eligible and direction.direction == "BEARISH"
            else "NO_TRADE"
        )
        payload["probability_up"] = direction.probability_up
        payload["expected_return_bps"] = direction.expected_return_bps
        payload["expected_abs_bps"] = max(
            abs(direction.expected_return_bps), float(payload.get("expected_abs_bps") or 0.0)
        )
        payload["validated_direction_edge"] = 2.0 * direction.probability_up - 1.0
        payload["trust"] = max(
            float(payload.get("trust") or 0.0), min(1.0, max(0.0, direction.strength))
        )
        payload["agreement"] = 1.0 if direction.eligible else float(payload.get("agreement") or 0.0)
        payload["model_version"] = direction.model_version
        payload["reasons"] = [
            "hgb_daily_refit_direction_signal"
            if direction.eligible
            else "hgb_direction_warming_or_below_strength"
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
