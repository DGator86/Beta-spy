from __future__ import annotations

from dataclasses import asdict
from datetime import UTC, datetime
from typing import Any

from .live import TradierMarketStream
from .v2_mtf import V2MTFStack


class V2TradierMarketStream(TradierMarketStream):
    """Beta V2 live stream: publish causal opportunity state, never an option strategy.

    Legacy Beta can still run on the same tape for comparison, but V2's authoritative
    output is `v2_opportunity`. It deliberately does not open Beta-led option positions.
    """

    def __init__(self, *args: Any, v2_stack: V2MTFStack | None = None, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.v2_stack = v2_stack or V2MTFStack()

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

        opportunity = self.v2_stack.step(timestamp, snapshot.factors, float(spy.close))
        payload = opportunity.as_dict()
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
