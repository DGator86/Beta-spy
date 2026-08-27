from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from datetime import datetime
from typing import Any

from .engine import Tape500Engine
from .models import EngineSnapshot
from .v2_validation import V2MarketState, V2ValidationStack


@dataclass(frozen=True)
class V2EngineSnapshot(EngineSnapshot):
    """Legacy snapshot plus the strategy-agnostic V2 market state."""

    v2_state: dict[str, Any] | None = None


class V2Tape500Engine(Tape500Engine):
    """Tape500 engine with a maturity-delayed supervisory forecast stack.

    The legacy forecast/decision path remains in the snapshot for side-by-side
    diagnosis, but V2 consumers must use ``v2_state``. Beta V2 does not own
    option payoff geometry and therefore never treats ``decision.structure`` as
    authoritative.
    """

    def __init__(self, *args: Any, v2_stack: V2ValidationStack | None = None, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.v2 = v2_stack or V2ValidationStack()
        self.latest_v2_state: V2MarketState | None = None

    def build_snapshot(self, timestamp: datetime) -> V2EngineSnapshot | None:
        base = super().build_snapshot(timestamp)
        if base is None:
            return None
        spy = next((item for item in base.symbols if item.symbol == "SPY"), None)
        if spy is None or spy.close <= 0:
            return V2EngineSnapshot(
                timestamp=base.timestamp,
                factors=base.factors,
                forecasts=base.forecasts,
                decision=base.decision,
                symbols=base.symbols,
                v2_state=None,
            )
        state = self.v2.step(timestamp, base.factors, float(spy.close))
        self.latest_v2_state = state
        if self.store is not None:
            with self.store.lock:
                self.store.connection.execute(
                    """
                    CREATE TABLE IF NOT EXISTS v2_market_state (
                        timestamp TEXT PRIMARY KEY,
                        payload TEXT NOT NULL
                    )
                    """
                )
                self.store.connection.execute(
                    "INSERT OR REPLACE INTO v2_market_state(timestamp,payload) VALUES(?,?)",
                    (
                        timestamp.isoformat().replace("+00:00", "Z"),
                        json.dumps(asdict(state), default=str, separators=(",", ":")),
                    ),
                )
                self.store.connection.commit()
        return V2EngineSnapshot(
            timestamp=base.timestamp,
            factors=base.factors,
            forecasts=base.forecasts,
            decision=base.decision,
            symbols=base.symbols,
            v2_state=state.as_dict(),
        )
