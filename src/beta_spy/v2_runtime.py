from __future__ import annotations

import argparse
import os
import threading
from dataclasses import asdict
from datetime import datetime

import uvicorn

from .app import create_app
from .live import StateHub, TradierMarketStream
from .replay import HistoricalReplay
from .storage import Tape500Store
from .universe import fetch_current_spy_universe, load_universe_csv
from .v2_engine import V2Tape500Engine


class V2TradierMarketStream(TradierMarketStream):
    """Beta V2 publishes market state and never opens a new option position.

    Alpha V2 is the sole payoff/strategy authority. Any legacy Beta paper
    positions already present in the ledger continue to be marked, but this
    runtime never calls ``_option_plan`` or ``ledger.open_position``.
    """

    def _publish_snapshot(self, timestamp: datetime) -> None:
        alpha_signal = self._record_alpha_signal(timestamp)
        if alpha_signal is not None:
            self.hub.update(alpha_signal={**alpha_signal, "recorded_at": timestamp.isoformat()})
        snapshot = self.engine.build_snapshot(timestamp)
        if snapshot is None:
            ledger_stats = self.ledger.stats(timestamp) if self.ledger is not None else None
            self.hub.update(timestamp=timestamp.isoformat(), status="WARMING", ledger=ledger_stats)
            return
        spy_price = None
        spy = next((item for item in snapshot.symbols if item.symbol == "SPY"), None)
        if spy is not None and spy.close > 0:
            spy_price = float(spy.close)
        ledger_stats = None
        if self.ledger is not None:
            self._mark_ledger(timestamp, spy_price=spy_price)
            ledger_stats = self.ledger.stats(timestamp)
        v2_state = getattr(snapshot, "v2_state", None)
        self.hub.update(
            timestamp=timestamp.isoformat(),
            snapshot=asdict(snapshot),
            v2_state=v2_state,
            opportunity=v2_state,
            option_plan=None,
            ledger=ledger_stats,
            status="LIVE",
        )


def _load_holdings(path: str | None):
    return load_universe_csv(path) if path else fetch_current_spy_universe()


def main() -> None:
    parser = argparse.ArgumentParser(prog="beta-spy-v2")
    parser.add_argument("--db", default="data/beta-spy.sqlite")
    parser.add_argument("--universe")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8790)
    parser.add_argument("--warm-sessions", type=int, default=10)
    parser.add_argument(
        "--alpha-state-url",
        default=os.environ.get("ALPHA_SPY_STATE_URL", "http://127.0.0.1:8787/api/v1/state"),
    )
    args = parser.parse_args()

    token = os.environ.get("TRADIER_MARKET_ACCESS_TOKEN") or os.environ.get("TRADIER_ACCESS_TOKEN") or ""
    if not token:
        raise SystemExit("Set TRADIER_MARKET_ACCESS_TOKEN (or TRADIER_ACCESS_TOKEN)")
    holdings = _load_holdings(args.universe)
    store = Tape500Store(args.db)
    hub = StateHub()
    engine = V2Tape500Engine(holdings, store=store)

    def _warm_start() -> None:
        with store.lock:
            dates = [
                str(row[0])
                for row in store.connection.execute(
                    "SELECT DISTINCT substr(timestamp, 1, 10) FROM minute_bars ORDER BY 1"
                ).fetchall()
            ]
        recent = dates[-max(0, args.warm_sessions) :]
        if not recent:
            return
        start = datetime.fromisoformat(recent[0] + "T00:00:00+00:00")
        count = 0
        for _ in HistoricalReplay(store, engine).run(start=start):
            count += 1
        print(
            f"V2 warm-start: replayed {count} snapshots from {recent[0]}; "
            "all validators matured causally",
            flush=True,
        )

    stream = V2TradierMarketStream(
        token,
        engine,
        hub,
        ledger=None,
        warmup=_warm_start if args.warm_sessions > 0 else None,
        alpha_state_url=args.alpha_state_url,
    )
    thread = threading.Thread(target=stream.run_forever, daemon=True, name="tradier-v2-tape")
    thread.start()
    try:
        uvicorn.run(create_app(hub), host=args.host, port=args.port, log_level="info")
    finally:
        stream.stop()
        thread.join(timeout=5)
        stream.close()
        store.close()


if __name__ == "__main__":
    main()
