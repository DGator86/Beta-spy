from __future__ import annotations

import argparse
import os
import threading
from datetime import UTC, datetime

import uvicorn

from .app import create_app
from .engine import Tape500Engine
from .live import StateHub
from .replay import HistoricalReplay
from .storage import Tape500Store
from .universe import fetch_current_spy_universe, load_universe_csv
from .v2_hgb_direction import CausalHGBDirectionStack
from .v2_live import V2TradierMarketStream
from .v2_mtf import V2MTFStack


def _holdings(path: str | None):
    return load_universe_csv(path) if path else fetch_current_spy_universe()


def parser() -> argparse.ArgumentParser:
    root = argparse.ArgumentParser(
        prog="beta-spy-v2",
        description="Beta-SPY V2 causal HGB + MTF intelligence service",
    )
    root.add_argument("--db", default="data/beta-spy.sqlite")
    root.add_argument("--universe")
    root.add_argument("--host", default="127.0.0.1")
    root.add_argument("--port", type=int, default=8790)
    root.add_argument("--warm-sessions", type=int, default=10)
    root.add_argument(
        "--alpha-state-url",
        default=os.environ.get(
            "ALPHA_SPY_STATE_URL", "http://127.0.0.1:8787/api/v1/state"
        ),
    )
    return root


def main() -> None:
    args = parser().parse_args()
    token = (
        os.environ.get("TRADIER_MARKET_ACCESS_TOKEN")
        or os.environ.get("TRADIER_ACCESS_TOKEN")
        or ""
    )
    if not token:
        raise SystemExit("Set TRADIER_MARKET_ACCESS_TOKEN (or TRADIER_ACCESS_TOKEN)")

    holdings = _holdings(args.universe)
    store = Tape500Store(args.db)
    hub = StateHub()
    engine = Tape500Engine(holdings, store=store)
    v2_stack = V2MTFStack()
    direction_stack = CausalHGBDirectionStack()

    def warm_start() -> None:
        if args.warm_sessions <= 0:
            return
        with store.lock:
            dates = [
                str(row[0])
                for row in store.connection.execute(
                    "SELECT DISTINCT substr(timestamp,1,10) FROM minute_bars ORDER BY 1"
                ).fetchall()
            ]
        recent = dates[-args.warm_sessions :]
        if not recent:
            return
        start = datetime.fromisoformat(recent[0] + "T00:00:00+00:00")
        count = 0
        for snapshot in HistoricalReplay(store, engine).run(start=start):
            spy = next(
                (item for item in snapshot.symbols if item.symbol == "SPY" and item.close > 0),
                None,
            )
            if spy is None:
                continue
            v2_stack.step(snapshot.timestamp, snapshot.factors, float(spy.close))
            direction_stack.step(snapshot.timestamp, engine.states, float(spy.close))
            count += 1
        hub.update(
            v2_warm_start={
                "sessions": len(recent),
                "snapshots": count,
                "started_at": recent[0],
                "completed_at": datetime.now(UTC).isoformat(),
                "mtf_config_sha256": v2_stack.config.fingerprint(),
                "hgb_training_sessions": len(set(direction_stack.sample_dates)),
                "hgb_training_samples": len(direction_stack.y_bps),
            }
        )

    stream = V2TradierMarketStream(
        token,
        engine,
        hub,
        ledger=None,
        warmup=warm_start,
        alpha_state_url=args.alpha_state_url,
        v2_stack=v2_stack,
        direction_stack=direction_stack,
    )
    thread = threading.Thread(target=stream.run_forever, daemon=True, name="beta-v2-tape")
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
