from __future__ import annotations

import argparse
import os
import threading
from datetime import UTC, datetime, timedelta
from pathlib import Path

import uvicorn

from .app import create_app
from .demo import DemoFeed, demo_holdings
from .engine import Tape500Engine
from .historical import AlpacaHistoricalClient, TradierHistoricalClient
from .ledger import PaperLedger
from .live import StateHub, TradierMarketStream
from .backtest import run_backtest, write_report
from .replay import HistoricalReplay
from .storage import Tape500Store
from .universe import fetch_current_spy_universe, load_universe_csv, save_universe_csv


def _time(value: str) -> datetime:
    parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    return parsed if parsed.tzinfo else parsed.replace(tzinfo=UTC)


def _load_holdings(path: str | None):
    return load_universe_csv(path) if path else fetch_current_spy_universe()


def main() -> None:
    parser = argparse.ArgumentParser(prog="beta-spy")
    sub = parser.add_subparsers(dest="command", required=True)

    run = sub.add_parser("run", help="Run live Tradier tape and rich dashboard")
    run.add_argument("--db", default="data/beta-spy.sqlite")
    run.add_argument("--universe")
    run.add_argument("--host", default="127.0.0.1")
    run.add_argument("--port", type=int, default=8790)
    run.add_argument(
        "--risk",
        type=float,
        default=100.0,
        help="Fallback fixed per-trade risk in dollars, used only when --bankroll is 0",
    )
    run.add_argument(
        "--risk-fraction",
        type=float,
        default=0.15,
        help="Per-trade risk budget as a fraction of current equity (compounds uncapped)",
    )
    run.add_argument(
        "--daily-loss-limit",
        type=float,
        default=None,
        help="Absolute daily loss breaker in dollars (default: --daily-loss-fraction of equity)",
    )
    run.add_argument(
        "--daily-loss-fraction",
        type=float,
        default=0.25,
        help="Daily loss breaker as a fraction of the day's starting equity",
    )
    run.add_argument(
        "--warm-sessions",
        type=int,
        default=3,
        help="Replay this many stored sessions at startup so models open warm (0 disables)",
    )
    run.add_argument(
        "--bankroll",
        type=float,
        default=10_000.0,
        help="Paper starting equity; per-trade risk is --risk-fraction of it and compounds (0 disables)",
    )
    run.add_argument(
        "--alpha-state-url",
        default=os.environ.get("ALPHA_SPY_STATE_URL", "http://127.0.0.1:8787/api/v1/state"),
        help="Alpha-spy state endpoint to record side-by-side signals from (empty disables)",
    )

    demo = sub.add_parser("demo", help="Run the dashboard against a deterministic synthetic tape")
    demo.add_argument("--db", default="data/beta-spy-demo.sqlite")
    demo.add_argument("--host", default="127.0.0.1")
    demo.add_argument("--port", type=int, default=8790)

    universe = sub.add_parser("refresh-universe", help="Fetch current SPY weights and sector metadata")
    universe.add_argument("--output", default="config/universe.csv")

    for name in ("backfill-bars", "backfill-flow"):
        cmd = sub.add_parser(name)
        cmd.add_argument("--db", default="data/beta-spy.sqlite")
        cmd.add_argument("--universe", default="config/universe.csv")
        cmd.add_argument("--start", required=True)
        cmd.add_argument("--end", required=True)
        cmd.add_argument("--feed", default="iex")

    replay = sub.add_parser("replay")
    replay.add_argument("--db", default="data/beta-spy.sqlite")
    replay.add_argument("--universe", default="config/universe.csv")
    replay.add_argument("--start")
    replay.add_argument("--end")

    backtest = sub.add_parser("backtest", help="Run causal replay and write Markdown/JSON metrics")
    backtest.add_argument("--db", default="data/beta-spy.sqlite")
    backtest.add_argument("--universe", default="config/universe.csv")
    backtest.add_argument("--start")
    backtest.add_argument("--end")
    backtest.add_argument("--output", default="reports/backtest-latest")

    nightly = sub.add_parser("nightly", help="Fetch recent history and immediately run a causal backtest")
    nightly.add_argument("--db", default="data/beta-spy.sqlite")
    nightly.add_argument("--universe", default="config/universe.csv")
    nightly.add_argument("--days", type=int, default=20)
    nightly.add_argument("--provider", choices=("auto", "tradier", "alpaca"), default="auto")
    nightly.add_argument("--interval", choices=("1min", "5min", "15min"), default="1min")
    nightly.add_argument("--feed", default="iex")
    nightly.add_argument("--with-flow", action="store_true")
    nightly.add_argument("--no-refresh-universe", action="store_true")
    nightly.add_argument("--output", default="reports/backtest-latest")

    args = parser.parse_args()

    if args.command == "refresh-universe":
        holdings = fetch_current_spy_universe()
        save_universe_csv(args.output, holdings)
        print(f"saved {len(holdings)} holdings to {args.output}")
        return

    if args.command == "nightly":
        if args.no_refresh_universe and Path(args.universe).exists():
            holdings = load_universe_csv(args.universe)
        else:
            print("Refreshing current SPY holdings/weights…", flush=True)
            holdings = fetch_current_spy_universe()
            save_universe_csv(args.universe, holdings)
        symbols = [item.symbol for item in holdings]
        if "SPY" not in symbols:
            symbols.append("SPY")
        end = datetime.now(tz=UTC)
        start = end - timedelta(days=max(1, args.days))
        store = Tape500Store(args.db)
        try:
            tradier_token = os.environ.get("TRADIER_MARKET_ACCESS_TOKEN") or os.environ.get("TRADIER_ACCESS_TOKEN") or ""
            alpaca_key = os.environ.get("APCA_API_KEY_ID", "")
            alpaca_secret = os.environ.get("APCA_API_SECRET_KEY", "")
            provider = args.provider
            if provider == "auto":
                provider = "tradier" if tradier_token else "alpaca" if alpaca_key and alpaca_secret else ""
            if provider == "tradier":
                if not tradier_token:
                    raise SystemExit("nightly provider=tradier requires TRADIER_MARKET_ACCESS_TOKEN")
                print(f"Fetching {args.interval} Tradier bars for {len(symbols)} symbols…", flush=True)
                with TradierHistoricalClient(tradier_token) as client:
                    count = client.backfill_bars(store, symbols, start, end, interval=args.interval)
            elif provider == "alpaca":
                if not alpaca_key or not alpaca_secret:
                    raise SystemExit("nightly provider=alpaca requires APCA_API_KEY_ID and APCA_API_SECRET_KEY")
                print(f"Fetching Alpaca bars for {len(symbols)} symbols…", flush=True)
                timeframe = {"1min": "1Min", "5min": "5Min", "15min": "15Min"}[args.interval]
                with AlpacaHistoricalClient(alpaca_key, alpaca_secret) as client:
                    count = client.backfill_bars(store, symbols, start, end, timeframe=timeframe, feed=args.feed)
                    if args.with_flow:
                        print("Fetching historical trades/quotes for 500-name minute flow. This is intentionally slower.", flush=True)
                        flow_count = client.backfill_minute_flow(store, symbols, start, end, feed=args.feed)
                        print(f"stored {flow_count:,} minute-flow rows", flush=True)
            else:
                raise SystemExit("No historical provider credentials found. Set TRADIER_MARKET_ACCESS_TOKEN or Alpaca keys.")
            print(f"stored {count:,} historical bars", flush=True)
            print("Running causal replay/backtest…", flush=True)
            report, _ = run_backtest(store, holdings, start=start, end=end)
            md_path, json_path = write_report(report, args.output)
            print(f"report: {md_path}")
            print(f"metrics: {json_path}")
        finally:
            store.close()
        return

    if args.command == "backtest":
        holdings = load_universe_csv(args.universe)
        store = Tape500Store(args.db)
        try:
            report, _ = run_backtest(
                store, holdings,
                start=_time(args.start) if args.start else None,
                end=_time(args.end) if args.end else None,
            )
            md_path, json_path = write_report(report, args.output)
            print(f"report: {md_path}")
            print(f"metrics: {json_path}")
        finally:
            store.close()
        return

    if args.command == "demo":
        store = Tape500Store(args.db)
        hub = StateHub()
        feed = DemoFeed(Tape500Engine(demo_holdings(), store=store), hub)
        thread = threading.Thread(target=feed.run_forever, daemon=True, name="beta-spy-demo")
        thread.start()
        try:
            uvicorn.run(create_app(hub), host=args.host, port=args.port, log_level="info")
        finally:
            feed.stop()
            thread.join(timeout=2)
            store.close()
        return

    if args.command == "run":
        token = os.environ.get("TRADIER_MARKET_ACCESS_TOKEN") or os.environ.get("TRADIER_ACCESS_TOKEN") or ""
        if not token:
            raise SystemExit("Set TRADIER_MARKET_ACCESS_TOKEN (or TRADIER_ACCESS_TOKEN)")
        holdings = _load_holdings(args.universe)
        store = Tape500Store(args.db)
        hub = StateHub()
        engine = Tape500Engine(holdings, store=store)
        ledger = PaperLedger(
            store,
            daily_loss_limit_dollars=args.daily_loss_limit,
            daily_loss_fraction=args.daily_loss_fraction,
            starting_equity=args.bankroll if args.bankroll > 0 else None,
            risk_fraction_per_trade=args.risk_fraction,
        )

        def _warm_start() -> None:
            with store.lock:
                dates = [
                    str(row[0])
                    for row in store.connection.execute(
                        "SELECT DISTINCT substr(timestamp, 1, 10) FROM minute_bars ORDER BY 1"
                    ).fetchall()
                ]
            recent = dates[-args.warm_sessions :]
            if not recent:
                return
            start = datetime.fromisoformat(recent[0] + "T00:00:00+00:00")
            print(f"warm-start: replaying {len(recent)} stored sessions from {recent[0]}…", flush=True)
            count = 0
            for _ in HistoricalReplay(store, engine).run(start=start):
                count += 1
            print(f"warm-start: replayed {count} snapshots; models ready", flush=True)

        stream = TradierMarketStream(
            token,
            engine,
            hub,
            maximum_option_risk_dollars=args.risk,
            ledger=ledger,
            warmup=_warm_start if args.warm_sessions > 0 else None,
            alpha_state_url=args.alpha_state_url,
        )
        thread = threading.Thread(target=stream.run_forever, daemon=True, name="tradier-tape")
        thread.start()
        try:
            uvicorn.run(create_app(hub), host=args.host, port=args.port, log_level="info")
        finally:
            stream.stop()
            thread.join(timeout=5)
            stream.close()
            store.close()
        return

    holdings = load_universe_csv(args.universe)
    store = Tape500Store(args.db)
    try:
        if args.command in {"backfill-bars", "backfill-flow"}:
            key = os.environ.get("APCA_API_KEY_ID", "")
            secret = os.environ.get("APCA_API_SECRET_KEY", "")
            if not key or not secret:
                raise SystemExit("Set APCA_API_KEY_ID and APCA_API_SECRET_KEY")
            symbols = [item.symbol for item in holdings]
            if "SPY" not in symbols:
                symbols.append("SPY")
            with AlpacaHistoricalClient(key, secret) as client:
                if args.command == "backfill-bars":
                    count = client.backfill_bars(store, symbols, _time(args.start), _time(args.end), feed=args.feed)
                else:
                    count = client.backfill_minute_flow(store, symbols, _time(args.start), _time(args.end), feed=args.feed)
            print(f"stored {count} rows")
            return

        engine = Tape500Engine(holdings, store=store)
        count = 0
        last = None
        for last in HistoricalReplay(store, engine).run(
            start=_time(args.start) if args.start else None,
            end=_time(args.end) if args.end else None,
        ):
            count += 1
        if last is None:
            print("no replayable snapshots")
        else:
            print(f"replayed {count}; last={last.timestamp.isoformat()} {last.decision.action}/{last.decision.direction}")
    finally:
        store.close()


if __name__ == "__main__":
    main()
