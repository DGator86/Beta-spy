from __future__ import annotations

import json
import threading
import time
from collections.abc import Callable
from dataclasses import asdict
from datetime import UTC, datetime
from zoneinfo import ZoneInfo
from typing import Any

import httpx
from websockets.exceptions import ConnectionClosed
from websockets.sync.client import connect

from .engine import Tape500Engine
from .ledger import PaperLedger
from .models import EngineSnapshot
from .options import OptionPlan, plan_best_strategy
from .session_policy import is_rth_entry

TRADIER_API = "https://api.tradier.com/v1"
TRADIER_WS = "wss://ws.tradier.com/v1/markets/events"


class StateHub:
    def __init__(self) -> None:
        self._lock = threading.RLock()
        self._state: dict[str, Any] = {
            "status": "STARTING",
            "timestamp": datetime.now(UTC).isoformat(),
            "snapshot": None,
            "option_plan": None,
            "ledger": None,
            "stream": {"events": 0, "reconnects": 0, "last_error": None},
        }

    def update(self, **values: Any) -> None:
        with self._lock:
            self._state.update(values)

    def patch_stream(self, **values: Any) -> None:
        with self._lock:
            stream = dict(self._state.get("stream") or {})
            stream.update(values)
            self._state["stream"] = stream

    def snapshot(self) -> dict[str, Any]:
        with self._lock:
            return json.loads(json.dumps(self._state, default=_json_default))


class TradierMarketStream:
    """One consolidated Tradier stream for the SPY constituent tape.

    Only quote + uniquely sequenced timesale events are subscribed. This avoids
    double counting prints across Tradier's trade/timesale payload families.
    """

    def __init__(
        self,
        access_token: str,
        engine: Tape500Engine,
        hub: StateHub,
        *,
        snapshot_seconds: float = 60.0,
        maximum_option_risk_dollars: float = 100.0,
        ledger: PaperLedger | None = None,
        warmup: Callable[[], None] | None = None,
        alpha_state_url: str | None = None,
    ) -> None:
        self.token = access_token.strip()
        if not self.token:
            raise ValueError("Tradier access token is required")
        self.engine = engine
        self.hub = hub
        self.snapshot_seconds = snapshot_seconds
        self.maximum_option_risk_dollars = maximum_option_risk_dollars
        self.ledger = ledger
        self.warmup = warmup
        self.alpha_state_url = (alpha_state_url or "").strip() or None
        self.mark_seconds = 20.0
        self._stop = threading.Event()
        self._events = 0
        self._reconnects = 0
        self._last_option_refresh = 0.0
        self._cached_chain: list[dict[str, Any]] | None = None
        self._cached_expiration: str | None = None
        self._http = httpx.Client(
            base_url=TRADIER_API,
            headers={"Authorization": f"Bearer {self.token}", "Accept": "application/json"},
            timeout=20.0,
        )

    def stop(self) -> None:
        self._stop.set()

    def close(self) -> None:
        self._http.close()

    def run_forever(self) -> None:
        if self.warmup is not None:
            # Replay stored sessions through the engine so the online models,
            # calibrators, and meta gate are ready at the opening bell instead
            # of spending the first hours of the session warming from zero.
            self.hub.update(status="WARMSTART")
            try:
                self.warmup()
            except Exception as exc:  # noqa: BLE001 - warm-start must never block live start
                self.hub.patch_stream(last_error=f"warm-start failed: {exc}")
        self.engine.recover_spy_microstructure(datetime.now(UTC))
        symbols = list(self.engine.holdings)
        if "SPY" not in symbols:
            symbols.append("SPY")
        delay = 1.0
        while not self._stop.is_set():
            try:
                session_id = self._create_session()
                with connect(
                    TRADIER_WS,
                    open_timeout=20,
                    close_timeout=5,
                    ping_interval=20,
                    ping_timeout=20,
                    compression=None,
                ) as websocket:
                    websocket.send(
                        json.dumps(
                            {
                                "symbols": symbols,
                                "filter": ["quote", "timesale"],
                                "sessionid": session_id,
                                "linebreak": True,
                                "validOnly": True,
                                "advancedDetails": False,
                            }
                        )
                    )
                    delay = 1.0
                    next_snapshot = time.monotonic() + self.snapshot_seconds
                    next_mark = time.monotonic() + self.mark_seconds
                    self.hub.update(status="LIVE")
                    self.hub.patch_stream(last_error=None)
                    while not self._stop.is_set():
                        try:
                            raw = websocket.recv(timeout=1.0)
                        except TimeoutError:
                            raw = None
                        if raw:
                            self._consume(raw)
                        now = time.monotonic()
                        if now >= next_snapshot:
                            self._publish_snapshot(datetime.now(UTC))
                            next_snapshot = now + self.snapshot_seconds
                            next_mark = now + self.mark_seconds
                        elif self.ledger is not None and now >= next_mark:
                            # Fast marks between snapshots: pending entries
                            # fill sooner and stops/targets fire up to 40s
                            # earlier than a once-a-minute mark would allow.
                            self._mark_ledger(datetime.now(UTC))
                            next_mark = now + self.mark_seconds
            except (ConnectionClosed, OSError, RuntimeError, ValueError, httpx.HTTPError) as exc:
                self._reconnects += 1
                self.hub.update(status="DEGRADED")
                self.hub.patch_stream(reconnects=self._reconnects, last_error=str(exc))
                self._stop.wait(delay)
                delay = min(30.0, delay * 2.0)
        self.hub.update(status="STOPPED")

    def _create_session(self) -> str:
        response = self._http.post("/markets/events/session")
        response.raise_for_status()
        payload = response.json()
        session_id = str((payload.get("stream") or {}).get("sessionid") or "")
        if not session_id:
            raise RuntimeError("Tradier did not return a market stream session id")
        return session_id

    def _consume(self, raw: Any) -> None:
        text = raw.decode("utf-8") if isinstance(raw, bytes) else str(raw)
        for line in text.splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                event = json.loads(line)
            except json.JSONDecodeError:
                continue
            if not isinstance(event, dict) or event.get("error"):
                continue
            self.engine.ingest_tradier_event(event)
            self._events += 1
        self.hub.patch_stream(events=self._events, last_event_at=datetime.now(UTC).isoformat())

    def _record_alpha_signal(self, timestamp: datetime) -> dict[str, Any] | None:
        """Sample Alpha-spy's concurrent stance for future agreement analysis.

        Both systems forecast SPY independently; weeks of side-by-side
        recordings are the prerequisite for validating (via CPCV) whether an
        agreement gate between them adds edge. Failures are silent: Alpha
        being down must never affect Beta's own loop.
        """
        if self.alpha_state_url is None:
            return None
        try:
            response = httpx.get(self.alpha_state_url, timeout=5.0)
            response.raise_for_status()
            state = response.json()
        except (httpx.HTTPError, ValueError):
            return None
        if not isinstance(state, dict):
            return None
        market = state.get("market") or {}
        decision = state.get("decision") or {}
        horizons = state.get("forecast_horizons") or {}
        record: dict[str, Any] = {
            "spy_price": market.get("price"),
            "action": decision.get("action"),
            "direction": decision.get("direction") or decision.get("side"),
            "decision_created_at": decision.get("created_at"),
            "horizons": {
                str(name): {
                    "probability_up": (value or {}).get("probability_up"),
                    "expected_return": (value or {}).get("expected_return"),
                    "created_at": (value or {}).get("created_at"),
                }
                for name, value in horizons.items()
                if isinstance(value, dict)
            },
        }
        if self.engine.store is not None:
            self.engine.store.save_alpha_signal(timestamp, record)
        return record

    def _publish_snapshot(self, timestamp: datetime) -> None:
        alpha_signal = self._record_alpha_signal(timestamp)
        if alpha_signal is not None:
            self.hub.update(alpha_signal={**alpha_signal, "recorded_at": timestamp.isoformat()})
        snapshot = self.engine.build_snapshot(timestamp)
        if snapshot is None:
            # The realized track record stays visible even when the tape is
            # quiet (weekends, pre-open) and no snapshot can be built.
            ledger_stats = self.ledger.stats(timestamp) if self.ledger is not None else None
            self.hub.update(timestamp=timestamp.isoformat(), status="WARMING", ledger=ledger_stats)
            return
        plan: OptionPlan | None = None
        if snapshot.decision.action in {"TRADE", "TRADE_NEUTRAL"} and is_rth_entry(timestamp):
            try:
                plan = self._option_plan(snapshot)
            except Exception as exc:  # noqa: BLE001 - chain fetch must not kill the tape
                self.hub.patch_stream(last_error=f"option-plan failed: {exc}")
                plan = None
        ledger_stats = None
        spy_price = None
        spy = next((item for item in snapshot.symbols if item.symbol == "SPY"), None)
        if spy is not None and spy.close > 0:
            spy_price = float(spy.close)
        if self.ledger is not None:
            quotes = self._mark_ledger(timestamp, spy_price=spy_price)
            if plan is not None and is_rth_entry(timestamp) and snapshot.decision.action == "TRADE":
                self.ledger.flatten_for_reverse(plan, timestamp, quotes)
            if plan is not None and is_rth_entry(timestamp):
                self.ledger.open_position(plan, timestamp, spy_price=spy_price)
            ledger_stats = self.ledger.stats(timestamp)
        self.hub.update(
            timestamp=timestamp.isoformat(),
            snapshot=asdict(snapshot),
            option_plan=asdict(plan) if plan else None,
            ledger=ledger_stats,
            status="LIVE",
        )

    def _mark_ledger(
        self, timestamp: datetime, *, spy_price: float | None = None
    ) -> dict[str, tuple[float, float]]:
        assert self.ledger is not None
        quotes: dict[str, tuple[float, float]] = {}
        symbols = self.ledger.open_symbols()
        if not symbols:
            return quotes
        try:
            payload = self._get("/markets/quotes", {"symbols": ",".join(symbols), "greeks": "false"})
        except httpx.HTTPError:
            # A failed quote fetch only delays the next mark; never let it
            # take the snapshot loop down.
            return quotes
        rows = (payload.get("quotes") or {}).get("quote") or []
        if isinstance(rows, dict):
            rows = [rows]
        for row in rows:
            symbol = str(row.get("symbol") or "")
            bid, ask = row.get("bid"), row.get("ask")
            if symbol and bid is not None and ask is not None:
                quotes[symbol] = (float(bid), float(ask))
        self.ledger.mark_positions(
            quotes,
            timestamp,
            spy_price=spy_price,
            tradeable_impulse=bool(self.engine.decisions.tradeable_impulse),
        )
        return quotes

    def _option_plan(self, snapshot: EngineSnapshot) -> OptionPlan | None:
        now = time.monotonic()
        stale_cache = self._cached_chain is None or (now - self._last_option_refresh) >= 20.0
        if stale_cache:
            expirations = self._get("/markets/options/expirations", {"symbol": "SPY", "includeAllRoots": "true"})
            dates = (expirations.get("expirations") or {}).get("date") or []
            if isinstance(dates, str):
                dates = [dates]
            if not dates:
                return None
            today = datetime.now(ZoneInfo("America/New_York")).date().isoformat()
            # 0DTE only: never fall back to the next weekly/1DTE expiry.
            if today not in dates:
                self._cached_chain = None
                self._cached_expiration = None
                return None
            expiration = today
            chain = self._get(
                "/markets/options/chains",
                {"symbol": "SPY", "expiration": expiration, "greeks": "true"},
            )
            rows = (chain.get("options") or {}).get("option") or []
            if isinstance(rows, dict):
                rows = [rows]
            normalized: list[dict[str, Any]] = []
            for row in rows:
                greeks = row.get("greeks") or {}
                normalized.append(
                    {
                        "symbol": row.get("symbol"),
                        "expiration": row.get("expiration_date") or expiration,
                        "strike": row.get("strike"),
                        "right": "C" if str(row.get("option_type")).lower() == "call" else "P",
                        "bid": row.get("bid"),
                        "ask": row.get("ask"),
                        "open_interest": row.get("open_interest") or 0,
                        "delta": greeks.get("delta"),
                        "gamma": greeks.get("gamma"),
                        "theta": greeks.get("theta"),
                    }
                )
            if not normalized:
                return None
            self._cached_chain = normalized
            self._cached_expiration = expiration
            self._last_option_refresh = now
        session_day = datetime.now(ZoneInfo("America/New_York")).date().isoformat()
        if self._cached_expiration != session_day:
            self._cached_chain = None
            self._cached_expiration = None
            return None
        normalized = self._cached_chain or []
        expiration = session_day
        primary = next(
            (
                forecast
                for forecast in snapshot.forecasts
                if forecast.horizon_minutes == snapshot.decision.primary_horizon
            ),
            None,
        )
        # Equity-proportional sizing: the budget is a fraction of current
        # bankroll (so it compounds without a cap), scaled by the decision
        # layer's confidence multiplier and hard-bounded so no single trade
        # can exceed the max fraction of equity. Falls back to the fixed
        # dollar budget when no bankroll is configured.
        base_budget = None
        if self.ledger is not None:
            base_budget = self.ledger.risk_budget_dollars(datetime.now(UTC))
        if base_budget is None:
            base_budget = self.maximum_option_risk_dollars
        risk_budget = base_budget * snapshot.decision.risk_multiplier
        if self.ledger is not None:
            ceiling = self.ledger.max_trade_risk_dollars()
            if ceiling is not None:
                risk_budget = min(risk_budget, ceiling)
        spy = next((item for item in snapshot.symbols if item.symbol == "SPY"), None)
        spy_price = float(spy.close) if spy is not None and spy.close > 0 else None
        expected_move_dollars = 0.0
        if primary is not None and spy_price is not None and snapshot.decision.direction != "NEUTRAL":
            expected_move_dollars = spy_price * primary.expected_return_bps / 10_000.0
        try:
            expiry_close = datetime.strptime(expiration, "%Y-%m-%d").replace(
                hour=20, minute=0, tzinfo=UTC
            )
            minutes_to_expiry = max(
                (expiry_close - datetime.now(UTC)).total_seconds() / 60.0, 30.0
            )
        except ValueError:
            minutes_to_expiry = 390.0
        return plan_best_strategy(
            normalized,
            snapshot.decision.direction,
            maximum_risk_dollars=risk_budget,
            hold_minutes=float(snapshot.decision.hold_minutes or snapshot.decision.primary_horizon),
            spy_price=spy_price,
            expected_move_dollars=expected_move_dollars,
            minutes_to_expiry=minutes_to_expiry,
            max_width=3.5,
        )

    def _get(self, path: str, params: dict[str, str]) -> dict[str, Any]:
        response = self._http.get(path, params=params)
        response.raise_for_status()
        payload = response.json()
        return payload if isinstance(payload, dict) else {}


def _json_default(value: Any) -> Any:
    if isinstance(value, datetime):
        return value.isoformat()
    if hasattr(value, "item"):
        return value.item()
    if hasattr(value, "__dict__"):
        return value.__dict__
    return str(value)
