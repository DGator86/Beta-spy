from __future__ import annotations

import json
import threading
import time
from dataclasses import asdict
from datetime import UTC, datetime
from typing import Any

import httpx
from websockets.exceptions import ConnectionClosed
from websockets.sync.client import connect

from .engine import Tape500Engine
from .models import EngineSnapshot
from .options import OptionPlan, plan_debit_spread

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
    ) -> None:
        self.token = access_token.strip()
        if not self.token:
            raise ValueError("Tradier access token is required")
        self.engine = engine
        self.hub = hub
        self.snapshot_seconds = snapshot_seconds
        self.maximum_option_risk_dollars = maximum_option_risk_dollars
        self._stop = threading.Event()
        self._events = 0
        self._reconnects = 0
        self._last_option_refresh = 0.0
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
                    self.hub.update(status="LIVE")
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

    def _publish_snapshot(self, timestamp: datetime) -> None:
        snapshot = self.engine.build_snapshot(timestamp)
        if snapshot is None:
            self.hub.update(timestamp=timestamp.isoformat(), status="WARMING")
            return
        plan: OptionPlan | None = None
        if snapshot.decision.action == "TRADE":
            plan = self._option_plan(snapshot)
        self.hub.update(
            timestamp=timestamp.isoformat(),
            snapshot=asdict(snapshot),
            option_plan=asdict(plan) if plan else None,
            status="LIVE",
        )

    def _option_plan(self, snapshot: EngineSnapshot) -> OptionPlan | None:
        now = time.monotonic()
        if now - self._last_option_refresh < 20.0:
            return None
        self._last_option_refresh = now
        expirations = self._get("/markets/options/expirations", {"symbol": "SPY", "includeAllRoots": "true"})
        dates = (expirations.get("expirations") or {}).get("date") or []
        if isinstance(dates, str):
            dates = [dates]
        if not dates:
            return None
        today = datetime.now().date().isoformat()
        eligible = sorted(date for date in dates if date >= today)
        if not eligible:
            return None
        expiration = today if today in dates else eligible[0]
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
                }
            )
        primary = next(
            (
                forecast
                for forecast in snapshot.forecasts
                if forecast.horizon_minutes == snapshot.decision.primary_horizon
            ),
            None,
        )
        expected_move_dollars: float | None = None
        probability: float | None = None
        risk_budget = self.maximum_option_risk_dollars * snapshot.decision.risk_multiplier
        if primary is not None:
            spy = next((item for item in snapshot.symbols if item.symbol == "SPY"), None)
            if spy is not None and spy.close > 0:
                expected_move_dollars = spy.close * abs(primary.expected_return_bps) / 10_000.0
            probability = max(primary.probability_up, 1.0 - primary.probability_up)
        return plan_debit_spread(
            normalized,
            snapshot.decision.direction,
            maximum_risk_dollars=risk_budget,
            expected_move_dollars=expected_move_dollars,
            probability=probability,
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
