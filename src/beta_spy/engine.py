from __future__ import annotations

from collections import defaultdict
from collections.abc import Iterable
from dataclasses import dataclass, replace
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from .breadth import BreadthAggregator
from .decision import DecisionEngine
from .flow import FlowAccumulator
from .forecast import OnlineForecastStack, OnlineMetaGate, meta_vector, vectorize
from .indicators import SymbolIndicatorState
from .models import (
    EngineSnapshot,
    FlowFeatures,
    HoldingMeta,
    MinuteBar,
    QuoteTop,
    SymbolFeatures,
    TradePrint,
)
from .storage import Tape500Store


def _event_time(value: Any) -> datetime:
    try:
        raw = float(value)
    except (TypeError, ValueError):
        return datetime.now(tz=UTC)
    if raw > 10_000_000_000:
        raw /= 1000.0
    return datetime.fromtimestamp(raw, tz=UTC)


def _minute_key(timestamp: datetime) -> datetime:
    return timestamp.astimezone(UTC).replace(second=0, microsecond=0)


@dataclass
class _BarBuilder:
    symbol: str
    timestamp: datetime
    open: float
    high: float
    low: float
    close: float
    volume: float = 0.0
    trade_count: int = 0
    pv: float = 0.0

    @classmethod
    def start(cls, trade: TradePrint) -> _BarBuilder:
        return cls(
            symbol=trade.symbol,
            timestamp=_minute_key(trade.timestamp),
            open=trade.price,
            high=trade.price,
            low=trade.price,
            close=trade.price,
            volume=trade.size,
            trade_count=1,
            pv=trade.price * trade.size,
        )

    def add(self, trade: TradePrint) -> None:
        self.high = max(self.high, trade.price)
        self.low = min(self.low, trade.price)
        self.close = trade.price
        self.volume += trade.size
        self.trade_count += 1
        self.pv += trade.price * trade.size

    def finish(self) -> MinuteBar:
        return MinuteBar(
            symbol=self.symbol,
            timestamp=self.timestamp,
            open=self.open,
            high=self.high,
            low=self.low,
            close=self.close,
            volume=self.volume,
            trade_count=self.trade_count,
            vwap=self.pv / self.volume if self.volume > 0 else None,
        )


class Tape500Engine:
    """Simplified constituent-tape intelligence engine.

    The engine consumes ordinary minute bars plus optional top-of-book quotes and time-and-sales.
    All ~500 constituents act as sensors. Their features are aggregated twice: equally weighted
    and by SPY weight, with sector sub-aggregates retained for diagnosis.
    """

    def __init__(
        self,
        holdings: Iterable[HoldingMeta],
        *,
        store: Tape500Store | None = None,
        forecast_stack: OnlineForecastStack | None = None,
        decision_engine: DecisionEngine | None = None,
    ) -> None:
        holdings = tuple(holdings)
        self.holdings = {item.symbol: item for item in holdings}
        self.expected_symbol_count = len(self.holdings)
        self.states: dict[str, SymbolIndicatorState] = {
            item.symbol: SymbolIndicatorState(item) for item in holdings
        }
        self.states.setdefault("SPY", SymbolIndicatorState(HoldingMeta("SPY", "ETF", 0.0, "SPDR S&P 500 ETF")))
        self.flows: dict[str, FlowAccumulator] = defaultdict(FlowAccumulator)
        self.builders: dict[str, _BarBuilder] = {}
        self.flow_overrides: dict[str, FlowFeatures] = {}
        self.store = store
        self.aggregator = BreadthAggregator()
        self.forecasts = forecast_stack or OnlineForecastStack()
        self.decisions = decision_engine or DecisionEngine()
        self.meta_gate = OnlineMetaGate()

    @classmethod
    def from_holdings(
        cls,
        holdings: Iterable[Any],
        *,
        database: Path | str | None = None,
    ) -> Tape500Engine:
        metas = [
            HoldingMeta(
                symbol=str(item.symbol),
                sector=str(item.sector),
                weight=float(item.weight),
                name=str(getattr(item, "name", "")),
            )
            for item in holdings
        ]
        store = Tape500Store(database) if database is not None else None
        return cls(metas, store=store)

    def set_universe(self, holdings: Iterable[HoldingMeta]) -> None:
        metas = tuple(holdings)
        self.holdings = {item.symbol: item for item in metas}
        self.expected_symbol_count = len(metas)
        for meta in metas:
            if meta.symbol not in self.states:
                self.states[meta.symbol] = SymbolIndicatorState(meta)
            elif self.states[meta.symbol].meta != meta:
                old = self.states[meta.symbol]
                replacement = SymbolIndicatorState(meta, max_bars=old.max_bars)
                replacement.bars.extend(old.bars)
                replacement.session_pv = old.session_pv
                replacement.session_volume = old.session_volume
                replacement.session_date = old.session_date
                self.states[meta.symbol] = replacement

    def add_bar(
        self,
        bar: MinuteBar,
        *,
        persist: bool = True,
        flow: FlowFeatures | None = None,
    ) -> None:
        state = self.states.get(bar.symbol)
        if state is None:
            meta = self.holdings.get(bar.symbol, HoldingMeta(bar.symbol, "Unknown", 0.0, bar.symbol))
            state = self.states.setdefault(bar.symbol, SymbolIndicatorState(meta))
        state.add_bar(bar)
        if flow is not None:
            self.flow_overrides[bar.symbol] = flow
        if persist and self.store is not None:
            self.store.save_bar(bar)
            if flow is not None:
                self.store.save_flow(bar.timestamp, bar.symbol, flow)

    def on_quote(self, quote: QuoteTop) -> None:
        self.flows[quote.symbol].on_quote(quote)
        if self.store is not None and quote.symbol == "SPY":
            with self.store.lock:
                self.store.connection.execute(
                    "INSERT OR IGNORE INTO spy_quotes(timestamp,bid,ask,bid_size,ask_size) VALUES(?,?,?,?,?)",
                    (
                        quote.timestamp.isoformat().replace("+00:00", "Z"),
                        quote.bid, quote.ask, quote.bid_size, quote.ask_size,
                    ),
                )
                self.store.connection.commit()

    def on_trade(self, trade: TradePrint) -> None:
        self.flows[trade.symbol].on_trade(trade)
        bucket = _minute_key(trade.timestamp)
        current = self.builders.get(trade.symbol)
        if current is None:
            self.builders[trade.symbol] = _BarBuilder.start(trade)
        elif current.timestamp == bucket:
            current.add(trade)
        elif current.timestamp < bucket:
            self.add_bar(current.finish())
            self.builders[trade.symbol] = _BarBuilder.start(trade)
        if self.store is not None and trade.symbol == "SPY":
            with self.store.lock:
                self.store.connection.execute(
                    "INSERT OR IGNORE INTO spy_trades(timestamp,sequence,price,size,bid,ask) VALUES(?,?,?,?,?,?)",
                    (
                        trade.timestamp.isoformat().replace("+00:00", "Z"), trade.sequence,
                        trade.price, trade.size, trade.bid, trade.ask,
                    ),
                )
                self.store.connection.commit()

    def flush_completed_bars(self, timestamp: datetime) -> None:
        boundary = _minute_key(timestamp)
        for symbol, builder in list(self.builders.items()):
            if builder.timestamp < boundary:
                self.add_bar(builder.finish())
                del self.builders[symbol]

    def ingest_tradier_event(self, event: dict[str, Any]) -> None:
        symbol = str(event.get("symbol") or "")
        if not symbol:
            return
        kind = str(event.get("type") or "").lower()
        timestamp = _event_time(event.get("date") or event.get("askdate") or event.get("biddate"))
        if kind == "quote":
            try:
                bid = float(event.get("bid") or 0.0)
                ask = float(event.get("ask") or 0.0)
            except (TypeError, ValueError):
                return
            if bid <= 0 or ask < bid:
                return
            self.on_quote(
                QuoteTop(
                    symbol=symbol,
                    timestamp=timestamp,
                    bid=bid,
                    ask=ask,
                    bid_size=_float_or_none(event.get("bidsz")),
                    ask_size=_float_or_none(event.get("asksz")),
                )
            )
            return
        if kind != "timesale" or bool(event.get("cancel")) or bool(event.get("correction")):
            return
        try:
            price = float(event.get("last") or 0.0)
            size = float(event.get("size") or 0.0)
        except (TypeError, ValueError):
            return
        if price <= 0 or size <= 0:
            return
        self.on_trade(
            TradePrint(
                symbol=symbol,
                timestamp=timestamp,
                price=price,
                size=size,
                bid=_float_or_none(event.get("bid")),
                ask=_float_or_none(event.get("ask")),
                sequence=_int_or_none(event.get("seq")),
            )
        )

    def build_snapshot(self, timestamp: datetime) -> EngineSnapshot | None:
        self.flush_completed_bars(timestamp)
        symbol_features: list[SymbolFeatures] = []
        for symbol, state in self.states.items():
            flow = self.flow_overrides.get(symbol) or self.flows[symbol].snapshot()
            feature = state.features(flow, timestamp)
            if feature is not None:
                symbol_features.append(feature)
        factors = self.aggregator.aggregate(
            symbol_features,
            timestamp=timestamp,
            expected_symbol_count=self.expected_symbol_count,
        )
        spy = next((item for item in symbol_features if item.symbol == "SPY"), None)
        if spy is None or spy.close <= 0:
            return None
        if self.store is not None and self.decisions.session_open_price is None:
            self.decisions.recover_session_open(self.store.first_rth_spy_price(timestamp))
        forecasts = self.forecasts.step(timestamp, factors, spy.close)
        decision = self.decisions.decide(timestamp, factors, forecasts, spy_price=spy.close)
        self.meta_gate.mature(timestamp, spy.close)
        if decision.action == "TRADE":
            direction = 1 if decision.direction == "BULLISH" else -1
            x_meta = meta_vector(vectorize(factors), forecasts, direction)
            # Queue before filtering so the meta model trains on every gated
            # signal, including the ones it vetoes (no selection bias).
            self.meta_gate.queue(timestamp, x_meta, direction, spy.close)
            win_probability = self.meta_gate.win_probability(x_meta)
            if win_probability is not None and win_probability < self.meta_gate.threshold:
                decision = replace(
                    decision,
                    action="NO_TRADE",
                    gates={**decision.gates, "meta_filter": False},
                    reasons=(
                        f"Meta-model win probability {win_probability:.2f} is below "
                        f"{self.meta_gate.threshold:.2f}",
                    ),
                    structure=None,
                )
        snapshot = EngineSnapshot(
            timestamp=timestamp,
            factors=factors,
            forecasts=forecasts,
            decision=decision,
            symbols=tuple(symbol_features),
        )
        if self.store is not None:
            self.store.save_factors(factors)
            self.store.save_forecasts(timestamp, forecasts)
            self.store.save_decision(decision)
        for flow in self.flows.values():
            flow.reset()
        self.flow_overrides.clear()
        return snapshot


def _float_or_none(value: Any) -> float | None:
    try:
        return float(value) if value is not None else None
    except (TypeError, ValueError):
        return None


def _int_or_none(value: Any) -> int | None:
    try:
        return int(value) if value is not None else None
    except (TypeError, ValueError):
        return None
