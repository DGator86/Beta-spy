from __future__ import annotations

from datetime import datetime, timedelta
from typing import Iterator

from .engine import Tape500Engine
from .models import EngineSnapshot, MinuteBar, QuoteTop, TradePrint
from .storage import Tape500Store


class HistoricalReplay:
    """Bar replay that also rebuilds SPY auction/CVD from stored prints.

    minute_bars are the constituent sensor. spy_trades / spy_quotes are the
    microstructure tape. Flow windows come from minute_flow when present,
    otherwise from the prints applied into this minute's accumulator.
    """

    def __init__(self, store: Tape500Store, engine: Tape500Engine):
        self.store = store
        self.engine = engine
        self._trade_day = None
        self._trades: list[TradePrint] = []
        self._quotes: list[QuoteTop] = []
        self._ti = 0
        self._qi = 0

    def run(self, start: datetime | None = None, end: datetime | None = None) -> Iterator[EngineSnapshot]:
        current_time: datetime | None = None
        bucket: list[MinuteBar] = []
        for bar in self.store.iter_bars(start=start, end=end):
            if current_time is None:
                current_time = bar.timestamp
            if bar.timestamp != current_time:
                snapshot = self._step(current_time, bucket)
                if snapshot is not None:
                    yield snapshot
                bucket = []
                current_time = bar.timestamp
            bucket.append(bar)
        if current_time is not None and bucket:
            snapshot = self._step(current_time, bucket)
            if snapshot is not None:
                yield snapshot

    def _load_day(self, timestamp: datetime) -> None:
        day = timestamp.date()
        if self._trade_day == day:
            return
        self._trade_day = day
        self._trades = list(self.store.iter_spy_trades(day))
        try:
            self._quotes = list(self.store.iter_spy_quotes(day))
        except Exception:  # noqa: BLE001 - older DBs may lack spy_quotes
            self._quotes = []
        self._ti = 0
        self._qi = 0

    def _apply_spy_tape(self, timestamp: datetime) -> None:
        self._load_day(timestamp)
        horizon = timestamp + timedelta(minutes=1)
        while self._qi < len(self._quotes) and self._quotes[self._qi].timestamp <= horizon:
            self.engine.apply_quote(self._quotes[self._qi])
            self._qi += 1
        while self._ti < len(self._trades) and self._trades[self._ti].timestamp <= horizon:
            self.engine.apply_print(self._trades[self._ti])
            self._ti += 1

    def _restore_constituent_cvd(self, timestamp: datetime) -> None:
        for symbol, value in self.store.session_cvd_for_timestamp(timestamp).items():
            if symbol == "SPY":
                continue
            self.engine.cvd[symbol].restore(value, timestamp.date())

    def _step(self, timestamp: datetime, bars: list[MinuteBar]) -> EngineSnapshot | None:
        self._apply_spy_tape(timestamp)
        self._restore_constituent_cvd(timestamp)
        flows = self.store.flows_for_timestamp(timestamp)
        for bar in bars:
            self.engine.add_bar(bar, persist=False, flow=flows.get(bar.symbol))
        return self.engine.build_snapshot(timestamp)
