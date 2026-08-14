from __future__ import annotations

from datetime import datetime
from typing import Iterator

from .engine import Tape500Engine
from .models import EngineSnapshot, MinuteBar
from .storage import Tape500Store


class HistoricalReplay:
    def __init__(self, store: Tape500Store, engine: Tape500Engine):
        self.store = store
        self.engine = engine

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

    def _step(self, timestamp: datetime, bars: list[MinuteBar]) -> EngineSnapshot | None:
        flows = self.store.flows_for_timestamp(timestamp)
        for bar in bars:
            self.engine.add_bar(bar, persist=False, flow=flows.get(bar.symbol))
        return self.engine.build_snapshot(timestamp)
