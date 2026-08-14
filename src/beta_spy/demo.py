from __future__ import annotations

import math
import random
import threading
import time
from dataclasses import asdict
from datetime import UTC, datetime, timedelta

from .engine import Tape500Engine
from .live import StateHub
from .models import FlowFeatures, HoldingMeta, MinuteBar


def demo_holdings(count: int = 120) -> list[HoldingMeta]:
    sectors = ["Technology", "Financials", "Industrials", "Health Care", "Energy", "Consumer"]
    weights = [1.0 / (index + 8) for index in range(count)]
    total = sum(weights)
    return [
        HoldingMeta(f"D{index:03d}", sectors[index % len(sectors)], weights[index] / total, f"Demo {index}")
        for index in range(count)
    ]


class DemoFeed:
    def __init__(self, engine: Tape500Engine, hub: StateHub, *, tick_seconds: float = 0.15) -> None:
        self.engine = engine
        self.hub = hub
        self.tick_seconds = tick_seconds
        self._stop = threading.Event()
        self._minute = 0
        self._prices = {symbol: 80.0 + index * 0.7 for index, symbol in enumerate(engine.holdings)}
        self._prices["SPY"] = 640.0
        self._rng = random.Random(500)

    def stop(self) -> None:
        self._stop.set()

    def run_forever(self) -> None:
        now = datetime.now(UTC).replace(second=0, microsecond=0) - timedelta(minutes=90)
        while not self._stop.is_set():
            phase = self._minute / 13.0
            market = 0.00035 * math.sin(phase) + self._rng.gauss(0.0, 0.00025)
            for symbol in [*self.engine.holdings, "SPY"]:
                old = self._prices[symbol]
                beta = 1.0 if symbol == "SPY" else 0.7 + (hash(symbol) % 60) / 100.0
                move = beta * market + self._rng.gauss(0.0, 0.0007 if symbol != "SPY" else 0.00018)
                close = old * (1.0 + move)
                high = max(old, close) * (1.0 + abs(self._rng.gauss(0, 0.00025)))
                low = min(old, close) * (1.0 - abs(self._rng.gauss(0, 0.00025)))
                volume = 1_000_000 * (0.4 + self._rng.random() * 1.8)
                flow = max(-1.0, min(1.0, move * 850 + self._rng.gauss(0, 0.18)))
                self.engine.add_bar(
                    MinuteBar(symbol, now, old, high, low, close, volume, int(200 + self._rng.random() * 900), (old + close) / 2),
                    persist=True,
                    flow=FlowFeatures(
                        buy_volume=volume * (1 + flow) / 2,
                        sell_volume=volume * (1 - flow) / 2,
                        order_flow_imbalance=flow,
                        quote_imbalance=flow * 0.7,
                        average_spread_bps=0.35 if symbol == "SPY" else 2.0 + self._rng.random() * 3,
                        trade_intensity=500,
                        trades=500,
                    ),
                )
                self._prices[symbol] = close
            snapshot = self.engine.build_snapshot(now + timedelta(seconds=59))
            if snapshot:
                self.hub.update(status="DEMO", timestamp=now.isoformat(), snapshot=asdict(snapshot), option_plan=None)
            self._minute += 1
            now += timedelta(minutes=1)
            self._stop.wait(self.tick_seconds)
