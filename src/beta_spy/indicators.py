from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from datetime import datetime

import numpy as np

from .models import AuctionFeatures, FlowFeatures, HoldingMeta, MinuteBar, SymbolFeatures
from .structure import StructureEngine


def _pct_change(values: np.ndarray, periods: int) -> float | None:
    if len(values) <= periods or values[-1 - periods] <= 0:
        return None
    return float(values[-1] / values[-1 - periods] - 1.0)


def _ema(values: np.ndarray, span: int) -> float | None:
    if len(values) == 0:
        return None
    alpha = 2.0 / (span + 1.0)
    current = float(values[0])
    for value in values[1:]:
        current = alpha * float(value) + (1.0 - alpha) * current
    return current


def _rsi(values: np.ndarray, period: int = 14) -> float | None:
    if len(values) <= period:
        return None
    changes = np.diff(values[-(period + 1) :])
    gains = np.clip(changes, 0.0, None)
    losses = -np.clip(changes, None, 0.0)
    avg_gain = float(gains.mean())
    avg_loss = float(losses.mean())
    if avg_loss == 0:
        return 100.0 if avg_gain > 0 else 50.0
    rs = avg_gain / avg_loss
    return 100.0 - (100.0 / (1.0 + rs))


def _atr_bps(bars: list[MinuteBar], period: int = 14) -> float | None:
    if len(bars) <= period:
        return None
    recent = bars[-(period + 1) :]
    trs: list[float] = []
    for previous, current in zip(recent[:-1], recent[1:], strict=True):
        tr = max(
            current.high - current.low,
            abs(current.high - previous.close),
            abs(current.low - previous.close),
        )
        trs.append(tr)
    close = recent[-1].close
    return float(np.mean(trs) / close * 10_000.0) if close > 0 else None


@dataclass
class SymbolIndicatorState:
    meta: HoldingMeta
    max_bars: int = 260
    bars: deque[MinuteBar] = field(default_factory=deque)
    session_pv: float = 0.0
    session_volume: float = 0.0
    session_date: object | None = None
    _structure: StructureEngine = field(default_factory=StructureEngine)

    def add_bar(self, bar: MinuteBar) -> None:
        if self.session_date != bar.timestamp.date():
            self.session_date = bar.timestamp.date()
            self.session_pv = 0.0
            self.session_volume = 0.0
        typical = bar.vwap if bar.vwap is not None else (bar.high + bar.low + bar.close) / 3.0
        if bar.volume > 0:
            self.session_pv += typical * bar.volume
            self.session_volume += bar.volume
        self.bars.append(bar)
        while len(self.bars) > self.max_bars:
            self.bars.popleft()

    def features(
        self,
        flow: FlowFeatures,
        timestamp: datetime | None = None,
        auction: AuctionFeatures | None = None,
    ) -> SymbolFeatures | None:
        if not self.bars:
            return None
        bars = list(self.bars)
        closes = np.asarray([bar.close for bar in bars], dtype=float)
        volumes = np.asarray([max(bar.volume, 0.0) for bar in bars], dtype=float)
        latest = bars[-1]
        timestamp = timestamp or latest.timestamp
        vwap = self.session_pv / self.session_volume if self.session_volume > 0 else None
        ema8 = _ema(closes[-64:], 8)
        ema21 = _ema(closes[-96:], 21)
        ema8_prev = _ema(closes[-65:-1], 8) if len(closes) >= 2 else None
        ema21_prev = _ema(closes[-97:-1], 21) if len(closes) >= 2 else None
        ema8_slope = (
            (ema8 / ema8_prev - 1.0) * 10_000.0 if ema8 and ema8_prev and ema8_prev > 0 else None
        )
        ema21_slope = (
            (ema21 / ema21_prev - 1.0) * 10_000.0 if ema21 and ema21_prev and ema21_prev > 0 else None
        )
        rv20 = None
        if len(closes) >= 21:
            returns = np.diff(np.log(closes[-21:]))
            rv20 = float(np.std(returns, ddof=1) * 10_000.0) if len(returns) > 1 else 0.0
        rel_volume = None
        if len(volumes) >= 2:
            lookback = volumes[-21:-1] if len(volumes) >= 21 else volumes[:-1]
            baseline = float(np.mean(lookback)) if len(lookback) else 0.0
            rel_volume = float(volumes[-1] / baseline) if baseline > 0 else None
        range_expansion = None
        if len(bars) >= 2:
            ranges = np.asarray([(bar.high - bar.low) / bar.close for bar in bars if bar.close > 0])
            if len(ranges) >= 2:
                baseline = float(np.mean(ranges[-21:-1] if len(ranges) >= 21 else ranges[:-1]))
                current = float(ranges[-1])
                range_expansion = current / baseline if baseline > 0 else None
        return SymbolFeatures(
            symbol=self.meta.symbol,
            timestamp=timestamp,
            sector=self.meta.sector,
            weight=self.meta.weight,
            close=latest.close,
            return_1m=_pct_change(closes, 1),
            return_5m=_pct_change(closes, 5),
            return_15m=_pct_change(closes, 15),
            vwap=vwap,
            vwap_distance_bps=((latest.close / vwap - 1.0) * 10_000.0 if vwap and vwap > 0 else None),
            ema8=ema8,
            ema21=ema21,
            ema8_slope_bps=ema8_slope,
            ema21_slope_bps=ema21_slope,
            rsi14=_rsi(closes),
            atr14_bps=_atr_bps(bars),
            realized_vol20_bps=rv20,
            relative_volume20=rel_volume,
            range_expansion=range_expansion,
            flow=flow,
            structure=self._structure.extract(bars),
            auction=auction or AuctionFeatures(),
        )
