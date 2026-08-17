"""Multiscale swing structure, BOS/failed-break, sweep/acceptance, candle geometry.

These are observable price-path descriptors, not named candlesticks, Elliott
labels, or Wyckoff phases. Prominence windows are bar counts on the minute tape.
"""

from __future__ import annotations

from dataclasses import dataclass
from itertools import pairwise

from .models import MinuteBar, StructureFeatures


def _atr(bars: list[MinuteBar], period: int = 14) -> float | None:
    if len(bars) <= period:
        return None
    recent = bars[-(period + 1) :]
    trs: list[float] = []
    for previous, current in pairwise(recent):
        tr = max(
            current.high - current.low,
            abs(current.high - previous.close),
            abs(current.low - previous.close),
        )
        trs.append(tr)
    return float(sum(trs) / len(trs)) if trs else None


def _swing_indexes(values: list[float], left: int, *, kind: str) -> list[int]:
    """Confirmed fractal: unique max (high) or min (low) with `left` bars on each side."""
    if left < 1 or len(values) < 2 * left + 1:
        return []
    found: list[int] = []
    for i in range(left, len(values) - left):
        window = values[i - left : i + left + 1]
        pivot = values[i]
        if kind == "high" and pivot == max(window) and window.count(pivot) == 1:
            found.append(i)
        elif kind == "low" and pivot == min(window) and window.count(pivot) == 1:
            found.append(i)
    return found


def _last_swings(bars: list[MinuteBar], left: int) -> tuple[float | None, float | None, int]:
    """Return (last swing high, last swing low, last swing sign +1 high / -1 low)."""
    highs = [bar.high for bar in bars]
    lows = [bar.low for bar in bars]
    high_idx = _swing_indexes(highs, left, kind="high")
    low_idx = _swing_indexes(lows, left, kind="low")
    last_high = highs[high_idx[-1]] if high_idx else None
    last_low = lows[low_idx[-1]] if low_idx else None
    last_sign = 0
    last_high_i = high_idx[-1] if high_idx else -1
    last_low_i = low_idx[-1] if low_idx else -1
    if last_high_i > last_low_i:
        last_sign = 1
    elif last_low_i > last_high_i:
        last_sign = -1
    return last_high, last_low, last_sign


def _structure_state(bars: list[MinuteBar], left: int = 5) -> float | None:
    highs = [bar.high for bar in bars]
    lows = [bar.low for bar in bars]
    high_idx = _swing_indexes(highs, left, kind="high")
    low_idx = _swing_indexes(lows, left, kind="low")
    if len(high_idx) < 2 or len(low_idx) < 2:
        return None
    hh = highs[high_idx[-1]] > highs[high_idx[-2]]
    ll = lows[low_idx[-1]] < lows[low_idx[-2]]
    hl = lows[low_idx[-1]] > lows[low_idx[-2]]
    lh = highs[high_idx[-1]] < highs[high_idx[-2]]
    score = 0.0
    if hh:
        score += 1.0
    if hl:
        score += 1.0
    if lh:
        score -= 1.0
    if ll:
        score -= 1.0
    return score


def _candle(bar: MinuteBar) -> tuple[float | None, float | None, float | None, float | None]:
    span = bar.high - bar.low
    if span <= 0:
        return None, None, None, None
    body = abs(bar.close - bar.open)
    upper = bar.high - max(bar.open, bar.close)
    lower = min(bar.open, bar.close) - bar.low
    return upper / span, lower / span, body / span, (bar.close - bar.low) / span


@dataclass
class StructureEngine:
    """Pure function wrapper so tests can call extract() without indicator state."""

    def extract(self, bars: list[MinuteBar]) -> StructureFeatures:
        if not bars:
            return StructureFeatures()
        latest = bars[-1]
        atr = _atr(bars)
        prominences: dict[int, float | None] = {}
        last_high_5 = last_low_5 = None
        last_sign_5 = 0
        for left in (3, 5, 9, 17):
            high, low, sign = _last_swings(bars, left)
            if left == 5:
                last_high_5, last_low_5, last_sign_5 = high, low, sign
            if atr and atr > 0 and high is not None and low is not None:
                prominences[left] = sign * abs(high - low) / atr
            else:
                prominences[left] = None
        dist_high = None
        dist_low = None
        if atr and atr > 0 and last_high_5 is not None:
            dist_high = (latest.close - last_high_5) / atr
        if atr and atr > 0 and last_low_5 is not None:
            dist_low = (latest.close - last_low_5) / atr
        state = _structure_state(bars, 5)
        break_strength = None
        failed_break = None
        sweep_high = sweep_low = 0.0
        accept_above = accept_below = 0.0
        if atr and atr > 0 and last_high_5 is not None and last_low_5 is not None:
            if latest.close > last_high_5:
                break_strength = (latest.close - last_high_5) / atr
                if latest.low > last_high_5:
                    accept_above = min(break_strength, 3.0)
            elif latest.high > last_high_5 and latest.close < last_high_5:
                failed_break = (latest.high - last_high_5) / atr
                sweep_high = min(failed_break, 3.0)
            if latest.close < last_low_5:
                down = (last_low_5 - latest.close) / atr
                break_strength = -down if break_strength is None else break_strength
                if latest.high < last_low_5:
                    accept_below = min(down, 3.0)
            elif latest.low < last_low_5 and latest.close > last_low_5:
                failed = (last_low_5 - latest.low) / atr
                failed_break = failed if failed_break is None else failed_break
                sweep_low = min(failed, 3.0)
        upper, lower, body, close_loc = _candle(latest)
        span = latest.high - latest.low
        displacement = abs(latest.close - latest.open) / span if span > 0 else None
        effort = None
        if latest.volume > 0 and span > 0:
            effort = span / latest.volume
        return StructureFeatures(
            swing_prominence_3=prominences.get(3),
            swing_prominence_5=prominences.get(5),
            swing_prominence_9=prominences.get(9),
            swing_prominence_17=prominences.get(17),
            distance_to_last_swing_high_atr=dist_high,
            distance_to_last_swing_low_atr=dist_low,
            structure_state=state,
            structure_break_strength=break_strength,
            failed_break_strength=failed_break,
            upper_wick_ratio=upper,
            lower_wick_ratio=lower,
            body_ratio=body,
            close_location=close_loc,
            effort_result_ratio=effort,
            displacement_efficiency=displacement,
            sweep_high_score=sweep_high or None,
            sweep_low_score=sweep_low or None,
            acceptance_above_score=accept_above or None,
            acceptance_below_score=accept_below or None,
            last_swing_sign=float(last_sign_5) if last_sign_5 else None,
        )
