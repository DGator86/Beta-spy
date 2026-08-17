"""Multiscale swing structure, NFS/BOS, sweep/acceptance, candle geometry.

OHLCV geometry only. Not named candlesticks, Elliott labels, or Wyckoff phases.
Prominence is continuous: (pivot - max(neighbors)) / ATR, not a boolean flag.
"""

from __future__ import annotations

from dataclasses import dataclass
from itertools import pairwise

from .models import MinuteBar, StructureFeatures

SCALES = (3, 5, 9, 17, 33)


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


def _prominence(values: list[float], index: int, left: int, atr: float) -> float:
    """(pivot - max(left, right neighbors)) / ATR. Positive = more prominent."""
    if atr <= 0:
        return 0.0
    pivot = values[index]
    left_ext = max(values[index - left : index]) if index - left >= 0 else pivot
    right_ext = max(values[index + 1 : index + 1 + left]) if index + 1 < len(values) else pivot
    return (pivot - max(left_ext, right_ext)) / atr


def _prominence_low(values: list[float], index: int, left: int, atr: float) -> float:
    if atr <= 0:
        return 0.0
    pivot = values[index]
    left_ext = min(values[index - left : index]) if index - left >= 0 else pivot
    right_ext = min(values[index + 1 : index + 1 + left]) if index + 1 < len(values) else pivot
    return (min(left_ext, right_ext) - pivot) / atr


def _clip(value: float, lower: float = -1.0, upper: float = 1.0) -> float:
    return max(lower, min(upper, value))


def _structure_score(
    highs: list[float],
    lows: list[float],
    high_idx: list[int],
    low_idx: list[int],
    atr: float,
    n_bars: int,
) -> float | None:
    """Continuous S in [-1, 1]: +1 strong HH/HL, 0 range, -1 strong LH/LL."""
    if len(high_idx) < 2 or len(low_idx) < 2 or atr <= 0:
        return None
    h0, h1 = highs[high_idx[-2]], highs[high_idx[-1]]
    l0, l1 = lows[low_idx[-2]], lows[low_idx[-1]]
    hh = _clip((h1 - h0) / atr / 2.0)
    hl = _clip((l1 - l0) / atr / 2.0)
    recency = 1.0 - min(n_bars - 1 - max(high_idx[-1], low_idx[-1]), 20) / 20.0
    raw = 0.45 * hh + 0.45 * hl
    return _clip(raw * (0.6 + 0.4 * recency))


def _nfs(
    highs: list[float],
    lows: list[float],
    high_idx: list[int],
    low_idx: list[int],
    bars: list[MinuteBar],
    atr: float,
) -> tuple[float | None, float | None, float | None, float | None, float | None]:
    """Quality of a failed-structure / break-of-structure event, not a boolean."""
    if len(high_idx) < 2 or len(low_idx) < 2 or atr <= 0:
        return None, None, None, None, None
    h0_i, h1_i = high_idx[-2], high_idx[-1]
    l0_i, l1_i = low_idx[-2], low_idx[-1]
    hh = highs[h1_i] > highs[h0_i]
    ll = lows[l1_i] < lows[l0_i]
    hl = lows[l1_i] > lows[l0_i]
    lh = highs[h1_i] < highs[h0_i]
    # Bullish NFS: lower low, then higher high. Bearish: higher high, then lower low.
    if ll and hh and h1_i > l1_i:
        direction = 1.0
        start_i, end_i = l1_i, h1_i
        break_atr = (highs[h1_i] - highs[h0_i]) / atr
    elif hh and ll and l1_i > h1_i:
        direction = -1.0
        start_i, end_i = h1_i, l1_i
        break_atr = (lows[l0_i] - lows[l1_i]) / atr
    elif hl and hh:
        direction = 1.0
        start_i, end_i = l1_i, h1_i
        break_atr = (highs[h1_i] - highs[h0_i]) / atr
    elif lh and ll:
        direction = -1.0
        start_i, end_i = h1_i, l1_i
        break_atr = (lows[l0_i] - lows[l1_i]) / atr
    else:
        return None, None, None, None, None
    duration = float(max(end_i - start_i, 1))
    move = abs(bars[end_i].close - bars[start_i].close)
    path = sum(abs(bars[i].close - bars[i - 1].close) for i in range(start_i + 1, end_i + 1))
    efficiency = move / path if path > 0 else None
    volumes = [bar.volume for bar in bars[start_i : end_i + 1] if bar.volume > 0]
    baseline = [bar.volume for bar in bars[max(0, start_i - 20) : start_i] if bar.volume > 0]
    rel_vol = None
    if volumes and baseline:
        rel_vol = (sum(volumes) / len(volumes)) / (sum(baseline) / len(baseline))
    return direction, break_atr, duration, efficiency, rel_vol


def _sweep_acceptance(
    latest: MinuteBar,
    level_high: float | None,
    level_low: float | None,
    atr: float,
    recent: list[MinuteBar],
) -> tuple[float, float, float, float]:
    """Independent sweep and acceptance scores. They do not sum to one."""
    sweep_high = sweep_low = accept_up = accept_down = 0.0
    if atr <= 0:
        return sweep_high, sweep_low, accept_up, accept_down
    lookback = recent[-5:] if len(recent) >= 2 else recent
    if level_high is not None:
        penetration = (latest.high - level_high) / atr
        if penetration > 0:
            close_beyond = max(0.0, (latest.close - level_high) / atr)
            time_beyond = sum(1 for bar in lookback if bar.close > level_high) / max(len(lookback), 1)
            vol_beyond = 0.0
            if latest.close > level_high and latest.volume > 0:
                avg = sum(bar.volume for bar in lookback) / max(len(lookback), 1)
                vol_beyond = min(latest.volume / avg, 3.0) / 3.0 if avg > 0 else 0.0
            returned = latest.close < level_high
            if returned:
                sweep_high = _clip(min(penetration, 3.0) / 3.0 * (1.0 - close_beyond), 0.0, 1.0)
            accept_up = _clip(0.45 * min(close_beyond, 2.0) / 2.0 + 0.30 * time_beyond + 0.25 * vol_beyond, 0.0, 1.0)
    if level_low is not None:
        penetration = (level_low - latest.low) / atr
        if penetration > 0:
            close_beyond = max(0.0, (level_low - latest.close) / atr)
            time_beyond = sum(1 for bar in lookback if bar.close < level_low) / max(len(lookback), 1)
            vol_beyond = 0.0
            if latest.close < level_low and latest.volume > 0:
                avg = sum(bar.volume for bar in lookback) / max(len(lookback), 1)
                vol_beyond = min(latest.volume / avg, 3.0) / 3.0 if avg > 0 else 0.0
            returned = latest.close > level_low
            if returned:
                sweep_low = _clip(min(penetration, 3.0) / 3.0 * (1.0 - close_beyond), 0.0, 1.0)
            accept_down = _clip(0.45 * min(close_beyond, 2.0) / 2.0 + 0.30 * time_beyond + 0.25 * vol_beyond, 0.0, 1.0)
    return sweep_high, sweep_low, accept_up, accept_down


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
        highs = [bar.high for bar in bars]
        lows = [bar.low for bar in bars]
        prominences: dict[int, float | None] = {}
        last_high_5 = last_low_5 = None
        last_sign_5 = 0
        high_idx_5: list[int] = []
        low_idx_5: list[int] = []
        for left in SCALES:
            high_idx = _swing_indexes(highs, left, kind="high")
            low_idx = _swing_indexes(lows, left, kind="low")
            if left == 5:
                high_idx_5, low_idx_5 = high_idx, low_idx
                last_high_5 = highs[high_idx[-1]] if high_idx else None
                last_low_5 = lows[low_idx[-1]] if low_idx else None
                if high_idx and low_idx:
                    last_sign_5 = 1 if high_idx[-1] > low_idx[-1] else -1
            if atr and atr > 0 and high_idx:
                prominences[left] = _prominence(highs, high_idx[-1], left, atr)
            else:
                prominences[left] = None
        dist_high = (latest.close - last_high_5) / atr if atr and atr > 0 and last_high_5 is not None else None
        dist_low = (latest.close - last_low_5) / atr if atr and atr > 0 and last_low_5 is not None else None
        score = _structure_score(highs, lows, high_idx_5, low_idx_5, atr or 0.0, len(bars))
        nfs_dir, nfs_break, nfs_dur, nfs_eff, nfs_vol = _nfs(
            highs, lows, high_idx_5, low_idx_5, bars, atr or 0.0
        )
        break_strength = None
        failed_break = None
        if atr and atr > 0 and last_high_5 is not None and latest.close > last_high_5:
            break_strength = (latest.close - last_high_5) / atr
        elif atr and atr > 0 and last_low_5 is not None and latest.close < last_low_5:
            break_strength = -(last_low_5 - latest.close) / atr
        if atr and atr > 0 and last_high_5 is not None and latest.high > last_high_5 and latest.close < last_high_5:
            failed_break = (latest.high - last_high_5) / atr
        elif atr and atr > 0 and last_low_5 is not None and latest.low < last_low_5 and latest.close > last_low_5:
            failed_break = (last_low_5 - latest.low) / atr
        sweep_high, sweep_low, accept_up, accept_down = _sweep_acceptance(
            latest, last_high_5, last_low_5, atr or 0.0, bars
        )
        upper, lower, body, close_loc = _candle(latest)
        span = latest.high - latest.low
        displacement = abs(latest.close - latest.open) / span if span > 0 else None
        effort = None
        if latest.volume > 0 and span > 0:
            effort = abs(latest.close - latest.open) / latest.volume
        return StructureFeatures(
            swing_prominence_3=prominences.get(3),
            swing_prominence_5=prominences.get(5),
            swing_prominence_9=prominences.get(9),
            swing_prominence_17=prominences.get(17),
            swing_prominence_33=prominences.get(33),
            distance_to_last_swing_high_atr=dist_high,
            distance_to_last_swing_low_atr=dist_low,
            structure_state=score,
            structure_score=score,
            structure_break_strength=break_strength,
            failed_break_strength=failed_break,
            nfs_direction=nfs_dir,
            nfs_break_atr=nfs_break,
            nfs_duration=nfs_dur,
            nfs_displacement_efficiency=nfs_eff,
            nfs_relative_volume=nfs_vol,
            upper_wick_ratio=upper,
            lower_wick_ratio=lower,
            body_ratio=body,
            close_location=close_loc,
            effort_result_ratio=effort,
            displacement_efficiency=displacement,
            sweep_high_score=sweep_high or None,
            sweep_low_score=sweep_low or None,
            acceptance_above_score=accept_up or None,
            acceptance_below_score=accept_down or None,
            last_swing_sign=float(last_sign_5) if last_sign_5 else None,
        )
