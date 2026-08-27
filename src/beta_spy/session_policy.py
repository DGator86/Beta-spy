"""Causal session and structure policy from the Aug 18–20 0DTE tape.

No lookahead. These rules exist so the live loop cannot repeat the paper
bleed (overnight credits, next-day expiry, 15-minute horizon chops) and so
it can hold the structures that actually paid: $2–$3 0DTE debit verticals
on RTH impulses and defined-risk condors on sub-$3 range days, taken off
before the close probe.
"""
from __future__ import annotations

from datetime import datetime, time
from dataclasses import dataclass
from zoneinfo import ZoneInfo

ET = ZoneInfo("America/New_York")

ENTRY_START = time(9, 40)
ENTRY_STOP = time(15, 40)
CONDOR_FLATTEN = time(15, 50)
FORCED_FLAT = time(15, 55)

RANGE_DAY_DOLLARS = 3.0
FAILED_EXTREME_DOLLARS = 0.80
PREFERRED_DEBIT_WIDTH = (2.0, 3.5)
MIN_RANGE_MINUTES = 90
IMPULSE_DOLLARS = 1.40
REVERSAL_DOLLARS = 0.75
WARMUP_MINUTES = 10
ENTRY_STOP_MINUTES = 370
MIN_DEBIT_DOLLARS = 0.22
MIN_DEBIT_FRACTION = 0.07
MIN_HOLD_MINUTES = 8.0
CONDOR_MIN_WING = 2.0
CONDOR_SHORT_CLEARANCE = 2.0


def session_date(timestamp: datetime):
    return timestamp.astimezone(ET).date()


def _clock(timestamp: datetime) -> time:
    return timestamp.astimezone(ET).time().replace(tzinfo=None)


def is_weekday(timestamp: datetime) -> bool:
    return timestamp.astimezone(ET).weekday() < 5


def is_rth_entry(timestamp: datetime) -> bool:
    if not is_weekday(timestamp):
        return False
    clock = _clock(timestamp)
    return ENTRY_START <= clock < ENTRY_STOP


def minutes_from_open(timestamp: datetime) -> int:
    eastern = timestamp.astimezone(ET)
    return (eastern.hour * 60 + eastern.minute) - (9 * 60 + 30)


def remaining_to_flatten_minutes(timestamp: datetime) -> float:
    eastern = timestamp.astimezone(ET)
    end = eastern.replace(hour=15, minute=50, second=0, microsecond=0)
    return max(8.0, (end - eastern).total_seconds() / 60.0)


def should_force_flat(timestamp: datetime) -> bool:
    if not is_weekday(timestamp):
        return True
    return _clock(timestamp) >= FORCED_FLAT


def should_flatten_condor(timestamp: datetime) -> bool:
    if not is_weekday(timestamp):
        return True
    return _clock(timestamp) >= CONDOR_FLATTEN


def prefer_debit_width(width: float) -> bool:
    lo, hi = PREFERRED_DEBIT_WIDTH
    return lo <= float(width) <= hi


def condor_allowed(*, minutes_open: int, session_range: float | None, trend_day: bool, had_tradeable_impulse: bool = False) -> bool:
    """Condor only after 90 minutes and only while the session is still ≤ $3."""
    if had_tradeable_impulse or trend_day:
        return False
    if minutes_open < MIN_RANGE_MINUTES:
        return False
    if session_range is None:
        return False
    return session_range <= RANGE_DAY_DOLLARS


def condor_geometry_ok(
    *,
    put_width: float,
    call_width: float,
    short_put: float | None,
    short_call: float | None,
    spot: float | None,
) -> bool:
    if put_width < CONDOR_MIN_WING or call_width < CONDOR_MIN_WING:
        return False
    if spot is None:
        return True
    if short_put is not None and abs(float(short_put) - float(spot)) < CONDOR_SHORT_CLEARANCE:
        return False
    if short_call is not None and abs(float(short_call) - float(spot)) < CONDOR_SHORT_CLEARANCE:
        return False
    return True


def should_abort_condor(
    *,
    entry_spot: float | None,
    spot: float | None,
    tradeable: bool = False,
) -> bool:
    if not tradeable or entry_spot is None or spot is None:
        return False
    return abs(float(spot) - float(entry_spot)) >= IMPULSE_DOLLARS


def should_abort_debit(
    *,
    direction: str,
    entry_spot: float | None,
    spot: float | None,
) -> bool:
    """Flatten a directional debit when the impulse reversed $1.40 against it."""
    if entry_spot is None or spot is None:
        return False
    if abs(float(spot) - float(entry_spot)) < IMPULSE_DOLLARS:
        return False
    side = str(direction).upper()
    if side == "BULLISH":
        return float(spot) <= float(entry_spot) - IMPULSE_DOLLARS
    if side == "BEARISH":
        return float(spot) >= float(entry_spot) + IMPULSE_DOLLARS
    return False


@dataclass(frozen=True)
class SwingState:
    pivot: float
    extreme: float
    direction: int
    confirmed: bool
    tradeable: bool = False
    had_tradeable: bool = False
    reversed_from_tradeable: bool = False
    last_tradeable_direction: int = 0
    pending_direction: int = 0
    leg_started_at: datetime | None = None
    hold_minutes: float = 0.0
    leg_steps: int = 0


def empty_swing(spot: float = 0.0) -> SwingState:
    return SwingState(pivot=spot, extreme=spot, direction=0, confirmed=False)


def _hold_minutes(started_at: datetime | None, now: datetime | None, steps: int) -> float:
    if now is not None and started_at is not None:
        return max(0.0, (now - started_at).total_seconds() / 60.0)
    return float(max(0, steps))


def _with_leg(
    *,
    pivot: float,
    extreme: float,
    direction: int,
    prev: SwingState,
    now: datetime | None,
    impulse: float,
    reversed_from_tradeable: bool,
    new_leg: bool,
) -> SwingState:
    move = abs(float(extreme) - float(pivot)) if direction else 0.0
    confirmed = move >= impulse
    if new_leg:
        started = now
        steps = 1
        hold = 0.0 if now is not None else 1.0
    else:
        started = prev.leg_started_at if prev.leg_started_at is not None else now
        steps = prev.leg_steps + 1
        hold = _hold_minutes(started, now, steps)
    tradeable = confirmed and hold >= MIN_HOLD_MINUTES
    had = bool(prev.had_tradeable or prev.tradeable or tradeable)
    if tradeable:
        last_dir = direction
        pending = 0
    elif reversed_from_tradeable:
        last_dir = prev.direction if prev.tradeable else prev.last_tradeable_direction
        pending = direction
    elif new_leg:
        last_dir = prev.last_tradeable_direction
        pending = 0
    else:
        last_dir = prev.last_tradeable_direction
        pending = prev.pending_direction
    if prev.tradeable and not tradeable and not reversed_from_tradeable and not new_leg:
        last_dir = prev.direction
    return SwingState(
        pivot=pivot,
        extreme=extreme,
        direction=direction,
        confirmed=confirmed,
        tradeable=tradeable,
        had_tradeable=had,
        reversed_from_tradeable=reversed_from_tradeable,
        last_tradeable_direction=last_dir,
        pending_direction=pending,
        leg_started_at=started,
        hold_minutes=hold,
        leg_steps=steps,
    )


def swing_step(
    state: SwingState,
    spot: float,
    *,
    now: datetime | None = None,
    impulse: float = IMPULSE_DOLLARS,
    reversal: float = REVERSAL_DOLLARS,
) -> SwingState:
    px = float(spot)
    if state.pivot <= 0:
        return empty_swing(px)
    if state.direction == 0:
        if px == state.pivot:
            return state
        direction = 1 if px > state.pivot else -1
        return _with_leg(
            pivot=state.pivot,
            extreme=px,
            direction=direction,
            prev=state,
            now=now,
            impulse=impulse,
            reversed_from_tradeable=False,
            new_leg=True,
        )
    extending = (state.direction > 0 and px >= state.extreme) or (
        state.direction < 0 and px <= state.extreme
    )
    reversing = (state.direction > 0 and state.extreme - px >= reversal) or (
        state.direction < 0 and px - state.extreme >= reversal
    )
    if extending:
        return _with_leg(
            pivot=state.pivot,
            extreme=px,
            direction=state.direction,
            prev=state,
            now=now,
            impulse=impulse,
            reversed_from_tradeable=False,
            new_leg=False,
        )
    if reversing:
        return _with_leg(
            pivot=state.extreme,
            extreme=px,
            direction=-state.direction,
            prev=state,
            now=now,
            impulse=impulse,
            reversed_from_tradeable=state.tradeable,
            new_leg=True,
        )
    return _with_leg(
        pivot=state.pivot,
        extreme=state.extreme,
        direction=state.direction,
        prev=state,
        now=now,
        impulse=impulse,
        reversed_from_tradeable=False,
        new_leg=False,
    )


def classify_situation(
    *,
    minutes_open: int,
    session_range: float | None,
    confirmed_impulse: bool,
    trend_day: bool = False,
    had_tradeable_impulse: bool = False,
) -> str:
    if minutes_open < 0 or minutes_open >= ENTRY_STOP_MINUTES:
        return "CLOSED"
    if confirmed_impulse:
        return "IMPULSE"
    if had_tradeable_impulse:
        return "TREND"
    if session_range is None:
        return "UNKNOWN"
    range_val = float(session_range)
    if minutes_open < WARMUP_MINUTES:
        return "WARMUP"
    if trend_day or range_val > RANGE_DAY_DOLLARS:
        return "TREND"
    if minutes_open >= MIN_RANGE_MINUTES and range_val <= RANGE_DAY_DOLLARS:
        return "RANGE"
    return "WATCH"


def debit_premium_ok(entry_price: float | None, width: float | None) -> bool:
    if entry_price is None or width is None:
        return False
    debit = float(entry_price)
    span = float(width)
    low, high = PREFERRED_DEBIT_WIDTH
    if span < low or span > high:
        return False
    return debit >= max(MIN_DEBIT_DOLLARS, MIN_DEBIT_FRACTION * span)

