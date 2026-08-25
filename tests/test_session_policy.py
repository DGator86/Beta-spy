from datetime import datetime
from zoneinfo import ZoneInfo

from beta_spy.session_policy import (
    classify_situation,
    condor_allowed,
    condor_geometry_ok,
    debit_premium_ok,
    should_abort_condor,
    should_abort_debit,
    swing_step,
    empty_swing,
)

ET = ZoneInfo("America/New_York")


def test_condor_requires_90_minutes_and_sub_three_range() -> None:
    assert condor_allowed(minutes_open=30, session_range=1.0, trend_day=False) is False
    assert condor_allowed(minutes_open=47, session_range=2.10, trend_day=False) is False
    assert condor_allowed(minutes_open=91, session_range=3.54, trend_day=False) is False
    assert condor_allowed(minutes_open=91, session_range=2.10, trend_day=True) is False
    assert condor_allowed(minutes_open=91, session_range=2.10, trend_day=False) is True


def test_one_dollar_wings_against_spot_rejected() -> None:
    assert condor_geometry_ok(
        put_width=1.0,
        call_width=1.0,
        short_put=759.0,
        short_call=765.0,
        spot=764.78,
    ) is False
    assert condor_geometry_ok(
        put_width=2.0,
        call_width=2.0,
        short_put=762.0,
        short_call=768.0,
        spot=765.0,
    ) is True


def test_condor_aborts_after_dollar_forty_impulse() -> None:
    assert should_abort_condor(entry_spot=764.78, spot=766.20) is False
    assert should_abort_condor(entry_spot=764.78, spot=766.20, tradeable=True) is True
    assert should_abort_condor(entry_spot=764.78, spot=765.50, tradeable=True) is False


def test_debit_aborts_only_when_the_impulse_is_against_it() -> None:
    assert should_abort_debit(direction="BEARISH", entry_spot=764.78, spot=766.25) is True
    assert should_abort_debit(direction="BEARISH", entry_spot=764.78, spot=763.20) is False
    assert should_abort_debit(direction="BULLISH", entry_spot=764.78, spot=763.20) is True
    assert should_abort_debit(direction="BULLISH", entry_spot=764.78, spot=766.25) is False


def test_range_after_ninety_minutes_without_impulse() -> None:
    assert classify_situation(
        minutes_open=46, session_range=1.78, confirmed_impulse=False
    ) == "WATCH"
    assert classify_situation(
        minutes_open=91, session_range=1.78, confirmed_impulse=False
    ) == "RANGE"
    assert classify_situation(
        minutes_open=28, session_range=1.52, confirmed_impulse=True
    ) == "IMPULSE"
    assert classify_situation(
        minutes_open=91, session_range=1.78, confirmed_impulse=False, had_tradeable_impulse=True
    ) == "TREND"


def test_thin_debit_premium_rejected() -> None:
    assert debit_premium_ok(0.05, 3.0) is False
    assert debit_premium_ok(0.18, 3.0) is False
    assert debit_premium_ok(0.56, 3.0) is True


def test_zigzag_confirms_only_a_dollar_forty_leg() -> None:
    state = empty_swing(770.25)
    for px in (769.80, 769.20, 768.73):
        state = swing_step(state, px)
    assert state.confirmed is True
    assert state.direction == -1
    chop = empty_swing(763.75)
    for px in (764.60, 763.80, 764.55, 763.70):
        chop = swing_step(chop, px)
    assert chop.confirmed is False
