"""Tests for the short-premium engine.

The chain-hygiene cases use quotes lifted verbatim from captured tape so that
the regressions are anchored to something that actually happened rather than to
invented numbers.
"""

from __future__ import annotations

import pytest

from beta_spy.premium import (
    ChainGate,
    PremiumConfig,
    PositionState,
    PremiumStructure,
    StructureLeg,
    assess_day,
    build_condor,
    build_credit_spread,
    decay_remaining,
    expected_remaining_move,
    manage_position,
    monitor,
    required_credit_ratio,
    size_position,
    strike_pressure,
)


def row(side, strike, bid, ask, delta=None, oi=1000):
    return {"side": side, "strike": strike, "bid": bid, "ask": ask, "delta": delta, "oi": oi}


# --------------------------------------------------------------------------
# chain hygiene
# --------------------------------------------------------------------------
def test_gate_rejects_quotes_asked_below_intrinsic():
    """Captured 2026-07-23 09:30: spot 739.31, the 749 put asked at 1.96.

    Intrinsic is 9.69. Buying at the ask and exercising is a riskless profit,
    which is the signature of a quote carried over from the prior session.
    """
    gate = ChainGate()
    chain = gate.clean(739.31, [row("put", 749.0, 1.93, 1.96), row("put", 735.0, 0.40, 0.45)])
    assert 749.0 not in chain.puts
    assert 735.0 in chain.puts
    assert chain.rejected["below_intrinsic"] == 1


def test_gate_rejects_the_captured_752_753_put_pair():
    """Captured 2026-07-22 11:38: spot 748.815, 752P mid 2.82, 753P mid 5.52.

    Two separate things are wrong here and either one is disqualifying. The
    752 put is asked at 2.83 against 3.185 of intrinsic, and the pair implies a
    one-point spread worth 2.70. The intrinsic rule fires first, which is why
    ``below_intrinsic`` rather than ``vertical_breach`` is the recorded reason.
    """
    gate = ChainGate()
    chain = gate.clean(
        748.815,
        [row("put", 752.0, 2.81, 2.83), row("put", 753.0, 5.48, 5.56), row("put", 745.0, 0.38, 0.39)],
    )
    assert 752.0 not in chain.puts
    assert 745.0 in chain.puts
    assert chain.rejected["below_intrinsic"] == 1


def test_gate_rejects_verticals_wider_than_their_width():
    """A one-point spread cannot be worth more than one point.

    Both strikes are out of the money here, so nothing trips the intrinsic
    rule; only the vertical bound catches it. Neither quote can be trusted and
    we cannot tell which is wrong, so both leave the book.
    """
    gate = ChainGate()
    chain = gate.clean(
        750.0,
        [row("put", 740.0, 0.20, 0.22), row("put", 741.0, 1.70, 1.74), row("put", 735.0, 0.08, 0.09)],
    )
    assert 740.0 not in chain.puts and 741.0 not in chain.puts
    assert 735.0 in chain.puts
    assert chain.rejected["vertical_breach"] == 2


def test_gate_keeps_a_well_formed_chain_intact():
    gate = ChainGate()
    rows = [row("put", k, v - 0.01, v + 0.01) for k, v in ((745.0, 0.40), (746.0, 0.55), (747.0, 0.75))]
    chain = gate.clean(750.0, rows)
    assert sorted(chain.puts) == [745.0, 746.0, 747.0]
    assert not any(chain.rejected.values())


def test_gate_rejects_crossed_and_zero_quotes():
    gate = ChainGate()
    chain = gate.clean(750.0, [row("put", 745.0, 2.0, 1.0), row("put", 744.0, 0.0, 0.0)])
    assert chain.puts == {}
    assert chain.rejected["malformed"] == 2


# --------------------------------------------------------------------------
# structure geometry
# --------------------------------------------------------------------------
def _condor_chain():
    """Symmetric, arbitrage-free chain around 750."""
    gate = ChainGate()
    rows = []
    for k, mid, d in ((746.0, 0.30, -0.10), (748.0, 0.70, -0.16), (750.0, 2.00, -0.45)):
        rows.append(row("put", k, mid - 0.02, mid + 0.02, d))
    for k, mid, d in ((750.0, 2.00, 0.45), (752.0, 0.70, 0.16), (754.0, 0.30, 0.10)):
        rows.append(row("call", k, mid - 0.02, mid + 0.02, d))
    return gate.clean(750.0, rows)


def test_condor_risk_is_the_widest_wing_not_the_sum():
    """Only one side of an iron condor can finish in the money.

    Summing both verticals is how a structure ends up believing it can lose
    twice its actual maximum.
    """
    condor = build_condor(_condor_chain(), PremiumConfig(short_delta=0.16, wing_points=2.0))
    assert condor is not None
    assert condor.width == 2.0
    assert condor.credit == pytest.approx(0.80, abs=1e-6)
    assert condor.max_loss == pytest.approx(1.20, abs=1e-6)


def test_structure_value_is_clamped_to_its_own_bound():
    """The 2026-07-29 14:00:09 snapshot, verbatim.

    Summing the four legs off this torn snapshot gives 1.435 on a one-wide
    structure -- the mark that turned $60 of defined risk into a $942 loss.
    The clamp is what stops the position manager acting on it.
    """
    gate = ChainGate(PremiumConfig(reject_vertical_breach=False, reject_below_intrinsic=False,
                                   max_relative_spread=10.0))
    chain = gate.clean(
        738.15,
        [
            row("put", 735.0, 1.58, 1.88, -0.30),
            row("put", 734.0, 0.95, 1.50, -0.22),
            row("call", 735.0, 4.47, 4.96, 0.85),
            row("call", 736.0, 3.56, 4.01, 0.78),
        ],
    )
    fly = PremiumStructure(
        kind="iron_fly",
        legs=(
            StructureLeg("put", "SELL", 735.0, 1.58, 1.88, -0.30),
            StructureLeg("put", "BUY", 734.0, 0.95, 1.50, -0.22),
            StructureLeg("call", "SELL", 735.0, 4.47, 4.96, 0.85),
            StructureLeg("call", "BUY", 736.0, 3.56, 4.01, 0.78),
        ),
        credit=0.94,
        width=1.0,
        short_strikes=(735.0,),
    )
    raw = sum(
        (chain.mid(leg.side, leg.strike) or 0) * (1 if leg.action == "SELL" else -1)
        for leg in fly.legs
    )
    assert raw == pytest.approx(1.435, abs=1e-6), "the torn snapshot really does sum past the bound"
    assert fly.value_at(chain) == 1.0, "value must be clamped to the structure's own width"
    # Booked unclamped on ten lots this is a $942.50 loss against $60 of risk.
    assert (fly.credit - raw) * 100 * 10 == pytest.approx(-495.0, abs=1e-6)
    assert (fly.credit - fly.value_at(chain)) * 100 * 10 == pytest.approx(-60.0, abs=1e-6)


def test_credit_spread_picks_the_target_delta():
    chain = _condor_chain()
    spread = build_credit_spread(chain, "put", PremiumConfig(short_delta=0.16, wing_points=2.0))
    assert spread is not None
    assert spread.short_strikes == (748.0,)
    assert spread.credit == pytest.approx(0.40, abs=1e-6)


# --------------------------------------------------------------------------
# sizing
# --------------------------------------------------------------------------
def test_size_is_taken_off_max_loss_not_off_the_credit():
    """Sizing off the credit is sizing off the best case."""
    condor = build_condor(_condor_chain(), PremiumConfig(short_delta=0.16, wing_points=2.0))
    assert condor is not None
    cfg = PremiumConfig(risk_fraction=0.02)
    # max_loss 1.20/share -> $120 per contract; 2% of 10_000 is $200 -> 1 contract.
    assert size_position(10_000.0, condor, cfg) == 1
    assert size_position(100_000.0, condor, cfg) == 16
    assert size_position(0.0, condor, cfg) == 0


def test_size_respects_the_contract_ceiling():
    condor = build_condor(_condor_chain(), PremiumConfig(short_delta=0.16, wing_points=2.0))
    assert condor is not None
    assert size_position(10_000_000.0, condor, PremiumConfig(max_contracts=50)) == 50


# --------------------------------------------------------------------------
# the stand-aside gate
# --------------------------------------------------------------------------
def test_stands_aside_once_the_day_has_spent_its_range():
    """2026-08-03 opened and ran 210% of its implied range before 10:00.

    That was one of the two sessions that broke the naive backtest.
    """
    verdict = assess_day(
        {"expected_range": 1.86, "has_catalyst": False},
        session_open=740.0,
        spot=744.0,
        config=PremiumConfig(max_realised_fraction=0.60),
    )
    assert verdict.trade is False
    assert any("implied range" in r for r in verdict.reasons)


def test_stands_aside_on_a_catalyst():
    verdict = assess_day(
        {"expected_range": 3.0, "has_catalyst": True, "catalyst_label": "FOMC"},
        session_open=740.0, spot=740.2,
    )
    assert verdict.trade is False
    assert "FOMC" in verdict.reasons[0]


def test_trades_a_quiet_well_paid_day():
    verdict = assess_day(
        {"expected_range": 3.0, "has_catalyst": False, "net_gex": 1.5},
        session_open=740.0, spot=740.3,
    )
    assert verdict.trade is True
    assert verdict.reasons == ()


def test_gamma_filter_is_off_by_default_and_opt_in():
    market = {"expected_range": 3.0, "has_catalyst": False, "net_gex": 2.0}
    assert assess_day(market, session_open=740.0, spot=740.1).trade is True
    filtered = assess_day(market, session_open=740.0, spot=740.1,
                          config=PremiumConfig(gamma_filter=True))
    assert filtered.trade is False


# --------------------------------------------------------------------------
# management
# --------------------------------------------------------------------------
def test_target_fires_when_the_credit_has_decayed():
    chain = _condor_chain()
    condor = build_condor(chain, PremiumConfig(short_delta=0.16, wing_points=2.0))
    assert condor is not None
    cheap = ChainGate().clean(
        750.0,
        [row("put", 748.0, 0.03, 0.05, -0.02), row("put", 746.0, 0.01, 0.02, -0.01),
         row("call", 752.0, 0.03, 0.05, 0.02), row("call", 754.0, 0.01, 0.02, 0.01)],
    )
    reason, value = manage_position(condor, cheap, "13:00")
    assert reason == "target"
    assert value is not None and value < condor.credit


def test_no_mark_stop_by_default():
    """The long wing is the stop. A second stop inside it just realises noise."""
    chain = _condor_chain()
    condor = build_condor(chain, PremiumConfig(short_delta=0.16, wing_points=2.0))
    assert condor is not None
    expensive = ChainGate().clean(
        750.0,
        [row("put", 748.0, 1.60, 1.64, -0.40), row("put", 746.0, 0.20, 0.24, -0.08),
         row("call", 752.0, 0.05, 0.07, 0.03), row("call", 754.0, 0.01, 0.02, 0.01)],
    )
    assert manage_position(condor, expensive, "13:00")[0] is None
    stopped = manage_position(condor, expensive, "13:00", PremiumConfig(stop_multiple=1.5))
    assert stopped[0] == "stop"


def test_breach_exit_uses_spot_not_marks():
    """Spot is always quotable; option marks are not. The tail defence must not
    depend on the thing that breaks during a tail."""
    chain = _condor_chain()
    condor = build_condor(chain, PremiumConfig(short_delta=0.16, wing_points=2.0))
    assert condor is not None
    gate = ChainGate()
    through = gate.clean(
        755.0,
        [row("put", 748.0, 0.01, 0.02, -0.01), row("put", 746.0, 0.01, 0.02, -0.01),
         row("call", 752.0, 3.00, 3.05, 0.95), row("call", 754.0, 1.05, 1.10, 0.80)],
    )
    cfg = PremiumConfig(breach_exit_points=0.0)
    assert manage_position(condor, through, "13:00", cfg)[0] == "breach"
    assert manage_position(condor, through, "13:00")[0] is None


def test_flatten_at_the_close():
    chain = _condor_chain()
    condor = build_condor(chain, PremiumConfig(short_delta=0.16, wing_points=2.0))
    assert condor is not None
    reason, _ = manage_position(condor, chain, "15:45")
    assert reason == "flat_eod"


def test_unusable_snapshot_is_not_an_exit_signal():
    """Refusing to act on a mark you cannot produce is the whole point."""
    chain = _condor_chain()
    condor = build_condor(chain, PremiumConfig(short_delta=0.16, wing_points=2.0))
    assert condor is not None
    empty = ChainGate().clean(750.0, [])
    assert manage_position(condor, empty, "13:00") == (None, None)


# --------------------------------------------------------------------------
# the session clock
# --------------------------------------------------------------------------
def test_decay_remaining_runs_from_entry_window_to_flatten():
    cfg = PremiumConfig(entry_after="10:00", flatten_at="15:30")
    assert decay_remaining("10:00", cfg) == pytest.approx(1.0)
    assert decay_remaining("15:30", cfg) == pytest.approx(0.0)
    assert decay_remaining("12:45", cfg) == pytest.approx(0.5)
    # decay after the flatten time is decay this book never collects
    assert decay_remaining("15:59", cfg) == pytest.approx(0.0)


def test_late_entries_must_pay_more_for_the_same_tail():
    cfg = PremiumConfig(min_credit_ratio=0.12)
    early = required_credit_ratio("10:00", cfg)
    late = required_credit_ratio("14:00", cfg)
    assert early == pytest.approx(0.12)
    assert late > early * 3
    # floored so it cannot go to infinity at the flatten time
    assert required_credit_ratio("15:30", cfg) == pytest.approx(0.60)


def test_expected_remaining_move_shrinks_with_the_square_root_of_time():
    cfg = PremiumConfig(entry_after="10:00", flatten_at="15:30")
    market = {"expected_range": 4.0}
    assert expected_remaining_move(market, "10:00", cfg) == pytest.approx(4.0)
    assert expected_remaining_move(market, "12:45", cfg) == pytest.approx(4.0 * 0.5**0.5)
    assert expected_remaining_move({}, "10:00", cfg) is None


# --------------------------------------------------------------------------
# strike pressure
# --------------------------------------------------------------------------
def test_strike_pressure_measures_buffer_in_remaining_moves():
    condor = build_condor(_condor_chain(), PremiumConfig(short_delta=0.16, wing_points=2.0))
    assert condor is not None
    cfg = PremiumConfig(entry_after="10:00", flatten_at="15:30")
    market = {"expected_range": 2.0}
    # spot 750, shorts at 748/752 -> 2.0 points of buffer against a 2.0 move
    assert strike_pressure(condor, 750.0, market, "10:00", cfg) == pytest.approx(1.0)
    # same spot later in the day: less time, less expected travel, safer
    assert strike_pressure(condor, 750.0, market, "12:45", cfg) > 1.0
    # spot drifting toward the short put eats the buffer
    assert strike_pressure(condor, 749.0, market, "10:00", cfg) == pytest.approx(0.5)


def test_a_position_cannot_violate_its_own_entry_gate_at_inception():
    """An absolute pressure floor near the median entry value is an instruction
    to close immediately. The gate belongs on entry, the exit on the buffer."""
    condor = build_condor(_condor_chain(), PremiumConfig(short_delta=0.16, wing_points=2.0))
    assert condor is not None
    cfg = PremiumConfig(entry_after="10:00", flatten_at="15:30")
    entry = strike_pressure(condor, 750.0, {"expected_range": 2.0}, "10:00", cfg)
    assert entry is not None and entry >= cfg.min_entry_pressure


# --------------------------------------------------------------------------
# sizing
# --------------------------------------------------------------------------
def test_size_scales_down_as_decay_runs_out():
    condor = build_condor(_condor_chain(), PremiumConfig(short_delta=0.16, wing_points=2.0))
    assert condor is not None
    cfg = PremiumConfig(risk_fraction=0.25, late_entry_size_scale=True, max_contracts=10_000,
                        entry_after="10:00", flatten_at="15:30", min_survivable_losses=0)
    early = size_position(100_000.0, condor, cfg, now_hhmm="10:00")
    late = size_position(100_000.0, condor, cfg, now_hhmm="12:45")
    assert early > 0
    assert late == pytest.approx(early / 2, rel=0.02)


def test_size_never_exceeds_what_the_account_can_survive():
    """Fixed-fractional sizing does not give survivability for free once the
    one-lot floor binds."""
    condor = build_condor(_condor_chain(), PremiumConfig(short_delta=0.16, wing_points=2.0))
    assert condor is not None
    greedy = PremiumConfig(risk_fraction=0.90, min_survivable_losses=0, late_entry_size_scale=False)
    safe = PremiumConfig(risk_fraction=0.90, min_survivable_losses=4, late_entry_size_scale=False)
    # max_loss ~1.20/share -> ~$120/contract on a $1,200 account
    unconstrained = size_position(1_200.0, condor, greedy)
    survivable = size_position(1_200.0, condor, safe)
    assert unconstrained >= 8, "90% of equity buys most of the account"
    assert survivable == 2, "four max losses must still leave a tradeable account"
    assert survivable * 4 * condor.max_loss * 100 <= 1_200.0


# --------------------------------------------------------------------------
# monitoring
# --------------------------------------------------------------------------
def _state(structure, entry_pressure=2.0):
    return PositionState(structure=structure, contracts=1, entry_hhmm="10:00",
                         entry_pressure=entry_pressure)


def test_monitor_records_unusable_marks_without_exiting_on_them():
    condor = build_condor(_condor_chain(), PremiumConfig(short_delta=0.16, wing_points=2.0))
    assert condor is not None
    state = _state(condor)
    result = monitor(state, ChainGate().clean(750.0, []), {"expected_range": 2.0}, "13:00")
    assert result.hold
    assert result.value is None
    assert state.marks_unusable == 1 and state.consecutive_unusable == 1


def test_monitor_flattens_on_the_clock_even_without_a_usable_mark():
    """The clock must not be disableable by a bad chain."""
    condor = build_condor(_condor_chain(), PremiumConfig(short_delta=0.16, wing_points=2.0))
    assert condor is not None
    result = monitor(_state(condor), ChainGate().clean(750.0, []), {}, "15:30")
    assert result.exit_reason == "flat_eod"


def test_risk_exit_is_off_by_default_and_relative_when_enabled():
    """Cutting a defined-risk short structure early was destructive at every
    threshold measured; the wing is already the risk control."""
    condor = build_condor(_condor_chain(), PremiumConfig(short_delta=0.16, wing_points=2.0))
    assert condor is not None
    chain = _condor_chain()
    market = {"expected_range": 2.0}
    # buffer has halved from entry
    assert monitor(_state(condor, entry_pressure=2.0), chain, market, "10:00").hold
    enabled = PremiumConfig(pressure_exit_ratio=0.75)
    fired = monitor(_state(condor, entry_pressure=2.0), chain, market, "10:00", enabled)
    assert fired.exit_reason == "risk"
    assert "buffer down to" in fired.note
