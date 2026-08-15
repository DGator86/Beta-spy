from datetime import UTC, datetime, timedelta

from beta_spy.ledger import PaperLedger
from beta_spy.options import OptionLeg, OptionPlan
from beta_spy.storage import Tape500Store


NOW = datetime(2026, 8, 17, 15, 0, tzinfo=UTC)


def _leg(symbol: str, side: str, right: str, strike: float, bid: float, ask: float) -> OptionLeg:
    return OptionLeg(symbol=symbol, side=side, right=right, strike=strike, bid=bid, ask=ask, delta=None)


def _debit_plan() -> OptionPlan:
    return OptionPlan(
        strategy="CALL_DEBIT_SPREAD",
        direction="BULLISH",
        expiration="2026-08-17",
        debit=1.2,
        width=5.0,
        max_loss_dollars=120.0,
        max_profit_dollars=380.0,
        score=1.0,
        legs=(
            _leg("SPY260817C00600000", "BUY", "C", 600.0, 2.0, 2.1),
            _leg("SPY260817C00605000", "SELL", "C", 605.0, 0.9, 1.0),
        ),
        contracts=1,
        hold_minutes=15.0,
    )


def _condor_plan() -> OptionPlan:
    return OptionPlan(
        strategy="IRON_CONDOR",
        direction="NEUTRAL",
        expiration="2026-08-17",
        debit=0.5,
        width=5.0,
        max_loss_dollars=450.0,
        max_profit_dollars=50.0,
        score=0.1,
        legs=(
            _leg("SPY260817P00590000", "SELL", "P", 590.0, 0.5, 0.55),
            _leg("SPY260817P00585000", "BUY", "P", 585.0, 0.2, 0.25),
            _leg("SPY260817C00610000", "SELL", "C", 610.0, 0.5, 0.55),
            _leg("SPY260817C00615000", "BUY", "C", 615.0, 0.2, 0.25),
        ),
        contracts=1,
        hold_minutes=15.0,
    )


def _ledger(tmp_path, **kwargs) -> PaperLedger:
    store = Tape500Store(tmp_path / "ledger.sqlite")
    return PaperLedger(store, **kwargs)


def test_debit_spread_takes_profit_and_reports(tmp_path):
    ledger = _ledger(tmp_path)
    assert ledger.open_position(_debit_plan(), NOW) is not None
    # Long leg sells at 1.90, short leg buys back at 0.10: value 1.80 vs 1.20 in.
    quotes = {
        "SPY260817C00600000": (1.9, 2.0),
        "SPY260817C00605000": (0.05, 0.10),
    }
    closed = ledger.mark_positions(quotes, NOW + timedelta(minutes=3))
    assert len(closed) == 1
    assert closed[0]["exit_reason"] == "TAKE_PROFIT"
    assert closed[0]["realized_pnl_dollars"] == 60.0
    stats = ledger.stats(NOW + timedelta(minutes=4))
    assert stats["closed_count"] == 1
    assert stats["wins"] == 1
    assert stats["day_realized_pnl_dollars"] == 60.0


def test_debit_spread_stops_out(tmp_path):
    ledger = _ledger(tmp_path)
    ledger.open_position(_debit_plan(), NOW)
    quotes = {
        "SPY260817C00600000": (0.6, 0.7),
        "SPY260817C00605000": (0.05, 0.10),
    }
    closed = ledger.mark_positions(quotes, NOW + timedelta(minutes=3))
    assert closed[0]["exit_reason"] == "STOP_LOSS"
    assert closed[0]["realized_pnl_dollars"] == -70.0


def test_directional_position_closes_at_horizon(tmp_path):
    ledger = _ledger(tmp_path)
    ledger.open_position(_debit_plan(), NOW)
    # Mark near entry: no profit target or stop is touched.
    quotes = {
        "SPY260817C00600000": (2.0, 2.1),
        "SPY260817C00605000": (0.9, 1.0),
    }
    assert ledger.mark_positions(quotes, NOW + timedelta(minutes=5)) == []
    closed = ledger.mark_positions(quotes, NOW + timedelta(minutes=16))
    assert closed[0]["exit_reason"] == "HORIZON"


def test_condor_takes_profit_on_premium_decay(tmp_path):
    ledger = _ledger(tmp_path)
    ledger.open_position(_condor_plan(), NOW)
    # Shorts buy back for 0.10 each, longs go out worthless: keeps 0.30 of 0.50.
    quotes = {
        "SPY260817P00590000": (0.05, 0.10),
        "SPY260817P00585000": (0.0, 0.05),
        "SPY260817C00610000": (0.05, 0.10),
        "SPY260817C00615000": (0.0, 0.05),
    }
    closed = ledger.mark_positions(quotes, NOW + timedelta(minutes=30))
    assert closed[0]["exit_reason"] == "TAKE_PROFIT"
    assert closed[0]["realized_pnl_dollars"] == 30.0


def test_condor_stops_at_twice_credit(tmp_path):
    ledger = _ledger(tmp_path)
    ledger.open_position(_condor_plan(), NOW)
    quotes = {
        "SPY260817P00590000": (0.8, 0.85),
        "SPY260817P00585000": (0.05, 0.10),
        "SPY260817C00610000": (0.8, 0.85),
        "SPY260817C00615000": (0.05, 0.10),
    }
    closed = ledger.mark_positions(quotes, NOW + timedelta(minutes=30))
    assert closed[0]["exit_reason"] == "STOP_LOSS"
    assert closed[0]["realized_pnl_dollars"] == -110.0


def test_condor_force_closed_before_expiry(tmp_path):
    ledger = _ledger(tmp_path)
    ledger.open_position(_condor_plan(), NOW)
    quotes = {
        "SPY260817P00590000": (0.4, 0.45),
        "SPY260817P00585000": (0.15, 0.20),
        "SPY260817C00610000": (0.4, 0.45),
        "SPY260817C00615000": (0.15, 0.20),
    }
    near_expiry = datetime(2026, 8, 17, 19, 51, tzinfo=UTC)
    closed = ledger.mark_positions(quotes, near_expiry)
    assert closed[0]["exit_reason"] == "EXPIRY_CLOSE"


def test_duplicate_strategy_and_book_limit_refused(tmp_path):
    ledger = _ledger(tmp_path, max_open_positions=2)
    assert ledger.open_position(_debit_plan(), NOW) is not None
    assert ledger.open_position(_debit_plan(), NOW) is None
    assert ledger.open_position(_condor_plan(), NOW) is not None
    # Book is full: a third distinct strategy is refused too.
    put_plan = OptionPlan(
        strategy="PUT_CREDIT_SPREAD",
        direction="BULLISH",
        expiration="2026-08-17",
        debit=0.4,
        width=5.0,
        max_loss_dollars=460.0,
        max_profit_dollars=40.0,
        score=0.1,
        legs=(
            _leg("SPY260817P00595000", "SELL", "P", 595.0, 0.5, 0.55),
            _leg("SPY260817P00590000", "BUY", "P", 590.0, 0.1, 0.15),
        ),
        contracts=1,
        hold_minutes=15.0,
    )
    assert ledger.open_position(put_plan, NOW) is None


def test_daily_loss_breaker_blocks_new_positions(tmp_path):
    ledger = _ledger(tmp_path, daily_loss_limit_dollars=50.0)
    ledger.open_position(_debit_plan(), NOW)
    quotes = {
        "SPY260817C00600000": (0.6, 0.7),
        "SPY260817C00605000": (0.05, 0.10),
    }
    closed = ledger.mark_positions(quotes, NOW + timedelta(minutes=3))
    assert closed[0]["realized_pnl_dollars"] == -70.0
    assert ledger.breaker_tripped(NOW + timedelta(minutes=4))
    assert ledger.open_position(_condor_plan(), NOW + timedelta(minutes=5)) is None
    # A new day resets the breaker.
    next_day = NOW + timedelta(days=1)
    assert not ledger.breaker_tripped(next_day)
    assert ledger.open_position(_condor_plan(), next_day) is not None
