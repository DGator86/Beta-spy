import json
from datetime import UTC, datetime, timedelta
from types import SimpleNamespace

import beta_spy.live as live_module
from beta_spy.ledger import PaperLedger
from beta_spy.live import StateHub, TradierMarketStream
from beta_spy.options import OptionLeg, OptionPlan
from beta_spy.storage import Tape500Store


NOW = datetime(2026, 8, 17, 15, 0, tzinfo=UTC)

# Quotes that cross through the debit spread's resting mid (limit 1.10).
DEBIT_FILL_QUOTES = {
    "SPY260817C00600000": (1.95, 2.00),
    "SPY260817C00605000": (0.90, 0.95),
}
CONDOR_ENTRY_QUOTES = {
    "SPY260817P00590000": (0.5, 0.55),
    "SPY260817P00585000": (0.2, 0.25),
    "SPY260817C00610000": (0.5, 0.55),
    "SPY260817C00615000": (0.2, 0.25),
}


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


def _fill_debit(ledger: PaperLedger) -> None:
    assert ledger.open_position(_debit_plan(), NOW) is not None
    # Market crosses through the mid: the resting limit fills at 1.10.
    assert ledger.mark_positions(DEBIT_FILL_QUOTES, NOW + timedelta(seconds=20)) == []
    working = ledger._rows(("OPEN",))
    assert len(working) == 1
    assert working[0]["entry_price"] == 1.1


def _fill_condor(ledger: PaperLedger, opened: datetime = NOW) -> None:
    assert ledger.open_position(_condor_plan(), opened) is not None
    # Entry quotes never reach the 0.60 mid; the patience timeout pays up
    # and crosses at the 0.50 credit.
    ledger.mark_positions(CONDOR_ENTRY_QUOTES, opened + timedelta(minutes=2))
    assert ledger._rows(("OPEN",))[0]["entry_price"] == 0.5


def test_pending_entry_fills_at_mid_improving_on_cross(tmp_path):
    ledger = _ledger(tmp_path)
    _fill_debit(ledger)
    row = ledger._rows(("OPEN",))[0]
    # Mid fill at 1.10 beats the 1.20 cross the plan assumed.
    assert row["entry_price"] == 1.1
    assert row["max_loss_dollars"] == 110.0
    assert row["max_profit_dollars"] == 390.0


def test_pending_entry_times_out_and_pays_up(tmp_path):
    ledger = _ledger(tmp_path, patience_seconds=90.0)
    ledger.open_position(_condor_plan(), NOW)
    # Before the timeout nothing fills (0.50 cross < 0.60 mid limit).
    ledger.mark_positions(CONDOR_ENTRY_QUOTES, NOW + timedelta(seconds=30))
    assert ledger._rows(("PENDING",)) and not ledger._rows(("OPEN",))
    ledger.mark_positions(CONDOR_ENTRY_QUOTES, NOW + timedelta(seconds=120))
    row = ledger._rows(("OPEN",))[0]
    assert row["entry_price"] == 0.5


def test_debit_spread_keeps_running_winners(tmp_path):
    ledger = _ledger(tmp_path)
    _fill_debit(ledger)
    # +$55 is a working debit, not 78% of max profit — do not chop it at 15 minutes.
    quotes = {
        "SPY260817C00600000": (1.75, 1.85),
        "SPY260817C00605000": (0.05, 0.10),
    }
    assert ledger.mark_positions(quotes, NOW + timedelta(minutes=3)) == []
    assert ledger.mark_positions(quotes, NOW + timedelta(minutes=16)) == []
    row = ledger._rows(("OPEN",))[0]
    assert row["unrealized_pnl_dollars"] == 55.0


def test_debit_spread_trails_giveback(tmp_path):
    ledger = _ledger(tmp_path)
    _fill_debit(ledger)
    runup = {
        "SPY260817C00600000": (2.40, 2.50),
        "SPY260817C00605000": (0.05, 0.10),
    }
    assert ledger.mark_positions(runup, NOW + timedelta(minutes=5)) == []
    giveback = {
        "SPY260817C00600000": (1.55, 1.65),
        "SPY260817C00605000": (0.05, 0.10),
    }
    closed = ledger.mark_positions(giveback, NOW + timedelta(minutes=8))
    assert closed[0]["exit_reason"] == "PROFIT_TRAIL"


def test_debit_spread_stops_out(tmp_path):
    ledger = _ledger(tmp_path)
    _fill_debit(ledger)
    quotes = {
        "SPY260817C00600000": (0.6, 0.7),
        "SPY260817C00605000": (0.05, 0.10),
    }
    closed = ledger.mark_positions(quotes, NOW + timedelta(minutes=3))
    assert closed[0]["exit_reason"] == "STOP_LOSS"
    assert closed[0]["realized_pnl_dollars"] == -60.0


def test_directional_position_does_not_die_on_the_horizon_clock(tmp_path):
    ledger = _ledger(tmp_path)
    _fill_debit(ledger)
    quotes = {
        "SPY260817C00600000": (2.0, 2.1),
        "SPY260817C00605000": (0.9, 1.0),
    }
    assert ledger.mark_positions(quotes, NOW + timedelta(minutes=5)) == []
    assert ledger.mark_positions(quotes, NOW + timedelta(minutes=16)) == []
    closed = ledger.mark_positions(quotes, datetime(2026, 8, 17, 19, 55, tzinfo=UTC))
    assert closed[0]["exit_reason"] == "FORCED_FLAT"


def test_overnight_entries_are_refused(tmp_path):
    ledger = _ledger(tmp_path)
    overnight = datetime(2026, 8, 18, 2, 0, tzinfo=UTC)
    assert ledger.open_position(_debit_plan(), overnight) is None


def test_condor_takes_profit_on_premium_decay(tmp_path):
    ledger = _ledger(tmp_path)
    _fill_condor(ledger)
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
    _fill_condor(ledger)
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
    _fill_condor(ledger)
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
    _fill_debit(ledger)
    quotes = {
        "SPY260817C00600000": (0.6, 0.7),
        "SPY260817C00605000": (0.05, 0.10),
    }
    closed = ledger.mark_positions(quotes, NOW + timedelta(minutes=3))
    assert closed[0]["realized_pnl_dollars"] == -60.0
    assert ledger.breaker_tripped(NOW + timedelta(minutes=4))
    assert ledger.open_position(_condor_plan(), NOW + timedelta(minutes=5)) is None
    # A new day resets the breaker.
    next_day = NOW + timedelta(days=1)
    assert not ledger.breaker_tripped(next_day)
    assert ledger.open_position(_condor_plan(), next_day) is not None


def test_risk_budget_compounds_and_throttles(tmp_path):
    ledger = _ledger(tmp_path, starting_equity=1000.0, daily_loss_limit_dollars=10_000.0)
    # Fresh $1,000 account: 15% per trade, 25% single-trade ceiling.
    assert ledger.risk_budget_dollars(NOW) == 150.0
    assert ledger.max_trade_risk_dollars() == 250.0
    with ledger.store.lock:
        # A banked win raises equity and therefore size.
        ledger.store.connection.execute(
            """
            INSERT INTO paper_positions(
                opened_at,strategy,direction,expiration,contracts,entry_price,is_credit,
                max_loss_dollars,max_profit_dollars,hold_minutes,legs,status,closed_at,
                realized_pnl_dollars
            ) VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?)
            """,
            (
                "2026-08-16T15:00:00Z", "CALL_DEBIT_SPREAD", "BULLISH", "2026-08-16",
                1, 1.0, 0, 100.0, 400.0, 15.0, "[]", "CLOSED", "2026-08-16T15:20:00Z", 500.0,
            ),
        )
        ledger.store.connection.commit()
    # Equity is now $1,500: the dollar budget compounds proportionally.
    assert ledger.equity_dollars() == 1500.0
    assert ledger.risk_budget_dollars(NOW) == 225.0
    assert ledger.max_trade_risk_dollars() == 375.0
    # Three consecutive losses today halve the budget.
    with ledger.store.lock:
        for index in range(3):
            ledger.store.connection.execute(
                """
                INSERT INTO paper_positions(
                    opened_at,strategy,direction,expiration,contracts,entry_price,is_credit,
                    max_loss_dollars,max_profit_dollars,hold_minutes,legs,status,closed_at,
                    realized_pnl_dollars
                ) VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?)
                """,
                (
                    "2026-08-17T14:00:00Z", "CALL_DEBIT_SPREAD", "BULLISH", "2026-08-17",
                    1, 1.0, 0, 100.0, 400.0, 15.0, "[]", "CLOSED",
                    f"2026-08-17T14:{10 + index:02d}:00Z", -10.0,
                ),
            )
        ledger.store.connection.commit()
    assert ledger.consecutive_losses_today(NOW) == 3
    equity = 1000.0 + 500.0 - 30.0
    assert ledger.risk_budget_dollars(NOW) == equity * 0.15 * 0.5


def test_fractional_daily_breaker_scales_with_equity(tmp_path):
    # No absolute limit: the breaker is 25% of the day's starting equity.
    ledger = _ledger(tmp_path, starting_equity=1000.0, daily_loss_fraction=0.25)
    assert ledger.daily_loss_limit_now(NOW) == 250.0
    assert not ledger.breaker_tripped(NOW)
    with ledger.store.lock:
        ledger.store.connection.execute(
            """
            INSERT INTO paper_positions(
                opened_at,strategy,direction,expiration,contracts,entry_price,is_credit,
                max_loss_dollars,max_profit_dollars,hold_minutes,legs,status,closed_at,
                realized_pnl_dollars
            ) VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?)
            """,
            (
                "2026-08-17T14:00:00Z", "CALL_DEBIT_SPREAD", "BULLISH", "2026-08-17",
                1, 1.0, 0, 100.0, 400.0, 15.0, "[]", "CLOSED",
                "2026-08-17T14:10:00Z", -250.0,
            ),
        )
        ledger.store.connection.commit()
    # Day-start equity is still $1,000 (the loss happened today), so the
    # $250 realized loss trips the 25% breaker exactly.
    assert ledger.day_start_equity_dollars(NOW) == 1000.0
    assert ledger.breaker_tripped(NOW)
    assert ledger.open_position(_condor_plan(), NOW) is None


def test_alpha_signal_recorded_and_published(tmp_path, monkeypatch):
    store = Tape500Store(tmp_path / "alpha.sqlite")
    hub = StateHub()
    stream = TradierMarketStream(
        "token",
        SimpleNamespace(store=store, holdings={}),
        hub,
        alpha_state_url="http://127.0.0.1:8787/api/v1/state",
    )
    payload = {
        "market": {"price": 776.34},
        "decision": {"action": "NO_TRADE", "created_at": "2026-08-15T15:53:59Z"},
        "forecast_horizons": {
            "15m": {"probability_up": 0.49, "expected_return": 0.0009, "created_at": "x"}
        },
    }
    monkeypatch.setattr(
        live_module.httpx,
        "get",
        lambda url, timeout: SimpleNamespace(raise_for_status=lambda: None, json=lambda: payload),
    )
    record = stream._record_alpha_signal(NOW)
    assert record is not None
    assert record["action"] == "NO_TRADE"
    assert record["horizons"]["15m"]["probability_up"] == 0.49
    row = store.connection.execute("SELECT payload FROM alpha_signals").fetchone()
    assert json.loads(row[0])["spy_price"] == 776.34
    stream.close()
