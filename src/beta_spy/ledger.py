"""Paper-trading ledger: every emitted option plan becomes a tracked position.

The ledger turns Beta-spy from a signal generator into a system with a
realized track record. Position lifecycle:

PENDING -> OPEN -> CLOSED (or PENDING -> CANCELLED)

Entries are patient: a new position rests as a limit order at the structure
mid. On each quote pass it fills at the mid if the market crosses through it
(sampled, so conservative); after ``patience_seconds`` it pays up and crosses
the spread like the original pessimistic model. Working the mid recovers a
half-spread per leg, which at this scale is comparable to the entire
per-trade edge.

Exits are managed:
- Credit structures (credit spreads, iron condors) take profit when half of
  the collected premium has decayed, stop out when the loss reaches twice the
  credit, and are force-closed shortly before expiry.
- Debit spreads take profit at +50% of the debit, stop out at -50%, and are
  otherwise closed at their forecast horizon (the EV model prices the plan
  over that hold, so letting it run past the horizon is unmodelled risk).

Sizing feedback:
- A daily circuit breaker refuses new positions once the day's realized loss
  exceeds the configured limit.
- ``size_multiplier`` compounds the risk budget with the bankroll (realized
  P&L relative to starting equity, clipped) and halves it after three
  consecutive losing trades in a day.
"""

from __future__ import annotations

import json
import math
import sqlite3
from datetime import UTC, datetime, timedelta
from typing import Any, Iterable, Mapping

from .options import OptionPlan
from .storage import Tape500Store

CREDIT_STRATEGIES = {"PUT_CREDIT_SPREAD", "CALL_CREDIT_SPREAD", "IRON_CONDOR"}

LEDGER_SCHEMA = """
CREATE TABLE IF NOT EXISTS paper_positions (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    opened_at TEXT NOT NULL,
    strategy TEXT NOT NULL,
    direction TEXT NOT NULL,
    expiration TEXT NOT NULL,
    contracts INTEGER NOT NULL,
    entry_price REAL NOT NULL,
    is_credit INTEGER NOT NULL,
    max_loss_dollars REAL NOT NULL,
    max_profit_dollars REAL NOT NULL,
    hold_minutes REAL NOT NULL,
    legs TEXT NOT NULL,
    status TEXT NOT NULL DEFAULT 'OPEN',
    closed_at TEXT,
    exit_reason TEXT,
    exit_value REAL,
    realized_pnl_dollars REAL,
    mark_value REAL,
    unrealized_pnl_dollars REAL,
    marked_at TEXT
);
CREATE INDEX IF NOT EXISTS paper_positions_status ON paper_positions(status);
CREATE INDEX IF NOT EXISTS paper_positions_closed_at ON paper_positions(closed_at);
"""

# Columns added after the first release; applied idempotently at startup.
LEDGER_MIGRATIONS = (
    "ALTER TABLE paper_positions ADD COLUMN limit_cash REAL",
    "ALTER TABLE paper_positions ADD COLUMN width REAL",
    "ALTER TABLE paper_positions ADD COLUMN entry_style TEXT",
)


def _iso(value: datetime) -> str:
    return value.isoformat().replace("+00:00", "Z")


def _parse(value: str) -> datetime:
    return datetime.fromisoformat(str(value).replace("Z", "+00:00"))


class PaperLedger:
    """Persistent paper book over the engine's option plans."""

    def __init__(
        self,
        store: Tape500Store,
        *,
        daily_loss_limit_dollars: float = 300.0,
        max_open_positions: int = 3,
        credit_take_profit_fraction: float = 0.5,
        credit_stop_loss_multiple: float = 2.0,
        debit_take_profit_fraction: float = 0.5,
        debit_stop_loss_fraction: float = 0.5,
        expiry_buffer_minutes: float = 10.0,
        patience_seconds: float = 90.0,
        starting_equity: float | None = 10_000.0,
        bankroll_multiplier_cap: float = 3.0,
        loss_streak_trigger: int = 3,
    ) -> None:
        self.store = store
        self.daily_loss_limit_dollars = float(daily_loss_limit_dollars)
        self.max_open_positions = int(max_open_positions)
        self.credit_take_profit_fraction = float(credit_take_profit_fraction)
        self.credit_stop_loss_multiple = float(credit_stop_loss_multiple)
        self.debit_take_profit_fraction = float(debit_take_profit_fraction)
        self.debit_stop_loss_fraction = float(debit_stop_loss_fraction)
        self.expiry_buffer_minutes = float(expiry_buffer_minutes)
        self.patience_seconds = float(patience_seconds)
        self.starting_equity = float(starting_equity) if starting_equity else None
        self.bankroll_multiplier_cap = float(bankroll_multiplier_cap)
        self.loss_streak_trigger = int(loss_streak_trigger)
        with self.store.lock:
            self.store.connection.executescript(LEDGER_SCHEMA)
            for statement in LEDGER_MIGRATIONS:
                try:
                    self.store.connection.execute(statement)
                except sqlite3.OperationalError:
                    pass  # column already exists
            self.store.connection.commit()

    # -- opening ---------------------------------------------------------

    def open_position(self, plan: OptionPlan, timestamp: datetime) -> int | None:
        """Rest the plan as a PENDING mid-price entry; returns the row id.

        Refuses when the strategy is already working or open (the planner
        re-emits the same idea while a signal persists), when the book is
        full, or when the daily circuit breaker has tripped.
        """
        if self.breaker_tripped(timestamp):
            return None
        working = self._rows(("OPEN", "PENDING"))
        if len(working) >= self.max_open_positions:
            return None
        if any(row["strategy"] == plan.strategy for row in working):
            return None
        legs = [
            {
                "symbol": leg.symbol,
                "side": leg.side,
                "right": leg.right,
                "strike": leg.strike,
                "entry_bid": leg.bid,
                "entry_ask": leg.ask,
            }
            for leg in plan.legs
        ]
        # Signed cash to enter at the mid of every leg: credit positive,
        # debit negative. This is the resting limit.
        limit_cash = 0.0
        for leg in plan.legs:
            mid = (leg.bid + leg.ask) / 2.0
            limit_cash += mid if leg.side.upper() == "SELL" else -mid
        is_credit = plan.strategy in CREDIT_STRATEGIES
        with self.store.lock:
            cursor = self.store.connection.execute(
                """
                INSERT INTO paper_positions(
                    opened_at,strategy,direction,expiration,contracts,entry_price,is_credit,
                    max_loss_dollars,max_profit_dollars,hold_minutes,legs,status,
                    limit_cash,width,entry_style
                ) VALUES(?,?,?,?,?,?,?,?,?,?,?,'PENDING',?,?,?)
                """,
                (
                    _iso(timestamp),
                    plan.strategy,
                    plan.direction,
                    plan.expiration,
                    plan.contracts,
                    plan.debit,
                    int(is_credit),
                    plan.max_loss_dollars * plan.contracts,
                    plan.max_profit_dollars * plan.contracts,
                    plan.hold_minutes,
                    json.dumps(legs, separators=(",", ":")),
                    round(limit_cash, 4),
                    plan.width,
                    "MID_LIMIT",
                ),
            )
            self.store.connection.commit()
        return int(cursor.lastrowid or 0)

    # -- marking, fills, and exits ----------------------------------------

    def open_symbols(self) -> list[str]:
        symbols: list[str] = []
        for row in self._rows(("OPEN", "PENDING")):
            for leg in json.loads(row["legs"]):
                if leg["symbol"] and leg["symbol"] not in symbols:
                    symbols.append(leg["symbol"])
        return symbols

    def mark_positions(
        self,
        quotes: Mapping[str, tuple[float, float]],
        now: datetime,
    ) -> list[dict[str, Any]]:
        """Fill pending entries and mark/close open positions.

        Returns the positions closed on this pass. Rows whose legs are
        missing from ``quotes`` are left untouched.
        """
        self._process_pending(quotes, now)
        closed: list[dict[str, Any]] = []
        for row in self._rows(("OPEN",)):
            legs = json.loads(row["legs"])
            liquidation = self._liquidation_value(legs, quotes)
            if liquidation is None:
                continue
            entry = float(row["entry_price"])
            contracts = int(row["contracts"])
            is_credit = bool(row["is_credit"])
            # Entry cashflow: credit collected is money in, debit paid is money out.
            entry_cashflow = entry if is_credit else -entry
            # Cents precision: exit thresholds must not miss on float dust.
            pnl = round((entry_cashflow + liquidation) * 100.0 * contracts, 2)
            reason = self._exit_reason(row, pnl, entry, contracts, is_credit, now)
            if reason is None:
                with self.store.lock:
                    self.store.connection.execute(
                        "UPDATE paper_positions SET mark_value=?, unrealized_pnl_dollars=?, marked_at=? WHERE id=?",
                        (round(liquidation, 4), round(pnl, 2), _iso(now), row["id"]),
                    )
                    self.store.connection.commit()
                continue
            with self.store.lock:
                self.store.connection.execute(
                    """
                    UPDATE paper_positions
                    SET status='CLOSED', closed_at=?, exit_reason=?, exit_value=?,
                        realized_pnl_dollars=?, mark_value=?, unrealized_pnl_dollars=0, marked_at=?
                    WHERE id=?
                    """,
                    (
                        _iso(now),
                        reason,
                        round(liquidation, 4),
                        round(pnl, 2),
                        round(liquidation, 4),
                        _iso(now),
                        row["id"],
                    ),
                )
                self.store.connection.commit()
            closed.append(
                {
                    "id": row["id"],
                    "strategy": row["strategy"],
                    "direction": row["direction"],
                    "exit_reason": reason,
                    "realized_pnl_dollars": round(pnl, 2),
                }
            )
        return closed

    def _process_pending(self, quotes: Mapping[str, tuple[float, float]], now: datetime) -> None:
        for row in self._rows(("PENDING",)):
            legs = json.loads(row["legs"])
            cash_now = self._entry_cash(legs, quotes)
            if cash_now is None:
                continue
            limit_cash = float(row["limit_cash"] if row["limit_cash"] is not None else 0.0)
            timed_out = now >= _parse(row["opened_at"]) + timedelta(seconds=self.patience_seconds)
            fill_cash: float | None = None
            if cash_now >= limit_cash:
                # Crossing now is at least as good as our resting mid: a
                # resting limit would certainly have filled.
                fill_cash = limit_cash
            elif timed_out:
                fill_cash = cash_now
            if fill_cash is None:
                continue
            width = float(row["width"] or 0.0)
            entry = abs(fill_cash)
            viable = entry > 0.0 and (width <= 0.0 or entry < width)
            if not viable:
                with self.store.lock:
                    self.store.connection.execute(
                        "UPDATE paper_positions SET status='CANCELLED', closed_at=?, exit_reason='UNFILLABLE' WHERE id=?",
                        (_iso(now), row["id"]),
                    )
                    self.store.connection.commit()
                continue
            contracts = int(row["contracts"])
            is_credit = bool(row["is_credit"])
            if width > 0.0:
                if is_credit:
                    max_profit = entry * 100.0 * contracts
                    max_loss = (width - entry) * 100.0 * contracts
                else:
                    max_loss = entry * 100.0 * contracts
                    max_profit = (width - entry) * 100.0 * contracts
            else:
                max_loss = float(row["max_loss_dollars"])
                max_profit = float(row["max_profit_dollars"])
            with self.store.lock:
                self.store.connection.execute(
                    """
                    UPDATE paper_positions
                    SET status='OPEN', opened_at=?, entry_price=?,
                        max_loss_dollars=?, max_profit_dollars=?, marked_at=?
                    WHERE id=?
                    """,
                    (
                        _iso(now),
                        round(entry, 4),
                        round(max_loss, 2),
                        round(max_profit, 2),
                        _iso(now),
                        row["id"],
                    ),
                )
                self.store.connection.commit()

    def _exit_reason(
        self,
        row: Mapping[str, Any],
        pnl: float,
        entry: float,
        contracts: int,
        is_credit: bool,
        now: datetime,
    ) -> str | None:
        # Cents precision on the thresholds as well as the P&L: 1.1 * 100 is
        # 110.00000000000001 in floats, which would push a threshold a hair
        # past an exactly-at-target P&L.
        premium = round(entry * 100.0 * contracts, 2)
        if is_credit:
            if pnl >= round(self.credit_take_profit_fraction * premium, 2):
                return "TAKE_PROFIT"
            if pnl <= -round(self.credit_stop_loss_multiple * premium, 2):
                return "STOP_LOSS"
        else:
            if pnl >= round(self.debit_take_profit_fraction * premium, 2):
                return "TAKE_PROFIT"
            if pnl <= -round(self.debit_stop_loss_fraction * premium, 2):
                return "STOP_LOSS"
        opened_at = _parse(row["opened_at"])
        if row["strategy"] == "IRON_CONDOR":
            # Condors are an expiry-EV trade: hold toward expiry, but never
            # into the settlement print.
            expiry_close = self._expiration_close(str(row["expiration"]))
            if expiry_close is not None and now >= expiry_close - timedelta(
                minutes=self.expiry_buffer_minutes
            ):
                return "EXPIRY_CLOSE"
        elif now >= opened_at + timedelta(minutes=float(row["hold_minutes"])):
            # Directional structures are priced over the forecast horizon.
            return "HORIZON"
        return None

    @staticmethod
    def _expiration_close(expiration: str) -> datetime | None:
        try:
            return datetime.strptime(expiration, "%Y-%m-%d").replace(
                hour=20, minute=0, tzinfo=UTC
            )
        except ValueError:
            return None

    @staticmethod
    def _entry_cash(
        legs: Iterable[Mapping[str, Any]],
        quotes: Mapping[str, tuple[float, float]],
    ) -> float | None:
        """Signed cash to enter now by crossing: SELL legs at bid, BUY at ask."""
        total = 0.0
        for leg in legs:
            quote = quotes.get(str(leg["symbol"]))
            if quote is None:
                return None
            bid, ask = float(quote[0]), float(quote[1])
            if bid < 0 or ask <= 0 or ask < bid or not math.isfinite(bid + ask):
                return None
            total += bid if str(leg["side"]).upper() == "SELL" else -ask
        return total

    @staticmethod
    def _liquidation_value(
        legs: Iterable[Mapping[str, Any]],
        quotes: Mapping[str, tuple[float, float]],
    ) -> float | None:
        """Signed cash received per structure to flatten every leg now.

        Long legs sell at the bid (+), short legs buy back at the ask (-);
        the result is negative for credit structures, positive for debit.
        """
        total = 0.0
        for leg in legs:
            quote = quotes.get(str(leg["symbol"]))
            if quote is None:
                return None
            bid, ask = float(quote[0]), float(quote[1])
            if bid < 0 or ask <= 0 or ask < bid or not math.isfinite(bid + ask):
                return None
            total += bid if str(leg["side"]).upper() == "BUY" else -ask
        return total

    # -- reporting and risk ------------------------------------------------

    def day_realized_dollars(self, now: datetime) -> float:
        day = now.astimezone(UTC).date().isoformat()
        with self.store.lock:
            row = self.store.connection.execute(
                """
                SELECT COALESCE(SUM(realized_pnl_dollars), 0) FROM paper_positions
                WHERE status='CLOSED' AND substr(closed_at, 1, 10) = ?
                """,
                (day,),
            ).fetchone()
        return float(row[0] or 0.0)

    def total_realized_dollars(self) -> float:
        with self.store.lock:
            row = self.store.connection.execute(
                "SELECT COALESCE(SUM(realized_pnl_dollars), 0) FROM paper_positions WHERE status='CLOSED'"
            ).fetchone()
        return float(row[0] or 0.0)

    def breaker_tripped(self, now: datetime) -> bool:
        return self.day_realized_dollars(now) <= -abs(self.daily_loss_limit_dollars)

    def consecutive_losses_today(self, now: datetime) -> int:
        day = now.astimezone(UTC).date().isoformat()
        with self.store.lock:
            rows = self.store.connection.execute(
                """
                SELECT realized_pnl_dollars FROM paper_positions
                WHERE status='CLOSED' AND substr(closed_at, 1, 10) = ?
                ORDER BY closed_at DESC
                """,
                (day,),
            ).fetchall()
        streak = 0
        for row in rows:
            if float(row[0] or 0.0) < 0:
                streak += 1
            else:
                break
        return streak

    def size_multiplier(self, now: datetime) -> float:
        """Compound the risk budget with the bankroll; throttle loss streaks.

        Winners raise equity and therefore size (clipped so one hot week
        cannot triple exposure overnight); three consecutive losing trades in
        a day halve size for the next entry until a winner resets the streak.
        """
        multiplier = 1.0
        if self.starting_equity:
            equity = self.starting_equity + self.total_realized_dollars()
            multiplier *= min(max(equity / self.starting_equity, 0.5), self.bankroll_multiplier_cap)
        if self.consecutive_losses_today(now) >= self.loss_streak_trigger:
            multiplier *= 0.5
        return float(multiplier)

    def stats(self, now: datetime) -> dict[str, Any]:
        with self.store.lock:
            totals = self.store.connection.execute(
                """
                SELECT COUNT(*),
                       COALESCE(SUM(realized_pnl_dollars), 0),
                       COALESCE(SUM(CASE WHEN realized_pnl_dollars > 0 THEN 1 ELSE 0 END), 0)
                FROM paper_positions WHERE status='CLOSED'
                """
            ).fetchone()
            recent = self.store.connection.execute(
                """
                SELECT strategy, direction, exit_reason, realized_pnl_dollars, closed_at
                FROM paper_positions WHERE status='CLOSED'
                ORDER BY closed_at DESC LIMIT 8
                """
            ).fetchall()
        working = self._rows(("OPEN", "PENDING"))
        closed_count = int(totals[0])
        realized = float(totals[1])
        wins = int(totals[2])
        return {
            "open_positions": [
                {
                    "id": row["id"],
                    "strategy": row["strategy"],
                    "direction": row["direction"],
                    "status": row["status"],
                    "contracts": row["contracts"],
                    "entry_price": row["entry_price"],
                    "opened_at": row["opened_at"],
                    "unrealized_pnl_dollars": row["unrealized_pnl_dollars"],
                    "max_loss_dollars": row["max_loss_dollars"],
                }
                for row in working
            ],
            "open_count": len(working),
            "closed_count": closed_count,
            "wins": wins,
            "win_rate": (wins / closed_count) if closed_count else None,
            "realized_pnl_dollars": round(realized, 2),
            "day_realized_pnl_dollars": round(self.day_realized_dollars(now), 2),
            "unrealized_pnl_dollars": round(
                sum(float(row["unrealized_pnl_dollars"] or 0.0) for row in working), 2
            ),
            "breaker_tripped": self.breaker_tripped(now),
            "daily_loss_limit_dollars": self.daily_loss_limit_dollars,
            "size_multiplier": round(self.size_multiplier(now), 3),
            "equity": (
                round(self.starting_equity + realized, 2) if self.starting_equity else None
            ),
            "recent_closed": [
                {
                    "strategy": row[0],
                    "direction": row[1],
                    "exit_reason": row[2],
                    "realized_pnl_dollars": row[3],
                    "closed_at": row[4],
                }
                for row in recent
            ],
        }

    def _rows(self, statuses: tuple[str, ...]) -> list[dict[str, Any]]:
        placeholders = ",".join("?" for _ in statuses)
        with self.store.lock:
            cursor = self.store.connection.execute(
                f"""
                SELECT id, opened_at, strategy, direction, expiration, contracts, entry_price,
                       is_credit, max_loss_dollars, max_profit_dollars, hold_minutes, legs,
                       unrealized_pnl_dollars, status, limit_cash, width, entry_style
                FROM paper_positions WHERE status IN ({placeholders}) ORDER BY id
                """,
                statuses,
            )
            names = [column[0] for column in cursor.description]
            return [dict(zip(names, row)) for row in cursor.fetchall()]
