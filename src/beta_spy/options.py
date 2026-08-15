from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Iterable


@dataclass(frozen=True)
class OptionLeg:
    symbol: str
    side: str
    right: str
    strike: float
    bid: float
    ask: float
    delta: float | None


@dataclass(frozen=True)
class OptionPlan:
    strategy: str
    direction: str
    expiration: str
    debit: float
    width: float
    max_loss_dollars: float
    max_profit_dollars: float
    score: float
    legs: tuple[OptionLeg, OptionLeg]
    contracts: int = 1
    total_risk_dollars: float | None = None
    expected_value_dollars: float | None = None

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


def plan_debit_spread(
    options: Iterable[dict[str, Any]],
    direction: str,
    *,
    maximum_risk_dollars: float = 100.0,
    max_width: float = 5.0,
    min_open_interest: int = 50,
    max_relative_spread: float = 0.20,
    expected_move_dollars: float | None = None,
    probability: float | None = None,
) -> OptionPlan | None:
    """Select one deterministic defined-risk SPY debit spread.

    Quotes are treated as executable pessimistically: buy at ask, sell at bid. The planner
    only expresses a Tape-500 direction; it does not create edge or submit an order.

    When the forecast's expected move (in dollars, signed toward the trade direction)
    is supplied, spreads are ranked by expected value per dollar of risk: the position
    delta captures the forecast move, and the bid/ask friction actually paid at entry
    is charged against it. Without a forecast the ranking falls back to delta targets.

    Sizing: the risk budget buys ``contracts`` spreads (never exceeding the budget),
    so a stronger signal can express more risk through the same structure.
    """
    bullish = str(direction).upper() == "BULLISH"
    right = "C" if bullish else "P"
    rows = [row for row in options if str(row.get("right") or "").upper() == right]
    valid = [row for row in rows if _liquid(row, min_open_interest, max_relative_spread)]
    best: OptionPlan | None = None
    for long in valid:
        for short in valid:
            long_strike = _float(long.get("strike"))
            short_strike = _float(short.get("strike"))
            if long_strike is None or short_strike is None:
                continue
            if bullish and short_strike <= long_strike:
                continue
            if not bullish and short_strike >= long_strike:
                continue
            width = abs(short_strike - long_strike)
            if width <= 0 or width > max_width:
                continue
            long_ask = _float(long.get("ask")) or 0.0
            short_bid = _float(short.get("bid")) or 0.0
            debit = long_ask - short_bid
            if debit <= 0 or debit >= width:
                continue
            max_loss = debit * 100.0
            if max_loss > maximum_risk_dollars + 1e-9:
                continue
            max_profit = (width - debit) * 100.0
            long_delta = _float(long.get("delta"))
            short_delta = _float(short.get("delta"))
            expected_value: float | None = None
            if expected_move_dollars is not None and long_delta is not None and short_delta is not None:
                position_delta = abs(long_delta - short_delta)
                # Friction actually paid versus mid when entering pessimistically.
                friction = (_half_spread(long) + _half_spread(short)) * 100.0
                capture = position_delta * abs(expected_move_dollars) * 100.0
                win = max(probability if probability is not None else 0.5, 0.0)
                # Expected value of the mark move, charged for entry friction and
                # weighted by how often the direction call is right vs wrong.
                expected_value = (win * capture) - ((1.0 - win) * capture) - friction
                score = expected_value / max(max_loss, 1e-9)
            else:
                long_target = 0.55 if bullish else -0.55
                short_target = 0.30 if bullish else -0.30
                delta_error = (
                    abs((long_delta if long_delta is not None else long_target) - long_target)
                    + abs((short_delta if short_delta is not None else short_target) - short_target)
                )
                spread_cost = _relative_spread(long) + _relative_spread(short)
                risk_reward = max_profit / max(max_loss, 1e-9)
                score = risk_reward - 2.0 * delta_error - spread_cost
            contracts = max(int(maximum_risk_dollars // max_loss), 1)
            expiration = str(long.get("expiration") or short.get("expiration") or "")
            plan = OptionPlan(
                strategy="CALL_DEBIT_SPREAD" if bullish else "PUT_DEBIT_SPREAD",
                direction="BULLISH" if bullish else "BEARISH",
                expiration=expiration,
                debit=round(debit, 4),
                width=width,
                max_loss_dollars=round(max_loss, 2),
                max_profit_dollars=round(max_profit, 2),
                score=score,
                legs=(
                    _leg(long, "BUY", right),
                    _leg(short, "SELL", right),
                ),
                contracts=contracts,
                total_risk_dollars=round(max_loss * contracts, 2),
                expected_value_dollars=(
                    round(expected_value * contracts, 2) if expected_value is not None else None
                ),
            )
            if best is None or plan.score > best.score:
                best = plan
    if best is not None and best.expected_value_dollars is not None and best.expected_value_dollars <= 0:
        # A trade whose best expression still has negative expected value after
        # friction is not worth putting on at all.
        return None
    return best


def _leg(row: dict[str, Any], side: str, right: str) -> OptionLeg:
    return OptionLeg(
        symbol=str(row.get("symbol") or ""),
        side=side,
        right=right,
        strike=float(row["strike"]),
        bid=float(row["bid"]),
        ask=float(row["ask"]),
        delta=_float(row.get("delta")),
    )


def _liquid(row: dict[str, Any], min_oi: int, max_spread: float) -> bool:
    bid = _float(row.get("bid")) or 0.0
    ask = _float(row.get("ask")) or 0.0
    if bid <= 0 or ask < bid:
        return False
    if int(_float(row.get("open_interest")) or 0) < min_oi:
        return False
    return _relative_spread(row) <= max_spread


def _half_spread(row: dict[str, Any]) -> float:
    bid = _float(row.get("bid")) or 0.0
    ask = _float(row.get("ask")) or 0.0
    return max(ask - bid, 0.0) / 2.0


def _relative_spread(row: dict[str, Any]) -> float:
    bid = _float(row.get("bid")) or 0.0
    ask = _float(row.get("ask")) or 0.0
    mid = (bid + ask) / 2.0
    return (ask - bid) / mid if mid > 0 else float("inf")


def _float(value: Any) -> float | None:
    try:
        return float(value) if value is not None else None
    except (TypeError, ValueError):
        return None
