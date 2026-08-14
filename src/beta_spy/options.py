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

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


def plan_debit_spread(
    options: Iterable[dict[str, Any]],
    direction: str,
    *,
    maximum_risk_dollars: float = 100.0,
    max_width: float = 5.0,
    min_open_interest: int = 25,
    max_relative_spread: float = 0.35,
) -> OptionPlan | None:
    """Select one deterministic defined-risk SPY debit spread.

    Quotes are treated as executable pessimistically: buy at ask, sell at bid. The planner
    only expresses a Tape-500 direction; it does not create edge or submit an order.
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
            long_target = 0.55 if bullish else -0.55
            short_target = 0.30 if bullish else -0.30
            delta_error = (
                abs((long_delta if long_delta is not None else long_target) - long_target)
                + abs((short_delta if short_delta is not None else short_target) - short_target)
            )
            spread_cost = _relative_spread(long) + _relative_spread(short)
            risk_reward = max_profit / max(max_loss, 1e-9)
            score = risk_reward - 2.0 * delta_error - spread_cost
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
            )
            if best is None or plan.score > best.score:
                best = plan
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
