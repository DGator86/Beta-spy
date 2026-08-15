from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from typing import Any, Iterable, Iterator


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
    legs: tuple[OptionLeg, ...]
    contracts: int = 1
    total_risk_dollars: float | None = None
    expected_value_dollars: float | None = None
    hold_minutes: float = 15.0

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


TRADING_MINUTES_PER_DAY = 390.0


def implied_move_dollars(
    options: Iterable[dict[str, Any]],
    spy_price: float,
    *,
    hold_minutes: float,
    minutes_to_expiry: float,
) -> float | None:
    """Market-implied |move| over the holding period from the ATM straddle.

    The straddle mid approximates the expected absolute move to expiry
    (the usual 0.85 haircut applied); square-root-of-time scales it down
    to the holding window.
    """
    if spy_price <= 0:
        return None
    best: dict[str, dict[str, Any]] = {}
    for row in options:
        strike = _float(row.get("strike"))
        right = str(row.get("right") or "").upper()
        if strike is None or right not in {"C", "P"}:
            continue
        current = best.get(right)
        if current is None or abs(strike - spy_price) < abs(float(current["strike"]) - spy_price):
            best[right] = row
    call, put = best.get("C"), best.get("P")
    if call is None or put is None:
        return None
    call_mid = ((_float(call.get("bid")) or 0.0) + (_float(call.get("ask")) or 0.0)) / 2.0
    put_mid = ((_float(put.get("bid")) or 0.0) + (_float(put.get("ask")) or 0.0)) / 2.0
    straddle = call_mid + put_mid
    if straddle <= 0:
        return None
    horizon = max(minutes_to_expiry, hold_minutes, 1.0)
    return 0.85 * straddle * math.sqrt(hold_minutes / horizon)


def _greeks_expected_value(
    legs: list[tuple[dict[str, Any], int]],
    *,
    expected_move_dollars: float,
    move_scale_dollars: float,
    hold_minutes: float,
) -> float | None:
    """Expected mark-to-market P&L in dollars per structure over the hold.

    ``legs`` pairs a chain row with +1 (buy) or -1 (sell). Delta captures the
    signed forecast move, gamma charges (short) or credits (long) convexity
    against the squared move scale, theta accrues over the holding fraction of
    the session, and the bid/ask friction actually paid at entry is deducted.
    """
    net_delta = net_gamma = net_theta = friction = 0.0
    for row, sign in legs:
        delta = _float(row.get("delta"))
        if delta is None:
            return None
        net_delta += sign * delta
        net_gamma += sign * (_float(row.get("gamma")) or 0.0)
        net_theta += sign * (_float(row.get("theta")) or 0.0)
        friction += _half_spread(row)
    move_sq = max(abs(expected_move_dollars), move_scale_dollars) ** 2
    directional = net_delta * expected_move_dollars
    convexity = 0.5 * net_gamma * move_sq
    carry = net_theta * (hold_minutes / TRADING_MINUTES_PER_DAY)
    return (directional + convexity + carry - friction) * 100.0


def _norm_pdf(x: float) -> float:
    return math.exp(-0.5 * x * x) / math.sqrt(2.0 * math.pi)


def _norm_cdf(x: float) -> float:
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


def _expected_capped_call_payoff(spot: float, strike: float, width: float, sigma: float) -> float:
    """E[min(max(S_T - strike, 0), width)] with S_T ~ N(spot, sigma^2)."""
    if sigma <= 0:
        return min(max(spot - strike, 0.0), width)

    def _call(k: float) -> float:
        d = (k - spot) / sigma
        return sigma * _norm_pdf(d) - (k - spot) * _norm_cdf(-d)

    return max(_call(strike) - _call(strike + width), 0.0)


def _expiry_condor_expected_value(
    legs: list[tuple[dict[str, Any], int]],
    credit: float,
    spot: float,
    sigma: float,
) -> float:
    """Hold-to-expiry EV: credit collected minus expected capped losses per side.

    Entry friction is already embedded because the credit is computed at the
    bid for shorts and the ask for longs; settlement is frictionless.
    """
    expected_loss = 0.0
    shorts = [(row, sign) for row, sign in legs if sign < 0]
    longs = {str(row.get("right") or "").upper(): float(row["strike"]) for row, sign in legs if sign > 0}
    for row, _ in shorts:
        right = str(row.get("right") or "").upper()
        strike = float(row["strike"])
        wing = longs.get(right)
        if wing is None:
            return -float("inf")
        width = abs(strike - wing)
        if right == "C":
            expected_loss += _expected_capped_call_payoff(spot, strike, width, sigma)
        else:
            # Put payoff mirrors the call payoff around the spot.
            expected_loss += _expected_capped_call_payoff(spot, 2.0 * spot - strike, width, sigma)
    return (credit - expected_loss) * 100.0


def _vertical_candidates(
    valid: list[dict[str, Any]],
    direction: str,
    *,
    max_width: float,
    maximum_risk_dollars: float,
) -> Iterator[tuple[str, list[tuple[dict[str, Any], int]], float, float, float, str]]:
    """Yield (strategy, legs, debit_or_credit, max_loss, max_profit, expiration)."""
    bullish = str(direction).upper() == "BULLISH"
    debit_right = "C" if bullish else "P"
    credit_right = "P" if bullish else "C"
    by_right: dict[str, list[dict[str, Any]]] = {"C": [], "P": []}
    for row in valid:
        right = str(row.get("right") or "").upper()
        if right in by_right:
            by_right[right].append(row)

    for long in by_right[debit_right]:
        for short in by_right[debit_right]:
            long_strike, short_strike = _float(long.get("strike")), _float(short.get("strike"))
            if long_strike is None or short_strike is None:
                continue
            if bullish and short_strike <= long_strike:
                continue
            if not bullish and short_strike >= long_strike:
                continue
            width = abs(short_strike - long_strike)
            if width <= 0 or width > max_width:
                continue
            debit = (_float(long.get("ask")) or 0.0) - (_float(short.get("bid")) or 0.0)
            if debit <= 0 or debit >= width:
                continue
            max_loss = debit * 100.0
            if max_loss > maximum_risk_dollars + 1e-9:
                continue
            expiration = str(long.get("expiration") or short.get("expiration") or "")
            yield (
                "CALL_DEBIT_SPREAD" if bullish else "PUT_DEBIT_SPREAD",
                [(long, 1), (short, -1)],
                debit,
                max_loss,
                (width - debit) * 100.0,
                expiration,
            )

    for short in by_right[credit_right]:
        short_delta = _float(short.get("delta"))
        # Sell premium out of the money: the short strike must not be the
        # deep-in-the-money side of the market.
        if short_delta is not None and abs(short_delta) > 0.45:
            continue
        for long in by_right[credit_right]:
            long_strike, short_strike = _float(long.get("strike")), _float(short.get("strike"))
            if long_strike is None or short_strike is None:
                continue
            # Protection sits farther out than the short strike.
            if bullish and long_strike >= short_strike:
                continue
            if not bullish and long_strike <= short_strike:
                continue
            width = abs(short_strike - long_strike)
            if width <= 0 or width > max_width:
                continue
            credit = (_float(short.get("bid")) or 0.0) - (_float(long.get("ask")) or 0.0)
            if credit <= 0 or credit >= width:
                continue
            max_loss = (width - credit) * 100.0
            if max_loss <= 0 or max_loss > maximum_risk_dollars + 1e-9:
                continue
            expiration = str(short.get("expiration") or long.get("expiration") or "")
            yield (
                "PUT_CREDIT_SPREAD" if bullish else "CALL_CREDIT_SPREAD",
                [(short, -1), (long, 1)],
                credit,
                max_loss,
                credit * 100.0,
                expiration,
            )


def _condor_candidates(
    valid: list[dict[str, Any]],
    *,
    max_width: float,
    maximum_risk_dollars: float,
) -> Iterator[tuple[str, list[tuple[dict[str, Any], int]], float, float, float, str]]:
    calls = [row for row in valid if str(row.get("right") or "").upper() == "C"]
    puts = [row for row in valid if str(row.get("right") or "").upper() == "P"]

    def shorts(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
        picked = []
        for row in rows:
            delta = _float(row.get("delta"))
            if delta is not None and 0.10 <= abs(delta) <= 0.35:
                picked.append(row)
        return picked

    for short_put in shorts(puts):
        sp_strike = _float(short_put.get("strike"))
        if sp_strike is None:
            continue
        for long_put in puts:
            lp_strike = _float(long_put.get("strike"))
            if lp_strike is None or lp_strike >= sp_strike:
                continue
            put_width = sp_strike - lp_strike
            if put_width > max_width:
                continue
            for short_call in shorts(calls):
                sc_strike = _float(short_call.get("strike"))
                if sc_strike is None or sc_strike <= sp_strike:
                    continue
                for long_call in calls:
                    lc_strike = _float(long_call.get("strike"))
                    if lc_strike is None or lc_strike <= sc_strike:
                        continue
                    call_width = lc_strike - sc_strike
                    if call_width > max_width:
                        continue
                    credit = (
                        (_float(short_put.get("bid")) or 0.0)
                        - (_float(long_put.get("ask")) or 0.0)
                        + (_float(short_call.get("bid")) or 0.0)
                        - (_float(long_call.get("ask")) or 0.0)
                    )
                    if credit <= 0:
                        continue
                    max_loss = (max(put_width, call_width) - credit) * 100.0
                    if max_loss <= 0 or max_loss > maximum_risk_dollars + 1e-9:
                        continue
                    expiration = str(short_put.get("expiration") or short_call.get("expiration") or "")
                    yield (
                        "IRON_CONDOR",
                        [(short_put, -1), (long_put, 1), (short_call, -1), (long_call, 1)],
                        credit,
                        max_loss,
                        credit * 100.0,
                        expiration,
                    )


def plan_best_strategy(
    options: Iterable[dict[str, Any]],
    direction: str,
    *,
    maximum_risk_dollars: float = 100.0,
    hold_minutes: float = 15.0,
    spy_price: float | None = None,
    expected_move_dollars: float = 0.0,
    minutes_to_expiry: float = TRADING_MINUTES_PER_DAY,
    max_width: float = 5.0,
    min_open_interest: int = 50,
    max_relative_spread: float = 0.20,
) -> OptionPlan | None:
    """Pick the best defined-risk structure for the signal, long or short premium.

    BULLISH considers call debit spreads and put credit spreads; BEARISH the
    mirror pair; NEUTRAL considers iron condors. Every candidate is scored by
    the same greeks-based expected value per dollar of risk, so premium selling
    wins exactly when theta plus win-rate beats the directional capture of the
    debit alternative. ``expected_move_dollars`` is signed toward the forecast.
    A plan is only returned when its expected value is positive after friction.
    """
    rows = list(options)
    valid = [row for row in rows if _liquid(row, min_open_interest, max_relative_spread)]
    if not valid:
        return None
    neutral = str(direction).upper() == "NEUTRAL"
    move_scale = None
    sigma_to_expiry = None
    if spy_price is not None:
        move_scale = implied_move_dollars(
            rows, spy_price, hold_minutes=hold_minutes, minutes_to_expiry=minutes_to_expiry
        )
        full_move = implied_move_dollars(
            rows, spy_price, hold_minutes=minutes_to_expiry, minutes_to_expiry=minutes_to_expiry
        )
        if full_move is not None:
            # E|move| = sigma * sqrt(2/pi) for a normal terminal distribution.
            sigma_to_expiry = (full_move / 0.85) * math.sqrt(math.pi / 2.0)
    if move_scale is None:
        move_scale = abs(expected_move_dollars)
    if neutral and (spy_price is None or sigma_to_expiry is None):
        return None

    candidates = (
        _condor_candidates(valid, max_width=max_width, maximum_risk_dollars=maximum_risk_dollars)
        if neutral
        else _vertical_candidates(
            valid, direction, max_width=max_width, maximum_risk_dollars=maximum_risk_dollars
        )
    )
    best: OptionPlan | None = None
    best_score = 0.0
    for strategy, legs, price, max_loss, max_profit, expiration in candidates:
        if neutral:
            expected_value = _expiry_condor_expected_value(
                legs, price, float(spy_price), float(sigma_to_expiry)
            )
        else:
            expected_value = _greeks_expected_value(
                legs,
                expected_move_dollars=expected_move_dollars,
                move_scale_dollars=move_scale,
                hold_minutes=hold_minutes,
            )
        if expected_value is None or expected_value <= 0:
            continue
        score = expected_value / max(max_loss, 1e-9)
        if best is None or score > best_score:
            contracts = max(int(maximum_risk_dollars // max_loss), 1)
            strikes_by_right: dict[str, list[float]] = {}
            for row, _ in legs:
                strikes_by_right.setdefault(str(row.get("right") or "").upper(), []).append(
                    float(row["strike"])
                )
            width = max(max(vals) - min(vals) for vals in strikes_by_right.values())
            best_score = score
            best = OptionPlan(
                strategy=strategy,
                direction=str(direction).upper(),
                expiration=expiration,
                debit=round(price, 4),
                width=width,
                max_loss_dollars=round(max_loss, 2),
                max_profit_dollars=round(max_profit, 2),
                score=score,
                legs=tuple(_leg(row, "BUY" if sign > 0 else "SELL", str(row.get("right") or "")) for row, sign in legs),
                contracts=contracts,
                total_risk_dollars=round(max_loss * contracts, 2),
                expected_value_dollars=round(expected_value * contracts, 2),
                hold_minutes=hold_minutes,
            )
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
    # Cheap far-out-of-the-money wings always look wide in relative terms;
    # a few cents of absolute spread is still perfectly executable.
    if ask - bid <= 0.05:
        return True
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
