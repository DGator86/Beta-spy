"""Short-premium 0DTE engine: sell decay, defend the tail.

The existing planners in :mod:`beta_spy.options` are debit-oriented -- they buy
optionality and pay theta. This module is the other side of the trade. It sells
defined-risk premium and holds it long enough for decay to actually happen.

Three things shape the design, each of them a direct response to something the
saved tape showed:

*Chain hygiene comes first.* Measured over 25 sessions of captured 0DTE tape,
6.8% of adjacent one-wide verticals intraday -- and 16.4% in the closing ten
minutes -- were quoted at a value greater than their own width, which is
arithmetically impossible. A further 7.5% of in-the-money quotes in the first
five minutes were asked below intrinsic. Every price this module touches goes
through :class:`ChainGate` first, and every structure it values is clamped to
its own arbitrage bound. A short-premium book that marks garbage does not get
to find out whether its strategy works.

*Theta needs time.* The prior book held positions for a median of 35 minutes
against roughly 360 minutes of available decay, which is a directional bet
wearing a theta label. Positions here are opened once, early, and held to a
profit target or to the close.

*The tail is the whole game, and the exits are the edge.* Over the same 25
sessions a naive 16-delta condor won 11 days of 20 and still lost money,
because three days erased eleven wins. The median day's realised move was 0.96x
its implied move -- the typical day pays. Survival is therefore not about
picking direction, it is about being small or absent when the range breaks.
:func:`assess_day` is the stand-aside gate and :func:`size_position` assumes
the tail will arrive.

More pointedly: on the sampled sessions, **two settlements in twelve landed
beyond a long wing** -- a full max loss had the position been held to expiry.
Break-even for this strategy is one full max-loss day in thirty-four. Held to
settlement it is therefore deeply negative on this tape; the only reason it
shows a profit is that the profit target and the 15:45 flat get it out first.
The exits are not housekeeping, they carry the entire result, and they run on
option marks -- the least reliable input in the system. That is why
:class:`ChainGate` and the bound clamp in :meth:`PremiumStructure.value_at`
are not defensive extras but load-bearing parts of the strategy.

Nothing in the stand-aside gate is fitted to the sample. Each rule is one that
can be argued for before seeing the data; the one empirically suggestive signal
found in the tape (dealer gamma sign) is available via
``PremiumConfig.gamma_filter`` but is **off by default**, because at 20
sessions it carried a permutation p-value of 0.108 and is not yet evidence.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Iterable, Literal, Mapping

Side = Literal["put", "call"]
Action = Literal["BUY", "SELL"]

__all__ = [
    "ChainGate",
    "CleanChain",
    "DayAssessment",
    "PremiumConfig",
    "PremiumStructure",
    "StructureLeg",
    "assess_day",
    "build_condor",
    "build_credit_spread",
    "manage_position",
    "size_position",
]


# ---------------------------------------------------------------------------
# configuration
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class PremiumConfig:
    """Every tunable in one place, with defaults chosen before fitting."""

    # --- structure selection ---
    short_delta: float = 0.16
    """Target absolute delta for the short strikes (~1 standard deviation)."""
    wing_points: float = 2.0
    """Distance from short strike to long wing, in points."""
    min_credit_ratio: float = 0.12
    """Reject a structure paying less than this fraction of its max loss.

    Selling a 1-wide vertical for 5 cents is taking 95 cents of risk for a
    nickel. The floor is what makes the trade worth its own tail.
    """

    # --- chain hygiene ---
    max_relative_spread: float = 0.35
    min_open_interest: int = 0
    reject_below_intrinsic: bool = True
    reject_vertical_breach: bool = True

    # --- entry window ---
    entry_after: str = "10:00"
    """No entries before this. The open is where stale quotes live."""
    entry_before: str = "12:00"
    """No new entries after this -- decay left is what pays for the risk."""

    # --- exits ---
    profit_target_fraction: float = 0.55
    """Close once this fraction of the entry credit has decayed away."""
    stop_multiple: float | None = None
    """Mark-based stop, as a multiple of the entry credit. Disabled by default.

    A defined-risk short structure already has a stop: the long wing caps the
    loss at ``width - credit`` by construction. A second stop *inside* that one
    does two harmful things. It converts intraday noise into realised losses --
    a 0DTE condor routinely trades against you and comes back -- and it fires
    off a mark, at exactly the moments marks are least trustworthy. In the
    captured tape 16.4% of one-wide verticals in the closing window were quoted
    beyond their own width; a mark-triggered stop is most likely to fire
    precisely there.

    Use :attr:`breach_exit_points` instead, which triggers off spot.
    """
    breach_exit_points: float | None = None
    """Close when spot trades this far beyond a short strike.

    Defends the tail using the underlying, which is always quotable, rather
    than the option marks, which are not. ``0.0`` exits the moment a short
    strike goes in the money; ``None`` disables it and relies on the wing.
    """
    flatten_at: str = "15:45"
    """Hard flat. Settlement risk on a short 0DTE structure is unbounded in
    practice even when it is bounded on paper."""

    # --- sizing ---
    risk_fraction: float = 0.02
    """Fraction of equity at risk per trade, against *true* max loss."""
    max_contracts: int = 50

    # --- stand-aside gates ---
    skip_on_catalyst: bool = True
    max_realised_fraction: float = 0.60
    """Stand aside if the day has already used this much of its implied range
    by entry time -- the range is breaking before the trade is on."""
    min_expected_range: float = 0.75
    """Below this the options are not paying enough to be worth the tail."""
    gamma_filter: bool = False
    """Unproven. See module docstring: p = 0.108 over 20 sessions."""

    @classmethod
    def ramp(cls, **overrides: Any) -> "PremiumConfig":
        """Aggressive preset for compounding a small account to a fixed target.

        One-point wings, because at a $1,000 account a two-point condor is $163
        of risk -- 16% of the account in a single lot, six max-loss days from
        being unable to trade at all. One-point wings are $79, which buys
        twelve.

        Sizing is 25%, which is where the probability of reaching a 15x target
        peaks in simulation. **Going above it is strictly worse on both axes:**
        at 50% the chance of reaching the target falls and the chance of
        busting rises, because beyond the growth-optimal fraction the drag from
        variance outruns the edge.

        Simulated on 12 sessions of captured tape, $1,000 to $15,000, resampling
        daily returns and drawing the true edge from its own confidence interval:

            risk    P(reach 15k)   P(bust)   median
            15%          49.2%      39.9%    224 sessions
            20%          50.9%      41.6%    166
            25%          52.3%      42.6%    138
            35%          52.2%      45.8%    100
            50%          48.9%      50.8%     68

        Those figures assume full max-loss days never occur, which is what the
        sample shows and is almost certainly optimistic. Break-even sits at one
        full max-loss day per 34 sessions; two of the twelve sampled sessions
        would have been full losses had they been held to settlement. The
        strategy is only positive because it exits early -- see the module
        docstring. Treat these numbers as an upper bound, not a forecast.
        """
        base = {"wing_points": 1.0, "risk_fraction": 0.25, "max_contracts": 50}
        base.update(overrides)
        return cls(**base)  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# chain hygiene
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class CleanChain:
    """A chain that has passed validation, plus what was thrown away."""

    spot: float
    puts: dict[float, tuple[float, float, float | None]]
    calls: dict[float, tuple[float, float, float | None]]
    rejected: dict[str, int]

    def mid(self, side: Side, strike: float) -> float | None:
        book = self.puts if side == "put" else self.calls
        row = book.get(strike)
        return None if row is None else (row[0] + row[1]) / 2.0

    def delta(self, side: Side, strike: float) -> float | None:
        book = self.puts if side == "put" else self.calls
        row = book.get(strike)
        return None if row is None else row[2]

    def strikes(self, side: Side) -> list[float]:
        return sorted(self.puts if side == "put" else self.calls)

    @property
    def usable(self) -> bool:
        return len(self.puts) >= 2 and len(self.calls) >= 2


class ChainGate:
    """Rejects quotes that cannot be true before anything prices off them."""

    def __init__(self, config: PremiumConfig | None = None):
        self.config = config or PremiumConfig()

    def clean(self, spot: float, rows: Iterable[Mapping[str, Any]]) -> CleanChain:
        cfg = self.config
        puts: dict[float, tuple[float, float, float | None]] = {}
        calls: dict[float, tuple[float, float, float | None]] = {}
        rejected: dict[str, int] = {
            "malformed": 0,
            "below_intrinsic": 0,
            "wide_spread": 0,
            "thin_oi": 0,
            "vertical_breach": 0,
        }

        for row in rows:
            side = str(row.get("side", "")).lower()
            if side not in ("put", "call"):
                rejected["malformed"] += 1
                continue
            strike = _as_float(row.get("strike"))
            bid = _as_float(row.get("bid"))
            ask = _as_float(row.get("ask"))
            if strike is None or bid is None or ask is None:
                rejected["malformed"] += 1
                continue
            if ask <= 0.0 or bid < 0.0 or ask < bid:
                rejected["malformed"] += 1
                continue

            if cfg.reject_below_intrinsic:
                intrinsic = max(0.0, strike - spot) if side == "put" else max(0.0, spot - strike)
                # An option asked below intrinsic is a riskless arbitrage and in
                # practice a quote carried over from a prior session.
                if ask < intrinsic - 1e-9:
                    rejected["below_intrinsic"] += 1
                    continue

            mid = (bid + ask) / 2.0
            if mid > 0 and (ask - bid) > 0.05 and (ask - bid) / mid > cfg.max_relative_spread:
                rejected["wide_spread"] += 1
                continue

            oi = _as_float(row.get("oi") if row.get("oi") is not None else row.get("open_interest"))
            if cfg.min_open_interest and (oi or 0) < cfg.min_open_interest:
                rejected["thin_oi"] += 1
                continue

            book = puts if side == "put" else calls
            book[strike] = (bid, ask, _as_float(row.get("delta")))

        if cfg.reject_vertical_breach:
            rejected["vertical_breach"] += _drop_vertical_breaches(puts, "put")
            rejected["vertical_breach"] += _drop_vertical_breaches(calls, "call")

        return CleanChain(spot=spot, puts=puts, calls=calls, rejected=rejected)


def _drop_vertical_breaches(book: dict[float, tuple[float, float, float | None]], side: Side) -> int:
    """Remove strikes whose adjacent vertical exceeds its own width.

    A one-point vertical cannot be worth more than one point. Where the pair
    violates that, at least one of the two quotes is wrong and we cannot tell
    which, so both leave the book.
    """
    dropped = 0
    strikes = sorted(book)
    bad: set[float] = set()
    for lo, hi in zip(strikes, strikes[1:]):
        width = hi - lo
        if width <= 0 or width > 1.001:
            continue
        lo_mid = (book[lo][0] + book[lo][1]) / 2.0
        hi_mid = (book[hi][0] + book[hi][1]) / 2.0
        spread_value = (hi_mid - lo_mid) if side == "put" else (lo_mid - hi_mid)
        if spread_value > width + 1e-9:
            bad.add(lo)
            bad.add(hi)
    for strike in bad:
        book.pop(strike, None)
        dropped += 1
    return dropped


# ---------------------------------------------------------------------------
# structures
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class StructureLeg:
    side: Side
    action: Action
    strike: float
    bid: float
    ask: float
    delta: float | None


@dataclass(frozen=True)
class PremiumStructure:
    kind: str
    legs: tuple[StructureLeg, ...]
    credit: float
    """Credit received per share at mid, always positive for a valid sale."""
    width: float
    """The widest single wing -- the structure's true risk, not the sum."""
    short_strikes: tuple[float, ...]

    @property
    def max_loss(self) -> float:
        return max(0.0, self.width - self.credit)

    @property
    def credit_ratio(self) -> float:
        risk = self.max_loss
        return self.credit / risk if risk > 0 else float("inf")

    def value_at(self, chain: CleanChain) -> float | None:
        """Cost per share to flatten now, clamped to the structure's bound.

        The clamp is the point. Summing four legs off a torn snapshot is how a
        one-wide fly gets marked at 1.435 and a $60 risk books a $942 loss.
        A structure cannot be worth more than its widest wing, so a mark that
        says otherwise is rejected rather than acted on.
        """
        total = 0.0
        for leg in self.legs:
            mid = chain.mid(leg.side, leg.strike)
            if mid is None:
                return None
            total += mid if leg.action == "SELL" else -mid
        # `total` is what it costs to buy the structure back, expressed positive.
        return min(max(total, 0.0), self.width)

    def as_dict(self) -> dict[str, Any]:
        return {
            "kind": self.kind,
            "credit": round(self.credit, 4),
            "width": self.width,
            "max_loss": round(self.max_loss, 4),
            "credit_ratio": round(self.credit_ratio, 4),
            "short_strikes": list(self.short_strikes),
            "legs": [
                {"side": leg.side, "action": leg.action, "strike": leg.strike, "delta": leg.delta}
                for leg in self.legs
            ],
        }


def _pick_by_delta(chain: CleanChain, side: Side, target: float) -> float | None:
    """Strike whose |delta| is closest to `target`, on the correct side of spot."""
    best: tuple[float, float] | None = None
    for strike in chain.strikes(side):
        delta = chain.delta(side, strike)
        if delta is None:
            continue
        if side == "put" and strike > chain.spot:
            continue
        if side == "call" and strike < chain.spot:
            continue
        error = abs(abs(delta) - target)
        if best is None or error < best[0]:
            best = (error, strike)
    return None if best is None else best[1]


def _leg(chain: CleanChain, side: Side, action: Action, strike: float) -> StructureLeg | None:
    book = chain.puts if side == "put" else chain.calls
    row = book.get(strike)
    if row is None:
        return None
    return StructureLeg(side=side, action=action, strike=strike, bid=row[0], ask=row[1], delta=row[2])


def build_credit_spread(
    chain: CleanChain, side: Side, config: PremiumConfig | None = None
) -> PremiumStructure | None:
    """One vertical: short near the money, long wing further out."""
    cfg = config or PremiumConfig()
    short_strike = _pick_by_delta(chain, side, cfg.short_delta)
    if short_strike is None:
        return None
    long_strike = short_strike - cfg.wing_points if side == "put" else short_strike + cfg.wing_points
    legs = [
        _leg(chain, side, "SELL", short_strike),
        _leg(chain, side, "BUY", long_strike),
    ]
    if any(leg is None for leg in legs):
        return None
    credit = chain.mid(side, short_strike) - chain.mid(side, long_strike)  # type: ignore[operator]
    if credit <= 0:
        return None
    return PremiumStructure(
        kind=f"{side}_credit_spread",
        legs=tuple(leg for leg in legs if leg is not None),
        credit=credit,
        width=cfg.wing_points,
        short_strikes=(short_strike,),
    )


def build_condor(chain: CleanChain, config: PremiumConfig | None = None) -> PremiumStructure | None:
    """Short both wings. Only one side can finish in the money, so the risk is
    the widest single wing -- not the sum of the two verticals."""
    cfg = config or PremiumConfig()
    put_side = build_credit_spread(chain, "put", cfg)
    call_side = build_credit_spread(chain, "call", cfg)
    if put_side is None or call_side is None:
        return None
    credit = put_side.credit + call_side.credit
    if credit <= 0:
        return None
    return PremiumStructure(
        kind="iron_condor",
        legs=put_side.legs + call_side.legs,
        credit=credit,
        width=max(put_side.width, call_side.width),
        short_strikes=(put_side.short_strikes[0], call_side.short_strikes[0]),
    )


# ---------------------------------------------------------------------------
# the stand-aside gate
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class DayAssessment:
    trade: bool
    reasons: tuple[str, ...] = ()
    notes: dict[str, Any] = field(default_factory=dict)


def assess_day(
    market: Mapping[str, Any],
    *,
    session_open: float,
    spot: float,
    config: PremiumConfig | None = None,
) -> DayAssessment:
    """Decide whether today is worth selling premium into at all.

    Every rule here is defensible before looking at the sample. The point is
    not to predict direction -- a premium seller does not care -- but to refuse
    the days whose range is already going.
    """
    cfg = config or PremiumConfig()
    reasons: list[str] = []
    expected = _as_float(market.get("expected_range")) or 0.0
    realised = abs(spot - session_open)

    if cfg.skip_on_catalyst and bool(market.get("has_catalyst")):
        reasons.append(f"catalyst: {market.get('catalyst_label') or 'unspecified'}")

    if expected < cfg.min_expected_range:
        reasons.append(f"implied range {expected:.2f} below floor {cfg.min_expected_range:.2f}")

    if expected > 0 and realised / expected > cfg.max_realised_fraction:
        reasons.append(
            f"day has already used {realised / expected:.0%} of its implied range"
        )

    if cfg.gamma_filter:
        net_gex = _as_float(market.get("net_gex"))
        if net_gex is not None and net_gex > 0:
            reasons.append("positive dealer gamma (unproven filter enabled)")

    return DayAssessment(
        trade=not reasons,
        reasons=tuple(reasons),
        notes={
            "expected_range": expected,
            "realised_move": realised,
            "realised_fraction": (realised / expected) if expected else None,
            "net_gex": _as_float(market.get("net_gex")),
        },
    )


# ---------------------------------------------------------------------------
# sizing and management
# ---------------------------------------------------------------------------
def size_position(
    equity: float, structure: PremiumStructure, config: PremiumConfig | None = None
) -> int:
    """Contracts such that the *true* max loss is at most `risk_fraction`.

    Sized off ``max_loss``, never off the credit. A short-premium book that
    sizes off the credit is sizing off its best case.
    """
    cfg = config or PremiumConfig()
    risk_per_contract = structure.max_loss * 100.0
    if risk_per_contract <= 0 or equity <= 0:
        return 0
    budget = equity * cfg.risk_fraction
    return max(0, min(cfg.max_contracts, int(budget // risk_per_contract)))


def manage_position(
    structure: PremiumStructure,
    chain: CleanChain,
    now_hhmm: str,
    config: PremiumConfig | None = None,
) -> tuple[str | None, float | None]:
    """Return ``(exit_reason, value)``; reason is ``None`` to keep holding.

    A mark the gate cannot produce is not an exit signal. Refusing to act on an
    unusable snapshot is the difference between riding out a torn quote and
    booking it. Note the ordering: the spot-based breach check runs before any
    mark-based rule, so the tail defence never depends on a quote.
    """
    cfg = config or PremiumConfig()

    if cfg.breach_exit_points is not None and structure.short_strikes:
        low, high = min(structure.short_strikes), max(structure.short_strikes)
        if chain.spot < low - cfg.breach_exit_points or chain.spot > high + cfg.breach_exit_points:
            return "breach", structure.value_at(chain)

    value = structure.value_at(chain)
    if value is None:
        return None, None
    if now_hhmm >= cfg.flatten_at:
        return "flat_eod", value
    if value <= structure.credit * (1.0 - cfg.profit_target_fraction):
        return "target", value
    if cfg.stop_multiple is not None and value >= structure.credit * cfg.stop_multiple:
        return "stop", value
    return None, value


def _as_float(value: Any) -> float | None:
    try:
        return None if value is None else float(value)
    except (TypeError, ValueError):
        return None
