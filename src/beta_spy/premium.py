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
    "MonitorResult",
    "PositionState",
    "PremiumConfig",
    "PremiumStructure",
    "StructureLeg",
    "assess_day",
    "build_condor",
    "build_credit_spread",
    "decay_remaining",
    "expected_remaining_move",
    "manage_position",
    "monitor",
    "required_credit_ratio",
    "size_position",
    "strike_pressure",
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
    entry_before: str = "14:30"
    """No new entries after this.

    Late entries are not forbidden, they are *priced*: :func:`required_credit_ratio`
    demands proportionally more credit as decay runs out, so a 14:00 entry has
    to pay for the tail it is still taking. The hard cutoff exists only because
    below about an hour there is no decay left to sell at any price.
    """
    max_entries_per_day: int = 3
    """Re-enter after a close, up to this many positions per session.

    The previous lifecycle allowed exactly one. Measured over the sampled
    sessions it took profit at a median of 10:53 and then sat flat for 60% of
    the tradeable day. Taking half the credit in fifty minutes and then
    declining to sell the remaining five hours of decay is not a theta
    strategy.
    """
    reentry_cooldown_minutes: int = 15
    """Wait this long after a close before opening again. Stops a chopping
    market from cycling the book through fees."""

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
    flatten_at: str = "15:30"
    """Hard flat, deliberately before the close rather than at it.

    Settlement risk on a short 0DTE structure is unbounded in practice even
    when it is bounded on paper. The captured tape adds a second reason: in the
    15:50-16:00 window 16.4% of one-wide verticals were quoted beyond their own
    width, against 6.8% intraday. Flattening at 15:45 walks the book into the
    worst quotes of the day; 15:30 leaves before the lights go out.
    """
    session_end: str = "16:00"
    """Used only to measure how much decay is left, never to trade."""

    min_entry_pressure: float | None = 1.00
    """Refuse to open unless the short strikes start at least this many
    *remaining expected moves* away from spot.

    Delta targeting alone does not guarantee this. Measured at 10:00 across the
    sampled sessions, a 16-delta condor opened anywhere from 0.16 to 2.67
    remaining-moves from its shorts, median 1.20 -- so on some sessions the
    structure is already inside the day's expected travel at inception. Those
    are the entries worth declining.
    """
    pressure_exit_ratio: float | None = None
    """Close once the market has taken this fraction of the buffer the position
    opened with. **Disabled by default**, and the reason is the same one that
    disables :attr:`stop_multiple`.

    Relative rather than absolute is the right *shape* for the rule: an
    absolute floor near the median entry pressure closes immediately on
    sessions that open tight and never fires on ones that open wide. But
    measured over the sampled sessions, cutting a defined-risk short structure
    early is destructive at every threshold tried:

        exit ratio   positions   win rate   total
        off                 12      91.7%   +$196
        0.30                13      50.0%    -$62
        0.50                14      41.7%    -$72
        0.70                16      50.0%     -$8

    Sixteen of the risk exits taken across those runs closed for a loss and
    none for a gain. The wing is already the risk control; closing inside it
    converts a drawdown that would have recovered into a realised loss, and it
    does so on the days the market is moving fastest and quoting worst. Left
    available for callers who want it, off by default.
    """
    strike_pressure_exit: float | None = None
    """Absolute floor on strike pressure. Off by default in favour of
    :attr:`pressure_exit_ratio`; see the note there."""

    # --- sizing ---
    risk_fraction: float = 0.02
    """Fraction of equity at risk per trade, against *true* max loss."""
    max_contracts: int = 50
    min_survivable_losses: int = 4
    """Never size so large that fewer than this many max-loss trades in a row
    would leave the account unable to place a single contract.

    Fixed-fractional sizing alone does not guarantee this once the one-lot
    floor binds: at a small account ``risk_fraction`` can round up to a
    position the account cannot afford to lose twice.
    """
    late_entry_size_scale: bool = True
    """Scale size down with the decay remaining. A 14:00 entry takes the same
    tail risk as a 10:00 entry for a third of the theta, so it gets a third of
    the size."""

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
# the session clock
# ---------------------------------------------------------------------------
def _minutes(hhmm: str) -> int:
    return int(hhmm[:2]) * 60 + int(hhmm[3:5])


def decay_remaining(now_hhmm: str, config: PremiumConfig | None = None) -> float:
    """Fraction of the position's decay window still ahead, in ``[0, 1]``.

    Measured to :attr:`PremiumConfig.flatten_at` rather than to the bell,
    because decay after the flatten time is decay this book never collects.
    """
    cfg = config or PremiumConfig()
    start, end = _minutes(cfg.entry_after), _minutes(cfg.flatten_at)
    if end <= start:
        return 0.0
    left = end - _minutes(now_hhmm)
    return max(0.0, min(1.0, left / (end - start)))


def required_credit_ratio(now_hhmm: str, config: PremiumConfig | None = None) -> float:
    """Credit-to-risk a structure must pay to be worth opening *now*.

    The tail does not shrink as the day runs out, but the decay you are being
    paid for does. A 14:00 entry therefore has to pay materially more per unit
    of risk than a 10:00 entry to be the same trade.
    """
    cfg = config or PremiumConfig()
    remaining = max(decay_remaining(now_hhmm, cfg), 0.20)
    return cfg.min_credit_ratio / remaining


def expected_remaining_move(
    market: Mapping[str, Any], now_hhmm: str, config: PremiumConfig | None = None
) -> float | None:
    """Implied range still to come, by square-root-of-time scaling."""
    expected = _as_float(market.get("expected_range"))
    if expected is None or expected <= 0:
        return None
    return expected * decay_remaining(now_hhmm, config) ** 0.5


def strike_pressure(
    structure: PremiumStructure,
    spot: float,
    market: Mapping[str, Any],
    now_hhmm: str,
    config: PremiumConfig | None = None,
) -> float | None:
    """Distance from spot to the nearest short strike, in remaining moves.

    Above 1.0 the short strikes are further away than the market is expected to
    travel in the time left. Below 1.0 they are inside it. This is the number
    that should drive the risk exit, and it is computed entirely from spot and
    the implied range -- no option marks involved.
    """
    move = expected_remaining_move(market, now_hhmm, config)
    if move is None or move <= 0 or not structure.short_strikes:
        return None
    low, high = min(structure.short_strikes), max(structure.short_strikes)
    distance = min(spot - low, high - spot) if len(structure.short_strikes) > 1 else (
        min(abs(spot - s) for s in structure.short_strikes)
    )
    return distance / move


# ---------------------------------------------------------------------------
# sizing
# ---------------------------------------------------------------------------
def size_position(
    equity: float,
    structure: PremiumStructure,
    config: PremiumConfig | None = None,
    *,
    now_hhmm: str | None = None,
) -> int:
    """Contracts to trade, against *true* max loss and the account's survival.

    Three constraints, all binding:

    1. ``risk_fraction`` of equity, taken off ``max_loss`` and never off the
       credit -- sizing off the credit is sizing off the best case.
    2. Scaled by the decay still available, when ``late_entry_size_scale`` is
       set. Same tail, less theta, less size.
    3. Capped so ``min_survivable_losses`` consecutive max losses still leave
       the account able to place one contract. Fixed-fractional sizing does not
       give you this for free once the one-lot floor binds.
    """
    cfg = config or PremiumConfig()
    risk_per_contract = structure.max_loss * 100.0
    if risk_per_contract <= 0 or equity <= 0:
        return 0

    budget = equity * cfg.risk_fraction
    if cfg.late_entry_size_scale and now_hhmm is not None:
        budget *= decay_remaining(now_hhmm, cfg)

    contracts = int(budget // risk_per_contract)
    if cfg.min_survivable_losses > 0:
        survivable = int((equity / cfg.min_survivable_losses) // risk_per_contract)
        contracts = min(contracts, survivable)
    return max(0, min(cfg.max_contracts, contracts))


# ---------------------------------------------------------------------------
# monitoring
# ---------------------------------------------------------------------------
@dataclass
class PositionState:
    """Everything the monitor needs to remember between snapshots."""

    structure: PremiumStructure
    contracts: int
    entry_hhmm: str
    entry_pressure: float | None = None
    marks_seen: int = 0
    marks_unusable: int = 0
    consecutive_unusable: int = 0
    worst_value: float = 0.0
    last_value: float | None = None
    last_pressure: float | None = None


@dataclass(frozen=True)
class MonitorResult:
    exit_reason: str | None
    value: float | None
    pressure: float | None
    note: str = ""

    @property
    def hold(self) -> bool:
        return self.exit_reason is None


def monitor(
    state: PositionState,
    chain: CleanChain,
    market: Mapping[str, Any],
    now_hhmm: str,
    config: PremiumConfig | None = None,
) -> MonitorResult:
    """Decide whether to hold or close, and record what was seen.

    Ordering matters and is deliberate. The two rules that can fire without a
    usable option mark -- the clock and the spot-based risk exit -- are checked
    first, so neither the flatten nor the tail defence can be disabled by a bad
    chain. Only then does anything consult a price.
    """
    cfg = config or PremiumConfig()
    state.marks_seen += 1
    pressure = strike_pressure(state.structure, chain.spot, market, now_hhmm, cfg)
    state.last_pressure = pressure

    if now_hhmm >= cfg.flatten_at:
        return MonitorResult("flat_eod", state.structure.value_at(chain), pressure)

    if pressure is not None:
        if cfg.strike_pressure_exit is not None and pressure < cfg.strike_pressure_exit:
            return MonitorResult(
                "risk", state.structure.value_at(chain), pressure,
                f"spot within {pressure:.2f} remaining moves of a short strike",
            )
        if (
            cfg.pressure_exit_ratio is not None
            and state.entry_pressure is not None
            and state.entry_pressure > 0
            and pressure < state.entry_pressure * cfg.pressure_exit_ratio
        ):
            return MonitorResult(
                "risk", state.structure.value_at(chain), pressure,
                f"buffer down to {pressure / state.entry_pressure:.0%} of entry",
            )

    value = state.structure.value_at(chain)
    if value is None:
        state.marks_unusable += 1
        state.consecutive_unusable += 1
        return MonitorResult(None, None, pressure, "unusable mark, holding")

    state.consecutive_unusable = 0
    state.last_value = value
    state.worst_value = max(state.worst_value, value)

    if value <= state.structure.credit * (1.0 - cfg.profit_target_fraction):
        return MonitorResult("target", value, pressure)
    if cfg.stop_multiple is not None and value >= state.structure.credit * cfg.stop_multiple:
        return MonitorResult("stop", value, pressure)
    return MonitorResult(None, value, pressure)


def manage_position(
    structure: PremiumStructure,
    chain: CleanChain,
    now_hhmm: str,
    config: PremiumConfig | None = None,
) -> tuple[str | None, float | None]:
    """Stateless convenience wrapper over :func:`monitor`.

    Kept for callers that have no market snapshot and therefore no way to
    compute strike pressure; the risk exit is unavailable on this path.
    """
    cfg = config or PremiumConfig()
    if cfg.breach_exit_points is not None and structure.short_strikes:
        low, high = min(structure.short_strikes), max(structure.short_strikes)
        if chain.spot < low - cfg.breach_exit_points or chain.spot > high + cfg.breach_exit_points:
            return "breach", structure.value_at(chain)
    state = PositionState(structure=structure, contracts=1, entry_hhmm=now_hhmm)
    result = monitor(state, chain, {}, now_hhmm, cfg)
    return result.exit_reason, result.value


def _as_float(value: Any) -> float | None:
    try:
        return None if value is None else float(value)
    except (TypeError, ValueError):
        return None
