"""Named forecast vectors. Production uses baseline only.

Experimental families are calculated and persisted. They enter a challenger
vector here, not FEATURE_NAMES, until they survive incremental OOS tests.
"""

from __future__ import annotations

BASELINE_FEATURE_NAMES = (
    "coverage_ratio",
    "covered_weight",
    "trend_ew",
    "trend_weighted",
    "momentum_ew",
    "momentum_weighted",
    "volume_ew",
    "volume_weighted",
    "flow_ew",
    "flow_weighted",
    "volatility_ew",
    "volatility_weighted",
    "pct_above_vwap",
    "pct_ema_bullish",
    "pct_positive_5m",
    "pct_buy_flow",
    "participation",
    "concentration",
    "breadth_acceleration",
    "spy_return_1m",
    "spy_return_5m",
    "spy_vwap_distance_bps",
    "spy_flow",
    "spy_quote_imbalance",
    "spy_spread_bps",
)

EXPERIMENTAL_STRUCTURE_NAMES = (
    "structure_ew",
    "structure_weighted",
    "pct_structure_bullish",
    "pct_structure_bearish",
    "structure_divergence",
    "sweep_ew",
    "sweep_weighted",
    "acceptance_ew",
    "acceptance_weighted",
    "pct_breaking_highs",
    "pct_breaking_lows",
)

EXPERIMENTAL_FLOW_NAMES = (
    "absorption_ew",
    "absorption_weighted",
    "initiative_ew",
    "initiative_weighted",
    "cvd_ew",
    "cvd_weighted",
    "pct_positive_cvd",
    "pct_buy_absorption",
    "pct_sell_absorption",
)

EXPERIMENTAL_AUCTION_NAMES = (
    "spy_cvd",
    "spy_cvd_divergence",
    "spy_poc_distance",
    "spy_value_location",
)

FEATURE_SETS: dict[str, tuple[str, ...]] = {
    "baseline": BASELINE_FEATURE_NAMES,
    "structure_v1": BASELINE_FEATURE_NAMES + EXPERIMENTAL_STRUCTURE_NAMES,
    "flow_v2": BASELINE_FEATURE_NAMES + EXPERIMENTAL_FLOW_NAMES,
    "auction_v1": BASELINE_FEATURE_NAMES + EXPERIMENTAL_AUCTION_NAMES,
    "full_v1": (
        BASELINE_FEATURE_NAMES
        + EXPERIMENTAL_STRUCTURE_NAMES
        + EXPERIMENTAL_FLOW_NAMES
        + EXPERIMENTAL_AUCTION_NAMES
    ),
}


def names_for(feature_set: str) -> tuple[str, ...]:
    try:
        return FEATURE_SETS[feature_set]
    except KeyError as exc:
        raise KeyError(f"unknown feature set {feature_set!r}") from exc
