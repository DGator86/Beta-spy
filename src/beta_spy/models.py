from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import datetime
from typing import Any


@dataclass(frozen=True)
class HoldingMeta:
    symbol: str
    sector: str
    weight: float
    name: str = ""


@dataclass(frozen=True)
class TradePrint:
    symbol: str
    timestamp: datetime
    price: float
    size: float
    bid: float | None = None
    ask: float | None = None
    sequence: int | None = None


@dataclass(frozen=True)
class QuoteTop:
    symbol: str
    timestamp: datetime
    bid: float
    ask: float
    bid_size: float | None = None
    ask_size: float | None = None


@dataclass(frozen=True)
class MinuteBar:
    symbol: str
    timestamp: datetime
    open: float
    high: float
    low: float
    close: float
    volume: float
    trade_count: int = 0
    vwap: float | None = None


@dataclass(frozen=True)
class FlowFeatures:
    buy_volume: float = 0.0
    sell_volume: float = 0.0
    neutral_volume: float = 0.0
    order_flow_imbalance: float | None = None
    quote_imbalance: float | None = None
    average_spread_bps: float | None = None
    trade_intensity: float = 0.0
    average_trade_size: float | None = None
    price_impact_bps_per_10k: float | None = None
    absorption: float | None = None
    quote_updates: int = 0
    trades: int = 0
    signed_delta: float | None = None
    signed_aggressive_volume: float | None = None
    directional_volume: float | None = None
    price_displacement_bps: float | None = None
    flow_to_displacement: float | None = None
    displacement_per_10k_volume: float | None = None
    buy_absorption: float | None = None
    sell_absorption: float | None = None
    initiative_buy_efficiency: float | None = None
    initiative_sell_efficiency: float | None = None
    best_bid_size_persistence: float | None = None
    best_ask_size_persistence: float | None = None
    best_bid_replenishment: float | None = None
    best_ask_replenishment: float | None = None
    best_bid_withdrawal_rate: float | None = None
    best_ask_withdrawal_rate: float | None = None
    quote_imbalance_velocity: float | None = None
    quote_imbalance_persistence: float | None = None


@dataclass(frozen=True)
class StructureFeatures:
    swing_prominence_3: float | None = None
    swing_prominence_5: float | None = None
    swing_prominence_9: float | None = None
    swing_prominence_17: float | None = None
    swing_prominence_33: float | None = None
    distance_to_last_swing_high_atr: float | None = None
    distance_to_last_swing_low_atr: float | None = None
    structure_state: float | None = None
    structure_score: float | None = None
    structure_break_strength: float | None = None
    failed_break_strength: float | None = None
    nfs_direction: float | None = None
    nfs_break_atr: float | None = None
    nfs_duration: float | None = None
    nfs_displacement_efficiency: float | None = None
    nfs_relative_volume: float | None = None
    upper_wick_ratio: float | None = None
    lower_wick_ratio: float | None = None
    body_ratio: float | None = None
    close_location: float | None = None
    effort_result_ratio: float | None = None
    displacement_efficiency: float | None = None
    sweep_high_score: float | None = None
    sweep_low_score: float | None = None
    acceptance_above_score: float | None = None
    acceptance_below_score: float | None = None
    last_swing_sign: float | None = None


@dataclass(frozen=True)
class AuctionFeatures:
    cvd_session: float | None = None
    cvd_5m: float | None = None
    cvd_15m: float | None = None
    cvd_slope_5m: float | None = None
    cvd_slope_15m: float | None = None
    cvd_zscore: float | None = None
    price_cvd_divergence_5m: float | None = None
    price_cvd_divergence_15m: float | None = None
    session_poc: float | None = None
    session_vah: float | None = None
    session_val: float | None = None
    distance_to_poc: float | None = None
    distance_to_vah: float | None = None
    distance_to_val: float | None = None
    nearest_hvn_distance: float | None = None
    nearest_lvn_distance: float | None = None
    poc_migration: float | None = None
    value_area_width: float | None = None
    inside_value: float | None = None
    above_value: float | None = None
    below_value: float | None = None
    max_positive_delta_price: float | None = None
    max_negative_delta_price: float | None = None
    stacked_buy_imbalance_count: int = 0
    stacked_sell_imbalance_count: int = 0
    local_absorption_high: float | None = None
    local_absorption_low: float | None = None


@dataclass(frozen=True)
class SymbolFeatures:
    symbol: str
    timestamp: datetime
    sector: str
    weight: float
    close: float
    return_1m: float | None
    return_5m: float | None
    return_15m: float | None
    vwap: float | None
    vwap_distance_bps: float | None
    ema8: float | None
    ema21: float | None
    ema8_slope_bps: float | None
    ema21_slope_bps: float | None
    rsi14: float | None
    atr14_bps: float | None
    realized_vol20_bps: float | None
    relative_volume20: float | None
    range_expansion: float | None
    flow: FlowFeatures = field(default_factory=FlowFeatures)
    structure: StructureFeatures = field(default_factory=StructureFeatures)
    auction: AuctionFeatures = field(default_factory=AuctionFeatures)

    @property
    def above_vwap(self) -> bool | None:
        if self.vwap is None:
            return None
        return self.close > self.vwap

    @property
    def ema_bullish(self) -> bool | None:
        if self.ema8 is None or self.ema21 is None:
            return None
        return self.ema8 > self.ema21


@dataclass(frozen=True)
class SectorFactors:
    sector: str
    count: int
    covered_weight: float
    trend: float | None
    momentum: float | None
    volume: float | None
    flow: float | None
    volatility: float | None
    participation: float | None


@dataclass(frozen=True)
class MarketFactors:
    timestamp: datetime
    symbol_count: int
    expected_symbol_count: int
    coverage_ratio: float
    covered_weight: float
    trend_ew: float | None
    trend_weighted: float | None
    momentum_ew: float | None
    momentum_weighted: float | None
    volume_ew: float | None
    volume_weighted: float | None
    flow_ew: float | None
    flow_weighted: float | None
    volatility_ew: float | None
    volatility_weighted: float | None
    pct_above_vwap: float | None
    pct_ema_bullish: float | None
    pct_positive_5m: float | None
    pct_buy_flow: float | None
    participation: float | None
    concentration: float | None
    breadth_acceleration: float | None
    spy_return_1m: float | None
    spy_return_5m: float | None
    spy_vwap_distance_bps: float | None
    spy_flow: float | None
    spy_quote_imbalance: float | None
    spy_spread_bps: float | None
    structure_ew: float | None = None
    structure_weighted: float | None = None
    pct_structure_bullish: float | None = None
    pct_structure_bearish: float | None = None
    structure_divergence: float | None = None
    absorption_ew: float | None = None
    absorption_weighted: float | None = None
    cvd_ew: float | None = None
    cvd_weighted: float | None = None
    pct_positive_cvd: float | None = None
    pct_buy_absorption: float | None = None
    pct_sell_absorption: float | None = None
    pct_bullish_sweep: float | None = None
    pct_bearish_sweep: float | None = None
    sweep_ew: float | None = None
    sweep_weighted: float | None = None
    acceptance_ew: float | None = None
    acceptance_weighted: float | None = None
    initiative_ew: float | None = None
    initiative_weighted: float | None = None
    pct_breaking_highs: float | None = None
    pct_breaking_lows: float | None = None
    spy_cvd: float | None = None
    spy_cvd_divergence: float | None = None
    spy_poc_distance: float | None = None
    spy_value_location: float | None = None
    sectors: tuple[SectorFactors, ...] = ()

    def feature_dict(self) -> dict[str, float]:
        values = {
            "coverage_ratio": self.coverage_ratio,
            "covered_weight": self.covered_weight,
            "trend_ew": self.trend_ew,
            "trend_weighted": self.trend_weighted,
            "momentum_ew": self.momentum_ew,
            "momentum_weighted": self.momentum_weighted,
            "volume_ew": self.volume_ew,
            "volume_weighted": self.volume_weighted,
            "flow_ew": self.flow_ew,
            "flow_weighted": self.flow_weighted,
            "volatility_ew": self.volatility_ew,
            "volatility_weighted": self.volatility_weighted,
            "pct_above_vwap": self.pct_above_vwap,
            "pct_ema_bullish": self.pct_ema_bullish,
            "pct_positive_5m": self.pct_positive_5m,
            "pct_buy_flow": self.pct_buy_flow,
            "participation": self.participation,
            "concentration": self.concentration,
            "breadth_acceleration": self.breadth_acceleration,
            "spy_return_1m": self.spy_return_1m,
            "spy_return_5m": self.spy_return_5m,
            "spy_vwap_distance_bps": self.spy_vwap_distance_bps,
            "spy_flow": self.spy_flow,
            "spy_quote_imbalance": self.spy_quote_imbalance,
            "spy_spread_bps": self.spy_spread_bps,
            "structure_ew": self.structure_ew,
            "structure_weighted": self.structure_weighted,
            "pct_structure_bullish": self.pct_structure_bullish,
            "pct_structure_bearish": self.pct_structure_bearish,
            "structure_divergence": self.structure_divergence,
            "sweep_ew": self.sweep_ew,
            "sweep_weighted": self.sweep_weighted,
            "acceptance_ew": self.acceptance_ew,
            "acceptance_weighted": self.acceptance_weighted,
            "initiative_ew": self.initiative_ew,
            "initiative_weighted": self.initiative_weighted,
            "pct_breaking_highs": self.pct_breaking_highs,
            "pct_breaking_lows": self.pct_breaking_lows,
            "absorption_ew": self.absorption_ew,
            "absorption_weighted": self.absorption_weighted,
            "cvd_ew": self.cvd_ew,
            "cvd_weighted": self.cvd_weighted,
            "pct_positive_cvd": self.pct_positive_cvd,
            "pct_buy_absorption": self.pct_buy_absorption,
            "pct_sell_absorption": self.pct_sell_absorption,
            "spy_cvd": self.spy_cvd,
            "spy_cvd_divergence": self.spy_cvd_divergence,
            "spy_poc_distance": self.spy_poc_distance,
            "spy_value_location": self.spy_value_location,
        }
        return {key: float(value) if value is not None else 0.0 for key, value in values.items()}


@dataclass(frozen=True)
class HorizonForecast:
    horizon_minutes: int
    probability_up: float
    expected_return_bps: float
    confidence: float
    model_ready: bool
    sample_count: int

    @property
    def direction(self) -> int:
        if self.probability_up >= 0.5:
            return 1
        return -1


@dataclass(frozen=True)
class Decision:
    timestamp: datetime
    action: str
    direction: str
    confidence: float
    score: float
    primary_horizon: int
    gates: dict[str, bool]
    reasons: tuple[str, ...]
    structure: str | None = None
    # Risk-budget multiplier for the options layer: 0.5x at the decision
    # threshold, scaling with directional edge, capped at 2x. Validated
    # out-of-sample to roughly double direction-adjusted P&L versus flat
    # sizing on the same trade set.
    risk_multiplier: float = 1.0
    # 15-minute forecast default; session-trend overlay may extend this to the
    # 15:50 ET flatten so a grind is not chopped into friction-losing scalps.
    hold_minutes: float = 15.0


@dataclass(frozen=True)
class EngineSnapshot:
    timestamp: datetime
    factors: MarketFactors
    forecasts: tuple[HorizonForecast, ...]
    decision: Decision
    symbols: tuple[SymbolFeatures, ...] = ()

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)
