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


@dataclass(frozen=True)
class EngineSnapshot:
    timestamp: datetime
    factors: MarketFactors
    forecasts: tuple[HorizonForecast, ...]
    decision: Decision
    symbols: tuple[SymbolFeatures, ...] = ()

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)
