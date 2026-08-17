from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime
from math import sqrt

from .models import MarketFactors, SectorFactors, SymbolFeatures


def _clip(value: float, lower: float = -1.0, upper: float = 1.0) -> float:
    return max(lower, min(upper, value))


def _mean(values: list[float]) -> float | None:
    return sum(values) / len(values) if values else None


def _weighted(values: list[tuple[float, float]]) -> float | None:
    denominator = sum(weight for _, weight in values if weight > 0)
    if denominator <= 0:
        return None
    return sum(value * weight for value, weight in values if weight > 0) / denominator


def _fraction(values: list[bool | None]) -> float | None:
    known = [value for value in values if value is not None]
    if not known:
        return None
    return sum(bool(value) for value in known) / len(known)


def _trend_score(feature: SymbolFeatures) -> float | None:
    signals: list[float] = []
    if feature.vwap_distance_bps is not None:
        signals.append(_clip(feature.vwap_distance_bps / 15.0))
    if feature.ema8 is not None and feature.ema21 is not None and feature.close > 0:
        signals.append(_clip((feature.ema8 / feature.ema21 - 1.0) * 10_000.0 / 10.0))
    if feature.ema8_slope_bps is not None:
        signals.append(_clip(feature.ema8_slope_bps / 2.0))
    if feature.return_5m is not None:
        signals.append(_clip(feature.return_5m * 10_000.0 / 15.0))
    return _mean(signals)


def _momentum_score(feature: SymbolFeatures) -> float | None:
    signals: list[float] = []
    if feature.return_1m is not None:
        signals.append(_clip(feature.return_1m * 10_000.0 / 8.0))
    if feature.return_5m is not None:
        signals.append(_clip(feature.return_5m * 10_000.0 / 20.0))
    if feature.return_15m is not None:
        signals.append(_clip(feature.return_15m * 10_000.0 / 35.0))
    if feature.rsi14 is not None:
        signals.append(_clip((feature.rsi14 - 50.0) / 20.0))
    return _mean(signals)


def _volume_score(feature: SymbolFeatures) -> float | None:
    signals: list[float] = []
    if feature.relative_volume20 is not None:
        direction = 1.0 if (feature.return_1m or 0.0) >= 0 else -1.0
        signals.append(direction * _clip((feature.relative_volume20 - 1.0) / 1.5))
    if feature.range_expansion is not None:
        direction = 1.0 if (feature.return_1m or 0.0) >= 0 else -1.0
        signals.append(direction * _clip((feature.range_expansion - 1.0) / 1.5))
    return _mean(signals)


def _flow_score(feature: SymbolFeatures) -> float | None:
    signals: list[float] = []
    if feature.flow.order_flow_imbalance is not None:
        signals.append(_clip(feature.flow.order_flow_imbalance))
    if feature.flow.quote_imbalance is not None:
        signals.append(_clip(feature.flow.quote_imbalance))
    if feature.flow.price_impact_bps_per_10k is not None:
        signals.append(_clip(feature.flow.price_impact_bps_per_10k / 3.0))
    if feature.flow.absorption is not None and feature.flow.order_flow_imbalance is not None:
        signals.append(-feature.flow.absorption * (1.0 if feature.flow.order_flow_imbalance >= 0 else -1.0))
    return _mean(signals)


def _volatility_score(feature: SymbolFeatures) -> float | None:
    signals: list[float] = []
    if feature.range_expansion is not None:
        signals.append(_clip((feature.range_expansion - 1.0) / 1.5))
    if feature.atr14_bps is not None and feature.realized_vol20_bps is not None:
        denominator = max(feature.realized_vol20_bps, 1e-9)
        signals.append(_clip(feature.atr14_bps / denominator - 1.0))
    return _mean(signals)


@dataclass
class BreadthAggregator:
    previous_trend_ew: float | None = None

    def aggregate(
        self,
        features: list[SymbolFeatures],
        *,
        timestamp: datetime,
        expected_symbol_count: int,
    ) -> MarketFactors:
        constituents = [item for item in features if item.symbol != "SPY"]
        spy = next((item for item in features if item.symbol == "SPY"), None)
        covered_weight = sum(max(item.weight, 0.0) for item in constituents)
        total_count = len(constituents)
        coverage = total_count / expected_symbol_count if expected_symbol_count > 0 else 0.0

        trend_pairs: list[tuple[float, float]] = []
        momentum_pairs: list[tuple[float, float]] = []
        volume_pairs: list[tuple[float, float]] = []
        flow_pairs: list[tuple[float, float]] = []
        vol_pairs: list[tuple[float, float]] = []
        trend_values: list[float] = []
        momentum_values: list[float] = []
        volume_values: list[float] = []
        flow_values: list[float] = []
        vol_values: list[float] = []
        sector_map: dict[str, list[SymbolFeatures]] = defaultdict(list)
        for item in constituents:
            sector_map[item.sector].append(item)
            for score, target, pairs in [
                (_trend_score(item), trend_values, trend_pairs),
                (_momentum_score(item), momentum_values, momentum_pairs),
                (_volume_score(item), volume_values, volume_pairs),
                (_flow_score(item), flow_values, flow_pairs),
                (_volatility_score(item), vol_values, vol_pairs),
            ]:
                if score is not None:
                    target.append(score)
                    pairs.append((score, item.weight))

        pct_above = _fraction([item.above_vwap for item in constituents])
        pct_ema = _fraction([item.ema_bullish for item in constituents])
        pct_pos5 = _fraction([None if item.return_5m is None else item.return_5m > 0 for item in constituents])
        pct_buy = _fraction(
            [
                None if item.flow.order_flow_imbalance is None else item.flow.order_flow_imbalance > 0
                for item in constituents
            ]
        )
        def _struct(item: SymbolFeatures) -> float | None:
            return item.structure.structure_score if item.structure.structure_score is not None else item.structure.structure_state

        structure_values = [score for item in constituents if (score := _struct(item)) is not None]
        structure_pairs = [(score, item.weight) for item in constituents if (score := _struct(item)) is not None]
        structure_ew = _mean(structure_values)
        structure_weighted = _weighted(structure_pairs)
        structure_divergence = None
        if structure_ew is not None and structure_weighted is not None:
            structure_divergence = structure_ew - structure_weighted
        pct_struct_bull = _fraction(
            [None if _struct(item) is None else _struct(item) > 0 for item in constituents]
        )
        pct_struct_bear = _fraction(
            [None if _struct(item) is None else _struct(item) < 0 for item in constituents]
        )
        absorption_values = [item.flow.absorption for item in constituents if item.flow.absorption is not None]
        absorption_pairs = [
            (item.flow.absorption, item.weight) for item in constituents if item.flow.absorption is not None
        ]
        cvd_values = [item.auction.cvd_session for item in constituents if item.auction.cvd_session is not None]
        cvd_pairs = [(item.auction.cvd_session, item.weight) for item in constituents if item.auction.cvd_session is not None]
        sweep_values = []
        sweep_pairs = []
        accept_values = []
        accept_pairs = []
        init_values = []
        init_pairs = []
        for item in constituents:
            sweep = None
            if item.structure.sweep_high_score is not None or item.structure.sweep_low_score is not None:
                sweep = (item.structure.sweep_low_score or 0.0) - (item.structure.sweep_high_score or 0.0)
                sweep_values.append(sweep)
                sweep_pairs.append((sweep, item.weight))
            accept = None
            if item.structure.acceptance_above_score is not None or item.structure.acceptance_below_score is not None:
                accept = (item.structure.acceptance_above_score or 0.0) - (item.structure.acceptance_below_score or 0.0)
                accept_values.append(accept)
                accept_pairs.append((accept, item.weight))
            init = None
            buy_i = item.flow.initiative_buy_efficiency
            sell_i = item.flow.initiative_sell_efficiency
            if buy_i is not None or sell_i is not None:
                init = (buy_i or 0.0) - (sell_i or 0.0)
                init_values.append(init)
                init_pairs.append((init, item.weight))
        participation_parts = [value for value in (pct_above, pct_ema, pct_pos5, pct_buy) if value is not None]
        participation = _mean([2.0 * value - 1.0 for value in participation_parts])

        contributions = [abs((item.return_1m or 0.0) * item.weight) for item in constituents]
        contribution_total = sum(contributions)
        concentration = None
        if contribution_total > 0:
            shares = [value / contribution_total for value in contributions]
            hhi = sum(share * share for share in shares)
            baseline = 1.0 / max(len(shares), 1)
            concentration = _clip((sqrt(hhi) - sqrt(baseline)) / max(1.0 - sqrt(baseline), 1e-9), 0.0, 1.0)

        trend_ew = _mean(trend_values)
        breadth_acceleration = None
        if trend_ew is not None and self.previous_trend_ew is not None:
            breadth_acceleration = trend_ew - self.previous_trend_ew
        if trend_ew is not None:
            self.previous_trend_ew = trend_ew

        sectors: list[SectorFactors] = []
        for sector, items in sorted(sector_map.items()):
            sectors.append(
                SectorFactors(
                    sector=sector,
                    count=len(items),
                    covered_weight=sum(max(item.weight, 0.0) for item in items),
                    trend=_mean([score for item in items if (score := _trend_score(item)) is not None]),
                    momentum=_mean([score for item in items if (score := _momentum_score(item)) is not None]),
                    volume=_mean([score for item in items if (score := _volume_score(item)) is not None]),
                    flow=_mean([score for item in items if (score := _flow_score(item)) is not None]),
                    volatility=_mean([score for item in items if (score := _volatility_score(item)) is not None]),
                    participation=_mean(
                        [
                            1.0 if (item.return_5m or 0.0) > 0 else -1.0
                            for item in items
                            if item.return_5m is not None
                        ]
                    ),
                )
            )

        return MarketFactors(
            timestamp=timestamp,
            symbol_count=total_count,
            expected_symbol_count=expected_symbol_count,
            coverage_ratio=coverage,
            covered_weight=covered_weight,
            trend_ew=trend_ew,
            trend_weighted=_weighted(trend_pairs),
            momentum_ew=_mean(momentum_values),
            momentum_weighted=_weighted(momentum_pairs),
            volume_ew=_mean(volume_values),
            volume_weighted=_weighted(volume_pairs),
            flow_ew=_mean(flow_values),
            flow_weighted=_weighted(flow_pairs),
            volatility_ew=_mean(vol_values),
            volatility_weighted=_weighted(vol_pairs),
            pct_above_vwap=pct_above,
            pct_ema_bullish=pct_ema,
            pct_positive_5m=pct_pos5,
            pct_buy_flow=pct_buy,
            participation=participation,
            concentration=concentration,
            breadth_acceleration=breadth_acceleration,
            spy_return_1m=spy.return_1m if spy else None,
            spy_return_5m=spy.return_5m if spy else None,
            spy_vwap_distance_bps=spy.vwap_distance_bps if spy else None,
            spy_flow=spy.flow.order_flow_imbalance if spy else None,
            spy_quote_imbalance=spy.flow.quote_imbalance if spy else None,
            spy_spread_bps=spy.flow.average_spread_bps if spy else None,
            structure_ew=structure_ew,
            structure_weighted=structure_weighted,
            pct_structure_bullish=pct_struct_bull,
            pct_structure_bearish=pct_struct_bear,
            structure_divergence=structure_divergence,
            absorption_ew=_mean(absorption_values),
            absorption_weighted=_weighted(absorption_pairs),
            cvd_ew=_mean(cvd_values),
            cvd_weighted=_weighted(cvd_pairs),
            pct_positive_cvd=_fraction(
                [None if item.auction.cvd_session is None else item.auction.cvd_session > 0 for item in constituents]
            ),
            pct_buy_absorption=_fraction(
                [None if item.flow.buy_absorption is None else item.flow.buy_absorption > 0.5 for item in constituents]
            ),
            pct_sell_absorption=_fraction(
                [None if item.flow.sell_absorption is None else item.flow.sell_absorption > 0.5 for item in constituents]
            ),
            pct_bullish_sweep=_fraction(
                [None if item.structure.sweep_low_score is None else item.structure.sweep_low_score > 0 for item in constituents]
            ),
            pct_bearish_sweep=_fraction(
                [None if item.structure.sweep_high_score is None else item.structure.sweep_high_score > 0 for item in constituents]
            ),
            sweep_ew=_mean(sweep_values),
            sweep_weighted=_weighted(sweep_pairs),
            acceptance_ew=_mean(accept_values),
            acceptance_weighted=_weighted(accept_pairs),
            initiative_ew=_mean(init_values),
            initiative_weighted=_weighted(init_pairs),
            pct_breaking_highs=_fraction(
                [
                    None if item.structure.structure_break_strength is None else item.structure.structure_break_strength > 0
                    for item in constituents
                ]
            ),
            pct_breaking_lows=_fraction(
                [
                    None if item.structure.structure_break_strength is None else item.structure.structure_break_strength < 0
                    for item in constituents
                ]
            ),
            spy_cvd=spy.auction.cvd_session if spy else None,
            spy_cvd_divergence=spy.auction.price_cvd_divergence_5m if spy else None,
            spy_poc_distance=spy.auction.distance_to_poc if spy else None,
            spy_value_location=(
                1.0
                if spy and spy.auction.above_value
                else -1.0
                if spy and spy.auction.below_value
                else 0.0
                if spy
                else None
            ),
            sectors=tuple(sectors),
        )
