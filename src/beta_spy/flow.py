from __future__ import annotations

from dataclasses import dataclass

from .models import FlowFeatures, QuoteTop, TradePrint


@dataclass
class FlowAccumulator:
    """Top-of-book/tape microstructure accumulator for one symbol and one window.

    Aggressor classification is intentionally conservative: prints at/above ask are buys,
    prints at/below bid are sells, and inside-spread prints use a tick rule only when a prior
    trade exists. Nothing here claims Level-II depth.
    """

    buy_volume: float = 0.0
    sell_volume: float = 0.0
    neutral_volume: float = 0.0
    trade_count: int = 0
    quote_updates: int = 0
    spread_bps_sum: float = 0.0
    spread_samples: int = 0
    quote_imbalance_sum: float = 0.0
    quote_imbalance_samples: int = 0
    first_trade_price: float | None = None
    last_trade_price: float | None = None
    previous_trade_price: float | None = None
    total_notional: float = 0.0
    bid: float | None = None
    ask: float | None = None
    bid_size: float | None = None
    ask_size: float | None = None

    def on_quote(self, quote: QuoteTop) -> None:
        self.bid = quote.bid if quote.bid > 0 else self.bid
        self.ask = quote.ask if quote.ask > 0 else self.ask
        self.bid_size = quote.bid_size
        self.ask_size = quote.ask_size
        self.quote_updates += 1
        if quote.bid > 0 and quote.ask >= quote.bid:
            mid = (quote.bid + quote.ask) / 2.0
            if mid > 0:
                self.spread_bps_sum += (quote.ask - quote.bid) / mid * 10_000.0
                self.spread_samples += 1
        if quote.bid_size is not None and quote.ask_size is not None:
            total = quote.bid_size + quote.ask_size
            if total > 0:
                self.quote_imbalance_sum += (quote.bid_size - quote.ask_size) / total
                self.quote_imbalance_samples += 1

    def on_trade(self, trade: TradePrint) -> None:
        if trade.size <= 0 or trade.price <= 0:
            return
        bid = trade.bid if trade.bid is not None else self.bid
        ask = trade.ask if trade.ask is not None else self.ask
        side = 0
        if ask is not None and ask > 0 and trade.price >= ask:
            side = 1
        elif bid is not None and bid > 0 and trade.price <= bid:
            side = -1
        elif self.last_trade_price is not None:
            if trade.price > self.last_trade_price:
                side = 1
            elif trade.price < self.last_trade_price:
                side = -1
        if side > 0:
            self.buy_volume += trade.size
        elif side < 0:
            self.sell_volume += trade.size
        else:
            self.neutral_volume += trade.size
        self.trade_count += 1
        self.total_notional += trade.price * trade.size
        if self.first_trade_price is None:
            self.first_trade_price = trade.price
        self.previous_trade_price = self.last_trade_price
        self.last_trade_price = trade.price

    def snapshot(self, *, window_seconds: float = 60.0) -> FlowFeatures:
        directional = self.buy_volume + self.sell_volume
        total = directional + self.neutral_volume
        ofi = (self.buy_volume - self.sell_volume) / directional if directional > 0 else None
        qi = (
            self.quote_imbalance_sum / self.quote_imbalance_samples
            if self.quote_imbalance_samples > 0
            else None
        )
        spread = self.spread_bps_sum / self.spread_samples if self.spread_samples > 0 else None
        avg_size = total / self.trade_count if self.trade_count > 0 else None
        impact = None
        absorption = None
        if self.first_trade_price and self.last_trade_price and directional > 0:
            move_bps = (self.last_trade_price / self.first_trade_price - 1.0) * 10_000.0
            impact = move_bps / max(directional / 10_000.0, 1e-9)
            if ofi is not None:
                expected_sign = 1.0 if ofi >= 0 else -1.0
                signed_move = expected_sign * move_bps
                absorption = max(0.0, min(1.0, abs(ofi) * (1.0 - min(max(signed_move, 0.0) / 5.0, 1.0))))
        return FlowFeatures(
            buy_volume=self.buy_volume,
            sell_volume=self.sell_volume,
            neutral_volume=self.neutral_volume,
            order_flow_imbalance=ofi,
            quote_imbalance=qi,
            average_spread_bps=spread,
            trade_intensity=self.trade_count / max(window_seconds, 1.0),
            average_trade_size=avg_size,
            price_impact_bps_per_10k=impact,
            absorption=absorption,
            quote_updates=self.quote_updates,
            trades=self.trade_count,
        )

    def reset(self) -> None:
        self.buy_volume = 0.0
        self.sell_volume = 0.0
        self.neutral_volume = 0.0
        self.trade_count = 0
        self.quote_updates = 0
        self.spread_bps_sum = 0.0
        self.spread_samples = 0
        self.quote_imbalance_sum = 0.0
        self.quote_imbalance_samples = 0
        self.first_trade_price = None
        self.last_trade_price = None
        self.total_notional = 0.0
