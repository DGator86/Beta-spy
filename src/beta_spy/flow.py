from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from datetime import datetime

from .models import FlowFeatures, QuoteTop, TradePrint


@dataclass
class FlowAccumulator:
    """Top-of-book/tape microstructure accumulator for one symbol.

    Window fields reset after each engine snapshot. Session CVD lives in
    auction.SessionCvd — this object is the current-minute window only.
    Aggressor classification stays conservative: at/above ask = buy,
    at/below bid = sell, inside-spread uses a tick rule. Not Level-II.
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
    _bid_price: float | None = None
    _ask_price: float | None = None
    _bid_persist: int = 0
    _ask_persist: int = 0
    _bid_replenish: int = 0
    _ask_replenish: int = 0
    _bid_withdraw: int = 0
    _ask_withdraw: int = 0
    _qi_history: deque[float] = field(default_factory=lambda: deque(maxlen=32))
    _same_qi_sign: int = 0
    _last_qi: float | None = None

    def on_quote(self, quote: QuoteTop) -> None:
        prev_bid_size = self.bid_size
        prev_ask_size = self.ask_size
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
                qi = (quote.bid_size - quote.ask_size) / total
                self.quote_imbalance_sum += qi
                self.quote_imbalance_samples += 1
                if self._last_qi is not None and qi * self._last_qi > 0:
                    self._same_qi_sign += 1
                else:
                    self._same_qi_sign = 1
                self._last_qi = qi
                self._qi_history.append(qi)
        if quote.bid > 0:
            if self._bid_price == quote.bid:
                self._bid_persist += 1
                if quote.bid_size is not None and prev_bid_size is not None:
                    if quote.bid_size > prev_bid_size:
                        self._bid_replenish += 1
                    elif quote.bid_size < prev_bid_size:
                        self._bid_withdraw += 1
            else:
                self._bid_price = quote.bid
                self._bid_persist = 1
        if quote.ask > 0:
            if self._ask_price == quote.ask:
                self._ask_persist += 1
                if quote.ask_size is not None and prev_ask_size is not None:
                    if quote.ask_size > prev_ask_size:
                        self._ask_replenish += 1
                    elif quote.ask_size < prev_ask_size:
                        self._ask_withdraw += 1
            else:
                self._ask_price = quote.ask
                self._ask_persist = 1

    def classify(self, trade: TradePrint) -> int:
        bid = trade.bid if trade.bid is not None else self.bid
        ask = trade.ask if trade.ask is not None else self.ask
        if ask is not None and ask > 0 and trade.price >= ask:
            return 1
        if bid is not None and bid > 0 and trade.price <= bid:
            return -1
        if self.last_trade_price is not None:
            if trade.price > self.last_trade_price:
                return 1
            if trade.price < self.last_trade_price:
                return -1
        return 0

    def on_trade(self, trade: TradePrint) -> int:
        if trade.size <= 0 or trade.price <= 0:
            return 0
        side = self.classify(trade)
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
        return side

    def snapshot(self, *, window_seconds: float = 60.0, now: datetime | None = None) -> FlowFeatures:
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
        move_bps = None
        if self.first_trade_price and self.last_trade_price and directional > 0:
            move_bps = (self.last_trade_price / self.first_trade_price - 1.0) * 10_000.0
            impact = move_bps / max(directional / 10_000.0, 1e-9)
            if ofi is not None:
                expected_sign = 1.0 if ofi >= 0 else -1.0
                signed_move = expected_sign * move_bps
                absorption = max(0.0, min(1.0, abs(ofi) * (1.0 - min(max(signed_move, 0.0) / 5.0, 1.0))))
        buy_abs = sell_abs = None
        init_buy = init_sell = None
        if self.buy_volume > 0 and move_bps is not None:
            init_buy = max(move_bps, 0.0) / (self.buy_volume / 10_000.0)
            buy_abs = max(0.0, min(1.0, 1.0 - min(max(move_bps, 0.0) / 5.0, 1.0)))
        if self.sell_volume > 0 and move_bps is not None:
            init_sell = max(-move_bps, 0.0) / (self.sell_volume / 10_000.0)
            sell_abs = max(0.0, min(1.0, 1.0 - min(max(-move_bps, 0.0) / 5.0, 1.0)))
        flow_to_disp = None
        if ofi is not None and move_bps is not None and abs(ofi) > 1e-9:
            flow_to_disp = move_bps / ofi
        del now
        quotes = max(self.quote_updates, 1)
        qi_velocity = None
        if len(self._qi_history) >= 2:
            qi_velocity = self._qi_history[-1] - self._qi_history[0]
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
            signed_delta=self.buy_volume - self.sell_volume,
            signed_aggressive_volume=self.buy_volume - self.sell_volume,
            directional_volume=directional,
            price_displacement_bps=move_bps,
            flow_to_displacement=flow_to_disp,
            displacement_per_10k_volume=impact,
            buy_absorption=buy_abs,
            sell_absorption=sell_abs,
            initiative_buy_efficiency=init_buy,
            initiative_sell_efficiency=init_sell,
            best_bid_size_persistence=self._bid_persist / quotes,
            best_ask_size_persistence=self._ask_persist / quotes,
            best_bid_replenishment=self._bid_replenish / quotes,
            best_ask_replenishment=self._ask_replenish / quotes,
            best_bid_withdrawal_rate=self._bid_withdraw / quotes,
            best_ask_withdrawal_rate=self._ask_withdraw / quotes,
            quote_imbalance_velocity=qi_velocity,
            quote_imbalance_persistence=self._same_qi_sign / quotes,
        )

    def reset(self) -> None:
        """Clear the snapshot window. NBBO price memory stays; CVD does not live here."""
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
        self.total_notional = 0.0
        self._bid_replenish = 0
        self._ask_replenish = 0
        self._bid_withdraw = 0
        self._ask_withdraw = 0
        self._bid_persist = 0
        self._ask_persist = 0
        self._same_qi_sign = 0
