"""SPY-only executed volume-at-price and footprint-lite.

Constituents keep compressed flow metrics. This object is the richer SPY
auction substrate: session POC/VAH/VAL plus per-price aggressor delta.
There is no Level-II depth here — bins are prints we actually saw.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime

from .models import AuctionFeatures, TradePrint


def _bin_key(price: float, tick: float) -> int:
    return round(price / tick)


@dataclass
class _PriceBin:
    buy: float = 0.0
    sell: float = 0.0

    @property
    def total(self) -> float:
        return self.buy + self.sell

    @property
    def delta(self) -> float:
        return self.buy - self.sell


@dataclass
class SpyAuctionState:
    tick: float = 0.01
    value_area_fraction: float = 0.70
    session_date: object | None = None
    bins: dict[int, _PriceBin] = field(default_factory=dict)
    last_poc: float | None = None
    prior_poc: float | None = None

    def reset_session(self, session_date: object) -> None:
        self.session_date = session_date
        self.prior_poc = self.last_poc
        self.bins = {}
        self.last_poc = None

    def on_trade(self, trade: TradePrint, side: int) -> None:
        day = trade.timestamp.date()
        if self.session_date != day:
            self.reset_session(day)
        if trade.price <= 0 or trade.size <= 0:
            return
        key = _bin_key(trade.price, self.tick)
        bin_ = self.bins.setdefault(key, _PriceBin())
        if side > 0:
            bin_.buy += trade.size
        elif side < 0:
            bin_.sell += trade.size
        else:
            bin_.buy += trade.size * 0.5
            bin_.sell += trade.size * 0.5

    def features(self, price: float, timestamp: datetime | None = None) -> AuctionFeatures:
        del timestamp
        if not self.bins or price <= 0:
            return AuctionFeatures()
        items = sorted(
            ((key * self.tick, bin_) for key, bin_ in self.bins.items()),
            key=lambda item: item[0],
        )
        total = sum(bin_.total for _, bin_ in items)
        if total <= 0:
            return AuctionFeatures()
        poc_price, _ = max(items, key=lambda item: item[1].total)
        self.last_poc = poc_price
        # Value area: expand from POC until `value_area_fraction` of volume.
        by_price = {price_: bin_ for price_, bin_ in items}
        prices = [price_ for price_, _ in items]
        poc_index = prices.index(poc_price)
        lo = hi = poc_index
        covered = by_price[poc_price].total
        target = self.value_area_fraction * total
        while covered < target and (lo > 0 or hi < len(prices) - 1):
            left = by_price[prices[lo - 1]].total if lo > 0 else -1.0
            right = by_price[prices[hi + 1]].total if hi < len(prices) - 1 else -1.0
            if right >= left:
                hi += 1
                covered += by_price[prices[hi]].total
            else:
                lo -= 1
                covered += by_price[prices[lo]].total
        val = prices[lo]
        vah = prices[hi]
        volumes = [bin_.total for _, bin_ in items]
        mean_vol = sum(volumes) / len(volumes)
        hvn = min(
            (abs(price_ - price) for price_, bin_ in items if bin_.total >= mean_vol),
            default=None,
        )
        lvn = min(
            (abs(price_ - price) for price_, bin_ in items if bin_.total < mean_vol),
            default=None,
        )
        max_pos = max(items, key=lambda item: item[1].delta)
        max_neg = min(items, key=lambda item: item[1].delta)
        stacked_buy = 0
        stacked_sell = 0
        run_buy = run_sell = 0
        for _, bin_ in items:
            if bin_.delta > 0 and bin_.total > 0:
                run_buy += 1
                run_sell = 0
                stacked_buy = max(stacked_buy, run_buy)
            elif bin_.delta < 0 and bin_.total > 0:
                run_sell += 1
                run_buy = 0
                stacked_sell = max(stacked_sell, run_sell)
            else:
                run_buy = run_sell = 0
        high_abs = None
        low_abs = None
        if items:
            top = max(items, key=lambda item: item[0])
            bot = min(items, key=lambda item: item[0])
            if top[1].total > 0:
                high_abs = top[1].sell / top[1].total
            if bot[1].total > 0:
                low_abs = bot[1].buy / bot[1].total
        poc_migration = None
        if self.prior_poc and self.prior_poc > 0 and poc_price:
            poc_migration = (poc_price / self.prior_poc - 1.0) * 10_000.0
        inside = 1.0 if val <= price <= vah else 0.0
        return AuctionFeatures(
            session_poc=poc_price,
            session_vah=vah,
            session_val=val,
            distance_to_poc=(price - poc_price) / price * 10_000.0,
            distance_to_vah=(price - vah) / price * 10_000.0,
            distance_to_val=(price - val) / price * 10_000.0,
            nearest_hvn_distance=hvn,
            nearest_lvn_distance=lvn,
            poc_migration=poc_migration,
            value_area_width=(vah - val) / price * 10_000.0 if price > 0 else None,
            inside_value=inside,
            above_value=1.0 if price > vah else 0.0,
            below_value=1.0 if price < val else 0.0,
            max_positive_delta_price=max_pos[0] if max_pos[1].delta > 0 else None,
            max_negative_delta_price=max_neg[0] if max_neg[1].delta < 0 else None,
            stacked_buy_imbalance_count=stacked_buy,
            stacked_sell_imbalance_count=stacked_sell,
            local_absorption_high=high_abs,
            local_absorption_low=low_abs,
        )
