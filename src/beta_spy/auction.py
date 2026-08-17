"""Session-tape state: CVD plus SPY volume-at-price / footprint-lite.

CVD is session-scoped and must not live in FlowAccumulator, which resets
after every snapshot. Volume-at-price bins are SPY-only. Constituents get
compressed CVD features from SessionCvd.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field, replace
from datetime import datetime
from math import sqrt

from .models import AuctionFeatures, TradePrint


def _bin_key(price: float, tick: float) -> int:
    return round(price / tick)


def _zscore(values: list[float]) -> float | None:
    if len(values) < 5:
        return None
    mean = sum(values) / len(values)
    var = sum((item - mean) ** 2 for item in values) / (len(values) - 1)
    if var <= 0:
        return 0.0
    return (values[-1] - mean) / sqrt(var)


@dataclass
class SessionCvd:
    """Persistent signed volume. Survives the one-minute flow window reset."""

    cvd: float = 0.0
    session_date: object | None = None
    last_price: float | None = None
    _marks: deque[tuple[datetime, float, float]] = field(default_factory=deque)

    def restore(self, cvd: float, session_date, *, last_price: float | None = None) -> None:
        """Rehydrate compressed constituent CVD from minute_flow.payload."""
        self.session_date = session_date
        self.cvd = float(cvd)
        if last_price is not None and last_price > 0:
            self.last_price = last_price

    def on_trade(self, trade: TradePrint, side: int) -> None:
        day = trade.timestamp.date()
        if self.session_date != day:
            self.session_date = day
            self.cvd = 0.0
            self._marks.clear()
        if trade.size <= 0:
            return
        if side > 0:
            self.cvd += trade.size
        elif side < 0:
            self.cvd -= trade.size
        if trade.price > 0:
            self.last_price = trade.price
        self._marks.append((trade.timestamp, self.cvd, trade.price))
        while len(self._marks) > 4000:
            self._marks.popleft()

    def features(self, now: datetime | None = None) -> AuctionFeatures:
        del now
        if not self._marks:
            return AuctionFeatures(cvd_session=self.cvd if self.session_date else None)
        latest_t, latest_cvd, latest_px = self._marks[-1]

        def _ago(minutes: int) -> tuple[float, float] | None:
            cutoff = latest_t.timestamp() - minutes * 60
            for stamp, cvd, px in self._marks:
                if stamp.timestamp() >= cutoff:
                    return cvd, px
            return None

        cvd_5m = cvd_15m = slope_5 = slope_15 = div_5 = div_15 = None
        ago5 = _ago(5)
        ago15 = _ago(15)
        if ago5 is not None:
            cvd_5m = latest_cvd - ago5[0]
            slope_5 = cvd_5m / 5.0
            if ago5[1] > 0 and latest_px > 0:
                px_chg = latest_px - ago5[1]
                if px_chg * cvd_5m < 0:
                    div_5 = abs(cvd_5m) / max(abs(px_chg), 1e-6)
                else:
                    div_5 = 0.0
        if ago15 is not None:
            cvd_15m = latest_cvd - ago15[0]
            slope_15 = cvd_15m / 15.0
            if ago15[1] > 0 and latest_px > 0:
                px_chg = latest_px - ago15[1]
                if px_chg * cvd_15m < 0:
                    div_15 = abs(cvd_15m) / max(abs(px_chg), 1e-6)
                else:
                    div_15 = 0.0
        return AuctionFeatures(
            cvd_session=self.cvd,
            cvd_5m=cvd_5m,
            cvd_15m=cvd_15m,
            cvd_slope_5m=slope_5,
            cvd_slope_15m=slope_15,
            cvd_zscore=_zscore([mark[1] for mark in self._marks]),
            price_cvd_divergence_5m=div_5,
            price_cvd_divergence_15m=div_15,
        )


def attach_cvd(vap: AuctionFeatures, cvd: AuctionFeatures) -> AuctionFeatures:
    """SPY volume-at-price plus session CVD."""
    return replace(
        vap,
        cvd_session=cvd.cvd_session,
        cvd_5m=cvd.cvd_5m,
        cvd_15m=cvd.cvd_15m,
        cvd_slope_5m=cvd.cvd_slope_5m,
        cvd_slope_15m=cvd.cvd_slope_15m,
        cvd_zscore=cvd.cvd_zscore,
        price_cvd_divergence_5m=cvd.price_cvd_divergence_5m,
        price_cvd_divergence_15m=cvd.price_cvd_divergence_15m,
    )


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
