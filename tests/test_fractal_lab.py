from __future__ import annotations

from datetime import UTC, datetime, timedelta

from beta_spy.auction import SessionCvd, SpyAuctionState
from beta_spy.breadth import BreadthAggregator
from beta_spy.engine import Tape500Engine
from beta_spy.feature_sets import FEATURE_SETS, names_for
from beta_spy.flow import FlowAccumulator
from beta_spy.forecast import FEATURE_NAMES, vectorize
from beta_spy.models import (
    AuctionFeatures,
    FlowFeatures,
    HoldingMeta,
    MinuteBar,
    QuoteTop,
    StructureFeatures,
    SymbolFeatures,
    TradePrint,
)
from beta_spy.replay import HistoricalReplay
from beta_spy.storage import Tape500Store
from beta_spy.structure import StructureEngine


def _bar(i: int, close: float, *, high: float | None = None, low: float | None = None) -> MinuteBar:
    ts = datetime(2026, 8, 17, 14, 0, tzinfo=UTC) + timedelta(minutes=i)
    high = close + 0.4 if high is None else high
    low = close - 0.4 if low is None else low
    return MinuteBar("AAA", ts, close - 0.1, high, low, close, 10_000)


def test_structure_reads_higher_highs_and_higher_lows():
    # Zigzag that prints confirmed HH/HL at prominence 5: rise, pullback, rise higher.
    closes: list[float] = []
    price = 100.0
    for cycle in range(6):
        for _ in range(6):
            price += 0.4
            closes.append(price)
        for _ in range(4):
            price -= 0.15
            closes.append(price)
    bars = [_bar(i, close, high=close + 0.15, low=close - 0.15) for i, close in enumerate(closes)]
    features = StructureEngine().extract(bars)
    assert features.structure_state is not None
    assert features.structure_score is not None
    assert features.structure_score > 0
    assert -1.0 <= features.structure_score <= 1.0
    assert features.body_ratio is not None
    assert 0.0 <= features.close_location <= 1.0
    assert features.nfs_direction in (-1.0, 1.0)
    assert features.nfs_break_atr is not None
    assert features.nfs_duration is not None and features.nfs_duration >= 1


def test_sweep_high_is_a_failed_break_not_acceptance():
    closes = [100.0 + 0.03 * ((i % 7) - 3) for i in range(30)]
    closes += [100.4, 100.9, 101.5, 102.2, 101.4, 101.0, 100.8, 100.7, 100.65, 100.6]
    bars = [_bar(i, close, high=close + 0.08, low=close - 0.08) for i, close in enumerate(closes)]
    fail = _bar(len(bars), 100.7, high=103.4, low=100.5)
    features = StructureEngine().extract(bars + [fail])
    assert features.sweep_high_score is not None and features.sweep_high_score > 0
    assert features.acceptance_above_score in (None, 0.0)
    assert features.failed_break_strength is not None and features.failed_break_strength > 0


def test_nfs_and_prominence_33_are_populated():
    closes = [100.0] * 40 + [100.0 + i * 0.15 for i in range(1, 20)] + [103.0] * 40
    closes[39] = 99.4
    closes[60] = 104.8
    bars = [_bar(i, close, high=close + 0.12, low=close - 0.12) for i, close in enumerate(closes)]
    features = StructureEngine().extract(bars)
    assert features.swing_prominence_5 is not None
    assert features.swing_prominence_33 is not None


def test_cvd_lives_in_auction_not_the_flow_window():
    t = datetime(2026, 8, 17, 14, 0, tzinfo=UTC)
    flow = FlowAccumulator()
    cvd = SessionCvd()
    side = flow.on_trade(TradePrint("AAA", t, 100.02, 200, 100.0, 100.02))
    cvd.on_trade(TradePrint("AAA", t, 100.02, 200, 100.0, 100.02), side)
    assert cvd.features().cvd_session == 200
    flow.reset()
    side = flow.on_trade(TradePrint("AAA", t + timedelta(minutes=1), 99.99, 50, 100.0, 100.02))
    cvd.on_trade(TradePrint("AAA", t + timedelta(minutes=1), 99.99, 50, 100.0, 100.02), side)
    window = flow.snapshot(now=t + timedelta(minutes=1))
    assert window.signed_delta == -50
    assert cvd.features().cvd_session == 150


def test_absorption_splits_buy_and_sell():
    t = datetime(2026, 8, 17, 14, 0, tzinfo=UTC)
    flow = FlowAccumulator()
    flow.on_trade(TradePrint("AAA", t, 100.02, 10_000, 100.0, 100.02))
    flow.on_trade(TradePrint("AAA", t + timedelta(seconds=1), 100.03, 10_000, 100.0, 100.02))
    snap = flow.snapshot(now=t)
    assert snap.buy_absorption is not None
    assert snap.initiative_buy_efficiency is not None
    assert snap.price_displacement_bps is not None


def test_nbbo_persistence_is_not_a_liquidity_wall():
    t = datetime(2026, 8, 17, 14, 0, tzinfo=UTC)
    flow = FlowAccumulator()
    flow.on_quote(QuoteTop("AAA", t, 100.0, 100.02, 300, 100))
    flow.on_quote(QuoteTop("AAA", t + timedelta(seconds=1), 100.0, 100.02, 400, 100))
    flow.on_quote(QuoteTop("AAA", t + timedelta(seconds=2), 100.0, 100.02, 250, 90))
    snap = flow.snapshot(now=t)
    assert snap.best_bid_replenishment is not None and snap.best_bid_replenishment > 0
    assert snap.best_bid_withdrawal_rate is not None and snap.best_bid_withdrawal_rate > 0
    assert snap.best_bid_size_persistence is not None


def test_spy_auction_builds_poc_and_value_area():
    t = datetime(2026, 8, 17, 13, 30, tzinfo=UTC)
    auction = SpyAuctionState(tick=0.01)
    for i in range(20):
        auction.on_trade(TradePrint("SPY", t + timedelta(seconds=i), 100.00, 500, 99.99, 100.01), side=1)
    for i in range(5):
        auction.on_trade(TradePrint("SPY", t + timedelta(seconds=30 + i), 99.90, 100, 99.89, 99.91), side=-1)
    features = auction.features(100.00, t)
    assert features.session_poc == 100.00
    assert features.session_val is not None and features.session_vah is not None
    assert features.inside_value == 1.0
    assert features.max_positive_delta_price == 100.00


def test_breadth_exposes_structure_batch_a():
    t = datetime(2026, 8, 17, 14, 0, tzinfo=UTC)

    def feat(symbol: str, weight: float, state: float) -> SymbolFeatures:
        return SymbolFeatures(
            symbol=symbol,
            timestamp=t,
            sector="Technology",
            weight=weight,
            close=100.0,
            return_1m=0.0,
            return_5m=0.0,
            return_15m=0.0,
            vwap=100.0,
            vwap_distance_bps=0.0,
            ema8=100.0,
            ema21=100.0,
            ema8_slope_bps=0.0,
            ema21_slope_bps=0.0,
            rsi14=50.0,
            atr14_bps=10.0,
            realized_vol20_bps=10.0,
            relative_volume20=1.0,
            range_expansion=1.0,
            flow=FlowFeatures(),
            structure=StructureFeatures(structure_state=state, structure_score=state / 2.0),
            auction=AuctionFeatures(cvd_session=weight * 100),
        )

    factors = BreadthAggregator().aggregate(
        [
            feat("BIG", 0.8, -2.0),
            feat("SMALL1", 0.1, 2.0),
            feat("SMALL2", 0.1, 2.0),
            feat("SPY", 0.0, 1.0),
        ],
        timestamp=t,
        expected_symbol_count=3,
    )
    assert factors.structure_ew is not None and factors.structure_ew > 0
    assert factors.structure_weighted is not None and factors.structure_weighted < 0
    assert factors.structure_divergence is not None and factors.structure_divergence > 0
    vector = vectorize(factors)
    assert "structure_ew" not in FEATURE_NAMES
    assert "structure_ew" in names_for("structure_v1")
    assert vector.shape[0] == len(FEATURE_NAMES) + 2  # session fraction + fraction^2
    challenger = vectorize(factors, feature_set="structure_v1")
    assert challenger.shape[0] == len(FEATURE_SETS["structure_v1"]) + 2
    assert challenger.shape[0] > vector.shape[0]
    assert len(FEATURE_NAMES) == 25
    for name in ("structure_v1", "flow_v2", "auction_v1", "full_v1"):
        assert len(FEATURE_SETS[name]) > 25
        wider = vectorize(factors, feature_set=name)
        assert wider.shape[0] == len(FEATURE_SETS[name]) + 2


def test_empty_auction_is_not_reported_as_inside_value():
    t = datetime(2026, 8, 17, 14, 0, tzinfo=UTC)
    spy = SymbolFeatures(
        symbol="SPY",
        timestamp=t,
        sector="ETF",
        weight=0.0,
        close=100.0,
        return_1m=0.0,
        return_5m=0.0,
        return_15m=0.0,
        vwap=100.0,
        vwap_distance_bps=0.0,
        ema8=100.0,
        ema21=100.0,
        ema8_slope_bps=0.0,
        ema21_slope_bps=0.0,
        rsi14=50.0,
        atr14_bps=10.0,
        realized_vol20_bps=10.0,
        relative_volume20=1.0,
        range_expansion=1.0,
        flow=FlowFeatures(),
        structure=StructureFeatures(),
        auction=AuctionFeatures(),
    )
    factors = BreadthAggregator().aggregate([spy], timestamp=t, expected_symbol_count=0)
    assert factors.spy_value_location is None
    assert factors.spy_cvd is None
    assert factors.spy_poc_distance is None


def test_flow_payload_survives_store_roundtrip(tmp_path):
    store = Tape500Store(tmp_path / "flow.sqlite")
    t = datetime(2026, 8, 17, 14, 0, tzinfo=UTC)
    store.save_flow(
        t,
        "AAA",
        FlowFeatures(
            buy_volume=200,
            sell_volume=50,
            buy_absorption=0.8,
            initiative_buy_efficiency=1.2,
        ),
    )
    loaded = store.flows_for_timestamp(t)["AAA"]
    assert loaded.buy_absorption == 0.8
    assert loaded.initiative_buy_efficiency == 1.2
    store.close()


def test_engine_rebuilds_spy_auction_from_stored_prints(tmp_path):
    store = Tape500Store(tmp_path / "tape.sqlite")
    engine = Tape500Engine([HoldingMeta("AAA", "Technology", 1.0)], store=store)
    t = datetime(2026, 8, 17, 14, 0, tzinfo=UTC)
    for i in range(10):
        engine.on_trade(TradePrint("SPY", t, 100.00, 400, 99.99, 100.00, sequence=i))
    assert engine.auction.bins
    # Simulate a process restart: new engine, same store.
    restarted = Tape500Engine([HoldingMeta("AAA", "Technology", 1.0)], store=store)
    restarted.recover_spy_microstructure(t)
    features = restarted.auction.features(100.00, t)
    assert features.session_poc == 100.00
    assert restarted.cvd["SPY"].cvd == 4000
    store.close()


def test_snapshot_persists_minute_flow_and_cvd_payload(tmp_path):
    store = Tape500Store(tmp_path / "live.sqlite")
    engine = Tape500Engine([HoldingMeta("AAA", "Technology", 1.0)], store=store)
    t = datetime(2026, 8, 17, 14, 0, tzinfo=UTC)
    engine.add_bar(MinuteBar("AAA", t, 10.0, 10.1, 9.9, 10.0, 100), persist=True)
    engine.add_bar(MinuteBar("SPY", t, 100.0, 100.1, 99.9, 100.0, 1000), persist=True)
    engine.apply_print(TradePrint("SPY", t + timedelta(seconds=1), 100.02, 500, 100.0, 100.02, sequence=1))
    engine.apply_print(TradePrint("AAA", t + timedelta(seconds=2), 10.02, 80, 10.0, 10.02, sequence=1))
    snapshot = engine.build_snapshot(t)
    assert snapshot is not None
    assert snapshot.factors.spy_cvd == 500
    assert snapshot.factors.spy_poc_distance is not None
    assert snapshot.factors.spy_value_location in (-1.0, 0.0, 1.0)
    flows = store.flows_for_timestamp(t)
    assert flows["SPY"].buy_volume == 500
    assert flows["AAA"].buy_volume == 80
    cvds = store.session_cvd_for_timestamp(t)
    assert cvds["SPY"] == 500
    assert cvds["AAA"] == 80
    store.close()


def test_replay_rebuilds_auction_from_spy_trades_without_stored_flow(tmp_path):
    store = Tape500Store(tmp_path / "replay.sqlite")
    start = datetime(2026, 8, 17, 13, 30, tzinfo=UTC)
    bars: list[MinuteBar] = []
    for minute in range(4):
        ts = start + timedelta(minutes=minute)
        bars.append(MinuteBar("AAA", ts, 10.0, 10.1, 9.9, 10.0, 100))
        bars.append(MinuteBar("SPY", ts, 100.0, 100.1, 99.9, 100.0, 1000))
    store.save_bars(bars)
    with store.lock:
        for i in range(12):
            stamp = (start + timedelta(seconds=15 * i)).isoformat().replace("+00:00", "Z")
            store.connection.execute(
                "INSERT OR IGNORE INTO spy_trades(timestamp,sequence,price,size,bid,ask) VALUES(?,?,?,?,?,?)",
                (stamp, i, 100.00, 250, 99.99, 100.00),
            )
        store.connection.commit()
    assert store.flows_for_timestamp(start) == {}
    engine = Tape500Engine([HoldingMeta("AAA", "Technology", 1.0)], store=store)
    snapshots = list(HistoricalReplay(store, engine).run())
    assert snapshots
    last = snapshots[-1]
    assert last.factors.spy_cvd == 12 * 250
    assert last.factors.spy_poc_distance is not None
    assert last.factors.spy_value_location is not None
    spy_flow = store.flows_for_timestamp(start).get("SPY")
    assert spy_flow is not None and spy_flow.buy_volume > 0
    store.close()


def test_backfill_spy_minute_flow_from_prints(tmp_path):
    store = Tape500Store(tmp_path / "backfill.sqlite")
    start = datetime(2026, 8, 17, 13, 30, tzinfo=UTC)
    with store.lock:
        for i in range(8):
            stamp = (start + timedelta(seconds=20 * i)).isoformat().replace("+00:00", "Z")
            store.connection.execute(
                "INSERT OR IGNORE INTO spy_trades(timestamp,sequence,price,size,bid,ask) VALUES(?,?,?,?,?,?)",
                (stamp, i, 100.02, 100, 100.0, 100.02),
            )
        store.connection.commit()
    written = store.backfill_spy_minute_flow(start.date())
    assert written >= 1
    total_buy = 0.0
    for minute in range(3):
        row = store.flows_for_timestamp(start + timedelta(minutes=minute)).get("SPY")
        if row is not None:
            total_buy += row.buy_volume
    assert total_buy == 800
    store.close()
