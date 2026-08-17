from __future__ import annotations

from dataclasses import replace
from datetime import UTC, datetime, timedelta
from pathlib import Path

from beta_spy.breadth import BreadthAggregator
from beta_spy.engine import Tape500Engine
from beta_spy.flow import FlowAccumulator
from beta_spy.models import (
    FlowFeatures,
    HoldingMeta,
    MinuteBar,
    QuoteTop,
    SymbolFeatures,
    TradePrint,
)
from beta_spy.replay import HistoricalReplay
from beta_spy.storage import Tape500Store


def test_flow_classifies_tape_and_quote_pressure() -> None:
    t = datetime(2026, 8, 13, 14, 0, tzinfo=UTC)
    flow = FlowAccumulator()
    flow.on_quote(QuoteTop("AAPL", t, 100.0, 100.02, 300, 100))
    flow.on_trade(TradePrint("AAPL", t, 100.02, 200, 100.0, 100.02))
    flow.on_trade(TradePrint("AAPL", t + timedelta(seconds=1), 100.00, 100, 100.0, 100.02))
    result = flow.snapshot()
    assert result.order_flow_imbalance == (200 - 100) / 300
    assert result.quote_imbalance == 0.5
    assert result.average_spread_bps is not None
    assert result.trade_intensity > 0


def _feature(symbol: str, weight: float, sector: str, direction: float) -> SymbolFeatures:
    t = datetime(2026, 8, 13, 14, 0, tzinfo=UTC)
    close = 100.0 * (1.0 + direction * 0.001)
    return SymbolFeatures(
        symbol=symbol,
        timestamp=t,
        sector=sector,
        weight=weight,
        close=close,
        return_1m=direction * 0.001,
        return_5m=direction * 0.002,
        return_15m=direction * 0.003,
        vwap=100.0,
        vwap_distance_bps=direction * 10.0,
        ema8=100.2 if direction > 0 else 99.8,
        ema21=100.0,
        ema8_slope_bps=direction * 1.0,
        ema21_slope_bps=direction * 0.5,
        rsi14=60.0 if direction > 0 else 40.0,
        atr14_bps=12.0,
        realized_vol20_bps=10.0,
        relative_volume20=1.5,
        range_expansion=1.3,
        flow=FlowFeatures(
            order_flow_imbalance=direction * 0.6,
            quote_imbalance=direction * 0.4,
            average_spread_bps=1.0,
            price_impact_bps_per_10k=direction * 1.0,
            absorption=0.0,
        ),
    )


def test_breadth_exposes_equal_and_weighted_state() -> None:
    t = datetime(2026, 8, 13, 14, 0, tzinfo=UTC)
    items = [
        _feature("BIG", 0.8, "Technology", -1),
        _feature("SMALL1", 0.1, "Financials", 1),
        _feature("SMALL2", 0.1, "Industrials", 1),
        _feature("SPY", 0.0, "ETF", 1),
    ]
    factors = BreadthAggregator().aggregate(items, timestamp=t, expected_symbol_count=3)
    assert factors.trend_ew is not None and factors.trend_ew > 0
    assert factors.trend_weighted is not None and factors.trend_weighted < 0
    assert factors.pct_positive_5m == 2 / 3
    assert len(factors.sectors) == 3


def test_engine_builds_snapshot_after_history() -> None:
    holdings = [
        HoldingMeta("AAA", "Technology", 0.5),
        HoldingMeta("BBB", "Financials", 0.5),
    ]
    engine = Tape500Engine(holdings)
    start = datetime(2026, 8, 13, 13, 30, tzinfo=UTC)
    for minute in range(40):
        ts = start + timedelta(minutes=minute)
        for symbol, base in [("AAA", 100.0), ("BBB", 80.0), ("SPY", 600.0)]:
            close = base * (1.0 + minute * 0.0002)
            engine.add_bar(
                MinuteBar(symbol, ts, close * 0.9998, close * 1.0003, close * 0.9995, close, 1000 + minute)
            )
    snapshot = engine.build_snapshot(start + timedelta(minutes=40))
    assert snapshot is not None
    assert snapshot.factors.coverage_ratio == 1.0
    assert len(snapshot.forecasts) == 3
    assert snapshot.decision.action in {"TRADE", "NO_TRADE"}


def test_store_and_replay(tmp_path: Path) -> None:
    holdings = [HoldingMeta("AAA", "Technology", 1.0)]
    store = Tape500Store(tmp_path / "tape.db")
    start = datetime(2026, 8, 13, 13, 30, tzinfo=UTC)
    bars = []
    for minute in range(35):
        ts = start + timedelta(minutes=minute)
        bars.extend(
            [
                MinuteBar("AAA", ts, 100, 101, 99, 100 + minute * 0.01, 1000),
                MinuteBar("SPY", ts, 600, 601, 599, 600 + minute * 0.02, 5000),
            ]
        )
    store.save_bars(bars)
    engine = Tape500Engine(holdings)
    outputs = list(HistoricalReplay(store, engine).run())
    assert len(outputs) == 35
    assert outputs[-1].factors.coverage_ratio == 1.0
    store.close()


def test_historical_flow_aggregation_without_raw_tick_storage(tmp_path):
    from beta_spy.historical import AlpacaHistoricalClient

    class Fake(AlpacaHistoricalClient):
        def __init__(self):
            pass

        def iter_quotes(self, symbol, start, end, *, feed="sip"):
            yield {"t": "2026-08-13T14:30:01Z", "bp": 99.99, "ap": 100.01, "bs": 800, "as": 200}
            yield {"t": "2026-08-13T14:30:40Z", "bp": 100.00, "ap": 100.02, "bs": 700, "as": 300}

        def iter_trades(self, symbol, start, end, *, feed="sip"):
            yield {"t": "2026-08-13T14:30:02Z", "p": 100.01, "s": 100, "i": 1}
            yield {"t": "2026-08-13T14:30:41Z", "p": 100.00, "s": 50, "i": 2}

    store = Tape500Store(tmp_path / "flow.sqlite")
    try:
        start = datetime(2026, 8, 13, 14, 30, tzinfo=UTC)
        end = datetime(2026, 8, 13, 14, 31, tzinfo=UTC)
        count = Fake().backfill_minute_flow(store, ["AAA"], start, end)
        assert count == 1
        rows = store.flows_for_timestamp(start)
        flow = rows["AAA"]
        assert flow.buy_volume == 100
        assert flow.sell_volume == 50
        assert flow.quote_imbalance is not None and flow.quote_imbalance > 0
    finally:
        store.close()


def test_option_planner_selects_defined_risk_call_spread():
    from beta_spy.options import plan_debit_spread

    rows = [
        {"symbol": "C100", "expiration": "2026-08-13", "right": "C", "strike": 100, "bid": 1.00, "ask": 1.05, "delta": .56, "open_interest": 1000},
        {"symbol": "C101", "expiration": "2026-08-13", "right": "C", "strike": 101, "bid": .45, "ask": .50, "delta": .31, "open_interest": 800},
        {"symbol": "C102", "expiration": "2026-08-13", "right": "C", "strike": 102, "bid": .18, "ask": .22, "delta": .17, "open_interest": 700},
    ]
    plan = plan_debit_spread(rows, "BULLISH", maximum_risk_dollars=100)
    assert plan is not None
    assert plan.strategy == "CALL_DEBIT_SPREAD"
    assert plan.max_loss_dollars <= 100
    assert plan.legs[0].side == "BUY" and plan.legs[1].side == "SELL"


def test_option_planner_expected_value_path_sizes_and_rejects():
    from beta_spy.options import plan_debit_spread

    rows = [
        {"symbol": "C100", "expiration": "2026-08-13", "right": "C", "strike": 100, "bid": 1.00, "ask": 1.05, "delta": .56, "open_interest": 1000},
        {"symbol": "C101", "expiration": "2026-08-13", "right": "C", "strike": 101, "bid": .45, "ask": .50, "delta": .31, "open_interest": 800},
        {"symbol": "C102", "expiration": "2026-08-13", "right": "C", "strike": 102, "bid": .18, "ask": .22, "delta": .17, "open_interest": 700},
    ]
    plan = plan_debit_spread(
        rows,
        "BULLISH",
        maximum_risk_dollars=200,
        expected_move_dollars=1.20,
        probability=0.62,
    )
    assert plan is not None
    assert plan.expected_value_dollars is not None and plan.expected_value_dollars > 0
    assert plan.contracts >= 1
    assert plan.total_risk_dollars <= 200 + 1e-6

    # A negligible expected move cannot pay the entry friction: no trade.
    rejected = plan_debit_spread(
        rows,
        "BULLISH",
        maximum_risk_dollars=200,
        expected_move_dollars=0.01,
        probability=0.52,
    )
    assert rejected is None


def _chain_with_greeks():
    # SPY at 100. Tight markets, healthy OI, greeks present. Theta per day.
    rows = []
    for strike, cd, cg, ct, cb, ca in [
        (98, .72, .030, -.18, 2.10, 2.14), (99, .63, .040, -.22, 1.30, 1.34),
        (100, .51, .045, -.25, .70, .74), (101, .38, .042, -.23, .48, .52),
        (102, .26, .035, -.20, .32, .36), (103, .16, .026, -.15, .14, .18),
    ]:
        rows.append({"symbol": f"C{strike}", "expiration": "2026-08-14", "right": "C",
                     "strike": strike, "bid": cb, "ask": ca, "delta": cd, "gamma": cg,
                     "theta": ct, "open_interest": 900})
    for strike, pd_, pg, pt, pb, pa in [
        (102, -.74, .030, -.18, 2.00, 2.04), (101, -.62, .040, -.22, 1.25, 1.29),
        (100, -.49, .045, -.25, .68, .72), (99, -.37, .042, -.23, .46, .50),
        (98, -.25, .035, -.20, .30, .34), (97, -.15, .026, -.15, .13, .17),
    ]:
        rows.append({"symbol": f"P{strike}", "expiration": "2026-08-14", "right": "P",
                     "strike": strike, "bid": pb, "ask": pa, "delta": pd_, "gamma": pg,
                     "theta": pt, "open_interest": 900})
    return rows


def test_planner_offers_credit_spreads_for_directional_signals():
    from beta_spy.options import plan_best_strategy

    plan = plan_best_strategy(
        _chain_with_greeks(),
        "BULLISH",
        maximum_risk_dollars=300,
        hold_minutes=15,
        spy_price=100.0,
        expected_move_dollars=0.40,
        minutes_to_expiry=390,
    )
    assert plan is not None
    assert plan.strategy in {"CALL_DEBIT_SPREAD", "PUT_CREDIT_SPREAD"}
    assert plan.expected_value_dollars is not None and plan.expected_value_dollars > 0
    assert plan.total_risk_dollars <= 300 + 1e-6
    sides = {(leg.side, leg.right) for leg in plan.legs}
    if plan.strategy == "PUT_CREDIT_SPREAD":
        assert ("SELL", "P") in sides and ("BUY", "P") in sides


def test_planner_sells_iron_condor_on_neutral_quiet_signal():
    from beta_spy.options import plan_best_strategy

    plan = plan_best_strategy(
        _chain_with_greeks(),
        "NEUTRAL",
        maximum_risk_dollars=400,
        hold_minutes=15,
        spy_price=100.0,
        expected_move_dollars=0.0,
        minutes_to_expiry=390,
    )
    assert plan is not None
    assert plan.strategy == "IRON_CONDOR"
    assert len(plan.legs) == 4
    sells = [leg for leg in plan.legs if leg.side == "SELL"]
    buys = [leg for leg in plan.legs if leg.side == "BUY"]
    assert len(sells) == 2 and len(buys) == 2
    assert {leg.right for leg in sells} == {"C", "P"}
    assert plan.max_loss_dollars <= 400 + 1e-6
    assert plan.expected_value_dollars is not None and plan.expected_value_dollars > 0


def test_high_conviction_gates_require_agreement_magnitude_and_breadth():
    from beta_spy.decision import DecisionEngine
    from beta_spy.models import HorizonForecast, MarketFactors

    engine = DecisionEngine()
    ts = datetime(2026, 8, 10, 15, 0, tzinfo=UTC)  # 11:00 ET, open window

    def forecasts(prob: float, conf: float, exp_bps: float):
        return tuple(
            HorizonForecast(horizon_minutes=h, probability_up=prob, expected_return_bps=exp_bps,
                            confidence=conf, model_ready=True, sample_count=500)
            for h in (5, 15, 30)
        )

    def factors(trend: float = 0.6):
        return MarketFactors(
            timestamp=ts, symbol_count=500, expected_symbol_count=500,
            coverage_ratio=0.99, covered_weight=0.99,
            trend_ew=trend, trend_weighted=trend, momentum_ew=trend,
            momentum_weighted=trend, volume_ew=0.0, volume_weighted=0.0,
            flow_ew=trend, flow_weighted=trend, volatility_ew=0.10,
            volatility_weighted=0.10, pct_above_vwap=0.8, pct_ema_bullish=0.8,
            pct_positive_5m=0.8, pct_buy_flow=0.8, participation=trend,
            concentration=0.1, breadth_acceleration=0.0, spy_return_1m=0.0002,
            spy_return_5m=0.001, spy_vwap_distance_bps=2.0, spy_flow=trend,
            spy_quote_imbalance=0.2, spy_spread_bps=1.0,
        )

    # Everything aligned with conviction: trade.
    decision = engine.decide(ts, factors(), forecasts(0.66, 0.5, 8.0))
    assert decision.action == "TRADE"

    # Forecast move too small and probability not extreme enough to override.
    decision = engine.decide(ts, factors(), forecasts(0.58, 0.5, 1.0))
    assert decision.action == "NO_TRADE"
    assert decision.gates["forecast_magnitude"] is False

    # Horizons agree but without conviction: gated.
    decision = engine.decide(ts, factors(), forecasts(0.66, 0.2, 8.0))
    assert decision.action == "NO_TRADE"
    assert decision.gates["multi_horizon"] is False

    # Three breadth dissenters break a 5-factor majority: gated.
    dissent = factors()
    dissent = replace(dissent, trend_ew=-0.6, momentum_ew=-0.6, participation=-0.6)
    decision = engine.decide(ts, dissent, forecasts(0.66, 0.5, 8.0))
    assert decision.action == "NO_TRADE"
    assert decision.gates["breadth_confirmation"] is False


def test_neutral_trade_requires_quiet_tape_not_just_model_neutrality():
    from beta_spy.decision import DecisionEngine
    from beta_spy.models import HorizonForecast

    engine = DecisionEngine()
    ts = datetime(2026, 8, 10, 15, 0, tzinfo=UTC)
    neutral_forecasts = tuple(
        HorizonForecast(horizon_minutes=h, probability_up=0.51, expected_return_bps=0.5,
                        confidence=0.4, model_ready=True, sample_count=500)
        for h in (5, 15, 30)
    )

    def factors_at(minute: int, spy_return_1m: float):
        from beta_spy.models import MarketFactors
        return MarketFactors(
            timestamp=ts + timedelta(minutes=minute), symbol_count=500,
            expected_symbol_count=500, coverage_ratio=0.99, covered_weight=0.99,
            trend_ew=0.0, trend_weighted=0.0, momentum_ew=0.0, momentum_weighted=0.0,
            volume_ew=0.0, volume_weighted=0.0, flow_ew=0.0, flow_weighted=0.0,
            volatility_ew=0.10, volatility_weighted=0.10, pct_above_vwap=0.5,
            pct_ema_bullish=0.5, pct_positive_5m=0.5, pct_buy_flow=0.5,
            participation=0.0, concentration=0.1, breadth_acceleration=0.0,
            spy_return_1m=spy_return_1m, spy_return_5m=0.0, spy_vwap_distance_bps=0.0,
            spy_flow=0.0, spy_quote_imbalance=0.0, spy_spread_bps=1.0,
        )

    # Loud tape: model-neutral but 1m returns are large -> must NOT sell premium.
    for minute in range(20):
        decision = engine.decide(ts + timedelta(minutes=minute), factors_at(minute, 0.0015), neutral_forecasts)
    assert decision.action == "NO_TRADE"

    # Quiet tape: tiny 1m returns for a full window -> condor signal fires.
    engine = DecisionEngine()
    for minute in range(20):
        decision = engine.decide(ts + timedelta(minutes=minute), factors_at(minute, 0.00005), neutral_forecasts)
    assert decision.action == "TRADE_NEUTRAL"
    assert decision.structure == "IRON_CONDOR"
    assert decision.risk_multiplier == 0.5


def test_causal_backtest_report_scores_matured_forecasts(tmp_path: Path) -> None:
    from beta_spy.backtest import run_backtest, write_report

    holdings = [HoldingMeta("AAA", "Technology", 1.0)]
    store = Tape500Store(tmp_path / "bt.sqlite")
    start = datetime(2026, 8, 10, 13, 30, tzinfo=UTC)
    bars = []
    for minute in range(300):
        ts = start + timedelta(minutes=minute)
        drift = minute * 0.00005
        bars.append(MinuteBar("AAA", ts, 100, 101, 99, 100 * (1 + drift), 10000 + minute))
        bars.append(MinuteBar("SPY", ts, 600, 601, 599, 600 * (1 + drift * 0.8), 50000 + minute))
    store.save_bars(bars)
    report, observations = run_backtest(store, holdings)
    assert report.snapshots == 300
    assert observations
    by_horizon = {item.horizon_minutes: item for item in report.horizons}
    assert by_horizon[15].observations > 200
    assert by_horizon[15].model_ready_observations > 0
    md, js = write_report(report, tmp_path / "report")
    assert md.exists() and js.exists()
    store.close()


def test_tradier_historical_timesales_parsing(tmp_path: Path) -> None:
    import httpx
    from beta_spy.historical import TradierHistoricalClient

    def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.path.endswith("/markets/timesales")
        return httpx.Response(
            200,
            headers={"X-Ratelimit-Available": "100"},
            json={"series": {"data": [{
                "timestamp": 1786632600,
                "open": 100.0, "high": 101.0, "low": 99.5, "close": 100.5,
                "volume": 12345, "vwap": 100.4,
            }]}}
        )

    client = TradierHistoricalClient("token")
    client.client.close()
    client.client = httpx.Client(
        base_url="https://api.tradier.com/v1",
        transport=httpx.MockTransport(handler),
        headers={"Authorization": "Bearer token", "Accept": "application/json"},
    )
    try:
        rows = list(client.iter_bars(
            "AAA",
            datetime(2026, 8, 13, 13, 30, tzinfo=UTC),
            datetime(2026, 8, 13, 20, 0, tzinfo=UTC),
        ))
        assert len(rows) == 1
        assert rows[0].close == 100.5
        assert rows[0].vwap == 100.4
    finally:
        client.close()
