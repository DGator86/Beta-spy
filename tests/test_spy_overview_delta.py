from __future__ import annotations

import importlib.util
from pathlib import Path

MODULE = Path(__file__).parents[1] / "scripts" / "spy-overview-delta.py"
spec = importlib.util.spec_from_file_location("overview_delta", MODULE)
overview = importlib.util.module_from_spec(spec)
assert spec.loader is not None
spec.loader.exec_module(overview)


def symbol(symbol="MU", weight=0.015, close=100.0, bullish=True):
    s = 1 if bullish else -1
    return {
        "symbol": symbol,
        "timestamp": "2026-09-08T13:45:00+00:00",
        "sector": "Information Technology",
        "weight": weight,
        "close": close,
        "return_1m": s * 0.002,
        "return_5m": s * 0.006,
        "return_15m": s * 0.012,
        "vwap": close / (1 + s * 0.005),
        "vwap_distance_bps": s * 50.0,
        "ema8": close - s * 0.2,
        "ema21": close - s * 0.7,
        "ema8_slope_bps": s * 3.0,
        "ema21_slope_bps": s * 1.5,
        "rsi14": 68 if bullish else 32,
        "atr14_bps": 55.0,
        "realized_vol20_bps": 35.0,
        "relative_volume20": 2.2,
        "range_expansion": 1.8,
        "flow": {
            "order_flow_imbalance": s * 0.72,
            "quote_imbalance": s * 0.55,
            "price_impact_bps_per_10k": s * 2.5,
            "absorption": 0.1,
            "initiative_buy_efficiency": 0.8 if bullish else 0.1,
            "initiative_sell_efficiency": 0.1 if bullish else 0.8,
        },
        "structure": {
            "structure_score": s * 0.8,
            "structure_break_strength": 0.75 if bullish else -0.75,
            "failed_break_strength": 0.0,
            "acceptance_above_score": 0.7 if bullish else 0.0,
            "acceptance_below_score": 0.0 if bullish else 0.7,
            "sweep_low_score": 0.6 if bullish else 0.0,
            "sweep_high_score": 0.0 if bullish else 0.6,
        },
        "auction": {"cvd_slope_5m": s * 1000.0, "price_cvd_divergence_5m": s * 0.6},
    }


def sector_factor():
    return {
        "sector": "Information Technology",
        "trend": 0.8,
        "momentum": 0.8,
        "flow": 0.7,
        "participation": 0.8,
    }


def catalyst(reference_price=99.5):
    return {
        "symbol": "MU",
        "direction": "BULLISH",
        "direction_sign": 1.0,
        "effective_strength": 0.9,
        "strength": 0.95,
        "materiality": 0.95,
        "novelty": 0.9,
        "confidence": 0.9,
        "expected_horizon_minutes": 120,
        "reference_price": reference_price,
        "headline": "fresh catalyst",
        "catalyst_type": "PEER",
    }


def test_no_catalyst_never_promotes_individual_stock_active():
    row = overview.score_stock(symbol(), sector_factor(), None, {"atm_iv": 0.45}, 100000)
    assert row is not None
    assert row["state"] not in {"ACTIVE_A+", "ACTIVE_B"}


def test_fresh_catalyst_plus_confirmation_can_be_active_and_has_trade_plan():
    row = overview.score_stock(symbol(), sector_factor(), catalyst(), {"atm_iv": 0.45}, 100000)
    assert row is not None
    assert row["state"] in {"ACTIVE_A+", "ACTIVE_B"}
    assert row["bias"] == "LONG"
    assert row["trade"]["suggested_risk_dollars"] > 0
    assert row["trade"]["suggested_shares"] > 0
    assert row["trade"]["stop_invalidation"] < row["price"]
    assert row["trade"]["targets"]["t3"] > row["price"]
    assert row["expected_realization_minutes"] == 120


def test_extended_when_catalyst_move_already_consumed():
    row = overview.score_stock(
        symbol(close=110),
        sector_factor(),
        catalyst(reference_price=100),
        {"atm_iv": 0.35},
        100000,
    )
    assert row is not None
    assert row["state"] == "EXTENDED"
    assert row["trade"]["suggested_risk_dollars"] == 0


def test_constituent_pressure_top_125_and_full_index():
    symbols = [symbol(symbol=f"S{i:03d}", weight=1 / 500, bullish=(i % 3 != 0)) for i in range(500)]
    beta = {"symbols": symbols, "sectors": [sector_factor()]}
    pressure = overview.constituent_pressure(beta)
    assert pressure["top_125_count"] == 125
    assert 0.24 <= pressure["top_125_by_weight"]["covered_weight"] <= 0.26
    assert 0.99 <= pressure["full_index"]["covered_weight"] <= 1.01
    assert len(pressure["top_positive_contributors"]) == 10
    assert len(pressure["top_negative_contributors"]) == 10


def test_spy_predictor_keeps_options_directionless():
    alpha = {
        "available": True,
        "price": 770.0,
        "change_pct": -0.002,
        "spy_iv": 0.18,
        "directional_confidence": 0.75,
        "trust_score": 0.85,
    }
    horizons = [
        {"name": "5m", "horizon_minutes": 5, "expected_return": -0.001, "probability_up": 0.40},
        {"name": "15m", "horizon_minutes": 15, "expected_return": -0.002, "probability_up": 0.36},
        {"name": "30m", "horizon_minutes": 30, "expected_return": -0.004, "probability_up": 0.30},
        {"name": "120m", "horizon_minutes": 120, "expected_return": -0.007, "probability_up": 0.25},
        {"name": "EOD", "horizon_minutes": 390, "expected_return": -0.006, "probability_up": 0.30},
        {"name": "1D", "horizon_minutes": 390, "expected_return": -0.005, "probability_up": 0.35},
        {"name": "5D", "horizon_minutes": 1950, "expected_return": 0.01, "probability_up": 0.60},
    ]
    beta = {
        "available": True,
        "price": 770.0,
        "confidence": 0.8,
        "coverage_ratio": 0.99,
        "direction": "BEARISH",
        "forecasts": [
            {"horizon_minutes": 5, "expected_return_bps": -8, "probability_up": 0.35},
            {"horizon_minutes": 15, "expected_return_bps": -18, "probability_up": 0.30},
            {"horizon_minutes": 30, "expected_return_bps": -35, "probability_up": 0.25},
        ],
        "spy": symbol("SPY", 0, 770, bullish=False),
    }
    pressure = {
        "full_index": {"net_expected_spy_return_30m": -0.003},
        "top_125_by_weight": {},
        "top_positive_contributors": [],
        "top_negative_contributors": [],
        "sector_contributions": [],
    }
    result = overview.spy_predictor(alpha, beta, horizons, pressure, {"available": False}, 100000)
    h30 = next(x for x in result["horizons"] if x["name"] == "30m")
    assert h30["predicted_move"] < 0
    assert h30["options_implied_magnitude"] is not None
    assert h30["options_implied_magnitude"] > 0
    assert "directionless" in h30["note"].lower()


def test_compact_chatgpt_view_excludes_broker_account():
    payload = {
        "generated_at": overview.iso_now(),
        "competition": {"paper_equity": 100000},
        "market": {"symbol": "SPY", "price": 770},
        "alpha": {"available": True, "account": {"secret": "do not expose"}},
        "beta": {
            "available": True,
            "status": "LIVE",
            "coverage_ratio": 1,
            "symbol_count": 500,
            "expected_symbol_count": 500,
        },
        "gamma": {"available": False, "events": []},
        "delta": {
            "data_contract": {"options_telemetry_symbols": 40, "gamma_catalysts_loaded": 0},
            "spy_predictor": {},
            "constituent_pressure": {},
            "stock_scanner": {"counts": {}},
            "running_list": {
                "actionable": [],
                "armed_watch": [],
                "extended_do_not_chase": [],
                "suggested_open_risk": 0,
                "max_simultaneous_open_risk": 3000,
            },
        },
        "safety": {"read_only_convergence": True},
    }
    compact = overview.chatgpt_view(payload)
    assert "do not expose" not in str(compact)
    assert compact["competition"]["paper_equity"] == 100000
