from __future__ import annotations

import importlib.util
from pathlib import Path

MODULE = Path(__file__).parents[1] / "scripts" / "spy-overview-delta-v2.py"
spec = importlib.util.spec_from_file_location("overview_delta_v2", MODULE)
overview = importlib.util.module_from_spec(spec)
assert spec.loader is not None
spec.loader.exec_module(overview)


def scored(symbol: str, weight: float, move: float, realization: int, state: str = "ACTIVE_A+"):
    return {
        "symbol": symbol,
        "sector": "Information Technology",
        "weight": weight,
        "price": 100.0,
        "bias": "LONG" if move >= 0 else "SHORT",
        "state": state,
        "score": 90.0,
        "remaining_expected_move": move,
        "expected_realization_minutes": realization,
        "trade": {
            "available": True,
            "suggested_risk_dollars": 1000.0,
            "suggested_shares": 200,
            "position_value": 20000.0,
            "stop_invalidation": 95.0,
        },
    }


def beta_for(rows):
    symbols = [
        {"symbol": row["symbol"], "weight": row["weight"], "sector": row["sector"], "close": row["price"]}
        for row in rows
    ]
    return {"symbols": symbols, "sectors": [], "available": True, "price": 770.0}


def test_horizon_pressure_respects_each_constituents_realization_window():
    fast = scored("FAST", 0.10, 0.04, 30)
    slow = scored("SLOW", 0.10, 0.04, 390)
    pressure = overview.horizon_constituent_pressure(beta_for([fast, slow]), [fast, slow])

    p30 = pressure["horizons"]["30m"]["full_index"]["net_expected_spy_return"]
    pclose = pressure["horizons"]["close"]["full_index"]["net_expected_spy_return"]

    # FAST is fully realized at 30m while SLOW is only partially realized.
    expected_p30 = 0.10 * 0.04 + 0.10 * 0.04 * (30 / 390) ** 0.5
    assert abs(p30 - expected_p30) < 1e-12
    # Both displacements are fully incorporated by the close horizon.
    assert abs(pclose - 0.008) < 1e-12
    assert pclose > p30


def test_top_125_is_reported_but_full_index_is_not_renormalized():
    rows = [scored(f"S{i:03d}", 1 / 500, 0.01 if i % 2 else -0.01, 30) for i in range(500)]
    pressure = overview.horizon_constituent_pressure(beta_for(rows), rows)
    h30 = pressure["horizons"]["30m"]
    assert h30["top_125_count"] == 125
    assert 0.249 <= h30["top_125_by_weight"]["covered_weight"] <= 0.251
    assert 0.999 <= h30["full_index"]["covered_weight"] <= 1.001


def test_deteriorating_and_extended_constituents_are_downweighted():
    active = scored("ACTIVE", 0.10, 0.05, 30, "ACTIVE_A+")
    extended = scored("EXT", 0.10, 0.05, 30, "EXTENDED")
    deteriorating = scored("BAD", 0.10, 0.05, 30, "DETERIORATING")
    pressure = overview.horizon_constituent_pressure(
        beta_for([active, extended, deteriorating]), [active, extended, deteriorating]
    )
    by_symbol = {
        row["symbol"]: row
        for row in pressure["horizons"]["30m"]["top_positive_contributors"]
    }
    assert by_symbol["ACTIVE"]["realization_fraction"] == 1.0
    assert by_symbol["EXT"]["realization_fraction"] == 0.35
    assert by_symbol["BAD"]["realization_fraction"] == 0.25


def test_portfolio_risk_scaling_recalculates_shares_and_position_value():
    a = scored("AAA", 0.05, 0.05, 120)
    b = scored("BBB", 0.04, 0.05, 120)
    a["trade"]["suggested_risk_dollars"] = 2000.0
    b["trade"]["suggested_risk_dollars"] = 2000.0
    stocks = {
        "actionable_now": [a, b],
        "armed_watch": [],
        "extended_do_not_chase": [],
    }
    running = overview.merge_running_with_risk(stocks, {"state": "NO_TRADE"}, 100000.0)
    assert running["requested_open_risk"] == 4000.0
    assert running["suggested_open_risk"] == 3000.0
    assert running["max_simultaneous_open_risk"] == 3000.0
    for row in running["actionable"]:
        assert row["trade"]["suggested_risk_dollars"] == 1500.0
        # $1,500 / $5 stop = 300 shares, but the $25k capital cap limits this to 250.
        assert row["trade"]["suggested_shares"] == 250
        assert row["trade"]["position_value"] == 25000.0


def test_spy_predictor_uses_matching_horizon_constituent_pressure():
    alpha = {
        "available": True,
        "price": 770.0,
        "spy_iv": 0.18,
        "directional_confidence": 0.8,
        "trust_score": 0.8,
    }
    horizons = [
        {"name": "5m", "horizon_minutes": 5, "expected_return": 0.0},
        {"name": "30m", "horizon_minutes": 30, "expected_return": 0.0},
        {"name": "120m", "horizon_minutes": 120, "expected_return": 0.0},
        {"name": "EOD", "horizon_minutes": 390, "expected_return": 0.0},
        {"name": "1D", "horizon_minutes": 390, "expected_return": 0.0},
        {"name": "5D", "horizon_minutes": 1950, "expected_return": 0.0},
    ]
    beta = {
        "available": True,
        "price": 770.0,
        "confidence": 0.8,
        "coverage_ratio": 1.0,
        "forecasts": [
            {"horizon_minutes": 5, "expected_return_bps": 0.0},
            {"horizon_minutes": 30, "expected_return_bps": 0.0},
        ],
        "spy": {"symbol": "SPY", "close": 770.0, "atr14_bps": 25.0, "vwap": 770.0},
    }
    pressure = {
        "horizons": {
            name: {
                "full_index": {"net_expected_spy_return": value},
                "top_125_by_weight": {"net_expected_spy_return": value},
            }
            for name, value in {
                "open": 0.001,
                "30m": 0.002,
                "2h": -0.010,
                "close": 0.004,
                "3d": 0.005,
                "5d": 0.006,
            }.items()
        }
    }
    result = overview.horizon_spy_predictor(alpha, beta, horizons, pressure, {"available": False}, 100000.0)
    r30 = next(row for row in result["horizons"] if row["name"] == "30m")
    r2h = next(row for row in result["horizons"] if row["name"] == "2h")
    assert r30["predicted_move"] > 0
    assert r2h["predicted_move"] < 0
    assert r30["constituent_pressure"] == 0.002
    assert r2h["constituent_pressure"] == -0.010
