from __future__ import annotations

import importlib.util
import math
from pathlib import Path

MODULE = Path(__file__).parents[1] / "scripts" / "spy-overview-live.py"
spec = importlib.util.spec_from_file_location("spy_overview_live", MODULE)
live = importlib.util.module_from_spec(spec)
assert spec.loader is not None
spec.loader.exec_module(live)


def test_short_tenor_straddle_is_not_used_for_longer_horizon():
    alpha = {"spy_iv": 0.20}
    chain = {
        "available": True,
        "atm_straddle_pct": 0.01,
        "minutes_to_expiry": 30,
        "age_seconds": 10,
    }
    short, short_source = live.horizon_options_magnitude(alpha, chain, 15)
    long, long_source = live.horizon_options_magnitude(alpha, chain, 120)

    assert abs(short - 0.01 * math.sqrt(15 / 30)) < 1e-12
    assert "straddle" in short_source.lower()
    expected_long = 0.20 * math.sqrt(120 / (252 * 390))
    assert abs(long - expected_long) < 1e-12
    assert "iv" in long_source.lower()


def test_stale_straddle_falls_back_to_iv():
    alpha = {"spy_iv": 0.18}
    chain = {
        "available": True,
        "atm_straddle_pct": 0.02,
        "minutes_to_expiry": 300,
        "age_seconds": 181,
    }
    magnitude, source = live.horizon_options_magnitude(alpha, chain, 30)
    assert abs(magnitude - 0.18 * math.sqrt(30 / (252 * 390))) < 1e-12
    assert "iv" in source.lower()


def test_constituent_index_exposes_all_beta_names_without_broker_data(monkeypatch):
    payload = {
        "competition": {"paper_equity": 100000},
        "alpha": {"available": True, "account": {"secret": "not for bridge"}},
        "gamma": {"available": False, "by_symbol": {}},
        "beta": {
            "available": True,
            "symbols": [
                {
                    "symbol": "AAA",
                    "sector": "Tech",
                    "weight": 0.03,
                    "close": 100.0,
                    "return_1m": 0.001,
                    "return_5m": 0.002,
                    "return_15m": 0.003,
                    "vwap": 99.5,
                    "vwap_distance_bps": 50.0,
                    "ema8": 100.0,
                    "ema21": 99.0,
                    "rsi14": 60.0,
                    "atr14_bps": 40.0,
                    "relative_volume20": 1.6,
                    "range_expansion": 1.3,
                    "flow": {"order_flow_imbalance": 0.4, "quote_imbalance": 0.3},
                    "structure": {"structure_score": 0.5},
                    "auction": {"cvd_slope_5m": 100.0},
                },
                {
                    "symbol": "BBB",
                    "sector": "Health Care",
                    "weight": 0.02,
                    "close": 50.0,
                    "flow": {},
                    "structure": {},
                    "auction": {},
                },
                {"symbol": "SPY", "sector": "ETF", "weight": 0.0, "close": 770.0},
            ],
            "sectors": [],
        },
    }

    monkeypatch.setattr(
        live.D.V1,
        "alpha_options",
        lambda alpha: (
            {"available": False},
            {"AAA": {"atm_iv": 0.40, "skew": 0.02}},
            "/tmp/fake.db",
        ),
    )
    monkeypatch.setattr(
        live.D,
        "_all_scored_stocks",
        lambda beta, gamma, iv, equity: [
            {
                "symbol": "AAA",
                "state": "ARMED",
                "bias": "LONG",
                "score": 72.0,
                "convergence": 72.0,
                "signal_agreement_pct": 75.0,
                "predictive_strength": 70.0,
                "confirmation_strength": 74.0,
                "model_expected_move": 0.02,
                "remaining_expected_move": 0.02,
                "options_implied_move": 0.015,
                "expected_realization_minutes": 120,
            }
        ],
    )

    rows = live._constituent_index(payload)
    assert [row["symbol"] for row in rows] == ["AAA", "BBB"]
    aaa = rows[0]
    assert aaa["atm_iv"] == 0.40
    assert aaa["state"] == "ARMED"
    assert aaa["relative_volume20"] == 1.6
    assert "secret" not in str(rows)


def test_compact_constituent_has_no_execution_or_account_fields():
    raw = {
        "symbol": "XYZ",
        "sector": "Industrials",
        "weight": 0.01,
        "close": 25.0,
        "flow": {},
        "structure": {},
        "auction": {},
        "account_id": "must-not-leak",
        "order_token": "must-not-leak",
    }
    row = live._compact_constituent(raw, None, None)
    text = str(row)
    assert "must-not-leak" not in text
    assert "account_id" not in row
    assert "order_token" not in row
