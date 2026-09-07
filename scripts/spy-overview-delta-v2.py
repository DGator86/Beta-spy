#!/usr/bin/env python3
"""Horizon-aware read-only Delta convergence publisher.

This is the production-facing wrapper around ``spy-overview-delta.py``.  Alpha
and Beta remain independent engines.  Delta reads their published state,
optional Gamma catalysts and Alpha's read-only option telemetry, then publishes
competition research only.  It has no execution authority and never feeds a
value back into Alpha or Beta.
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import math
import os
from pathlib import Path
from typing import Any


def _load_v1():
    candidates = [
        os.getenv("SPY_OVERVIEW_DELTA_V1"),
        "/usr/local/lib/spy-overview-delta-v1.py",
        str(Path(__file__).with_name("spy-overview-delta.py")),
    ]
    for value in candidates:
        if not value:
            continue
        path = Path(value)
        if not path.is_file():
            continue
        spec = importlib.util.spec_from_file_location("spy_overview_delta_v1", path)
        if spec is None or spec.loader is None:
            continue
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module
    raise RuntimeError("Delta v1 module is unavailable")


V1 = _load_v1()
num = V1.num
clamp = V1.clamp

HORIZONS: tuple[tuple[str, int], ...] = (
    ("open", 5),
    ("30m", 30),
    ("2h", 120),
    ("close", 390),
    ("3d", 1170),
    ("5d", 1950),
)


def _all_scored_stocks(
    beta: dict[str, Any],
    gamma: dict[str, Any],
    option_map: dict[str, dict[str, Any]],
    equity: float,
) -> list[dict[str, Any]]:
    sectors = {
        str(row.get("sector")): row
        for row in beta.get("sectors", [])
        if isinstance(row, dict)
    }
    by_symbol = gamma.get("by_symbol") if isinstance(gamma.get("by_symbol"), dict) else {}
    rows: list[dict[str, Any]] = []
    for raw in beta.get("symbols", []):
        if not isinstance(raw, dict):
            continue
        symbol = str(raw.get("symbol") or "").upper()
        scored = V1.score_stock(
            raw,
            sectors.get(str(raw.get("sector"))),
            by_symbol.get(symbol),
            option_map.get(symbol),
            equity,
        )
        if scored is not None:
            rows.append(scored)
    return rows


def _realization_fraction(row: dict[str, Any], horizon_minutes: int) -> float:
    """Fraction of a stock thesis expected to have realized by a horizon.

    Before the stock's own realization window we use a square-root arrival
    curve.  Once the expected window has elapsed the modeled displacement is
    treated as incorporated into the price level, rather than decaying back to
    zero.  Deteriorating/extended setups are down-weighted because little fresh
    directional information remains.
    """
    expected = max(1, int(num(row.get("expected_realization_minutes")) or 120))
    fraction = min(1.0, math.sqrt(max(horizon_minutes, 1) / expected))
    state = str(row.get("state") or "")
    if state == "DETERIORATING":
        fraction *= 0.25
    elif state == "EXTENDED":
        fraction *= 0.35
    elif state == "PREDICTIVE_SETUP":
        fraction *= 0.75
    return fraction


def horizon_constituent_pressure(
    beta: dict[str, Any], scored_rows: list[dict[str, Any]]
) -> dict[str, Any]:
    """Weight every current S&P constituent into SPY at each requested horizon."""
    by_symbol = {str(row.get("symbol")): row for row in scored_rows}
    weights: dict[str, tuple[float, str]] = {}
    for raw in beta.get("symbols", []):
        if not isinstance(raw, dict):
            continue
        symbol = str(raw.get("symbol") or "")
        if not symbol or symbol == "SPY":
            continue
        weight = num(raw.get("weight")) or 0.0
        if weight > 0:
            weights[symbol] = (weight, str(raw.get("sector") or "UNKNOWN"))

    ordered = sorted(weights.items(), key=lambda item: item[1][0], reverse=True)
    top_symbols = {symbol for symbol, _ in ordered[:125]}
    horizon_rows: dict[str, Any] = {}

    for name, minutes in HORIZONS:
        contributions: list[dict[str, Any]] = []
        sector_map: dict[str, float] = {}
        for symbol, (weight, sector) in ordered:
            scored = by_symbol.get(symbol)
            expected_move = num((scored or {}).get("remaining_expected_move"))
            if expected_move is None:
                expected_move = 0.0
            fraction = _realization_fraction(scored, minutes) if scored else 0.0
            horizon_return = expected_move * fraction
            contribution = weight * horizon_return
            sector_map[sector] = sector_map.get(sector, 0.0) + contribution
            contributions.append(
                {
                    "symbol": symbol,
                    "sector": sector,
                    "weight": weight,
                    "expected_stock_move": expected_move,
                    "realization_fraction": fraction,
                    "expected_return_at_horizon": horizon_return,
                    "spy_contribution": contribution,
                    "state": (scored or {}).get("state"),
                    "score": (scored or {}).get("score"),
                }
            )

        def aggregate(items: list[dict[str, Any]]) -> dict[str, Any]:
            return {
                "covered_weight": sum(row["weight"] for row in items),
                "bullish_weight": sum(row["weight"] for row in items if row["expected_return_at_horizon"] > 0),
                "bearish_weight": sum(row["weight"] for row in items if row["expected_return_at_horizon"] < 0),
                "neutral_weight": sum(row["weight"] for row in items if row["expected_return_at_horizon"] == 0),
                "net_expected_spy_return": sum(row["spy_contribution"] for row in items),
            }

        top = [row for row in contributions if row["symbol"] in top_symbols]
        positives = sorted(contributions, key=lambda row: row["spy_contribution"], reverse=True)[:10]
        negatives = sorted(contributions, key=lambda row: row["spy_contribution"])[:10]
        horizon_rows[name] = {
            "horizon_minutes": minutes,
            "full_index": aggregate(contributions),
            "top_125_by_weight": aggregate(top),
            "top_125_count": len(top),
            "top_positive_contributors": positives,
            "top_negative_contributors": negatives,
            "sector_contributions": [
                {"sector": sector, "spy_contribution": contribution}
                for sector, contribution in sorted(
                    sector_map.items(), key=lambda item: abs(item[1]), reverse=True
                )
            ],
        }

    return {
        "method": (
            "Each constituent's remaining model move is weighted by current S&P weight and "
            "its own expected realization window; the top 125 are reported separately, while "
            "the full current constituent set remains in the index calculation."
        ),
        "scored_constituents": len(scored_rows),
        "weighted_constituents": len(ordered),
        "top_125_count": min(125, len(ordered)),
        "horizons": horizon_rows,
    }


def _alpha_return(horizons: list[dict[str, Any]], *, name: str | None = None, minutes: int | None = None) -> float | None:
    row = V1.alpha_row(horizons, name=name, minutes=minutes)
    return num((row or {}).get("expected_return"))


def _beta_return(beta: dict[str, Any], minutes: int) -> float | None:
    row = V1.beta_row(beta, minutes)
    bps = num((row or {}).get("expected_return_bps"))
    return bps / 10000.0 if bps is not None else None


def _pressure_return(pressure: dict[str, Any], name: str) -> float | None:
    return num(
        pressure.get("horizons", {})
        .get(name, {})
        .get("full_index", {})
        .get("net_expected_spy_return")
    )


def _blend(parts: list[tuple[float | None, float]]) -> float | None:
    known = [(float(value), weight) for value, weight in parts if value is not None and weight > 0]
    total = sum(weight for _, weight in known)
    if not total:
        return None
    return sum(value * weight for value, weight in known) / total


def horizon_spy_predictor(
    alpha: dict[str, Any],
    beta: dict[str, Any],
    alpha_horizons: list[dict[str, Any]],
    pressure: dict[str, Any],
    chain: dict[str, Any],
    equity: float,
) -> dict[str, Any]:
    """Blend Alpha, Beta and matching-horizon constituent pressure into SPY."""
    p5 = _pressure_return(pressure, "open")
    p30 = _pressure_return(pressure, "30m")
    p120 = _pressure_return(pressure, "2h")
    pclose = _pressure_return(pressure, "close")
    p3d = _pressure_return(pressure, "3d")
    p5d = _pressure_return(pressure, "5d")

    a5 = _alpha_return(alpha_horizons, minutes=5)
    a30 = _alpha_return(alpha_horizons, minutes=30)
    a120 = _alpha_return(alpha_horizons, minutes=120)
    aeod = _alpha_return(alpha_horizons, name="EOD")
    a1d = _alpha_return(alpha_horizons, name="1D")
    a5d = _alpha_return(alpha_horizons, name="5D")
    b5, b30 = _beta_return(beta, 5), _beta_return(beta, 30)

    a3d = a1d + 0.5 * (a5d - a1d) if a1d is not None and a5d is not None else None

    model_moves = {
        "open": _blend([(a5, 0.45), (b5, 0.30), (p5, 0.25)]),
        "30m": _blend([(a30, 0.45), (b30, 0.30), (p30, 0.25)]),
        "2h": _blend([(a120, 0.60), (b30, 0.10), (p120, 0.30)]),
        "close": _blend([(aeod, 0.68), (b30, 0.05), (pclose, 0.27)]),
        "3d": _blend([(a3d, 0.78), (p3d, 0.22)]),
        "5d": _blend([(a5d, 0.82), (p5d, 0.18)]),
    }

    consensus = V1.BASE.consensus(alpha, beta) or {}
    agreement = num(consensus.get("horizon_match_rate"))
    confidence = V1.mean(
        [
            num(alpha.get("directional_confidence")),
            num(alpha.get("trust_score")),
            num(beta.get("confidence")),
            num(beta.get("coverage_ratio")),
        ]
    ) or 0.0
    if agreement is not None:
        confidence = clamp(0.80 * confidence + 0.20 * agreement)

    price = num(alpha.get("price")) or num(beta.get("price"))
    rows: list[dict[str, Any]] = []
    for name, minutes in HORIZONS:
        move = model_moves.get(name)
        market_mag, market_source = V1.spy_market_move(alpha, chain, minutes)
        h_conf = confidence * (0.82 if name in {"3d", "5d"} else 1.0)
        if move is None:
            h_conf *= 0.4
        pressure_row = pressure.get("horizons", {}).get(name, {})
        rows.append(
            {
                "name": name,
                "horizon_minutes": minutes,
                "predicted_move": move,
                "predicted_level": price * (1 + move) if price and move is not None else None,
                "direction": (
                    "BULLISH" if (move or 0) > 0 else "BEARISH" if (move or 0) < 0 else "UNAVAILABLE"
                ),
                "confidence": h_conf,
                "constituent_pressure": num(
                    pressure_row.get("full_index", {}).get("net_expected_spy_return")
                ),
                "top_125_pressure": num(
                    pressure_row.get("top_125_by_weight", {}).get("net_expected_spy_return")
                ),
                "options_implied_magnitude": market_mag,
                "options_implied_source": market_source,
                "model_minus_market_magnitude": (
                    abs(move) - market_mag
                    if move is not None and market_mag is not None
                    else None
                ),
                "note": (
                    "Options implied magnitude is directionless. Delta direction is the blend of "
                    "Alpha, Beta and horizon-matched weighted constituent forecasts."
                ),
            }
        )

    primary = next((row for row in rows if row["name"] == "2h" and row["predicted_move"] is not None), None)
    if primary is None:
        primary = next((row for row in rows if row["name"] == "30m" and row["predicted_move"] is not None), None)

    state = "NO_TRADE"
    if primary:
        magnitude = abs(float(primary["predicted_move"]))
        conf = float(primary["confidence"])
        market = num(primary.get("options_implied_magnitude"))
        edge_ok = market is None or magnitude >= max(0.70 * market, 0.0025)
        if conf >= 0.78 and magnitude >= 0.0045 and edge_ok:
            state = "ACTIVE_A+"
        elif conf >= 0.66 and magnitude >= 0.0030 and edge_ok:
            state = "ACTIVE_B"
        elif conf >= 0.55 and magnitude >= 0.0020:
            state = "ARMED"

    plan = {"available": False}
    if primary and state != "NO_TRADE":
        plan = V1.spy_plan(
            beta,
            float(primary["predicted_move"]),
            int(primary["horizon_minutes"]),
            state if state in {"ACTIVE_A+", "ACTIVE_B"} else "ARMED",
            float(primary["confidence"]),
            equity,
        )

    return {
        "state": state,
        "price": price,
        "horizons": rows,
        "primary_trade_horizon": primary["name"] if primary else None,
        "primary_direction": primary["direction"] if primary else None,
        "confidence": primary["confidence"] if primary else confidence,
        "trade": plan,
        "spy_options": chain,
        "method": (
            "Read-only Delta blend. Constituents use their own expected realization windows; "
            "Alpha/Beta remain independent and retain all execution authority outside Delta."
        ),
    }


def _resize_trade_for_risk(row: dict[str, Any], new_risk: float, equity: float) -> None:
    trade = row.get("trade") if isinstance(row.get("trade"), dict) else None
    if not trade or not trade.get("available"):
        return
    price = num(row.get("price")) or 0.0
    stop = num(trade.get("stop_invalidation"))
    if price <= 0 or stop is None:
        trade["suggested_risk_dollars"] = round(new_risk, 2)
        return
    stop_distance = abs(price - stop)
    capital_cap = min(30000.0, equity * 0.30) if row.get("symbol") == "SPY" else min(25000.0, equity * 0.25)
    shares_by_risk = math.floor(new_risk / stop_distance) if stop_distance > 0 and new_risk > 0 else 0
    shares_by_capital = math.floor(capital_cap / price)
    shares = max(0, min(shares_by_risk, shares_by_capital))
    trade["suggested_risk_dollars"] = round(new_risk, 2)
    trade["suggested_shares"] = shares
    trade["position_value"] = round(shares * price, 2)


def merge_running_with_risk(
    stocks: dict[str, Any], spy: dict[str, Any], equity: float
) -> dict[str, Any]:
    rows = [dict(row) for row in stocks.get("actionable_now", [])]
    if spy.get("state") in {"ACTIVE_A+", "ACTIVE_B"}:
        horizon = next(
            (
                row
                for row in spy.get("horizons", [])
                if row.get("name") == spy.get("primary_trade_horizon")
            ),
            {},
        )
        rows.append(
            {
                "symbol": "SPY",
                "sector": "INDEX",
                "weight": 1.0,
                "price": spy.get("price"),
                "bias": "LONG" if spy.get("primary_direction") == "BULLISH" else "SHORT",
                "state": spy.get("state"),
                "score": round((num(spy.get("confidence")) or 0.0) * 100, 1),
                "model_expected_move": horizon.get("predicted_move"),
                "remaining_expected_move": horizon.get("predicted_move"),
                "options_implied_move": horizon.get("options_implied_magnitude"),
                "expected_realization_minutes": horizon.get("horizon_minutes"),
                "trade": dict(spy.get("trade") or {}),
                "catalyst": {
                    "catalyst_type": "INDEX_CONVERGENCE",
                    "headline": "Alpha/Beta/horizon-matched constituent convergence",
                },
            }
        )

    rows.sort(
        key=lambda row: (
            row.get("state") == "ACTIVE_A+",
            num(row.get("score")) or 0.0,
            abs(num(row.get("remaining_expected_move")) or 0.0),
        ),
        reverse=True,
    )

    requested = [num((row.get("trade") or {}).get("suggested_risk_dollars")) or 0.0 for row in rows]
    total_requested = sum(requested)
    cap = min(3000.0, equity * 0.03)
    scale = min(1.0, cap / total_requested) if total_requested > 0 else 1.0
    for row, risk in zip(rows, requested, strict=True):
        if risk > 0:
            adjusted = risk * scale
            _resize_trade_for_risk(row, adjusted, equity)
            if scale < 1.0:
                row["trade"]["portfolio_risk_adjustment"] = (
                    f"Scaled to {scale:.3f}x so simultaneous modeled risk stays at or below ${cap:,.0f}."
                )

    actual_risk = sum(
        num((row.get("trade") or {}).get("suggested_risk_dollars")) or 0.0 for row in rows
    )
    return {
        "actionable": rows[:12],
        "armed_watch": stocks.get("armed_watch", [])[:12],
        "extended_do_not_chase": stocks.get("extended_do_not_chase", [])[:8],
        "max_simultaneous_open_risk": cap,
        "requested_open_risk": total_requested,
        "suggested_open_risk": min(actual_risk, cap),
    }


def collect() -> dict[str, Any]:
    payload = V1.BASE.collect()
    alpha = payload.get("alpha") or {}
    beta = payload.get("beta") or {}
    alpha_horizons = V1.raw_alpha_horizons()
    spy_options, option_map, alpha_db = V1.alpha_options(alpha)
    gamma = V1.load_gamma()
    equity = num(os.getenv("COMPETITION_EQUITY")) or 100000.0

    if beta.get("available"):
        scored = _all_scored_stocks(beta, gamma, option_map, equity)
        stocks = V1.build_stock_list(beta, gamma, option_map, equity)
        pressure = horizon_constituent_pressure(beta, scored)
    else:
        scored = []
        stocks = {
            "scanned": 0,
            "actionable_now": [],
            "armed_watch": [],
            "extended_do_not_chase": [],
            "deteriorating": [],
            "counts": {},
        }
        pressure = {
            "method": "unavailable",
            "scored_constituents": 0,
            "weighted_constituents": 0,
            "top_125_count": 0,
            "horizons": {},
        }

    spy = horizon_spy_predictor(alpha, beta, alpha_horizons, pressure, spy_options, equity)
    running = merge_running_with_risk(stocks, spy, equity)

    payload.update(
        {
            "schema_version": 4,
            "generated_at": V1.BASE.iso_now(),
            "product": {"name": "SPY Command / Delta", "version": "2.1.0"},
            "competition": {
                "paper_equity": equity,
                "normal_risk_dollars": [500, 750],
                "a_plus_risk_dollars": [1000, 1250],
                "max_simultaneous_open_risk": min(3000.0, equity * 0.03),
                "minimum_preferred_reward_risk": 2.0,
                "execution": "research/paper only",
            },
            "gamma": gamma,
            "delta": {
                "spy_predictor": spy,
                "constituent_pressure": pressure,
                "stock_scanner": stocks,
                "running_list": running,
                "data_contract": {
                    "all_sp500_scanned": bool(
                        len(scored) >= 450 and (num(beta.get("coverage_ratio")) or 0.0) >= 0.90
                    ),
                    "weighted_constituent_count": pressure.get("weighted_constituents", 0),
                    "top_125_integrated": pressure.get("top_125_count", 0) >= 125,
                    "horizon_specific_constituent_pressure": True,
                    "options_telemetry_symbols": len(option_map),
                    "gamma_catalysts_loaded": len(gamma.get("events", [])),
                    "execution_authority": False,
                },
            },
            "options": {
                "spy": spy_options,
                "constituent_symbol_count": len(option_map),
                "alpha_db": alpha_db,
            },
            "safety": {
                "read_only_convergence": True,
                "feeds_back_into_alpha": False,
                "feeds_back_into_beta": False,
                "places_orders": False,
                "options_implied_moves_are_directionless": True,
                "broker_accounts_exposed_in_chatgpt_view": False,
            },
        }
    )
    payload.setdefault("sources", {})["alpha_db"] = alpha_db
    payload["sources"]["gamma"] = gamma.get("source")
    return payload


def chatgpt_view(payload: dict[str, Any]) -> dict[str, Any]:
    compact = V1.chatgpt_view(payload)
    compact["schema_version"] = "chatgpt-market-state-v2"
    compact["constituent_pressure"] = payload["delta"]["constituent_pressure"]
    compact["instructions"] = {
        "interpretation": (
            "Use with fresh public news. Options implied moves are magnitude only. "
            "Recheck current price before presenting any entry, stop or target."
        ),
        "promotion": (
            "Prefer ACTIVE_A+/ACTIVE_B. ARMED/CONFIRMING require fresh catalyst or "
            "live confirmation. EXTENDED is do-not-chase."
        ),
    }
    return compact


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default=V1.OUTPUT)
    parser.add_argument("--chatgpt-output", default=V1.CHATGPT_OUTPUT)
    parser.add_argument("--stdout", action="store_true")
    args = parser.parse_args()
    payload = collect()
    if args.stdout:
        print(json.dumps(payload, indent=2))
    else:
        V1.write_atomic(Path(args.output), payload)
        V1.write_atomic(Path(args.chatgpt_output), chatgpt_view(payload))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
