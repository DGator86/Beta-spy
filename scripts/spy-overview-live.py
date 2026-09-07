#!/usr/bin/env python3
"""Final live launcher for the read-only SPY Command Delta bridge.

Loads the horizon-aware Delta implementation, applies live-feed correctness
rules that are deployment-specific, and publishes a compact quantitative row for
all scanned constituents so ChatGPT can marry newly discovered news to the live
market state even when a ticker was not already top-ranked.
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import math
import os
from pathlib import Path
from typing import Any


def _load_delta():
    candidates = [
        os.getenv("SPY_OVERVIEW_DELTA_V2"),
        "/usr/local/lib/spy-overview-delta-v2.py",
        str(Path(__file__).with_name("spy-overview-delta-v2.py")),
    ]
    for value in candidates:
        if not value:
            continue
        path = Path(value)
        if not path.is_file():
            continue
        spec = importlib.util.spec_from_file_location("spy_overview_delta_v2", path)
        if spec is None or spec.loader is None:
            continue
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module
    raise RuntimeError("horizon-aware Delta module is unavailable")


D = _load_delta()
num = D.num


def horizon_options_magnitude(
    alpha: dict[str, Any], chain: dict[str, Any], minutes: int
) -> tuple[float | None, str | None]:
    """Return an options-implied magnitude whose tenor actually covers horizon.

    A 0DTE/short-tenor ATM straddle is valid only while its remaining tenor
    covers the requested forecast.  Longer horizons use SPY IV scaled to the
    horizon instead of pretending the short option prices multi-day movement.
    """
    straddle = num(chain.get("atm_straddle_pct"))
    ttl = num(chain.get("minutes_to_expiry"))
    age = num(chain.get("age_seconds"))
    if (
        chain.get("available")
        and straddle is not None
        and ttl is not None
        and ttl > 0
        and minutes <= ttl
        and (age is None or age <= 180)
    ):
        return (
            straddle * math.sqrt(max(minutes, 1) / ttl),
            "live ATM straddle scaled within remaining tenor (directionless)",
        )

    iv = num(alpha.get("spy_iv"))
    if iv is not None and iv > 0:
        return (
            iv * math.sqrt(max(minutes, 1) / (252 * 390)),
            "SPY IV scaled to requested horizon (directionless)",
        )
    return None, None


# V2 deliberately calls through V1 for this function. Patch only this pure
# magnitude function; no Alpha/Beta state or execution method is modified.
D.V1.spy_market_move = horizon_options_magnitude


def _compact_constituent(
    raw: dict[str, Any], scored: dict[str, Any] | None, iv: dict[str, Any] | None
) -> dict[str, Any]:
    flow = raw.get("flow") if isinstance(raw.get("flow"), dict) else {}
    structure = raw.get("structure") if isinstance(raw.get("structure"), dict) else {}
    auction = raw.get("auction") if isinstance(raw.get("auction"), dict) else {}
    row: dict[str, Any] = {
        "symbol": raw.get("symbol"),
        "sector": raw.get("sector"),
        "weight": num(raw.get("weight")),
        "price": num(raw.get("close")),
        "return_1m": num(raw.get("return_1m")),
        "return_5m": num(raw.get("return_5m")),
        "return_15m": num(raw.get("return_15m")),
        "vwap": num(raw.get("vwap")),
        "vwap_distance_bps": num(raw.get("vwap_distance_bps")),
        "ema8": num(raw.get("ema8")),
        "ema21": num(raw.get("ema21")),
        "rsi14": num(raw.get("rsi14")),
        "atr14_bps": num(raw.get("atr14_bps")),
        "relative_volume20": num(raw.get("relative_volume20")),
        "range_expansion": num(raw.get("range_expansion")),
        "order_flow_imbalance": num(flow.get("order_flow_imbalance")),
        "quote_imbalance": num(flow.get("quote_imbalance")),
        "structure_score": num(structure.get("structure_score")),
        "structure_break_strength": num(structure.get("structure_break_strength")),
        "cvd_slope_5m": num(auction.get("cvd_slope_5m")),
        "atm_iv": num((iv or {}).get("atm_iv")),
        "iv_skew": num((iv or {}).get("skew")),
    }
    if scored:
        row.update(
            {
                "state": scored.get("state"),
                "bias": scored.get("bias"),
                "score": num(scored.get("score")),
                "convergence": num(scored.get("convergence")),
                "signal_agreement_pct": num(scored.get("signal_agreement_pct")),
                "predictive_strength": num(scored.get("predictive_strength")),
                "confirmation_strength": num(scored.get("confirmation_strength")),
                "model_expected_move": num(scored.get("model_expected_move")),
                "remaining_expected_move": num(scored.get("remaining_expected_move")),
                "options_implied_move": num(scored.get("options_implied_move")),
                "expected_realization_minutes": scored.get("expected_realization_minutes"),
            }
        )
    return row


def _constituent_index(payload: dict[str, Any]) -> list[dict[str, Any]]:
    beta = payload.get("beta") if isinstance(payload.get("beta"), dict) else {}
    if not beta.get("available"):
        return []
    alpha = payload.get("alpha") if isinstance(payload.get("alpha"), dict) else {}
    gamma = payload.get("gamma") if isinstance(payload.get("gamma"), dict) else {"by_symbol": {}}
    equity = num(payload.get("competition", {}).get("paper_equity")) or 100000.0
    _, iv_map, _ = D.V1.alpha_options(alpha)
    scored_rows = D._all_scored_stocks(beta, gamma, iv_map, equity)
    scored_map = {str(row.get("symbol")): row for row in scored_rows}
    output: list[dict[str, Any]] = []
    for raw in beta.get("symbols", []):
        if not isinstance(raw, dict):
            continue
        symbol = str(raw.get("symbol") or "")
        if not symbol or symbol == "SPY":
            continue
        output.append(_compact_constituent(raw, scored_map.get(symbol), iv_map.get(symbol)))
    output.sort(key=lambda row: num(row.get("weight")) or 0.0, reverse=True)
    return output


def collect() -> dict[str, Any]:
    payload = D.collect()
    signals = _constituent_index(payload)
    payload["delta"]["constituent_signals"] = signals
    payload["delta"]["data_contract"]["chatgpt_constituent_rows"] = len(signals)
    payload["delta"]["data_contract"]["short_tenor_options_not_used_beyond_expiry"] = True
    return payload


def chatgpt_view(payload: dict[str, Any]) -> dict[str, Any]:
    compact = D.chatgpt_view(payload)
    compact["schema_version"] = "chatgpt-market-state-v3"
    compact["constituents"] = payload.get("delta", {}).get("constituent_signals", [])
    compact["data_health"]["chatgpt_constituent_rows"] = len(compact["constituents"])
    compact["instructions"] = {
        "interpretation": (
            "Scan fresh public news across the S&P 500, join news by ticker to the constituent rows, "
            "and use Delta/Alpha/Beta convergence for confirmation. Options implied moves are magnitude only."
        ),
        "entries": (
            "Recheck the current price before presenting entry/stop/targets. Use live VWAP, ATR, flow, "
            "relative volume, structure and sector confirmation. Do not chase EXTENDED names."
        ),
        "promotion": (
            "ACTIVE_A+/ACTIVE_B are directly actionable research states. CONFIRMING/ARMED/PREDICTIVE_SETUP "
            "can be promoted only when fresh external catalyst evidence supports the same direction."
        ),
    }
    return compact


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default=D.V1.OUTPUT)
    parser.add_argument("--chatgpt-output", default=D.V1.CHATGPT_OUTPUT)
    parser.add_argument("--stdout", action="store_true")
    args = parser.parse_args()
    payload = collect()
    if args.stdout:
        print(json.dumps(payload, indent=2))
    else:
        D.V1.write_atomic(Path(args.output), payload)
        D.V1.write_atomic(Path(args.chatgpt_output), chatgpt_view(payload))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
