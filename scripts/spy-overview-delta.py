#!/usr/bin/env python3
"""Read-only Delta convergence layer for SPY Command.

The existing Alpha-SPY and Beta-spy engines remain independent and unchanged.
This process reads their published state plus Alpha's SQLite option telemetry,
optionally reads a Gamma catalyst feed, and publishes a competition-oriented
running list and SPY predictor. It never places orders or feeds values back into
Alpha/Beta.
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import math
import os
import sqlite3
import tempfile
from datetime import UTC, datetime
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo

ET = ZoneInfo("America/New_York")
OUTPUT = os.getenv("OVERVIEW_STATUS_PATH", "/var/www/spy-overview/status.json")
CHATGPT_OUTPUT = os.getenv("CHATGPT_STATUS_PATH", "/var/www/spy-overview/chatgpt.json")
GAMMA_PATH = os.getenv("GAMMA_CATALYSTS_PATH", "/var/lib/spy-overview/gamma-catalysts.json")
GAMMA_URL = os.getenv("GAMMA_CATALYSTS_URL", "")


def _load_base():
    candidates = [
        os.getenv("SPY_OVERVIEW_BASE"),
        "/usr/local/lib/spy-overview-base.py",
        str(Path(__file__).with_name("spy-overview-status.py")),
    ]
    for value in candidates:
        if not value:
            continue
        path = Path(value)
        if not path.is_file():
            continue
        spec = importlib.util.spec_from_file_location("spy_overview_base", path)
        if spec is None or spec.loader is None:
            continue
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module
    raise RuntimeError("SPY Command base collector is unavailable")


BASE = _load_base()
num = BASE.num
clamp = BASE.clamp
iso_now = BASE.iso_now


def clip(value: float, lo: float = -1.0, hi: float = 1.0) -> float:
    return max(lo, min(hi, value))


def mean(values: list[float | None]) -> float | None:
    known = [float(v) for v in values if v is not None and math.isfinite(float(v))]
    return sum(known) / len(known) if known else None


def parse_time(value: Any) -> datetime | None:
    if value is None:
        return None
    try:
        dt = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
    except ValueError:
        return None
    return (dt if dt.tzinfo else dt.replace(tzinfo=UTC)).astimezone(UTC)


def age_seconds(value: Any) -> float | None:
    dt = parse_time(value)
    return max(0.0, (datetime.now(UTC) - dt).total_seconds()) if dt else None


def first_file(env: str, defaults: list[str]) -> Path | None:
    candidates = ([os.environ[env]] if os.getenv(env) else []) + defaults
    return next((Path(x) for x in candidates if x and Path(x).exists()), None)


def direction_sign(value: Any) -> float:
    text = str(value or "").strip().upper()
    if text in {"BULLISH", "LONG", "UP", "BUY", "1", "+1"}:
        return 1.0
    if text in {"BEARISH", "SHORT", "DOWN", "SELL", "-1"}:
        return -1.0
    value_num = num(value)
    return 1.0 if (value_num or 0) > 0 else -1.0 if (value_num or 0) < 0 else 0.0


def write_atomic(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent, text=True)
    try:
        os.fchmod(fd, 0o644)
        with os.fdopen(fd, "w") as handle:
            json.dump(payload, handle, separators=(",", ":"))
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(tmp, path)
    finally:
        try:
            os.unlink(tmp)
        except FileNotFoundError:
            pass


# ---------- Alpha forecast + option telemetry ----------

HORIZON_MINUTES = {"5m": 5, "15m": 15, "30m": 30, "60m": 60, "120m": 120, "eod": 390, "1d": 390, "5d": 1950}


def raw_alpha_horizons() -> list[dict[str, Any]]:
    token = os.getenv("ALPHA_DASHBOARD_TOKEN") or os.getenv("DASHBOARD_VIEW_TOKEN")
    raw, _ = BASE.fetch_json(BASE.ALPHA_URL, token)
    source = (raw or {}).get("forecast_horizons") or {}
    rows = []
    if not isinstance(source, dict):
        return rows
    for name, item in source.items():
        if not isinstance(item, dict):
            continue
        minutes = num(item.get("horizon_minutes"))
        if minutes is None:
            minutes = HORIZON_MINUTES.get(str(name).lower())
        if minutes is None:
            continue
        rows.append({
            "name": str(name),
            "horizon_minutes": int(minutes),
            "expected_return": num(item.get("expected_return")),
            "probability_up": num(item.get("probability_up")),
            "predicted_price": num(item.get("predicted_price")),
            "predicted_low": num(item.get("predicted_low")),
            "predicted_high": num(item.get("predicted_high")),
            "sigma_return": num(item.get("sigma_return")),
            "integrity": item.get("integrity"),
        })
    return sorted(rows, key=lambda row: (row["horizon_minutes"], row["name"]))


def alpha_options(alpha: dict[str, Any]) -> tuple[dict[str, Any], dict[str, dict[str, Any]], str | None]:
    db = first_file("ALPHA_DB", ["/var/lib/alpha-spy/journal/alpha-spy.db", "/opt/alpha-spy/data/alpha-spy.db"])
    if not db:
        return {"available": False, "reason": "alpha_db_not_found"}, {}, None
    try:
        con = sqlite3.connect(f"file:{db}?mode=ro", uri=True, timeout=2)
        con.row_factory = sqlite3.Row
        chain = con.execute(
            "SELECT * FROM option_chain_snapshots WHERE underlying='SPY' AND purpose='strategy' ORDER BY captured_at DESC LIMIT 1"
        ).fetchone()
        spy: dict[str, Any] = {"available": False, "source": "alpha_sqlite"}
        if chain:
            q = con.execute(
                "SELECT * FROM option_quotes WHERE chain_snapshot_id=? ORDER BY right,strike",
                (chain["chain_snapshot_id"],),
            ).fetchall()
            spot = num(chain["underlying_price"]) or num(alpha.get("price"))
            calls, puts = [r for r in q if r["right"] == "C"], [r for r in q if r["right"] == "P"]
            call = min(calls, key=lambda r: abs(float(r["strike"]) - float(spot)), default=None) if spot else None
            put = min(puts, key=lambda r: abs(float(r["strike"]) - float(spot)), default=None) if spot else None

            def mid(row):
                if row is None:
                    return None
                value = num(row["midpoint"])
                if value and value > 0:
                    return value
                bid, ask = num(row["bid"]), num(row["ask"])
                return (bid + ask) / 2 if bid is not None and ask is not None else ask or bid

            cm, pm = mid(call), mid(put)
            straddle = cm + pm if cm is not None and pm is not None else None
            minutes_to_expiry = None
            try:
                d = datetime.fromisoformat(str(chain["expiration"])).date()
                close = datetime(d.year, d.month, d.day, 16, 0, tzinfo=ET)
                minutes_to_expiry = max(0.0, (close.astimezone(UTC) - datetime.now(UTC)).total_seconds() / 60)
            except ValueError:
                pass
            call_vol, put_vol = sum(float(r["volume"] or 0) for r in calls), sum(float(r["volume"] or 0) for r in puts)
            call_oi, put_oi = sum(float(r["open_interest"] or 0) for r in calls), sum(float(r["open_interest"] or 0) for r in puts)
            spy = {
                "available": bool(straddle and spot),
                "source": "alpha_sqlite_spy_chain",
                "captured_at": chain["captured_at"],
                "age_seconds": age_seconds(chain["captured_at"]),
                "expiration": chain["expiration"],
                "minutes_to_expiry": minutes_to_expiry,
                "underlying_price": spot,
                "atm_strike": num(call["strike"]) if call is not None else num(put["strike"]) if put is not None else None,
                "atm_straddle": straddle,
                "atm_straddle_pct": straddle / spot if straddle is not None and spot else None,
                "call_put_volume_ratio": call_vol / put_vol if put_vol else None,
                "call_put_oi_ratio": call_oi / put_oi if put_oi else None,
                "quote_count": len(q),
                "integrity": chain["integrity"],
                "note": "ATM straddle is directionless magnitude to expiry.",
            }
        rows = con.execute(
            """SELECT o.* FROM constituent_iv_observations o JOIN (
                   SELECT symbol,MAX(captured_at) captured_at FROM constituent_iv_observations GROUP BY symbol
               ) x ON x.symbol=o.symbol AND x.captured_at=o.captured_at ORDER BY o.weight DESC"""
        ).fetchall()
        con.close()
        options = {}
        for row in rows:
            iv = num(row["atm_iv"])
            if not row["symbol"] or not iv or iv <= 0:
                continue
            options[str(row["symbol"])] = {
                "captured_at": row["captured_at"],
                "age_seconds": age_seconds(row["captured_at"]),
                "atm_iv": iv,
                "skew": num(row["skew"]),
                "expiration": row["expiration"],
                "dte": int(row["dte"] or 0),
                "iv_implied_move_1d": iv / math.sqrt(252),
                "iv_implied_move_5d": iv * math.sqrt(5 / 252),
                "integrity": row["integrity"],
            }
        return spy, options, str(db)
    except sqlite3.Error as exc:
        return {"available": False, "reason": f"alpha_db_error:{exc}"}, {}, str(db)


# ---------- Gamma catalyst contract ----------
def load_gamma() -> dict[str, Any]:
    raw = None
    source: dict[str, Any] | None = None
    if GAMMA_URL:
        raw, source = BASE.fetch_json(GAMMA_URL)
    elif Path(GAMMA_PATH).is_file():
        try:
            raw = json.loads(Path(GAMMA_PATH).read_text(encoding="utf-8"))
            source = {"ok": True, "type": "file", "path": GAMMA_PATH}
        except (OSError, json.JSONDecodeError) as exc:
            source = {"ok": False, "type": "file", "path": GAMMA_PATH, "error": str(exc)}
    if not isinstance(raw, dict):
        return {"available": False, "source": source, "events": [], "by_symbol": {}}
    events = raw.get("events") if isinstance(raw.get("events"), list) else []
    out, by_symbol = [], {}
    half_life = {"FDA": 240, "CLINICAL": 360, "M&A": 360, "EARNINGS": 360, "GUIDANCE": 480, "ANALYST": 240, "MACRO": 180, "PEER": 180}
    for row in events:
        if not isinstance(row, dict):
            continue
        symbol = str(row.get("symbol") or "").upper().strip()
        if not symbol:
            continue
        strength = clamp(num(row.get("strength")) if num(row.get("strength")) is not None else 0.7)
        materiality = clamp(num(row.get("materiality")) if num(row.get("materiality")) is not None else strength)
        novelty = clamp(num(row.get("novelty")) if num(row.get("novelty")) is not None else strength)
        confidence = clamp(num(row.get("confidence")) if num(row.get("confidence")) is not None else 0.7)
        kind = str(row.get("catalyst_type") or "OTHER").upper()
        published = row.get("published_at") or row.get("timestamp")
        age_min = (age_seconds(published) or 0) / 60 if published else None
        hl = num(row.get("half_life_minutes")) or half_life.get(kind, 360)
        decay = 0.5 ** (age_min / max(hl, 1)) if age_min is not None else 0.75
        event = {
            "symbol": symbol,
            "published_at": published,
            "age_minutes": age_min,
            "direction": str(row.get("direction") or "NEUTRAL").upper(),
            "direction_sign": direction_sign(row.get("direction")),
            "strength": strength,
            "materiality": materiality,
            "novelty": novelty,
            "confidence": confidence,
            "effective_strength": clamp(strength * materiality * novelty * confidence * decay),
            "catalyst_type": kind,
            "headline": row.get("headline") or row.get("title"),
            "source": row.get("source"),
            "url": row.get("url"),
            "expected_horizon_minutes": int(num(row.get("expected_horizon_minutes")) or 0) or None,
            "reference_price": num(row.get("reference_price")),
        }
        out.append(event)
        prior = by_symbol.get(symbol)
        if prior is None or event["effective_strength"] > prior["effective_strength"]:
            by_symbol[symbol] = event
    out.sort(key=lambda r: r["effective_strength"], reverse=True)
    return {"available": bool(out), "source": source, "generated_at": raw.get("generated_at"), "events": out, "by_symbol": by_symbol}


# ---------- Constituent scoring ----------
def nested(row: dict[str, Any], key: str) -> dict[str, Any]:
    return row.get(key) if isinstance(row.get(key), dict) else {}


def components(row: dict[str, Any], sector: dict[str, Any] | None) -> dict[str, float | None]:
    r1, r5, r15 = num(row.get("return_1m")), num(row.get("return_5m")), num(row.get("return_15m"))
    vd, e8, e21 = num(row.get("vwap_distance_bps")), num(row.get("ema8")), num(row.get("ema21"))
    slope, rsi = num(row.get("ema8_slope_bps")), num(row.get("rsi14"))
    rv, expansion = num(row.get("relative_volume20")), num(row.get("range_expansion"))
    flow, st, auc = nested(row, "flow"), nested(row, "structure"), nested(row, "auction")
    trend = mean([
        clip(vd / 15) if vd is not None else None,
        clip((e8 / e21 - 1) * 1000) if e8 and e21 else None,
        clip(slope / 2) if slope is not None else None,
        clip(r5 * 10000 / 15) if r5 is not None else None,
    ])
    momentum = mean([
        clip(r1 * 10000 / 8) if r1 is not None else None,
        clip(r5 * 10000 / 20) if r5 is not None else None,
        clip(r15 * 10000 / 35) if r15 is not None else None,
        clip((rsi - 50) / 20) if rsi is not None else None,
    ])
    sign = 1 if (r1 or r5 or 0) >= 0 else -1
    volume = mean([
        sign * clip((rv - 1) / 1.5) if rv is not None else None,
        sign * clip((expansion - 1) / 1.5) if expansion is not None else None,
    ])
    ofi, qi = num(flow.get("order_flow_imbalance")), num(flow.get("quote_imbalance"))
    impact, absorption = num(flow.get("price_impact_bps_per_10k")), num(flow.get("absorption"))
    buy, sell = num(flow.get("initiative_buy_efficiency")), num(flow.get("initiative_sell_efficiency"))
    flow_score = mean([
        clip(ofi) if ofi is not None else None,
        clip(qi) if qi is not None else None,
        clip(impact / 3) if impact is not None else None,
        -clip(absorption) * (1 if (ofi or 0) >= 0 else -1) if absorption is not None else None,
        clip((buy or 0) - (sell or 0)) if buy is not None or sell is not None else None,
    ])
    ss = num(st.get("structure_score"))
    ss = num(st.get("structure_state")) if ss is None else ss
    acceptance = (num(st.get("acceptance_above_score")) or 0) - (num(st.get("acceptance_below_score")) or 0)
    sweep = (num(st.get("sweep_low_score")) or 0) - (num(st.get("sweep_high_score")) or 0)
    structure = mean([clip(ss) if ss is not None else None, clip(acceptance), clip(sweep), clip(num(auc.get("price_cvd_divergence_5m")) or 0)])
    sector_score = mean([num(sector.get(k)) for k in ("trend", "momentum", "flow", "participation")]) if sector else None
    return {"trend": trend, "momentum": momentum, "volume": volume, "flow": flow_score, "structure": structure, "sector": sector_score,
            "technical": mean([trend, momentum, volume, flow_score, structure, sector_score])}


def implied_move(iv: dict[str, Any] | None, minutes: int) -> float | None:
    value = num((iv or {}).get("atm_iv"))
    return value * math.sqrt(max(minutes, 1) / (252 * 390)) if value and value > 0 else None


def expected_magnitude(row: dict[str, Any], event: dict[str, Any] | None, minutes: int, iv: dict[str, Any] | None) -> tuple[float, float | None]:
    atr = (num(row.get("atr14_bps")) or 15) / 10000
    rv = (num(row.get("realized_vol20_bps")) or 10) / 10000
    r15 = abs(num(row.get("return_15m")) or 0)
    mag = max(r15 * min(2.5, math.sqrt(minutes / 15)), atr * math.sqrt(minutes / 14), rv * math.sqrt(minutes / 20), 0.0015)
    mag *= 1 + 0.15 * clamp(((num(row.get("relative_volume20")) or 1) - 1) / 2)
    mag *= 1 + (0.9 * event["effective_strength"] if event else 0)
    market = implied_move(iv, minutes)
    if market is not None:
        mag = min(mag, market * (2.2 if event and event["effective_strength"] >= 0.6 else 1.35))
        mag = max(mag, min(market * 0.35, 0.02))
    return min(mag, 0.20 if minutes <= 390 else 0.35), market


def trade_plan(row: dict[str, Any], side: str, remaining: float, minutes: int, state: str, equity: float) -> dict[str, Any]:
    price = num(row.get("close")) or 0
    if price <= 0:
        return {"available": False}
    atr = price * max((num(row.get("atr14_bps")) or 20) / 10000, 0.0015)
    stop_dist = max(0.85 * atr, price * 0.0035)
    target_dist = max(abs(remaining) * price, atr)
    s = 1 if side == "LONG" else -1
    vwap = num(row.get("vwap"))
    zone = [price - 0.2 * atr, price + 0.1 * atr] if s > 0 else [price - 0.1 * atr, price + 0.2 * atr]
    confirmation = price + s * 0.25 * atr
    if vwap is not None:
        confirmation = max(confirmation, vwap + 0.1 * atr) if s > 0 else min(confirmation, vwap - 0.1 * atr)
    stop = price - s * stop_dist
    t1, t2, t3 = price + s * stop_dist, price + s * min(target_dist * 0.75, 2 * stop_dist), price + s * target_dist
    rr = target_dist / stop_dist
    risk = (min(1250, max(1000, equity * 0.01)) if state == "ACTIVE_A+" else min(750, max(500, equity * 0.006)) if state == "ACTIVE_B" else 0)
    shares = min(math.floor(risk / stop_dist) if risk else 0, math.floor(min(25000, equity * 0.25) / price))
    hold, time_stop = ("10–45 minutes", 30) if minutes <= 30 else ("30 minutes–3 hours", 90) if minutes <= 120 else ("1–6 hours", 150) if minutes <= 390 else ("1–3 sessions", min(minutes, 1170)) if minutes <= 1170 else ("3–5 sessions", 1950)
    return {
        "available": True,
        "primary_entry_zone": [round(x, 4) for x in zone],
        "confirmation_entry": round(confirmation, 4),
        "retest_entry": round(vwap if vwap is not None else price - s * 0.1 * atr, 4),
        "do_not_chase": round(price + s * max(1.2 * atr, 0.55 * target_dist), 4),
        "stop_invalidation": round(stop, 4),
        "targets": {"t1": round(t1, 4), "t2": round(t2, 4), "t3": round(t3, 4)},
        "reward_risk_to_model_target": round(rr, 2),
        "expected_hold": hold,
        "time_stop_minutes": time_stop,
        "time_exit": f"Exit/reassess after {time_stop} minutes if the move has not begun to realize.",
        "thesis_exit": "Exit early on VWAP/flow/sector reversal or contradictory catalyst; never widen the stop.",
        "suggested_risk_dollars": round(risk, 2),
        "suggested_shares": shares,
        "position_value": round(shares * price, 2),
    }


def score_stock(row: dict[str, Any], sector: dict[str, Any] | None, event: dict[str, Any] | None, iv: dict[str, Any] | None, equity: float) -> dict[str, Any] | None:
    ticker, price = str(row.get("symbol") or "").upper(), num(row.get("close"))
    if not ticker or ticker == "SPY" or not price:
        return None
    comp = components(row, sector)
    technical = num(comp.get("technical"))
    if technical is None:
        return None
    cat_power = event["effective_strength"] if event else 0
    cat_signed = (event["direction_sign"] * cat_power) if event else 0
    predictive = clip(0.62 * technical + 0.38 * cat_signed) if event else clip(technical)
    sign = 1 if predictive > 0 else -1 if predictive < 0 else 0
    if not sign:
        return None
    flow, st, auc = nested(row, "flow"), nested(row, "structure"), nested(row, "auction")
    confirms: list[tuple[str, float]] = []
    vd = num(row.get("vwap_distance_bps")); e8, e21 = num(row.get("ema8")), num(row.get("ema21")); r1 = num(row.get("return_1m"))
    if vd is not None: confirms.append(("vwap", 1 if vd > 0 else -1 if vd < 0 else 0))
    if e8 is not None and e21 is not None: confirms.append(("ema8_21", 1 if e8 > e21 else -1 if e8 < e21 else 0))
    if r1 is not None: confirms.append(("price_1m", 1 if r1 > 0 else -1 if r1 < 0 else 0))
    for name, value in (("order_flow", num(flow.get("order_flow_imbalance"))), ("quote_imbalance", num(flow.get("quote_imbalance")))):
        if value is not None: confirms.append((name, clip(value)))
    ss = num(st.get("structure_score")); ss = num(st.get("structure_state")) if ss is None else ss
    if ss is not None: confirms.append(("structure", clip(ss)))
    cvd = num(auc.get("cvd_slope_5m"))
    if cvd is not None: confirms.append(("cvd", 1 if cvd > 0 else -1 if cvd < 0 else 0))
    if comp.get("sector") is not None: confirms.append(("sector", clip(float(comp["sector"]))))
    if event and event["direction_sign"]: confirms.append(("catalyst", event["direction_sign"]))
    conf_strength = mean([clamp(sign * v) for _, v in confirms]) or 0
    agreement = sum(1 for _, v in confirms if sign * v > 0) / len(confirms) if confirms else 0
    rvol = num(row.get("relative_volume20")) or 1
    cat_align = cat_power if event and event["direction_sign"] == sign else 0
    cat_conflict = cat_power if event and event["direction_sign"] == -sign else 0
    convergence = clamp(0.32 * abs(predictive) + 0.27 * conf_strength + 0.20 * cat_align + 0.13 * agreement + 0.08 * clamp((rvol - 1) / 2) - 0.35 * cat_conflict)
    minutes = int((event or {}).get("expected_horizon_minutes") or (30 if convergence >= 0.82 and rvol >= 1.5 else 120))
    mag, market_mag = expected_magnitude(row, event, minutes, iv)
    model_move = sign * mag
    ref = num((event or {}).get("reference_price")); realized = price / ref - 1 if ref else None
    remaining = sign * max(0, mag - max(0, sign * realized)) if realized is not None else model_move
    consumed = max(0, sign * realized) / max(mag, 1e-9) if realized is not None else 0
    rr_proxy = abs(remaining) / max(max((num(row.get("atr14_bps")) or 20) / 10000 * 0.85, 0.0035), 1e-9)
    state = "PREDICTIVE_SETUP"
    if cat_conflict >= 0.4: state = "DETERIORATING"
    elif realized is not None and consumed >= 0.8: state = "EXTENDED"
    elif event and cat_power >= 0.60 and convergence >= 0.82 and agreement >= 0.70 and rr_proxy >= 2: state = "ACTIVE_A+"
    elif event and cat_power >= 0.42 and convergence >= 0.70 and agreement >= 0.60 and rr_proxy >= 2: state = "ACTIVE_B"
    elif convergence >= 0.68 and conf_strength >= 0.58: state = "CONFIRMING"
    elif convergence >= 0.55: state = "ARMED"
    side = "LONG" if sign > 0 else "SHORT"
    plan = trade_plan(row, side, remaining, minutes, state, equity)
    if state in {"ACTIVE_A+", "ACTIVE_B"} and plan["reward_risk_to_model_target"] < 2:
        state, plan = "ARMED", trade_plan(row, side, remaining, minutes, "ARMED", equity)
    score = convergence * 100 * (0.55 if state == "EXTENDED" else 0.4 if state == "DETERIORATING" else 1)
    return {
        "symbol": ticker, "sector": row.get("sector"), "weight": num(row.get("weight")) or 0, "price": price,
        "bias": side, "state": state, "score": round(score, 1), "convergence": round(convergence * 100, 1),
        "signal_agreement_pct": round(agreement * 100, 1), "predictive_strength": round(abs(predictive) * 100, 1),
        "confirmation_strength": round(conf_strength * 100, 1), "catalyst": event, "options": iv,
        "options_implied_move": market_mag, "model_expected_move": model_move, "move_already_realized": realized,
        "remaining_expected_move": remaining, "model_minus_implied_magnitude": abs(model_move) - market_mag if market_mag is not None else None,
        "expected_realization_minutes": minutes, "trade": plan,
    }


def build_stock_list(beta: dict[str, Any], gamma: dict[str, Any], option_map: dict[str, dict[str, Any]], equity: float) -> dict[str, Any]:
    sectors = {str(x.get("sector")): x for x in beta.get("sectors", []) if isinstance(x, dict)}
    rows = []
    for raw in beta.get("symbols", []):
        if isinstance(raw, dict):
            scored = score_stock(raw, sectors.get(str(raw.get("sector"))), gamma["by_symbol"].get(str(raw.get("symbol") or "").upper()), option_map.get(str(raw.get("symbol") or "").upper()), equity)
            if scored: rows.append(scored)
    priority = {"ACTIVE_A+": 6, "ACTIVE_B": 5, "CONFIRMING": 4, "ARMED": 3, "PREDICTIVE_SETUP": 2, "EXTENDED": 0, "DETERIORATING": -1}
    rows.sort(key=lambda r: (priority.get(r["state"], -2), r["score"], abs(r["remaining_expected_move"])), reverse=True)
    seen = set()
    for r in rows:
        trade, sector = r["trade"], str(r.get("sector") or "UNKNOWN")
        if trade.get("suggested_risk_dollars", 0) and sector in seen:
            trade["suggested_risk_dollars"] = round(trade["suggested_risk_dollars"] * 0.6, 2)
            stop = abs(r["price"] - trade["stop_invalidation"])
            trade["suggested_shares"] = min(math.floor(trade["suggested_risk_dollars"] / stop), math.floor(25000 / r["price"])) if stop else 0
            trade["position_value"] = round(trade["suggested_shares"] * r["price"], 2)
            trade["correlation_adjustment"] = "60% risk: higher-ranked same-sector setup already present."
        elif trade.get("suggested_risk_dollars", 0):
            seen.add(sector); trade["correlation_adjustment"] = "Full model risk: first actionable setup in sector."
    return {
        "scanned": len(rows),
        "actionable_now": [r for r in rows if r["state"] in {"ACTIVE_A+", "ACTIVE_B"}][:10],
        "armed_watch": [r for r in rows if r["state"] in {"CONFIRMING", "ARMED", "PREDICTIVE_SETUP"}][:15],
        "extended_do_not_chase": [r for r in rows if r["state"] == "EXTENDED"][:10],
        "deteriorating": [r for r in rows if r["state"] == "DETERIORATING"][:10],
        "counts": {name: sum(r["state"] == name for r in rows) for name in ("ACTIVE_A+", "ACTIVE_B", "CONFIRMING", "ARMED", "PREDICTIVE_SETUP", "EXTENDED", "DETERIORATING")},
    }


# ---------- SPY constituent pressure + Delta ----------
def constituent_pressure(beta: dict[str, Any]) -> dict[str, Any]:
    sectors = {str(x.get("sector")): x for x in beta.get("sectors", []) if isinstance(x, dict)}
    rows, by_sector = [], {}
    for raw in beta.get("symbols", []):
        if not isinstance(raw, dict) or raw.get("symbol") == "SPY": continue
        weight, price = num(raw.get("weight")) or 0, num(raw.get("close")) or 0
        if weight <= 0 or price <= 0: continue
        tech = num(components(raw, sectors.get(str(raw.get("sector")))).get("technical")) or 0
        if abs(tech) < 0.03: predicted = 0.0
        else:
            atr, r15 = (num(raw.get("atr14_bps")) or 15) / 10000, abs(num(raw.get("return_15m")) or 0)
            predicted = (1 if tech > 0 else -1) * min(max(r15 * 1.35, atr * math.sqrt(30 / 14), 0.0005), 0.035) * min(1, 0.35 + 0.65 * abs(tech))
        contrib = weight * predicted; sector = str(raw.get("sector") or "UNKNOWN")
        by_sector[sector] = by_sector.get(sector, 0) + contrib
        rows.append({"symbol": raw.get("symbol"), "sector": sector, "weight": weight, "technical_score": tech, "expected_return_30m": predicted, "spy_contribution_30m": contrib})
    rows.sort(key=lambda r: r["weight"], reverse=True); top = rows[:125]
    def agg(items):
        return {"covered_weight": sum(r["weight"] for r in items), "bullish_weight": sum(r["weight"] for r in items if r["expected_return_30m"] > 0),
                "bearish_weight": sum(r["weight"] for r in items if r["expected_return_30m"] < 0), "neutral_weight": sum(r["weight"] for r in items if r["expected_return_30m"] == 0),
                "net_expected_spy_return_30m": sum(r["spy_contribution_30m"] for r in items)}
    return {"full_index": agg(rows), "top_125_by_weight": agg(top), "top_125_count": len(top),
            "top_positive_contributors": sorted(rows, key=lambda r: r["spy_contribution_30m"], reverse=True)[:10],
            "top_negative_contributors": sorted(rows, key=lambda r: r["spy_contribution_30m"])[:10],
            "sector_contributions": [{"sector": k, "spy_contribution_30m": v} for k, v in sorted(by_sector.items(), key=lambda x: abs(x[1]), reverse=True)]}


def alpha_row(horizons: list[dict[str, Any]], name: str | None = None, minutes: int | None = None):
    if name: return next((r for r in horizons if r["name"].lower() == name.lower()), None)
    return next((r for r in horizons if r["horizon_minutes"] == minutes), None)


def beta_row(beta: dict[str, Any], minutes: int):
    return next((r for r in beta.get("forecasts", []) if int(r.get("horizon_minutes") or 0) == minutes), None)


def blend(parts: list[tuple[float | None, float]]) -> float | None:
    known = [(v, w) for v, w in parts if v is not None and w > 0]
    total = sum(w for _, w in known)
    return sum(float(v) * w for v, w in known) / total if total else None


def spy_market_move(alpha: dict[str, Any], chain: dict[str, Any], minutes: int) -> tuple[float | None, str | None]:
    straddle, ttl, age = num(chain.get("atm_straddle_pct")), num(chain.get("minutes_to_expiry")), num(chain.get("age_seconds"))
    if chain.get("available") and straddle is not None and ttl and ttl > 0 and (age is None or age <= 180):
        return straddle * math.sqrt(min(1, max(minutes, 1) / ttl)), "live ATM straddle (directionless)"
    iv = num(alpha.get("spy_iv"))
    return (iv * math.sqrt(max(minutes, 1) / (252 * 390)), "SPY IV scaled to horizon (directionless)") if iv and iv > 0 else (None, None)


def spy_plan(beta: dict[str, Any], move: float, minutes: int, state: str, confidence: float, equity: float) -> dict[str, Any]:
    raw = dict(beta.get("spy") or {}); raw["close"] = num(raw.get("close")) or num(beta.get("price"))
    plan = trade_plan(raw, "LONG" if move > 0 else "SHORT", move, minutes, state, equity)
    cap = min(30000, equity * 0.30)
    if plan.get("available") and plan.get("suggested_shares"):
        plan["suggested_shares"] = min(plan["suggested_shares"], math.floor(cap / raw["close"])); plan["position_value"] = round(plan["suggested_shares"] * raw["close"], 2)
    plan["confidence"] = confidence; plan["max_capital_per_spy"] = cap
    return plan


def spy_predictor(alpha: dict[str, Any], beta: dict[str, Any], horizons: list[dict[str, Any]], pressure: dict[str, Any], chain: dict[str, Any], equity: float) -> dict[str, Any]:
    cp = num(pressure.get("full_index", {}).get("net_expected_spy_return_30m"))
    ar = lambda name=None, minutes=None: num((alpha_row(horizons, name, minutes) or {}).get("expected_return"))
    br = lambda minutes: (num((beta_row(beta, minutes) or {}).get("expected_return_bps")) / 10000) if num((beta_row(beta, minutes) or {}).get("expected_return_bps")) is not None else None
    r30 = blend([(ar(minutes=30), .50), (br(30), .30), (cp, .20)])
    r120 = blend([(ar(minutes=120), .65), (br(30), .10), (cp, .25)])
    rclose = blend([(ar(name="EOD"), .75), (cp, .20), (br(30), .05)])
    r1d, r5d_a = blend([(ar(name="1D"), .85), (cp, .15)]), ar(name="5D")
    r3d = r1d + .5 * (r5d_a - r1d) if r1d is not None and r5d_a is not None else 3 * r1d if r1d is not None else None
    r5d = blend([(r5d_a, .90), (cp, .10)])
    now = datetime.now(UTC).astimezone(ET); pre = now.weekday() < 5 and (now.hour > 4 or now.hour == 4) and (now.hour < 9 or (now.hour == 9 and now.minute < 30))
    observed = num(alpha.get("change_pct")); open_gap = observed + .25 * (blend([(ar(minutes=5), .6), (br(5), .4)]) or 0) if pre and observed is not None else None
    agreement = (BASE.consensus(alpha, beta) or {}).get("horizon_match_rate")
    conf = mean([num(alpha.get("directional_confidence")), num(alpha.get("trust_score")), num(beta.get("confidence")), num(beta.get("coverage_ratio"))]) or 0
    if agreement is not None: conf = clamp(.8 * conf + .2 * agreement)
    price = num(alpha.get("price")) or num(beta.get("price")); out = []
    for name, minutes, move in (("open", 5, open_gap), ("30m", 30, r30), ("2h", 120, r120), ("close", 390, rclose), ("3d", 1170, r3d), ("5d", 1950, r5d)):
        market, source = spy_market_move(alpha, chain, minutes); hconf = conf * (.8 if name in {"3d", "5d"} else 1) * (.4 if move is None else 1)
        out.append({"name": name, "horizon_minutes": minutes, "predicted_move": move, "predicted_level": price * (1 + move) if price and move is not None else None,
                    "direction": "BULLISH" if (move or 0) > 0 else "BEARISH" if (move or 0) < 0 else "UNAVAILABLE", "confidence": hconf,
                    "options_implied_magnitude": market, "options_implied_source": source, "model_minus_market_magnitude": abs(move) - market if move is not None and market is not None else None,
                    "note": "Options implied magnitude is directionless; Delta direction comes from Alpha/Beta/constituent convergence."})
    primary = next((x for x in out if x["name"] == "2h" and x["predicted_move"] is not None), None) or next((x for x in out if x["name"] == "30m" and x["predicted_move"] is not None), None)
    state = "NO_TRADE"
    if primary:
        mag, c, market = abs(primary["predicted_move"]), primary["confidence"], primary["options_implied_magnitude"]
        edge_ok = market is None or mag >= max(.7 * market, .0025)
        if c >= .78 and mag >= .0045 and edge_ok: state = "ACTIVE_A+"
        elif c >= .66 and mag >= .003 and edge_ok: state = "ACTIVE_B"
        elif c >= .55 and mag >= .002: state = "ARMED"
    plan = spy_plan(beta, primary["predicted_move"], primary["horizon_minutes"], state if state in {"ACTIVE_A+", "ACTIVE_B"} else "ARMED", primary["confidence"], equity) if primary and state != "NO_TRADE" else {"available": False}
    return {"state": state, "price": price, "horizons": out, "primary_trade_horizon": primary["name"] if primary else None, "primary_direction": primary["direction"] if primary else None,
            "confidence": primary["confidence"] if primary else conf, "trade": plan, "spy_options": chain, "observed_premarket_gap": observed if pre else None,
            "method": "Delta research blend of Alpha, Beta and weighted constituent pressure; no execution authority."}


def merge_running(stocks: dict[str, Any], spy: dict[str, Any]) -> dict[str, Any]:
    rows = list(stocks["actionable_now"])
    if spy["state"] in {"ACTIVE_A+", "ACTIVE_B"}:
        h = next((x for x in spy["horizons"] if x["name"] == spy["primary_trade_horizon"]), {})
        rows.append({"symbol": "SPY", "sector": "INDEX", "price": spy["price"], "bias": "LONG" if spy["primary_direction"] == "BULLISH" else "SHORT", "state": spy["state"],
                     "score": round(spy["confidence"] * 100, 1), "model_expected_move": h.get("predicted_move"), "remaining_expected_move": h.get("predicted_move"),
                     "options_implied_move": h.get("options_implied_magnitude"), "expected_realization_minutes": h.get("horizon_minutes"), "trade": spy["trade"],
                     "catalyst": {"catalyst_type": "INDEX_CONVERGENCE", "headline": "Alpha/Beta/constituent convergence"}})
    rows.sort(key=lambda r: (r["state"] == "ACTIVE_A+", r.get("score", 0), abs(r.get("remaining_expected_move") or 0)), reverse=True)
    total = sum(num((r.get("trade") or {}).get("suggested_risk_dollars")) or 0 for r in rows)
    if total > 3000:
        scale = 3000 / total
        for r in rows:
            t = r.get("trade") or {}; risk = num(t.get("suggested_risk_dollars")) or 0
            if risk: t["suggested_risk_dollars"] = round(risk * scale, 2); t["portfolio_risk_adjustment"] = "Scaled to $3,000 simultaneous-risk cap."
    return {"actionable": rows[:12], "armed_watch": stocks["armed_watch"][:12], "extended_do_not_chase": stocks["extended_do_not_chase"][:8], "max_simultaneous_open_risk": 3000.0,
            "suggested_open_risk": min(total, 3000.0)}


# ---------- Public / ChatGPT contract ----------
def compact(row: dict[str, Any]) -> dict[str, Any]:
    keys = ("symbol", "sector", "weight", "price", "bias", "state", "score", "convergence", "signal_agreement_pct", "predictive_strength", "confirmation_strength",
            "model_expected_move", "move_already_realized", "remaining_expected_move", "options_implied_move", "model_minus_implied_magnitude", "expected_realization_minutes", "catalyst", "trade")
    return {k: row.get(k) for k in keys if k in row}


def chatgpt_view(payload: dict[str, Any]) -> dict[str, Any]:
    delta, beta, gamma = payload["delta"], payload.get("beta") or {}, payload.get("gamma") or {}
    running = delta["running_list"]
    return {
        "schema_version": "chatgpt-market-state-v1", "generated_at": payload["generated_at"], "freshness_seconds": age_seconds(payload["generated_at"]),
        "competition": payload["competition"],
        "data_health": {"alpha_available": bool((payload.get("alpha") or {}).get("available")), "beta_available": bool(beta.get("available")), "beta_status": beta.get("status"),
                        "beta_coverage_ratio": beta.get("coverage_ratio"), "beta_symbol_count": beta.get("symbol_count"), "expected_symbol_count": beta.get("expected_symbol_count"),
                        "options_telemetry_symbols": delta["data_contract"]["options_telemetry_symbols"], "gamma_catalysts_loaded": delta["data_contract"]["gamma_catalysts_loaded"]},
        "market": {k: (payload.get("market") or {}).get(k) for k in ("symbol", "price", "bid", "ask", "change_pct", "market_open", "exchange_time")},
        "spy_predictor": delta["spy_predictor"], "constituent_pressure": delta["constituent_pressure"],
        "running_list": {"actionable": [compact(r) for r in running["actionable"]], "armed_watch": [compact(r) for r in running["armed_watch"]],
                         "extended_do_not_chase": [compact(r) for r in running["extended_do_not_chase"]], "suggested_open_risk": running["suggested_open_risk"],
                         "max_simultaneous_open_risk": running["max_simultaneous_open_risk"]},
        "gamma": {"available": gamma.get("available"), "generated_at": gamma.get("generated_at"), "events": gamma.get("events", [])[:20]},
        "safety": payload["safety"],
        "instructions": {"interpretation": "Combine this quantitative state with fresh public news. Options implied moves are magnitude only. Recalculate entries against current price before presenting a trade.",
                         "promotion": "Prefer ACTIVE_A+/ACTIVE_B. ARMED requires fresh catalyst or live confirmation."},
    }


def collect() -> dict[str, Any]:
    payload = BASE.collect()
    alpha, beta = payload.get("alpha") or {}, payload.get("beta") or {}
    horizons = raw_alpha_horizons()
    spy_opts, iv_map, alpha_db = alpha_options(alpha)
    gamma = load_gamma(); equity = num(os.getenv("COMPETITION_EQUITY")) or 100000.0
    stocks = build_stock_list(beta, gamma, iv_map, equity) if beta.get("available") else {"scanned": 0, "actionable_now": [], "armed_watch": [], "extended_do_not_chase": [], "deteriorating": [], "counts": {}}
    pressure = constituent_pressure(beta) if beta.get("available") else {"full_index": {}, "top_125_by_weight": {}, "top_125_count": 0, "top_positive_contributors": [], "top_negative_contributors": [], "sector_contributions": []}
    spy = spy_predictor(alpha, beta, horizons, pressure, spy_opts, equity); running = merge_running(stocks, spy)
    payload.update({
        "schema_version": 3, "generated_at": BASE.iso_now(), "product": {"name": "SPY Command / Delta", "version": "2.0.0"},
        "competition": {"paper_equity": equity, "normal_risk_dollars": [500, 750], "a_plus_risk_dollars": [1000, 1250], "max_simultaneous_open_risk": 3000,
                        "minimum_preferred_reward_risk": 2.0, "execution": "research/paper only"},
        "gamma": gamma,
        "delta": {"spy_predictor": spy, "constituent_pressure": pressure, "stock_scanner": stocks, "running_list": running,
                  "data_contract": {"all_sp500_scanned": bool(stocks["scanned"] >= 450 and (beta.get("coverage_ratio") or 0) >= .90),
                                    "top_125_integrated": pressure.get("top_125_count", 0) >= 125, "options_telemetry_symbols": len(iv_map),
                                    "gamma_catalysts_loaded": len(gamma["events"]), "execution_authority": False}},
        "options": {"spy": spy_opts, "constituent_symbol_count": len(iv_map), "alpha_db": alpha_db},
        "safety": {"read_only_convergence": True, "feeds_back_into_alpha": False, "feeds_back_into_beta": False, "places_orders": False, "options_implied_moves_are_directionless": True},
    })
    payload.setdefault("sources", {})["alpha_db"] = alpha_db; payload["sources"]["gamma"] = gamma.get("source")
    return payload


def main() -> int:
    parser = argparse.ArgumentParser(); parser.add_argument("--output", default=OUTPUT); parser.add_argument("--chatgpt-output", default=CHATGPT_OUTPUT); parser.add_argument("--stdout", action="store_true")
    args = parser.parse_args(); payload = collect()
    if args.stdout: print(json.dumps(payload, indent=2))
    else: write_atomic(Path(args.output), payload); write_atomic(Path(args.chatgpt_output), chatgpt_view(payload))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
