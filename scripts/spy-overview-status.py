#!/usr/bin/env python3
"""Generate the read-only SPY Command overview status document.

Runs from systemd, reads Alpha-SPY + Beta-spy HTTP state, optional Beta SQLite
history/backtest data, and local service health, then atomically writes status.json.
No cross-engine value is ever fed back into either trading engine.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import shutil
import sqlite3
import subprocess
import tempfile
import time
import urllib.error
import urllib.request
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

ALPHA_URL = os.getenv("ALPHA_STATE_URL", "http://127.0.0.1:8788/api/v1/dashboard/state")
BETA_URL = os.getenv("BETA_STATE_URL", "http://127.0.0.1:8790/api/state")
OUTPUT = os.getenv("OVERVIEW_STATUS_PATH", "/var/www/spy-overview/status.json")


def iso_now() -> str:
    return datetime.now(UTC).isoformat().replace("+00:00", "Z")


def num(value: Any) -> float | None:
    try:
        value = float(value)
    except (TypeError, ValueError):
        return None
    return value if math.isfinite(value) else None


def clamp(value: float, lo: float = 0.0, hi: float = 1.0) -> float:
    return max(lo, min(hi, value))


def fetch_json(url: str, token: str | None = None) -> tuple[dict[str, Any] | None, dict[str, Any]]:
    started = time.monotonic()
    headers = {"Accept": "application/json", "User-Agent": "spy-command/1.0"}
    if token:
        headers["X-Dashboard-Token"] = token
    try:
        req = urllib.request.Request(url, headers=headers)
        with urllib.request.urlopen(req, timeout=4) as response:  # nosec B310 - configured/local endpoint
            data = json.load(response)
        if not isinstance(data, dict):
            raise ValueError("endpoint returned non-object JSON")
        return data, {"ok": True, "url": url, "latency_ms": round((time.monotonic()-started)*1000, 1), "error": None}
    except (urllib.error.URLError, urllib.error.HTTPError, TimeoutError, ValueError, json.JSONDecodeError) as exc:
        return None, {"ok": False, "url": url, "latency_ms": round((time.monotonic()-started)*1000, 1), "error": str(exc)}


def direction(prob_up: float | None, expected: float | None = None) -> str:
    if prob_up is not None:
        if prob_up >= .52:
            return "BULLISH"
        if prob_up <= .48:
            return "BEARISH"
    if expected is not None:
        return "BULLISH" if expected > 0 else "BEARISH" if expected < 0 else "NEUTRAL"
    return "NEUTRAL"


def alpha_horizons(raw: dict[str, Any]) -> list[dict[str, Any]]:
    source = raw.get("forecast_horizons") or raw.get("forecast") or {}
    if isinstance(source, dict):
        # Live Alpha publishes a dict keyed "5m"/"15m"/"30m" whose values do
        # not carry horizon_minutes; recover it from the key.
        values = []
        for key, item in source.items():
            if isinstance(item, dict) and num(item.get("horizon_minutes")) is None:
                text = str(key).strip().lower()
                if text.endswith("m") and text[:-1].isdigit():
                    item = {**item, "horizon_minutes": int(text[:-1])}
            values.append(item)
    else:
        values = source if isinstance(source, list) else []
    rows = []
    for item in values:
        if not isinstance(item, dict) or num(item.get("horizon_minutes")) is None:
            continue
        rows.append({
            "horizon_minutes": int(float(item["horizon_minutes"])),
            "probability_up": num(item.get("probability_up")),
            "expected_return": num(item.get("expected_return")),
            "predicted_price": num(item.get("predicted_price")),
            "predicted_low": num(item.get("predicted_low")),
            "predicted_high": num(item.get("predicted_high")),
            "integrity": item.get("integrity"),
            "role": item.get("role"),
        })
    return sorted(rows, key=lambda x: x["horizon_minutes"])


def normalize_alpha(raw: dict[str, Any] | None, endpoint: dict[str, Any]) -> dict[str, Any]:
    if not raw:
        return {"available": False, "endpoint": endpoint}
    market = raw.get("market") if isinstance(raw.get("market"), dict) else {}
    decision = raw.get("decision") if isinstance(raw.get("decision"), dict) else {}
    health = raw.get("health") if isinstance(raw.get("health"), dict) else {}
    horizons = alpha_horizons(raw)
    h15 = next((x for x in horizons if x["horizon_minutes"] == 15), {})
    prob = num(h15.get("probability_up"))
    if prob is None:
        prob = num(market.get("probability_up"))
    expected = num(h15.get("expected_return"))
    if expected is None:
        expected = num(market.get("expected_return_15m"))
    return {
        "available": True, "endpoint": endpoint, "timestamp": raw.get("timestamp"),
        "price": num(market.get("price")), "bid": num(market.get("bid")), "ask": num(market.get("ask")),
        "change": num(market.get("change")), "change_pct": num(market.get("change_pct")),
        "direction": direction(prob, expected), "probability_up": prob,
        "directional_confidence": clamp(abs(prob-.5)*2) if prob is not None else None,
        "expected_return_15m": expected, "trust_score": num(health.get("trust_score")),
        "health_state": health.get("state"), "health": health,
        "market": market, "session": raw.get("session") or {}, "engine": raw.get("engine") or {},
        "decision": decision, "gates": decision.get("gates") or [], "failed_gates": decision.get("failed_gates") or [],
        "candidates": raw.get("candidates") or [], "position": raw.get("position") or {}, "account": raw.get("account") or {},
        "audit": raw.get("audit") or raw.get("prediction_metrics") or {}, "services": raw.get("services") or [],
        "alerts": raw.get("alerts") or [], "security": raw.get("security") or {}, "horizons": horizons,
        "regime": market.get("regime"), "regime_hierarchy": market.get("regime_hierarchy") or {},
        "gamma_state": market.get("gamma_state"), "liquidity_state": market.get("liquidity_state"),
        "event_state": market.get("event_state"), "breadth": num(market.get("breadth")),
        "pressure": num(market.get("pressure")), "concentration": num(market.get("concentration")),
        "correlation": num(market.get("correlation")), "spy_iv": num(market.get("spy_iv")),
    }


def normalize_beta(raw: dict[str, Any] | None, endpoint: dict[str, Any]) -> dict[str, Any]:
    if not raw:
        return {"available": False, "endpoint": endpoint}
    snap = raw.get("snapshot") if isinstance(raw.get("snapshot"), dict) else {}
    factors = snap.get("factors") if isinstance(snap.get("factors"), dict) else {}
    decision_row = snap.get("decision") if isinstance(snap.get("decision"), dict) else {}
    forecasts = []
    for item in snap.get("forecasts") or []:
        if not isinstance(item, dict):
            continue
        forecasts.append({
            "horizon_minutes": int(num(item.get("horizon_minutes")) or 0),
            "probability_up": num(item.get("probability_up")), "expected_return_bps": num(item.get("expected_return_bps")),
            "confidence": num(item.get("confidence")), "model_ready": bool(item.get("model_ready")),
            "sample_count": int(num(item.get("sample_count")) or 0),
        })
    symbols = snap.get("symbols") if isinstance(snap.get("symbols"), list) else []
    spy = next((x for x in symbols if isinstance(x, dict) and x.get("symbol") == "SPY"), {})
    h15 = next((x for x in forecasts if x["horizon_minutes"] == 15), {})
    beta_dir = decision_row.get("direction") or direction(num(h15.get("probability_up")))
    stream = raw.get("stream") if isinstance(raw.get("stream"), dict) else {}
    return {
        "available": True, "endpoint": endpoint, "timestamp": raw.get("timestamp"), "status": raw.get("status"),
        "price": num(spy.get("close")), "spy": spy, "direction": beta_dir,
        "confidence": num(decision_row.get("confidence")), "score": num(decision_row.get("score")),
        "decision_action": decision_row.get("action"), "decision": decision_row,
        "gates": decision_row.get("gates") or {}, "reasons": decision_row.get("reasons") or [],
        "forecasts": sorted(forecasts, key=lambda x: x["horizon_minutes"]), "factors": factors, "symbols": symbols,
        "sectors": factors.get("sectors") or [], "coverage_ratio": num(factors.get("coverage_ratio")),
        "symbol_count": int(num(factors.get("symbol_count")) or 0), "expected_symbol_count": int(num(factors.get("expected_symbol_count")) or 0),
        "trend_ew": num(factors.get("trend_ew")), "trend_weighted": num(factors.get("trend_weighted")),
        "momentum_ew": num(factors.get("momentum_ew")), "momentum_weighted": num(factors.get("momentum_weighted")),
        "flow_ew": num(factors.get("flow_ew")), "flow_weighted": num(factors.get("flow_weighted")),
        "pct_above_vwap": num(factors.get("pct_above_vwap")), "pct_ema_bullish": num(factors.get("pct_ema_bullish")),
        "pct_positive_5m": num(factors.get("pct_positive_5m")), "pct_buy_flow": num(factors.get("pct_buy_flow")),
        "participation": num(factors.get("participation")), "concentration": num(factors.get("concentration")),
        "breadth_acceleration": num(factors.get("breadth_acceleration")), "stream": stream,
        "option_plan": raw.get("option_plan") if isinstance(raw.get("option_plan"), dict) else None,
    }


def consensus(alpha: dict[str, Any], beta: dict[str, Any]) -> dict[str, Any]:
    ad = alpha.get("direction", "NEUTRAL") if alpha.get("available") else "NEUTRAL"
    bd = beta.get("direction", "NEUTRAL") if beta.get("available") else "NEUTRAL"
    ac = num(alpha.get("directional_confidence"))
    if ac is None:
        ac = num(alpha.get("trust_score"))
    bc = num(beta.get("confidence"))
    ac, bc = clamp(ac or 0), clamp(bc or 0)
    strength = (ac + bc) / 2
    if ad == bd and ad not in {"NEUTRAL", None}:
        score, state, out_dir = 50 + 50*strength, "STRONG AGREEMENT" if strength >= .56 else "AGREEMENT", ad
    elif {ad, bd} == {"BULLISH", "BEARISH"}:
        score, state, out_dir = 50 - 50*strength, "STRONG DIVERGENCE" if strength >= .56 else "DIVERGENCE", "DIVERGENT"
    else:
        score, state, out_dir = 50, "MIXED / NEUTRAL", "NEUTRAL"
    a_by = {x["horizon_minutes"]: x for x in alpha.get("horizons", [])}
    b_by = {x["horizon_minutes"]: x for x in beta.get("forecasts", [])}
    compared = []
    for minutes in sorted(set(a_by) | set(b_by)):
        a, b = a_by.get(minutes), b_by.get(minutes)
        ar = num(a.get("expected_return")) if a else None
        br_bps = num(b.get("expected_return_bps")) if b else None
        br = br_bps/10000 if br_bps is not None else None
        ap = num(a.get("probability_up")) if a else None
        bp = num(b.get("probability_up")) if b else None
        compared.append({"horizon_minutes": minutes, "alpha_probability_up": ap, "beta_probability_up": bp,
                         "alpha_expected_return": ar, "beta_expected_return": br,
                         "direction_agreement": direction(ap, ar) == direction(bp, br) if a and b else None})
    matched = [x for x in compared if x["direction_agreement"] is not None]
    match_rate = sum(bool(x["direction_agreement"]) for x in matched)/len(matched) if matched else None
    return {"score": round(score, 1), "state": state, "direction": out_dir,
            "alpha_direction": ad, "beta_direction": bd, "alpha_confidence": ac, "beta_confidence": bc,
            "horizon_match_rate": match_rate, "forecasts": compared,
            "method": "Descriptive Live Agreement Index only; not a validated trading signal."}


def first_file(env: str, defaults: list[str]) -> Path | None:
    candidates = ([os.environ[env]] if os.getenv(env) else []) + defaults
    return next((Path(x) for x in candidates if Path(x).exists()), None)


def beta_history(limit: int = 240) -> tuple[list[dict[str, Any]], str | None]:
    db = first_file("BETA_DB", ["/opt/beta-spy/src/data/beta-spy.sqlite", "/opt/beta-spy/data/beta-spy.sqlite", "/var/lib/beta-spy/beta-spy.sqlite", "data/beta-spy.sqlite"])
    if not db:
        return [], None
    try:
        con = sqlite3.connect(f"file:{db}?mode=ro", uri=True, timeout=2)
        rows = con.execute("SELECT timestamp,close,volume,vwap FROM minute_bars WHERE symbol='SPY' ORDER BY timestamp DESC LIMIT ?", (limit,)).fetchall()
        con.close(); rows.reverse()
        return [{"timestamp": r[0], "price": num(r[1]), "volume": num(r[2]), "vwap": num(r[3])} for r in rows], str(db)
    except sqlite3.Error:
        return [], str(db)


def beta_backtest() -> tuple[dict[str, Any] | None, str | None]:
    path = first_file("BETA_BACKTEST_JSON", ["/opt/beta-spy/src/reports/backtest-latest.json", "/opt/beta-spy/reports/backtest-latest.json", "reports/backtest-latest.json"])
    if not path:
        return None, None
    try:
        data = json.loads(path.read_text())
        return data if isinstance(data, dict) else None, str(path)
    except (OSError, json.JSONDecodeError):
        return None, str(path)


def unit(name: str) -> dict[str, Any]:
    try:
        p = subprocess.run(["systemctl", "is-active", name], capture_output=True, text=True, timeout=2, check=False)
        state = (p.stdout or p.stderr or "unknown").strip()
        return {"name": name, "state": state, "ok": state == "active"}
    except (OSError, subprocess.SubprocessError):
        return {"name": name, "state": "unknown", "ok": False}


def collect() -> dict[str, Any]:
    token = os.getenv("ALPHA_DASHBOARD_TOKEN") or os.getenv("DASHBOARD_VIEW_TOKEN")
    ar, aep = fetch_json(ALPHA_URL, token)
    br, bep = fetch_json(BETA_URL)
    alpha, beta = normalize_alpha(ar, aep), normalize_beta(br, bep)
    history, db_path = beta_history()
    backtest, report_path = beta_backtest()
    alerts = [x for x in alpha.get("alerts", []) if isinstance(x, dict)]
    if beta.get("stream", {}).get("last_error"):
        alerts.insert(0, {"severity": "warning", "title": "Beta stream", "message": beta["stream"]["last_error"], "source": "beta-spy", "timestamp": beta.get("timestamp")})
    if not aep["ok"]:
        alerts.insert(0, {"severity": "critical", "title": "Alpha unavailable", "message": aep["error"], "source": "overview"})
    if not bep["ok"]:
        alerts.insert(0, {"severity": "critical", "title": "Beta unavailable", "message": bep["error"], "source": "overview"})
    disk = shutil.disk_usage("/")
    units = os.getenv("OVERVIEW_UNITS", "alpha-spy,beta-spy,spy-overview-status.timer,spy-tunnel,nginx")
    return {
        "schema_version": 2, "generated_at": iso_now(), "product": {"name": "SPY Command", "version": "1.0.0"},
        "market": {"symbol": "SPY", "price": alpha.get("price") if alpha.get("price") is not None else beta.get("price"),
                   "bid": alpha.get("bid"), "ask": alpha.get("ask"), "change": alpha.get("change"), "change_pct": alpha.get("change_pct"),
                   "market_open": alpha.get("session", {}).get("market_open"), "exchange_time": alpha.get("session", {}).get("exchange_time"), "history": history},
        "alpha": alpha, "beta": beta, "consensus": consensus(alpha, beta),
        "performance": {"alpha": alpha.get("audit") if alpha.get("available") else None, "beta": backtest, "beta_backtest_path": report_path},
        "alerts": alerts[:50],
        "system": {"alpha_endpoint": aep, "beta_endpoint": bep, "units": [unit(x.strip()) for x in units.split(",") if x.strip()],
                   "hostname": os.uname().nodename if hasattr(os, "uname") else None,
                   "loadavg": list(os.getloadavg()) if hasattr(os, "getloadavg") else None,
                   "disk": {"total": disk.total, "used": disk.used, "free": disk.free, "used_ratio": disk.used/disk.total if disk.total else None}},
        "sources": {"beta_db": db_path, "beta_backtest": report_path},
        "links": {"alpha": os.getenv("ALPHA_PUBLIC_URL"), "beta": os.getenv("BETA_PUBLIC_URL")},
    }


def write_atomic(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temp = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent, text=True)
    try:
        # mkstemp creates mode-600 files; nginx (www-data) must be able to
        # read the published document.
        os.fchmod(fd, 0o644)
        with os.fdopen(fd, "w") as f:
            json.dump(payload, f, separators=(",", ":")); f.write("\n"); f.flush(); os.fsync(f.fileno())
        os.replace(temp, path)
    finally:
        try: os.unlink(temp)
        except FileNotFoundError: pass


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default=OUTPUT)
    parser.add_argument("--stdout", action="store_true")
    args = parser.parse_args()
    payload = collect()
    if args.stdout:
        print(json.dumps(payload, indent=2))
    else:
        write_atomic(Path(args.output), payload)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
