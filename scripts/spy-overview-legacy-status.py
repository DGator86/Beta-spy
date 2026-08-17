#!/usr/bin/env python3
"""Write the ops half of the overview page to /var/www/spy-overview/status.json.

The live trading data (Alpha/Beta state, ledger) is proxied straight from the
services by nginx; this script covers what only the box itself knows: unit
health, timer schedules, watchdog verdicts, the ledger's realized history
(equity curve, daily P&L), tape freshness, the latest backtest and CPCV
verdicts, disk and memory. Runs from a systemd timer once a minute; stdlib
only.
"""

from __future__ import annotations

import glob
import json
import re
import shutil
import sqlite3
import subprocess
from datetime import UTC, datetime
from pathlib import Path

UNITS = [
    "alpha-spy-market",
    "alpha-spy-engine",
    "alpha-spy-decision",
    "alpha-spy-confirmation",
    "alpha-spy-settlement",
    "alpha-spy-dashboard",
    "beta-spy",
    "nginx",
    "spy-tunnel",
]

TIMERS = [
    "spy-watchdog.timer",
    "beta-spy-backtest.timer",
    "beta-spy-backup.timer",
    "alpha-spy-backup.timer",
]

BETA_DB = "/var/lib/beta-spy/beta-spy.sqlite"
REPORTS = Path("/var/lib/beta-spy/reports")


def _run(*argv: str) -> str:
    try:
        return subprocess.run(
            argv, capture_output=True, text=True, timeout=15
        ).stdout.strip()
    except Exception:  # noqa: BLE001 - status must always render
        return ""


def _starting_equity() -> float:
    try:
        unit = Path("/etc/systemd/system/beta-spy.service").read_text()
        match = re.search(r"--bankroll\s+(\d+(?:\.\d+)?)", unit)
        if match:
            return float(match.group(1))
    except OSError:
        pass
    return 1000.0


def _ledger_history() -> dict:
    """Equity curve and per-day realized P&L from the paper ledger."""
    out: dict = {"equity_curve": [], "daily_pnl": [], "tape_age_seconds": None}
    try:
        con = sqlite3.connect(f"file:{BETA_DB}?mode=ro", uri=True, timeout=5)
    except sqlite3.Error:
        return out
    try:
        equity = _starting_equity()
        curve = [{"t": None, "equity": equity}]
        daily: dict[str, float] = {}
        rows = con.execute(
            """
            SELECT closed_at, realized_pnl_dollars FROM paper_positions
            WHERE status='CLOSED' AND realized_pnl_dollars IS NOT NULL
            ORDER BY closed_at
            """
        ).fetchall()
        for closed_at, pnl in rows:
            equity += float(pnl or 0.0)
            curve.append({"t": str(closed_at), "equity": round(equity, 2)})
            daily[str(closed_at)[:10]] = daily.get(str(closed_at)[:10], 0.0) + float(pnl or 0.0)
        out["equity_curve"] = curve[-200:]
        out["daily_pnl"] = [
            {"date": day, "pnl": round(value, 2)} for day, value in sorted(daily.items())
        ][-30:]
        age = con.execute(
            "SELECT CAST(strftime('%s','now') - strftime('%s', MAX(timestamp)) AS INTEGER) FROM minute_bars"
        ).fetchone()
        out["tape_age_seconds"] = int(age[0]) if age and age[0] is not None else None
    except sqlite3.Error:
        pass
    finally:
        con.close()
    return out


def _grade(
    calls: list[tuple[str, float]],
    closes: dict[str, float],
    horizon_minutes: int = 15,
    deadband: float = 0.02,
) -> dict:
    """Grade (minute_timestamp, probability_up) calls against realized SPY moves.

    A call is graded when the entry and exit minute both exist on the tape and
    the tape actually moved. "Decisive" restricts to calls with conviction
    beyond the deadband — the calls a trader would actually act on.
    """
    from datetime import datetime as _dt, timedelta as _td

    graded = decisive = correct = decisive_correct = 0
    captured_bps = 0.0
    by_day: dict[str, list[int]] = {}
    for minute, probability in calls:
        entry = closes.get(minute)
        try:
            exit_minute = (
                _dt.fromisoformat(minute.replace("Z", "+00:00")) + _td(minutes=horizon_minutes)
            ).strftime("%Y-%m-%dT%H:%M:00Z")
        except ValueError:
            continue
        exit_price = closes.get(exit_minute)
        if entry is None or exit_price is None or probability is None:
            continue
        move = exit_price - entry
        if move == 0.0 or probability == 0.5:
            continue
        direction = 1 if probability > 0.5 else -1
        hit = (move > 0) == (direction > 0)
        graded += 1
        correct += int(hit)
        captured_bps += direction * (move / entry) * 10_000.0
        day = minute[:10]
        by_day.setdefault(day, [0, 0])
        by_day[day][0] += int(hit)
        by_day[day][1] += 1
        if abs(probability - 0.5) >= deadband:
            decisive += 1
            decisive_correct += int(hit)
    return {
        "graded": graded,
        "accuracy": round(correct / graded, 4) if graded else None,
        "decisive": decisive,
        "decisive_accuracy": round(decisive_correct / decisive, 4) if decisive else None,
        "avg_captured_bps": round(captured_bps / graded, 3) if graded else None,
        "daily": [
            {"date": day, "accuracy": round(hits / total, 3), "n": total}
            for day, (hits, total) in sorted(by_day.items())[-7:]
        ],
    }


def _scoreboard() -> dict:
    """Head-to-head: Alpha's and Beta's 15m calls graded on the same SPY tape."""
    out: dict = {"beta": None, "alpha": None, "horizon_minutes": 15}
    try:
        con = sqlite3.connect(f"file:{BETA_DB}?mode=ro", uri=True, timeout=5)
    except sqlite3.Error:
        return out
    try:
        # Keys are floored to the minute in the same shape _grade produces.
        closes = {
            str(ts)[:17] + "00Z": float(close)
            for ts, close in con.execute(
                "SELECT timestamp, close FROM minute_bars WHERE symbol='SPY'"
            )
        }
        beta_calls = [
            (str(ts)[:17] + "00Z", float(p))
            for ts, p in con.execute(
                "SELECT timestamp, probability_up FROM forecasts "
                "WHERE horizon_minutes=15 AND model_ready=1"
            )
        ]
        out["beta"] = _grade(beta_calls, closes)
        # The number that matters: only the decisions that survived every
        # gate. The raw model hovers near a coin flip by design; the gate
        # stack is where the edge is concentrated.
        # Gate configs change; grading mixed eras would blur what the current
        # gates actually do. A marker file written at deploy time scopes the
        # gated-trade grade to the live configuration only.
        era_file = Path("/var/lib/beta-spy/gates-era.txt")
        gates_era = era_file.read_text().strip() if era_file.exists() else ""
        gated_calls: list[tuple[str, float]] = []
        for ts, payload in con.execute("SELECT timestamp, payload FROM decisions"):
            if gates_era and str(ts).replace(" ", "T") < gates_era:
                continue
            try:
                record = json.loads(payload)
            except json.JSONDecodeError:
                continue
            if record.get("action") != "TRADE":
                continue
            direction = str(record.get("direction") or "")
            if direction not in ("BULLISH", "BEARISH"):
                continue
            minute = str(ts).replace(" ", "T")[:17] + "00Z"
            gated_calls.append((minute, 1.0 if direction == "BULLISH" else 0.0))
        out["beta_gated"] = _grade(gated_calls, closes)
        alpha_calls: list[tuple[str, float]] = []
        seen: set[str] = set()
        for ts, payload in con.execute("SELECT timestamp, payload FROM alpha_signals"):
            try:
                record = json.loads(payload)
                horizon = (record.get("horizons") or {}).get("15m") or {}
                created = str(horizon.get("created_at") or "")
                probability = horizon.get("probability_up")
            except (json.JSONDecodeError, AttributeError):
                continue
            # A stale forecast repeats its created_at; grade each fresh
            # forecast once, at the minute it was first recorded.
            if probability is None or created in seen:
                continue
            seen.add(created)
            alpha_calls.append((str(ts)[:17] + "00Z", float(probability)))
        out["alpha"] = _grade(alpha_calls, closes)
    except sqlite3.Error:
        pass
    finally:
        con.close()
    return out


def _backtest_summary() -> dict | None:
    # Newest by mtime, not name: "backtest-robust.json" must not permanently
    # shadow the nightly date-stamped reports.
    files = sorted(glob.glob(str(REPORTS / "backtest-*.json")), key=lambda f: Path(f).stat().st_mtime)
    if not files:
        return None
    try:
        data = json.loads(Path(files[-1]).read_text())
    except (OSError, json.JSONDecodeError):
        return None
    decisions = data.get("decisions") or {}
    accuracy_15 = next(
        (
            h.get("model_ready_direction_accuracy")
            for h in data.get("horizons") or []
            if h.get("horizon_minutes") == 15
        ),
        None,
    )
    return {
        "report": Path(files[-1]).name,
        "created_at": data.get("created_at"),
        "snapshots": data.get("snapshots"),
        "trade_signals": decisions.get("trade_signals"),
        "trade_accuracy": decisions.get("direction_accuracy"),
        "avg_return_bps": decisions.get("avg_underlying_return_bps"),
        "neutral_signals": decisions.get("neutral_signals"),
        "model_accuracy_15m": accuracy_15,
        "gate_failures": decisions.get("gate_failure_counts"),
    }


def _cpcv_verdict() -> list[str]:
    files = sorted(glob.glob(str(REPORTS / "cscv-*.txt")), key=lambda f: Path(f).stat().st_mtime)
    if not files:
        return []
    try:
        return Path(files[-1]).read_text().splitlines()[-4:]
    except OSError:
        return []


def main() -> None:
    services = {
        unit: _run("systemctl", "is-active", unit) or "unknown" for unit in UNITS
    }

    timers = {}
    for timer in TIMERS:
        timers[timer] = (
            _run(
                "systemctl", "show", timer, "--property=NextElapseUSecRealtime", "--value"
            )
            or "unknown"
        )

    watchdog_tail = []
    log = Path("/var/log/spy-watchdog.log")
    if log.exists():
        watchdog_tail = log.read_text().splitlines()[-5:]

    disk = shutil.disk_usage("/")
    memory = {}
    for line in Path("/proc/meminfo").read_text().splitlines():
        if line.startswith(("MemTotal", "MemAvailable")):
            key, value = line.split(":", 1)
            memory[key] = int(value.strip().split()[0]) // 1024  # MiB

    payload = {
        "generated_at": datetime.now(UTC).isoformat(),
        "services": services,
        "all_services_active": all(state == "active" for state in services.values()),
        "timers_next": timers,
        "watchdog_tail": watchdog_tail,
        "backtest": _backtest_summary(),
        "cpcv_verdict": _cpcv_verdict(),
        "scoreboard": _scoreboard(),
        "disk_free_gb": round(disk.free / 1e9, 1),
        "memory_available_mb": memory.get("MemAvailable"),
        "memory_total_mb": memory.get("MemTotal"),
        **_ledger_history(),
    }

    out = Path("/var/www/spy-overview/status.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    tmp = out.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(payload, indent=1))
    tmp.replace(out)


if __name__ == "__main__":
    main()
