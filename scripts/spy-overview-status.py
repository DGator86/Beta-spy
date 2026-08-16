#!/usr/bin/env python3
"""Write the ops half of the overview page to /var/www/spy-overview/status.json.

The live trading data (Alpha/Beta state, ledger) is proxied straight from the
services by nginx; this script covers what only the box itself knows: unit
health, timer schedules, watchdog verdicts, backups, disk and memory. Runs
from a systemd timer once a minute; stdlib only.
"""

from __future__ import annotations

import json
import shutil
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
]

TIMERS = [
    "spy-watchdog.timer",
    "beta-spy-backtest.timer",
    "beta-spy-backup.timer",
    "alpha-spy-backup.timer",
]


def _run(*argv: str) -> str:
    try:
        return subprocess.run(
            argv, capture_output=True, text=True, timeout=15
        ).stdout.strip()
    except Exception:  # noqa: BLE001 - status must always render
        return ""


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

    reports = sorted(Path("/var/lib/beta-spy/reports").glob("backtest-*.md"))
    latest_backtest = reports[-1].name if reports else None

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
        "latest_backtest_report": latest_backtest,
        "disk_free_gb": round(disk.free / 1e9, 1),
        "memory_available_mb": memory.get("MemAvailable"),
        "memory_total_mb": memory.get("MemTotal"),
    }

    out = Path("/var/www/spy-overview/status.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    tmp = out.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(payload, indent=1))
    tmp.replace(out)


if __name__ == "__main__":
    main()
