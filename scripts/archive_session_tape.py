#!/usr/bin/env python3
"""Export today's RTH tape and copy it to the Google Drive backup remote.

Writes a compressed archive under session-tapes/YYYY-MM-DD/ on the same
rclone remote the nightly database snapshots use.  Tape schema v2 preserves
Beta's own factors/forecasts in addition to the market tape and sampled Alpha
signals, so cross-model replay does not lose the Beta witness stream.
"""
from __future__ import annotations

import csv
import json
import os
import sqlite3
import subprocess
import tarfile
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from zoneinfo import ZoneInfo

ET = ZoneInfo("America/New_York")
BETA_DB = Path("/var/lib/beta-spy/beta-spy.sqlite")
ALPHA_DB = Path("/var/lib/alpha-spy/journal/alpha-spy.db")
TAPE_SCHEMA_VERSION = 2


def session_bounds(day: str) -> tuple[str, str]:
    start = datetime.fromisoformat(day).replace(
        tzinfo=ET,
        hour=9,
        minute=30,
        second=0,
        microsecond=0,
    )
    end = start.replace(hour=16, minute=0)
    return (
        start.astimezone(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        end.astimezone(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
    )


def write_rows(rows: list[dict], dest: Path) -> int:
    dest.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        dest.write_text("[]\n" if dest.suffix != ".csv" else "", encoding="utf-8")
        return 0
    if dest.suffix == ".csv":
        with dest.open("w", newline="", encoding="utf-8") as fh:
            writer = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)
    else:
        dest.write_text(json.dumps(rows, default=str) + "\n", encoding="utf-8")
    return len(rows)


def dump_between(
    conn: sqlite3.Connection,
    table: str,
    ts_col: str,
    dest: Path,
    start: str,
    end: str,
) -> int:
    cols = [row[1] for row in conn.execute(f"pragma table_info({table})")]
    if ts_col not in cols:
        return write_rows([], dest)
    conn.row_factory = sqlite3.Row
    rows = [
        dict(row)
        for row in conn.execute(
            f"SELECT * FROM {table} WHERE {ts_col} >= ? AND {ts_col} < ? "
            f"ORDER BY {ts_col}",
            (start, end),
        )
    ]
    return write_rows(rows, dest)


def main() -> int:
    day = os.environ.get("TAPE_DAY") or datetime.now(ET).date().isoformat()
    start, end = session_bounds(day)
    host = os.environ.get("HOST") or os.uname().nodename.split(".")[0]
    remote = os.environ.get("ALPHA_SPY_BACKUP_REMOTE") or f"gdrive:SPY Trading Backups/{host}"
    remote = remote.strip().strip("'\"")
    rclone_config = os.environ.get("RCLONE_CONFIG") or "/root/.config/rclone/rclone.conf"
    work = Path(tempfile.mkdtemp(prefix=f"session-tape-{day}-", dir="/var/tmp"))
    out_dir = work / day
    out_dir.mkdir()
    counts: dict[str, int] = {}
    try:
        if BETA_DB.exists():
            beta = sqlite3.connect(f"file:{BETA_DB}?mode=ro", uri=True)
            counts["minute_bars"] = dump_between(
                beta,
                "minute_bars",
                "timestamp",
                out_dir / "minute_bars.csv",
                start,
                end,
            )
            counts["spy_trades"] = dump_between(
                beta,
                "spy_trades",
                "timestamp",
                out_dir / "spy_trades.csv",
                start,
                end,
            )
            counts["spy_quotes"] = dump_between(
                beta,
                "spy_quotes",
                "timestamp",
                out_dir / "spy_quotes.csv",
                start,
                end,
            )
            counts["beta_factors"] = dump_between(
                beta,
                "factor_snapshots",
                "timestamp",
                out_dir / "beta_factors.json",
                start,
                end,
            )
            counts["beta_forecasts"] = dump_between(
                beta,
                "forecasts",
                "timestamp",
                out_dir / "beta_forecasts.csv",
                start,
                end,
            )
            counts["decisions"] = dump_between(
                beta,
                "decisions",
                "timestamp",
                out_dir / "decisions.json",
                start,
                end,
            )
            # Explicit provenance alias for schema-v2 consumers.  Keep the old
            # file name above so existing tape readers remain compatible.
            counts["beta_decisions"] = dump_between(
                beta,
                "decisions",
                "timestamp",
                out_dir / "beta_decisions.json",
                start,
                end,
            )
            counts["paper_positions"] = dump_between(
                beta,
                "paper_positions",
                "opened_at",
                out_dir / "paper_positions.json",
                start,
                end,
            )
            counts["alpha_signals"] = dump_between(
                beta,
                "alpha_signals",
                "timestamp",
                out_dir / "alpha_signals.json",
                start,
                end,
            )
            beta.close()
        if ALPHA_DB.exists():
            alpha = sqlite3.connect(f"file:{ALPHA_DB}?mode=ro", uri=True)
            alpha.row_factory = sqlite3.Row
            rows = [
                dict(row)
                for row in alpha.execute(
                    """
                    SELECT captured_at, spy_price, exchange_state
                    FROM market_snapshots
                    WHERE spy_price IS NOT NULL
                      AND captured_at >= ?
                      AND captured_at < ?
                    ORDER BY captured_at
                    """,
                    (start, end),
                )
            ]
            counts["alpha_snapshots"] = write_rows(rows, out_dir / "alpha_spy_path.csv")
            alpha.close()
        (out_dir / "manifest.txt").write_text(
            "\n".join(
                [
                    f"tape_schema_version={TAPE_SCHEMA_VERSION}",
                    "tape_source=beta-spy",
                    f"day={day}",
                    f"session_start_utc={start}",
                    f"session_end_utc={end}",
                    f"exported_at={datetime.now(ET).isoformat()}",
                    *[f"{key}={value}" for key, value in counts.items()],
                    "",
                ]
            ),
            encoding="utf-8",
        )
        archive = work / f"tape-{day}.tar"
        with tarfile.open(archive, "w") as tar:
            tar.add(out_dir, arcname=day)
        zst = Path(str(archive) + ".zst")
        subprocess.run(["zstd", "-f", "-q", str(archive), "-o", str(zst)], check=True)
        dest = f"{remote}/session-tapes/{day}/tape-{day}.tar.zst"
        subprocess.run(
            [
                "rclone",
                "copyto",
                str(zst),
                dest,
                "--config",
                rclone_config,
                "--retries",
                "3",
                "--timeout",
                "10m",
            ],
            check=True,
        )
        print(f"uploaded {dest} counts={counts}")
        return 0
    finally:
        subprocess.run(["rm", "-rf", str(work)], check=False)


if __name__ == "__main__":
    raise SystemExit(main())
