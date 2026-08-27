#!/usr/bin/env python3
"""Export today's RTH tape and copy it to the Google Drive backup remote.

Writes a compressed archive under session-tapes/YYYY-MM-DD/ on the same
rclone remote the nightly database snapshots use. V2 also includes the exact
Alpha option-chain evidence used for payoff selection so future replay never
has to synthesize historical option quotes.
"""
from __future__ import annotations

import csv
import json
import os
import shutil
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
ALPHA_STATE_ROOT = Path(os.environ.get("ALPHA_SPY_STATE_ROOT") or "/var/lib/alpha-spy")


def session_bounds(day: str) -> tuple[str, str]:
    start = datetime.fromisoformat(day).replace(tzinfo=ET, hour=9, minute=30, second=0, microsecond=0)
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
            f"SELECT * FROM {table} WHERE {ts_col} >= ? AND {ts_col} < ? ORDER BY {ts_col}",
            (start, end),
        )
    ]
    return write_rows(rows, dest)


def copy_jsonl_if_present(source: Path, destination: Path) -> int:
    if not source.exists() or source.stat().st_size <= 0:
        return 0
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, destination)
    with destination.open("rb") as handle:
        return sum(1 for line in handle if line.strip())


def dump_alpha_option_db(
    conn: sqlite3.Connection,
    destination: Path,
    start: str,
    end: str,
) -> int:
    """Fallback DB export of every RTH SPY strategy chain and its raw quote payload."""
    conn.row_factory = sqlite3.Row
    tables = {
        str(row[0])
        for row in conn.execute("SELECT name FROM sqlite_master WHERE type='table'").fetchall()
    }
    if not {"option_chain_snapshots", "option_quotes"}.issubset(tables):
        return write_rows([], destination)
    chains = [
        dict(row)
        for row in conn.execute(
            """
            SELECT * FROM option_chain_snapshots
            WHERE underlying='SPY' AND purpose='strategy'
              AND captured_at >= ? AND captured_at < ?
            ORDER BY captured_at
            """,
            (start, end),
        )
    ]
    if not chains:
        return write_rows([], destination)
    output: list[dict] = []
    for chain in chains:
        quotes = [
            dict(row)
            for row in conn.execute(
                """
                SELECT * FROM option_quotes
                WHERE chain_snapshot_id=?
                ORDER BY right,strike
                """,
                (chain["chain_snapshot_id"],),
            )
        ]
        output.append({"chain": chain, "options": quotes})
    write_rows(output, destination)
    return len(output)


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
                beta, "minute_bars", "timestamp", out_dir / "minute_bars.csv", start, end
            )
            counts["spy_trades"] = dump_between(
                beta, "spy_trades", "timestamp", out_dir / "spy_trades.csv", start, end
            )
            counts["spy_quotes"] = dump_between(
                beta, "spy_quotes", "timestamp", out_dir / "spy_quotes.csv", start, end
            )
            counts["decisions"] = dump_between(
                beta, "decisions", "timestamp", out_dir / "decisions.json", start, end
            )
            counts["paper_positions"] = dump_between(
                beta, "paper_positions", "opened_at", out_dir / "paper_positions.json", start, end
            )
            counts["alpha_signals"] = dump_between(
                beta, "alpha_signals", "timestamp", out_dir / "alpha_signals.json", start, end
            )
            counts["v2_market_state"] = dump_between(
                beta, "v2_market_state", "timestamp", out_dir / "v2_market_state.json", start, end
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
            counts["alpha_option_db_chains"] = dump_alpha_option_db(
                alpha,
                out_dir / "alpha_option_chains.json",
                start,
                end,
            )
            alpha.close()

        counts["v2_exact_chain_records"] = copy_jsonl_if_present(
            ALPHA_STATE_ROOT / "replay" / f"v2-option-chain-{day}.jsonl",
            out_dir / "v2_option_chain_decisions.jsonl",
        )
        counts["alpha_raw_chain_records"] = copy_jsonl_if_present(
            ALPHA_STATE_ROOT / "market" / f"spy-options-{day}.jsonl",
            out_dir / "alpha_spy_options_raw.jsonl",
        )
        counts["alpha_v2_candidate_records"] = copy_jsonl_if_present(
            ALPHA_STATE_ROOT / "candidates" / f"v2-candidates-{day}.jsonl",
            out_dir / "alpha_v2_candidates.jsonl",
        )

        (out_dir / "manifest.txt").write_text(
            "\n".join(
                [
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
