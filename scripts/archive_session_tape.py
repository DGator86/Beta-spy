#!/usr/bin/env python3
"""Export today's RTH Beta/Alpha tape and copy it to the Google Drive backup remote.

V2 expands the tape from a SPY-path summary into deterministic replay evidence:
full Alpha market snapshots/quotes, full 0DTE option-chain snapshots and contracts,
features, predictions, candidates, decisions, orders, positions and matured outcomes.
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


def session_bounds(day: str) -> tuple[str, str]:
    start = datetime.fromisoformat(day).replace(
        tzinfo=ET, hour=9, minute=30, second=0, microsecond=0
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
            f"SELECT * FROM {table} WHERE {ts_col} >= ? AND {ts_col} < ? ORDER BY {ts_col}",
            (start, end),
        )
    ]
    return write_rows(rows, dest)


def dump_child_by_parent_time(
    conn: sqlite3.Connection,
    *,
    child: str,
    parent: str,
    join_col: str,
    parent_ts_col: str,
    dest: Path,
    start: str,
    end: str,
) -> int:
    conn.row_factory = sqlite3.Row
    try:
        rows = [
            dict(row)
            for row in conn.execute(
                f"""
                SELECT child.*
                FROM {child} AS child
                JOIN {parent} AS parent
                  ON parent.{join_col}=child.{join_col}
                WHERE parent.{parent_ts_col} >= ?
                  AND parent.{parent_ts_col} < ?
                ORDER BY parent.{parent_ts_col}, child.rowid
                """,
                (start, end),
            )
        ]
    except sqlite3.OperationalError:
        rows = []
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

            # Keep the legacy lightweight path for backwards-compatible analysis.
            path_rows = [
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
            counts["alpha_snapshots"] = write_rows(
                path_rows, out_dir / "alpha_spy_path.csv"
            )

            counts["alpha_market_snapshots"] = dump_between(
                alpha,
                "market_snapshots",
                "captured_at",
                out_dir / "alpha_market_snapshots.csv",
                start,
                end,
            )
            counts["alpha_snapshot_quotes"] = dump_child_by_parent_time(
                alpha,
                child="snapshot_quotes",
                parent="market_snapshots",
                join_col="snapshot_id",
                parent_ts_col="captured_at",
                dest=out_dir / "alpha_snapshot_quotes.csv",
                start=start,
                end=end,
            )
            counts["alpha_option_chain_snapshots"] = dump_between(
                alpha,
                "option_chain_snapshots",
                "captured_at",
                out_dir / "alpha_option_chain_snapshots.csv",
                start,
                end,
            )
            counts["alpha_option_quotes"] = dump_child_by_parent_time(
                alpha,
                child="option_quotes",
                parent="option_chain_snapshots",
                join_col="chain_snapshot_id",
                parent_ts_col="captured_at",
                dest=out_dir / "alpha_option_quotes.csv",
                start=start,
                end=end,
            )
            counts["alpha_surface_metrics"] = dump_between(
                alpha,
                "surface_metrics",
                "created_at",
                out_dir / "alpha_surface_metrics.csv",
                start,
                end,
            )
            counts["alpha_features"] = dump_between(
                alpha, "features", "created_at", out_dir / "alpha_features.csv", start, end
            )
            counts["alpha_predictions"] = dump_between(
                alpha,
                "predictions",
                "created_at",
                out_dir / "alpha_predictions.csv",
                start,
                end,
            )
            counts["alpha_candidates"] = dump_between(
                alpha,
                "candidates",
                "created_at",
                out_dir / "alpha_candidates.csv",
                start,
                end,
            )
            counts["alpha_decisions"] = dump_between(
                alpha,
                "decisions",
                "created_at",
                out_dir / "alpha_decisions.csv",
                start,
                end,
            )
            counts["alpha_orders"] = dump_between(
                alpha, "orders", "created_at", out_dir / "alpha_orders.csv", start, end
            )
            counts["alpha_positions"] = dump_between(
                alpha,
                "positions",
                "opened_at",
                out_dir / "alpha_positions.csv",
                start,
                end,
            )
            counts["alpha_prediction_outcomes"] = dump_between(
                alpha,
                "prediction_outcomes",
                "confirmed_at",
                out_dir / "alpha_prediction_outcomes.csv",
                start,
                end,
            )
            counts["alpha_candidate_outcomes"] = dump_between(
                alpha,
                "candidate_outcomes",
                "confirmed_at",
                out_dir / "alpha_candidate_outcomes.csv",
                start,
                end,
            )
            alpha.close()

        (out_dir / "manifest.txt").write_text(
            "\n".join(
                [
                    f"day={day}",
                    f"session_start_utc={start}",
                    f"session_end_utc={end}",
                    f"exported_at={datetime.now(ET).isoformat()}",
                    "schema=v2_full_replay_evidence",
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
