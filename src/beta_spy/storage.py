from __future__ import annotations

import json
import sqlite3
import threading
from dataclasses import asdict
from datetime import datetime
from pathlib import Path
from typing import Iterable, Iterator

from .models import Decision, FlowFeatures, HorizonForecast, HoldingMeta, MarketFactors, MinuteBar


SCHEMA = """
PRAGMA journal_mode=WAL;
PRAGMA synchronous=NORMAL;
CREATE TABLE IF NOT EXISTS universe (
    asof TEXT NOT NULL,
    symbol TEXT NOT NULL,
    sector TEXT NOT NULL,
    weight REAL NOT NULL,
    name TEXT NOT NULL,
    PRIMARY KEY (asof, symbol)
);
CREATE TABLE IF NOT EXISTS minute_bars (
    timestamp TEXT NOT NULL,
    symbol TEXT NOT NULL,
    open REAL NOT NULL,
    high REAL NOT NULL,
    low REAL NOT NULL,
    close REAL NOT NULL,
    volume REAL NOT NULL,
    trade_count INTEGER NOT NULL,
    vwap REAL,
    PRIMARY KEY (timestamp, symbol)
);
CREATE INDEX IF NOT EXISTS minute_bars_symbol_time ON minute_bars(symbol, timestamp);
CREATE TABLE IF NOT EXISTS minute_flow (
    timestamp TEXT NOT NULL,
    symbol TEXT NOT NULL,
    buy_volume REAL NOT NULL,
    sell_volume REAL NOT NULL,
    neutral_volume REAL NOT NULL,
    order_flow_imbalance REAL,
    quote_imbalance REAL,
    average_spread_bps REAL,
    trade_intensity REAL NOT NULL,
    average_trade_size REAL,
    price_impact_bps_per_10k REAL,
    absorption REAL,
    quote_updates INTEGER NOT NULL,
    trades INTEGER NOT NULL,
    PRIMARY KEY (timestamp, symbol)
);
CREATE INDEX IF NOT EXISTS minute_flow_symbol_time ON minute_flow(symbol, timestamp);
CREATE TABLE IF NOT EXISTS factor_snapshots (
    timestamp TEXT PRIMARY KEY,
    payload TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS forecasts (
    timestamp TEXT NOT NULL,
    horizon_minutes INTEGER NOT NULL,
    probability_up REAL NOT NULL,
    expected_return_bps REAL NOT NULL,
    confidence REAL NOT NULL,
    model_ready INTEGER NOT NULL,
    sample_count INTEGER NOT NULL,
    PRIMARY KEY (timestamp, horizon_minutes)
);
CREATE TABLE IF NOT EXISTS decisions (
    timestamp TEXT PRIMARY KEY,
    payload TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS spy_trades (
    timestamp TEXT NOT NULL,
    sequence INTEGER,
    price REAL NOT NULL,
    size REAL NOT NULL,
    bid REAL,
    ask REAL,
    PRIMARY KEY (timestamp, sequence)
);
CREATE TABLE IF NOT EXISTS spy_quotes (
    timestamp TEXT NOT NULL,
    bid REAL NOT NULL,
    ask REAL NOT NULL,
    bid_size REAL,
    ask_size REAL,
    PRIMARY KEY (timestamp, bid, ask)
);
CREATE TABLE IF NOT EXISTS alpha_signals (
    timestamp TEXT PRIMARY KEY,
    payload TEXT NOT NULL
);
"""


def _iso(value: datetime) -> str:
    return value.isoformat().replace("+00:00", "Z")


class Tape500Store:
    def __init__(self, path: Path | str):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.lock = threading.RLock()
        self.connection = sqlite3.connect(self.path, check_same_thread=False, timeout=30)
        self.connection.execute("PRAGMA busy_timeout=30000")
        self.connection.executescript(SCHEMA)
        self.connection.commit()

    def close(self) -> None:
        self.connection.close()

    def save_universe(self, asof: datetime, holdings: Iterable[HoldingMeta]) -> None:
        rows = [(_iso(asof), item.symbol, item.sector, item.weight, item.name) for item in holdings]
        with self.lock:
            self.connection.executemany(
                "INSERT OR REPLACE INTO universe(asof,symbol,sector,weight,name) VALUES(?,?,?,?,?)",
                rows,
            )
            self.connection.commit()

    def save_bar(self, bar: MinuteBar) -> None:
        self.save_bars([bar])

    def save_bars(self, bars: Iterable[MinuteBar]) -> None:
        with self.lock:
            self.connection.executemany(
                """
                INSERT OR REPLACE INTO minute_bars
                (timestamp,symbol,open,high,low,close,volume,trade_count,vwap)
                VALUES(?,?,?,?,?,?,?,?,?)
                """,
                [
                    (
                        _iso(bar.timestamp),
                        bar.symbol,
                        bar.open,
                        bar.high,
                        bar.low,
                        bar.close,
                        bar.volume,
                        bar.trade_count,
                        bar.vwap,
                    )
                    for bar in bars
                ],
            )
            self.connection.commit()

    def save_flow(self, timestamp: datetime, symbol: str, flow: FlowFeatures) -> None:
        self.save_flows([(timestamp, symbol, flow)])

    def save_flows(self, rows: Iterable[tuple[datetime, str, FlowFeatures]]) -> None:
        with self.lock:
            self.connection.executemany(
                """
                INSERT OR REPLACE INTO minute_flow(
                    timestamp,symbol,buy_volume,sell_volume,neutral_volume,order_flow_imbalance,
                    quote_imbalance,average_spread_bps,trade_intensity,average_trade_size,
                    price_impact_bps_per_10k,absorption,quote_updates,trades
                ) VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?)
                """,
                [
                    (
                        _iso(timestamp), symbol, flow.buy_volume, flow.sell_volume, flow.neutral_volume,
                        flow.order_flow_imbalance, flow.quote_imbalance, flow.average_spread_bps,
                        flow.trade_intensity, flow.average_trade_size, flow.price_impact_bps_per_10k,
                        flow.absorption, flow.quote_updates, flow.trades,
                    )
                    for timestamp, symbol, flow in rows
                ],
            )
            self.connection.commit()

    def flows_for_timestamp(self, timestamp: datetime) -> dict[str, FlowFeatures]:
        with self.lock:
            cursor = self.connection.execute(
                """
                SELECT symbol,buy_volume,sell_volume,neutral_volume,order_flow_imbalance,quote_imbalance,
                       average_spread_bps,trade_intensity,average_trade_size,price_impact_bps_per_10k,
                       absorption,quote_updates,trades
                FROM minute_flow WHERE timestamp=?
                """,
                (_iso(timestamp),),
            )
            rows = cursor.fetchall()
        return {
            str(row[0]): FlowFeatures(
                buy_volume=float(row[1]), sell_volume=float(row[2]), neutral_volume=float(row[3]),
                order_flow_imbalance=float(row[4]) if row[4] is not None else None,
                quote_imbalance=float(row[5]) if row[5] is not None else None,
                average_spread_bps=float(row[6]) if row[6] is not None else None,
                trade_intensity=float(row[7]),
                average_trade_size=float(row[8]) if row[8] is not None else None,
                price_impact_bps_per_10k=float(row[9]) if row[9] is not None else None,
                absorption=float(row[10]) if row[10] is not None else None,
                quote_updates=int(row[11]), trades=int(row[12]),
            )
            for row in rows
        }

    def save_factors(self, factors: MarketFactors) -> None:
        with self.lock:
            self.connection.execute(
                "INSERT OR REPLACE INTO factor_snapshots(timestamp,payload) VALUES(?,?)",
                (_iso(factors.timestamp), json.dumps(asdict(factors), default=str, separators=(",", ":"))),
            )
            self.connection.commit()

    def save_forecasts(self, timestamp: datetime, forecasts: Iterable[HorizonForecast]) -> None:
        with self.lock:
            self.connection.executemany(
                """
                INSERT OR REPLACE INTO forecasts
                (timestamp,horizon_minutes,probability_up,expected_return_bps,confidence,model_ready,sample_count)
                VALUES(?,?,?,?,?,?,?)
                """,
                [
                    (
                        _iso(timestamp),
                        item.horizon_minutes,
                        item.probability_up,
                        item.expected_return_bps,
                        item.confidence,
                        int(item.model_ready),
                        item.sample_count,
                    )
                    for item in forecasts
                ],
            )
            self.connection.commit()

    def save_alpha_signal(self, timestamp: datetime, payload: dict) -> None:
        """Record Alpha-spy's concurrent stance for future agreement analysis."""
        with self.lock:
            self.connection.execute(
                "INSERT OR REPLACE INTO alpha_signals(timestamp,payload) VALUES(?,?)",
                (_iso(timestamp), json.dumps(payload, default=str, separators=(",", ":"))),
            )
            self.connection.commit()

    def save_decision(self, decision: Decision) -> None:
        with self.lock:
            self.connection.execute(
                "INSERT OR REPLACE INTO decisions(timestamp,payload) VALUES(?,?)",
                (_iso(decision.timestamp), json.dumps(asdict(decision), default=str, separators=(",", ":"))),
            )
            self.connection.commit()

    def iter_bars(self, start: datetime | None = None, end: datetime | None = None) -> Iterator[MinuteBar]:
        clauses: list[str] = []
        args: list[str] = []
        if start is not None:
            clauses.append("timestamp >= ?")
            args.append(_iso(start))
        if end is not None:
            clauses.append("timestamp <= ?")
            args.append(_iso(end))
        where = f"WHERE {' AND '.join(clauses)}" if clauses else ""
        with self.lock:
            rows = self.connection.execute(
                f"""
                SELECT timestamp,symbol,open,high,low,close,volume,trade_count,vwap
                FROM minute_bars {where}
                ORDER BY timestamp, symbol
                """,
                args,
            ).fetchall()
        for row in rows:
            yield MinuteBar(
                timestamp=datetime.fromisoformat(str(row[0]).replace("Z", "+00:00")),
                symbol=str(row[1]),
                open=float(row[2]),
                high=float(row[3]),
                low=float(row[4]),
                close=float(row[5]),
                volume=float(row[6]),
                trade_count=int(row[7]),
                vwap=float(row[8]) if row[8] is not None else None,
            )
