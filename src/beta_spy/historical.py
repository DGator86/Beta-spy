from __future__ import annotations

from collections.abc import Iterable, Iterator
from datetime import UTC, datetime
from typing import Any

import httpx

from .models import MinuteBar
from .storage import Tape500Store


def alpaca_symbol(symbol: str) -> str:
    return symbol.replace("/", ".")


def alpha_symbol(symbol: str) -> str:
    return symbol.replace(".", "/")


class AlpacaHistoricalClient:
    """Minimal adapter for Alpaca's ordinary multi-symbol historical stock data."""

    def __init__(
        self,
        api_key: str,
        api_secret: str,
        *,
        base_url: str = "https://data.alpaca.markets",
        timeout: float = 30.0,
    ) -> None:
        self.client = httpx.Client(
            base_url=base_url.rstrip("/"),
            headers={
                "APCA-API-KEY-ID": api_key,
                "APCA-API-SECRET-KEY": api_secret,
                "Accept": "application/json",
            },
            timeout=timeout,
        )

    def close(self) -> None:
        self.client.close()

    def __enter__(self) -> "AlpacaHistoricalClient":
        return self

    def __exit__(self, *_: object) -> None:
        self.close()

    def iter_bars(
        self,
        symbols: Iterable[str],
        start: datetime,
        end: datetime,
        *,
        timeframe: str = "1Min",
        feed: str = "sip",
        adjustment: str = "split",
        chunk_size: int = 50,
    ) -> Iterator[MinuteBar]:
        normalized = list(dict.fromkeys(str(symbol) for symbol in symbols))
        for offset in range(0, len(normalized), chunk_size):
            chunk = normalized[offset : offset + chunk_size]
            yield from self._iter_bar_chunk(chunk, start, end, timeframe, feed, adjustment)

    def _iter_bar_chunk(
        self,
        symbols: list[str],
        start: datetime,
        end: datetime,
        timeframe: str,
        feed: str,
        adjustment: str,
    ) -> Iterator[MinuteBar]:
        page_token: str | None = None
        while True:
            params: dict[str, Any] = {
                "symbols": ",".join(alpaca_symbol(symbol) for symbol in symbols),
                "timeframe": timeframe,
                "start": _rfc3339(start),
                "end": _rfc3339(end),
                "feed": feed,
                "adjustment": adjustment,
                "limit": 10_000,
                "sort": "asc",
            }
            if page_token:
                params["page_token"] = page_token
            response = self.client.get("/v2/stocks/bars", params=params)
            response.raise_for_status()
            payload = response.json()
            bars = payload.get("bars") or {}
            if isinstance(bars, dict):
                for provider_symbol, rows in bars.items():
                    symbol = alpha_symbol(str(provider_symbol))
                    for row in rows or []:
                        timestamp = _row_time(row)
                        if timestamp is None:
                            continue
                        yield MinuteBar(
                            symbol=symbol,
                            timestamp=timestamp,
                            open=float(row["o"]),
                            high=float(row["h"]),
                            low=float(row["l"]),
                            close=float(row["c"]),
                            volume=float(row.get("v") or 0.0),
                            trade_count=int(row.get("n") or 0),
                            vwap=float(row["vw"]) if row.get("vw") is not None else None,
                        )
            page_token = payload.get("next_page_token")
            if not page_token:
                break

    def backfill_bars(
        self,
        store: Tape500Store,
        symbols: Iterable[str],
        start: datetime,
        end: datetime,
        **kwargs: Any,
    ) -> int:
        batch: list[MinuteBar] = []
        total = 0
        for bar in self.iter_bars(symbols, start, end, **kwargs):
            batch.append(bar)
            if len(batch) >= 5000:
                store.save_bars(batch)
                total += len(batch)
                batch.clear()
        if batch:
            store.save_bars(batch)
            total += len(batch)
        return total

    def iter_trades(
        self,
        symbol: str,
        start: datetime,
        end: datetime,
        *,
        feed: str = "sip",
    ) -> Iterator[dict[str, Any]]:
        yield from self._iter_rows("trades", symbol, start, end, feed=feed)

    def iter_quotes(
        self,
        symbol: str,
        start: datetime,
        end: datetime,
        *,
        feed: str = "sip",
    ) -> Iterator[dict[str, Any]]:
        yield from self._iter_rows("quotes", symbol, start, end, feed=feed)

    def _iter_rows(
        self,
        kind: str,
        symbol: str,
        start: datetime,
        end: datetime,
        *,
        feed: str,
    ) -> Iterator[dict[str, Any]]:
        page_token: str | None = None
        provider_symbol = alpaca_symbol(symbol)
        while True:
            params: dict[str, Any] = {
                "symbols": provider_symbol,
                "start": _rfc3339(start),
                "end": _rfc3339(end),
                "feed": feed,
                "limit": 10_000,
                "sort": "asc",
            }
            if page_token:
                params["page_token"] = page_token
            response = self.client.get(f"/v2/stocks/{kind}", params=params)
            response.raise_for_status()
            payload = response.json()
            rows_by_symbol = payload.get(kind) or {}
            rows = rows_by_symbol.get(provider_symbol) if isinstance(rows_by_symbol, dict) else []
            for row in rows or []:
                yield row
            page_token = payload.get("next_page_token")
            if not page_token:
                break

    def backfill_minute_flow(
        self,
        store: Tape500Store,
        symbols: Iterable[str],
        start: datetime,
        end: datetime,
        *,
        feed: str = "sip",
    ) -> int:
        """Reduce historical trades/quotes into one compact flow row per symbol/minute."""
        from .flow import FlowAccumulator
        from .models import QuoteTop, TradePrint

        total = 0
        batch: list[tuple[datetime, str, Any]] = []
        for symbol in dict.fromkeys(str(value) for value in symbols):
            trades = iter(self.iter_trades(symbol, start, end, feed=feed))
            quotes = iter(self.iter_quotes(symbol, start, end, feed=feed))
            trade = next(trades, None)
            quote = next(quotes, None)
            accumulator = FlowAccumulator()
            bucket: datetime | None = None

            def flush() -> None:
                nonlocal total
                if bucket is None:
                    return
                batch.append((bucket, symbol, accumulator.snapshot(window_seconds=60.0)))
                total += 1
                if len(batch) >= 5000:
                    store.save_flows(batch)
                    batch.clear()
                accumulator.reset()

            while trade is not None or quote is not None:
                trade_time = _row_time(trade)
                quote_time = _row_time(quote)
                use_quote = quote_time is not None and (trade_time is None or quote_time <= trade_time)
                event_time = quote_time if use_quote else trade_time
                if event_time is None:
                    if use_quote:
                        quote = next(quotes, None)
                    else:
                        trade = next(trades, None)
                    continue
                minute = event_time.replace(second=0, microsecond=0)
                if bucket is None:
                    bucket = minute
                elif minute != bucket:
                    flush()
                    bucket = minute
                if use_quote:
                    bid = _number((quote or {}).get("bp"))
                    ask = _number((quote or {}).get("ap"))
                    if bid is not None and ask is not None and bid > 0 and ask >= bid:
                        accumulator.on_quote(
                            QuoteTop(
                                symbol=symbol,
                                timestamp=event_time,
                                bid=bid,
                                ask=ask,
                                bid_size=_number((quote or {}).get("bs")),
                                ask_size=_number((quote or {}).get("as")),
                            )
                        )
                    quote = next(quotes, None)
                else:
                    price = _number((trade or {}).get("p"))
                    size = _number((trade or {}).get("s"))
                    if price is not None and size is not None and price > 0 and size > 0:
                        accumulator.on_trade(
                            TradePrint(
                                symbol=symbol,
                                timestamp=event_time,
                                price=price,
                                size=size,
                                sequence=_integer((trade or {}).get("i")),
                            )
                        )
                    trade = next(trades, None)
            flush()
        if batch:
            store.save_flows(batch)
        return total


def _rfc3339(value: datetime) -> str:
    if value.tzinfo is None:
        value = value.replace(tzinfo=UTC)
    return value.astimezone(UTC).isoformat().replace("+00:00", "Z")


def _row_time(row: dict[str, Any] | None) -> datetime | None:
    if not row or not row.get("t"):
        return None
    parsed = datetime.fromisoformat(str(row["t"]).replace("Z", "+00:00"))
    return parsed if parsed.tzinfo is not None else parsed.replace(tzinfo=UTC)


def _number(value: Any) -> float | None:
    try:
        return float(value) if value is not None else None
    except (TypeError, ValueError):
        return None


def _integer(value: Any) -> int | None:
    try:
        return int(value) if value is not None else None
    except (TypeError, ValueError):
        return None


class TradierHistoricalClient:
    """Tradier production Time & Sales downloader for recent intraday bars."""

    def __init__(
        self,
        access_token: str,
        *,
        base_url: str = "https://api.tradier.com/v1",
        timeout: float = 45.0,
    ) -> None:
        import time

        self._time = time
        self.client = httpx.Client(
            base_url=base_url.rstrip("/"),
            headers={"Authorization": f"Bearer {access_token}", "Accept": "application/json"},
            timeout=timeout,
        )

    def close(self) -> None:
        self.client.close()

    def __enter__(self) -> "TradierHistoricalClient":
        return self

    def __exit__(self, *_: object) -> None:
        self.close()

    def _get(self, path: str, *, params: dict[str, Any]) -> httpx.Response:
        attempts = 0
        while True:
            attempts += 1
            response = self.client.get(path, params=params)
            if response.status_code != 429:
                response.raise_for_status()
                available = _int_number(response.headers.get("X-Ratelimit-Available"))
                expiry_ms = _int_number(response.headers.get("X-Ratelimit-Expiry"))
                if available is not None and available <= 2 and expiry_ms is not None:
                    wait = max(0.0, expiry_ms / 1000.0 - self._time.time()) + 0.15
                    self._time.sleep(min(wait, 61.0))
                return response
            if attempts >= 6:
                response.raise_for_status()
            retry = response.headers.get("Retry-After")
            wait = float(retry) if retry else min(60.0, 2.0 ** attempts)
            self._time.sleep(wait)

    def iter_bars(
        self,
        symbol: str,
        start: datetime,
        end: datetime,
        *,
        interval: str = "1min",
        session_filter: str = "open",
    ) -> Iterator[MinuteBar]:
        if interval not in {"1min", "5min", "15min"}:
            raise ValueError("Tradier historical bars support 1min, 5min, or 15min")
        response = self._get(
            "/markets/timesales",
            params={
                "symbol": symbol,
                "interval": interval,
                "start": start.astimezone(UTC).strftime("%Y-%m-%d %H:%M"),
                "end": end.astimezone(UTC).strftime("%Y-%m-%d %H:%M"),
                "session_filter": session_filter,
            },
        )
        payload = response.json()
        series = payload.get("series") or {}
        rows = series.get("data") or [] if isinstance(series, dict) else []
        if isinstance(rows, dict):
            rows = [rows]
        for row in rows:
            if not isinstance(row, dict):
                continue
            timestamp = None
            raw_timestamp = _number(row.get("timestamp"))
            if raw_timestamp is not None:
                timestamp = datetime.fromtimestamp(raw_timestamp, tz=UTC)
            if timestamp is None:
                timestamp = _row_time(row)
            if timestamp is None:
                continue
            close = _number(row.get("close")) or _number(row.get("price"))
            if close is None or close <= 0:
                continue
            yield MinuteBar(
                symbol=symbol,
                timestamp=timestamp,
                open=_number(row.get("open")) or close,
                high=_number(row.get("high")) or close,
                low=_number(row.get("low")) or close,
                close=close,
                volume=_number(row.get("volume")) or 0.0,
                trade_count=0,
                vwap=_number(row.get("vwap")),
            )

    def backfill_bars(
        self,
        store: Tape500Store,
        symbols: Iterable[str],
        start: datetime,
        end: datetime,
        *,
        interval: str = "1min",
        session_filter: str = "open",
        progress_every: int = 25,
    ) -> int:
        total = 0
        symbols = list(dict.fromkeys(str(symbol) for symbol in symbols))
        for index, symbol in enumerate(symbols, start=1):
            rows = list(self.iter_bars(symbol, start, end, interval=interval, session_filter=session_filter))
            if rows:
                store.save_bars(rows)
                total += len(rows)
            if progress_every and (index % progress_every == 0 or index == len(symbols)):
                print(f"Tradier historical: {index}/{len(symbols)} symbols, {total:,} bars", flush=True)
        return total


def _int_number(value: Any) -> int | None:
    try:
        return int(float(value)) if value is not None else None
    except (TypeError, ValueError):
        return None
