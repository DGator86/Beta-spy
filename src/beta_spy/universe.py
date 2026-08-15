from __future__ import annotations

import csv
import io
import re
from pathlib import Path

import httpx
import pandas as pd

from .models import HoldingMeta

SSGA_SPY_HOLDINGS_URL = (
    "https://www.ssga.com/us/en/intermediary/library-content/products/"
    "fund-data/etfs/us/holdings-daily-us-en-spy.xlsx"
)
WIKIPEDIA_SP500_URL = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"


def normalize_symbol(symbol: str) -> str:
    return re.sub(r"\s+", "", str(symbol).strip().upper().replace(".", "/"))


def load_universe_csv(path: str | Path) -> list[HoldingMeta]:
    rows: list[HoldingMeta] = []
    with Path(path).open("r", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            symbol = normalize_symbol(row.get("symbol") or row.get("ticker") or "")
            if not symbol:
                continue
            try:
                weight = float(row.get("weight") or 0.0)
            except (TypeError, ValueError):
                continue
            if weight > 1.0:
                weight /= 100.0
            if weight <= 0:
                continue
            rows.append(
                HoldingMeta(
                    symbol=symbol,
                    sector=str(row.get("sector") or "Unknown").strip() or "Unknown",
                    weight=weight,
                    name=str(row.get("name") or symbol).strip() or symbol,
                )
            )
    return _normalize_weights(rows)


def fetch_current_spy_universe(timeout: float = 30.0) -> list[HoldingMeta]:
    """Fetch current SPY holdings/weights and enrich sectors when possible."""
    headers = {"User-Agent": "Beta-spy/0.1 (+research workstation)"}
    with httpx.Client(timeout=timeout, follow_redirects=True, headers=headers) as client:
        response = client.get(SSGA_SPY_HOLDINGS_URL)
        response.raise_for_status()
        holdings = _parse_ssga_xlsx(response.content)

    sectors: dict[str, tuple[str, str]] = {}
    try:
        tables = pd.read_html(WIKIPEDIA_SP500_URL)
        table = tables[0]
        for _, row in table.iterrows():
            symbol = normalize_symbol(row.get("Symbol", ""))
            if symbol:
                sectors[symbol] = (
                    str(row.get("GICS Sector", "Unknown")),
                    str(row.get("Security", symbol)),
                )
    except Exception:
        sectors = {}

    enriched = [
        HoldingMeta(
            symbol=item.symbol,
            sector=sectors.get(item.symbol, (item.sector, item.name))[0],
            weight=item.weight,
            name=sectors.get(item.symbol, (item.sector, item.name))[1],
        )
        for item in holdings
    ]
    return _normalize_weights(enriched)


def save_universe_csv(path: str | Path, holdings: list[HoldingMeta]) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["symbol", "name", "sector", "weight"])
        for item in holdings:
            writer.writerow([item.symbol, item.name, item.sector, f"{item.weight:.12g}"])


def _parse_ssga_xlsx(payload: bytes) -> list[HoldingMeta]:
    raw = pd.read_excel(io.BytesIO(payload), header=None)
    header_row = None
    for index in range(min(30, len(raw))):
        cells = [str(cell).strip().lower() for cell in raw.iloc[index].tolist()]
        if "ticker" in cells and any("weight" in cell for cell in cells):
            header_row = index
            break
    if header_row is None:
        raise ValueError("Could not locate Ticker/Weight header in SSGA workbook")

    frame = pd.read_excel(io.BytesIO(payload), header=header_row)
    columns = {str(column).strip().lower(): column for column in frame.columns}
    ticker_col = columns.get("ticker")
    name_col = columns.get("name")
    sector_col = columns.get("sector")
    weight_col = next((column for key, column in columns.items() if "weight" in key), None)
    if ticker_col is None or weight_col is None:
        raise ValueError("SSGA workbook is missing ticker/weight columns")

    holdings: list[HoldingMeta] = []
    for _, row in frame.iterrows():
        symbol = normalize_symbol(row.get(ticker_col, ""))
        if not symbol or symbol in {"-", "NAN", "CASH", "USD"}:
            continue
        # Index files carry placeholder identifiers (e.g. "2602335D") for
        # cash/pending lines; real US equity tickers are letters with an
        # optional class suffix.
        if not re.fullmatch(r"[A-Z]+(/[A-Z]+)?", symbol):
            continue
        try:
            raw_weight = str(row.get(weight_col, "0")).replace("%", "").replace(",", "")
            weight = float(raw_weight) / 100.0
        except (TypeError, ValueError):
            continue
        if weight <= 0:
            continue
        sector = str(row.get(sector_col, "Unknown")) if sector_col is not None else "Unknown"
        if sector.strip() in {"", "-", "--", "nan", "N/A"}:
            sector = "Unknown"
        name = str(row.get(name_col, symbol)) if name_col is not None else symbol
        holdings.append(HoldingMeta(symbol=symbol, sector=sector, weight=weight, name=name))
    return _normalize_weights(holdings)


def _normalize_weights(items: list[HoldingMeta]) -> list[HoldingMeta]:
    total = sum(max(item.weight, 0.0) for item in items)
    if total <= 0:
        raise ValueError("Universe weights sum to zero")
    return [
        HoldingMeta(item.symbol, item.sector, item.weight / total, item.name)
        for item in sorted(items, key=lambda item: item.weight, reverse=True)
    ]
