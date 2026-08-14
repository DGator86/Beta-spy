# Beta-spy

Beta-spy is the deliberately simpler sibling of Alpha-Spy: **all SPY constituents are sensors, SPY is the trading target, and the predictive input set is restricted to data that is ordinary, observable, archivable, and replayable.**

It uses:

- 1-minute constituent price/volume bars
- VWAP, EMA 8/21, RSI, ATR, realized volatility, relative volume and momentum
- consolidated top-of-book quotes
- uniquely sequenced time-and-sales prints
- aggressor-volume and quote imbalance proxies
- equal-weight and SPY-weighted breadth
- sector participation
- causal 5/15/30-minute online forecasts
- a fail-closed decision engine with `NO_TRADE` as a normal outcome
- SPY options **only after** the tape creates a qualified directional signal

The options layer does not feed the forecast. It only expresses the signal as a defined-risk call or put debit spread.

## Architecture

```text
Tradier consolidated stream
       │
       ├── quote ───────────────┐
       └── uniquely-sequenced   │
           timesale prints      │
                               ▼
                     500 constituent states
                   ┌───────────┼────────────┐
                   │           │            │
              indicators     order flow    volume
                   └───────────┼────────────┘
                               ▼
                   equal-weight + SPY-weighted
                   breadth + sector participation
                               ▼
                       5 / 15 / 30m models
                               ▼
                         decision gates
                               ▼
                       TRADE or NO_TRADE
                               │
                       (only if TRADE)
                               ▼
                     SPY debit-spread planner
```

Every minute can be persisted to SQLite and replayed through the same feature/forecast path.

## Quick start: GUI without credentials

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e '.[dev]'
beta-spy demo --host 0.0.0.0 --port 8790
```

Open `http://127.0.0.1:8790`.

The demo deliberately accelerates synthetic market minutes so the online forecast models warm up quickly.

## Live Tradier tape

```bash
export TRADIER_MARKET_ACCESS_TOKEN='...'
beta-spy refresh-universe --output config/universe.csv
beta-spy run --universe config/universe.csv --host 0.0.0.0 --port 8790
```

The live stream subscribes to `quote` and `timesale` only. It does **not** also count `trade`, because Tradier's time-and-sale stream is uniquely sequenced and counting both families can double-count prints.

## Historical data

Historical minute bars and historical trade/quote aggregation are supported through Alpaca:

```bash
export APCA_API_KEY_ID='...'
export APCA_API_SECRET_KEY='...'

beta-spy backfill-bars \
  --universe config/universe.csv \
  --start 2026-08-01T13:30:00Z \
  --end   2026-08-01T20:00:00Z

beta-spy backfill-flow \
  --universe config/universe.csv \
  --start 2026-08-01T13:30:00Z \
  --end   2026-08-01T20:00:00Z

beta-spy replay --universe config/universe.csv
```

`iex` is the default Alpaca historical feed. Use SIP historical data when your plan provides it and you need historical flow to resemble the consolidated live feed more closely.

## Historical fetch + causal backtest

Beta-spy can bootstrap a recent full-universe intraday corpus directly from Tradier. The `nightly` command refreshes current SPY holdings, downloads recent Time & Sales bars for every constituent plus SPY, replays the production indicator/breadth/forecast pipeline in timestamp order, and writes Markdown and JSON metrics.

```bash
export TRADIER_MARKET_ACCESS_TOKEN='...'
beta-spy nightly --days 20 --interval 1min --output reports/backtest-latest
```

The report separates all forecasts from **model-ready** forecasts after the online learner has accumulated matured labels. It reports 5m/15m/30m direction accuracy, Brier score, expected-return MAE, decision-layer direction accuracy, and the most common `NO_TRADE` gates.

If Alpaca credentials are available, Beta-spy can optionally add historical trade/quote-derived minute flow:

```bash
export APCA_API_KEY_ID='...'
export APCA_API_SECRET_KEY='...'
beta-spy nightly --provider alpaca --feed iex --with-flow --days 5
```

A bars-only Tradier backtest validates the 500-stock indicator/breadth engine. Historical order-flow validation requires historical trades+quotes or Beta-spy's own captured tape. The report states which mode was actually tested.

## Backtest limitation that matters

A long backtest requires **point-in-time S&P 500 membership and weights**. Replaying old periods with today's constituents introduces survivorship bias. Beta-spy stores the universe used for each research run going forward; historical membership should be supplied separately for older periods.

## Safety

Beta-spy does not submit orders. The options component returns a proposed, pessimistically-priced, defined-risk debit spread. Broker execution remains outside this package.

## Tests

```bash
pytest
```

See [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md) for factor definitions and the replay contract.
