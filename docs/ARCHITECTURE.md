# Beta-spy architecture

Beta-spy is a standalone SPY research and paper-signal workstation. Its core hypothesis is deliberately falsifiable: **does the observable behavior of the stocks underneath SPY improve 5/15/30-minute SPY forecasts over SPY-only signals?**

## Design rules

1. Every core predictive input must come from ordinary stock bars, trades, or quotes.
2. All constituent features are computed point-in-time; future prices are used only after labels mature.
3. Equal-weight and SPY-weighted internals are both first-class signals.
4. `NO_TRADE` is a normal decision outcome.
5. SPY options are downstream expressions of a qualified signal, never inputs to the directional forecast.
6. The package does not submit broker orders.

## Live pipeline

```text
Tradier quote + uniquely sequenced timesale stream
                    │
                    ▼
           per-symbol minute state
                    │
        ┌───────────┼────────────┐
        ▼           ▼            ▼
   indicators     flow         volume
        └───────────┼────────────┘
                    ▼
        equal-weight / SPY-weighted
        breadth + sector participation
                    │
                    ▼
             5m / 15m / 30m
             causal online models
                    │
                    ▼
             decision gates
               │          │
          NO_TRADE      TRADE
                          │
                          ▼
             SPY debit-spread plan
```

## Per-symbol state

The rolling feature engine calculates 1/5/15-minute returns, session VWAP, EMA 8/21, RSI(14), ATR(14), 20-minute realized volatility, relative volume, range expansion, top-of-book spread/imbalance, aggressor-volume imbalance, trade intensity, price-impact proxy, and absorption proxy.

"Order flow" means consolidated tape plus top-of-book inference. It is not Level-II depth.

## 500-stock aggregation

Each minute constituent features are aggregated both equal-weight and by current SPY weight. The factor vector includes trend, momentum, volume, order-flow, volatility, percent above VWAP, percent EMA-bullish, positive-return breadth, positive-flow breadth, participation, concentration, breadth acceleration, sector factors, and SPY's own state.

The EW-vs-SPY-weighted difference exposes broad participation versus narrow mega-cap leadership.

## Forecasting and causality

`OnlineForecastStack` maintains independent 5-, 15-, and 30-minute models. Each horizon has a logistic direction model and expected-return regression model. A feature vector is not trained against a future price until that horizon has actually matured. Historical replay and live processing use the same classes.

## Storage

SQLite stores compact point-in-time universe snapshots, minute bars, minute flow aggregates, factors, forecasts, decisions, and optional raw SPY tape. All-constituent raw ticks are intentionally optional; minute aggregates are the permanent research substrate.

## Historical research

Two layers are supported:

- **Bars:** sufficient for indicators, breadth, sectors, concentration and the baseline forecast.
- **Trades + quotes:** reduced directly into minute flow rows so historical research does not need to retain every raw event.

Recent bars can be bootstrapped from Tradier with `beta-spy nightly`. Alpaca can provide longer bar history and optional historical trade/quote flow where the account/feed permits it.

## Backtest contract

Replay proceeds timestamp by timestamp and never reads observations later than the simulated clock. Reports distinguish cold-start/fallback forecasts from model-ready forecasts. Long historical tests require point-in-time SPY membership and weights; otherwise survivorship/weight bias must be disclosed.

## Deployment

The package includes a FastAPI/WebSocket workstation, a demo generator, a one-shot nightly backtest shell script, and systemd service/timer examples. Runtime secrets belong outside Git in environment variables or `/etc/beta-spy/secrets.env`.
