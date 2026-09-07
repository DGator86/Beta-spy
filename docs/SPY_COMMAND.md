# SPY Command

SPY Command is the combined, read-only operating console for Alpha-SPY and Beta-spy.
Alpha and Beta remain independent decision engines. Delta is a separate convergence/research layer and never feeds a value back into either engine or submits an order.

## Product surfaces

- **Command** — SPY live chart, Alpha/Beta posture, live agreement/divergence, Alpha execution decision, gate ladder, tape/regime context, alerts and active position.
- **Alpha** — multi-horizon forecast stack, regime hierarchy, entry gates, option candidate book, forecast audit and engine services.
- **Beta** — full S&P constituent sensor matrix, 5/15/30m forecast stack, breadth/flow posture, sector tape and Beta decision gates.
- **Gamma** — optional fresh-catalyst contract. News/catalyst events are time-decayed and may promote a technically confirmed individual constituent from watch/confirming into ACTIVE_B or ACTIVE_A+.
- **Delta** — merges Alpha, Beta, Gamma and Alpha's read-only options telemetry into the $100k competition research list and multi-horizon SPY predictor.
- **Performance** — Alpha live forecast audit, Beta causal backtest metrics, Beta decision-layer metrics and the current cross-engine read.
- **System** — Alpha/Beta endpoint latency, systemd service health, host load/disk posture and data-source diagnostics.

The dashboard is a single self-contained HTML file with no frontend build step and no CDN dependency.

## Competition / Delta contract

Delta scans every currently available S&P constituent from Beta. The current top 125 constituents by index weight are reported separately, but the full constituent set is retained in the SPY calculation and is **not** renormalized to make the top 125 equal 100%.

For each constituent, Delta combines Beta's trend, momentum, VWAP, relative volume, order-flow, structure/auction and sector signals with an optional Gamma catalyst and Alpha constituent IV observations. Options-implied movement is treated as magnitude only; direction comes from the predictive/confirming signal stack.

A stock can be classified as:

- `PREDICTIVE_SETUP`
- `ARMED`
- `CONFIRMING`
- `ACTIVE_B`
- `ACTIVE_A+`
- `EXTENDED`
- `DETERIORATING`

Individual active trades include entry zone, confirmation/retest entry, do-not-chase level, hard invalidation, T1/T2/T3, time stop, thesis exit, expected hold, modeled reward/risk and $100k-paper-account sizing. Delta will not promote an individual name to `ACTIVE_A+` or `ACTIVE_B` without a fresh Gamma catalyst.

### Horizon-aware SPY constituent pressure

Delta does not reuse one constituent pressure number for every forecast horizon. Each constituent carries its own `expected_realization_minutes`. Before that window completes, its expected displacement enters SPY on a square-root realization curve; once the window is complete the modeled displacement remains incorporated in the expected price level. `EXTENDED` and `DETERIORATING` states are explicitly down-weighted.

Delta produces matching constituent pressure for:

- open / 5 minutes
- 30 minutes
- 2 hours
- close
- 3 days
- 5 days

Those horizon-specific weighted constituent forecasts are blended with the matching Alpha/Beta forecasts. SPY's own ATM straddle or IV supplies a **directionless** implied-magnitude benchmark; it never supplies Delta's direction.

## Gamma catalyst input

Default path:

```text
/var/lib/spy-overview/gamma-catalysts.json
```

Optional URL:

```text
GAMMA_CATALYSTS_URL=https://...
```

See `config/gamma-catalysts.example.json` for the schema. Important fields are ticker, publication timestamp, direction, strength, materiality, novelty, confidence, catalyst type, expected realization horizon, and `reference_price` captured when the catalyst is discovered. The reference price is how Delta estimates how much of the expected move has already been consumed.

## ChatGPT live-state bridge

The Delta deployer generates a random persistent filename under:

```text
/var/www/spy-overview/chatgpt/<random-slug>.json
```

`config/nginx-spy.conf` exempts **only** `/chatgpt/` from dashboard basic auth. The raw dashboard, Alpha/Beta APIs, account state and control surfaces remain authenticated. The ChatGPT document is deliberately scrubbed: no broker credentials, no account identifiers, no order API and no execution authority.

`spy-tunnel-url.sh` combines the current TryCloudflare tunnel URL with that random path and uploads the resulting URL to the existing Google Drive backup folder as:

```text
chatgpt-url.txt
```

That allows a connected ChatGPT session to locate the current live market-state endpoint without receiving the dashboard password or brokerage secrets. The quick tunnel URL can rotate; the existing tunnel URL timer republishes the new address when it changes.

## Source and deployment

Core source:

- `config/overview-index.html`
- `config/nginx-spy.conf`
- `scripts/spy-overview-status.py` — existing Alpha/Beta overview collector
- `scripts/spy-overview-delta.py` — Delta v1 compatibility layer
- `scripts/spy-overview-delta-v2.py` — horizon-aware production-facing Delta publisher
- `scripts/spy-tunnel-url.sh` — dashboard + ChatGPT URL publisher
- `systemd/spy-overview-status.service`
- `systemd/spy-overview-status.timer`
- `systemd/spy-tunnel-url.service`
- `systemd/spy-tunnel-url.timer`

Runtime copies:

- `/var/www/spy-overview/index.html`
- `/usr/local/lib/spy-overview-base.py`
- `/usr/local/lib/spy-overview-delta-v1.py`
- `/usr/local/sbin/spy-overview-status`
- `/var/www/spy-overview/status.json`
- `/var/www/spy-overview/chatgpt/<random-slug>.json`

Deploy the overview/Delta bridge only:

```bash
sudo bash ./scripts/deploy-spy-overview.sh
```

The deployer preserves the existing `/etc/spy-overview.env`, generates the ChatGPT slug once, validates Python and shell assets, discovers and safely updates the existing source-controlled SPY nginx server block when present, validates nginx before reload, installs the status/tunnel timers, generates the live document immediately and asks the existing Drive URL publisher to publish its current URL.

This deploy script intentionally does **not** touch `src/beta_spy/`.

### Important Beta-spy packaging rule

Anything under `src/beta_spy/`, including Beta's internal dashboard static files, still requires the package reinstall path on the VPS:

```bash
cd /opt/beta-spy/src
/opt/beta-spy/venv/bin/pip install .
systemctl restart beta-spy
```

An rsync into `/opt/beta-spy/src/src/beta_spy/static/` alone does not update the installed runtime copy under site-packages.

## Aggregator inputs

By default the generator reads:

- Alpha: `http://127.0.0.1:8788/api/v1/dashboard/state`
- Beta: `http://127.0.0.1:8790/api/state`
- Alpha SQLite: `/var/lib/alpha-spy/journal/alpha-spy.db` for read-only SPY option chains and constituent IV observations
- Beta SQLite candidates, including `/opt/beta-spy/src/data/beta-spy.sqlite`
- Beta backtest candidates, including `/opt/beta-spy/src/reports/backtest-latest.json`
- optional Gamma catalyst JSON
- local systemd state for Alpha, Beta, the overview timer, tunnel and nginx

Optional `/etc/spy-overview.env` variables include:

```bash
ALPHA_STATE_URL=http://127.0.0.1:8788/api/v1/dashboard/state
BETA_STATE_URL=http://127.0.0.1:8790/api/state
ALPHA_DASHBOARD_TOKEN=...
ALPHA_DB=/var/lib/alpha-spy/journal/alpha-spy.db
BETA_DB=/opt/beta-spy/src/data/beta-spy.sqlite
BETA_BACKTEST_JSON=/opt/beta-spy/src/reports/backtest-latest.json
GAMMA_CATALYSTS_PATH=/var/lib/spy-overview/gamma-catalysts.json
COMPETITION_EQUITY=100000
ALPHA_PUBLIC_URL=https://your-host:8081/
BETA_PUBLIC_URL=https://your-host:8082/
OVERVIEW_STATUS_PATH=/var/www/spy-overview/status.json
CHATGPT_STATUS_PATH=/var/www/spy-overview/chatgpt/<generated-slug>.json
OVERVIEW_UNITS=alpha-spy,beta-spy,spy-overview-status.timer,spy-tunnel,nginx
```

The deploy script manages `CHATGPT_STATUS_PATH`; do not hard-code the generated slug in source control.

## Agreement index

The original Alpha × Beta **Live Agreement Index** remains observational. Delta uses Alpha/Beta/constituent convergence only in its own research output; it does not change Alpha/Beta execution decisions. Any future execution use must be separately validated on captured, timestamp-valid data.

## Failure behavior

The overview/Delta layer is fail-visible and fail-closed:

- an unavailable engine is marked offline;
- endpoint errors become overview alerts;
- missing Gamma means individual names cannot become `ACTIVE_A+`/`ACTIVE_B` from catalyst logic;
- missing Alpha options telemetry leaves implied-move fields unavailable rather than fabricated;
- missing SQLite history leaves the chart explicitly empty;
- partial state renders unavailable values rather than `NaN`;
- status files are replaced atomically;
- no Delta failure can place an order or alter Alpha/Beta decision state.
