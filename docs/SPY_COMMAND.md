# SPY Command

SPY Command is the combined, read-only operating console for Alpha-SPY and Beta-spy.
It deliberately does **not** merge the engines or feed Alpha/Beta consensus back into either decision path.

## Product surfaces

- **Command** — SPY live chart, Alpha/Beta posture, live agreement/divergence, Alpha execution decision, gate ladder, tape/regime context, alerts and active position.
- **Alpha** — multi-horizon forecast stack, regime hierarchy, entry gates, option candidate book, forecast audit and engine services.
- **Beta** — 500-name constituent sensor matrix, 5/15/30m forecast stack, breadth/flow posture, sector tape and Beta decision gates.
- **Performance** — Alpha live forecast audit, Beta causal backtest metrics, Beta decision-layer metrics and the current cross-engine read.
- **System** — Alpha/Beta endpoint latency, systemd service health, host load/disk posture and data-source diagnostics.

The dashboard is a single self-contained HTML file with no frontend build step and no CDN dependency.

## Source and deployment

Source:

- `config/overview-index.html`
- `scripts/spy-overview-status.py`
- `systemd/spy-overview-status.service`
- `systemd/spy-overview-status.timer`

Runtime copies:

- `/var/www/spy-overview/index.html`
- `/usr/local/sbin/spy-overview-status`
- `/var/www/spy-overview/status.json`

Deploy the overview only:

```bash
sudo bash ./scripts/deploy-spy-overview.sh
```

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
- Beta SQLite candidates, including `/opt/beta-spy/src/data/beta-spy.sqlite`
- Beta backtest candidates, including `/opt/beta-spy/src/reports/backtest-latest.json`
- local systemd state for Alpha, Beta, the overview timer, tunnel and nginx

Optional `/etc/spy-overview.env` variables:

```bash
ALPHA_STATE_URL=http://127.0.0.1:8788/api/v1/dashboard/state
BETA_STATE_URL=http://127.0.0.1:8790/api/state
ALPHA_DASHBOARD_TOKEN=...
BETA_DB=/opt/beta-spy/src/data/beta-spy.sqlite
BETA_BACKTEST_JSON=/opt/beta-spy/src/reports/backtest-latest.json
ALPHA_PUBLIC_URL=https://your-host:8081/
BETA_PUBLIC_URL=https://your-host:8082/
OVERVIEW_STATUS_PATH=/var/www/spy-overview/status.json
OVERVIEW_UNITS=alpha-spy,beta-spy,spy-overview-status.timer,spy-tunnel,nginx
```

If `ALPHA_PUBLIC_URL` / `BETA_PUBLIC_URL` are omitted, the browser creates links using the current hostname and ports 8081/8082.

## Agreement index

The Alpha × Beta **Live Agreement Index** is intentionally labeled observational. It is based on current independent engine direction and directional confidence and reports shared-horizon direction matches. It is **not** promoted as a trading signal and should not become an execution input until separately validated on captured historical data.

## Failure behavior

The overview is fail-visible rather than fail-silent:

- an unavailable engine is marked offline;
- endpoint errors become overview alerts;
- missing SQLite history leaves the chart explicitly empty rather than fabricating data;
- missing backtest JSON leaves the performance table empty;
- partial Alpha/Beta state renders `—` instead of `NaN`;
- `status.json` is replaced atomically so nginx never serves a partially written document.
