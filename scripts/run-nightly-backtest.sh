#!/usr/bin/env bash
set -Eeuo pipefail
cd "${BETA_SPY_HOME:-/opt/beta-spy}"
if [[ -f /etc/beta-spy/secrets.env ]]; then
  set -a
  # shellcheck disable=SC1091
  source /etc/beta-spy/secrets.env
  set +a
fi
STAMP="$(date -u +%Y-%m-%d)"
exec "${BETA_SPY_BIN:-beta-spy}" nightly \
  --db "${BETA_SPY_DB:-/var/lib/beta-spy/beta-spy.sqlite}" \
  --universe "${BETA_SPY_UNIVERSE:-/var/lib/beta-spy/universe.csv}" \
  --days "${BETA_SPY_BACKTEST_DAYS:-20}" \
  --interval "${BETA_SPY_BACKTEST_INTERVAL:-1min}" \
  --output "${BETA_SPY_REPORT_DIR:-/var/lib/beta-spy/reports}/backtest-${STAMP}"
