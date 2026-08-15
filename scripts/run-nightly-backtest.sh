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
DB="${BETA_SPY_DB:-/var/lib/beta-spy/beta-spy.sqlite}"
UNIVERSE="${BETA_SPY_UNIVERSE:-/var/lib/beta-spy/universe.csv}"
REPORTS="${BETA_SPY_REPORT_DIR:-/var/lib/beta-spy/reports}"
BIN="${BETA_SPY_BIN:-beta-spy}"
PYTHON="${BETA_SPY_PYTHON:-/opt/beta-spy/venv/bin/python}"
SCRIPTS="${BETA_SPY_SCRIPTS:-/opt/beta-spy/src/scripts}"

"$BIN" nightly \
  --db "$DB" \
  --universe "$UNIVERSE" \
  --days "${BETA_SPY_BACKTEST_DAYS:-20}" \
  --interval "${BETA_SPY_BACKTEST_INTERVAL:-1min}" \
  --output "$REPORTS/backtest-${STAMP}"

# Self-audit: re-run combinatorial purged cross-validation on the freshly
# extended tape so overfitting of the production gates is re-measured every
# night as data grows. The report lands in the reports directory, which the
# daily Drive backup uploads. A single rolling signals dump is kept locally.
echo "self-audit: dumping signals and running CPCV…"
SIGNALS="$REPORTS/signals-latest.csv"
"$PYTHON" "$SCRIPTS/dump_signals.py" --db "$DB" --universe "$UNIVERSE" --output "$SIGNALS"
"$PYTHON" "$SCRIPTS/cscv_eval.py" --signals "$SIGNALS" | tee "$REPORTS/cscv-${STAMP}.txt"
echo "self-audit: report written to $REPORTS/cscv-${STAMP}.txt"
