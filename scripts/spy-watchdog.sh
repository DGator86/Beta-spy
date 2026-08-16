#!/usr/bin/env bash
# Watchdog for Alpha-spy and Beta-spy.
#
# Every run it: restarts any down service, verifies each HTTP health
# endpoint, detects a code revert of the improved beta-spy build (a stale
# redeploy once silently wiped it), and checks tape freshness during the US
# session. Alerts are appended to /var/log/spy-watchdog.log and uploaded to
# the same Google Drive remote the backups use, deduplicated so an
# unchanged alert set re-uploads at most every six hours.
set -uo pipefail

STATE_DIR=/var/lib/spy-watchdog
LOG=/var/log/spy-watchdog.log
mkdir -p "$STATE_DIR"

if [[ -f /etc/alpha-spy/backup.env ]]; then
  set -a
  # shellcheck disable=SC1091
  source /etc/alpha-spy/backup.env
  set +a
fi
export RCLONE_CONFIG="${RCLONE_CONFIG:-/root/.config/rclone/rclone.conf}"
REMOTE="${ALPHA_SPY_BACKUP_REMOTE:-gdrive:SPY Trading Backups/$(hostname -s)}"

ALERTS=()
note() { ALERTS+=("$1"); }
log() { printf '%s %s\n' "$(date -Is)" "$*" >>"$LOG"; }

# 1. Services: restart anything that is down.
UNITS=(
  alpha-spy-market alpha-spy-engine alpha-spy-decision
  alpha-spy-confirmation alpha-spy-settlement alpha-spy-dashboard
  beta-spy nginx
)
for unit in "${UNITS[@]}"; do
  if ! systemctl is-active --quiet "$unit"; then
    systemctl restart "$unit" >/dev/null 2>&1
    sleep 2
    if systemctl is-active --quiet "$unit"; then
      note "$unit was down and has been restarted"
    else
      note "$unit is DOWN and restart FAILED"
    fi
  fi
done

# 2. HTTP health endpoints.
check_http() {
  local name="$1" url="$2" code
  code="$(curl -s -o /dev/null -m 10 -w '%{http_code}' "$url" 2>/dev/null)" || code=000
  [[ "$code" == "200" ]] || note "$name health endpoint returned HTTP $code"
}
check_http "alpha-spy decision API" http://127.0.0.1:8787/health
check_http "alpha-spy dashboard" http://127.0.0.1:8788/health
check_http "beta-spy dashboard" http://127.0.0.1:8790/api/health

# 3. Code-revert detection for the improved beta-spy build.
PKG=/opt/beta-spy/venv/lib/python3.12/site-packages/beta_spy
if [[ -d "$PKG" ]] && ! grep -q plan_best_strategy "$PKG/options.py" 2>/dev/null; then
  note "beta-spy installed code has REVERTED to the pre-improvement build"
fi

# 4. Tape freshness during the US session (Mon-Fri 13:35-20:00 UTC).
dow="$(date -u +%u)"
hm="$(date -u +%H%M)"
if (( dow <= 5 )) && [[ "$hm" > "1335" && "$hm" < "2000" ]]; then
  last="$(sqlite3 "file:/var/lib/beta-spy/beta-spy.sqlite?mode=ro" \
    "SELECT CAST(strftime('%s','now') - strftime('%s', MAX(timestamp)) AS INTEGER) FROM minute_bars" \
    2>/dev/null || echo "")"
  if [[ -z "$last" ]]; then
    note "beta-spy tape freshness unknown (freshness query failed)"
  elif (( last > 300 )); then
    note "beta-spy tape is stale: last minute bar is ${last}s old during market hours"
  fi
fi

# 5. Deliver alerts.
if ((${#ALERTS[@]})); then
  for alert in "${ALERTS[@]}"; do log "ALERT $alert"; done
  HASH="$(printf '%s\n' "${ALERTS[@]}" | sha256sum | cut -d' ' -f1)"
  LAST_HASH="$(cat "$STATE_DIR/last-alert-hash" 2>/dev/null || true)"
  LAST_TS="$(cat "$STATE_DIR/last-alert-ts" 2>/dev/null || echo 0)"
  NOW_TS="$(date +%s)"
  if [[ "$HASH" != "$LAST_HASH" ]] || (( NOW_TS - LAST_TS > 21600 )); then
    STAMP="$(date -u +%Y%m%dT%H%M%SZ)"
    FILE="$STATE_DIR/alert-$STAMP.txt"
    {
      echo "spy-watchdog alerts from $(hostname) at $(date -Is)"
      printf ' - %s\n' "${ALERTS[@]}"
    } >"$FILE"
    if rclone copyto "$FILE" "$REMOTE/alerts/$STAMP.txt" \
        --retries 3 --low-level-retries 10 --timeout 2m >/dev/null 2>&1; then
      log "alert uploaded to $REMOTE/alerts/$STAMP.txt"
    else
      log "alert upload FAILED"
    fi
    echo "$HASH" >"$STATE_DIR/last-alert-hash"
    echo "$NOW_TS" >"$STATE_DIR/last-alert-ts"
  fi
else
  log "OK all checks passed"
fi
exit 0
