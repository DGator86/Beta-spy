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
# /api/v1/state, not /health: the decision API's /health handler can block
# indefinitely outside market hours, while state is the endpoint the rest of
# the stack actually depends on.
check_http "alpha-spy decision API" http://127.0.0.1:8787/api/v1/state
check_http "alpha-spy dashboard" http://127.0.0.1:8788/health
check_http "beta-spy dashboard" http://127.0.0.1:8790/api/health

# 2b. Alpha forecast freshness during the session. The decision API can
# answer 200 with days-old state while the engine crash-loops (it did for
# two days after a pandas upgrade), so liveness must be judged by the
# forecast timestamp, not the HTTP code.
dow="$(date -u +%u)"
hm="$(date -u +%H%M)"
if (( dow <= 5 )) && [[ "$hm" > "1340" && "$hm" < "2000" ]]; then
  VIEW_TOKEN="$(grep -s '^ALPHA_SPY_VIEW_TOKEN=' /etc/alpha-spy/secrets.env | cut -d= -f2)"
  if [[ -n "$VIEW_TOKEN" ]]; then
    AGE="$(curl -s -m 10 -H "X-Dashboard-Token: $VIEW_TOKEN" \
        http://127.0.0.1:8788/api/v1/dashboard/state 2>/dev/null \
      | python3 -c '
import json, sys
from datetime import UTC, datetime
try:
    d = json.load(sys.stdin)
    created = d["forecast_horizons"]["15m"]["created_at"]
    dt = datetime.fromisoformat(created.replace("Z", "+00:00"))
    print(int((datetime.now(UTC) - dt).total_seconds()))
except Exception:
    print(-1)
')"
    if [[ "$AGE" == "-1" ]]; then
      note "alpha-spy forecast freshness unknown (dashboard state unreadable)"
    elif (( AGE > 600 )); then
      note "alpha-spy 15m forecast is STALE: ${AGE}s old during market hours (engine likely failing)"
    fi
  fi
fi

# 2c. Disk headroom. Alpha's collector writes ~20 GB/day; a full disk takes
# down every service on the box at once.
DISK_PCT="$(df --output=pcent / | tail -1 | tr -dc 0-9)"
if (( DISK_PCT >= 90 )); then
  note "disk is ${DISK_PCT}% full: services will fail when it reaches 100%"
fi

# 3. Code-revert detection for the improved beta-spy build.
PKG=/opt/beta-spy/venv/lib/python3.12/site-packages/beta_spy
if [[ -d "$PKG" ]] && ! grep -q plan_best_strategy "$PKG/options.py" 2>/dev/null; then
  note "beta-spy installed code has REVERTED to the pre-improvement build"
fi

# 4. Tape freshness during the US session (Mon-Fri 13:35-20:00 UTC).
# A 502 handshake used to kill the Tradier thread while systemd still
# reported beta-spy as active. Restart when the tape is stale so the next
# session is not spent serving yesterday's DEGRADED snapshot.
if (( dow <= 5 )) && [[ "$hm" > "1335" && "$hm" < "2000" ]]; then
  last="$(sqlite3 "file:/var/lib/beta-spy/beta-spy.sqlite?mode=ro" \
    "SELECT CAST(strftime('%s','now') - strftime('%s', MAX(timestamp)) AS INTEGER) FROM minute_bars" \
    2>/dev/null || echo "")"
  if [[ -z "$last" ]]; then
    note "beta-spy tape freshness unknown (freshness query failed)"
  elif (( last > 300 )); then
    note "beta-spy tape is stale: last minute bar is ${last}s old during market hours"
    LAST_RESTART="$(cat "$STATE_DIR/last-beta-tape-restart" 2>/dev/null || echo 0)"
    NOW_TS="$(date +%s)"
    if (( NOW_TS - LAST_RESTART > 600 )); then
      if systemctl restart beta-spy >/dev/null 2>&1; then
        echo "$NOW_TS" >"$STATE_DIR/last-beta-tape-restart"
        note "beta-spy restarted because the Tradier tape was stale"
      else
        note "beta-spy restart FAILED while the tape was stale"
      fi
    else
      note "beta-spy restart skipped; last tape restart was $((NOW_TS - LAST_RESTART))s ago"
    fi
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
