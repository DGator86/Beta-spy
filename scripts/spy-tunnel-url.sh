#!/usr/bin/env bash
# Publish the current quick-tunnel URLs where the user and ChatGPT can find them.
#
# The dashboard stays behind basic auth. Delta's scrubbed read-only market-state
# document lives under a random /chatgpt/<slug>.json path that nginx exposes
# without dashboard credentials. The current URLs are uploaded to the same
# Google Drive backup folder already used by SPY Command.
set -uo pipefail

STATE_DIR=/var/lib/spy-watchdog
DASH_STATE="$STATE_DIR/tunnel-url.txt"
CHATGPT_STATE="$STATE_DIR/chatgpt-url.txt"
OUT=/var/www/spy-overview/tunnel-url.txt

URL="$(journalctl -u spy-tunnel --no-pager -o cat --since '-7 days' \
  | grep -Eo 'https://[a-z0-9-]+\.trycloudflare\.com' | tail -1)"
[[ -n "$URL" ]] || exit 0

echo "$URL" >"$OUT"

# The Delta deployer persists only the scrubbed output path here. It contains no
# broker token or basic-auth password.
if [[ -f /etc/spy-overview.env ]]; then
  set -a
  # shellcheck disable=SC1091
  source /etc/spy-overview.env
  set +a
fi

CHATGPT_URL=""
if [[ -n "${CHATGPT_STATUS_PATH:-}" ]]; then
  case "$CHATGPT_STATUS_PATH" in
    /var/www/spy-overview/*)
      REL="${CHATGPT_STATUS_PATH#/var/www/spy-overview/}"
      CHATGPT_URL="$URL/$REL"
      ;;
  esac
fi

if [[ -f /etc/alpha-spy/backup.env ]]; then
  set -a
  # shellcheck disable=SC1091
  source /etc/alpha-spy/backup.env
  set +a
fi
export RCLONE_CONFIG="${RCLONE_CONFIG:-/root/.config/rclone/rclone.conf}"
REMOTE="${ALPHA_SPY_BACKUP_REMOTE:-gdrive:SPY Trading Backups/$(hostname -s)}"
mkdir -p "$STATE_DIR"

DASH_LAST="$(cat "$DASH_STATE" 2>/dev/null || true)"
if [[ "$URL" != "$DASH_LAST" ]]; then
  {
    echo "SPY trading overview page (basic auth user: spy)"
    echo "$URL"
    echo "updated $(date -Is)"
  } >/tmp/dashboard-url.txt
  if rclone copyto /tmp/dashboard-url.txt "$REMOTE/dashboard-url.txt" \
      --retries 3 --low-level-retries 10 --timeout 2m >/dev/null 2>&1; then
    echo "$URL" >"$DASH_STATE"
    printf '%s tunnel URL published: %s\n' "$(date -Is)" "$URL" >>/var/log/spy-watchdog.log
  fi
fi

if [[ -n "$CHATGPT_URL" ]]; then
  CHATGPT_LAST="$(cat "$CHATGPT_STATE" 2>/dev/null || true)"
  if [[ "$CHATGPT_URL" != "$CHATGPT_LAST" ]]; then
    {
      echo "SPY Command Delta read-only ChatGPT market-state endpoint"
      echo "$CHATGPT_URL"
      echo "contains scrubbed market/model data only; no broker credentials or order API"
      echo "updated $(date -Is)"
    } >/tmp/chatgpt-url.txt
    if rclone copyto /tmp/chatgpt-url.txt "$REMOTE/chatgpt-url.txt" \
        --retries 3 --low-level-retries 10 --timeout 2m >/dev/null 2>&1; then
      echo "$CHATGPT_URL" >"$CHATGPT_STATE"
      printf '%s ChatGPT market-state URL published: %s\n' \
        "$(date -Is)" "$CHATGPT_URL" >>/var/log/spy-watchdog.log
    fi
  fi
fi

exit 0
