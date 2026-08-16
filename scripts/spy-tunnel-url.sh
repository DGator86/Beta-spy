#!/usr/bin/env bash
# Publish the current quick-tunnel URL where the user can find it.
#
# Extracts the newest trycloudflare.com URL from the spy-tunnel journal,
# writes it to /var/www/spy-overview/tunnel-url.txt (so the overview page and
# anyone on the box can see it) and, when it changes, uploads it to the same
# Google Drive folder the backups use as dashboard-url.txt.
set -uo pipefail

STATE=/var/lib/spy-watchdog/tunnel-url.txt
OUT=/var/www/spy-overview/tunnel-url.txt

URL="$(journalctl -u spy-tunnel --no-pager -o cat --since '-7 days' \
  | grep -Eo 'https://[a-z0-9-]+\.trycloudflare\.com' | tail -1)"
[[ -n "$URL" ]] || exit 0

echo "$URL" >"$OUT"
LAST="$(cat "$STATE" 2>/dev/null || true)"
[[ "$URL" == "$LAST" ]] && exit 0

if [[ -f /etc/alpha-spy/backup.env ]]; then
  set -a
  # shellcheck disable=SC1091
  source /etc/alpha-spy/backup.env
  set +a
fi
export RCLONE_CONFIG="${RCLONE_CONFIG:-/root/.config/rclone/rclone.conf}"
REMOTE="${ALPHA_SPY_BACKUP_REMOTE:-gdrive:SPY Trading Backups/$(hostname -s)}"

{
  echo "SPY trading overview page (basic auth user: spy)"
  echo "$URL"
  echo "updated $(date -Is)"
} >/tmp/dashboard-url.txt

if rclone copyto /tmp/dashboard-url.txt "$REMOTE/dashboard-url.txt" \
    --retries 3 --low-level-retries 10 --timeout 2m >/dev/null 2>&1; then
  mkdir -p "$(dirname "$STATE")"
  echo "$URL" >"$STATE"
  printf '%s tunnel URL published: %s\n' "$(date -Is)" "$URL" >>/var/log/spy-watchdog.log
fi
exit 0
