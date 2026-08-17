#!/usr/bin/env bash
set -euo pipefail

# Deploy ONLY the copy-based SPY Command overview layer.
# This script intentionally does not touch src/beta_spy/. Changes there still
# require: sync /opt/beta-spy/src -> pip install . -> restart beta-spy.

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
WEB_ROOT="${SPY_OVERVIEW_WEB_ROOT:-/var/www/spy-overview}"
STATUS_BIN="${SPY_OVERVIEW_STATUS_BIN:-/usr/local/sbin/spy-overview-status}"

if [[ ${EUID:-$(id -u)} -ne 0 ]]; then
  echo "deploy-spy-overview.sh must run as root" >&2
  exit 1
fi

install -d -m 0755 "$WEB_ROOT"
install -m 0644 "$ROOT/config/overview-index.html" "$WEB_ROOT/index.html"
install -m 0755 "$ROOT/scripts/spy-overview-status.py" "$STATUS_BIN"

python3 -m py_compile "$STATUS_BIN"

if [[ -f "$ROOT/systemd/spy-overview-status.service" ]]; then
  install -m 0644 "$ROOT/systemd/spy-overview-status.service" /etc/systemd/system/spy-overview-status.service
fi
if [[ -f "$ROOT/systemd/spy-overview-status.timer" ]]; then
  install -m 0644 "$ROOT/systemd/spy-overview-status.timer" /etc/systemd/system/spy-overview-status.timer
fi

systemctl daemon-reload
systemctl enable --now spy-overview-status.timer
systemctl start spy-overview-status.service

if command -v nginx >/dev/null 2>&1; then
  nginx -t
fi

echo "SPY Command overview deployed"
echo "  UI:     $WEB_ROOT/index.html"
echo "  status: $WEB_ROOT/status.json"
echo "  timer:  $(systemctl is-active spy-overview-status.timer || true)"
