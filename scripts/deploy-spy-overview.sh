#!/usr/bin/env bash
set -euo pipefail

# Deploy ONLY the copy-based SPY Command overview / Delta layer.
# This script intentionally does not touch src/beta_spy/. Changes there still
# require: sync /opt/beta-spy/src -> pip install . -> restart beta-spy.

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
WEB_ROOT="${SPY_OVERVIEW_WEB_ROOT:-/var/www/spy-overview}"
STATUS_BIN="${SPY_OVERVIEW_STATUS_BIN:-/usr/local/sbin/spy-overview-status}"
BASE_BIN="${SPY_OVERVIEW_BASE_BIN:-/usr/local/lib/spy-overview-base.py}"

if [[ ${EUID:-$(id -u)} -ne 0 ]]; then
  echo "deploy-spy-overview.sh must run as root" >&2
  exit 1
fi

install -d -m 0755 "$WEB_ROOT"
install -d -m 0755 "$(dirname "$BASE_BIN")"
install -d -m 0755 /var/lib/spy-overview
install -m 0644 "$ROOT/config/overview-index.html" "$WEB_ROOT/index.html"

# Keep the existing collector as a stable base module and install Delta as the
# executable publisher. Delta is read-only and never feeds data back into either
# trading engine.
install -m 0644 "$ROOT/scripts/spy-overview-status.py" "$BASE_BIN"
install -m 0755 "$ROOT/scripts/spy-overview-delta.py" "$STATUS_BIN"

python3 -m py_compile "$BASE_BIN" "$STATUS_BIN"

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

echo "SPY Command / Delta overview deployed"
echo "  UI:       $WEB_ROOT/index.html"
echo "  status:   $WEB_ROOT/status.json"
echo "  ChatGPT:  $WEB_ROOT/chatgpt.json"
echo "  Gamma:    ${GAMMA_CATALYSTS_PATH:-/var/lib/spy-overview/gamma-catalysts.json} (optional)"
echo "  timer:    $(systemctl is-active spy-overview-status.timer || true)"
