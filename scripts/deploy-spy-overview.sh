#!/usr/bin/env bash
set -euo pipefail

# Deploy ONLY the copy-based SPY Command overview / Delta layer.
# This script intentionally does not touch src/beta_spy/. Changes there still
# require: sync /opt/beta-spy/src -> pip install . -> restart beta-spy.

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
WEB_ROOT="${SPY_OVERVIEW_WEB_ROOT:-/var/www/spy-overview}"
STATUS_BIN="${SPY_OVERVIEW_STATUS_BIN:-/usr/local/sbin/spy-overview-status}"
BASE_BIN="${SPY_OVERVIEW_BASE_BIN:-/usr/local/lib/spy-overview-base.py}"
DELTA_V1_BIN="${SPY_OVERVIEW_DELTA_V1_BIN:-/usr/local/lib/spy-overview-delta-v1.py}"
ENV_FILE="${SPY_OVERVIEW_ENV_FILE:-/etc/spy-overview.env}"
SLUG_FILE="${SPY_OVERVIEW_CHATGPT_SLUG_FILE:-/etc/spy-overview-chatgpt-slug}"

if [[ ${EUID:-$(id -u)} -ne 0 ]]; then
  echo "deploy-spy-overview.sh must run as root" >&2
  exit 1
fi

install -d -m 0755 "$WEB_ROOT"
install -d -m 0755 "$WEB_ROOT/chatgpt"
install -d -m 0755 "$(dirname "$BASE_BIN")"
install -d -m 0755 /var/lib/spy-overview
install -m 0644 "$ROOT/config/overview-index.html" "$WEB_ROOT/index.html"

# Generate the public-but-unguessable filename once and preserve it across
# deployments.  The document itself is scrubbed and read-only; the slug keeps
# casual scanners from discovering the strategy feed while dashboard/API auth
# remains unchanged.
if [[ -s "$SLUG_FILE" ]]; then
  CHATGPT_SLUG="$(tr -cd 'A-Za-z0-9_-' < "$SLUG_FILE")"
else
  CHATGPT_SLUG="$(python3 - <<'PY'
import secrets
print(secrets.token_urlsafe(24))
PY
)"
  umask 077
  printf '%s\n' "$CHATGPT_SLUG" > "$SLUG_FILE"
  chmod 0600 "$SLUG_FILE"
fi
if [[ -z "$CHATGPT_SLUG" ]]; then
  echo "failed to establish ChatGPT bridge slug" >&2
  exit 1
fi
CHATGPT_PATH="$WEB_ROOT/chatgpt/$CHATGPT_SLUG.json"

# Preserve every existing overview setting/secret while ensuring Delta sees the
# generated output path through the systemd EnvironmentFile.
python3 - "$ENV_FILE" "$CHATGPT_PATH" <<'PY'
from pathlib import Path
import sys
path = Path(sys.argv[1])
value = sys.argv[2]
lines = path.read_text(encoding="utf-8").splitlines() if path.exists() else []
lines = [line for line in lines if not line.startswith("CHATGPT_STATUS_PATH=")]
lines.append(f"CHATGPT_STATUS_PATH={value}")
path.parent.mkdir(parents=True, exist_ok=True)
path.write_text("\n".join(lines) + "\n", encoding="utf-8")
path.chmod(0o600)
PY

# Keep each layer separate so rollback is trivial.  The old overview collector
# remains the base, Delta v1 remains available for compatibility, and the
# horizon-aware v2 wrapper is the executable publisher.
install -m 0644 "$ROOT/scripts/spy-overview-status.py" "$BASE_BIN"
install -m 0644 "$ROOT/scripts/spy-overview-delta.py" "$DELTA_V1_BIN"
install -m 0755 "$ROOT/scripts/spy-overview-delta-v2.py" "$STATUS_BIN"

python3 -m py_compile "$BASE_BIN" "$DELTA_V1_BIN" "$STATUS_BIN"

if [[ -f "$ROOT/systemd/spy-overview-status.service" ]]; then
  install -m 0644 "$ROOT/systemd/spy-overview-status.service" /etc/systemd/system/spy-overview-status.service
fi
if [[ -f "$ROOT/systemd/spy-overview-status.timer" ]]; then
  install -m 0644 "$ROOT/systemd/spy-overview-status.timer" /etc/systemd/system/spy-overview-status.timer
fi

# If this machine already has the source-controlled SPY nginx config installed,
# update only that discovered file.  Do not invent a second server block on an
# unfamiliar host.  A failed nginx validation is rolled back before exiting.
NGINX_UPDATED=0
if command -v nginx >/dev/null 2>&1; then
  NGINX_TARGET="${SPY_NGINX_CONFIG_TARGET:-}"
  if [[ -z "$NGINX_TARGET" ]]; then
    while IFS= read -r candidate; do
      [[ -n "$candidate" ]] || continue
      NGINX_TARGET="$(readlink -f "$candidate")"
      break
    done < <(
      grep -rlF '# SPY trading external access' \
        /etc/nginx/sites-enabled /etc/nginx/sites-available /etc/nginx/conf.d \
        2>/dev/null || true
    )
  fi
  if [[ -n "$NGINX_TARGET" && -f "$NGINX_TARGET" ]]; then
    BACKUP="${NGINX_TARGET}.pre-delta"
    cp -a "$NGINX_TARGET" "$BACKUP"
    install -m 0644 "$ROOT/config/nginx-spy.conf" "$NGINX_TARGET"
    if nginx -t; then
      systemctl reload nginx
      NGINX_UPDATED=1
      rm -f "$BACKUP"
    else
      cp -a "$BACKUP" "$NGINX_TARGET"
      rm -f "$BACKUP"
      nginx -t || true
      echo "nginx validation failed; restored prior SPY config" >&2
      exit 1
    fi
  else
    nginx -t
    echo "warning: existing SPY nginx config was not discovered; /chatgpt/ remains unavailable until config/nginx-spy.conf is installed" >&2
  fi
fi

systemctl daemon-reload
systemctl enable --now spy-overview-status.timer
systemctl start spy-overview-status.service

# Produce the document now rather than waiting for the next minute boundary.
if [[ ! -s "$CHATGPT_PATH" ]]; then
  echo "Delta did not produce $CHATGPT_PATH" >&2
  exit 1
fi

printf 'SPY Command / Delta overview deployed\n'
printf '  UI:       %s/index.html\n' "$WEB_ROOT"
printf '  status:   %s/status.json\n' "$WEB_ROOT"
printf '  ChatGPT:  %s\n' "$CHATGPT_PATH"
printf '  Gamma:    %s (optional)\n' "${GAMMA_CATALYSTS_PATH:-/var/lib/spy-overview/gamma-catalysts.json}"
printf '  nginx:    %s\n' "$([[ $NGINX_UPDATED -eq 1 ]] && echo updated || echo unchanged)"
printf '  timer:    %s\n' "$(systemctl is-active spy-overview-status.timer || true)"
