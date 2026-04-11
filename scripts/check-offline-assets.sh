#!/usr/bin/env bash
# Verify local CSS/JS assets referenced by base.html exist (blocking load, no CDN).
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
BASE="${ROOT}/app/templates/base.html"
for needle in "/static/css/tailwind.css" "/static/css/fonts.css" "/static/css/app.css" \
  "/static/vendor/htmx-2.0.4.min.js" "/static/vendor/alpine-3.14.3.min.js" "/static/vendor/chart-4.4.7.umd.min.js"; do
  if ! grep -qF "$needle" "$BASE"; then
    echo "ERROR: $BASE must reference fallback path: $needle"
    exit 1
  fi
done
for f in tailwind.css fonts.css; do
  if [[ ! -f "${ROOT}/app/static/css/$f" ]]; then
    echo "ERROR: Missing app/static/css/$f (run npm run build:css for tailwind.css)."
    exit 1
  fi
done
for f in htmx-2.0.4.min.js alpine-3.14.3.min.js chart-4.4.7.umd.min.js; do
  if [[ ! -f "${ROOT}/app/static/vendor/$f" ]]; then
    echo "ERROR: Missing app/static/vendor/$f"
    exit 1
  fi
done
echo "OK: UI static assets present (local blocking load in base.html)."
