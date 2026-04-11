#!/usr/bin/env bash
# Re-download vendored JS and font files (pinned versions). Run from repo root when upgrading versions.
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
V="$ROOT/app/static/vendor"
F="$ROOT/app/static/fonts"
mkdir -p "$V" "$F"

curl -fsSL "https://cdn.jsdelivr.net/npm/htmx.org@2.0.4/dist/htmx.min.js" -o "$V/htmx-2.0.4.min.js"
curl -fsSL "https://cdn.jsdelivr.net/npm/alpinejs@3.14.3/dist/cdn.min.js" -o "$V/alpine-3.14.3.min.js"
curl -fsSL "https://cdn.jsdelivr.net/npm/chart.js@4.4.7/dist/chart.umd.min.js" -o "$V/chart-4.4.7.umd.min.js"

for w in 300 400 500 600 700 800 900; do
  curl -fsSL "https://cdn.jsdelivr.net/npm/@fontsource/inter@5.0.20/files/inter-latin-${w}-normal.woff2" -o "$F/inter-latin-${w}-normal.woff2"
done
for w in 400 600 700; do
  curl -fsSL "https://cdn.jsdelivr.net/npm/@fontsource/jetbrains-mono@5.0.20/files/jetbrains-mono-latin-${w}-normal.woff2" -o "$F/jetbrains-mono-latin-${w}-normal.woff2"
done

echo "Done. Re-run: npm run build:css"
