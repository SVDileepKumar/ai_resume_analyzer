#!/usr/bin/env bash
# One-shot local setup: Python venv + requirements, Ollama + models (optional), .env.
# Run from repo root: bash scripts/setup-local.sh
# First run needs internet for pip, Ollama install, and ollama pull. The web UI then works offline.

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

info() { printf '\033[0;34m%s\033[0m\n' "$*"; }
ok() { printf '\033[0;32m%s\033[0m\n' "$*"; }
warn() { printf '\033[0;33m%s\033[0m\n' "$*"; }
err() { printf '\033[0;31m%s\033[0m\n' "$*" >&2; }

if [[ -n "${HTTP_PROXY:-}" || -n "${HTTPS_PROXY:-}" ]]; then
  warn "HTTP(S)_PROXY is set. For local Ollama, ensure NO_PROXY includes localhost,127.0.0.1,::1"
  warn "Example: export NO_PROXY=localhost,127.0.0.1,::1"
fi

# --- Python ---
if ! command -v python3 >/dev/null 2>&1; then
  err "python3 not found. Install Python 3.12+ and retry."
  exit 1
fi

PYVER="$(python3 -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")')"
if ! python3 -c 'import sys; sys.exit(0 if sys.version_info >= (3, 10) else 1)'; then
  err "Python 3.10+ required (found $PYVER)."
  exit 1
fi

VENV="${ROOT}/.venv"
if [[ ! -d "$VENV" ]]; then
  info "Creating virtual environment at .venv …"
  python3 -m venv "$VENV"
fi
# shellcheck source=/dev/null
source "$VENV/bin/activate"

info "Installing Python dependencies …"
pip install -U pip >/dev/null
pip install -r requirements.txt

if [[ ! -f "${ROOT}/.env" ]]; then
  info "Creating .env from .env.example …"
  cp .env.example .env
  ok "Created .env — edit if you need different models or limits."
fi

# --- Ollama models (read from .env or .env.example) ---
ENV_SRC="${ROOT}/.env"
[[ -f "$ENV_SRC" ]] || ENV_SRC="${ROOT}/.env.example"
CHAT_MODEL="$(grep -E '^OLLAMA_CHAT_MODEL=' "$ENV_SRC" | head -1 | cut -d= -f2- | tr -d '\r' | xargs)"
EMBED_MODEL="$(grep -E '^OLLAMA_EMBED_MODEL=' "$ENV_SRC" | head -1 | cut -d= -f2- | tr -d '\r' | xargs)"
CHAT_MODEL="${CHAT_MODEL:-qwen3.5:2b}"
EMBED_MODEL="${EMBED_MODEL:-qwen3-embedding:0.6b}"

install_ollama_macos() {
  if command -v brew >/dev/null 2>&1; then
    info "Installing Ollama via Homebrew …"
    brew install ollama
  else
    warn "Homebrew not found. Install Ollama from https://ollama.com/download (macOS), then re-run this script."
    return 1
  fi
}

install_ollama_linux() {
  info "Installing Ollama (official install script from ollama.com) …"
  warn "Review https://ollama.com/install.sh if your organization restricts curl|sh installers."
  curl -fsSL https://ollama.com/install.sh | sh
}

if ! command -v ollama >/dev/null 2>&1; then
  OS="$(uname -s)"
  case "$OS" in
    Darwin) install_ollama_macos || true ;;
    Linux) install_ollama_linux ;;
    *)
      err "Unsupported OS: $OS. Install Ollama manually from https://ollama.com/download"
      ;;
  esac
fi

if command -v ollama >/dev/null 2>&1; then
  info "Pulling Ollama models (chat: $CHAT_MODEL, embed: $EMBED_MODEL) …"
  ollama pull "$CHAT_MODEL"
  ollama pull "$EMBED_MODEL"
  ok "Ollama models ready."
else
  warn "Ollama CLI not available — skip model pull. Install Ollama and run: ollama pull $CHAT_MODEL && ollama pull $EMBED_MODEL"
fi

# --- Frontend assets (Tailwind CSS + vendored JS — required for the UI to render) ---
if [[ ! -f "${ROOT}/app/static/css/tailwind.css" || ! -f "${ROOT}/app/static/css/fonts.css" ]]; then
  warn "Missing frontend assets (tailwind.css / fonts.css)."
  if command -v npm >/dev/null 2>&1; then
    info "Building Tailwind CSS …"
    npm --prefix "$ROOT" install
    npm --prefix "$ROOT" run build:css
    ok "Frontend assets built."
  else
    err "npm not found. Install Node.js from https://nodejs.org, then run: npm install && npm run build:css"
    err "The app will not render correctly without these assets."
  fi
else
  ok "Frontend assets already present."
fi

ok "Setup complete."

echo ""
info "Next steps:"
echo "  1. Start Ollama (if not running as a service):  ollama serve"
echo "  2. Activate venv and run the app:"
echo "       source .venv/bin/activate"
echo "       uvicorn app.main:app --host 0.0.0.0 --port 8899 --timeout-keep-alive 1800"
echo "  3. Open http://localhost:8899"
echo ""
info "After template changes, rebuild Tailwind:  npm install && npm run build:css"
