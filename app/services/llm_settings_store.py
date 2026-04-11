"""Optional operator JSON file for OpenAI-compatible API key and base URL."""

from __future__ import annotations

import json
import logging
import os
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

_settings_mtime: float | None = None
_cached: dict[str, Any] | None = None


def _path() -> Path | None:
    from app.config import LLM_USER_SETTINGS_PATH
    if not LLM_USER_SETTINGS_PATH:
        return None
    return Path(LLM_USER_SETTINGS_PATH).expanduser()


def load_operator_settings() -> dict[str, str]:
    """Return {openai_api_key, openai_base_url} from file if present."""
    global _settings_mtime, _cached
    p = _path()
    if p is None or not p.is_file():
        _cached = {}
        return {}
    try:
        mtime = p.stat().st_mtime
        if _cached is not None and _settings_mtime == mtime:
            return dict(_cached)
        with open(p, encoding="utf-8") as f:
            raw = json.load(f)
        if not isinstance(raw, dict):
            return {}
        out = {
            "openai_api_key": str(raw.get("openai_api_key", "") or "").strip(),
            "openai_base_url": str(raw.get("openai_base_url", "") or "").strip(),
        }
        _cached = out
        _settings_mtime = mtime
        return dict(out)
    except OSError as e:
        logger.warning("Could not read LLM user settings: %s", e)
        return {}


def save_operator_settings(api_key: str, base_url: str) -> None:
    """Write settings file (directory created if needed)."""
    p = _path()
    if p is None:
        raise ValueError("LLM_USER_SETTINGS_PATH is not configured")
    p.parent.mkdir(parents=True, exist_ok=True)
    data = {
        "openai_api_key": api_key.strip(),
        "openai_base_url": base_url.strip(),
    }
    tmp = p.with_suffix(".tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)
        f.write("\n")
    os.replace(tmp, p)
    global _settings_mtime, _cached
    _cached = dict(data)
    _settings_mtime = p.stat().st_mtime


def invalidate_cache() -> None:
    global _settings_mtime, _cached
    _settings_mtime = None
    _cached = None
