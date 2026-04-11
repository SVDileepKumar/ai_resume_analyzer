"""Load model catalogs, validate chat model ids, RAM detection, Ollama tag listing."""

from __future__ import annotations

import json
import logging
import re
from pathlib import Path
from typing import Any, AsyncIterator

import httpx

from app.config import (
    LLM_ALLOW_ANY_OLLAMA_MODEL,
    LLM_ASSUMED_RAM_GB,
    LLM_DETECT_RAM,
    LLM_RAM_SAFETY_FACTOR,
    LLM_BACKEND,
    OLLAMA_BASE_URL,
    OLLAMA_CHAT_MODEL,
    OPENAI_CHAT_MODEL,
)

logger = logging.getLogger(__name__)

_DATA_DIR = Path(__file__).resolve().parents[1] / "data"


def _load_json(name: str) -> dict[str, Any]:
    path = _DATA_DIR / name
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def get_ollama_catalog_entries() -> list[dict[str, Any]]:
    data = _load_json("ollama_models.json")
    return list(data.get("models") or [])


def get_openai_catalog_entries() -> list[dict[str, Any]]:
    data = _load_json("openai_models.json")
    return list(data.get("models") or [])


def ollama_catalog_ids() -> set[str]:
    return {str(m["id"]).strip() for m in get_ollama_catalog_entries() if m.get("id")}


def openai_catalog_ids() -> set[str]:
    return {str(m["id"]).strip() for m in get_openai_catalog_entries() if m.get("id")}


def detect_ram_gb() -> float | None:
    """Return available RAM in GB for *this server process* (heuristic)."""
    from app.config import LLM_ASSUMED_RAM_GB

    if LLM_ASSUMED_RAM_GB is not None:
        return float(LLM_ASSUMED_RAM_GB)
    if not LLM_DETECT_RAM:
        return None
    try:
        import psutil  # type: ignore[import-untyped]

        return round(psutil.virtual_memory().total / (1024**3), 2)
    except Exception:
        logger.debug("RAM detection unavailable", exc_info=True)
        return None


def fetch_ollama_installed_tags() -> set[str]:
    """Return model names/tags reported by Ollama (empty if unreachable)."""
    base = OLLAMA_BASE_URL.replace("/v1", "").rstrip("/")
    try:
        resp = httpx.get(f"{base}/api/tags", timeout=5)
        resp.raise_for_status()
        data = resp.json()
    except Exception as e:
        logger.debug("Ollama /api/tags failed: %s", e)
        return set()
    out: set[str] = set()
    for item in data.get("models") or []:
        name = item.get("name") or item.get("model")
        if isinstance(name, str) and name.strip():
            out.add(name.strip())
    return out


def _pull_event_payload(model: str, data: dict[str, Any]) -> dict[str, Any]:
    """Normalize Ollama pull stream events for the UI progress bar."""
    status = str(data.get("status") or "pulling")
    total_raw = data.get("total")
    completed_raw = data.get("completed")
    try:
        total = int(total_raw) if total_raw is not None else None
    except (TypeError, ValueError):
        total = None
    try:
        completed = int(completed_raw) if completed_raw is not None else None
    except (TypeError, ValueError):
        completed = None
    percent: int | None = None
    if total and completed is not None and total > 0:
        percent = max(0, min(100, int((completed / total) * 100)))
    done = bool(data.get("done")) or status.lower() in {
        "success",
        "verifying sha256 digest",
        "writing manifest",
        "removing any unused layers",
    }
    return {
        "model": model,
        "status": status,
        "digest": data.get("digest"),
        "total": total,
        "completed": completed,
        "percent": 100 if done else percent,
        "done": done,
    }


async def stream_ollama_model_pull(model: str) -> AsyncIterator[dict[str, Any]]:
    """Yield normalized progress events while ensuring an Ollama model is present."""
    installed = fetch_ollama_installed_tags()
    if model in installed:
        yield {
            "model": model,
            "status": "already installed",
            "percent": 100,
            "done": True,
            "total": None,
            "completed": None,
            "digest": None,
        }
        return

    base = OLLAMA_BASE_URL.replace("/v1", "").rstrip("/")
    timeout = httpx.Timeout(connect=10.0, read=None, write=120.0, pool=30.0)
    async with httpx.AsyncClient(timeout=timeout) as client:
        async with client.stream("POST", f"{base}/api/pull", json={"name": model, "stream": True}) as resp:
            resp.raise_for_status()
            emitted_done = False
            async for line in resp.aiter_lines():
                line = (line or "").strip()
                if not line:
                    continue
                try:
                    data = json.loads(line)
                except json.JSONDecodeError:
                    logger.debug("Skipping non-JSON Ollama pull line: %s", line[:200])
                    continue
                event = _pull_event_payload(model, data)
                emitted_done = emitted_done or bool(event.get("done"))
                yield event
            if not emitted_done:
                yield {
                    "model": model,
                    "status": "ready",
                    "percent": 100,
                    "done": True,
                    "total": None,
                    "completed": None,
                    "digest": None,
                }


def _parse_min_ram(entry: dict[str, Any]) -> float:
    v = entry.get("min_ram_gb", 8)
    try:
        return float(v)
    except (TypeError, ValueError):
        return 8.0


def _parse_ollama_size(entry: dict[str, Any]) -> float:
    """Return actual Ollama download/runtime size in GB. Falls back to min_ram_gb."""
    v = entry.get("ollama_size_gb")
    if v is not None:
        try:
            return float(v)
        except (TypeError, ValueError):
            pass
    return _parse_min_ram(entry)


def _parse_tau2(entry: dict[str, Any]) -> float:
    """Return TAU2 agent/tool-use benchmark score (0–100). Primary quality signal.

    TAU2 directly measures structured tool-calling ability — the most relevant
    benchmark for ATS resume scoring which relies on JSON function calling.
    Falls back to 0.0 when not available.
    """
    benchmarks = entry.get("benchmarks") or {}
    v = benchmarks.get("tau2")
    if v is None:
        return 0.0
    try:
        return float(v)
    except (TypeError, ValueError):
        return 0.0


def _catalog_deprecated(entry: dict[str, Any]) -> bool:
    """True if row is legacy; JSON flag OR known legacy id prefixes (stale catalogs)."""
    if entry.get("deprecated") is True:
        return True
    eid = str(entry.get("id", "")).lower()
    if eid.startswith("gemma3") or eid.startswith("gemma2"):
        return True
    return False


def _catalog_order_index(entry: dict[str, Any], catalog: list[dict[str, Any]]) -> int:
    """Stable order from JSON listing (lower = earlier row = preferred tie-break)."""
    eid = str(entry.get("id", ""))
    for i, row in enumerate(catalog):
        if str(row.get("id", "")) == eid:
            return i
    return 10**6


# Embed model overhead reserved when checking if a chat model fits in RAM.
# Assumes the default embedding model (qwen3-embedding:0.6b ≈ 0.64 GB) is always
# co-loaded. This prevents recommending a chat model that leaves no headroom for it.
_EMBED_MODEL_OVERHEAD_GB: float = 0.7


def _model_fits_with_embed(entry: dict[str, Any], usable_ram_gb: float) -> bool:
    """Return True if the model fits in usable RAM with embed model overhead reserved.

    Uses actual Ollama size (ollama_size_gb) rather than min_ram_gb to catch
    models like gemma4:e2b (7.2GB actual) which nominally claim min_ram_gb=8
    but leave no room for the embedding model co-loaded alongside.
    """
    model_size = _parse_ollama_size(entry)
    return (model_size + _EMBED_MODEL_OVERHEAD_GB) <= usable_ram_gb


def recommend_ollama_model(installed: set[str], ram_gb: float | None) -> str | None:
    """Pick best catalog model for available RAM.

    Ranking priority (descending):
    1. Not deprecated
    2. Fits in RAM WITH embed model overhead (uses actual ollama_size_gb, not min_ram_gb)
    3. TAU2 score (highest first) — best proxy for structured JSON/tool-use quality
    4. Catalog order (tie-break, preserves intentional row ordering)
    5. Already installed (prefer pulling nothing over pulling something equivalent)
    """
    entries = get_ollama_catalog_entries()
    usable_ram: float | None = None
    if ram_gb is not None:
        usable_ram = max(0.5, ram_gb * LLM_RAM_SAFETY_FACTOR)

    all_ids = [e for e in entries if e.get("id")]
    non_deprecated = [e for e in all_ids if not _catalog_deprecated(e)]
    candidates = non_deprecated if non_deprecated else all_ids
    if not candidates:
        return OLLAMA_CHAT_MODEL or None

    # Split into fits-with-embed and overflow — prefer models that actually leave room
    # for the embedding model rather than just using min_ram_gb as the floor.
    if usable_ram is not None:
        fits = [e for e in candidates if _model_fits_with_embed(e, usable_ram)]
        # Fall back to min_ram_gb check if ollama_size_gb data is absent for all entries
        if not fits:
            fits = [e for e in candidates if _parse_min_ram(e) <= usable_ram]
        if fits:
            candidates = fits
        # If still nothing fits, keep full list (better than returning None)

    candidates.sort(
        key=lambda e: (
            # Primary: TAU2 score descending (best structured-task quality first)
            -_parse_tau2(e),
            # Tie-break: catalog row order (intentional priority ordering in JSON)
            _catalog_order_index(e, entries),
            # Final tie-break: prefer already-installed (avoid unnecessary download)
            1 if str(e.get("id", "")) not in installed else 0,
        ),
    )
    return str(candidates[0]["id"])


def best_catalog_model_in_tier(
    entries: list[dict[str, Any]],
    installed: set[str],
    ram_gb: float | None,
    tier: str,
    *,
    apply_ram_heuristic: bool = True,
) -> str | None:
    """Best catalog id within a single tier (fast / balanced / quality), for UI quick picks.

    Uses actual ollama_size_gb + embed overhead for RAM fitting (same logic as recommend).
    Falls back to min_ram_gb if ollama_size_gb is absent. Within a tier, TAU2 score wins.
    """
    usable_ram: float | None = None
    if apply_ram_heuristic and ram_gb is not None:
        usable_ram = max(0.5, ram_gb * LLM_RAM_SAFETY_FACTOR)

    tier_entries = [
        e for e in entries
        if e.get("id") and str(e.get("tier", "balanced")) == tier
    ]
    if not tier_entries:
        return None

    nd = [e for e in tier_entries if not _catalog_deprecated(e)]
    pool = nd if nd else tier_entries

    if usable_ram is not None:
        fits = [e for e in pool if _model_fits_with_embed(e, usable_ram)]
        if not fits:
            # Secondary: check min_ram_gb only (no size data in catalog)
            fits = [e for e in pool if _parse_min_ram(e) <= usable_ram]
        # If still nothing fits this tier for this RAM, return None so the UI
        # doesn't suggest a model the user cannot run.
        if fits:
            pool = fits
        else:
            return None

    pool.sort(
        key=lambda e: (
            -_parse_tau2(e),
            _catalog_order_index(e, entries),
            1 if str(e.get("id", "")) not in installed else 0,
        ),
    )
    return str(pool[0]["id"])


def validate_ollama_chat_model(model: str) -> tuple[bool, str]:
    """Return (ok, normalized_model_id)."""
    m = (model or "").strip()
    if not m:
        return False, OLLAMA_CHAT_MODEL
    if LLM_ALLOW_ANY_OLLAMA_MODEL:
        return True, m
    if m == OLLAMA_CHAT_MODEL.strip():
        return True, m
    if m in ollama_catalog_ids():
        return True, m
    return False, m


def validate_openai_chat_model(model: str) -> tuple[bool, str]:
    m = (model or "").strip()
    if not m:
        return False, OPENAI_CHAT_MODEL
    if m == OPENAI_CHAT_MODEL.strip():
        return True, m
    if m in openai_catalog_ids():
        return True, m
    return False, m


def validate_openai_base_url(url: str) -> bool:
    """Basic SSRF guard: http(s) only; optional block of obvious private hosts when strict."""
    u = (url or "").strip()
    if not u:
        return False
    if not re.match(r"^https?://", u, re.I):
        return False
    from urllib.parse import urlparse

    parsed = urlparse(u)
    host = (parsed.hostname or "").lower()
    if host in ("localhost", "127.0.0.1", "::1"):
        return True
    # Allow public hostnames; block literal private IPv4 in hostname
    if re.match(r"^(10\.|172\.(1[6-9]|2[0-9]|3[01])\.|192\.168\.)", host):
        return False
    return True


def build_llm_options_payload() -> dict[str, Any]:
    """JSON-serializable body for GET /api/llm/options."""
    ram = detect_ram_gb()
    installed = fetch_ollama_installed_tags() if LLM_BACKEND == "ollama" else set()
    rec: str | None = None
    tier_picks: dict[str, str | None] = {"fast": None, "balanced": None, "quality": None}
    if LLM_BACKEND == "ollama":
        rec = recommend_ollama_model(installed, ram)
        o_entries = get_ollama_catalog_entries()
        tier_picks["fast"] = best_catalog_model_in_tier(o_entries, installed, ram, "fast")
        tier_picks["balanced"] = best_catalog_model_in_tier(o_entries, installed, ram, "balanced")
        tier_picks["quality"] = best_catalog_model_in_tier(o_entries, installed, ram, "quality")
    else:
        rec = OPENAI_CHAT_MODEL
        oa_entries = get_openai_catalog_entries()
        tier_picks["fast"] = best_catalog_model_in_tier(
            oa_entries, set(), None, "fast", apply_ram_heuristic=False
        )
        tier_picks["balanced"] = best_catalog_model_in_tier(
            oa_entries, set(), None, "balanced", apply_ram_heuristic=False
        )
        tier_picks["quality"] = best_catalog_model_in_tier(
            oa_entries, set(), None, "quality", apply_ram_heuristic=False
        )
    effective_ram = None
    if ram is not None:
        effective_ram = round(max(0.5, ram * LLM_RAM_SAFETY_FACTOR), 2)

    ollama_models: list[dict[str, Any]] = []
    for e in get_ollama_catalog_entries():
        mid = str(e.get("id", ""))
        ollama_models.append(
            {
                "id": mid,
                "label": e.get("label", mid),
                "min_ram_gb": _parse_min_ram(e),
                "ollama_size_gb": _parse_ollama_size(e),
                "tier": e.get("tier", "balanced"),
                "installed": mid in installed if LLM_BACKEND == "ollama" else False,
                "deprecated": _catalog_deprecated(e),
                "tau2_score": _parse_tau2(e),
                "json_reliability_note": e.get("json_reliability_note", ""),
                "description": e.get("description", ""),
                "best_for": e.get("best_for", ""),
            },
        )

    openai_models = [
        {
            "id": m["id"],
            "label": m.get("label", m["id"]),
            "tier": m.get("tier", "balanced"),
            "notes": m.get("notes", ""),
            "description": m.get("description", ""),
            "best_for": m.get("best_for", ""),
        }
        for m in get_openai_catalog_entries()
        if m.get("id")
    ]

    from app.config import OPENAI_API_KEY
    from app.services.llm_settings_store import load_operator_settings

    op = load_operator_settings()
    hosted_configured = bool(OPENAI_API_KEY) or bool(op.get("openai_api_key"))

    from app.config import ENABLE_HOSTED_LLM_UI

    return {
        "backend": LLM_BACKEND,
        "hosted_configured": hosted_configured,
        "detected_ram_gb": ram,
        "effective_recommended_ram_gb": effective_ram,
        "recommended_model": rec,
        "default_chat_model": OLLAMA_CHAT_MODEL if LLM_BACKEND == "ollama" else OPENAI_CHAT_MODEL,
        "tier_picks": tier_picks,
        "models": ollama_models if LLM_BACKEND == "ollama" else openai_models,
        "download_once_note": (
            "Each local model is downloaded once if it is not already on disk; later runs reuse it."
            if LLM_BACKEND == "ollama"
            else ""
        ),
        "recommendation_reason": (
            f"Best overall pick for ~{effective_ram} GB usable RAM on this server (download once if missing)."
            if LLM_BACKEND == "ollama" and effective_ram is not None
            else "Best overall pick from the catalog for this server's RAM (download once if missing)."
            if LLM_BACKEND == "ollama"
            else "Recommended default hosted model for balanced cost and reliability."
        ),
        "ram_disclaimer": (
            "Recommendation uses this server’s RAM, not your laptop, when the app runs remotely."
        ),
        "enable_hosted_llm_ui": ENABLE_HOSTED_LLM_UI,
    }
