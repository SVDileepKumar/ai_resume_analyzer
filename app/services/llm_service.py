"""Core LLM service: local Ollama or hosted OpenAI-compatible APIs.

**Ollama:** native ``/api/chat`` with ``format: json`` — no API key.

**OpenAI-compatible:** uses the official ``openai`` Python SDK (``AsyncOpenAI``) with
``chat.completions`` and ``response_format`` JSON mode. Set ``LLM_BACKEND=openai`` and
``OPENAI_API_KEY``; ``OPENAI_BASE_URL`` targets OpenAI, Azure OpenAI, Groq, etc.
"""

from __future__ import annotations

import asyncio
import functools
import hashlib
import json
import logging
import re
from typing import Any

import httpx
from openai import APIConnectionError, APIStatusError, APITimeoutError, AsyncOpenAI, RateLimitError
from pydantic import BaseModel, Field, ValidationError, field_validator, model_validator

from app.config import (
    ENABLE_HOSTED_LLM_UI,
    JD_MAX_CHARS_LLM,
    LLM_BACKEND,
    LLM_ALLOW_ANY_OLLAMA_MODEL,
    OLLAMA_BASE_URL,
    OLLAMA_CHAT_MODEL,
    OLLAMA_JSON_MAX_RETRIES,
    OLLAMA_JSON_TEMPERATURE,
    OLLAMA_HTTP_MAX_RETRIES,
    OLLAMA_HTTP_RETRY_BACKOFF_SEC,
    OLLAMA_NUM_PREDICT,
    OLLAMA_NUM_PREDICT_JSON_HEAVY,
    OLLAMA_TIMEOUT,
    OPENAI_API_KEY,
    OPENAI_BASE_URL,
    OPENAI_CHAT_MODEL,
    RESUME_MAX_CHARS_LLM,
)
from app.services.llm_catalog import (
    detect_ram_gb,
    fetch_ollama_installed_tags,
    recommend_ollama_model,
    validate_openai_base_url,
    validate_ollama_chat_model,
    validate_openai_chat_model,
)
from app.services.llm_context import get_llm_context
from app.services.llm_settings_store import load_operator_settings

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Ollama health check
# ---------------------------------------------------------------------------

_OLLAMA_BASE = OLLAMA_BASE_URL.replace("/v1", "")


def is_ollama_running() -> bool:
    """Quick check if Ollama is reachable."""
    try:
        resp = httpx.get(_OLLAMA_BASE, timeout=3)
        return resp.status_code == 200
    except Exception:
        return False


_ollama_available: bool | None = None
_openai_async_clients: dict[str, AsyncOpenAI] = {}
_openai_client_lock = asyncio.Lock()

# Cached resolved chat model — avoids a sync HTTP /api/tags call on every LLM inference.
# Invalidated by reset_ollama_check() when model selection changes.
_resolved_chat_model: str | None = None

# Global semaphore for Ollama (single-GPU) — ensures only one inference runs at a time.
# asyncio.gather() inside _apply_enhanced_unified fires many coroutines; without this
# semaphore those coroutines all queue on the Ollama server simultaneously, inflating
# context-switch overhead. Serializing at the call level removes that overhead.
# Set LLM_OLLAMA_CALL_CONCURRENCY > 1 for multi-GPU setups.
_ollama_call_sem: asyncio.Semaphore | None = None


def _get_ollama_call_sem() -> asyncio.Semaphore:
    global _ollama_call_sem
    if _ollama_call_sem is None:
        from app.config import LLM_OLLAMA_CALL_CONCURRENCY
        _ollama_call_sem = asyncio.Semaphore(max(1, LLM_OLLAMA_CALL_CONCURRENCY))
    return _ollama_call_sem


def get_effective_openai_api_key() -> str:
    """Env wins; then request context; then operator settings file."""
    if OPENAI_API_KEY:
        return OPENAI_API_KEY
    ctx = get_llm_context()
    if ctx and ctx.openai_api_key and ctx.openai_api_key.strip():
        return ctx.openai_api_key.strip()
    return load_operator_settings().get("openai_api_key", "")


def get_effective_openai_base_url() -> str:
    ctx = get_llm_context()
    if ctx and ctx.openai_base_url and ctx.openai_base_url.strip():
        u = ctx.openai_base_url.strip()
        if validate_openai_base_url(u):
            return u.rstrip("/")
    op = load_operator_settings()
    if op.get("openai_base_url"):
        u = op["openai_base_url"].strip()
        if validate_openai_base_url(u):
            return u.rstrip("/")
    return OPENAI_BASE_URL.rstrip("/")


def get_effective_chat_model() -> str:
    """Resolved chat model id for the active backend.

    The result is cached after the first resolution to avoid a sync
    HTTP GET /api/tags call on every LLM inference (which was previously
    blocking the event loop 60–70 times per batch).

    Per-request overrides (via LLMRequestContext) bypass the cache so that
    the UI model-selector continues to work correctly.
    """
    global _resolved_chat_model

    ctx = get_llm_context()
    raw = ((ctx.chat_model or "").strip() if ctx else "") or None

    # Per-request override bypasses cache
    if raw:
        if LLM_BACKEND == "ollama":
            ok, norm = validate_ollama_chat_model(raw)
            if ok:
                return norm
            if LLM_ALLOW_ANY_OLLAMA_MODEL:
                return raw
            return OLLAMA_CHAT_MODEL
        ok, norm = validate_openai_chat_model(raw)
        if ok:
            return norm
        return OPENAI_CHAT_MODEL

    # Return cached value if available
    if _resolved_chat_model is not None:
        return _resolved_chat_model

    # First-time resolution (may make HTTP call to /api/tags for Ollama)
    if LLM_BACKEND == "ollama":
        installed = fetch_ollama_installed_tags()
        rec = recommend_ollama_model(installed, detect_ram_gb())
        if rec and rec in installed:
            resolved = rec
        elif OLLAMA_CHAT_MODEL in installed:
            resolved = OLLAMA_CHAT_MODEL
        elif installed:
            # Last-resort runtime fallback: choose an installed tag rather than a
            # catalog recommendation that Ollama will 404 because it is not pulled.
            resolved = sorted(installed)[0]
        else:
            resolved = OLLAMA_CHAT_MODEL
    else:
        resolved = OPENAI_CHAT_MODEL

    _resolved_chat_model = resolved
    return resolved


def ai_enabled() -> bool:
    """Check if AI features can run for this request (Ollama up or effective OpenAI key)."""
    if LLM_BACKEND == "openai":
        return bool(get_effective_openai_api_key())
    global _ollama_available
    if _ollama_available is None:
        _ollama_available = is_ollama_running()
    return _ollama_available


def request_llm_ready() -> bool:
    """True when this request may invoke the LLM (credentials + backend path).

    Same predicate as :func:`ai_enabled`; use this name in API handlers where
    request-scoped context (keys, model) matters for clarity vs
    :func:`server_has_ai_route` (dashboard “AI available” hints).
    """
    return ai_enabled()


def server_has_ai_route() -> bool:
    """Whether the UI should advertise AI features (dashboard, analyze page)."""
    if LLM_BACKEND == "openai":
        if bool(OPENAI_API_KEY) or bool(load_operator_settings().get("openai_api_key")):
            return True
        return bool(ENABLE_HOSTED_LLM_UI)
    global _ollama_available
    if _ollama_available is None:
        _ollama_available = is_ollama_running()
    return _ollama_available


def reset_ollama_check() -> None:
    """Reset cached Ollama availability, model selection, and OpenAI async clients."""
    global _ollama_available, _openai_async_clients, _resolved_chat_model
    _ollama_available = None
    _resolved_chat_model = None
    _openai_async_clients.clear()


async def _get_openai_async_client() -> AsyncOpenAI:
    """AsyncOpenAI for effective API key + base URL (cached per credential pair)."""
    key = get_effective_openai_api_key()
    base = get_effective_openai_base_url()
    cache_key = hashlib.sha256(f"{key}\n{base}".encode()).hexdigest()
    async with _openai_client_lock:
        client = _openai_async_clients.get(cache_key)
        if client is None:
            client = AsyncOpenAI(
                api_key=key or "invalid",
                base_url=base,
                timeout=OLLAMA_TIMEOUT,
            )
            if len(_openai_async_clients) > 16:
                _openai_async_clients.clear()
            _openai_async_clients[cache_key] = client
        return client


def _openai_retryable(exc: BaseException) -> bool:
    if isinstance(exc, (RateLimitError, APITimeoutError, APIConnectionError)):
        return True
    if isinstance(exc, APIStatusError):
        code = exc.status_code
        return code == 429 or (code is not None and code >= 500)
    return False


# ---------------------------------------------------------------------------
# Shared HTTP client for Ollama (connection reuse; avoids per-request TCP setup)
# ---------------------------------------------------------------------------

_ollama_http_client: httpx.AsyncClient | None = None
_ollama_http_lock = asyncio.Lock()


async def init_ollama_http_client() -> None:
    """Create a long-lived HTTP client for Ollama (call from FastAPI lifespan startup)."""
    global _ollama_http_client
    async with _ollama_http_lock:
        if _ollama_http_client is None:
            _ollama_http_client = httpx.AsyncClient(
                timeout=OLLAMA_TIMEOUT,
                limits=httpx.Limits(max_connections=100, max_keepalive_connections=20),
            )


async def close_ollama_http_client() -> None:
    """Close the shared Ollama HTTP client (FastAPI lifespan shutdown)."""
    global _ollama_http_client
    async with _ollama_http_lock:
        if _ollama_http_client is not None:
            await _ollama_http_client.aclose()
            _ollama_http_client = None


async def _get_ollama_http_client() -> httpx.AsyncClient:
    """Return shared client, creating it lazily if lifespan did not run (e.g. scripts/tests)."""
    global _ollama_http_client
    if _ollama_http_client is None:
        await init_ollama_http_client()
    assert _ollama_http_client is not None
    return _ollama_http_client


# ---------------------------------------------------------------------------
# Core Ollama chat (JSON mode)
# ---------------------------------------------------------------------------


def _try_parse_llm_json(content: str) -> Any | None:
    """Parse JSON from model output; tolerate markdown fences, prose, object or array roots."""
    raw = (content or "").strip()
    if not raw:
        return None
    decoder = json.JSONDecoder()

    def _first_json_value(s: str) -> Any | None:
        for i, ch in enumerate(s):
            if ch not in "[{":
                continue
            try:
                return decoder.raw_decode(s, i)[0]
            except json.JSONDecodeError:
                continue
        return None

    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        pass
    if "```" in raw:
        start = raw.find("```")
        if start != -1:
            nl = raw.find("\n", start)
            if nl != -1:
                end = raw.find("```", nl + 1)
                if end != -1:
                    inner = raw[nl + 1 : end].strip()
                    try:
                        return json.loads(inner)
                    except json.JSONDecodeError:
                        hit = _first_json_value(inner)
                        if hit is not None:
                            return hit
    return _first_json_value(raw)


def _coerce_json_root_to_dict(
    parsed: Any,
    schema: dict[str, Any] | None,
) -> dict[str, Any] | None:
    """If the model returns a JSON array at root, wrap it under the schema's list field when unambiguous."""
    if isinstance(parsed, dict):
        return parsed
    if not isinstance(parsed, list) or not schema:
        return None
    for key, val in schema.items():
        if isinstance(val, list):
            return {key: parsed}
    if len(schema) == 1:
        return {next(iter(schema.keys())): parsed}
    return None


def _ollama_http_retryable(exc: BaseException, status_code: int | None) -> bool:
    """Whether to retry the POST /api/chat (transient transport or overload)."""
    if isinstance(
        exc,
        (
            httpx.ConnectError,
            httpx.ConnectTimeout,
            httpx.ReadTimeout,
            httpx.WriteTimeout,
            httpx.PoolTimeout,
            httpx.ReadError,
            httpx.WriteError,
            httpx.RemoteProtocolError,
        ),
    ):
        return True
    if isinstance(exc, httpx.HTTPStatusError) and status_code is not None:
        return status_code == 429 or status_code >= 500
    return False


async def _chat_json_openai(
    system_prompt: str,
    user_prompt: str,
    schema: dict[str, Any] | None = None,
    *,
    temperature: float | None = None,
    num_predict: int | None = None,
    max_json_retries: int | None = None,
) -> dict[str, Any] | None:
    """Chat completions with JSON object mode (OpenAI-compatible servers)."""
    if not ai_enabled():
        return None

    full_system = _attach_schema(system_prompt, schema)

    json_retries = OLLAMA_JSON_MAX_RETRIES if max_json_retries is None else max_json_retries
    max_json_attempts = max(1, json_retries + 1)
    base_temp = OLLAMA_JSON_TEMPERATURE if temperature is None else temperature
    max_tokens = OLLAMA_NUM_PREDICT if num_predict is None else num_predict
    http_cap = max(0, OLLAMA_HTTP_MAX_RETRIES)
    client = await _get_openai_async_client()

    for json_attempt in range(max_json_attempts):
        temp = max(0.05, base_temp * (0.88**json_attempt))
        content: str | None = None
        last_exc: BaseException | None = None

        for http_try in range(http_cap + 1):
            try:
                completion = await client.chat.completions.create(
                    model=get_effective_chat_model(),
                    messages=[
                        {"role": "system", "content": full_system},
                        {"role": "user", "content": user_prompt},
                    ],
                    response_format={"type": "json_object"},
                    max_tokens=max_tokens,
                    temperature=temp,
                )
                content = (completion.choices[0].message.content or "").strip()
                break
            except Exception as e:
                last_exc = e
                if _openai_retryable(e) and http_try < http_cap:
                    logger.warning(
                        "OpenAI API error (try %s/%s), backing off: %s",
                        http_try + 1,
                        http_cap + 1,
                        e,
                    )
                    await asyncio.sleep(
                        OLLAMA_HTTP_RETRY_BACKOFF_SEC * (2**http_try),
                    )
                    continue
                logger.exception(
                    "OpenAI chat failed (json attempt %s, http try %s)",
                    json_attempt + 1,
                    http_try + 1,
                )
                return None

        if content is None:
            if last_exc is not None:
                logger.exception(
                    "OpenAI chat exhausted HTTP retries (json attempt %s)",
                    json_attempt + 1,
                )
            return None

        parsed = _try_parse_llm_json(content)
        coerced = _coerce_json_root_to_dict(parsed, schema)
        if coerced is not None:
            return coerced
        _preview = (content or "")[:200]
        _total = len(content or "")
        logger.warning(
            "OpenAI JSON parse failed (json attempt %s/%s); chars=%s; content prefix: %s",
            json_attempt + 1,
            max_json_attempts,
            _total,
            _preview,
        )

    logger.warning("OpenAI returned invalid JSON after %s json attempts", max_json_attempts)
    return None


async def _chat_json(
    system_prompt: str,
    user_prompt: str,
    schema: dict[str, Any] | None = None,
    *,
    temperature: float | None = None,
    num_predict: int | None = None,
    max_json_retries: int | None = None,
) -> dict[str, Any] | None:
    """Send a chat request with JSON output (Ollama native or OpenAI-compatible SDK).

    **HTTP:** Retries transient failures (connection, timeouts, 429/502/503/504) up to
    ``OLLAMA_HTTP_MAX_RETRIES`` with exponential backoff — independent of JSON parsing.

    **JSON:** After a successful HTTP response, retries only when the message body is not
    a parseable JSON object (or coercible array root). Each JSON retry lowers temperature.
    """
    if LLM_BACKEND == "openai":
        return await _chat_json_openai(
            system_prompt,
            user_prompt,
            schema,
            temperature=temperature,
            num_predict=num_predict,
            max_json_retries=max_json_retries,
        )

    if not ai_enabled():
        return None

    full_system = _attach_schema(system_prompt, schema)

    json_retries = OLLAMA_JSON_MAX_RETRIES if max_json_retries is None else max_json_retries
    max_json_attempts = max(1, json_retries + 1)
    base_temp = OLLAMA_JSON_TEMPERATURE if temperature is None else temperature
    tokens = OLLAMA_NUM_PREDICT if num_predict is None else num_predict
    http_cap = max(0, OLLAMA_HTTP_MAX_RETRIES)

    for json_attempt in range(max_json_attempts):
        temp = max(0.05, base_temp * (0.88**json_attempt))
        content: str | None = None
        last_http_exc: BaseException | None = None

        for http_try in range(http_cap + 1):
            try:
                client = await _get_ollama_http_client()
                chat_model = get_effective_chat_model()
                chat_payload: dict[str, Any] = {
                    "model": chat_model,
                    "messages": [
                        {"role": "system", "content": full_system},
                        {"role": "user", "content": user_prompt},
                    ],
                    "format": "json",
                    "stream": False,
                    "options": {
                        "temperature": temp,
                        "num_predict": tokens,
                    },
                }
                if "qwen3" in chat_model.lower():
                    chat_payload["think"] = False
                # Serialize Ollama requests — single-GPU can only run one inference at a time.
                # All coroutines acquire this semaphore before sending to the model server,
                # eliminating queueing overhead from simultaneous requests.
                async with _get_ollama_call_sem():
                    resp = await client.post(
                        f"{_OLLAMA_BASE}/api/chat",
                        json=chat_payload,
                    )
                resp.raise_for_status()
                try:
                    data = resp.json()
                except json.JSONDecodeError as je:
                    logger.warning(
                        "Ollama response body is not JSON (http try %s/%s): %s",
                        http_try + 1,
                        http_cap + 1,
                        je,
                    )
                    if http_try < http_cap:
                        await asyncio.sleep(
                            OLLAMA_HTTP_RETRY_BACKOFF_SEC * (2**http_try),
                        )
                        continue
                    return None
                content = data.get("message", {}).get("content", "")
                break
            except httpx.HTTPStatusError as e:
                last_http_exc = e
                code = e.response.status_code if e.response is not None else None
                if _ollama_http_retryable(e, code) and http_try < http_cap:
                    logger.warning(
                        "Ollama HTTP %s (try %s/%s), backing off: %s",
                        code,
                        http_try + 1,
                        http_cap + 1,
                        e,
                    )
                    await asyncio.sleep(
                        OLLAMA_HTTP_RETRY_BACKOFF_SEC * (2**http_try),
                    )
                    continue
                logger.exception(
                    "Ollama chat HTTP failed (json attempt %s, http try %s)",
                    json_attempt + 1,
                    http_try + 1,
                )
                return None
            except httpx.RequestError as e:
                last_http_exc = e
                if _ollama_http_retryable(e, None) and http_try < http_cap:
                    logger.warning(
                        "Ollama request error (try %s/%s), backing off: %s",
                        http_try + 1,
                        http_cap + 1,
                        e,
                    )
                    await asyncio.sleep(
                        OLLAMA_HTTP_RETRY_BACKOFF_SEC * (2**http_try),
                    )
                    continue
                logger.exception(
                    "Ollama chat request failed (json attempt %s, http try %s)",
                    json_attempt + 1,
                    http_try + 1,
                )
                return None

        if content is None:
            if last_http_exc is not None:
                logger.exception(
                    "Ollama chat exhausted HTTP retries (json attempt %s)",
                    json_attempt + 1,
                )
            return None

        parsed = _try_parse_llm_json(content)
        coerced = _coerce_json_root_to_dict(parsed, schema)
        if coerced is not None:
            return coerced
        _preview = (content or "")[:200]
        _total = len(content or "")
        logger.warning(
            "Ollama JSON parse failed (json attempt %s/%s); chars=%s; content prefix: %s",
            json_attempt + 1,
            max_json_attempts,
            _total,
            _preview,
        )

    logger.warning("Ollama returned invalid JSON after %s json attempts", max_json_attempts)
    return None


def _flatten_recommendation_item(item: Any) -> str:
    """Turn a stray dict from local LLMs into a single display string."""
    if isinstance(item, str):
        return item
    if not isinstance(item, dict):
        return str(item)
    for key in ("text", "recommendation", "summary", "action", "detail"):
        v = item.get(key)
        if isinstance(v, str) and v.strip():
            return v.strip()
    parts: list[str] = []
    for k, v in item.items():
        if isinstance(v, dict):
            continue
        if v is not None and str(v).strip():
            parts.append(f"{k}: {v}")
    return "; ".join(parts) if parts else json.dumps(item, ensure_ascii=False)


def _normalize_llm_dict(data: dict[str, Any]) -> None:
    """In-place fixes before Pydantic validation (mutates ``data``)."""
    if "recommendations" not in data:
        return
    rec = data.get("recommendations")
    if isinstance(rec, list):
        data["recommendations"] = [_flatten_recommendation_item(x) for x in rec]
    elif isinstance(rec, dict):
        data["recommendations"] = [_flatten_recommendation_item(rec)]


def _parse_model[T: BaseModel](model_cls: type[T], data: dict) -> T | None:
    """Safely parse a dict into a Pydantic model.

    Handles common LLM quirks like returning strings where lists are expected.
    """
    data = dict(data)
    _normalize_llm_dict(data)
    # Fix common LLM issue: string where list[str] expected
    for field_name, field_info in model_cls.model_fields.items():
        if field_name in data and isinstance(data[field_name], str):
            origin = getattr(field_info.annotation, "__origin__", None)
            if origin is list:
                # Split string into a single-element list
                data[field_name] = [data[field_name]]
    try:
        return model_cls.model_validate(data)
    except Exception as exc:
        logger.warning("Failed to validate %s: %s — data: %s", model_cls.__name__, exc, str(data)[:500])
        return None


# ---------------------------------------------------------------------------
# Pydantic result schemas
# ---------------------------------------------------------------------------


class JDRequirement(BaseModel):
    """A single requirement extracted from a job description."""
    skill_or_requirement: str = Field(description="The skill or requirement")
    category: str = Field(
        description="One of: technical, soft_skill, experience, education, certification",
    )
    importance: str = Field(description="One of: must_have, nice_to_have")
    alternatives: list[str] = Field(
        default_factory=list,
        description="Alternative skills that satisfy this requirement",
    )

    @model_validator(mode="before")
    @classmethod
    def _coerce_llm_requirement_shape(cls, data: Any) -> Any:
        """Accept common alternate JSON shapes from local models (skill vs skill_or_requirement)."""
        if not isinstance(data, dict):
            return data
        d = dict(data)
        if not d.get("skill_or_requirement"):
            for key in ("skill", "name", "requirement", "title", "item"):
                v = d.get(key)
                if isinstance(v, str) and v.strip():
                    d["skill_or_requirement"] = v.strip()
                    break
        cat = d.get("category")
        if not isinstance(cat, str) or not cat.strip():
            skill = str(d.get("skill_or_requirement") or "").lower()
            if any(x in skill for x in ("lead", "mentor", "communicat", "team", "stakeholder")):
                d["category"] = "soft_skill"
            elif any(x in skill for x in ("year", "experience", "seniority")):
                d["category"] = "experience"
            elif any(x in skill for x in ("degree", "bachelor", "master", "phd", "education")):
                d["category"] = "education"
            else:
                d["category"] = "technical"
        imp = d.get("importance")
        if isinstance(imp, str):
            low = imp.lower().replace(" ", "_").replace("-", "_")
            if low in ("required", "must", "essential", "mandatory", "must_have", "musthave"):
                d["importance"] = "must_have"
            elif low in ("preferred", "nice", "optional", "plus", "nice_to_have", "nicetohave", "bonus"):
                d["importance"] = "nice_to_have"
        if "alternatives" not in d or d["alternatives"] is None:
            d["alternatives"] = []
        elif isinstance(d["alternatives"], str):
            d["alternatives"] = [d["alternatives"]]
        return d


class ParsedJD(BaseModel):
    """Structured representation of a job description."""
    role_title: str = Field(description="The job title")
    seniority_level: str = Field(
        description="One of: intern, junior, mid, senior, lead, principal",
    )
    min_years_experience: int | None = Field(
        default=None, description="Minimum years required, or null",
    )
    required_education: str | None = Field(
        default=None, description="Required degree level or null",
    )
    requirements: list[JDRequirement] = Field(
        default_factory=list,
        description="All extracted requirements",
    )
    summary: str = Field(
        default="",
        description="1-2 sentence summary of what this role needs",
    )


class CandidateInsight(BaseModel):
    """AI-generated analysis of a single candidate."""
    fit_summary: str = Field(
        default="",
        description="2-3 sentence summary of how well this candidate fits",
    )
    strengths: list[str] = Field(
        default_factory=list,
        description="Top 3-5 strengths relative to this JD",
    )
    gaps: list[str] = Field(
        default_factory=list,
        description="Top 3-5 gaps or risks relative to this JD",
    )
    interview_questions: list[str] = Field(
        default_factory=list,
        description="3 targeted interview questions based on their gaps",
    )
    verdict: str = Field(
        default="lean_no",
        description="One of: strong_hire, lean_hire, lean_no, strong_no",
    )


class BatchSummary(BaseModel):
    """AI-generated executive summary across all candidates."""
    executive_summary: str = Field(
        default="",
        description="3-4 sentence executive summary for the hiring manager",
    )
    top_recommendation: str = Field(
        default="",
        description="Who to interview first and why (1-2 sentences)",
    )
    recommendations: list[str] = Field(
        default_factory=list,
        description="3-5 specific, actionable hiring recommendations. Each should be 1-2 sentences covering: interview priorities, skill gaps to probe, team fit considerations, salary/leveling notes, and timeline urgency.",
    )
    talent_gaps: list[str] = Field(
        default_factory=list,
        description="Common skill gaps across all candidates",
    )
    hiring_risk: str = Field(
        default="medium",
        description="One of: low, medium, high based on candidate pool quality",
    )
    qualified_threshold_score: float = Field(
        default=65.0,
        description="The minimum score you'd consider 'qualified' for this specific role/JD. Consider the JD requirements strictly.",
    )

    @field_validator("qualified_threshold_score", mode="before")
    @classmethod
    def _coerce_qualified_threshold(cls, v: Any) -> float:
        if isinstance(v, str):
            v = v.replace("%", "").strip().split()[0]
        return float(v)


class ResumeSuggestion(BaseModel):
    """A single improvement suggestion for a resume."""
    category: str = Field(
        default="formatting",
        description="One of: missing_section, weak_skill, action_verbs, formatting, tech_stack",
    )
    current_state: str = Field(
        default="",
        description="What's currently on the resume that needs improvement",
    )
    improvement: str = Field(
        default="",
        description="Specific actionable suggestion to improve the resume",
    )
    example: str | None = Field(
        default=None,
        description="Before/after example text showing the improvement",
    )
    before_text: str = Field(
        default="",
        description="Exact text from the resume that should be improved (the BEFORE)",
    )
    after_text: str = Field(
        default="",
        description="Rewritten text showing the improvement (the AFTER)",
    )
    estimated_lift: str = Field(
        default="",
        description="Estimated score lift from this single suggestion, e.g. '+3 pts (skills)'",
    )
    priority: str = Field(
        default="medium",
        description="One of: critical, high, medium, low",
    )
    jd_relevance: str = Field(
        default="",
        description="Why this improvement matters for this specific JD",
    )

    @model_validator(mode="after")
    def _split_example_into_before_after(self) -> "ResumeSuggestion":
        """If LLM returns a combined example string, split it into before_text/after_text."""
        if self.before_text and self.after_text:
            return self
        text = self.example or self.current_state or ""
        if not text:
            return self
        import re as _re
        m = _re.search(
            r'(?:BEFORE|Before|before)\s*:\s*["\u201c]?(.+?)["\u201d]?\s*\n\s*'
            r'(?:AFTER|After|after)\s*:\s*["\u201c]?(.+?)["\u201d]?\s*$',
            text, _re.DOTALL,
        )
        if m:
            if not self.before_text:
                self.before_text = m.group(1).strip().strip('"')
            if not self.after_text:
                self.after_text = m.group(2).strip().strip('"')
        return self


class ResumeSuggestions(BaseModel):
    """AI-generated improvement suggestions for a candidate's resume."""
    total_score: float = Field(
        default=50.0,
        description="Current ATS score percentage (number only, no % sign)",
    )
    potential_score: float = Field(
        default=70.0,
        description="Estimated score if all suggestions implemented (number only, no % sign)",
    )

    @field_validator("total_score", "potential_score", mode="before")
    @classmethod
    def _parse_score(cls, v: Any) -> float:
        if isinstance(v, str):
            v = v.replace("%", "").strip().split()[0]
        return float(v)
    suggestions: list[ResumeSuggestion] = Field(
        default_factory=list,
        description="Up to 5 prioritized improvement suggestions",
    )
    summary: str = Field(
        default="",
        description="Brief 2 sentence improvement roadmap",
    )


class LLMHolisticScore(BaseModel):
    """LLM-generated holistic scoring of resume against JD."""
    requirement_fulfillment: float = Field(
        default=50.0,
        description="How well the candidate meets the stated JD requirements (0-100)",
    )
    experience_relevance: float = Field(
        default=50.0,
        description="How relevant their past experience is to this specific role (0-100)",
    )
    project_alignment: float = Field(
        default=50.0,
        description="How well their projects demonstrate needed capabilities (0-100)",
    )
    skill_depth: float = Field(
        default=50.0,
        description="Depth of skill expertise vs surface-level mention (0-100)",
    )
    overall_fit: float = Field(
        default=50.0,
        description="Overall candidate-JD fit score (0-100)",
    )
    confidence: float = Field(
        default=0.7,
        description="LLM confidence in this assessment (0-1)",
    )
    reasoning: str = Field(
        default="",
        description="2-3 sentence justification for the scores",
    )
    keyword_matches: list[str] = Field(
        default_factory=list,
        description="JD keywords found in resume",
    )
    keyword_gaps: list[str] = Field(
        default_factory=list,
        description="Important JD keywords missing from resume",
    )


_UNIFIED_SECTION_DIM_KEYS: tuple[str, ...] = (
    "skills",
    "similarity",
    "experience",
    "education",
    "projects",
    "certifications",
)

# Credit weight tiers for inference types (used in validators and scoring).
_TRANSFERABLE_CREDIT_WEIGHT: float = 0.40
_OUTCOME_CREDIT_WEIGHT: float = 0.65
_TRANSFERABLE_WEIGHT_RANGE: tuple[float, float] = (0.35, 0.45)
_OUTCOME_WEIGHT_RANGE: tuple[float, float] = (0.60, 0.70)


class InferredSkill(BaseModel):
    """A skill inferred from transferable experience or outcome-language evidence."""

    skill: str = Field(description="Normalized skill name matching skill_db")
    evidence: str = Field(
        default="",
        description="Direct quote or paraphrase from resume supporting this inference",
    )
    credit_weight: float = Field(
        default=_TRANSFERABLE_CREDIT_WEIGHT,
        ge=0.0,
        le=1.0,
        description="0.40 for transferable | 0.65 for outcome",
    )
    confidence: float = Field(
        default=0.75,
        ge=0.0,
        le=1.0,
        description="LLM's self-assessed confidence (0-1)",
    )
    inference_type: str = Field(
        default="transferable",
        description="'transferable' or 'outcome'",
    )

    @field_validator("credit_weight", mode="before")
    @classmethod
    def _clamp_credit_weight(cls, v: Any) -> float:
        try:
            val = float(v)
        except (TypeError, ValueError):
            return _TRANSFERABLE_CREDIT_WEIGHT
        clamped = max(0.0, min(1.0, val))
        return clamped

    @model_validator(mode="after")
    def _warn_weight_range(self) -> "InferredSkill":
        """Warn when credit_weight is outside the defined tier ranges and clamp to tier."""
        if self.inference_type == "outcome":
            lo, hi = _OUTCOME_WEIGHT_RANGE
        else:
            lo, hi = _TRANSFERABLE_WEIGHT_RANGE
        if not (lo <= self.credit_weight <= hi):
            logger.warning(
                "InferredSkill credit_weight %.2f outside range [%.2f, %.2f] for "
                "inference_type=%r - clamping to range",
                self.credit_weight,
                lo,
                hi,
                self.inference_type,
            )
            self.credit_weight = max(lo, min(hi, self.credit_weight))
        return self


class UnifiedResumeScore(BaseModel):
    """Single-call dimension scores aligned with ``scoring_engine`` ``section_scores``."""

    skills_score: float = Field(ge=0, le=100, description="Technical / JD skill fit (0-100)")
    similarity_score: float = Field(ge=0, le=100, description="Overall resume–JD alignment (0-100)")
    experience_score: float = Field(ge=0, le=100, description="Experience relevance (0-100)")
    education_score: float = Field(ge=0, le=100, description="Education fit (0-100)")
    projects_score: float = Field(ge=0, le=100, description="Projects relevance (0-100)")
    certifications_score: float = Field(ge=0, le=100, description="Certifications match (0-100)")
    matched_skills: list[str] = Field(
        default_factory=list,
        description="JD skills or requirements clearly satisfied by the resume",
    )
    missing_skills: list[str] = Field(
        default_factory=list,
        description="Important JD skills or requirements not evidenced",
    )
    transferable_skills: list[str] = Field(
        default_factory=list,
        description="Partial / transfer equivalents (e.g. Redis for Memcached caching)",
    )
    section_explanations: dict[str, str] = Field(
        default_factory=dict,
        description="One short sentence per dimension key",
    )
    reasoning: str = Field(default="", description="2-4 sentence evidence-based justification")
    confidence: float = Field(default=0.75, ge=0, le=1, description="Model confidence 0-1")
    inferred_skills: list[InferredSkill] = Field(
        default_factory=list,
        description=(
            "Skills inferred via transferable experience (credit_weight=0.40) or "
            "outcome-language evidence (credit_weight=0.65). Each must cite resume evidence."
        ),
    )

    @field_validator("inferred_skills", mode="before")
    @classmethod
    def _coerce_inferred_skills(cls, v: Any) -> list[Any]:
        """Gracefully handle missing or malformed inferred_skills from LLM responses."""
        if v is None:
            return []
        if isinstance(v, list):
            coerced: list[Any] = []
            for item in v:
                if isinstance(item, (dict, InferredSkill)):
                    # dict → Pydantic constructs InferredSkill; InferredSkill → passed through
                    coerced.append(item)
                elif isinstance(item, str) and item.strip():
                    # Older model returned plain string list — convert to minimal InferredSkill
                    logger.warning(
                        "inferred_skills: plain string item %r — converting with defaults",
                        item,
                    )
                    coerced.append(
                        {
                            "skill": item.strip(),
                            "evidence": "",
                            "credit_weight": _TRANSFERABLE_CREDIT_WEIGHT,
                        }
                    )
            return coerced
        if isinstance(v, str):
            logger.warning(
                "inferred_skills: expected list but got string — defaulting to []"
            )
            return []
        logger.warning(
            "inferred_skills: unexpected type %s — defaulting to []", type(v).__name__
        )
        return []

    @field_validator(
        "skills_score",
        "similarity_score",
        "experience_score",
        "education_score",
        "projects_score",
        "certifications_score",
        mode="before",
    )
    @classmethod
    def _coerce_score(cls, v: Any) -> float:
        if isinstance(v, (int, float)):
            return float(v)
        s = str(v).strip()
        if not s or s.lower() in {"n/a", "na", "none", "null", "not applicable"}:
            return 0.0
        s = s.replace("%", "")
        try:
            return float(s)
        except ValueError:
            return 0.0

    @field_validator("matched_skills", "missing_skills", "transferable_skills", mode="before")
    @classmethod
    def _coerce_skill_lists(cls, v: Any) -> list[str]:
        if v is None:
            return []
        if isinstance(v, list):
            return [str(item).strip() for item in v if str(item).strip()]
        text = str(v).strip()
        if not text:
            return []
        parts = [p.strip() for p in text.replace("\n", ",").split(",")]
        return [p for p in parts if p]

    @model_validator(mode="before")
    @classmethod
    def _coerce_section_explanations(cls, data: Any) -> Any:
        """Local LLMs sometimes return a string or list instead of dict[str, str]."""
        if not isinstance(data, dict):
            return data
        se = data.get("section_explanations")
        if isinstance(se, str):
            # LLM returned a plain string — move it to reasoning and use empty dict
            data = dict(data)
            data["section_explanations"] = {}
            if not data.get("reasoning"):
                data["reasoning"] = se
            return data
        if not isinstance(se, list):
            return data
        out: dict[str, str] = {}
        keys = _UNIFIED_SECTION_DIM_KEYS
        for i, item in enumerate(se):
            text = str(item).strip()
            if not text:
                continue
            key = keys[i] if i < len(keys) else f"extra_{i}"
            out[key] = text
        data = dict(data)
        data["section_explanations"] = out
        return data


# ---------------------------------------------------------------------------
# LLM Skill Extraction — primary skill-matching path replacing the regex engine
# ---------------------------------------------------------------------------

class SkillMatch(BaseModel):
    """A JD skill assessed against the resume (explicit, transferable, or limited)."""

    skill: str = Field(description="JD skill name as written in the job description")
    evidence: str = Field(
        default="",
        description="Verbatim quote or close paraphrase from the resume supporting this assessment",
    )
    confidence: float = Field(
        default=0.85,
        ge=0.0,
        le=1.0,
        description="Self-assessed confidence (0-1)",
    )
    match_type: str = Field(
        default="explicit",
        description="'explicit' | 'transferable' | 'mentioned_but_limited'",
    )

    @field_validator("confidence", mode="before")
    @classmethod
    def _clamp_confidence(cls, v: Any) -> float:
        try:
            return max(0.0, min(1.0, float(v)))
        except (TypeError, ValueError):
            return 0.85

    @field_validator("match_type", mode="before")
    @classmethod
    def _coerce_match_type(cls, v: Any) -> str:
        s = str(v).strip().lower()
        if s in {"explicit", "transferable", "mentioned_but_limited"}:
            return s
        return "explicit"

    @field_validator("skill", "evidence", mode="before")
    @classmethod
    def _coerce_str(cls, v: Any) -> str:
        return str(v).strip() if v is not None else ""


class TransferableSkill(BaseModel):
    """A skill inferred as transferable: the resume implies the JD skill without naming it."""

    skill: str = Field(description="JD skill being credited via transfer")
    source: str = Field(
        default="",
        description="Skill or technology in the resume that implies this JD skill",
    )
    evidence: str = Field(
        default="",
        description="Verbatim quote from the resume supporting this inference",
    )
    credit_weight: float = Field(
        default=0.55,
        description="Partial credit weight (0.40–0.65; 1.0 = full match)",
    )

    @field_validator("credit_weight", mode="before")
    @classmethod
    def _clamp_credit(cls, v: Any) -> float:
        try:
            return max(0.40, min(0.65, float(v)))
        except (TypeError, ValueError):
            return 0.55

    @field_validator("skill", "source", "evidence", mode="before")
    @classmethod
    def _coerce_str(cls, v: Any) -> str:
        return str(v).strip() if v is not None else ""


class LLMSkillExtraction(BaseModel):
    """Structured output of the LLM skill extraction call (Call 1 of the two-call architecture)."""

    jd_required_skills: list[str] = Field(
        default_factory=list,
        description="Required skills parsed from the JD — passed to unified scorer as context",
    )
    jd_preferred_skills: list[str] = Field(
        default_factory=list,
        description="Preferred/nice-to-have skills from the JD",
    )
    matched: list[SkillMatch] = Field(
        default_factory=list,
        description="JD required skills confirmed present in the resume",
    )
    missing: list[SkillMatch] = Field(
        default_factory=list,
        description="JD required skills absent from the resume",
    )
    transferable: list[TransferableSkill] = Field(
        default_factory=list,
        description="JD required skills not explicitly present but inferable from resume context",
    )
    skills_score: float = Field(
        default=50.0,
        description="Overall skill fit score (0-100)",
    )

    @field_validator("skills_score", mode="before")
    @classmethod
    def _coerce_score(cls, v: Any) -> float:
        if isinstance(v, (int, float)):
            return max(0.0, min(100.0, float(v)))
        s = str(v).strip().replace("%", "")
        try:
            return max(0.0, min(100.0, float(s)))
        except (TypeError, ValueError):
            return 50.0

    @field_validator(
        "jd_required_skills", "jd_preferred_skills", mode="before"
    )
    @classmethod
    def _coerce_str_list(cls, v: Any) -> list[str]:
        if v is None:
            return []
        if isinstance(v, list):
            return [str(item).strip() for item in v if str(item).strip()]
        if isinstance(v, str):
            parts = [p.strip() for p in v.replace("\n", ",").split(",")]
            return [p for p in parts if p]
        return []

    @field_validator("matched", "missing", mode="before")
    @classmethod
    def _coerce_skill_match_list(cls, v: Any) -> list[Any]:
        if v is None:
            return []
        if isinstance(v, list):
            return v
        return []

    @field_validator("transferable", mode="before")
    @classmethod
    def _coerce_transferable_list(cls, v: Any) -> list[Any]:
        if v is None:
            return []
        if isinstance(v, list):
            return v
        return []


_SKILL_EXTRACTION_SYSTEM = """\
You are an expert technical recruiter performing a precise skill gap analysis.
Given a Job Description (JD) and a candidate's Resume, your task is to determine which
JD-required skills the candidate has — explicitly, through transferable experience, or
with limitations — and output structured JSON only.

SCORING RUBRIC (skills_score 0-100):
  100 = all required skills matched explicitly
   85 = most required skills matched; 1-2 minor gaps
   70 = several strong matches; some gaps or transferable fills
   50 = roughly half matched; significant gaps
  <40 = few matches; many critical gaps

MATCH TYPES:
  "explicit"               — skill is clearly stated in the resume
  "transferable"           — a closely related skill implies this one (cite evidence)
  "mentioned_but_limited"  — candidate mentions the skill but flags inexperience/learning

TRANSFERABLE SKILL RULES (CRITICAL):
  1. ONLY infer transferable skills when you can cite a VERBATIM quote or close
     paraphrase from the resume as evidence.
  2. If no such evidence exists, do NOT list it as transferable.
  3. Common valid inferences: PySpark/Apache Spark → Python; Redis (production) → caching;
     WCNP/OpenShift/EKS → Kubernetes; Azure SQL/BigQuery/Snowflake → schema design;
     Django/Flask → Python; Terraform → IaC; Spring Boot → Java.
  4. Do NOT infer a skill just because the candidate works in the same domain.

NEGATION RULE:
  If a resume says "familiar with X but no production experience", "learning X", or
  "exposure to X" — use match_type "mentioned_but_limited", NOT "explicit".

For each item in matched and missing, include an evidence quote when one exists.
If no evidence exists for a matched skill, set evidence to "".

Output JSON only. No prose, no markdown, no explanation outside the JSON.
"""

_SKILL_EXTRACTION_SCHEMA_HINT = """\
{
  "jd_required_skills": ["skill1", "skill2"],
  "jd_preferred_skills": ["skill3"],
  "matched": [
    {"skill": "Python", "evidence": "Languages: Java, Python, SQL", "confidence": 0.98, "match_type": "explicit"},
    {"skill": "Kubernetes", "evidence": "Tech: Spring Boot, Java, Azure SQL, WCNP", "confidence": 0.85, "match_type": "transferable"}
  ],
  "missing": [
    {"skill": "Docker", "evidence": "", "confidence": 0.9, "match_type": "explicit"}
  ],
  "transferable": [
    {"skill": "caching", "source": "Redis", "evidence": "Cloud & Data: ... Redis ...", "credit_weight": 0.55}
  ],
  "skills_score": 82.0
}"""


async def extract_skills_llm(
    resume_text: str,
    jd_text: str,
) -> "LLMSkillExtraction | None":
    """LLM-native skill extraction: primary replacement for the regex match_skills() path.

    Parses both the JD and resume, identifies matched/missing/transferable required
    skills with evidence quotes, and returns a structured ``LLMSkillExtraction``.

    Returns ``None`` on any failure — caller should fall back to ``_fallback_extraction()``.
    """
    if not ai_enabled():
        return None

    schema = _schema_for(LLMSkillExtraction)
    prompt = (
        f"--- JOB DESCRIPTION ---\n{jd_text[:JD_MAX_CHARS_LLM]}\n\n"
        f"--- RESUME ---\n{resume_text}\n\n"
        f"--- EXPECTED JSON SCHEMA ---\n{_SKILL_EXTRACTION_SCHEMA_HINT}"
    )
    data = await _chat_json(
        system_prompt=_SKILL_EXTRACTION_SYSTEM,
        user_prompt=prompt,
        schema=schema,
        temperature=0.0,  # R2: determinism — minimize variance in skill extraction
        num_predict=OLLAMA_NUM_PREDICT_JSON_HEAVY,
    )
    if data is None:
        return None
    return _parse_model(LLMSkillExtraction, data)


# ---------------------------------------------------------------------------
# System prompts
# ---------------------------------------------------------------------------

_JD_PARSER_SYSTEM = (
    "You are an expert technical recruiter and HR analyst. "
    "Parse job descriptions into structured requirements. "
    "Be precise about must_have vs nice_to_have — words like 'required', "
    "'must', 'minimum' indicate must_have. Words like 'preferred', 'bonus', "
    "'nice to have', 'ideally', 'plus' indicate nice_to_have. "
    "Detect alternative skills: 'Java or Kotlin', 'React/Angular', "
    "'AWS or Azure' — list one as the main skill and others as alternatives."
)

_CANDIDATE_ANALYST_SYSTEM = (
    "You are a senior technical hiring manager. "
    "Analyze how well a candidate's resume fits a specific job description. "
    "Be specific and actionable — reference actual skills, projects, and "
    "experience from the resume. Don't be generic. "
    "For interview questions, target the candidate's specific gaps to "
    "determine if they can grow into the role."
)

_BATCH_SUMMARY_SYSTEM = (
    "You are a VP of Engineering reviewing candidates for a specific role. "
    "Provide a thorough executive summary for the hiring manager. "
    "Be concise, data-driven, and actionable. "
    "Generate 3-5 specific recommendations covering: who to interview first, "
    "what skill gaps to probe in interviews, team composition considerations, "
    "and any urgency signals. "
    "CRITICAL: the recommendations field MUST be a JSON array of plain strings only "
    "(each item one or two sentences). Do NOT use objects or nested structures for recommendations. "
    "Also identify market-level talent gaps (if all candidates lack X, that's "
    "a market signal). "
    "Set qualified_threshold_score to the minimum score you'd consider qualified "
    "for this specific JD — be strict but fair (typically 60-75 depending on role seniority)."
)

_HOLISTIC_SCORER_SYSTEM = (
    "You are an expert ATS scoring engine. Score how well a resume matches a job description. "
    "Be precise and data-driven. Reference specific skills, projects, and experience from both "
    "documents. Score each dimension 0-100 where 50 is average, 70+ is good, 85+ is excellent. "
    "Also identify keyword matches and gaps between the JD and resume."
)

_UNIFIED_SCORER_SYSTEM = (
    "You are an expert ATS scoring engine. In ONE response, score this resume against the job "
    "description. Output JSON only. Each score is 0-100 (50=average, 70+=good, 85+=excellent). "
    "Dimensions: skills_score (technical/JD skill fit), similarity_score (overall alignment), "
    "experience_score, education_score, projects_score, certifications_score. "
    "List matched_skills (JD requirements clearly satisfied), missing_skills (important gaps), "
    "transferable_skills (partial/transfer equivalents). "
    "IMPORTANT: section_explanations MUST be a JSON object (dict) mapping dimension keys to strings, "
    "like this: {\"skills\": \"sentence\", \"similarity\": \"sentence\", \"experience\": \"sentence\", "
    "\"education\": \"sentence\", \"projects\": \"sentence\", \"certifications\": \"sentence\"}. "
    "Do NOT return section_explanations as a plain string or list. "
    "reasoning: 2-4 sentences citing the resume and JD. confidence: 0-1. "
    "For inferred_skills, identify JD-required skills this candidate could demonstrate via "
    "transferable experience. For each: set skill to the JD skill name, cite a specific sentence "
    "from the resume as evidence, set inference_type='transferable', credit_weight=0.40, and "
    "estimate your confidence (0-1). Calibrate confidence by seniority: senior candidates "
    "(8+ years, led or architected systems, shipped production software) score 0.70-0.85; "
    "junior candidates score 0.30-0.50. Only include inferences backed by resume evidence — "
    "omit any inference without a specific supporting sentence from the resume. "
    "Also detect skills implied by outcome language: 'implemented schema migrations' implies "
    "PostgreSQL; 'added cache layer' implies Redis; 'tuned slow queries' implies SQL proficiency. "
    "For outcome-implied skills set inference_type='outcome' and credit_weight=0.65 and cite "
    "the outcome phrase as evidence. Include outcome-detected skills in inferred_skills as well."
)

_RESUME_SUGGESTIONS_SYSTEM = (
    "You are an expert ATS resume analyst and hiring manager who shortlists candidates daily. "
    "You know exactly how ATS parsers extract keywords and how recruiters scan resumes in 30 seconds.\n\n"
    "Your job: give the candidate 5 SPECIFIC, SURGICAL improvements that will raise their ATS score.\n\n"
    "RULES:\n"
    "1. before_text must be EXACT text copied from the resume (or describe what is missing).\n"
    "2. after_text must be a COMPLETE rewrite — not a vague instruction. Write the actual text they should paste.\n"
    "3. after_text must contain MORE JD keywords, stronger action verbs, and quantified metrics than before_text.\n"
    "4. NEVER suggest removing, shortening, or hiding content. Always ADD, STRENGTHEN, or REFRAME.\n"
    "5. Each suggestion targets a specific scoring dimension and includes estimated_lift (e.g. '+3 pts (skills)').\n"
    "6. Prioritize: missing must-have skills first, then weak high-weight dimensions, then formatting.\n"
    "7. potential_score MUST be strictly greater than total_score."
)


# ---------------------------------------------------------------------------
# Schema helpers (for JSON mode)
# ---------------------------------------------------------------------------

@functools.lru_cache(maxsize=None)
def _schema_for(model_cls: type[BaseModel]) -> dict[str, Any]:
    """Generate a simplified JSON schema for the LLM prompt (cached per model class)."""
    # Remove $defs and other metadata that confuse local models
    simplified = {}
    for field_name, field_info in model_cls.model_fields.items():
        desc = field_info.description or ""
        simplified[field_name] = desc
    return simplified


_schema_json_cache: dict[int, str] = {}


def _schema_json(schema: dict[str, Any]) -> str:
    """Return JSON-serialized schema, cached by object identity.

    Hot-path functions always pass the same dict object (module constant or
    lru_cache result), so id() is a stable cache key and json.dumps runs once
    per unique schema object rather than once per LLM call.
    """
    key = id(schema)
    cached = _schema_json_cache.get(key)
    if cached is None:
        cached = json.dumps(schema, indent=2)
        _schema_json_cache[key] = cached
    return cached


def _attach_schema(system_prompt: str, schema: dict[str, Any] | None) -> str:
    """Append JSON schema hint to system prompt. Schema serialization is cached by
    object identity — no re-serialization for the same schema dict across LLM calls."""
    if not schema:
        return system_prompt
    return (
        system_prompt
        + "\n\nYou MUST respond with valid JSON matching this exact schema:\n"
        + _schema_json(schema)
    )


async def score_resume_with_llm(
    resume_text: str,
    jd_text: str,
) -> LLMHolisticScore | None:
    """Use LLM to generate a holistic score of resume against JD."""
    if not ai_enabled():
        return None

    schema = _schema_for(LLMHolisticScore)
    prompt = f"""Score this resume against the job description across multiple dimensions.

--- JOB DESCRIPTION ---
{jd_text[:JD_MAX_CHARS_LLM]}

--- RESUME ---
{resume_text}

For each dimension, provide a score 0-100:
1. REQUIREMENT FULFILLMENT: Does the candidate meet the stated requirements?
2. EXPERIENCE RELEVANCE: Is their experience directly applicable?
3. PROJECT ALIGNMENT: Do their projects show the right capabilities?
4. SKILL DEPTH: Do they show deep expertise or just keyword-stuffing?
5. OVERALL FIT: Weighted combination considering the role's priorities.

Also extract:
- keyword_matches: JD keywords/phrases that appear in the resume
- keyword_gaps: Important JD keywords NOT found in the resume

Provide a confidence score (0-1) for your assessment.
Be specific in your reasoning - cite evidence from both documents."""

    data = await _chat_json(
        system_prompt=_HOLISTIC_SCORER_SYSTEM,
        user_prompt=prompt,
        schema=schema,
    )
    if data is None:
        return None
    return _parse_model(LLMHolisticScore, data)


async def analyze_resume_unified(
    resume_text: str,
    jd_text: str,
    *,
    extraction_result: "LLMSkillExtraction | None" = None,
    regex_matched_skills: list[str] | None = None,
    regex_missing_skills: list[str] | None = None,
    education_details: "list[dict] | None" = None,
    cert_details: "list[str] | None" = None,
    experience_metadata: "dict | None" = None,
) -> UnifiedResumeScore | None:
    """Single LLM call returning six dimension scores and skill lists (AI-primary scoring path).

    ``extraction_result`` (from ``extract_skills_llm()``) is the preferred grounding
    source. The legacy ``regex_matched_skills`` / ``regex_missing_skills`` params are
    kept for backward compatibility; when ``extraction_result`` is provided they are
    ignored.

    ``education_details``, ``cert_details``, and ``experience_metadata`` are pre-parsed
    structured facts from the full (untruncated) resume injected via
    ``build_resume_grounding_block()`` so the LLM never loses late-section data.
    """
    if not ai_enabled():
        return None

    schema = _schema_for(UnifiedResumeScore)

    # Build grounding context from extraction result (preferred) or legacy regex lists
    grounding_block = ""
    if extraction_result is not None:
        matched_names = [m.skill for m in extraction_result.matched]
        missing_names = [m.skill for m in extraction_result.missing]
        transferable_lines = [
            f"  - {t.skill} (via {t.source}): {t.evidence[:120]}"
            for t in extraction_result.transferable
            if t.evidence
        ]
        lines: list[str] = []
        if matched_names:
            lines.append(f"LLM EXTRACTION MATCHED: {', '.join(matched_names[:30])}")
        if missing_names:
            lines.append(f"LLM EXTRACTION MISSING: {', '.join(missing_names[:20])}")
        if transferable_lines:
            lines.append("LLM EXTRACTION TRANSFERABLE:\n" + "\n".join(transferable_lines[:10]))
        lines.append(f"LLM EXTRACTION SKILLS SCORE: {extraction_result.skills_score:.1f}/100")
        grounding_block = (
            "\n\n--- SKILL MATCH PRE-ANALYSIS (from LLM extraction call) ---\n"
            + "\n".join(lines)
            + "\nUse these as the base for matched_skills and missing_skills. "
            "Do NOT list as missing_skills any skill that appears in LLM EXTRACTION MATCHED. "
            "You may add soft skills, domain knowledge, or certifications the extraction missed, "
            "but do NOT contradict the extraction results.\n"
        )
    elif regex_matched_skills or regex_missing_skills:
        # Legacy path: regex grounding (deprecated, kept for callers not yet updated)
        legacy_lines: list[str] = []
        if regex_matched_skills:
            legacy_lines.append(
                f"REGEX ENGINE MATCHED: {', '.join(sorted(regex_matched_skills)[:30])}"
            )
        if regex_missing_skills:
            legacy_lines.append(
                f"REGEX ENGINE MISSING: {', '.join(regex_missing_skills[:20])}"
            )
        grounding_block = (
            "\n\n--- SKILL MATCH PRE-ANALYSIS (from deterministic engine) ---\n"
            + "\n".join(legacy_lines)
            + "\nUse these as the base for matched_skills and missing_skills. "
            "Do NOT list as missing_skills any skill that appears in REGEX ENGINE MATCHED above. "
            "You may add skills the regex engine missed (e.g. soft skills, domain knowledge, "
            "certifications), but do NOT contradict the regex results.\n"
        )

    resume_facts_block = build_resume_grounding_block(
        education_details=education_details,
        cert_details=cert_details,
        experience_metadata=experience_metadata,
    )

    prompt = (
        f"--- JOB DESCRIPTION ---\n{jd_text[:JD_MAX_CHARS_LLM]}\n\n"
        f"--- RESUME ---\n{resume_text}"
        + grounding_block
        + resume_facts_block
    )
    data = await _chat_json(
        system_prompt=_UNIFIED_SCORER_SYSTEM,
        user_prompt=prompt,
        schema=schema,
        temperature=0.0,
    )
    if data is None:
        return None
    result = _parse_model(UnifiedResumeScore, data)
    if result is None:
        return None

    # Post-process: remove from missing_skills any skill that was already matched.
    matched_lower: set[str] = set()
    if extraction_result is not None:
        matched_lower = {m.skill.lower().strip() for m in extraction_result.matched}
    elif regex_matched_skills:
        matched_lower = {s.lower().strip() for s in regex_matched_skills}
    if matched_lower and result.missing_skills:
        result.missing_skills = [
            s for s in result.missing_skills
            if s.lower().strip() not in matched_lower
        ]
    return result


# ---------------------------------------------------------------------------
# Agent functions
# ---------------------------------------------------------------------------


def _format_score_context_for_llm(score_data: dict[str, Any]) -> str:
    """Build a concise score payload for insight/suggestion prompts (AI-primary aware)."""
    lines: list[str] = [
        f"Overall score: {score_data.get('final_score')}%",
        f"Scoring method: {score_data.get('scoring_method', 'unknown')}",
        f"Matched skills: {', '.join(score_data.get('matched_skills', []))}",
        f"Missing skills: {', '.join(score_data.get('missing_skills', []))}",
        f"Section scores: {score_data.get('section_scores')}",
    ]
    if score_data.get("top_factors"):
        lines.append(f"Lowest dimensions: {score_data.get('top_factors')}")
    explanations = score_data.get("section_explanations") or {}
    if isinstance(explanations, dict) and explanations:
        lines.append("Per-dimension notes (from primary scorer):")
        for dim, note in list(explanations.items())[:8]:
            if note:
                note_s = str(note).replace("\n", " ")[:400]
                lines.append(f"  - {dim}: {note_s}")
    return "\n".join(lines)


def build_resume_grounding_block(
    education_details: "list[dict] | None" = None,
    cert_details: "list[str] | None" = None,
    experience_metadata: "dict | None" = None,
) -> str:
    """Build a structured grounding block for any LLM prompt that receives a resume.

    The LLM only sees the first ``RESUME_MAX_CHARS_LLM`` characters of the resume
    text.  For longer resumes, sections that appear near the end (Education,
    Certifications, some Skills) are silently dropped before the model ever reads
    them.  This function converts the deterministically extracted facts (which were
    computed from the full, untruncated resume) into a compact block that is
    appended to the prompt so the LLM never loses late-section information.

    Args:
        education_details: Output of ``extract_education_details(resume_text)``
        cert_details:      Output of ``extract_certifications_details(resume_text)``
        experience_metadata: Optional dict with candidate_years / match_type keys

    Returns:
        A non-empty string block (with leading newlines) when any data is present,
        or an empty string when nothing was detected — safe to concatenate always.
    """
    sections: list[str] = []

    # --- Education ---
    if education_details:
        edu_lines: list[str] = []
        for ed in education_details:
            parts: list[str] = [ed.get("degree", "")]
            if ed.get("field"):
                parts.append(f"in {ed['field']}")
            if ed.get("institution"):
                parts.append(f"from {ed['institution']}")
            if ed.get("raw_score") and ed.get("score_label"):
                parts.append(f"({ed['score_label']}: {ed['raw_score']})")
            edu_lines.append(" ".join(p for p in parts if p))
        sections.append(
            "EDUCATION (verified from full resume — never flag as missing):\n"
            + "\n".join(f"  • {line}" for line in edu_lines)
        )

    # --- Certifications ---
    if cert_details:
        sections.append(
            "CERTIFICATIONS (verified from full resume):\n"
            + "\n".join(f"  • {c}" for c in cert_details)
        )

    # --- Experience summary (when available) ---
    if experience_metadata:
        years = experience_metadata.get("candidate_years")
        seniority = experience_metadata.get("candidate_seniority")
        if years or seniority:
            exp_parts: list[str] = []
            if years:
                exp_parts.append(f"{years} years")
            if seniority and seniority != "unknown":
                exp_parts.append(f"{seniority}-level")
            sections.append(f"EXPERIENCE LEVEL (verified): {', '.join(exp_parts)}")

    if not sections:
        return ""

    return (
        "\n\n--- RESUME FACTS (parsed from full document, use these authoritatively) ---\n"
        + "\n".join(sections)
        + "\nIMPORTANT: Do NOT contradict or ignore the verified facts above. "
        "They were extracted from the complete resume before any truncation was applied.\n"
    )


async def parse_jd(jd_text: str) -> ParsedJD | None:
    """Use LLM to extract structured requirements from a JD."""
    schema = _schema_for(ParsedJD)
    data = await _chat_json(
        system_prompt=_JD_PARSER_SYSTEM,
        user_prompt=(
            "Parse this job description into structured requirements:\n\n"
            + jd_text[:JD_MAX_CHARS_LLM]
        ),
        schema=schema,
    )
    if data is None:
        return None
    return _parse_model(ParsedJD, data)


async def analyze_candidate(
    resume_text: str,
    jd_text: str,
    score_data: dict[str, Any],
) -> CandidateInsight | None:
    """Use LLM to generate deep insights for a single candidate."""
    schema = _schema_for(CandidateInsight)
    resume_facts_block = build_resume_grounding_block(
        education_details=score_data.get("education_details"),
        cert_details=score_data.get("cert_details"),
        experience_metadata=score_data.get("experience_metadata"),
    )
    prompt = (
        f"Analyze this candidate for the role.\n\n"
        f"--- JOB DESCRIPTION ---\n{jd_text[:JD_MAX_CHARS_LLM]}\n\n"
        f"--- RESUME ---\n{resume_text}\n\n"
        f"--- SCORING DATA ---\n"
        f"{_format_score_context_for_llm(score_data)}\n"
        + resume_facts_block
    )
    data = await _chat_json(
        system_prompt=_CANDIDATE_ANALYST_SYSTEM,
        user_prompt=prompt,
        schema=schema,
    )
    if data is None:
        return None
    return _parse_model(CandidateInsight, data)


async def generate_batch_summary(
    jd_text: str,
    candidates: list[dict[str, Any]],
) -> BatchSummary | None:
    """Use LLM to generate an executive summary across all candidates."""
    schema = _schema_for(BatchSummary)
    candidate_lines = []
    for i, c in enumerate(candidates[:15], 1):
        candidate_lines.append(
            f"{i}. {c['candidate']} \u2014 Score: {c['final_score']}%, "
            f"Decision: {c['decision']}, "
            f"Skills: {c['skill_match_ratio']}, "
            f"Matched: {', '.join(c.get('matched_skills', [])[:8])}"
        )

    prompt = (
        f"Generate an executive hiring summary.\n\n"
        f"--- JOB DESCRIPTION ---\n{jd_text[:JD_MAX_CHARS_LLM]}\n\n"
        f"--- CANDIDATES (ranked by score) ---\n"
        + "\n".join(candidate_lines)
    )
    data = await _chat_json(
        system_prompt=_BATCH_SUMMARY_SYSTEM,
        user_prompt=prompt,
        schema=schema,
    )
    if data is None:
        return None
    return _parse_model(BatchSummary, data)


_BACKWARD_PATTERNS = re.compile(
    r"\b(?:remov(?:e|ing)|delet(?:e|ing)|drop|cut\s+out|take\s+out|hide|omit|"
    r"shorten\s+(?:your|the)\s+resume|reduce\s+(?:your|the)\s+resume|"
    r"less\s+detail|simplify\s+by\s+removing)\b",
    re.IGNORECASE,
)


def _validate_suggestions_upward(
    suggestions: list[ResumeSuggestion],
) -> list[ResumeSuggestion]:
    """Filter out suggestions that could move the score backward.

    Backward means: telling the candidate to remove content, strip skills,
    or reduce detail — anything that would lower keyword density, section
    completeness, or evidence depth that the scoring engine rewards.
    """
    validated: list[ResumeSuggestion] = []
    for s in suggestions:
        combined = f"{s.improvement} {s.example or ''}"
        if _BACKWARD_PATTERNS.search(combined):
            if "replac" in combined.lower() or "rewrit" in combined.lower() or "add" in combined.lower():
                validated.append(s)
            else:
                continue
        else:
            validated.append(s)
    return validated


def _sanitize_resume_for_suggestions(text: str) -> str:
    """Strip PDF-extraction artefacts from resume text before passing to LLM.

    Some PDF fonts (ligature-heavy or embedded CIDFont) produce garbled tokens
    like "RECOGNIT", "ﬁ", "ﬂ", "ﬀ" etc. when extracted by PyMuPDF.  These
    confuse the LLM into echoing them as if they are real resume content and
    generating corrections that are nonsensical for the candidate.

    Strategy:
    - Replace Unicode ligature characters with their ASCII equivalents.
    - Remove standalone ALLCAPS tokens of 6+ chars that contain no vowels
      (classic sign of font-encoding garbage, e.g. "RECOGNIT", "PRGRMMR").
    - Collapse runs of whitespace introduced by the above replacements.
    """
    # Unicode ligature substitution (common PDF extraction artefacts)
    _LIGATURES = {
        "\ufb00": "ff",  # ﬀ
        "\ufb01": "fi",  # ﬁ
        "\ufb02": "fl",  # ﬂ
        "\ufb03": "ffi", # ﬃ
        "\ufb04": "ffl", # ﬄ
        "\ufb05": "st",  # ﬅ
        "\ufb06": "st",  # ﬆ
    }
    for lig, rep in _LIGATURES.items():
        text = text.replace(lig, rep)

    # Remove garbled ALL-CAPS tokens: 6+ chars, all uppercase, fewer than 2 vowels
    # These are almost never real words and are always PDF encoding artefacts.
    _VOWELS = set("AEIOUaeiou")
    _GARBAGE_TOKEN = re.compile(r"\b([A-Z]{6,})\b")
    def _drop_garbage(m: re.Match) -> str:
        token = m.group(1)
        vowel_count = sum(1 for c in token if c in _VOWELS)
        # Real English all-caps acronyms/abbreviations typically have at least
        # one vowel per 4 chars (e.g. "PYTHON" = 2/6, "SENIOR" = 3/6).
        # Garbage tokens like "RECOGNIT" or "PRGRMMR" have very few.
        if vowel_count < max(1, len(token) // 4):
            return ""
        return token
    text = _GARBAGE_TOKEN.sub(_drop_garbage, text)

    # Collapse multiple spaces/blank lines left behind
    text = re.sub(r"[ \t]{2,}", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


async def generate_resume_suggestions(
    resume_text: str,
    jd_text: str,
    score_data: dict[str, Any],
) -> ResumeSuggestions | None:
    """Use LLM to generate specific resume improvement suggestions.

    Returns up to 5 prioritized suggestions with actionable examples.
    Gracefully returns None if Ollama is unavailable or times out.
    """
    from app.config import FEATURE_IMPROVEMENT_SUGGESTIONS

    # Sanitize PDF artefacts before any LLM or fallback path sees the text.
    resume_text = _sanitize_resume_for_suggestions(resume_text)

    total_score = score_data.get("final_score", 0)
    sec = score_data.get("section_scores") or {}
    skills_score = sec.get("skills", 0)
    experience_score = sec.get("experience", 0)
    similarity_score = sec.get("similarity", 0)
    education_score = sec.get("education", 0)
    projects_score = sec.get("projects", 0)
    cert_score = sec.get("certifications", 0)
    matched_skills = score_data.get("matched_skills", [])[:10]
    missing_skills = score_data.get("missing_skills", [])[:10]

    # Prefer normalized scores for all prompt construction so the LLM reasons
    # about actual score contributions rather than raw pre-normalization values.
    sec_norm = score_data.get("section_scores_normalized") or sec
    skills_score_n = sec_norm.get("skills", skills_score)
    experience_score_n = sec_norm.get("experience", experience_score)
    similarity_score_n = sec_norm.get("similarity", similarity_score)
    education_score_n = sec_norm.get("education", education_score)
    projects_score_n = sec_norm.get("projects", projects_score)
    cert_score_n = sec_norm.get("certifications", cert_score)

    def _score_value(value: Any) -> float:
        try:
            return max(0.0, min(100.0, float(value)))
        except (TypeError, ValueError):
            return 0.0

    def _normalize_resume_suggestions(data: dict[str, Any] | None) -> ResumeSuggestions | None:
        if not data:
            return None

        engine_total = _score_value(total_score)
        # Ceiling matches scoring_engine: no resume can score 100 (always room to improve)
        _SCORE_CEILING = 99.0
        data["total_score"] = engine_total

        llm_potential = _score_value(data.get("potential_score", 0))
        gain = max(0.0, llm_potential - engine_total)
        gain = min(gain, 25.0)
        # For high-scoring resumes (≥90) ensure gain is at least 3 pts so the banner
        # shows meaningful headroom, not a trivial "+1 pt".
        min_gain = 3.0 if engine_total >= 90 else 4.0
        if gain < min_gain:
            # Count dimensions whose normalized score is below 60 (meaningful gap)
            low_dims = sum(
                1 for v in (skills_score_n, experience_score_n, similarity_score_n,
                            education_score_n, projects_score_n, cert_score_n)
                if _score_value(v) < 60
            )
            gain = max(min_gain, min(25.0, min_gain + len(missing_skills) * 2.0 + low_dims * 2.0))
        data["potential_score"] = min(_SCORE_CEILING, engine_total + gain)

        parsed = _parse_model(ResumeSuggestions, data)
        if parsed is None:
            return None

        parsed.total_score = engine_total
        if parsed.potential_score <= parsed.total_score:
            parsed.potential_score = min(_SCORE_CEILING, parsed.total_score + min_gain)

        parsed.suggestions = _validate_suggestions_upward(parsed.suggestions)

        return parsed

    def _extract_resume_bullets(text: str, max_count: int = 8) -> list[str]:
        """Extract distinct bullet points from the resume, weakest first."""
        bullet_pat = re.compile(r"(?:^|\n)\s*[\u2022\-\*\u25cf]\s*(.{20,200})", re.MULTILINE)
        raw = [c.strip() for c in bullet_pat.findall(text)]
        if not raw:
            line_pat = re.compile(r"(?:^|\n)(.{30,200})")
            raw = [c.strip() for c in line_pat.findall(text)]

        weak_verbs = {"helped", "assisted", "participated", "worked on", "responsible for",
                      "involved in", "contributed to", "supported", "was part of"}
        weak = [b for b in raw if any(wv in b.lower() for wv in weak_verbs)]
        other = [b for b in raw if b not in weak]
        ordered = weak + other
        seen: set[str] = set()
        unique: list[str] = []
        for b in ordered:
            key = b[:50].lower()
            if key not in seen:
                seen.add(key)
                unique.append(b)
            if len(unique) >= max_count:
                break
        return unique if unique else ["Worked on various projects and tasks"]

    def _est_lift(dim_name: str, dim_score: float, dim_weight: float) -> str:
        """Compute estimated point lift for improving a normalized dimension toward 100%.

        Uses normalized score and weight so the estimate reflects actual contribution
        to the final score rather than a pre-normalization raw value.
        """
        # How far the normalized dimension can still grow (capped at headroom to 100)
        gap = max(0, 100 - dim_score)
        # Realistic improvement: assume the suggestion closes ~30% of the gap
        realistic_improvement = gap * 0.30
        pts = round(realistic_improvement * dim_weight, 0)
        pts = max(1, min(pts, 8))
        return f"+{int(pts)} pts ({dim_name})"

    def _fallback_resume_suggestions() -> ResumeSuggestions:
        suggestions: list[ResumeSuggestion] = []
        resume_lower = resume_text.lower()
        bullets = _extract_resume_bullets(resume_text)
        bullet_idx = 0

        def _next_bullet() -> str:
            nonlocal bullet_idx
            b = bullets[bullet_idx % len(bullets)]
            bullet_idx += 1
            return b

        # Sort by weighted contribution (normalized score × weight) so the weakest
        # dimension in terms of actual score impact is addressed first.
        low_sections = sorted(
            (
                ("skills", _score_value(skills_score_n), 0.33),
                ("experience", _score_value(experience_score_n), 0.24),
                ("similarity", _score_value(similarity_score_n), 0.19),
                ("projects", _score_value(projects_score_n), 0.11),
                ("education", _score_value(education_score_n), 0.08),
                ("certifications", _score_value(cert_score_n), 0.05),
            ),
            key=lambda item: item[1] * item[2],  # weighted contribution
        )

        # --- Suggestion 1: Missing JD skills (critical) ---
        if missing_skills:
            ms_top = missing_skills[:3]
            ms_all = ", ".join(missing_skills[:5])
            bullet = _next_bullet()
            suggestions.append(
                ResumeSuggestion(
                    category="tech_stack",
                    current_state=(
                        f"The JD requires {ms_all} but these don't appear in your resume. "
                        f"Skills score is {_score_value(skills_score):.0f}% — the highest-weighted dimension (33%)."
                    ),
                    improvement=(
                        f"Weave {', '.join(ms_top)} into your strongest bullets and add them to your skills section. "
                        "Use the JD's exact phrasing so ATS parsers match them."
                    ),
                    before_text=bullet,
                    after_text=(
                        f"{bullet.rstrip('.')}, utilizing {ms_top[0]}"
                        + (f" and {ms_top[1]}" if len(ms_top) > 1 else "")
                        + " to improve system reliability by 35% and reduce deployment time from days to hours"
                    ),
                    estimated_lift=_est_lift("skills", _score_value(skills_score), 0.33),
                    priority="critical",
                    jd_relevance=(
                        f"The JD explicitly lists {', '.join(ms_top)} as requirements. "
                        "Each missing keyword directly lowers the skills dimension, which carries 33% of the final score."
                    ),
                )
            )

        # --- Suggestion 2: Weakest dimension (skip projects if it gets its own suggestion below) ---
        will_add_projects_section = "project" not in resume_lower
        weak_dim_candidates = [
            (n, s, w) for n, s, w in low_sections
            if s < 75 and not (n == "projects" and will_add_projects_section)
        ]
        if weak_dim_candidates:
            w_name, w_score, w_weight = weak_dim_candidates[0]
            bullet = _next_bullet()
            section_label = w_name.replace("_", " ").title()
            suggestions.append(
                ResumeSuggestion(
                    category="weak_skill" if w_name in {"skills", "similarity"} else "missing_section",
                    current_state=(
                        f"Your {section_label} dimension scores {int(round(w_score))}% — "
                        f"{'the lowest area' if w_name == low_sections[0][0] else 'a weak area'}. "
                        f"It carries {w_weight:.0%} of the final score."
                    ),
                    improvement=(
                        f"Add specific, quantified evidence for {section_label.lower()}. "
                        "Name the exact tools, scope, and measurable outcome."
                    ),
                    before_text=bullet,
                    after_text=(
                        f"Led {section_label.lower()}-focused initiative across 3 teams, "
                        f"implementing {missing_skills[0] if missing_skills else 'modern'} solutions "
                        "that reduced operational costs by 25% and improved delivery velocity by 40%"
                    ),
                    estimated_lift=_est_lift(w_name, w_score, w_weight),
                    priority="high",
                    jd_relevance=(
                        f"Raising {section_label} from {int(round(w_score))}% toward 70% "
                        f"adds ~{max(1, int((70 - w_score) * w_weight))} points to your overall score."
                    ),
                )
            )

        # --- Suggestion 3: Add metrics ---
        has_metrics = bool(re.search(r"\d+%|\$\d|\d+x|\d+\s*(?:million|thousand|users|customers|req)", resume_lower))
        if not has_metrics:
            bullet = _next_bullet()
            suggestions.append(
                ResumeSuggestion(
                    category="action_verbs",
                    current_state=(
                        "No quantified impact detected — no percentages, dollar figures, or scale numbers. "
                        "Hiring managers scan for numbers first."
                    ),
                    improvement=(
                        "Add a metric to your top 3 bullets: % improvement, $ saved, users served, "
                        "time reduced, or team size. Even estimates are better than nothing."
                    ),
                    before_text=bullet,
                    after_text=(
                        f"{bullet.rstrip('.')}, reducing processing time by 40% "
                        "and handling 10K+ daily requests with 99.9% uptime"
                    ),
                    estimated_lift=_est_lift("similarity", _score_value(similarity_score), 0.19),
                    priority="high",
                    jd_relevance=(
                        "Quantified impact lifts both the Similarity dimension (numbers match JD language patterns) "
                        "and recruiter confidence during a fast resume scan."
                    ),
                )
            )

        # --- Suggestion 4: Projects section ---
        if "project" not in resume_lower:
            stack = ", ".join(missing_skills[:2]) if missing_skills else "the JD's core stack"
            suggestions.append(
                ResumeSuggestion(
                    category="missing_section",
                    current_state=(
                        f"No 'Projects' section detected. Projects dimension is {_score_value(projects_score):.0f}% (weight 11%)."
                    ),
                    improvement=(
                        "Add a 'Projects' heading with 2-3 entries. Each names the stack, "
                        "what you built, and a quantified outcome."
                    ),
                    before_text="(no projects section)",
                    after_text=(
                        f"Projects\n"
                        f"- Built a production {missing_skills[0] if missing_skills else 'full-stack'} "
                        f"application with CI/CD pipeline, serving 5K+ monthly users with 99.9% uptime\n"
                        f"- Developed internal tool using {stack} that automated manual reporting, "
                        "saving the team 15 hours per week"
                    ),
                    estimated_lift=_est_lift("projects", _score_value(projects_score), 0.11),
                    priority="medium",
                    jd_relevance=(
                        f"The Projects dimension is at {_score_value(projects_score):.0f}%. "
                        f"Adding a project section using {stack} directly addresses missing keywords."
                    ),
                )
            )

        # --- Suggestion 5: Front-load keywords in summary ---
        if len(matched_skills) < 5:
            matched_str = ", ".join(matched_skills[:3]) if matched_skills else "your strongest skills"
            missing_list = [s for s in missing_skills[:3]]
            missing_str = ", ".join(missing_list) if missing_list else "key JD requirements"
            suggestions.append(
                ResumeSuggestion(
                    category="formatting",
                    current_state=(
                        f"Only {len(matched_skills)} skill(s) clearly match the JD in the first 200 words. "
                        "ATS parsers weight the top of the resume more heavily."
                    ),
                    improvement=(
                        "Rewrite your summary/headline to front-load the JD's top keywords "
                        "in the first 2-3 lines."
                    ),
                    before_text="Experienced software engineer with diverse background",
                    after_text=(
                        f"{missing_str} engineer with hands-on {matched_str} experience, "
                        "delivering production systems at scale with measurable business impact"
                    ),
                    estimated_lift=_est_lift("similarity", _score_value(similarity_score), 0.19),
                    priority="medium",
                    jd_relevance=(
                        "Front-loading keywords improves ATS rank position and lifts the "
                        "Similarity dimension by aligning your language with the JD."
                    ),
                )
            )

        # --- Fill remaining slots with bullet rewrites ---
        while len(suggestions) < 5:
            bullet = _next_bullet()
            slot_dim = low_sections[len(suggestions) % len(low_sections)]
            dim_label = slot_dim[0].replace("_", " ")
            suggestions.append(
                ResumeSuggestion(
                    category="action_verbs",
                    current_state=(
                        f"This bullet lacks ownership language and quantified impact. "
                        f"It doesn't clearly contribute to the {dim_label} dimension."
                    ),
                    improvement=(
                        "Rewrite using the formula: [Strong Verb] + [What] + [Using What] + [Result with Number]. "
                        f"Align with {dim_label} scoring."
                    ),
                    before_text=bullet,
                    after_text=(
                        f"Architected and delivered {dim_label}-aligned solution "
                        f"using {missing_skills[0] if missing_skills else 'modern tooling'}, "
                        "achieving 30% efficiency gain and reducing manual overhead by 20 hours/month"
                    ),
                    estimated_lift=_est_lift(slot_dim[0], slot_dim[1], slot_dim[2]),
                    priority="medium",
                    jd_relevance=(
                        f"Strengthening this bullet with JD keywords and metrics lifts the "
                        f"{dim_label} dimension ({slot_dim[1]:.0f}%, weight {slot_dim[2]:.0%})."
                    ),
                )
            )

        low_scores_count = sum(1 for _, s, _ in low_sections[:3] if s < 70)
        estimated_gain = max(5.0, min(20.0, 5.0 + len(missing_skills) * 1.5 + low_scores_count * 2.0))
        engine_total = _score_value(total_score)
        # Ceiling 99: consistent with scoring_engine — no resume is ever perfect
        potential_score = min(99.0, engine_total + estimated_gain)

        return ResumeSuggestions(
            total_score=engine_total,
            potential_score=potential_score,
            suggestions=suggestions[:5],
            summary=(
                f"Your biggest score drag is {low_sections[0][0]} at {low_sections[0][1]:.0f}% "
                f"and {low_sections[1][0]} at {low_sections[1][1]:.0f}% — target these first. "
                + (f"Adding {', '.join(missing_skills[:3])} and quantifying your impact "
                   "will create the largest upward lift."
                   if missing_skills else
                   "Quantifying your impact and strengthening weak bullets with ownership verbs "
                   "will create the largest upward lift.")
            ),
        )

    if not FEATURE_IMPROVEMENT_SUGGESTIONS:
        return _fallback_resume_suggestions()

    schema = _schema_for(ResumeSuggestions)

    # Use normalized scores so "weakest dimension" is ranked by actual score
    # contribution (normalized × weight), not by raw pre-normalization value.
    sorted_dims = sorted(
        [("skills", skills_score_n, 0.33), ("experience", experience_score_n, 0.24),
         ("similarity", similarity_score_n, 0.19), ("projects", projects_score_n, 0.11),
         ("education", education_score_n, 0.08), ("certifications", cert_score_n, 0.05)],
        key=lambda d: _score_value(d[1]) * d[2],  # sort by weighted contribution
    )
    weakest_dims = ", ".join(
        f"{n} ({_score_value(s):.0f}% normalized, contributes {_score_value(s) * w:.1f} pts to overall)"
        for n, s, w in sorted_dims[:3]
    )

    resume_facts_block = build_resume_grounding_block(
        education_details=score_data.get("education_details"),
        cert_details=score_data.get("cert_details"),
        experience_metadata=score_data.get("experience_metadata"),
    )

    # When the score is already very high, tailor the framing so the LLM gives
    # precise, non-trivial suggestions rather than generic "add keyword X" advice.
    _score_val = _score_value(total_score)
    _high_score_note = ""
    if _score_val >= 90:
        _high_score_note = (
            "\n## HIGH-SCORE CONTEXT\n"
            f"This resume already scores {_score_val:.0f}%. Major gaps don't exist. "
            "Focus only on:\n"
            "  a) Specific JD keywords that are genuinely absent from the resume text (missing_skills list).\n"
            "  b) Bullets that lack quantified metrics (no %, $, or count numbers).\n"
            "  c) Phrasing that could be sharpened for ATS keyword density.\n"
            "Do NOT invent problems. Do NOT suggest fixing text that is already correct. "
            "Do NOT reference parsed/garbled text — only reference text that clearly appears in the resume.\n"
        )

    # Pre-compute weighted contributions (normalized score × weight) for the prompt.
    # This is the actual points each dimension adds to the final score.
    _wc = {
        "skills":          (_score_value(skills_score_n),      0.33),
        "experience":      (_score_value(experience_score_n),  0.24),
        "similarity":      (_score_value(similarity_score_n),  0.19),
        "projects":        (_score_value(projects_score_n),    0.11),
        "education":       (_score_value(education_score_n),   0.08),
        "certifications":  (_score_value(cert_score_n),        0.05),
    }
    def _contrib(dim: str) -> str:
        s, w = _wc[dim]
        return f"{s:.0f}% → {s * w:.1f} pts"

    prompt = f"""Analyze this resume against the JD. Return exactly 5 surgical improvements as JSON.

--- JOB DESCRIPTION ---
{jd_text[:JD_MAX_CHARS_LLM]}

--- RESUME ---
{resume_text}

--- CURRENT SCORES (normalized score → weighted pts out of 100) ---
Overall: {total_score}%
  Skills (wt 33%):         {_contrib("skills")}
  Experience (wt 24%):     {_contrib("experience")}
  Similarity (wt 19%):     {_contrib("similarity")}
  Projects (wt 11%):       {_contrib("projects")}
  Education (wt 8%):       {_contrib("education")}
  Certifications (wt 5%):  {_contrib("certifications")}
Weakest by weighted contribution: {weakest_dims}
Matched skills: {', '.join(matched_skills) or 'none'}
Missing skills: {', '.join(missing_skills) or 'none'}

{_format_score_context_for_llm(score_data)}{resume_facts_block}{_high_score_note}

## JSON FORMAT (follow exactly)
```json
{{
  "total_score": {total_score},
  "potential_score": <{total_score} + 3 to 15, max 99>,
  "summary": "<2-sentence roadmap targeting weakest dimensions>",
  "suggestions": [
    {{
      "category": "tech_stack",
      "current_state": "<describe the problem: what is weak or missing>",
      "improvement": "<1-2 sentence instruction: what to do>",
      "before_text": "<EXACT verbatim text from the resume OR '(missing)' if section doesn't exist — NEVER invent or paraphrase>",
      "after_text": "<COMPLETE rewritten text the candidate should use — not instructions, actual text>",
      "estimated_lift": "+3 pts (skills)",
      "priority": "critical",
      "jd_relevance": "<why this matters for THIS JD, referencing a specific requirement>"
    }}
  ]
}}
```

## RULES
1. before_text: copy EXACT verbatim text from the resume, or write "(missing)" if the section doesn't exist.
   - NEVER paraphrase, reconstruct, or fix garbled text — if you see it, quote it exactly or skip it.
   - NEVER reference a university name or company name unless it clearly appears in the resume text above.
2. after_text: write the COMPLETE replacement text. Not an instruction — the actual rewritten bullet/section they paste in.
   - Must have MORE JD keywords, stronger verbs, and quantified metrics than before_text.
3. estimated_lift: estimate how many points this single change adds and which dimension it lifts.
   Format: "+N pts (dimension)" e.g. "+4 pts (skills)", "+2 pts (experience)", "+3 pts (similarity)"
4. NEVER suggest removing content. Always ADD, STRENGTHEN, or REFRAME.
5. total_score must be exactly {total_score}.
6. Target the weakest high-weight dimensions first for maximum score lift.
7. category: one of missing_section, weak_skill, action_verbs, formatting, tech_stack.
8. priority: critical > high > medium > low. At least one must be critical if missing_skills exist.
9. potential_score must be > total_score and ≤ 99. Never 100.

Return valid JSON only."""

    try:
        data = await _chat_json(
            system_prompt=_RESUME_SUGGESTIONS_SYSTEM,
            user_prompt=prompt,
            schema=schema,
            num_predict=OLLAMA_NUM_PREDICT_JSON_HEAVY,
        )
    except Exception:
        logger.warning("Structured resume suggestions failed; using deterministic fallback", exc_info=True)
        return _fallback_resume_suggestions()

    parsed = _normalize_resume_suggestions(data)
    if parsed is not None and parsed.suggestions:
        return parsed
    return _fallback_resume_suggestions()


# ---------------------------------------------------------------------------
# Phase 3: Career Trajectory Analysis
# ---------------------------------------------------------------------------

_VALID_TIER_PROGRESSIONS = frozenset({
    "startup_to_enterprise", "enterprise_to_startup", "consistent", "varied",
})
_VALID_PROGRESSION_TYPES = frozenset({
    "rapid_growth", "steady_growth", "lateral", "stagnant", "declining", "pivot", "mixed",
})
_VALID_RISK_LEVELS = frozenset({"low", "medium", "high"})


class CareerTrajectory(BaseModel):
    """Analysis of career progression patterns."""
    progression_type: str = Field(
        default="mixed",
        description="One of: rapid_growth, steady_growth, lateral, stagnant, declining, pivot",
    )
    average_tenure_months: int = Field(
        default=24,
        description="Average time at each company",
    )

    @field_validator("average_tenure_months", mode="before")
    @classmethod
    def _parse_tenure(cls, v: Any) -> int:
        if isinstance(v, str):
            # Extract first number from strings like "29 months (TechCorp: 48...)"
            import re
            match = re.search(r'\d+', v)
            if match:
                return int(match.group())
            return 0
        if v is None:
            return 0
        return int(v)

    job_hopping_risk: str = Field(
        default="medium",
        description="One of: low, medium, high based on tenure patterns",
    )
    employment_gaps: list[dict] = Field(
        default_factory=list,
        description="List of gaps: {start, end, months, explanation_found}",
    )

    @field_validator("employment_gaps", mode="before")
    @classmethod
    def _filter_bogus_gaps(cls, v: Any) -> list[dict]:
        """Discard hallucinated gaps (e.g. current role mistaken for a gap)."""
        if not isinstance(v, list):
            return []
        import re as _re
        cleaned: list[dict] = []
        for item in v:
            if not isinstance(item, dict):
                continue
            end = str(item.get("end", "")).strip().lower()
            start = str(item.get("start", "")).strip().lower()
            if "present" in end or "present" in start or "current" in end:
                continue
            if not _re.search(r"\d{4}", start) or not _re.search(r"\d{4}", end):
                continue
            try:
                months = int(item.get("months", 0))
            except (ValueError, TypeError):
                months = 0
            if months < 3:
                continue
            cleaned.append(item)
        return cleaned

    title_progression: list[str] = Field(
        default_factory=list,
        description="Sequence of job titles from oldest to newest",
    )

    @field_validator("title_progression", mode="before")
    @classmethod
    def _clean_title_progression(cls, v: Any) -> list[str]:
        if not isinstance(v, list):
            return []
        seen: set[str] = set()
        cleaned: list[str] = []
        for item in v:
            t = str(item).strip()
            if t and t.lower() not in seen:
                seen.add(t.lower())
                cleaned.append(t)
        return cleaned

    company_tier_progression: str = Field(
        default="consistent",
        description="One of: startup_to_enterprise, enterprise_to_startup, consistent, varied",
    )

    @field_validator("company_tier_progression", mode="before")
    @classmethod
    def _normalize_tier_progression(cls, v: Any) -> str:
        s = str(v).strip().lower().replace(" ", "_").replace("-", "_")
        if s in _VALID_TIER_PROGRESSIONS:
            return s
        return "consistent"

    @field_validator("progression_type", mode="before")
    @classmethod
    def _normalize_progression_type(cls, v: Any) -> str:
        s = str(v).strip().lower().replace(" ", "_").replace("-", "_")
        if s in _VALID_PROGRESSION_TYPES:
            return s
        return "mixed"

    pivot_detected: bool = Field(
        default=False,
        description="True if career pivot detected",
    )
    pivot_details: str | None = Field(
        default=None,
        description="Details of career pivot if detected",
    )
    growth_potential: str = Field(
        default="medium",
        description="One of: high, medium, low based on trajectory",
    )

    @field_validator("growth_potential", "job_hopping_risk", mode="before")
    @classmethod
    def _normalize_risk_level(cls, v: Any) -> str:
        s = str(v).strip().lower()
        if s in _VALID_RISK_LEVELS:
            return s
        return "medium"

    red_flags: list[str] = Field(
        default_factory=list,
        description="Career red flags detected",
    )
    green_flags: list[str] = Field(
        default_factory=list,
        description="Positive career indicators",
    )


_CAREER_TRAJECTORY_SYSTEM = (
    "You are a senior technical recruiter with 15+ years experience. "
    "Analyze career progression patterns from resumes.\n\n"
    "CRITICAL: Be SPECIFIC — cite actual company names, titles, dates, and durations. "
    "Never produce generic observations. Every flag must reference concrete resume content "
    "and explain what it means for a hiring decision.\n"
    "Context matters — short tenures at startups differ from enterprises."
)


async def analyze_career_trajectory(
    resume_text: str,
) -> CareerTrajectory | None:
    """Analyze career progression patterns and detect red flags.

    Args:
        resume_text: The candidate's resume text

    Returns:
        CareerTrajectory with progression analysis
    """
    from app.config import ENABLE_CAREER_TRAJECTORY

    if not ENABLE_CAREER_TRAJECTORY or not ai_enabled():
        return None

    schema = _schema_for(CareerTrajectory)

    prompt = f"""Analyze this resume for career trajectory patterns.

## Resume
{resume_text}

## Analysis Required

IMPORTANT: Use ONLY facts explicitly stated in the resume. Do NOT infer promotions,
fabricate dates, or guess company names that aren't written in the resume text.

### 1. PROGRESSION TYPE
- rapid_growth: Quick promotions, increasing responsibility faster than normal
- steady_growth: Normal 2-3 year cycles with promotions
- lateral: Same-level moves between companies
- stagnant: Same role for 5+ years without growth
- declining: Decreasing responsibility over time
- pivot: Career change to different field
If the resume shows only ONE employer, judge based on title changes within that company.
If only ONE title is visible, this is likely steady_growth or stagnant, NOT rapid_growth.

### 2. JOB HOPPING ANALYSIS
- Average tenure at each company (single-employer = low risk by definition)
- Pattern: Are short tenures at startups (acceptable) or enterprises (concerning)?
- Risk assessment: low (>3yr avg or single employer), medium (2-3yr), high (<2yr)

### 3. EMPLOYMENT GAPS
A gap is ONLY the idle period BETWEEN the end date of one role and the start date of the next.
- Only report gaps > 3 months between two DISTINCT employers/positions.
- A current role (ending "Present") is NOT a gap.
- Continuous employment at one company = EMPTY list [].
- Note if a real gap is explained (education, travel, family, personal projects).

### 4. TITLE PROGRESSION
Extract EVERY distinct job title from the resume, ordered OLDEST to NEWEST.
Include the company name with each, e.g. ["Junior Dev @ StartupX", "Senior Dev @ BigCorp"].
If the resume lists multiple roles at the SAME company, list each separately.
NEVER return a single-element list unless the resume truly shows only one role ever held.

### 5. COMPANY TIER PROGRESSION
Must be exactly one of: startup_to_enterprise, enterprise_to_startup, consistent, varied.
Use "consistent" when all roles are at the same company or same tier.

### 6. PIVOT DETECTION
- Has the candidate changed career direction significantly?
- If yes, what was the pivot and when?

### 7. GROWTH POTENTIAL
- high: Strong upward trajectory, learning new skills
- medium: Steady progression
- low: Stagnant or declining

### 8. RED FLAGS
RULES:
1. Each flag MUST cite a specific fact from the resume (company, title, date, duration).
2. Do NOT speculate about skills the candidate "might" lack or "may need".
3. Do NOT restate what the progression_type or other stats already convey.
4. Only flag things a hiring manager should PROBE in an interview.
BAD: "Focus is highly specialized which requires the next role to align" (speculation)
BAD: "Specialization may limit opportunities" (generic, not from resume)
GOOD: "No people-management experience visible despite 8 years tenure — probe if role requires team leadership"
GOOD: "All 3 roles focused on internal tooling at Acme Corp — no customer-facing product experience"

### 9. GREEN FLAGS
RULES:
1. Each flag MUST cite a specific fact from the resume.
2. Do NOT restate the progression_type (e.g. don't say "rapid growth" as a flag).
3. Do NOT restate generic traits ("demonstrated leadership") — cite the evidence.
GOOD: "Promoted from SDE-II to Principal Engineer at Walmart in 4 years (2019→2023)"
GOOD: "Built RASCE from zero to production in <12 months — shows end-to-end ownership"

Respond with valid JSON only."""

    data = await _chat_json(
        system_prompt=_CAREER_TRAJECTORY_SYSTEM,
        user_prompt=prompt,
        schema=schema,
    )
    if data is None:
        return None
    return _parse_model(CareerTrajectory, data)


# ---------------------------------------------------------------------------
# Phase 3: Deep Resume Analysis (replaces 5 separate enricher LLM calls)
# ---------------------------------------------------------------------------

class DeepExperienceAnalysis(BaseModel):
    """Embedded experience analysis within DeepResumeAnalysis."""
    relevance_score: float = Field(default=50.0)
    total_years: float = Field(default=0.0)
    relevant_years: float = Field(default=0.0)
    career_progression: str = Field(default="mixed")
    domain_match: str = Field(default="different")
    technology_currency: str = Field(default="current")
    highlights: list[str] = Field(default_factory=list)
    concerns: list[str] = Field(default_factory=list)
    verdict: str = Field(default="adequate")

    @field_validator("total_years", "relevant_years", "relevance_score", mode="before")
    @classmethod
    def _coerce_float(cls, v: Any) -> float:
        if isinstance(v, (int, float)):
            return float(v)
        import re as _re
        s = _re.sub(r"[+~\s%].*$", "", str(v).strip())
        try:
            return float(s)
        except ValueError:
            return 0.0


class DeepAchievementItem(BaseModel):
    statement: str = Field(default="")
    action_verb: str = Field(default="")
    action_strength: str = Field(default="moderate")
    impact_level: str = Field(default="individual")
    jd_relevance: float = Field(default=0.5)

    @field_validator("jd_relevance", mode="before")
    @classmethod
    def _coerce_relevance(cls, v: Any) -> float:
        if isinstance(v, (int, float)):
            return float(v)
        s = str(v).strip()
        if not s or s.lower() in {"n/a", "na", "none", "null"}:
            return 0.0
        try:
            return float(s)
        except ValueError:
            return 0.0


class DeepAchievementAnalysis(BaseModel):
    achievements: list[DeepAchievementItem] = Field(default_factory=list)
    quantified_count: int = Field(default=0)
    average_impact: float = Field(default=0.0)
    top_achievements: list[str] = Field(default_factory=list)
    weak_statements: list[str] = Field(default_factory=list)
    score: float = Field(default=0.0)

    @model_validator(mode="before")
    @classmethod
    def _coerce_fields(cls, data: Any) -> Any:
        if not isinstance(data, dict):
            return data
        import re as _re
        qc = data.get("quantified_count")
        if isinstance(qc, str):
            m = _re.search(r"-?\d+", qc.strip())
            data["quantified_count"] = int(m.group()) if m else 0
        return data


class DeepCandidateFit(BaseModel):
    technical_fit: float = Field(default=50.0)
    experience_fit: float = Field(default=50.0)
    domain_fit: float = Field(default=50.0)
    seniority_fit: float = Field(default=50.0)
    culture_indicators: float = Field(default=50.0)
    growth_potential: float = Field(default=50.0)
    overall_fit: float = Field(default=50.0)
    hiring_confidence: str = Field(default="medium")
    strongest_dimensions: list[str] = Field(default_factory=list)
    weakest_dimensions: list[str] = Field(default_factory=list)
    unique_value: str = Field(default="")
    risk_factors: list[str] = Field(default_factory=list)
    ideal_for_role: bool = Field(default=False)
    compensation_tier: str = Field(default="mid")

    @field_validator(
        "technical_fit",
        "experience_fit",
        "domain_fit",
        "seniority_fit",
        "culture_indicators",
        "growth_potential",
        "overall_fit",
        mode="before",
    )
    @classmethod
    def _coerce_fit_score(cls, v: Any) -> float:
        if isinstance(v, (int, float)):
            return float(v)
        s = str(v).strip().lower()
        if not s or s in {"n/a", "na", "none", "null"}:
            return 0.0
        if s == "high":
            return 85.0
        if s == "medium":
            return 60.0
        if s == "low":
            return 30.0
        s = s.replace("%", "")
        try:
            return float(s)
        except ValueError:
            return 0.0


class DeepRedFlagItem(BaseModel):
    flag: str = Field(default="")
    evidence: str = Field(default="")
    severity: str = Field(default="high")
    recommendation: str = Field(default="")


class DeepWarningFlagItem(BaseModel):
    flag: str = Field(default="")
    evidence: str = Field(default="")
    questions_to_ask: list[str] = Field(default_factory=list)


class DeepRedFlags(BaseModel):
    critical_flags: list[DeepRedFlagItem] = Field(default_factory=list)
    warning_flags: list[DeepWarningFlagItem] = Field(default_factory=list)
    green_flags: list[str] = Field(default_factory=list)
    risk_level: str = Field(default="medium")
    proceed_with_interview: bool = Field(default=True)
    interview_focus_areas: list[str] = Field(default_factory=list)


class DeepResumeAnalysis(BaseModel):
    """All five enricher analyses returned in a single LLM call."""
    experience: DeepExperienceAnalysis = Field(default_factory=DeepExperienceAnalysis)
    achievements: DeepAchievementAnalysis = Field(default_factory=DeepAchievementAnalysis)
    trajectory: CareerTrajectory = Field(default_factory=CareerTrajectory)
    fit: DeepCandidateFit = Field(default_factory=DeepCandidateFit)
    red_flags: DeepRedFlags = Field(default_factory=DeepRedFlags)


_DEEP_ANALYSIS_SYSTEM = (
    "You are a senior technical recruiter with 15+ years experience performing a comprehensive "
    "resume review. Analyze the candidate across five dimensions simultaneously and return "
    "structured JSON.\n\n"
    "CRITICAL RULES:\n"
    "- Be SPECIFIC: always cite actual company names, job titles, dates, and technologies "
    "from the resume. Never produce generic observations like 'consistent progression' — "
    "instead say 'promoted from X to Y at Company in N years'.\n"
    "- Be EVIDENCE-BASED: every flag, concern, or highlight must reference something "
    "concrete from the resume text.\n"
    "- Be JD-AWARE: evaluate the candidate through the lens of the specific job description "
    "provided — a 'red flag' is only a flag if it matters for this particular role.\n"
    "- Be ACTIONABLE: flags should help a hiring manager decide what to probe in an "
    "interview, not just state obvious facts.\n"
    "- Be fair but thorough."
)

_DEEP_ANALYSIS_SCHEMA = {
    "experience": {
        "relevance_score": "0-100",
        "total_years": "float",
        "relevant_years": "float",
        "career_progression": "ascending|lateral|descending|mixed",
        "domain_match": "exact|related|different",
        "technology_currency": "current|slightly_outdated|outdated",
        "highlights": ["top 3 relevant experiences"],
        "concerns": ["gaps or concerns"],
        "verdict": "strong|adequate|weak",
    },
    "achievements": {
        "achievements": [
            {
                "statement": "achievement text",
                "action_verb": "verb used",
                "action_strength": "weak|moderate|strong",
                "impact_level": "individual|team|department|company|industry",
                "jd_relevance": "0-1 float",
            }
        ],
        "quantified_count": "integer",
        "average_impact": "0-100 float",
        "top_achievements": ["top 3 statements"],
        "weak_statements": ["statements needing stronger verbs"],
        "score": "0-100 float",
    },
    "trajectory": {
        "progression_type": "rapid_growth|steady_growth|lateral|stagnant|declining|pivot",
        "average_tenure_months": "integer calculated from resume dates",
        "job_hopping_risk": "low|medium|high (single employer = low)",
        "employment_gaps": [],
        "title_progression": ["Role @ Company (oldest)", "Role @ Company", "Role @ Company (newest)"],
        "company_tier_progression": "startup_to_enterprise|enterprise_to_startup|consistent|varied",
        "pivot_detected": "bool",
        "pivot_details": "string or null",
        "growth_potential": "high|medium|low",
        "red_flags": ["each cites specific resume facts — never generic speculation"],
        "green_flags": ["each cites specific resume facts — never restates stat cards"],
    },
    "fit": {
        "technical_fit": "0-100",
        "experience_fit": "0-100",
        "domain_fit": "0-100",
        "seniority_fit": "0-100",
        "culture_indicators": "0-100",
        "growth_potential": "0-100",
        "overall_fit": "0-100",
        "hiring_confidence": "high|medium|low",
        "strongest_dimensions": ["top 2 strengths"],
        "weakest_dimensions": ["top 2 weaknesses"],
        "unique_value": "what they uniquely bring",
        "risk_factors": ["risks"],
        "ideal_for_role": "bool",
        "compensation_tier": "entry|mid|senior|executive",
    },
    "red_flags": {
        "critical_flags": [{"flag": "name", "evidence": "evidence", "severity": "critical|high", "recommendation": "action"}],
        "warning_flags": [{"flag": "name", "evidence": "evidence", "questions_to_ask": ["questions"]}],
        "green_flags": ["positive signals"],
        "risk_level": "low|medium|high|critical",
        "proceed_with_interview": "bool",
        "interview_focus_areas": ["areas"],
    },
}


async def analyze_resume_deep(
    resume_text: str,
    jd_text: str,
    role: str | None = None,
    *,
    inferred_skill_names: list[str] | None = None,
    education_details: "list[dict] | None" = None,
    cert_details: "list[str] | None" = None,
    experience_metadata: "dict | None" = None,
) -> DeepResumeAnalysis | None:
    """Run all five enricher analyses in a single LLM call.

    Replaces five separate asyncio.gather() calls (experience, achievements,
    career trajectory, candidate fit, red flags) with one round-trip to the
    model. On local Ollama this cuts model queue occupancy from 5 slots to 1,
    dramatically reducing wall-clock time per resume.

    When ``inferred_skill_names`` is provided, a suppression block is injected
    into the prompt so the LLM does not flag those skills as missing — they have
    already been credited via transferable inference or outcome-language evidence.
    """
    if not ai_enabled():
        return None

    role_ctx = f" for the {role} role" if role else ""

    suppression_block = ""
    if inferred_skill_names:
        skill_list = ", ".join(inferred_skill_names)
        suppression_block = (
            f"\nCREDITED INFERRED SKILLS — do NOT flag these as missing or concerning:\n"
            f"{skill_list}\n"
            "These skills have been credited via transferable experience or outcome evidence.\n"
        )

    resume_facts_block = build_resume_grounding_block(
        education_details=education_details,
        cert_details=cert_details,
        experience_metadata=experience_metadata,
    )

    prompt = (
        f"Perform a comprehensive analysis of this candidate{role_ctx}. "
        "Return a single JSON object with five top-level keys: "
        "experience, achievements, trajectory, fit, red_flags.\n\n"
        f"## Job Description\n{jd_text[:JD_MAX_CHARS_LLM]}\n\n"
        f"## Resume\n{resume_text}\n\n"
        + resume_facts_block
        + suppression_block
        + "## Instructions\n\n"
        "### experience — relevance of work history to this JD\n"
        "  relevance_score (0-100), total_years (float), relevant_years (float), "
        "  career_progression (ascending|lateral|descending|mixed), "
        "  domain_match (exact|related|different), "
        "  technology_currency (current|slightly_outdated|outdated), "
        "  highlights (top 3 relevant experiences), concerns (gaps/concerns), "
        "  verdict (strong|adequate|weak)\n\n"
        "### achievements — quantified impact statements\n"
        "  achievements (list, each: statement, action_verb, action_strength weak|moderate|strong, "
        "  impact_level individual|team|department|company|industry, jd_relevance 0-1), "
        "  quantified_count (int), average_impact (0-100), "
        "  top_achievements (top 3 statements), weak_statements (list), score (0-100)\n\n"
        "### trajectory — career progression pattern\n"
        "  IMPORTANT: Use ONLY facts explicitly stated in the resume. "
        "Do NOT infer promotions, fabricate dates, or guess company names.\n\n"
        "  progression_type (rapid_growth|steady_growth|lateral|stagnant|declining|pivot):\n"
        "    - rapid_growth = promoted faster than industry norm (e.g. junior→senior in <3 yrs)\n"
        "    - steady_growth = normal 2-4 year cycles with clear upward moves\n"
        "    - lateral = same-level moves across companies without seniority gain\n"
        "    - stagnant = same role/level for 5+ years without visible growth\n"
        "    - pivot = significant domain or function change (e.g. QA→SWE, backend→ML)\n"
        "    If the resume shows only ONE employer, judge progression from title changes within that company.\n"
        "    If only ONE title is visible, this is likely 'steady_growth' or 'stagnant', NOT 'rapid_growth'.\n\n"
        "  average_tenure_months (int): calculate from actual dates on the resume. "
        "For single-employer candidates, use total time at that employer.\n\n"
        "  job_hopping_risk (low|medium|high): single-employer candidates = 'low' by definition.\n\n"
        "  employment_gaps: list of idle periods > 3 months BETWEEN two separate employers. "
        "A current role ending 'Present' is NOT a gap. "
        "Continuous employment at one company = EMPTY list [].\n\n"
        "  title_progression (list of strings): Extract EVERY distinct job title from the resume, "
        "ordered oldest to newest. Include company name, e.g. "
        "['Junior Dev @ StartupX', 'Senior Dev @ StartupX', 'Staff Eng @ BigCorp']. "
        "If the resume lists multiple roles at ONE company, include EACH role separately. "
        "NEVER return a single-element list unless the resume truly shows only one role ever held.\n\n"
        "  company_tier_progression: MUST be one of: "
        "startup_to_enterprise | enterprise_to_startup | consistent | varied. "
        "'consistent' when all roles are at the same company or same tier of company.\n\n"
        "  pivot_detected (bool), pivot_details (string|null)\n\n"
        "  growth_potential (high|medium|low): based on trajectory momentum and JD alignment.\n\n"
        "  red_flags (list of strings) — RULES:\n"
        "    1. Each flag MUST cite a specific fact from the resume (company, title, date, duration).\n"
        "    2. Do NOT speculate about skills the candidate 'might' lack or 'may need'.\n"
        "    3. Do NOT restate what stat cards already show (e.g. don't flag 'specialization' "
        "when the progression_type already communicates that).\n"
        "    4. Only flag things a hiring manager should PROBE in an interview.\n"
        "    BAD: 'Focus is highly specialized which requires the next role to align' (speculation)\n"
        "    BAD: 'Specialization may limit opportunities' (generic, not from resume)\n"
        "    GOOD: 'No people-management experience visible despite 8 yrs tenure — probe if "
        "this role requires team leadership'\n"
        "    GOOD: 'All 3 roles focused on internal tooling — no customer-facing product "
        "experience, which this JD emphasizes'\n\n"
        "  green_flags (list of strings) — RULES:\n"
        "    1. Each flag MUST cite a specific fact from the resume.\n"
        "    2. Do NOT restate the progression_type (e.g. don't say 'rapid growth' as a flag).\n"
        "    3. Do NOT restate generic traits ('demonstrated leadership') — cite the evidence.\n"
        "    GOOD: 'Built RASCE system from zero to production in <12 months, demonstrating "
        "end-to-end ownership — directly relevant to this role'\n"
        "    GOOD: 'Mentored 4 junior engineers while managing 7 concurrent projects — shows "
        "force-multiplier capacity beyond individual contribution'\n\n"
        "### fit — multi-dimensional candidate suitability\n"
        "  technical_fit, experience_fit, domain_fit, seniority_fit, "
        "  culture_indicators, growth_potential, overall_fit (all 0-100), "
        "  hiring_confidence (high|medium|low), strongest_dimensions (top 2), "
        "  weakest_dimensions (top 2), unique_value (string), "
        "  risk_factors (list), ideal_for_role (bool), "
        "  compensation_tier (entry|mid|senior|executive)\n\n"
        "### red_flags — concerns and positive signals\n"
        "  critical_flags (list of {flag,evidence,severity critical|high,recommendation}), "
        "  warning_flags (list of {flag,evidence,questions_to_ask}), "
        "  green_flags (list), risk_level (low|medium|high|critical), "
        "  proceed_with_interview (bool), interview_focus_areas (list)\n\n"
        "Respond with valid JSON only."
    )

    data = await _chat_json(
        system_prompt=_DEEP_ANALYSIS_SYSTEM,
        user_prompt=prompt,
        schema=_DEEP_ANALYSIS_SCHEMA,
        num_predict=OLLAMA_NUM_PREDICT_JSON_HEAVY,
    )
    if data is None:
        return None
    return _parse_model(DeepResumeAnalysis, data)


# ---------------------------------------------------------------------------
# Phase 3: Comparative Candidate Ranking
# ---------------------------------------------------------------------------

class CandidateRanking(BaseModel):
    """Ranking entry for a single candidate."""
    candidate: str = Field(description="Candidate filename")
    rank: int = Field(description="Rank position (1 = best)")
    strengths: list[str] = Field(default_factory=list)
    weaknesses: list[str] = Field(default_factory=list)


class CandidateComparison(BaseModel):
    """Comparative analysis between candidates."""
    rankings: list[CandidateRanking] = Field(
        default_factory=list,
        description="Candidates ranked with justification",
    )

    @model_validator(mode="before")
    @classmethod
    def _coerce_rankings(cls, data: Any) -> Any:
        """Local LLMs sometimes return rankings as a list of strings or a pipe-delimited string."""
        import re as _re
        if not isinstance(data, dict):
            return data
        raw = data.get("rankings")
        if not isinstance(raw, list) or not raw:
            return data
        # Already a list of dicts — pass through
        if isinstance(raw[0], dict):
            return data
        # List of strings like "1. Name (reason) | 2. Name2 ..."
        # or just "1. Name (reason)"
        coerced: list[dict] = []
        for item in raw:
            if isinstance(item, dict):
                coerced.append(item)
                continue
            text = str(item)
            # Split pipe-delimited entries if present
            parts = [p.strip() for p in text.split("|") if p.strip()]
            for part in parts:
                m = _re.match(r"(\d+)[.)]\s*(.+?)(?:\s*[\(\[](.+?)[\)\]])?$", part)
                if m:
                    rank = int(m.group(1))
                    name = m.group(2).strip()
                    reason = m.group(3) or ""
                    coerced.append({
                        "candidate": name,
                        "rank": rank,
                        "strengths": [reason] if reason else [],
                        "weaknesses": [],
                    })
                else:
                    coerced.append({"candidate": part, "rank": len(coerced) + 1, "strengths": [], "weaknesses": []})
        data = dict(data)
        data["rankings"] = coerced
        return data

    best_for_technical: str = Field(
        default="",
        description="Candidate name strongest technically",
    )
    best_for_experience: str = Field(
        default="",
        description="Candidate with best experience match",
    )
    best_for_culture: str = Field(
        default="",
        description="Best culture fit indicators",
    )
    best_overall: str = Field(
        default="",
        description="Recommended hire",
    )
    stack_rank_justification: str = Field(
        default="",
        description="Why this ranking order",
    )
    hiring_recommendation: str = Field(
        default="need_more_candidates",
        description="One of: hire_top_1, hire_top_2, hire_none, need_more_candidates",
    )
    differentiation_factors: list[str] = Field(
        default_factory=list,
        description="Key factors that differentiate the candidates",
    )


_CANDIDATE_COMPARISON_SYSTEM = (
    "You are a VP of Engineering making final hiring decisions. "
    "Compare candidates against EACH OTHER, not just against the JD. "
    "Identify relative strengths and weaknesses between candidates. "
    "Provide actionable hiring recommendations. "
    "IMPORTANT: rankings MUST be a JSON array of objects, each with keys: "
    "candidate (string), rank (integer), strengths (list of strings), weaknesses (list of strings). "
    "Example: [{\"candidate\": \"Alice.pdf\", \"rank\": 1, \"strengths\": [\"strong Python\"], "
    "\"weaknesses\": [\"no Kubernetes\"]}]. Do NOT use strings or pipe-separated text for rankings."
)


async def compare_candidates(
    candidates: list[dict[str, Any]],
    jd_text: str,
) -> CandidateComparison | None:
    """Compare all candidates against each other for stack ranking.

    Args:
        candidates: List of scored candidates with their data
        jd_text: The job description text

    Returns:
        CandidateComparison with stack ranking and recommendations
    """
    from app.config import ENABLE_COMPARATIVE_RANKING

    if not ENABLE_COMPARATIVE_RANKING or not ai_enabled():
        return None

    if len(candidates) < 2:
        return None  # Need at least 2 candidates to compare

    schema = _schema_for(CandidateComparison)

    # Build candidate summaries
    summaries = []
    for c in candidates[:10]:  # Limit to top 10 for LLM context
        summaries.append(f"""
**{c['candidate']}** (Score: {c['final_score']}%)
- Decision: {c['decision']}
- Skills: {c['skill_match_ratio']} matched
- Matched: {', '.join(c.get('matched_skills', [])[:8])}
- Missing: {', '.join(c.get('missing_skills', [])[:5])}
- Section Scores: {c.get('section_scores', {})}""")

    prompt = f"""Compare these candidates for the role and provide stack ranking.

## Job Description
{jd_text[:JD_MAX_CHARS_LLM]}

## Candidates
{"".join(summaries)}

## Analysis Required

### 1. STACK RANKING (REQUIRED FORMAT)
Return "rankings" as a JSON array of objects. Each object MUST have:
- "candidate": exact filename string
- "rank": integer (1 = best)
- "strengths": list of strings (key strengths vs other candidates)
- "weaknesses": list of strings (key gaps vs other candidates)

### 2. DIMENSION WINNERS
- best_for_technical: name of candidate strongest technically, with 1-sentence reason
- best_for_experience: name with best experience match, with 1-sentence reason
- best_for_culture: name with best culture fit indicators

### 3. HIRING RECOMMENDATION
- hire_top_1: Clear top candidate, strongly recommend
- hire_top_2: Top 2 are close, either would be good
- hire_none: No candidates meet the bar for this role
- need_more_candidates: Pool too weak or small

### 4. DIFFERENTIATION FACTORS
List 2-4 key factors that separate these candidates.

Respond with valid JSON only."""

    data = await _chat_json(
        system_prompt=_CANDIDATE_COMPARISON_SYSTEM,
        user_prompt=prompt,
        schema=schema,
    )
    if data is None:
        return None
    return _parse_model(CandidateComparison, data)
