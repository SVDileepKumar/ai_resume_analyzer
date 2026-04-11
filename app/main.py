"""ATS Resume Analyzer — FastAPI application with LLM integration."""

from __future__ import annotations

import asyncio
import json
import logging
import time
import uuid
from collections import deque
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import Annotated, Any, AsyncIterator

from fastapi import FastAPI, File, Form, HTTPException, Request, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, JSONResponse, Response, StreamingResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel

from app.config import (
    MAX_RESUMES, MAX_FILE_SIZE_MB, MAX_JD_CHARS, MIN_JD_CHARS, MAX_RESUME_CHARS,
    FEATURE_JD_FILE_UPLOAD, FEATURE_OPTIONAL_ROLE, FEATURE_IMPROVEMENT_SUGGESTIONS,
    CORS_ORIGINS, RATE_LIMIT_REQUESTS, RATE_LIMIT_WINDOW_SEC,
    LLM_BACKEND, OPENAI_API_KEY,
    ENABLE_HOSTED_LLM_UI, LLM_ALLOW_ANY_OLLAMA_MODEL, LLM_USER_SETTINGS_PATH,
)
from app.services.pdf_parser import extract_text
from app.services.text_normalizer import strip_boilerplate
from app.services.scoring_engine import analyze_batch
from app.services.skill_matcher import get_available_roles
from app.services.llm_catalog import (
    build_llm_options_payload,
    stream_ollama_model_pull,
    validate_openai_base_url,
    validate_ollama_chat_model,
    validate_openai_chat_model,
)
from app.services.llm_context import LLMRequestContext, reset_llm_context, set_llm_context
from app.services.llm_service import (
    ai_enabled,
    get_effective_chat_model,
    parse_jd,
    request_llm_ready,
    server_has_ai_route,
)
from app.services.llm_settings_store import save_operator_settings

import re


# ---------------------------------------------------------------------------
# In-memory batch result store with TTL eviction
# ---------------------------------------------------------------------------

@dataclass
class _BatchEntry:
    results: dict[str, Any]
    expires_at: float  # monotonic timestamp


class _BatchResultStore:
    """Thread-safe (asyncio) in-memory store for completed batch results.

    Results are keyed by a UUID and expire after ``ttl_seconds`` (default 30 min).
    Eviction is lazy — expired entries are removed on the next write or explicit purge.
    This gives the browser a window to fetch results via GET /api/results/{batch_id}
    even after the SSE/HTMX connection drops.
    """

    def __init__(self, ttl_seconds: int = 1800, max_entries: int = 200) -> None:
        self._store: dict[str, _BatchEntry] = {}
        self._lock = asyncio.Lock()
        self._ttl = ttl_seconds
        self._max = max_entries

    def _evict_expired(self) -> None:
        now = time.monotonic()
        expired = [k for k, v in self._store.items() if v.expires_at <= now]
        for k in expired:
            del self._store[k]

    async def put(self, results: dict[str, Any]) -> str:
        """Store results and return the assigned batch_id."""
        batch_id = str(uuid.uuid4())
        async with self._lock:
            self._evict_expired()
            if len(self._store) >= self._max:
                # Drop oldest entry when at capacity
                oldest = min(self._store, key=lambda k: self._store[k].expires_at)
                del self._store[oldest]
            self._store[batch_id] = _BatchEntry(
                results=results,
                expires_at=time.monotonic() + self._ttl,
            )
        return batch_id

    async def get(self, batch_id: str) -> dict[str, Any] | None:
        """Return stored results or None if not found / expired."""
        async with self._lock:
            entry = self._store.get(batch_id)
            if entry is None:
                return None
            if entry.expires_at <= time.monotonic():
                del self._store[batch_id]
                return None
            return entry.results


_result_store = _BatchResultStore()


def _sanitize_filename(name: str) -> str:
    """Strip dangerous characters from uploaded filenames."""
    return re.sub(r'[^\w\-. ]', '_', name or "unnamed")[:255]


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# App setup
# ---------------------------------------------------------------------------
BASE_DIR = Path(__file__).resolve().parent


def load_jd_templates() -> dict[str, str]:
    """Load role → sample JD text from app/data/jd_templates.json."""
    tpl_path = BASE_DIR / "data" / "jd_templates.json"
    with open(tpl_path, encoding="utf-8") as f:
        data = json.load(f)
    return {str(k): str(v) for k, v in (data or {}).items()}


@asynccontextmanager
async def lifespan(fastapi_app: FastAPI) -> AsyncIterator[None]:
    """Startup: init HTTP client, warm up the LLM model, pre-warm skill embeddings.
    Shutdown: close shared connections.
    """
    from app.services.llm_service import init_ollama_http_client, close_ollama_http_client, ai_enabled, _chat_json
    from app.services.semantic_skill_matcher import _init_skill_embeddings_async

    await init_ollama_http_client()

    if ai_enabled():
        # Warm up LLM model — loads weights into VRAM before first real request.
        try:
            logger.info("Warming up LLM model...")
            warmup = await _chat_json(
                "You are a helpful assistant.",
                "Reply with: ok",
                max_json_retries=0,
            )
            if warmup is None:
                logger.warning("LLM warm-up failed (non-fatal): no JSON response returned")
            else:
                logger.info("LLM model warm-up complete")
        except Exception as e:
            logger.warning("LLM warm-up failed (non-fatal): %s", e)

        # Pre-warm skill embeddings — amortizes the 332-skill batch embedding call
        # at startup instead of blocking the first batch request.
        try:
            logger.info("Pre-warming skill embeddings...")
            await _init_skill_embeddings_async()
            logger.info("Skill embeddings ready")
        except Exception as e:
            logger.warning("Skill embedding warm-up failed (non-fatal): %s", e)

    yield
    await close_ollama_http_client()


app = FastAPI(title="ATS Resume Analyzer", version="0.0-dev", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=CORS_ORIGINS,
    allow_methods=["GET", "POST"],
    allow_headers=["Content-Type"],
)

# In-process rate limit state for POST /api/analyze and /api/review (per IP)
_rate_limit_store: dict[str, deque[float]] = {}
_rate_limit_lock = asyncio.Lock()


@app.middleware("http")
async def rate_limit_middleware(request: Request, call_next):
    """Per-IP rate limit for heavy endpoints (configurable via RATE_LIMIT_*)."""
    rate_limited = ("/api/analyze", "/api/review", "/api/llm/settings", "/api/llm/pull")
    if request.method != "POST" or request.url.path not in rate_limited:
        return await call_next(request)
    client_ip = request.client.host if request.client else "unknown"
    now = time.monotonic()
    async with _rate_limit_lock:
        if client_ip not in _rate_limit_store:
            _rate_limit_store[client_ip] = deque(maxlen=RATE_LIMIT_REQUESTS * 2)
        q = _rate_limit_store[client_ip]
        cutoff = now - RATE_LIMIT_WINDOW_SEC
        while q and q[0] < cutoff:
            q.popleft()
        if len(q) >= RATE_LIMIT_REQUESTS:
            from fastapi.responses import JSONResponse
            return JSONResponse(
                status_code=429,
                content={"detail": "Too many requests. Try again later."},
                headers={"Retry-After": str(RATE_LIMIT_WINDOW_SEC)},
            )
        q.append(now)
    return await call_next(request)


@app.middleware("http")
async def add_security_headers(request: Request, call_next):
    response = await call_next(request)
    response.headers["X-Content-Type-Options"] = "nosniff"
    response.headers["X-Frame-Options"] = "DENY"
    response.headers["Referrer-Policy"] = "strict-origin-when-cross-origin"
    return response


app.mount("/static", StaticFiles(directory=BASE_DIR / "static"), name="static")
templates = Jinja2Templates(directory=BASE_DIR / "templates")


# ---------------------------------------------------------------------------
# Health (for load balancers / Kubernetes)
# ---------------------------------------------------------------------------
@app.get("/health")
async def health():
    """Health check endpoint. Returns 200 when app is ready to serve.
    Optionally reports degraded when Ollama is unreachable (AI features disabled).
    """
    from app.services.llm_service import get_effective_openai_api_key, is_ollama_running

    if LLM_BACKEND == "openai":
        if not get_effective_openai_api_key():
            return {
                "status": "degraded",
                "llm": "openai_not_configured",
                "message": "AI features disabled (set OPENAI_API_KEY or operator LLM settings)",
            }
        return {"status": "ok", "llm": "openai"}
    if not is_ollama_running():
        return {"status": "degraded", "ollama": "unreachable", "message": "AI features disabled"}
    return {"status": "ok", "llm": "ollama"}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _apply_llm_request_context(
    chat_model: str | None,
    openai_api_key: str | None,
    openai_base_url: str | None,
) -> str | None:
    """Validate form fields and set per-request LLM context. Returns error message or None."""
    cm = (chat_model or "").strip() or None
    key = (openai_api_key or "").strip() or None
    base = (openai_base_url or "").strip() or None
    if LLM_BACKEND == "ollama" and cm:
        ok, _ = validate_ollama_chat_model(cm)
        if not ok and not LLM_ALLOW_ANY_OLLAMA_MODEL:
            return (
                f"Unknown Ollama model «{cm}». Pull it with `ollama pull {cm}`, pick a catalog "
                "model, set OLLAMA_CHAT_MODEL, or set LLM_ALLOW_ANY_OLLAMA_MODEL=true."
            )
    if LLM_BACKEND == "openai" and cm:
        ok, _ = validate_openai_chat_model(cm)
        if not ok:
            return (
                f"Unknown hosted model «{cm}». Choose a catalog id or your OPENAI_CHAT_MODEL default."
            )
    if base and not validate_openai_base_url(base):
        return "Invalid API base URL (use https:// for remote APIs, or http://127.0.0.1 for local gateways)."
    set_llm_context(LLMRequestContext(chat_model=cm, openai_api_key=key, openai_base_url=base))
    return None


def _error_html(request: Request, message: str) -> HTMLResponse:
    """Return an error as an HTML partial that HTMX can swap in."""
    return templates.TemplateResponse(
        "partials/error_banner.html",
        {"request": request, "error_message": message},
        status_code=200,
    )


# ---------------------------------------------------------------------------
# Page routes
# ---------------------------------------------------------------------------
@app.get("/", response_class=HTMLResponse)
async def dashboard(request: Request):
    return templates.TemplateResponse("dashboard.html", {
        "request": request,
        "roles": get_available_roles(),
        "ai_enabled": server_has_ai_route(),
    })


@app.get("/favicon.ico", include_in_schema=False)
async def favicon() -> Response:
    """Return an explicit empty favicon response to avoid noisy 404s."""
    return Response(status_code=204)


@app.get("/analyze", response_class=HTMLResponse)
async def analyze_page(request: Request):
    return templates.TemplateResponse("analyze.html", {
        "request": request,
        "roles": get_available_roles(),
        "jd_templates": load_jd_templates(),
        "ai_enabled": server_has_ai_route(),
        "llm_backend": LLM_BACKEND,
    })


# ---------------------------------------------------------------------------
# LLM options (JSON for UI)
# ---------------------------------------------------------------------------
@app.get("/api/llm/options")
async def api_llm_options():
    return JSONResponse(build_llm_options_payload())


class PullModelRequest(BaseModel):
    chat_model: str


@app.post("/api/llm/pull")
async def api_llm_pull(payload: PullModelRequest):
    """Stream Ollama model download progress as NDJSON for the selected chat model."""
    if LLM_BACKEND != "ollama":
        raise HTTPException(status_code=400, detail="Model pull is only available for Ollama.")
    model = (payload.chat_model or "").strip()
    if not model:
        raise HTTPException(status_code=400, detail="chat_model is required.")
    ok, normalized = validate_ollama_chat_model(model)
    if not ok and not LLM_ALLOW_ANY_OLLAMA_MODEL:
        raise HTTPException(
            status_code=400,
            detail=(
                f"Unknown Ollama model '{model}'. Pick a catalog entry, set "
                "OLLAMA_CHAT_MODEL, or enable LLM_ALLOW_ANY_OLLAMA_MODEL."
            ),
        )

    async def _stream() -> AsyncIterator[bytes]:
        yield (json.dumps({
            "model": normalized,
            "status": "starting",
            "percent": 0,
            "done": False,
        }) + "\n").encode("utf-8")
        try:
            async for event in stream_ollama_model_pull(normalized):
                yield (json.dumps(event) + "\n").encode("utf-8")
        except Exception as exc:
            logger.exception("Failed to pull Ollama model %s", normalized)
            yield (json.dumps({
                "model": normalized,
                "status": "error",
                "error": str(exc),
                "done": True,
            }) + "\n").encode("utf-8")

    return StreamingResponse(_stream(), media_type="application/x-ndjson")


@app.post("/api/llm/settings")
async def api_llm_settings(
    openai_api_key: Annotated[str | None, Form()] = None,
    openai_base_url: Annotated[str | None, Form()] = None,
):
    """Persist operator OpenAI-compatible credentials (same-origin; opt-in)."""
    if not ENABLE_HOSTED_LLM_UI:
        return JSONResponse(
            {"detail": "Hosted LLM settings UI is disabled (ENABLE_HOSTED_LLM_UI)."},
            status_code=403,
        )
    if not LLM_USER_SETTINGS_PATH.strip():
        return JSONResponse(
            {"detail": "Set LLM_USER_SETTINGS_PATH to a writable JSON file path."},
            status_code=400,
        )
    key = (openai_api_key or "").strip()
    base = (openai_base_url or "").strip() or "https://api.openai.com/v1"
    if not key:
        return JSONResponse({"detail": "openai_api_key is required."}, status_code=400)
    if not validate_openai_base_url(base):
        return JSONResponse({"detail": "Invalid openai_base_url."}, status_code=400)
    try:
        save_operator_settings(key, base.rstrip("/"))
    except OSError as e:
        logger.exception("Failed to save LLM settings")
        return JSONResponse({"detail": str(e)}, status_code=500)
    return {"ok": True}


# ---------------------------------------------------------------------------
# API routes (return HTML partials for HTMX)
# ---------------------------------------------------------------------------
@app.post("/api/analyze", response_class=HTMLResponse)
async def api_analyze(
    request: Request,
    resumes: Annotated[list[UploadFile], File(...)],
    job_description: Annotated[str | None, Form()] = None,
    job_description_file: Annotated[UploadFile | None, File()] = None,
    role: Annotated[str | None, Form()] = None,
    chat_model: Annotated[str | None, Form()] = None,
    openai_api_key: Annotated[str | None, Form()] = None,
    openai_base_url: Annotated[str | None, Form()] = None,
):
    """Analyze uploaded resumes against JD. Returns HTML partial for HTMX.

    JD Input (one required):
    - job_description: Paste JD text directly
    - job_description_file: Upload JD as PDF/DOCX (if FEATURE_JD_FILE_UPLOAD enabled)

    Role (optional if FEATURE_OPTIONAL_ROLE enabled):
    - If provided, used as metadata/advisory only
    - Scoring uses unified weights regardless of role selection
    """
    # --- Extract JD text from file or form field ---
    jd_text: str = ""

    if job_description_file and job_description_file.filename and FEATURE_JD_FILE_UPLOAD:
        # JD uploaded as file
        if job_description and job_description.strip():
            return _error_html(request, "Provide JD via paste OR upload, not both.")
        try:
            jd_text = await extract_text(job_description_file)
        except ValueError as exc:
            return _error_html(request, f"JD file error: {exc}")
        except Exception:
            logger.exception("Error parsing JD file %s", job_description_file.filename)
            return _error_html(request, "Failed to parse JD file. Use PDF or DOCX.")
    elif job_description:
        # JD pasted as text
        jd_text = job_description.strip()
    else:
        return _error_html(request, "Job description is required. Paste text or upload PDF/DOCX.")

    # --- Validate JD length ---
    if len(jd_text) < MIN_JD_CHARS:
        return _error_html(request, f"Job description too short (min {MIN_JD_CHARS} characters).")
    if len(jd_text) > MAX_JD_CHARS:
        jd_text = jd_text[:MAX_JD_CHARS]
        logger.warning("JD truncated to %d chars", MAX_JD_CHARS)

    # --- Validate role (optional if feature flag enabled) ---
    if not FEATURE_OPTIONAL_ROLE:
        if not role or not role.strip():
            return _error_html(request, "Please select a role.")

    # Normalize role - empty string or None both mean "no role selected"
    role_normalized = (role or "").strip() or None

    # --- Validate resumes ---
    if not resumes or (len(resumes) == 1 and not resumes[0].filename):
        return _error_html(request, "Please upload at least one resume.")
    if len(resumes) > MAX_RESUMES:
        return _error_html(request, f"Maximum {MAX_RESUMES} resumes allowed.")

    # --- Extract text from each resume ---
    parsed_resumes: list[dict[str, str]] = []
    errors: list[str] = []

    for resume_file in resumes:
        try:
            resume_file.file.seek(0, 2)
            size_mb = resume_file.file.tell() / (1024 * 1024)
            resume_file.file.seek(0)
            if size_mb > MAX_FILE_SIZE_MB:
                errors.append(f"{resume_file.filename}: exceeds {MAX_FILE_SIZE_MB}MB limit")
                continue

            text = await extract_text(resume_file)
            text = strip_boilerplate(text)
            if not text.strip():
                errors.append(f"{resume_file.filename}: could not extract text")
                continue
            if len(text) > MAX_RESUME_CHARS:
                text = text[:MAX_RESUME_CHARS]
                logger.debug("Resume %s truncated to %d chars", resume_file.filename, MAX_RESUME_CHARS)

            parsed_resumes.append({
                "filename": _sanitize_filename(resume_file.filename),
                "text": text,
            })
        except ValueError as exc:
            errors.append(f"{resume_file.filename}: {exc}")
        except Exception:
            logger.exception("Error parsing %s", resume_file.filename)
            errors.append(f"{resume_file.filename}: parsing error")

    if not parsed_resumes:
        error_detail = "No valid resumes could be parsed."
        if errors:
            error_detail += " Issues: " + "; ".join(errors)
        return _error_html(request, error_detail)

    ctx_err = _apply_llm_request_context(chat_model, openai_api_key, openai_base_url)
    if ctx_err:
        return _error_html(request, ctx_err)

    if not ai_enabled():
        return _error_html(
            request,
            "AI scoring is required but unavailable. "
            "Please start Ollama or configure your OpenAI API key before uploading resumes."
        )

    try:
        logger.info(
            "analyze batch llm backend=%s model=%s",
            LLM_BACKEND,
            get_effective_chat_model(),
        )
        results = await analyze_batch(parsed_resumes, jd_text, role_normalized)
        results["errors"] = errors

        # Persist results so the browser can re-fetch via GET /api/results/{batch_id}
        # (e.g. after page reload or when JS needs the raw JSON for charting).
        batch_id = await _result_store.put(results)

        return templates.TemplateResponse("partials/results_content.html", {
            "request": request,
            "results": results,
            "jd_text": jd_text,
            "batch_id": batch_id,
        })
    except RuntimeError as exc:
        return _error_html(request, str(exc))
    finally:
        reset_llm_context()


@app.get("/review", response_class=HTMLResponse)
async def review_page(request: Request):
    return templates.TemplateResponse("review.html", {
        "request": request,
        "ai_enabled": server_has_ai_route(),
        "roles": get_available_roles(),
        "jd_templates": load_jd_templates(),
        "llm_backend": LLM_BACKEND,
    })


@app.post("/api/review", response_class=HTMLResponse)
async def api_review(
    request: Request,
    resume: Annotated[UploadFile, File(...)],
    job_description: Annotated[str | None, Form()] = None,
    job_description_file: Annotated[UploadFile | None, File()] = None,
    role: Annotated[str | None, Form()] = None,
    chat_model: Annotated[str | None, Form()] = None,
    openai_api_key: Annotated[str | None, Form()] = None,
    openai_base_url: Annotated[str | None, Form()] = None,
):
    """Individual resume review — single resume, optional JD."""
    from app.services.scoring_engine import analyze_single_resume

    # Extract JD (optional)
    jd_text: str | None = None
    if job_description_file and job_description_file.filename:
        try:
            jd_text = await extract_text(job_description_file)
        except Exception:
            pass
    elif job_description and job_description.strip():
        jd_text = job_description.strip()

    # Validate resume
    if not resume or not resume.filename:
        return _error_html(request, "Please upload a resume.")

    resume.file.seek(0, 2)
    size_mb = resume.file.tell() / (1024 * 1024)
    resume.file.seek(0)
    if size_mb > MAX_FILE_SIZE_MB:
        return _error_html(request, f"File exceeds {MAX_FILE_SIZE_MB}MB limit.")

    try:
        resume_text = await extract_text(resume)
    except Exception:
        return _error_html(request, "Could not parse resume. Use PDF or DOCX.")
    resume_text = strip_boilerplate(resume_text)

    if not resume_text.strip():
        return _error_html(request, "Could not extract text from resume.")
    if len(resume_text) > MAX_RESUME_CHARS:
        resume_text = resume_text[:MAX_RESUME_CHARS]

    ctx_err = _apply_llm_request_context(chat_model, openai_api_key, openai_base_url)
    if ctx_err:
        return _error_html(request, ctx_err)

    if not ai_enabled():
        return _error_html(
            request,
            "AI scoring is required but unavailable. "
            "Please start Ollama or configure your OpenAI API key."
        )

    try:
        logger.info(
            "review llm backend=%s model=%s",
            LLM_BACKEND,
            get_effective_chat_model(),
        )
        role_normalized = role.strip() if role and role.strip() else None
        result = await analyze_single_resume(
            resume_text, jd_text, role=role_normalized, filename=resume.filename or "resume",
        )

        has_jd = bool(jd_text and jd_text.strip())
        score = result.get("final_score", 0)

        return templates.TemplateResponse("partials/review_results.html", {
            "request": request,
            "result": result,
            "score": score,
            "has_jd": has_jd,
            "ai_enabled": ai_enabled(),
            "feature_improvement_suggestions": FEATURE_IMPROVEMENT_SUGGESTIONS,
        })
    except RuntimeError as exc:
        return _error_html(request, str(exc))
    finally:
        reset_llm_context()


@app.post("/api/parse-jd", response_class=HTMLResponse)
async def api_parse_jd(
    request: Request,
    job_description: Annotated[str | None, Form()] = None,
    job_description_file: Annotated[UploadFile | None, File()] = None,
    chat_model: Annotated[str | None, Form()] = None,
    openai_api_key: Annotated[str | None, Form()] = None,
    openai_base_url: Annotated[str | None, Form()] = None,
):
    """Parse JD and return a preview of extracted requirements."""
    # Extract JD text (same logic as analyze)
    jd_text = ""
    if job_description_file and job_description_file.filename:
        try:
            jd_text = await extract_text(job_description_file)
        except Exception:
            jd_text = ""
    elif job_description:
        jd_text = job_description.strip()

    if len(jd_text) < MIN_JD_CHARS:
        return _error_html(request, f"JD too short for preview (min {MIN_JD_CHARS} chars).")

    ctx_err = _apply_llm_request_context(chat_model, openai_api_key, openai_base_url)
    if ctx_err:
        return _error_html(request, ctx_err)

    try:
        parsed = None
        if request_llm_ready():
            logger.info(
                "parse-jd llm backend=%s model=%s",
                LLM_BACKEND,
                get_effective_chat_model(),
            )
            parsed = await parse_jd(jd_text[:MAX_JD_CHARS])

        return templates.TemplateResponse("partials/jd_parsed_preview.html", {
            "request": request,
            "parsed": parsed,
        })
    finally:
        reset_llm_context()


@app.get("/api/roles")
async def api_roles():
    return {"roles": get_available_roles()}


@app.get("/api/jd-templates")
async def api_jd_templates():
    """Return pre-built JD templates for each role."""
    return load_jd_templates()


@app.get("/api/results/{batch_id}")
async def api_get_results(batch_id: str):
    """Return a previously completed batch analysis as JSON.

    Results are stored in-memory for 30 minutes after the POST /api/analyze
    completes. Useful for:
    - Fetching raw JSON for client-side charting / export
    - Re-rendering results after a page reload without re-running the LLM
    - Programmatic access by agents / scripts

    Returns 404 if the batch_id is unknown or expired.
    """
    # Validate UUID format to avoid log spam from malformed IDs
    try:
        uuid.UUID(batch_id)
    except ValueError:
        raise HTTPException(status_code=400, detail="Invalid batch_id format.")

    results = await _result_store.get(batch_id)
    if results is None:
        raise HTTPException(
            status_code=404,
            detail="Results not found or expired. Re-run the analysis to generate fresh results.",
        )
    return JSONResponse(results)
