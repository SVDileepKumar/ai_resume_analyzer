# CODEBASE_CONTEXT
**Last Updated:** 2026-04-09

## 1. Project Mission & Goal
ATS Resume Analyzer is a FastAPI web app that scores PDF/DOCX resumes against a job description (JD-first), with optional LLM features via Ollama (local) or any OpenAI-compatible API (embeddings, insights, semantic skills, trajectory/fit/red-flag analysis). It serves HTML+HTMX UIs for batch comparison (`/analyze`) and single-resume review (`/review`).

## 2. Tech Stack & Infrastructure
- **Primary Language/Framework:** Python / FastAPI 0.128, Jinja2 templates, HTMX-driven partials (see `requirements.txt` for pinned versions).
- **Core Dependencies:** `fastapi`, `uvicorn`, `jinja2`, `pydantic`; document parsing `pdfplumber`, `python-docx`; scoring `scikit-learn`, `numpy`; LLM client `httpx` to Ollama's OpenAI-compatible API and `openai` SDK for hosted APIs; optional PDF reports `reportlab`; RAM detection `psutil`; tests `pytest`.
- **Infrastructure/Deployment:** No Dockerfiles in-repo; run with `uvicorn` (see README). Optional Vercel deployment via `server.py` + `vercel.json`. LLM backend selectable via `LLM_BACKEND` env var (`ollama` or `openai`); app degrades gracefully when LLM is unreachable (`/health` reports `degraded`).

## 3. Architecture & Patterns
Monolithic FastAPI app: routes and middleware in `app/main.py`, business logic in `app/services/*`, JSON data in `app/data/`, server-rendered UI with static assets. No database -- stateless scoring; only in-process per-IP rate limiting for heavy POSTs. Per-request LLM context (`llm_context.py`) allows different chat models / API keys per request without global state.

## 4. Module Map
- `/app/main.py` - FastAPI app, CORS + rate-limit + security headers, page routes (`/`, `/analyze`, `/review`), API routes returning HTML partials (`/api/analyze`, `/api/review`, `/api/parse-jd`), LLM management endpoints (`/api/llm/options`, `/api/llm/pull`, `/api/llm/settings`), `/health`, static mount, JSON helpers (`/api/roles`, `/api/jd-templates`).
- `/app/config.py` - Env-driven limits, Ollama/OpenAI URLs/models, feature flags, hiring bands, scoring weights, concurrency settings, CORS/rate-limit settings. Code-only constants for token budgets and HTTP retry behavior.
- `/app/services/scoring_engine.py` - Orchestrates dimension scores, optional LLM enrichments, batch vs single entry points (`analyze_batch`, `analyze_single_resume`, `score_resume`).
- `/app/services/skill_matcher.py` - JD/resume skill matching (900+ skills, 27 categories), role list (24 roles), `get_role_weights` (unified vs `role_profiles.json` when `USE_ROLE_WEIGHTS`).
- `/app/services/section_scorer.py` - Experience/education/projects/certs scoring and normalization.
- `/app/services/similarity.py` - TF-IDF and/or embedding-based JD similarity (Ollama/OpenAI embeddings when available).
- `/app/services/llm_service.py` - Ollama/OpenAI chat/embeddings, candidate insights, batch summary, suggestions, comparative ranking, JD parsing. Supports per-request model override via `llm_context`.
- `/app/services/llm_catalog.py` - Model JSON catalogs (`ollama_models.json`, `openai_models.json`), RAM hint, Ollama tag merge, model pull streaming.
- `/app/services/llm_context.py` - Per-request LLM context (chat model, optional API key/base URL). Thread-local via `contextvars`.
- `/app/services/llm_settings_store.py` - Operator LLM settings persistence to disk (BYOK UI, gated by `ENABLE_HOSTED_LLM_UI`).
- `/app/services/semantic_skill_matcher.py`, `experience_analyzer.py`, `achievement_analyzer.py`, `fit_analyzer.py`, `red_flag_detector.py` - LLM-assisted analyses (gated by config flags).
- `/app/services/pdf_parser.py` - PDF/DOCX text extraction; `text_normalizer.py` - resume boilerplate stripping.
- `/app/data/` - `skill_db.json` (900+ skills, 27 categories), `role_profiles.json` (24 roles), `jd_templates.json`, `ollama_models.json` (curated tags + RAM heuristics), `openai_models.json` (allowlisted hosted model ids).
- `/app/templates/` - Jinja2 pages and `partials/` for HTMX swaps (includes `jd_templates_bootstrap.html` for inline template data).
- `/app/static/` - CSS (Tailwind, custom), JS (app.js), self-hosted fonts (Inter, JetBrains Mono), vendored libs (htmx, Alpine.js, Chart.js).
- `/tests/` - Pytest integration and unit tests for scoring, matchers, LLM JSON parsing, similarity, achievement analysis, LLM catalog, JD templates.
- `/scripts/` - `setup-local.sh` (guided setup), `download-frontend-assets.sh`, `check-offline-assets.sh`, `hm_wrap_jd_templates.py`.

## 5. Core Data Flows
1. **Batch ATS analyze**: `POST /api/analyze` -> validate JD (paste or file) + resumes -> `pdf_parser.extract_text` -> `text_normalizer.strip_boilerplate` -> set per-request LLM context -> `scoring_engine.analyze_batch` -> Jinja `partials/results_content.html`.
2. **Single resume review**: `POST /api/review` -> optional JD -> extract resume -> set per-request LLM context -> `scoring_engine.analyze_single_resume` -> `partials/review_results.html`.
3. **JD preview**: `POST /api/parse-jd` -> optional `llm_service.parse_jd` if AI enabled -> `partials/jd_parsed_preview.html`.
4. **Health / AI availability**: `GET /health` -> checks LLM backend (Ollama connectivity or OpenAI API key presence) -> `ok` vs `degraded`.
5. **LLM management**: `GET /api/llm/options` returns available models + RAM hints; `POST /api/llm/pull` streams Ollama model download; `POST /api/llm/settings` persists operator BYOK credentials.

## 6. Known Quirks & Constraints
- **LLM optional**: Many features skip or simplify when the LLM backend is unavailable; do not assume embeddings/LLM calls succeed without checking `ai_enabled()` / config flags.
- **No persistence**: No DB; rate-limit state is in-memory only (not suitable for multi-instance without external store).
- **Role weights**: When `USE_ROLE_WEIGHTS` and a valid role are set, weights come from `role_profiles.json`; otherwise unified defaults (`app/config.py`, `get_role_weights`).
- **Concurrency**: `LLM_OLLAMA_CALL_CONCURRENCY` (default 1) serializes Ollama calls for single-GPU; `LLM_MAX_CONCURRENT_RESUMES` (default 10) caps parallel resume-level processing.
- **Enricher skip**: Resumes scoring below `ENRICHER_SCORE_THRESHOLD` (default 35) skip the deep enricher LLM pass to reduce cost/latency for weak candidates.
