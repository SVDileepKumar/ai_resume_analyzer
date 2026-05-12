# 📄 ATS Resume Analyzer

An enterprise-grade Applicant Tracking System (ATS) resume analyzer built with **FastAPI**, **HTMX**, **Alpine.js**, **Tailwind CSS**, and **Chart.js**. Features deep LLM integration for semantic analysis, career trajectory evaluation, and AI-powered resume improvement coaching.

![Python](https://img.shields.io/badge/Python-3.12-blue)
![FastAPI](https://img.shields.io/badge/FastAPI-0.128-green)
![HTMX](https://img.shields.io/badge/HTMX-2.0-purple)
![Ollama](https://img.shields.io/badge/Ollama-AI--Powered-orange)

---

## ✨ Features

### 📊 Batch Analyzer (`/analyze`)
- **Built for HR & recruiters** — Compare, rank, and shortlist multiple candidates against one JD (no per-candidate resume coaching here; use `/review` for improvement suggestions)
- **Multi-Resume Upload** — Drag & drop up to 50 PDF/DOCX resumes at once with upload progress bars
- **JD-First Scoring** — Job description drives all scoring; role selection is optional/advisory
- **JD File Upload** — Upload JD as PDF/DOCX alongside text paste option
- **Hiring-manager sample JDs** — Templates are embedded with the page (same data as `GET /api/jd-templates`); choose a role, then **Use template** to fill the JD in paste mode
- **Smart Skill Matching** — 900+ skills across 27 categories matched against JD
- **24 Job Role Profiles** — Core engineering, testing (QA, SDET, Test Lead, Performance), enterprise platforms (SAP, ServiceNow, Salesforce, Snowflake), plus non-IT samples (HRBP, Financial Analyst, etc.)
- **6-Dimension Scoring** — Skills (33%), Experience (24%), Similarity (19%), Projects (11%), Education (8%), Certifications (5%) — weights vary by role or use unified defaults
- **Score Normalization** — Prevents dimension bias across scoring dimensions
- **Radar Charts** — Visual score breakdown per candidate (Chart.js)
- **Ranked Results** — Candidates sorted by ATS score with color-coded hiring decisions
- **Qualified Tab** — Filter to only candidates meeting qualification threshold
- **Comparison Matrix** — Side-by-side candidate comparison for 2+ resumes
- **CSV Export** — Download results for offline review

### 🔍 Individual Resume Review (`/review`)
- **Single Resume Deep Dive** — Upload one resume for personalized feedback
- **Optional JD Targeting** — Score against a specific JD or get general quality feedback
- **Use template (Review)** — With a target role selected, paste-mode JD can be filled from the same hiring-manager sample library as Analyze
- **Score Ring Visualization** — Animated circular score indicator
- **Dimension Breakdown** — Visual bars for skills, experience, projects, education, and similarity
- **AI Action Plan** — 5 prioritized improvement suggestions with before/after examples
- **Career Trajectory Analysis** — Progression patterns, tenure analysis, growth potential
- **Personalized Insights** — AI-generated strengths, gaps, and interview preparation

### 🧠 AI Features (Ollama-Powered)

All AI features are optional and gracefully degrade when Ollama is not running.

| Feature | Description |
|---------|-------------|
| **Semantic Similarity** | Embeddings via `qwen3-embedding:8b` for deep JD matching |
| **Semantic Skill Matching** | LLM identifies skill equivalences (e.g., "REST APIs" ≈ "API Development") |
| **LLM Holistic Scoring** | Multi-dimensional AI scoring blended with deterministic scores |
| **Per-Candidate Insights** | Strengths, gaps, and targeted interview questions |
| **Resume Improvement Suggestions** | 5 prioritized, actionable recommendations with before/after examples (**`/review` only**; not shown in batch Analyze) |
| **Career Trajectory Analysis** | Progression type, tenure patterns, green/red flags |
| **Achievement Impact Analysis** | Evaluates quantified achievements and impact statements |
| **Candidate Fit Analysis** | Culture fit, growth potential, team composition signals |
| **Red Flag Detection** | Employment gaps, job hopping, career inconsistencies |
| **Experience Relevance** | Context-aware analysis of past role relevance |
| **Executive Hiring Summary** | Batch-level analysis with 3-5 actionable recommendations |
| **Comparative Ranking** | AI-powered candidate ranking across multiple dimensions |

**Scoring modes:** With `AI_PRIMARY_SCORING=true` (default), one structured LLM call sets all six dimension scores when the LLM is available; regex anchors skill divergence when unified vs regex scores diverge by more than `SCORE_DIVERGENCE_THRESHOLD`.

### 🎨 Design & Accessibility
- **WCAG 2.2 AA Compliant** — Keyboard navigable, proper contrast ratios, screen reader support
- **Brand palette** — Primary blue (`#0053e2`) + accent gold (`#ffc220`)
- **Dark Mode** — Full dark theme support (press `D` to toggle)
- **Responsive** — Mobile-friendly layout

---

## 🚀 Quick Start

### Prerequisites

- **Python 3.12+** ([python.org](https://www.python.org/downloads/)) — Required for the backend
- **Git** ([git-scm.com](https://git-scm.com)) — To clone the repository
- **Ollama** (optional, but recommended) — For AI features. See [AI Setup](#-ai-setup-optional-but-recommended) below.

### One-command setup (macOS / Linux)

For a guided install (Python venv, `pip install`, Ollama + models from `.env` / `.env.example`, and `.env` creation):

```bash
git clone https://github.com/SVDileepKumar/ai_resume_analyzer.git
cd ai_resume_analyzer
bash scripts/setup-local.sh
```

Then start Ollama if it is not already running (`ollama serve`), activate `.venv`, and run Uvicorn as shown below. The script prints exact commands when it finishes.

**Security note:** On Linux, Ollama is installed via the official [ollama.com/install.sh](https://ollama.com/install.sh) script. Review it if your organization restricts `curl | sh` installers.

### Manual setup

#### All platforms:

```bash
# 1. Clone the repository
git clone https://github.com/SVDileepKumar/ai_resume_analyzer.git
cd ai_resume_analyzer

# 2. Create virtual environment
python3 -m venv .venv
# Or if you have uv installed:
# uv venv

# 3. Activate virtual environment
# On macOS / Linux:
source .venv/bin/activate
# On Windows:
# .venv\Scripts\activate

# 4. Install dependencies
pip install -r requirements.txt

# 5. Copy environment template (optional, but recommended)
cp .env.example .env

# 6. Run the server
uvicorn app.main:app --host 0.0.0.0 --port 8899 --timeout-keep-alive 1800

# 7. Open in browser
# macOS:
open http://localhost:8899
# Linux:
xdg-open http://localhost:8899
# Windows:
start http://localhost:8899
```

**Port configuration:** You can change `8899` to any available port. Useful if 8899 is already in use:
```bash
uvicorn app.main:app --host 0.0.0.0 --port 8888 --timeout-keep-alive 1800
```

### Frontend assets: CDN-first, local fallback

[`app/templates/base.html`](app/templates/base.html) tries **public CDNs first** (Tailwind Play, Google Fonts, jsDelivr for HTMX / Alpine / Chart.js) so you typically get **current CDN builds** without extra setup.

**For most users: No action needed** — the app works out of the box.

#### Offline / air-gapped environments:
After one install that includes `app/static` assets, the UI still loads via fallbacks without those CDNs.

#### For maintainers (modifying CSS):
Rebuild Tailwind when templates change:
```bash
npm install
npm run build:css
```

Refresh vendored mirrors:
```bash
bash scripts/download-frontend-assets.sh  # Requires network access
```

#### CI / local guards:
Verify that fallback paths in `base.html` exist on disk:
```bash
bash scripts/check-offline-assets.sh
```

### Corporate proxy / air-gapped networks

On **corporate networks** with `HTTP_PROXY` / `HTTPS_PROXY`, set `NO_PROXY` so loopback is not sent through the proxy (for curl, browser, or tools outside Python):

```bash
export NO_PROXY=localhost,127.0.0.1,::1
```

The app merges `localhost`, `127.0.0.1`, and `::1` into `NO_PROXY` and `no_proxy` when `app.config` loads, so `httpx` calls to Ollama on `127.0.0.1` use the proxy bypass automatically.

---

## 🧠 AI Setup (Optional but Recommended)

To enable AI-powered features, install and run [Ollama](https://ollama.ai). Defaults match [`app/config.py`](app/config.py) and [`.env.example`](.env.example):

```bash
# macOS (Homebrew)
brew install ollama

# Linux — official installer (review https://ollama.com/install.sh if required by policy)
curl -fsSL https://ollama.com/install.sh | sh

# Windows — download from https://ollama.ai/download

# Pull required models (override names in .env if you use different models)
ollama pull gemma4:e2b            # Chat / analysis (8GB-friendly default)
ollama pull qwen3.5:4b            # Optional fast multilingual chat (catalog alternative)
ollama pull qwen3-embedding:8b    # Embeddings for semantic similarity

# Start Ollama server
ollama serve
```

The app automatically detects Ollama and enables AI features. Without Ollama, all deterministic scoring features work normally.

---

## ⚙️ Configuration

Most settings use **environment variables**. Copy `.env.example` to `.env` to customize:

```bash
cp .env.example .env
```

### Ollama tuning (code, not `.env`)

Token budgets, JSON parse retries, JSON sampling temperature, and **HTTP retry** behavior for transient Ollama failures are defined as **constants in `app/config.py`** (e.g. `OLLAMA_NUM_PREDICT`, `OLLAMA_TEMPERATURE`).

### Environment Variables

#### LLM backend & models

| Variable | Default | Description |
|----------|---------|-------------|
| `LLM_BACKEND` | `ollama` | `ollama` (local) or `openai` (hosted / OpenAI-compatible API) |
| `NO_PROXY` / `no_proxy` | (merged at startup) | App merges `localhost`, `127.0.0.1`, `::1` into both vars on `app.config` import so local Ollama is not routed through `HTTP_PROXY`. Set `NO_PROXY` on startup for corporate networks. |
| `OLLAMA_BASE_URL` | `http://localhost:11434/v1` | Ollama API endpoint |
| `OLLAMA_CHAT_MODEL` | `gemma4:e2b` | LLM model for analysis |
| `OLLAMA_EMBED_MODEL` | `qwen3-embedding:8b` | Embedding model |
| `OLLAMA_TIMEOUT` | `1200` | LLM request timeout (seconds) |
| `OPENAI_API_KEY` | (empty) | API key for OpenAI-compatible endpoint (when `LLM_BACKEND=openai`) |
| `OPENAI_BASE_URL` | `https://api.openai.com/v1` | Base URL for hosted API (Groq, Azure OpenAI, etc.) |
| `OPENAI_CHAT_MODEL` | `gpt-4o-mini` | Default hosted chat model |
| `OPENAI_EMBED_MODEL` | `text-embedding-3-small` | Default hosted embedding model |

#### Model catalog & RAM hints

| Variable | Default | Description |
|----------|---------|-------------|
| `LLM_RAM_SAFETY_FACTOR` | `0.6` | Heuristic: `min_ram_gb` must be ≤ `RAM × factor` for auto-recommend |
| `LLM_DETECT_RAM` | `true` | Use `psutil` for server RAM (set `LLM_ASSUMED_RAM_GB` in Docker if wrong) |
| `LLM_ASSUMED_RAM_GB` | (empty) | Override detected RAM in Docker/remote hosts |
| `LLM_ALLOW_ANY_OLLAMA_MODEL` | `false` | Allow any pulled Ollama tag not listed in `app/data/ollama_models.json` |
| `ENABLE_HOSTED_LLM_UI` | `false` | Show optional API key fields; enable `POST /api/llm/settings` when combined with `LLM_USER_SETTINGS_PATH` |
| `LLM_USER_SETTINGS_PATH` | (empty) | Writable JSON path for operator-saved OpenAI-compatible key/base URL |

#### Concurrency & performance

| Variable | Default | Description |
|----------|---------|-------------|
| `LLM_MAX_CONCURRENT_RESUMES` | `10` | Max parallel LLM-heavy batch tasks per phase (`0` = unlimited) |
| `LLM_OLLAMA_CALL_CONCURRENCY` | `1` | Concurrent Ollama inference calls (`1` = serialize for single-GPU; `0` or higher for hosted APIs) |
| `LLM_ANALYSIS_TIMEOUT` | `1200` | Per-analysis LLM timeout (seconds) |

#### File & text limits

| Variable | Default | Description |
|----------|---------|-------------|
| `MAX_RESUMES` | `50` | Maximum resumes per batch |
| `MAX_FILE_SIZE_MB` | `10` | Maximum file size per upload |
| `MAX_JD_CHARS` | `50000` | Maximum JD text length |
| `MIN_JD_CHARS` | `50` | Minimum JD length to accept |
| `MAX_RESUME_CHARS` | `100000` | Max resume text length after extraction (longer is truncated) |
| `JD_MAX_CHARS_LLM` | `3000` | Max JD chars sent per LLM request |
| `RESUME_MAX_CHARS_LLM` | `4000` | Max resume chars sent per LLM request |
| `SIMILARITY_EMBED_MAX_CHARS` | `8000` | Max chars sent to embedding API |

#### Scoring & thresholds

| Variable | Default | Description |
|----------|---------|-------------|
| `AI_PRIMARY_SCORING` | `true` | One unified LLM call sets all six dimension scores (set `false` for hybrid) |
| `SCORE_DIVERGENCE_THRESHOLD` | `30` | Blend unified vs regex skills when they diverge by more than this |
| `ENRICHER_SCORE_THRESHOLD` | `35` | Skip deep enricher LLM call for resumes below this score (`0` = disable) |
| `HIRING_BAND_STRONG` | `80` | Score threshold for "Strong Match" |
| `HIRING_BAND_POTENTIAL` | `65` | Score threshold for "Potential Match" |
| `HIRING_BAND_NEEDS_REVIEW` | `50` | Score threshold for "Needs Review" |
| `USE_ROLE_WEIGHTS` | `true` | Use role-specific dimension weights when a role is selected |
| `DIMENSION_BOUNDS_OVERRIDE` | (empty) | Score normalization bounds override, e.g. `similarity:20,90` (semicolon-separated) |

#### Semantic matching

| Variable | Default | Description |
|----------|---------|-------------|
| `SEMANTIC_SKILL_THRESHOLD` | `0.75` | Minimum similarity for semantic skill partial match |
| `SEMANTIC_FULL_MATCH_THRESHOLD` | `0.9` | Minimum similarity for semantic skill full match |
| `SKILL_COSINE_FULL_MATCH` | `0.60` | Offline TF-IDF cosine threshold for full fuzzy skill match |
| `SKILL_COSINE_PARTIAL_MATCH` | `0.40` | Offline TF-IDF cosine threshold for partial fuzzy skill match (0.7× credit) |

#### Security & rate limiting

| Variable | Default | Description |
|----------|---------|-------------|
| `CORS_ORIGINS` | `http://localhost:8899,...` | Comma-separated allowed CORS origins |
| `RATE_LIMIT_REQUESTS` | `20` | Max requests per window for POST `/api/analyze` and `/api/review` (per IP) |
| `RATE_LIMIT_WINDOW_SEC` | `60` | Rate limit window duration (seconds) |

### LLM model selection & API keys

- **Analyze / Review** include a **Language model** control. The server sends your choice to Ollama or your OpenAI-compatible API as the chat model for that run.
- **Auto-recommendation** uses **this server's RAM** and `GET /api/tags` from Ollama (if `LLM_BACKEND=ollama`). If the app runs on a remote host, the hint describes the **server**, not your laptop.
- **Catalogs:** `app/data/ollama_models.json` and `app/data/openai_models.json` (include Gemma 4–style tags such as `gemma4:e2b` for ~8GB-class hardware — heuristics only; pull models with `ollama pull <tag>`).
- **Hosted APIs:** Set `LLM_BACKEND=openai` and `OPENAI_API_KEY` (and optional `OPENAI_BASE_URL` for Groq, Azure OpenAI, etc.). Token usage scales with resumes and features — monitor spend on your provider's dashboard.
- **Optional operator UI:** With `ENABLE_HOSTED_LLM_UI=true` and `LLM_USER_SETTINGS_PATH`, `POST /api/llm/settings` saves credentials to disk (gitignore local file). Prefer env vars on shared servers.

### Why there is no vector database

This app scores **one JD against a small batch of resumes** per request. Similarity is **pairwise** (embeddings in memory, cosine + lexical blend). A dedicated vector DB helps **large-scale ANN searches** (thousands of candidates). For small batches, in-memory similarity is simpler and faster.

### Feature Flags

| Flag | Default | Description |
|------|---------|-------------|
| `FEATURE_JD_FILE_UPLOAD` | `true` | Enable JD file upload (PDF/DOCX) |
| `FEATURE_OPTIONAL_ROLE` | `true` | Make role selection optional |
| `FEATURE_IMPROVEMENT_SUGGESTIONS` | `true` | AI resume improvement suggestions |
| `FEATURE_SCORE_NORMALIZATION` | `true` | Dimension score normalization |
| `ENABLE_SEMANTIC_SKILLS` | `true` | Semantic skill matching via LLM |
| `ENABLE_LLM_EXPERIENCE` | `true` | LLM-powered experience analysis |
| `ENABLE_ACHIEVEMENT_ANALYSIS` | `true` | Achievement impact scoring |
| `ENABLE_CAREER_TRAJECTORY` | `true` | Career progression analysis |
| `ENABLE_MULTI_DIM_FIT` | `true` | Multi-dimensional fit analysis |
| `ENABLE_RED_FLAG_DETECTION` | `true` | Red flag detection |
| `ENABLE_COMPARATIVE_RANKING` | `true` | AI comparative ranking |

---

## 📁 Project Structure

```
ai_resume_analyzer/
├── requirements.txt              # Python dependencies
├── package.json                  # Tailwind build (npm run build:css)
├── tailwind.config.js            # Tailwind theme (matches former Play CDN config)
├── .env.example                  # Environment variable template
├── .gitignore                    # Git ignore rules
├── README.md                     # This file
├── AGENTS.md                     # Agent / LLM notes (codebase context skill)
├── CONTRIBUTING.md               # Contribution guidelines
├── SECURITY.md                   # Security policy
├── LICENSE                       # MIT License
├── server.py                     # Re-exports app (Vercel entry point)
├── vercel.json                   # Vercel serverless config
├── scripts/
│   ├── setup-local.sh            # Venv + pip + Ollama models (guided setup)
│   ├── download-frontend-assets.sh  # Re-fetch vendored JS/fonts (maintainers)
│   ├── check-offline-assets.sh   # CI/local guard: local fallback files exist
│   └── hm_wrap_jd_templates.py   # JD template migration utility
├── tests/
│   ├── conftest.py               # Pytest fixtures
│   ├── test_skill_matcher.py
│   ├── test_llm_json_parse.py
│   ├── test_scoring_engine_integration.py
│   ├── test_similarity.py
│   ├── test_unified_resume_score.py
│   ├── test_openai_llm_backend.py
│   ├── test_section_scorer.py
│   ├── test_achievement_analyzer.py
│   ├── test_llm_catalog.py
│   └── test_jd_templates_bootstrap.py
└── app/
    ├── main.py                   # FastAPI app, routes, middleware
    ├── config.py                 # Environment config + feature flags
    ├── services/
    │   ├── pdf_parser.py         # PDF/DOCX text extraction (pdfplumber + python-docx)
    │   ├── text_normalizer.py    # Resume text cleanup / boilerplate stripping
    │   ├── scoring_engine.py     # Central ATS scoring orchestrator
    │   ├── skill_matcher.py      # Skill extraction & JD-aware matching (900+ skills)
    │   ├── similarity.py         # JD–resume similarity: cosine on embeddings or TF–IDF (+ Jaccard blend)
    │   ├── section_scorer.py     # Experience/education/project scoring + normalization
    │   ├── llm_service.py        # Ollama / OpenAI-compatible chat, insights, suggestions
    │   ├── llm_catalog.py        # Model JSON catalogs, RAM hint, Ollama tag merge
    │   ├── llm_context.py        # Per-request chat model + optional API credentials
    │   ├── llm_settings_store.py # Operator LLM settings persistence (BYOK)
    │   ├── semantic_skill_matcher.py  # LLM-powered semantic skill equivalence
    │   ├── experience_analyzer.py     # Context-aware experience relevance analysis
    │   ├── achievement_analyzer.py    # Achievement impact scoring
    │   ├── fit_analyzer.py            # Multi-dimensional candidate fit analysis
    │   └── red_flag_detector.py       # Employment red flag detection
    ├── data/
    │   ├── ollama_models.json    # Curated Ollama chat tags + min_ram_gb heuristics
    │   ├── openai_models.json    # Allowlisted hosted chat model ids
    │   ├── skill_db.json         # 900+ skills across 27 categories
    │   ├── role_profiles.json    # 24 role profiles (skills + weights)
    │   └── jd_templates.json     # Pre-built JD templates (sample posting styles)
    ├── templates/
    │   ├── base.html             # Base layout (CDN-first; local fallbacks)
    │   ├── dashboard.html        # Landing page
    │   ├── analyze.html          # Batch analyzer page
    │   ├── review.html           # Individual resume review page
    │   └── partials/
    │       ├── results_content.html      # Batch results (rankings, charts, tabs)
    │       ├── review_results.html       # Individual review results
    │       ├── ai_candidate_insight.html # AI insight card per candidate
    │       ├── ai_executive_summary.html # Executive hiring summary
    │       ├── comparison_matrix.html    # Side-by-side candidate comparison
    │       ├── jd_parsed_preview.html    # JD parsing preview
    │       ├── jd_templates_bootstrap.html # Inline JD template data for analyze/review
    │       ├── loading.html              # Loading animation
    │       └── error_banner.html         # Error display
    └── static/
        ├── css/
        │   ├── tailwind.css      # Built utility CSS (commit this; rebuild with npm)
        │   ├── tailwind-input.css
        │   ├── fonts.css         # @font-face for self-hosted fonts
        │   └── app.css           # Design system + component styles
        ├── fonts/                # Inter + JetBrains Mono (woff2)
        ├── vendor/               # htmx, Alpine, Chart.js (pinned minified)
        └── js/app.js             # Score rings, theme toggle, keyboard shortcuts
```

Optional: **`CODEBASE_CONTEXT.md`** at the repo root — dense, LLM-oriented map of the project (generated/updated via the **init-codebase-context** skill: e.g. ask for `/init` or "refresh codebase context").

---

## 🏗️ Scoring Architecture

```
JD Text ─────────────────────────────────────────────────────────────┐
  │                                                                  │
  ├─→ Skill Extraction (from JD) ────────→ Required Skills           │
  │                                              │                   │
  │   Resume Text                                │                   │
  │        │                                     │                   │
  │        ├─→ Skill Matching ←──────────────────┘                   │
  │        │       └─→ skills_score (33%)                            │
  │        │                                                         │
  │        ├─→ Semantic/TF-IDF Similarity ───→ similarity_score (19%)│
  │        │                                                         │
  │        ├─→ Experience Scoring ───────────→ experience_score (24%)│
  │        │       └─→ Years + Seniority (inferred if missing)       │
  │        │                                                         │
  │        ├─→ Education Scoring ────────────→ education_score (8%)  │
  │        │                                                         │
  │        ├─→ Certifications Scoring ───────→ certifications (5%)  │
  │        │                                                         │
  │        └─→ Projects Scoring ─────────────→ projects_score (11%) │
  │                                                                  │
  └─→ Score Normalization ───────────→ Normalized Scores (0-100)     │
          │                                                          │
          └─→ Weighted Average ──────→ Base ATS Score                │
                    │                                                │
                    │    ┌─── AI Enhancement (when Ollama running) ──┘
                    │    │
                    │    ├─→ Achievement Impact Bonus (-5 to +5 pts)
                    │    ├─→ Candidate Fit Modifier (-5 to +5 pts)
                    │    ├─→ Red Flag Penalty (-15 to +2 pts)
                    │    └─→ LLM Holistic Score (40% blend)
                    │
                    └─→ Final Score (0-100)
                            │
                            └─→ Hiring Decision:
                                 ≥80% Strong Match (green)
                                 ≥65% Potential Match (blue)
                                 ≥50% Needs Review (amber)
                                 <50% Weak Match (red)
```

### JD–resume text similarity (already cosine-based)

The **similarity** dimension is computed in [`app/services/similarity.py`](app/services/similarity.py). It does **not** need a separate "add cosine similarity" feature—**cosine is already the standard**.

| Path | What gets compared | Cosine role |
|------|-------------------|-------------|
| **Semantic** (when embeddings work: Ollama or OpenAI) | One embedding vector for the JD and one for the resume (text truncated per `SIMILARITY_EMBED_MAX_CHARS`) | **Cosine similarity** between vectors |
| **Lexical fallback** (no embedding) | TF–IDF vectors built from the JD + resume pair | **Cosine similarity** of TF–IDF vectors (`sklearn.metrics.pairwise.cosine_similarity`), blended with Jaccard / token overlap |

Tuning blend weights or adding chunking would be an **optional product change** only after evaluation—see project discussions / plans on similarity. For most users, enabling **`qwen3-embedding:8b`** in Ollama solves this.

---

## 🎨 Tech Stack

| Layer | Technology |
|-------|------------|
| Backend | FastAPI + Uvicorn |
| Frontend | HTMX + Tailwind + Alpine.js (CDN-first in `base.html`, fallbacks under `app/static/`) |
| Charts | Chart.js (radar, bar, horizontal bar; vendored UMD bundle) |
| PDF Parsing | pdfplumber + python-docx |
| NLP | scikit-learn (TF–IDF + cosine similarity for JD–resume fallback); embeddings + cosine when Ollama/OpenAI embed is available ([`app/services/similarity.py`](app/services/similarity.py)) |
| AI/LLM | Ollama (local) or any OpenAI-compatible API (Groq, Azure OpenAI, etc.) |
| Data Validation | Pydantic v2 (structured LLM output) |
| HTTP Client | httpx (async Ollama communication) |
| Design System | Primary blue (#0053e2) + accent (#ffc220) |

---

## 📝 Pages

| Route | Purpose |
|-------|---------|
| `/` | Dashboard — quick start, feature overview, AI status |
| `/analyze` | Batch Analyzer — multi-resume scoring against a JD |
| `/review` | Resume Review — single resume deep-dive with AI coaching |

---

## 🔒 Security

- Security headers (X-Content-Type-Options, X-Frame-Options, Referrer-Policy)
- CORS restricted to localhost origins
- File type validation via magic bytes (not just extension)
- Filename sanitization on upload
- No data persistence — all analysis is ephemeral
- Ollama runs 100% locally — no data leaves your machine (when using `LLM_BACKEND=ollama`)
- Per-IP rate limiting on analysis endpoints (configurable via `RATE_LIMIT_REQUESTS` / `RATE_LIMIT_WINDOW_SEC`)

---

## Acknowledgments

Local Ollama chat/embedding defaults and parts of the model-catalog approach were informed by earlier personal prototyping (including a capstone-style CLI pipeline for resume–JD ranking). This app expands that into an enterprise-grade web experience.

---

## License

This project is released under the [MIT License](LICENSE).

---

## Open source

Contributions are welcome. See [CONTRIBUTING.md](CONTRIBUTING.md). To report security issues privately, see [SECURITY.md](SECURITY.md).

---

## Deploying on Vercel (Hobby / free tier)

Vercel supports **FastAPI with zero extra wiring** when a FastAPI instance named `app` is exported from a [supported entry file](https://vercel.com/docs/frameworks/backend/fastapi) (this repo uses `server.py`).

1. Push the repository to GitHub (or GitLab / Bitbucket).
2. Import the project in the [Vercel dashboard](https://vercel.com/new) and connect the repo.
3. Set **environment variables** in the Vercel project (same names as `.env.example` where applicable). **Important:**
   - **`OLLAMA_BASE_URL`**: The app cannot reach **localhost** from Vercel's cloud. Point this to an **OpenAI-compatible HTTPS endpoint** (e.g. a hosted model API, or a secure tunnel to a machine on your network).
   - **`CORS_ORIGINS`**: Include your production site origin (e.g. `https://your-app.vercel.app`).
4. Deploy. CLI option: install the [Vercel CLI](https://vercel.com/docs/cli) and run `vercel` from the repo root.

**Limits to know (serverless):**

| Limit | Implication |
|-------|-------------|
| Request / response body **~4.5 MB** | Large PDF batches may hit limits; prefer smaller uploads on Vercel or use a container host for huge files. |
| Function **max duration** (see `vercel.json`) | Set to **300s** on Hobby (max); long analyses need a reachable LLM within that window. |
| Cold start | Heavy deps (e.g. scikit-learn) can make the first request slower. |

For **full local-Ollama workflows** (no remote API) or **very large** uploads, consider **Render**, **Railway**, **Fly.io**, or a **VPS** running `uvicorn` with Ollama on the same private network.

See also [CONTRIBUTING.md](CONTRIBUTING.md) for local `uvicorn` development.
