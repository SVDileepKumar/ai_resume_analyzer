"""Application configuration.

Most tunables use environment variables with defaults. The following are **code-only**
(edit ``app/config.py``, redeploy): Ollama JSON token limits (``OLLAMA_NUM_PREDICT*``),
``OLLAMA_JSON_MAX_RETRIES``, ``OLLAMA_JSON_TEMPERATURE``, and HTTP retry settings
(``OLLAMA_HTTP_MAX_RETRIES``, ``OLLAMA_HTTP_RETRY_BACKOFF_SEC``).
"""

from __future__ import annotations

import os

# --- File limits ---
MAX_RESUMES: int = int(os.getenv("MAX_RESUMES", "50"))
MAX_FILE_SIZE_MB: int = int(os.getenv("MAX_FILE_SIZE_MB", "10"))
MAX_JD_CHARS: int = int(os.getenv("MAX_JD_CHARS", "50000"))
MIN_JD_CHARS: int = int(os.getenv("MIN_JD_CHARS", "50"))
# Max resume text length after extraction; longer text is truncated for consistency (default 100k)
MAX_RESUME_CHARS: int = int(os.getenv("MAX_RESUME_CHARS", "100000"))
# Max chars sent to LLM per document (used by llm_service, semantic_skill_matcher, etc.)
JD_MAX_CHARS_LLM: int = int(os.getenv("JD_MAX_CHARS_LLM", "3000"))
RESUME_MAX_CHARS_LLM: int = int(os.getenv("RESUME_MAX_CHARS_LLM", "20000"))

# --- Ollama (local LLM) ---
# Ollama serves an OpenAI-compatible API on localhost.
# No API key needed — everything runs locally on your machine.
OLLAMA_BASE_URL: str = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434/v1")
OLLAMA_CHAT_MODEL: str = os.getenv("OLLAMA_CHAT_MODEL", "qwen3.5:2b")
OLLAMA_EMBED_MODEL: str = os.getenv("OLLAMA_EMBED_MODEL", "qwen3-embedding:0.6b")
OLLAMA_TIMEOUT: int = int(os.getenv("OLLAMA_TIMEOUT", "1200"))
# Max tokens per Ollama /api/chat completion — tuned in code only (see app/config.py).
# Higher values reduce JSON truncation on long resumes; increase latency slightly.
OLLAMA_NUM_PREDICT: int = 2048
# Deep analysis (achievements, trajectory, fit, red flags) kept at 6144 to prevent
# JSON truncation on complex resumes. Lower increases throughput but risks cut-off JSON.
OLLAMA_NUM_PREDICT_JSON_HEAVY: int = 6144
# JD skill extraction output is smaller than resume skill lists; lower default latency.
OLLAMA_NUM_PREDICT_JD_SKILLS: int = 4096
OLLAMA_JSON_MAX_RETRIES: int = 3
OLLAMA_JSON_TEMPERATURE: float = 0.22
# Transient Ollama/network failures (separate from JSON parse retries in llm_service._chat_json).
OLLAMA_HTTP_MAX_RETRIES: int = 3
OLLAMA_HTTP_RETRY_BACKOFF_SEC: float = 0.5

# --- LLM backend: ollama (local) | openai (hosted via openai Python SDK, OpenAI-compatible base URL) ---
_raw_llm_backend = os.getenv("LLM_BACKEND", "ollama").strip().lower()
LLM_BACKEND: str = _raw_llm_backend if _raw_llm_backend in ("ollama", "openai") else "ollama"
OPENAI_API_KEY: str = os.getenv("OPENAI_API_KEY", "").strip()
OPENAI_BASE_URL: str = os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1").strip()
OPENAI_CHAT_MODEL: str = os.getenv("OPENAI_CHAT_MODEL", "gpt-4o-mini").strip()
OPENAI_EMBED_MODEL: str = os.getenv("OPENAI_EMBED_MODEL", "text-embedding-3-small").strip()

# --- Hiring decision bands (configurable per company/campaign) ---
HIRING_BAND_STRONG: float = float(os.getenv("HIRING_BAND_STRONG", "80"))
HIRING_BAND_POTENTIAL: float = float(os.getenv("HIRING_BAND_POTENTIAL", "65"))
HIRING_BAND_NEEDS_REVIEW: float = float(os.getenv("HIRING_BAND_NEEDS_REVIEW", "50"))

# --- Unified Scoring Weights ---
# JD-first approach. When USE_ROLE_WEIGHTS is true and a role is selected, role_profiles weights are used instead.
UNIFIED_WEIGHTS: dict[str, float] = {
    "skills": 0.33,          # Technical skill match from JD
    "experience": 0.24,      # Years & seniority alignment
    "similarity": 0.19,      # Semantic similarity to JD
    "projects": 0.11,        # Relevant project experience
    "education": 0.08,       # Educational background match
    "certifications": 0.05,  # Certifications (e.g. PMP, AWS) when JD requires
}
USE_ROLE_WEIGHTS: bool = os.getenv("USE_ROLE_WEIGHTS", "true").lower() == "true"

# --- Feature Flags ---
# Enable/disable features for safe rollback during deployment
FEATURE_JD_FILE_UPLOAD: bool = os.getenv("FEATURE_JD_FILE_UPLOAD", "true").lower() == "true"
FEATURE_OPTIONAL_ROLE: bool = os.getenv("FEATURE_OPTIONAL_ROLE", "true").lower() == "true"
FEATURE_IMPROVEMENT_SUGGESTIONS: bool = os.getenv("FEATURE_IMPROVEMENT_SUGGESTIONS", "true").lower() == "true"
FEATURE_SCORE_NORMALIZATION: bool = os.getenv("FEATURE_SCORE_NORMALIZATION", "true").lower() == "true"

# --- LLM-Enhanced Scoring Feature Flags ---
# Phase 3: Deep LLM integration for semantic analysis
ENABLE_SEMANTIC_SKILLS: bool = os.getenv("ENABLE_SEMANTIC_SKILLS", "true").lower() == "true"
ENABLE_LLM_EXPERIENCE: bool = os.getenv("ENABLE_LLM_EXPERIENCE", "true").lower() == "true"
ENABLE_ACHIEVEMENT_ANALYSIS: bool = os.getenv("ENABLE_ACHIEVEMENT_ANALYSIS", "true").lower() == "true"
ENABLE_CAREER_TRAJECTORY: bool = os.getenv("ENABLE_CAREER_TRAJECTORY", "true").lower() == "true"
ENABLE_MULTI_DIM_FIT: bool = os.getenv("ENABLE_MULTI_DIM_FIT", "true").lower() == "true"
ENABLE_RED_FLAG_DETECTION: bool = os.getenv("ENABLE_RED_FLAG_DETECTION", "true").lower() == "true"
ENABLE_COMPARATIVE_RANKING: bool = os.getenv("ENABLE_COMPARATIVE_RANKING", "true").lower() == "true"

# Semantic matching configuration (must be in 0-1)
def _clamp_float_env(key: str, default: str, low: float, high: float) -> float:
    val = float(os.getenv(key, default))
    return max(low, min(high, val))


# --- LLM model catalog & RAM hints (see app/data/ollama_models.json) ---
LLM_RAM_SAFETY_FACTOR: float = _clamp_float_env("LLM_RAM_SAFETY_FACTOR", "0.6", 0.2, 0.95)
_raw_detect = os.getenv("LLM_DETECT_RAM", "true").strip().lower()
LLM_DETECT_RAM: bool = _raw_detect in ("1", "true", "yes")
LLM_ASSUMED_RAM_GB: float | None = None
_assumed_ram = os.getenv("LLM_ASSUMED_RAM_GB", "").strip()
if _assumed_ram:
    try:
        LLM_ASSUMED_RAM_GB = float(_assumed_ram)
    except ValueError:
        LLM_ASSUMED_RAM_GB = None

LLM_ALLOW_ANY_OLLAMA_MODEL: bool = os.getenv("LLM_ALLOW_ANY_OLLAMA_MODEL", "false").lower() == "true"
ENABLE_HOSTED_LLM_UI: bool = os.getenv("ENABLE_HOSTED_LLM_UI", "false").lower() == "true"
LLM_USER_SETTINGS_PATH: str = os.getenv("LLM_USER_SETTINGS_PATH", "").strip()
# Cap concurrent LLM-heavy tasks in batch scoring.
# Default 15: with the global Ollama call semaphore (LLM_OLLAMA_CALL_CONCURRENCY) now
# controlling actual GPU/CPU serialization, the resume-level cap can be higher to allow
# maximum pipelining of CPU-bound work while the model is busy with another resume.
_mc_raw = os.getenv("LLM_MAX_CONCURRENT_RESUMES", "15").strip()
try:
    LLM_MAX_CONCURRENT_RESUMES: int = max(0, int(_mc_raw))
except ValueError:
    LLM_MAX_CONCURRENT_RESUMES = 10

# Number of concurrent Ollama inference calls allowed.
# Match this to OLLAMA_NUM_PARALLEL in your Ollama server config:
#   8GB  RAM: OLLAMA_NUM_PARALLEL=1 → set this to 1
#   16GB RAM: OLLAMA_NUM_PARALLEL=2 → set this to 2
#   32GB RAM: OLLAMA_NUM_PARALLEL=2 → set this to 2
# Hosted OpenAI-compatible APIs: set to 0 (unlimited) or match your RPM limit.
# Setting higher than OLLAMA_NUM_PARALLEL causes unnecessary queuing in Ollama.
_oc_raw = os.getenv("LLM_OLLAMA_CALL_CONCURRENCY", "1").strip()
try:
    LLM_OLLAMA_CALL_CONCURRENCY: int = max(1, int(_oc_raw))
except ValueError:
    LLM_OLLAMA_CALL_CONCURRENCY = 1

# Partial match: skill similarity >= this threshold counts as partial match
SEMANTIC_SKILL_THRESHOLD: float = _clamp_float_env("SEMANTIC_SKILL_THRESHOLD", "0.75", 0.0, 1.0)
# Full match: skill similarity >= this counts as full match
SEMANTIC_FULL_MATCH_THRESHOLD: float = _clamp_float_env("SEMANTIC_FULL_MATCH_THRESHOLD", "0.9", 0.0, 1.0)
# Offline TF-IDF cosine threshold for fuzzy skill matching in the regex path.
# Skills with cosine >= this are treated as equivalent (e.g. "React.js" ≈ "React").
SKILL_COSINE_FULL_MATCH: float = _clamp_float_env("SKILL_COSINE_FULL_MATCH", "0.60", 0.3, 1.0)
# Skills with cosine >= this but < FULL are partial credit (0.7×).
SKILL_COSINE_PARTIAL_MATCH: float = _clamp_float_env("SKILL_COSINE_PARTIAL_MATCH", "0.40", 0.2, 1.0)
# Max chars sent to embedding API (longer text is truncated; document this for callers)
SIMILARITY_EMBED_MAX_CHARS: int = int(os.getenv("SIMILARITY_EMBED_MAX_CHARS", "8000"))
# Score normalization bounds: (low, high) per dimension; raw scores clamped then mapped to 0-100
# Format: "similarity:20,90" etc. — parsed in section_scorer
DIMENSION_BOUNDS_OVERRIDE: str = os.getenv("DIMENSION_BOUNDS_OVERRIDE", "")

# LLM timeout configuration per analysis type (seconds)
LLM_ANALYSIS_TIMEOUT: int = int(os.getenv("LLM_ANALYSIS_TIMEOUT", "1200"))

# --- AI-primary scoring (unified LLM dimension scorer) ---
AI_PRIMARY_SCORING: bool = os.getenv("AI_PRIMARY_SCORING", "true").lower() == "true"
SCORE_DIVERGENCE_THRESHOLD: float = float(os.getenv("SCORE_DIVERGENCE_THRESHOLD", "30"))

# Resumes whose unified score is below this threshold skip the deep enricher LLM call
# (experience, achievements, trajectory, fit, red flags). This halves LLM calls for
# weak candidates who clearly won't be shortlisted. Set to 0 to disable.
# Raised from 35 to 50: candidates below 50 are already "Needs Review"/"Weak Match"
# and benefit less from expensive enricher analysis.
ENRICHER_SCORE_THRESHOLD: float = float(os.getenv("ENRICHER_SCORE_THRESHOLD", "50"))

# Maximum additive post-blend inference credit (in score points, 0-100 scale).
# Applied after the LLM/regex blending step to avoid the divergence paradox.
# Default 15: large enough to close the ~8-pt gap between a strong-but-contextual
# senior and a keyword-optimized junior, yet conservative enough to avoid
# pathological over-inflation for resumes that game transferable inferences.
MAX_INFERENCE_BOOST: float = float(os.getenv("MAX_INFERENCE_BOOST", "15"))

# --- CORS & Security (env-driven for deploy) ---
# Comma-separated list of allowed origins (e.g. http://localhost:8000,https://app.example.com)
CORS_ORIGINS: list[str] = [
    o.strip() for o in os.getenv("CORS_ORIGINS", "http://localhost:8899,http://127.0.0.1:8899,http://localhost:8000,http://127.0.0.1:8000").split(",")
    if o.strip()
]

# Rate limiting: max requests per window for POST /api/analyze and POST /api/review (per IP)
# Production may use nginx or API gateway for stricter or per-tenant limits.
RATE_LIMIT_REQUESTS: int = int(os.getenv("RATE_LIMIT_REQUESTS", "20"))
RATE_LIMIT_WINDOW_SEC: int = int(os.getenv("RATE_LIMIT_WINDOW_SEC", "60"))

# --- Scoring reference (all thresholds in one place) ---
# Hiring bands: _hiring_decision() in scoring_engine.py (configurable in Phase 1)
#   Strong Match >= 80, Potential >= 65, Needs Review >= 50, Weak < 50
# Normalization bounds: section_scorer._DIMENSION_BOUNDS_DEFAULT, override via DIMENSION_BOUNDS_OVERRIDE
# Semantic thresholds: SEMANTIC_SKILL_THRESHOLD (0.75 partial), SEMANTIC_FULL_MATCH_THRESHOLD (0.9 full)
# Modifiers: achievement_analyzer get_achievement_bonus (-5 to +5), red_flag_detector get_red_flag_penalty (-15 to +2), fit_analyzer get_fit_modifier (-5 to +5)
# Combined modifier clamped to [-15, +10] in scoring_engine._apply_enhanced_unified
# AI_PRIMARY_SCORING / SCORE_DIVERGENCE_THRESHOLD: see scoring_engine unified path


def _merge_no_proxy(existing: str, extra_hosts: str) -> str:
    parts: list[str] = []
    seen: set[str] = set()
    for segment in f"{existing},{extra_hosts}".replace(";", ",").split(","):
        h = segment.strip()
        if h and h not in seen:
            seen.add(h)
            parts.append(h)
    return ",".join(parts)


_EXTRA_NO_PROXY_LOOPBACK = "localhost,127.0.0.1,::1"


def _ensure_localhost_no_proxy() -> None:
    """Merge loopback hosts into NO_PROXY/no_proxy so httpx (Ollama) skips corporate proxies.

    When HTTP(S)_PROXY is set, traffic to 127.0.0.1 must bypass the proxy; urllib/httpx
    read NO_PROXY and no_proxy from the environment.
    """
    for key in ("NO_PROXY", "no_proxy"):
        cur = os.environ.get(key)
        if cur is None:
            os.environ[key] = _EXTRA_NO_PROXY_LOOPBACK
        else:
            os.environ[key] = _merge_no_proxy(cur, _EXTRA_NO_PROXY_LOOPBACK)


_ensure_localhost_no_proxy()
