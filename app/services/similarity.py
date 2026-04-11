"""Semantic text similarity using AI embeddings with TF-IDF fallback.

Primary path: embeddings via Ollama ``/api/embed`` or OpenAI ``embeddings.create``.
Fallback path: TF-IDF cosine similarity (sklearn) when embeddings fail or are unavailable.

When embeddings succeed, blends semantic cosine (65%) + weighted Jaccard (35%).
When embeddings fail, blends TF-IDF cosine (60%) + weighted Jaccard (40%).
"""

from __future__ import annotations

import asyncio
import hashlib
import logging
import re

import httpx
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity as sklearn_cosine

from app.config import (
    LLM_BACKEND,
    OLLAMA_BASE_URL,
    OLLAMA_EMBED_MODEL,
    OLLAMA_TIMEOUT,
    OPENAI_EMBED_MODEL,
    SIMILARITY_EMBED_MAX_CHARS,
)

logger = logging.getLogger(__name__)

_OLLAMA_EMBED_BASE = OLLAMA_BASE_URL.replace("/v1", "")

# Cached async OpenAI clients for embedding — avoids creating a new TCP connection
# on every embedding call.  Keyed by sha256(api_key + base_url) same as llm_service.
_openai_embed_clients: dict[str, "AsyncOpenAI"] = {}  # type: ignore[name-defined]  # noqa: F821
_openai_embed_client_lock: asyncio.Lock | None = None


def _get_embed_client_lock() -> asyncio.Lock:
    global _openai_embed_client_lock
    if _openai_embed_client_lock is None:
        _openai_embed_client_lock = asyncio.Lock()
    return _openai_embed_client_lock


async def _get_openai_embed_client() -> "AsyncOpenAI":  # type: ignore[name-defined]  # noqa: F821
    from openai import AsyncOpenAI
    from app.services.llm_service import get_effective_openai_api_key, get_effective_openai_base_url

    api_key = get_effective_openai_api_key()
    base_url = get_effective_openai_base_url()
    cache_key = hashlib.sha256(f"{api_key}\n{base_url}".encode()).hexdigest()

    async with _get_embed_client_lock():
        client = _openai_embed_clients.get(cache_key)
        if client is None:
            client = AsyncOpenAI(api_key=api_key or "invalid", base_url=base_url, timeout=OLLAMA_TIMEOUT)
            if len(_openai_embed_clients) > 8:
                _openai_embed_clients.clear()
            _openai_embed_clients[cache_key] = client
        return client


# ---------------------------------------------------------------------------
# Synchronous single embedding (kept for backward-compat with non-async callers)
# ---------------------------------------------------------------------------

def _get_embedding(text: str) -> list[float] | None:
    """Get a single embedding vector (sync). Prefer async batch version in hot paths."""
    if LLM_BACKEND == "openai":
        return _get_embedding_openai(text)

    truncated = text[:SIMILARITY_EMBED_MAX_CHARS]
    try:
        resp = httpx.post(
            f"{_OLLAMA_EMBED_BASE}/api/embed",
            json={"model": OLLAMA_EMBED_MODEL, "input": truncated},
            timeout=OLLAMA_TIMEOUT,
        )
        resp.raise_for_status()
        data = resp.json()
        embeddings = data.get("embeddings")
        if embeddings and len(embeddings) > 0:
            return embeddings[0]
        return None
    except Exception as e:
        logger.warning("Ollama embedding failed: %s", getattr(e, "message", str(e)))
        return None


_openai_sync_embed_clients: dict[str, "OpenAI"] = {}  # type: ignore[name-defined]  # noqa: F821


def _get_openai_sync_embed_client() -> "OpenAI":  # type: ignore[name-defined]  # noqa: F821
    from openai import OpenAI
    from app.services.llm_service import get_effective_openai_api_key, get_effective_openai_base_url

    api_key = get_effective_openai_api_key()
    base_url = get_effective_openai_base_url()
    cache_key = hashlib.sha256(f"{api_key}\n{base_url}".encode()).hexdigest()
    if cache_key not in _openai_sync_embed_clients:
        if len(_openai_sync_embed_clients) > 4:
            _openai_sync_embed_clients.clear()
        _openai_sync_embed_clients[cache_key] = OpenAI(
            api_key=api_key or "invalid", base_url=base_url, timeout=OLLAMA_TIMEOUT
        )
    return _openai_sync_embed_clients[cache_key]


def _get_embedding_openai(text: str) -> list[float] | None:
    """Embedding vector via OpenAI SDK (sync, cached client)."""
    from app.services.llm_service import get_effective_openai_api_key

    if not get_effective_openai_api_key():
        return None

    truncated = text[:SIMILARITY_EMBED_MAX_CHARS]
    try:
        client = _get_openai_sync_embed_client()
        response = client.embeddings.create(model=OPENAI_EMBED_MODEL, input=truncated)
        if response.data:
            return list(response.data[0].embedding)
        return None
    except Exception as e:
        logger.warning("OpenAI embedding failed: %s", getattr(e, "message", str(e)))
        return None


# ---------------------------------------------------------------------------
# Async batch embedding (primary path for performance)
# ---------------------------------------------------------------------------

async def get_embeddings_batch_async(texts: list[str]) -> list[list[float] | None]:
    """Embed multiple texts in a single API call (async).

    Returns a list of embedding vectors (or None for failed items) in the same
    order as the input texts. Ollama /api/embed accepts a list input natively.
    """
    if not texts:
        return []

    if LLM_BACKEND == "openai":
        return await _get_embeddings_batch_openai(texts)

    truncated = [t[:SIMILARITY_EMBED_MAX_CHARS] for t in texts]
    try:
        from app.services.llm_service import _get_ollama_http_client
        client = await _get_ollama_http_client()
        resp = await client.post(
            f"{_OLLAMA_EMBED_BASE}/api/embed",
            json={"model": OLLAMA_EMBED_MODEL, "input": truncated},
        )
        resp.raise_for_status()
        data = resp.json()
        embeddings = data.get("embeddings") or []
        result: list[list[float] | None] = []
        for i in range(len(texts)):
            result.append(embeddings[i] if i < len(embeddings) else None)
        return result
    except Exception as e:
        logger.warning("Ollama batch embedding failed: %s", getattr(e, "message", str(e)))
        return [None] * len(texts)


async def _get_embeddings_batch_openai(texts: list[str]) -> list[list[float] | None]:
    """Batch embedding via OpenAI async SDK (cached client — no per-call TCP setup)."""
    from app.services.llm_service import get_effective_openai_api_key

    api_key = get_effective_openai_api_key()
    if not api_key:
        return [None] * len(texts)

    truncated = [t[:SIMILARITY_EMBED_MAX_CHARS] for t in texts]
    try:
        client = await _get_openai_embed_client()
        response = await client.embeddings.create(model=OPENAI_EMBED_MODEL, input=truncated)
        result: list[list[float] | None] = [None] * len(texts)
        for item in response.data:
            result[item.index] = list(item.embedding)
        return result
    except Exception as e:
        logger.warning("OpenAI batch embedding failed: %s", getattr(e, "message", str(e)))
        return [None] * len(texts)


async def get_embedding_async(text: str) -> list[float] | None:
    """Async single embedding — delegates to batch for code reuse."""
    results = await get_embeddings_batch_async([text])
    return results[0] if results else None


# ---------------------------------------------------------------------------
# Cosine similarity helpers
# ---------------------------------------------------------------------------

def _cosine_similarity_vecs(vec_a: list[float], vec_b: list[float]) -> float:
    a = np.array(vec_a)
    b = np.array(vec_b)
    dot = np.dot(a, b)
    norm = np.linalg.norm(a) * np.linalg.norm(b)
    return float(dot / norm) if norm > 0 else 0.0


# ---------------------------------------------------------------------------
# Jaccard overlap (lightweight lexical signal, combined with semantic)
# ---------------------------------------------------------------------------

_STOPWORDS = {
    "the", "and", "for", "with", "that", "this", "are", "was",
    "will", "from", "have", "has", "been", "our", "your", "you",
    "not", "but", "all", "can", "had", "her", "one", "who",
    "their", "there", "what", "about", "which", "when", "make",
    "like", "than", "each", "other", "into", "more", "some",
}


def _jaccard_similarity(text_a: str, text_b: str) -> float:
    def tokenize(t: str) -> set[str]:
        return set(re.findall(r"\b[a-z][a-z0-9+#.]{1,}\b", t.lower())) - _STOPWORDS

    a, b = tokenize(text_a), tokenize(text_b)
    if not a or not b:
        return 0.0
    return len(a & b) / len(a | b)


def _weighted_jaccard(resume_text: str, jd_text: str) -> float:
    """Section-aware Jaccard that weights skills/experience sections higher.

    Technical keywords and role-relevant terms count 2x toward overlap to
    better capture skill relevance vs boilerplate text overlap.
    """
    _technical_boost = {
        "python", "java", "javascript", "typescript", "react", "angular",
        "docker", "kubernetes", "aws", "azure", "gcp", "sql", "nosql",
        "machine learning", "api", "microservices", "devops", "agile",
        "terraform", "node.js", "django", "spring", "fastapi", "flask",
        "tensorflow", "pytorch", "redis", "kafka", "graphql", "ci/cd",
    }

    def tokenize(t: str) -> dict[str, float]:
        tokens = re.findall(r"\b[a-z][a-z0-9+#./]{1,}\b", t.lower())
        weights: dict[str, float] = {}
        for tok in tokens:
            if tok in _STOPWORDS:
                continue
            w = 2.0 if tok in _technical_boost else 1.0
            weights[tok] = max(weights.get(tok, 0), w)
        return weights

    a_weights = tokenize(resume_text)
    b_weights = tokenize(jd_text)
    all_tokens = set(a_weights) | set(b_weights)
    if not all_tokens:
        return 0.0

    intersection_sum = sum(
        min(a_weights.get(t, 0), b_weights.get(t, 0))
        for t in all_tokens
    )
    union_sum = sum(
        max(a_weights.get(t, 0), b_weights.get(t, 0))
        for t in all_tokens
    )
    return intersection_sum / union_sum if union_sum > 0 else 0.0


# ---------------------------------------------------------------------------
# TF-IDF fallback (no Ollama / OpenAI needed)
# ---------------------------------------------------------------------------

def _tfidf_cosine_similarity(text_a: str, text_b: str) -> float:
    """Compute TF-IDF cosine similarity between two documents.

    Used as fallback when embeddings are unavailable. Character 2-4 grams
    capture sub-word overlap (e.g. "reactjs" ≈ "react.js"). Returns 0–1.
    """
    try:
        vectorizer = TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 4), min_df=1)
        tfidf_matrix = vectorizer.fit_transform([text_a.lower(), text_b.lower()])
        return float(sklearn_cosine(tfidf_matrix[0], tfidf_matrix[1])[0, 0])
    except Exception:
        return 0.0


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

async def calculate_similarity_async(
    resume_text: str,
    jd_text: str,
) -> tuple[float, str]:
    """Return (score 0–100, method_used). Async — uses batch embedding.

    Falls back to TF-IDF cosine + weighted Jaccard when embeddings fail.
    """
    if not resume_text.strip() or not jd_text.strip():
        return 0.0, "none"

    vecs = await get_embeddings_batch_async([resume_text, jd_text])
    vec_a, vec_b = vecs[0], vecs[1]

    w_jaccard = _weighted_jaccard(resume_text, jd_text)

    if vec_a is None or vec_b is None:
        # Embedding unavailable — fall back to TF-IDF blend instead of returning 0.0
        tfidf_score = _tfidf_cosine_similarity(resume_text, jd_text)
        blended = (tfidf_score * 0.60) + (w_jaccard * 0.40)
        score = round(min(max(blended * 100, 0.0), 100.0), 2)
        return score, "tfidf_fallback"

    semantic = _cosine_similarity_vecs(vec_a, vec_b)
    blended = (semantic * 0.65) + (w_jaccard * 0.35)
    score = round(min(max(blended * 100, 0.0), 100.0), 2)
    return score, "semantic"


def calculate_similarity(
    resume_text: str,
    jd_text: str,
) -> tuple[float, str]:
    """Sync wrapper kept for code paths that cannot yet be made async.

    In AI-primary mode, this result is overridden by the unified LLM call so
    the blocking nature is acceptable. For pure-async callers, use
    calculate_similarity_async() instead.

    Falls back to TF-IDF cosine + weighted Jaccard when embeddings fail.
    """
    if not resume_text.strip() or not jd_text.strip():
        return 0.0, "none"

    w_jaccard = _weighted_jaccard(resume_text, jd_text)
    vec_a = _get_embedding(resume_text)
    vec_b = _get_embedding(jd_text) if vec_a is not None else None

    if vec_a is None or vec_b is None:
        # Embedding unavailable — fall back to TF-IDF blend instead of returning 0.0
        tfidf_score = _tfidf_cosine_similarity(resume_text, jd_text)
        blended = (tfidf_score * 0.60) + (w_jaccard * 0.40)
        score = round(min(max(blended * 100, 0.0), 100.0), 2)
        return score, "tfidf_fallback"

    semantic = _cosine_similarity_vecs(vec_a, vec_b)
    blended = (semantic * 0.65) + (w_jaccard * 0.35)
    score = round(min(max(blended * 100, 0.0), 100.0), 2)
    return score, "semantic"
