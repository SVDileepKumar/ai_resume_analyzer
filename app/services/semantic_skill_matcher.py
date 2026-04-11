"""Semantic skill matching using embeddings and LLM.

.. deprecated:: 2026-04-09
    SemanticSkillMatcher is superseded by ``extract_skills_llm()`` in
    ``app.services.llm_service``, which provides LLM-native skill extraction
    with transferable inference in a single structured call.
    This module is retained for the ``ENABLE_SEMANTIC_SKILLS=False`` gate
    and will be removed in a future release.

Replaces regex-based skill matching with embedding-powered semantic matching.
Understands skill variations (React vs React.js), detects proficiency levels,
and identifies transferable skills.
"""

from __future__ import annotations

import asyncio
import json
import logging
import re as _re
import warnings
from typing import Any

import numpy as np
from pydantic import BaseModel, Field, ValidationError, field_validator

from app.config import (
    ENABLE_SEMANTIC_SKILLS,
    SEMANTIC_SKILL_THRESHOLD,
    SEMANTIC_FULL_MATCH_THRESHOLD,
    JD_MAX_CHARS_LLM,
    RESUME_MAX_CHARS_LLM,
    LLM_BACKEND,
    OLLAMA_EMBED_MODEL,
    OPENAI_EMBED_MODEL,
    OLLAMA_NUM_PREDICT_JSON_HEAVY,
    OLLAMA_NUM_PREDICT_JD_SKILLS,
)
from app.services.llm_service import ai_enabled, _chat_json, _parse_model
from app.services.similarity import get_embeddings_batch_async
from app.services.skill_matcher import (
    ALL_KNOWN_SKILLS,
    get_skills_that_satisfy_requirement,
)

logger = logging.getLogger(__name__)

warnings.warn(
    "SemanticSkillMatcher (app.services.semantic_skill_matcher) is deprecated as of "
    "2026-04-09 and will be removed in a future release. "
    "Use extract_skills_llm() from app.services.llm_service instead.",
    DeprecationWarning,
    stacklevel=2,
)


# ---------------------------------------------------------------------------
# Pydantic Models for LLM Structured Output
# ---------------------------------------------------------------------------

class ExtractedSkill(BaseModel):
    """A skill extracted from text with context."""
    name: str = Field(description="Normalized skill name (e.g., 'React' not 'ReactJS')")
    proficiency: str = Field(
        default="intermediate",
        description="One of: basic, intermediate, advanced, expert"
    )
    years: float | None = Field(
        default=None,
        description="Years of experience with this skill if mentioned"
    )
    context: str = Field(
        default="",
        description="Brief quote showing how skill was used"
    )

    @field_validator("years", mode="before")
    @classmethod
    def _coerce_years(cls, v: object) -> float | None:
        if v is None:
            return None
        if isinstance(v, (int, float)):
            return float(v)
        s = _re.sub(r"[+~\s].*$", "", str(v).strip())
        try:
            return float(s)
        except ValueError:
            return None

    @field_validator("proficiency", mode="before")
    @classmethod
    def _coerce_proficiency(cls, v: object) -> str:
        valid = {"basic", "intermediate", "advanced", "expert"}
        s = str(v).lower().strip() if v else "intermediate"
        return s if s in valid else "intermediate"


class ExtractedSkills(BaseModel):
    """Collection of skills extracted from a document."""
    skills: list[ExtractedSkill] = Field(default_factory=list)


class JDRequiredSkill(BaseModel):
    """A skill required by a job description."""
    name: str = Field(description="Skill name")
    importance: str = Field(
        default="required",
        description="One of: required, preferred, nice_to_have"
    )
    min_years: float | None = Field(
        default=None,
        description="Minimum years required if specified"
    )

    @field_validator("min_years", mode="before")
    @classmethod
    def _coerce_min_years(cls, v: object) -> float | None:
        if v is None:
            return None
        if isinstance(v, (int, float)):
            return float(v)
        s = _re.sub(r"[+~\s].*$", "", str(v).strip())
        try:
            return float(s)
        except ValueError:
            return None

    @field_validator("importance", mode="before")
    @classmethod
    def _coerce_importance(cls, v: object) -> str:
        valid = {"required", "preferred", "nice_to_have"}
        s = str(v).lower().strip().replace(" ", "_").replace("-", "_") if v else "required"
        return s if s in valid else "required"


class JDSkillRequirements(BaseModel):
    """Skills extracted from a job description."""
    skills: list[JDRequiredSkill] = Field(default_factory=list)


class PartialSkillMatch(BaseModel):
    """A partial/transferable skill match."""
    required: str
    found: str
    similarity: float
    note: str = Field(default="", description="Why this is a partial match")


class SemanticSkillResult(BaseModel):
    """Result of semantic skill matching."""
    matched: list[str] = Field(default_factory=list)
    partial: list[PartialSkillMatch] = Field(default_factory=list)
    missing: list[str] = Field(default_factory=list)
    extra: list[str] = Field(default_factory=list)
    proficiency_levels: dict[str, str] = Field(default_factory=dict)
    score: float = Field(default=0.0)
    method: str = Field(default="semantic")


# ---------------------------------------------------------------------------
# Skill Embedding Cache (async batch init: 1 HTTP call for all skills)
# ---------------------------------------------------------------------------

_SKILL_EMBEDDINGS: dict[str, list[float]] = {}
_SKILL_EMBEDDINGS_KEY: str | None = None
_SKILL_EMBEDDINGS_LOCK = asyncio.Lock()


def _skill_embedding_cache_key() -> str:
    """Partition cache by backend + embed model so vectors stay in one space."""
    model = OPENAI_EMBED_MODEL if LLM_BACKEND == "openai" else OLLAMA_EMBED_MODEL
    return f"{LLM_BACKEND}:{model}"


async def _init_skill_embeddings_async() -> None:
    """Pre-compute embeddings for all skills in skill_db.json using a single batch call.

    Replaces the old sync loop (332 individual HTTP calls) with one batched request.
    Protected by a lock so concurrent requests don't double-init.
    """
    global _SKILL_EMBEDDINGS, _SKILL_EMBEDDINGS_KEY

    if not ai_enabled():
        return

    key = _skill_embedding_cache_key()
    if _SKILL_EMBEDDINGS and _SKILL_EMBEDDINGS_KEY == key:
        return

    async with _SKILL_EMBEDDINGS_LOCK:
        # Re-check inside lock in case another coroutine just finished
        if _SKILL_EMBEDDINGS and _SKILL_EMBEDDINGS_KEY == key:
            return

        _SKILL_EMBEDDINGS = {}
        _SKILL_EMBEDDINGS_KEY = key

        skills_list = list(ALL_KNOWN_SKILLS)
        logger.info("Initializing skill embeddings for %d skills (batch)...", len(skills_list))

        embeddings = await get_embeddings_batch_async(skills_list)

        for skill, emb in zip(skills_list, embeddings):
            if emb:
                _SKILL_EMBEDDINGS[skill] = emb

        logger.info("Cached %d skill embeddings", len(_SKILL_EMBEDDINGS))


def _cosine_similarity(vec_a: list[float], vec_b: list[float]) -> float:
    """Compute cosine similarity between two vectors."""
    a = np.array(vec_a)
    b = np.array(vec_b)
    dot = np.dot(a, b)
    norm = np.linalg.norm(a) * np.linalg.norm(b)
    return float(dot / norm) if norm > 0 else 0.0


# ---------------------------------------------------------------------------
# LLM System Prompts
# ---------------------------------------------------------------------------

_SKILL_EXTRACTOR_SYSTEM = (
    "You are an expert technical recruiter and resume parser. "
    "Extract technical skills from text with proficiency assessment. "
    "Normalize skill names (ReactJS → React, NodeJS → Node.js). "
    "Proficiency levels based on context: "
    "- expert: led, architected, 8+ years, mentored, designed systems "
    "- advanced: implemented, optimized, 5-7 years, complex projects "
    "- intermediate: developed, built, 2-4 years, contributed "
    "- basic: familiar, exposure, coursework, learning, <2 years"
)

_JD_SKILL_EXTRACTOR_SYSTEM = (
    "You are an expert technical recruiter. "
    "Extract skill requirements from job descriptions. "
    "Classify importance: "
    "- required: 'must have', 'required', 'essential', without qualifiers "
    "- preferred: 'preferred', 'strongly preferred', 'ideally' "
    "- nice_to_have: 'nice to have', 'bonus', 'plus', 'optional'"
)


def _parse_extracted_skills(data: Any) -> list[ExtractedSkill]:
    """Parse resume skill payloads item-by-item so one bad entry does not poison the batch."""
    if not isinstance(data, dict):
        return []
    raw_skills = data.get("skills")
    if not isinstance(raw_skills, list):
        return []

    parsed: list[ExtractedSkill] = []
    dropped = 0
    for item in raw_skills:
        try:
            parsed.append(ExtractedSkill.model_validate(item))
        except ValidationError:
            dropped += 1

    if dropped:
        logger.warning("Dropped %d malformed extracted resume skills", dropped)
    return parsed


def _parse_jd_required_skills(data: Any) -> list[JDRequiredSkill]:
    """Parse JD skill payloads item-by-item so one bad entry does not discard valid skills."""
    if not isinstance(data, dict):
        return []
    raw_skills = data.get("skills")
    if not isinstance(raw_skills, list):
        return []

    parsed: list[JDRequiredSkill] = []
    dropped = 0
    for item in raw_skills:
        try:
            parsed.append(JDRequiredSkill.model_validate(item))
        except ValidationError:
            dropped += 1

    if dropped:
        logger.warning("Dropped %d malformed extracted JD skills", dropped)
    return parsed


# ---------------------------------------------------------------------------
# LLM Extraction Functions
# ---------------------------------------------------------------------------

async def _extract_resume_skills_llm(resume_text: str) -> list[ExtractedSkill]:
    """Use LLM to extract skills WITH proficiency levels from resume."""

    schema = {
        "skills": [
            {
                "name": "skill_name (normalized)",
                "proficiency": "basic|intermediate|advanced|expert",
                "years": "number or null",
                "context": "brief quote showing usage"
            }
        ]
    }

    prompt = f"""Analyze this resume and extract ALL technical skills with proficiency levels.

## Resume
{resume_text}

## Instructions
For each skill found:
1. Normalize the skill name (ReactJS → React, NodeJS → Node.js)
2. Determine proficiency from context clues
3. Extract years if mentioned
4. Note context where skill was used — **each context must be at most 120 characters** (truncate if needed)

Focus on technical skills, programming languages, frameworks, tools, and platforms.
Return at most **60** skills (prioritize the most relevant). Omit duplicate or near-duplicate skills.
"""

    data = await _chat_json(
        _SKILL_EXTRACTOR_SYSTEM,
        prompt,
        schema,
        num_predict=OLLAMA_NUM_PREDICT_JSON_HEAVY,
    )
    if data is None:
        return []

    skills = _parse_extracted_skills(data)
    if skills:
        return skills
    logger.warning("Failed to parse LLM skill extraction: %s", data)
    return []


async def extract_jd_skills_llm(jd_text: str) -> list[JDRequiredSkill]:
    """Use LLM to extract required skills from job description."""

    schema = {
        "skills": [
            {
                "name": "skill_name",
                "importance": "required|preferred|nice_to_have",
                "min_years": "number or null"
            }
        ]
    }

    prompt = f"""Extract all technical skill requirements from this job description.

## Job Description
{jd_text[:JD_MAX_CHARS_LLM]}

## Instructions
For each skill:
1. Identify the skill name
2. Classify importance (required/preferred/nice_to_have)
3. Note minimum years if specified

Include: programming languages, frameworks, tools, platforms, methodologies.
Return at most **40** distinct skills (merge synonyms).
"""

    data = await _chat_json(
        _JD_SKILL_EXTRACTOR_SYSTEM,
        prompt,
        schema,
        num_predict=OLLAMA_NUM_PREDICT_JD_SKILLS,
    )
    if data is None:
        return []

    skills = _parse_jd_required_skills(data)
    if skills:
        return skills
    logger.warning("Failed to parse LLM JD skill extraction: %s", data)
    return []


# ---------------------------------------------------------------------------
# Semantic Matching Core
# ---------------------------------------------------------------------------

async def semantic_skill_match(
    resume_text: str,
    jd_text: str,
    threshold: float | None = None,
    jd_skills_preextracted: list[JDRequiredSkill] | None = None,
) -> SemanticSkillResult:
    """Match skills semantically using batch embeddings and LLM (AI-only, no regex fallback).

    Default path runs two Ollama JSON completions (JD + resume skills) in parallel unless
    ``jd_skills_preextracted`` is supplied. All embeddings are fetched in a single batch call.

    Args:
        resume_text: The candidate's resume text
        jd_text: The job description text
        threshold: Similarity threshold for partial matches (default from config)
        jd_skills_preextracted: If set, skip per-call JD LLM extraction (batch mode).

    Returns:
        SemanticSkillResult with matched, partial, missing skills and score.
        Returns an empty result (score=0) if AI is unavailable.
    """
    if threshold is None:
        threshold = SEMANTIC_SKILL_THRESHOLD

    if not ENABLE_SEMANTIC_SKILLS or not ai_enabled():
        logger.warning("Semantic skill match skipped: AI not available")
        return SemanticSkillResult(score=0.0, method="unavailable")

    if not (jd_text and jd_text.strip()):
        logger.warning("Empty JD text for semantic skill match")
        return SemanticSkillResult(score=0.0, method="unavailable")

    # Ensure skill embedding cache is warm (single batch call)
    await _init_skill_embeddings_async()

    if jd_skills_preextracted is not None:
        jd_skills = jd_skills_preextracted
        resume_skills = await _extract_resume_skills_llm(resume_text)
    else:
        logger.info("Semantic skill match: extracting JD + resume skills in parallel")
        jd_skills, resume_skills = await asyncio.gather(
            extract_jd_skills_llm(jd_text),
            _extract_resume_skills_llm(resume_text),
        )

    if not jd_skills:
        logger.warning("JD skill extraction returned no skills")
        return SemanticSkillResult(score=0.0, method="unavailable")

    # Build lookup for resume skills by normalized name
    resume_skill_map: dict[str, ExtractedSkill] = {
        s.name.lower().strip(): s for s in resume_skills
    }

    # Batch-embed all distinct resume skill names in ONE API call
    distinct_resume_names = list({(s.name or "").strip() for s in resume_skills if s.name})
    if distinct_resume_names:
        batch_vecs = await get_embeddings_batch_async(distinct_resume_names)
        resume_emb_by_name: dict[str, list[float]] = {
            name: vec
            for name, vec in zip(distinct_resume_names, batch_vecs)
            if vec is not None
        }
    else:
        resume_emb_by_name = {}

    # Batch-embed all unresolved JD skill names (those needing semantic comparison)
    # We can pre-filter: only skills that didn't match directly or via implication need embeddings
    needs_embedding: list[str] = []
    direct_matches: dict[str, tuple[str, str]] = {}  # jd_skill_name → (matched_name, proficiency)

    for jd_skill in jd_skills:
        req_name = jd_skill.name.lower().strip()
        if req_name in resume_skill_map:
            direct_matches[jd_skill.name] = (
                jd_skill.name,
                resume_skill_map[req_name].proficiency,
            )
            continue
        satisfies = get_skills_that_satisfy_requirement(jd_skill.name)
        implied = satisfies & set(resume_skill_map.keys())
        if implied:
            first = next(iter(implied))
            direct_matches[jd_skill.name] = (jd_skill.name, resume_skill_map[first].proficiency)
            continue
        needs_embedding.append(jd_skill.name)

    jd_emb_by_name: dict[str, list[float]] = {}
    if needs_embedding:
        jd_vecs = await get_embeddings_batch_async(needs_embedding)
        jd_emb_by_name = {
            name: vec
            for name, vec in zip(needs_embedding, jd_vecs)
            if vec is not None
        }

    # Match each JD requirement
    matched: list[str] = []
    partial: list[PartialSkillMatch] = []
    missing: list[str] = []
    proficiency_levels: dict[str, str] = {}

    for jd_skill in jd_skills:
        if jd_skill.name in direct_matches:
            _, prof = direct_matches[jd_skill.name]
            matched.append(jd_skill.name)
            proficiency_levels[jd_skill.name] = prof
            continue

        req_emb = jd_emb_by_name.get(jd_skill.name)
        if not req_emb:
            missing.append(jd_skill.name)
            continue

        best_match: ExtractedSkill | None = None
        best_score = 0.0
        for resume_skill in resume_skills:
            res_emb = resume_emb_by_name.get((resume_skill.name or "").strip())
            if res_emb:
                sim = _cosine_similarity(req_emb, res_emb)
                if sim > best_score:
                    best_score = sim
                    best_match = resume_skill

        full_match_threshold = max(SEMANTIC_FULL_MATCH_THRESHOLD, threshold)
        if best_score >= full_match_threshold:
            matched.append(jd_skill.name)
            if best_match:
                proficiency_levels[jd_skill.name] = best_match.proficiency
        elif best_score >= threshold and best_match:
            partial.append(PartialSkillMatch(
                required=jd_skill.name,
                found=best_match.name,
                similarity=round(best_score, 3),
                note=f"Transferable: {best_match.name} → {jd_skill.name}",
            ))
        else:
            missing.append(jd_skill.name)

    # Extra skills (in resume but not required by JD)
    required_names = {s.name.lower().strip() for s in jd_skills}
    matched_names = {s.lower() for s in matched}
    partial_found = {p.found.lower() for p in partial}
    extra = [
        s.name for s in resume_skills
        if s.name.lower().strip() not in required_names
        and s.name.lower().strip() not in matched_names
        and s.name.lower().strip() not in partial_found
    ][:20]

    # Score: 65% required, 20% preferred, 10% partial credit, 5% extra-skill depth.
    # When a category has zero skills in the JD, its bucket is collapsed (weight → 0)
    # so missing categories don't give free points or penalize the candidate.
    required_count = sum(1 for s in jd_skills if s.importance == "required")
    preferred_count = sum(1 for s in jd_skills if s.importance == "preferred")
    nice_count = sum(1 for s in jd_skills if s.importance == "nice_to_have")
    matched_required = sum(1 for s in jd_skills if s.importance == "required" and s.name in matched)
    matched_preferred = sum(1 for s in jd_skills if s.importance == "preferred" and s.name in matched)
    matched_nice = sum(1 for s in jd_skills if s.importance == "nice_to_have" and s.name in matched)

    # Partial matches contribute weighted credit based on similarity
    partial_credit = sum(p.similarity for p in partial)
    total_required = len(jd_skills) or 1

    # When required_count == 0 use 0.0 (not 1.0) — avoid awarding 65 free points
    required_score = matched_required / required_count if required_count > 0 else 0.0
    # When preferred/nice counts are 0, exclude their weight from the score (rescale instead)
    preferred_score = matched_preferred / preferred_count if preferred_count > 0 else None
    nice_score = matched_nice / nice_count if nice_count > 0 else None

    # Dynamic weights: drop buckets with no JD skills, keep proportions
    req_weight = 65 if required_count > 0 else 0
    pref_weight = 18 if preferred_count > 0 else 0
    nice_weight = 2 if nice_count > 0 else 0
    active_weight = req_weight + pref_weight + nice_weight or 85  # fallback to 85 to avoid /0

    # Depth bonus: having many extra relevant skills signals strong technical breadth
    depth_bonus = min(len(extra) * 0.5, 5.0) if extra else 0.0

    score = (
        required_score * req_weight / active_weight * 85
        + (preferred_score or 0.0) * pref_weight / active_weight * 85
        + (nice_score or 0.0) * nice_weight / active_weight * 85
        + (partial_credit / total_required) * 10
        + depth_bonus
    )
    score = min(score, 100.0)

    result = SemanticSkillResult(
        matched=matched,
        partial=partial,
        missing=missing,
        extra=extra,
        proficiency_levels=proficiency_levels,
        score=round(score, 1),
        method="semantic",
    )
    if result.proficiency_levels:
        result = result.model_copy(
            update={"score": adjust_score_for_proficiency(result, jd_skills)}
        )
    return result


# ---------------------------------------------------------------------------
# Proficiency-Adjusted Scoring
# ---------------------------------------------------------------------------

def adjust_score_for_proficiency(
    result: SemanticSkillResult,
    jd_skills: list[JDRequiredSkill],
) -> float:
    """Adjust skill score based on proficiency levels.

    Expert proficiency gives bonus points, basic gives penalty.
    """
    if not result.proficiency_levels:
        return result.score

    proficiency_multipliers = {
        "expert": 1.15,
        "advanced": 1.05,
        "intermediate": 1.0,
        "basic": 0.85,
    }

    adjustments = []
    for skill in result.matched:
        prof = result.proficiency_levels.get(skill, "intermediate")
        multiplier = proficiency_multipliers.get(prof, 1.0)
        adjustments.append(multiplier)

    if not adjustments:
        return result.score

    avg_multiplier = sum(adjustments) / len(adjustments)
    adjusted = result.score * avg_multiplier

    return round(min(adjusted, 100), 1)
