"""LLM-powered experience relevance analysis.

Evaluates if candidate experience is RELEVANT to JD, not just counting years.
Understands domain relevance, technology currency, and career progression.
"""

from __future__ import annotations

import logging
from typing import Any

import re as _re

from pydantic import BaseModel, Field, ValidationError, field_validator

from app.config import ENABLE_LLM_EXPERIENCE, JD_MAX_CHARS_LLM, RESUME_MAX_CHARS_LLM
from app.services.llm_service import ai_enabled, _chat_json, _parse_model

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Pydantic Models
# ---------------------------------------------------------------------------

def _parse_years(v: object) -> float:
    """Parse year values from LLM output — handles floats, '9+', '5.5', etc."""
    if isinstance(v, (int, float)):
        return float(v)
    s = str(v).strip()
    # Strip trailing '+', '~', spaces and extract first numeric part
    s = _re.sub(r"[+~\s].*$", "", s)
    try:
        return float(s)
    except ValueError:
        return 0.0


class ExperienceAnalysis(BaseModel):
    """Structured analysis of candidate experience relevance."""
    relevance_score: float = Field(
        description="0-100 score for JD relevance"
    )
    total_years: float = Field(
        default=0.0,
        description="Total years of professional experience"
    )
    relevant_years: float = Field(
        default=0.0,
        description="Years directly relevant to this JD"
    )

    @field_validator("total_years", "relevant_years", mode="before")
    @classmethod
    def _coerce_years(cls, v: object) -> float:
        return _parse_years(v)

    @field_validator("relevance_score", mode="before")
    @classmethod
    def _coerce_relevance(cls, v: object) -> float:
        if isinstance(v, (int, float)):
            return float(v)
        return float(str(v).replace("%", "").strip().split()[0])
    career_progression: str = Field(
        default="mixed",
        description="One of: ascending, lateral, descending, mixed"
    )
    domain_match: str = Field(
        default="different",
        description="One of: exact, related, different"
    )
    technology_currency: str = Field(
        default="current",
        description="One of: current, slightly_outdated, outdated"
    )
    highlights: list[str] = Field(
        default_factory=list,
        description="Top 3 most relevant experiences"
    )
    concerns: list[str] = Field(
        default_factory=list,
        description="Gaps or concerns about experience"
    )
    verdict: str = Field(
        default="adequate",
        description="One of: strong, adequate, weak"
    )


# ---------------------------------------------------------------------------
# System Prompt
# ---------------------------------------------------------------------------

_EXPERIENCE_ANALYZER_SYSTEM = (
    "You are a senior technical recruiter with 15+ years experience. "
    "Analyze candidate work experience for relevance to specific job requirements. "
    "Weight recent experience (last 3 years) more heavily. "
    "Consider: role similarity, tech stack overlap, domain expertise, seniority match. "
    "Be precise and specific - reference actual experience from the resume."
)


# ---------------------------------------------------------------------------
# Main Analysis Function
# ---------------------------------------------------------------------------

async def analyze_experience_relevance(
    resume_text: str,
    jd_text: str,
) -> ExperienceAnalysis | None:
    """Use LLM to deeply analyze experience relevance.

    Args:
        resume_text: The candidate's resume text
        jd_text: The job description text

    Returns:
        ExperienceAnalysis with detailed relevance scoring, or None if unavailable
    """
    if not ENABLE_LLM_EXPERIENCE or not ai_enabled():
        return None

    schema = {
        "relevance_score": "0-100 score for JD relevance",
        "total_years": "total years of professional experience",
        "relevant_years": "years directly relevant to this JD",
        "career_progression": "ascending|lateral|descending|mixed",
        "domain_match": "exact|related|different",
        "technology_currency": "current|slightly_outdated|outdated",
        "highlights": ["top 3 most relevant experiences"],
        "concerns": ["gaps or concerns"],
        "verdict": "strong|adequate|weak"
    }

    prompt = f"""Analyze the candidate's work experience for relevance to this specific job.

## Job Description
{jd_text[:JD_MAX_CHARS_LLM]}

## Resume
{resume_text}

## Analysis Required

### 1. RELEVANCE SCORING (0-100)
- How well does their experience align with what this job needs?
- Weight recent experience (last 3 years) more heavily
- Consider: role similarity, tech stack overlap, domain expertise, seniority match
- 80-100: Strong match, directly relevant experience
- 60-79: Good match, mostly relevant with some gaps
- 40-59: Moderate match, some relevant experience
- 20-39: Weak match, limited relevant experience
- 0-19: Poor match, experience doesn't align

### 2. CAREER PROGRESSION
- ascending: Junior → Mid → Senior (positive signal for growth)
- lateral: Same level roles, different companies (neutral)
- descending: Senior → Mid (yellow flag - investigate why)
- mixed: Unclear trajectory or role pivots

### 3. DOMAIN MATCH
- exact: Same industry (e.g., fintech → fintech company)
- related: Adjacent industry (e.g., fintech → banking, insurance)
- different: Unrelated industry (e.g., fintech → gaming, retail)

### 4. TECHNOLOGY CURRENCY
Are their skills current or outdated for 2025?
- current: Modern tech stack within last 2-3 years (React 18, Python 3.10+, K8s, modern cloud)
- slightly_outdated: Slightly older versions (React 16-17, Python 3.7-3.9, older patterns)
- outdated: Legacy tech (jQuery as primary, Python 2, PHP 5, no cloud experience)

### 5. HIGHLIGHTS
Quote the top 3 experiences most relevant to this JD.
Be specific - mention actual projects, achievements, or responsibilities.

### 6. CONCERNS
Note any red flags, gaps, or mismatches discovered.
Examples: employment gaps, missing expected skills, short tenures, title mismatches.

### 7. VERDICT
- strong: Highly relevant, meets or exceeds requirements
- adequate: Relevant enough to consider, some gaps to address
- weak: Not well-matched, significant gaps

Respond with valid JSON only."""

    data = await _chat_json(_EXPERIENCE_ANALYZER_SYSTEM, prompt, schema)
    if data is None:
        return None
    if not isinstance(data, dict) or "relevance_score" not in data:
        logger.warning("Experience analysis LLM returned empty or incomplete payload")
        return None

    return _parse_model(ExperienceAnalysis, data)


def determine_match_type(analysis: ExperienceAnalysis) -> str:
    """Determine overqualified/underqualified from experience analysis.

    Returns: 'overqualified', 'underqualified', or 'exact'
    """
    # Use relevance score and verdict to determine fit
    if analysis.relevance_score >= 85 and analysis.verdict == "strong":
        # Could be overqualified if senior applying for mid-level
        if analysis.total_years > analysis.relevant_years + 5:
            return "overqualified"
        return "exact"
    elif analysis.relevance_score < 50 or analysis.verdict == "weak":
        return "underqualified"
    else:
        return "exact"


def get_experience_metadata(analysis: ExperienceAnalysis) -> dict[str, Any]:
    """Extract metadata from experience analysis for UI display."""
    return {
        "match_type": determine_match_type(analysis),
        "career_progression": analysis.career_progression,
        "domain_match": analysis.domain_match,
        "technology_currency": analysis.technology_currency,
        "highlights": analysis.highlights[:3],
        "concerns": analysis.concerns[:3],
        "total_years": analysis.total_years,
        "relevant_years": analysis.relevant_years,
        "verdict": analysis.verdict,
    }
