"""Multi-dimensional candidate fit analysis.

Scores candidates across multiple dimensions: technical, experience, domain,
seniority, culture indicators, and growth potential.
"""

from __future__ import annotations

import logging
from typing import Any

from pydantic import BaseModel, Field, ValidationError

from app.config import ENABLE_MULTI_DIM_FIT, JD_MAX_CHARS_LLM, RESUME_MAX_CHARS_LLM
from app.services.llm_service import ai_enabled, _chat_json, _parse_model

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Pydantic Models
# ---------------------------------------------------------------------------

class CandidateFit(BaseModel):
    """Comprehensive fit analysis across multiple dimensions."""

    # Core fit dimensions (0-100 each)
    technical_fit: float = Field(
        default=50.0,
        description="Technical skills match"
    )
    experience_fit: float = Field(
        default=50.0,
        description="Experience level & relevance"
    )
    domain_fit: float = Field(
        default=50.0,
        description="Industry/domain knowledge"
    )
    seniority_fit: float = Field(
        default=50.0,
        description="Seniority level alignment"
    )
    culture_indicators: float = Field(
        default=50.0,
        description="Soft skills & culture signals"
    )
    growth_potential: float = Field(
        default=50.0,
        description="Potential to grow in role"
    )

    # Composite scores
    overall_fit: float = Field(
        default=50.0,
        description="Weighted overall fit score"
    )
    hiring_confidence: str = Field(
        default="medium",
        description="One of: high, medium, low"
    )

    # Detailed insights
    strongest_dimensions: list[str] = Field(
        default_factory=list,
        description="Top 2 strongest fit areas"
    )
    weakest_dimensions: list[str] = Field(
        default_factory=list,
        description="Top 2 weakest fit areas"
    )
    unique_value: str = Field(
        default="",
        description="What this candidate uniquely brings"
    )
    risk_factors: list[str] = Field(
        default_factory=list,
        description="Potential risks in hiring"
    )

    # Comparison context
    ideal_for_role: bool = Field(
        default=False,
        description="Is this an ideal candidate?"
    )
    compensation_tier: str = Field(
        default="mid",
        description="One of: entry, mid, senior, executive"
    )


# ---------------------------------------------------------------------------
# System Prompt
# ---------------------------------------------------------------------------

_FIT_ANALYZER_SYSTEM = (
    "You are a VP of Engineering evaluating candidates for hiring. "
    "Provide comprehensive, data-driven fit analysis across multiple dimensions. "
    "Be specific and reference actual resume content. "
    "Consider both strengths and risks - no candidate is perfect."
)


# ---------------------------------------------------------------------------
# Main Analysis Function
# ---------------------------------------------------------------------------

async def analyze_candidate_fit(
    resume_text: str,
    jd_text: str,
    role: str | None = None,
) -> CandidateFit | None:
    """Comprehensive multi-dimensional fit analysis.

    Args:
        resume_text: The candidate's resume text
        jd_text: The job description text
        role: Optional role name for context

    Returns:
        CandidateFit with scores across all dimensions
    """
    if not ENABLE_MULTI_DIM_FIT or not ai_enabled():
        return None

    role_context = f" for the {role} position" if role else ""

    schema = {
        "technical_fit": "0-100 technical skills match",
        "experience_fit": "0-100 experience level & relevance",
        "domain_fit": "0-100 industry/domain knowledge",
        "seniority_fit": "0-100 seniority level alignment",
        "culture_indicators": "0-100 soft skills & culture signals",
        "growth_potential": "0-100 potential to grow in role",
        "overall_fit": "0-100 weighted overall score",
        "hiring_confidence": "high|medium|low",
        "strongest_dimensions": ["top 2 strengths"],
        "weakest_dimensions": ["top 2 weaknesses"],
        "unique_value": "what they uniquely bring",
        "risk_factors": ["potential risks"],
        "ideal_for_role": "true/false",
        "compensation_tier": "entry|mid|senior|executive"
    }

    prompt = f"""Evaluate this candidate{role_context}. Provide comprehensive fit analysis.

## Job Description
{jd_text[:JD_MAX_CHARS_LLM]}

## Candidate Resume
{resume_text}

## Analysis Dimensions (score each 0-100)

### 1. TECHNICAL FIT
- Do they have the required technical skills?
- Are their skills at the right DEPTH (not just mentioned, but actually USED)?
- Technology currency: are skills current or outdated for 2025?
- Score 80+: Exceeds technical requirements
- Score 60-79: Meets most technical requirements
- Score 40-59: Meets some, gaps exist
- Score <40: Significant technical gaps

### 2. EXPERIENCE FIT
- Years of experience vs what's required
- Relevance of past experience to this role
- Quality of past companies/projects
- Depth vs breadth of experience

### 3. DOMAIN FIT
- Industry knowledge match
- Problem domain familiarity (e.g., fintech, healthcare, e-commerce)
- Business context understanding
- Regulatory/compliance knowledge if applicable

### 4. SENIORITY FIT
- Is their career level right for this role?
- Overqualified: -10 to -30 points (flight risk, salary expectations)
- Underqualified: -20 to -50 points (not ready, needs ramp-up)
- Exact match: full points

### 5. CULTURE INDICATORS
- Communication quality (how well is resume written?)
- Collaboration signals (team achievements vs solo work)
- Leadership indicators (if role requires)
- Autonomy indicators (self-started projects, initiatives)
- Remote work indicators if relevant

### 6. GROWTH POTENTIAL
- Learning trajectory (new skills acquired over time)
- Adaptability signals (industry/tech pivots)
- Career ambition alignment (are they growing toward this role?)
- Education & certifications (continuous learning)

### OVERALL FIT CALCULATION
Weights: Technical 30% + Experience 25% + Domain 15% + Seniority 15% + Culture 10% + Growth 5%

### HIRING CONFIDENCE
- high: Clear yes, strong candidate
- medium: Worth interviewing, some questions to address
- low: Significant concerns, proceed with caution

### UNIQUE VALUE
What does this candidate uniquely bring that others might not?
(e.g., rare skill combination, specific domain expertise, leadership experience)

### RISK FACTORS
What could go wrong if we hire this candidate?
(e.g., flight risk, culture mismatch, skill gaps, salary expectations)

### COMPENSATION TIER
Based on their value, what compensation tier are they likely in?
- entry: New to field, learning phase
- mid: Established professional, independent contributor
- senior: Expert level, leads initiatives
- executive: Leadership/strategic level

Respond with valid JSON only."""

    data = await _chat_json(_FIT_ANALYZER_SYSTEM, prompt, schema)
    if data is None:
        return None

    return _parse_model(CandidateFit, data)


def get_fit_summary(fit: CandidateFit) -> dict[str, Any]:
    """Get a summary of candidate fit for UI display."""
    dimensions = {
        "Technical": fit.technical_fit,
        "Experience": fit.experience_fit,
        "Domain": fit.domain_fit,
        "Seniority": fit.seniority_fit,
        "Culture": fit.culture_indicators,
        "Growth": fit.growth_potential,
    }

    return {
        "dimensions": dimensions,
        "overall_fit": fit.overall_fit,
        "hiring_confidence": fit.hiring_confidence,
        "strongest": fit.strongest_dimensions,
        "weakest": fit.weakest_dimensions,
        "unique_value": fit.unique_value,
        "risk_factors": fit.risk_factors,
        "ideal": fit.ideal_for_role,
        "compensation_tier": fit.compensation_tier,
    }


def get_fit_modifier(fit: CandidateFit) -> float:
    """Calculate score adjustment based on multi-dimensional fit.

    Returns a modifier between -5 and +5 to adjust final score.
    """
    if fit.hiring_confidence == "high" and fit.overall_fit >= 80:
        return 5.0
    elif fit.hiring_confidence == "high":
        return 3.0
    elif fit.hiring_confidence == "medium" and fit.overall_fit >= 60:
        return 1.0
    elif fit.hiring_confidence == "low" and fit.overall_fit < 40:
        return -5.0
    elif fit.hiring_confidence == "low":
        return -2.0
    return 0.0
