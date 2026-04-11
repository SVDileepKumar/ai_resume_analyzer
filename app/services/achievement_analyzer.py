"""Extract and score achievements from resumes.

Detects quantified achievements, action verb strength, and impact metrics.
Scores achievements relative to JD requirements.
"""

from __future__ import annotations

import logging
import re
from typing import Any

from pydantic import BaseModel, Field, ValidationError, model_validator

from app.config import (
    ENABLE_ACHIEVEMENT_ANALYSIS,
    JD_MAX_CHARS_LLM,
    OLLAMA_NUM_PREDICT_JSON_HEAVY,
    RESUME_MAX_CHARS_LLM,
)
from app.services.llm_service import ai_enabled, _chat_json, _parse_model

logger = logging.getLogger(__name__)

# Bound LLM JSON size (local models truncate long outputs).
ACHIEVEMENT_LLM_MAX_ITEMS: int = 15
ACHIEVEMENT_STATEMENT_MAX_CHARS: int = 240


# ---------------------------------------------------------------------------
# Pydantic Models
# ---------------------------------------------------------------------------

class Achievement(BaseModel):
    """A single quantified achievement."""
    statement: str = Field(description="The achievement statement")
    action_verb: str = Field(description="The action verb used")
    action_strength: str = Field(
        default="moderate",
        description="One of: weak, moderate, strong"
    )
    metric_type: str | None = Field(
        default=None,
        description="One of: percentage, currency, time, count, none"
    )
    metric_value: str | None = Field(
        default=None,
        description="The actual metric if present"
    )
    impact_level: str = Field(
        default="individual",
        description="One of: individual, team, department, company, industry"
    )
    jd_relevance: float = Field(
        default=0.5,
        description="0-1 relevance to JD requirements"
    )


class AchievementAnalysis(BaseModel):
    """Analysis of all achievements in a resume."""
    achievements: list[Achievement] = Field(default_factory=list)
    quantified_count: int = Field(
        default=0,
        description="Number of achievements with metrics"
    )
    average_impact: float = Field(
        default=0.0,
        description="Average impact score 0-100"
    )
    top_achievements: list[str] = Field(
        default_factory=list,
        description="Top 3 most impactful achievements"
    )
    weak_statements: list[str] = Field(
        default_factory=list,
        description="Statements that need stronger verbs"
    )
    score: float = Field(
        default=0.0,
        description="Overall achievement score 0-100"
    )

    @model_validator(mode="before")
    @classmethod
    def _coerce_quantified_count(cls, data: Any) -> Any:
        if not isinstance(data, dict):
            return data
        qc = data.get("quantified_count")
        if isinstance(qc, str):
            m = re.search(r"-?\d+", qc.strip())
            data["quantified_count"] = int(m.group()) if m else 0
        elif qc is None:
            data["quantified_count"] = 0
        return data


# ---------------------------------------------------------------------------
# Action Verb Classifications
# ---------------------------------------------------------------------------

STRONG_VERBS = {
    "led", "architected", "designed", "implemented", "increased", "decreased",
    "reduced", "saved", "generated", "launched", "built", "scaled", "optimized",
    "automated", "mentored", "negotiated", "transformed", "pioneered", "spearheaded",
    "orchestrated", "streamlined", "revolutionized", "delivered", "achieved",
    "drove", "accelerated", "established", "founded", "invented", "created"
}

WEAK_VERBS = {
    "helped", "assisted", "participated", "worked on", "was responsible for",
    "involved in", "contributed to", "supported", "collaborated", "aided",
    "attended", "part of", "member of"
}


# ---------------------------------------------------------------------------
# System Prompt
# ---------------------------------------------------------------------------

_ACHIEVEMENT_ANALYZER_SYSTEM = (
    "You are an expert resume coach specializing in ATS optimization. "
    "Analyze resumes for quantified achievements and impact. "
    "Strong verbs: led, architected, designed, implemented, increased, reduced, saved. "
    "Weak verbs: helped, assisted, participated, worked on, was responsible for. "
    "Score achievements higher when they have metrics (%, $, time, count) and company-wide impact."
)


# ---------------------------------------------------------------------------
# Main Analysis Function
# ---------------------------------------------------------------------------

async def analyze_achievements(
    resume_text: str,
    jd_text: str,
) -> AchievementAnalysis | None:
    """Extract and analyze achievements from resume.

    Args:
        resume_text: The candidate's resume text
        jd_text: The job description text

    Returns:
        AchievementAnalysis with extracted achievements and scoring
    """
    if not ENABLE_ACHIEVEMENT_ANALYSIS or not ai_enabled():
        return None

    schema = {
        "achievements": [
            {
                "statement": (
                    f"achievement text, max {ACHIEVEMENT_STATEMENT_MAX_CHARS} chars (truncate in JSON if longer)"
                ),
                "action_verb": "the verb used",
                "action_strength": "weak|moderate|strong",
                "metric_type": "percentage|currency|time|count|none",
                "metric_value": "the metric value or null",
                "impact_level": "individual|team|department|company|industry",
                "jd_relevance": "0-1 relevance score",
            }
        ],
        "quantified_count": "number with metrics",
        "average_impact": "0-100 average impact",
        "top_achievements": ["top 3 achievements"],
        "weak_statements": ["statements needing improvement"],
        "score": "0-100 overall score"
    }

    prompt = f"""Analyze this resume for quantified achievements and impact.

## Job Description (for relevance scoring)
{jd_text[:JD_MAX_CHARS_LLM]}

## Resume
{resume_text}

## Instructions

1. **Extract up to {ACHIEVEMENT_LLM_MAX_ITEMS} achievement statements** — the most impactful and JD-relevant bullets (prioritize recent roles). Do not exceed this count.
   - Each **statement** must be at most **{ACHIEVEMENT_STATEMENT_MAX_CHARS} characters** (truncate wording in JSON if needed).

2. **For each achievement analyze:**
   - **Action verb**: What verb starts or drives the statement?
   - **Action strength**:
     - strong: led, architected, designed, implemented, increased, reduced, saved, launched, scaled
     - moderate: developed, managed, created, built, improved, wrote, organized
     - weak: helped, assisted, participated, worked on, was responsible for
   - **Metrics**: Does it have quantified results?
     - percentage: "increased by 40%", "reduced 25%"
     - currency: "saved $1M", "generated $500K"
     - time: "reduced time by 2 hours", "cut latency by 50ms"
     - count: "managed team of 5", "processed 1M records"
     - none: no quantification
   - **Impact level**:
     - individual: affected only the person's work
     - team: affected the team (5-15 people)
     - department: affected the department (15-100 people)
     - company: affected the company or major business line
     - industry: affected industry standards or practices
   - **JD relevance**: 0-1 score for how relevant this achievement is to the job

3. **Identify weak statements** that should be rewritten with stronger verbs

4. **Score calculation (0-100)**:
   - +5 points per achievement with strong verb
   - +10 points per achievement with quantified metric
   - +5/10/15/20/25 points based on impact level (individual→industry)
   - Multiply by JD relevance factor
   - Cap at 100

5. **Top 3 achievements**: Select the most impactful and JD-relevant achievements

Respond with valid JSON only."""

    data = await _chat_json(
        _ACHIEVEMENT_ANALYZER_SYSTEM,
        prompt,
        schema,
        num_predict=OLLAMA_NUM_PREDICT_JSON_HEAVY,
    )
    if data is None:
        return None

    return _parse_model(AchievementAnalysis, data)


def get_achievement_bonus(analysis: AchievementAnalysis) -> float:
    """Calculate score adjustment based on achievement quality.

    Returns a modifier between -10 and +10 to adjust final score.
    """
    if analysis.score >= 80:
        return 5.0  # Excellent achievements = +5 bonus
    elif analysis.score >= 60:
        return 2.0  # Good achievements = +2 bonus
    elif analysis.score >= 40:
        return 0.0  # Average
    elif analysis.score >= 20:
        return -2.0  # Weak achievements = -2 penalty
    else:
        return -5.0  # Poor achievements = -5 penalty


def get_achievement_summary(analysis: AchievementAnalysis) -> dict[str, Any]:
    """Get a summary of achievement analysis for UI display."""
    return {
        "total_achievements": len(analysis.achievements),
        "quantified_count": analysis.quantified_count,
        "quantified_ratio": (
            f"{analysis.quantified_count}/{len(analysis.achievements)}"
            if analysis.achievements else "0/0"
        ),
        "average_impact": round(analysis.average_impact, 1),
        "top_achievements": analysis.top_achievements[:3],
        "weak_statements_count": len(analysis.weak_statements),
        "score": analysis.score,
    }
