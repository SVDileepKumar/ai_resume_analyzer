"""Detect red flags and concerning patterns in resumes.

Automatically identifies potential issues like job hopping, unexplained gaps,
title inflation, and other warning signs for hiring decisions.
"""

from __future__ import annotations

import logging
from typing import Any

from pydantic import BaseModel, Field, ValidationError

from app.config import ENABLE_RED_FLAG_DETECTION, JD_MAX_CHARS_LLM, RESUME_MAX_CHARS_LLM
from app.services.llm_service import ai_enabled, _chat_json, _parse_model

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Pydantic Models
# ---------------------------------------------------------------------------

class CriticalFlag(BaseModel):
    """A critical red flag that may be a deal-breaker."""
    flag: str = Field(description="Name of the red flag")
    evidence: str = Field(description="Specific evidence from resume")
    severity: str = Field(
        default="high",
        description="One of: critical, high"
    )
    recommendation: str = Field(
        default="",
        description="What to do about this flag"
    )


class WarningFlag(BaseModel):
    """A warning flag that should be investigated."""
    flag: str = Field(description="Name of the warning")
    evidence: str = Field(description="Specific evidence from resume")
    questions_to_ask: list[str] = Field(
        default_factory=list,
        description="Interview questions to address this flag"
    )


class RedFlagAnalysis(BaseModel):
    """Comprehensive red flag detection."""

    # Critical red flags (likely deal-breakers)
    critical_flags: list[CriticalFlag] = Field(default_factory=list)

    # Warning flags (investigate further)
    warning_flags: list[WarningFlag] = Field(default_factory=list)

    # Positive signals (green flags)
    green_flags: list[str] = Field(default_factory=list)

    # Overall assessment
    risk_level: str = Field(
        default="medium",
        description="One of: low, medium, high, critical"
    )
    proceed_with_interview: bool = Field(
        default=True,
        description="Should we interview this candidate?"
    )
    interview_focus_areas: list[str] = Field(
        default_factory=list,
        description="Areas to probe in interview"
    )


# ---------------------------------------------------------------------------
# System Prompt
# ---------------------------------------------------------------------------

_RED_FLAG_DETECTOR_SYSTEM = (
    "You are a senior technical recruiter specializing in candidate risk assessment. "
    "Analyze resumes for red flags, warning signs, and positive indicators. "
    "Be thorough but fair - not every short tenure is job hopping, "
    "and not every gap is unexplained. Context matters. "
    "Focus on patterns, not isolated incidents."
)


# ---------------------------------------------------------------------------
# Main Analysis Function
# ---------------------------------------------------------------------------

async def detect_red_flags(
    resume_text: str,
    jd_text: str,
) -> RedFlagAnalysis | None:
    """Detect red flags and concerning patterns.

    Args:
        resume_text: The candidate's resume text
        jd_text: The job description text (for context)

    Returns:
        RedFlagAnalysis with flags and recommendations
    """
    if not ENABLE_RED_FLAG_DETECTION or not ai_enabled():
        return None

    schema = {
        "critical_flags": [
            {
                "flag": "flag name",
                "evidence": "specific evidence",
                "severity": "critical|high",
                "recommendation": "what to do"
            }
        ],
        "warning_flags": [
            {
                "flag": "flag name",
                "evidence": "specific evidence",
                "questions_to_ask": ["interview questions"]
            }
        ],
        "green_flags": ["positive indicators"],
        "risk_level": "low|medium|high|critical",
        "proceed_with_interview": "true/false",
        "interview_focus_areas": ["areas to probe"]
    }

    prompt = f"""Analyze this resume for red flags and concerning patterns.

## Job Description (context)
{jd_text[:JD_MAX_CHARS_LLM]}

## Resume
{resume_text}

## RED FLAGS TO DETECT

### CRITICAL FLAGS (Likely deal-breakers)
1. **Fabrication indicators**:
   - Inconsistent dates (overlapping jobs, impossible timelines)
   - Vague achievements without specifics
   - Claims that don't match job level
2. **Major unexplained gaps**:
   - >6 months gap with no explanation
   - Multiple gaps in succession
3. **Severe role mismatch**:
   - Applying for senior role with only internship experience
   - No relevant experience for core requirements
4. **Buzzword stuffing**:
   - Lists 50+ technologies without demonstrating depth
   - Every technology ever invented on the resume
5. **Copy-paste resume**:
   - Generic statements not specific to their experience
   - Obvious template language

### WARNING FLAGS (Investigate in interview)
1. **Job hopping pattern**:
   - <1 year at multiple companies without explanation
   - Pattern suggests commitment issues
   - Exception: startups that closed, layoffs, contract roles
2. **Title inflation**:
   - Senior/Lead titles at unknown companies with junior work
   - Title progression doesn't match responsibility growth
3. **Vague achievements**:
   - "Worked on", "Helped with", "Participated in"
   - No quantified results
4. **Missing sections**:
   - No education dates (hiding graduation year?)
   - No company names or dates
5. **Overqualification**:
   - PhD applying for junior role (flight risk?)
   - Senior exec applying for IC role
6. **Technology mismatch**:
   - Claims expertise in conflicting stacks
   - Lists competitors' products as experience
7. **No progression**:
   - Same title for 7+ years without promotion
   - Lateral moves only

### GREEN FLAGS (Positive signals)
1. Promotions within same company (loyalty + growth)
2. Quantified achievements with metrics
3. Open source contributions (visible work quality)
4. Consistent, logical career progression
5. Education matches career path
6. Rehired by former employer
7. Long tenures at strong companies
8. Side projects showing initiative

## OUTPUT INSTRUCTIONS
- Be specific with evidence - quote or reference actual resume content
- For critical flags, provide actionable recommendations
- For warning flags, provide interview questions to investigate
- Green flags help balance the assessment
- Risk level considers the overall pattern, not just count of flags
- proceed_with_interview should be false only for critical+high combo

Respond with valid JSON only."""

    data = await _chat_json(_RED_FLAG_DETECTOR_SYSTEM, prompt, schema)
    if data is None:
        return None

    return _parse_model(RedFlagAnalysis, data)


def get_red_flag_penalty(analysis: RedFlagAnalysis) -> float:
    """Calculate score penalty based on red flags detected.

    Returns a penalty between 0 and -15 to adjust final score.
    """
    if analysis.risk_level == "critical":
        return -15.0
    elif analysis.risk_level == "high":
        return -10.0
    elif analysis.risk_level == "medium" and len(analysis.critical_flags) > 0:
        return -5.0
    elif analysis.risk_level == "medium":
        return -2.0
    elif analysis.risk_level == "low":
        # Green flags can give a small bonus
        if len(analysis.green_flags) >= 3:
            return 2.0
        return 0.0
    return 0.0


def get_red_flag_summary(analysis: RedFlagAnalysis) -> dict[str, Any]:
    """Get a summary of red flag analysis for UI display."""
    return {
        "risk_level": analysis.risk_level,
        "critical_count": len(analysis.critical_flags),
        "warning_count": len(analysis.warning_flags),
        "green_count": len(analysis.green_flags),
        "proceed": analysis.proceed_with_interview,
        "critical_flags": [
            {"flag": f.flag, "evidence": f.evidence[:100], "severity": f.severity}
            for f in analysis.critical_flags
        ],
        "warning_flags": [
            {"flag": f.flag, "evidence": f.evidence[:100]}
            for f in analysis.warning_flags[:5]  # Limit to top 5
        ],
        "green_flags": analysis.green_flags[:5],  # Limit to top 5
        "focus_areas": analysis.interview_focus_areas[:3],
    }


def should_proceed_with_candidate(analysis: RedFlagAnalysis) -> tuple[bool, str]:
    """Determine if candidate should proceed based on red flags.

    Returns (proceed, reason)
    """
    if not analysis.proceed_with_interview:
        if analysis.critical_flags:
            top_flag = analysis.critical_flags[0]
            return False, f"Critical issue: {top_flag.flag}"
        return False, f"Risk level too high: {analysis.risk_level}"

    if analysis.risk_level == "high":
        return True, "Proceed with caution - address flagged concerns in interview"

    if analysis.risk_level == "medium":
        return True, "Proceed - some areas to explore in interview"

    return True, "Low risk candidate"
