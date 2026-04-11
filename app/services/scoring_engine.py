"""Central ATS scoring orchestrator with LLM integration.

Every dimension scores a resume AGAINST the JD.
When Ollama is running locally, AI provides:
  - Semantic similarity (embeddings via ``OLLAMA_EMBED_MODEL`` / OpenAI embed model)
  - Per-candidate deep insights (strengths, gaps, interview Qs)
  - Executive hiring summary
  - Resume improvement suggestions (individual /review only; not batch HR view)
  - Semantic skill matching (Phase 3)
  - Context-aware experience analysis (Phase 3)
  - Achievement impact analysis (Phase 3)
  - Multi-dimensional fit scoring (Phase 3)
  - Career trajectory analysis (Phase 3)
  - Red flag detection (Phase 3)
  - Comparative candidate ranking (Phase 3)
"""

from __future__ import annotations

import asyncio
import hashlib
import logging
import re
from typing import Any

from app.services.skill_matcher import match_skills, get_role_weights
from app.services.similarity import calculate_similarity
from app.services.section_scorer import (
    score_experience,
    score_education,
    score_projects,
    score_certifications,
    normalize_scores,
    _extract_equivalent_years_from_jd,
    extract_education_details,
    extract_certifications_details,
)
from app.services.llm_service import (
    ai_enabled,
    analyze_candidate,
    analyze_resume_unified,
    analyze_resume_deep,
    extract_skills_llm,
    generate_batch_summary,
    generate_resume_suggestions,
    compare_candidates,
    score_resume_with_llm,
    LLMSkillExtraction,
    SkillMatch,
    TransferableSkill,
    UnifiedResumeScore,
)

# Phase 3 imports - LLM-powered analysis
from app.services.semantic_skill_matcher import (
    extract_jd_skills_llm,
    JDRequiredSkill,
)

from app.config import (
    AI_PRIMARY_SCORING,
    ENABLE_SEMANTIC_SKILLS,
    ENRICHER_SCORE_THRESHOLD,
    FEATURE_IMPROVEMENT_SUGGESTIONS,
    LLM_MAX_CONCURRENT_RESUMES,
    LLM_BACKEND,
    MAX_INFERENCE_BOOST,
    SCORE_DIVERGENCE_THRESHOLD,
)
from app.services.experience_analyzer import get_experience_metadata
from app.services.achievement_analyzer import get_achievement_bonus
from app.services.fit_analyzer import get_fit_modifier
from app.services.red_flag_detector import get_red_flag_penalty

logger = logging.getLogger(__name__)


async def _gather_with_concurrency_cap(
    tasks: list[Any],
    max_concurrent: int,
    *,
    return_exceptions: bool = False,
) -> list[Any]:
    """Run awaitables with an optional semaphore (0 = unlimited)."""
    if max_concurrent <= 0:
        return await asyncio.gather(*tasks, return_exceptions=return_exceptions)
    sem = asyncio.Semaphore(max_concurrent)

    async def _run(awaitable: Any) -> Any:
        async with sem:
            return await awaitable

    return await asyncio.gather(*(_run(t) for t in tasks), return_exceptions=return_exceptions)


def _hiring_decision(score: float) -> tuple[str, str]:
    """Return (label, color) for a hiring recommendation. Thresholds from config."""
    from app.config import HIRING_BAND_STRONG, HIRING_BAND_POTENTIAL, HIRING_BAND_NEEDS_REVIEW
    if score >= HIRING_BAND_STRONG:
        return "Strong Match", "green"
    if score >= HIRING_BAND_POTENTIAL:
        return "Potential Match", "blue"
    if score >= HIRING_BAND_NEEDS_REVIEW:
        return "Needs Review", "amber"
    return "Weak Match", "red"


def _error_candidate_result(filename: str, role: str | None, error: str) -> dict[str, Any]:
    """Shape-preserving error result used by batch views/templates.

    The results template assumes every candidate has weights, section scores,
    and a few explanatory fields. Returning a full-shaped error card keeps the
    page renderable even when AI scoring fails for one or more candidates.
    """
    weights = get_role_weights(role)
    return {
        "candidate": filename,
        "final_score": 0.0,
        "decision": "Error",
        "decision_color": "red",
        "error": error,
        "matched_skills": [],
        "missing_skills": [],
        "extra_skills": [],
        "skill_match_ratio": "0/0",
        "section_scores": {
            "skills": 0.0,
            "similarity": 0.0,
            "experience": 0.0,
            "education": 0.0,
            "projects": 0.0,
            "certifications": 0.0,
        },
        "section_scores_normalized": {
            "skills": 0.0,
            "similarity": 0.0,
            "experience": 0.0,
            "education": 0.0,
            "projects": 0.0,
            "certifications": 0.0,
        },
        "section_explanations": {},
        "top_factors": [],
        "weights": weights,
        "match_type": "unknown",
        "experience_metadata": {},
        "semantic_skills": None,
        "experience_analysis": None,
        "achievement_analysis": None,
        "career_trajectory": None,
        "candidate_fit": None,
        "red_flags": None,
        "llm_holistic_score": None,
        "ai_insight": None,
        "scoring_method": "error",
    }


def _fallback_extraction(skill_result: dict[str, Any]) -> LLMSkillExtraction:
    """Wrap regex match_skills() output in the LLMSkillExtraction schema.

    Called when extract_skills_llm() fails or AI is unavailable, ensuring all
    downstream code has a single consuming interface regardless of which path ran.
    """
    matched_items = [
        SkillMatch(skill=str(s), evidence="", confidence=1.0, match_type="explicit")
        for s in (skill_result.get("matched") or [])
    ]
    missing_items = [
        SkillMatch(skill=str(s), evidence="", confidence=0.9, match_type="explicit")
        for s in (skill_result.get("missing") or [])
    ]
    score = float(skill_result.get("score", 50.0))
    # jd_required_skills is approximated from matched + missing skill names
    all_jd_skills = [m.skill for m in matched_items] + [m.skill for m in missing_items]
    return LLMSkillExtraction(
        jd_required_skills=all_jd_skills,
        jd_preferred_skills=[],
        matched=matched_items,
        missing=missing_items,
        transferable=[],
        skills_score=score,
    )


async def _apply_enhanced_unified(
    base_result: dict[str, Any],
    unified: UnifiedResumeScore,
    resume_text: str,
    jd_text: str,
    role: str | None,
    *,
    extraction_result: LLMSkillExtraction | None = None,
) -> dict[str, Any]:
    """AI-primary path: unified dimension scores + enrichers only (no semantic/holistic blend)."""
    # Extraction skills_score is authoritative — it was produced by a dedicated LLM call
    # that saw both the JD and resume with few-shot transferable inference instructions.
    # When extraction succeeded (non-None), use its score directly. When only the
    # regex fallback ran, base_result["section_scores"]["skills"] already holds that
    # fallback score (set in score_resume_enhanced before this call).
    if extraction_result is not None:
        blended_skills = float(extraction_result.skills_score)
    else:
        # Graceful degradation: no extraction result available, blend unified + regex
        u_skills = float(unified.skills_score)
        regex_skills = float(base_result["section_scores"]["skills"])
        blended_skills = 0.70 * u_skills + 0.30 * regex_skills

    # Post-blend inference credit: apply evidence-backed inferred skill boost AFTER blending
    evidence_valid = [s for s in unified.inferred_skills if s.evidence.strip()]
    # Use extraction jd_required_skills for more accurate JD skill set (R4 of plan)
    if extraction_result is not None and extraction_result.jd_required_skills:
        jd_required_skill_set = {sk.lower() for sk in extraction_result.jd_required_skills}
    else:
        jd_required_skill_set = {
            sk.lower() for sk in (
                list(base_result.get("matched_skills") or [])
                + list(base_result.get("missing_skills") or [])
            )
        }
    # De-duplicate: do not count inference credit for skills already in extraction.matched
    # (they are already reflected in extraction_result.skills_score)
    if extraction_result is not None:
        extraction_matched_lower = {m.skill.lower() for m in extraction_result.matched}
        evidence_valid = [
            s for s in evidence_valid
            if s.skill.lower() not in extraction_matched_lower
        ]
    jd_skill_count = max(len(jd_required_skill_set), 1)
    contributions = [
        s.credit_weight
        for s in evidence_valid
        if s.skill.lower() in jd_required_skill_set
    ]
    inference_scaled = sum(contributions) / jd_skill_count * 100
    inference_credit = min(inference_scaled, MAX_INFERENCE_BOOST)
    blended_skills = min(blended_skills + inference_credit, 100.0)
    if inference_credit > 0:
        logger.debug(
            "Inference credit: +%.2f pts from %d skill(s) %s → blended_skills=%.2f",
            inference_credit,
            len(contributions),
            [s.skill for s in evidence_valid if s.skill.lower() in jd_required_skill_set],
            blended_skills,
        )

    raw_scores = {
        "skills": blended_skills,
        "similarity": float(unified.similarity_score),
        "experience": float(unified.experience_score),
        "education": float(unified.education_score),
        "projects": float(unified.projects_score),
        "certifications": float(unified.certifications_score),
    }

    equiv_years = _extract_equivalent_years_from_jd(jd_text)
    exp_meta = base_result.get("experience_metadata") or {}
    if equiv_years is not None and exp_meta.get("candidate_years", 0) >= equiv_years:
        raw_scores["education"] = max(raw_scores["education"], 70.0)

    normalized = normalize_scores(raw_scores, ai_sourced=True)
    weights = base_result["weights"]
    total_weight = sum(weights.values()) or 1.0
    weighted_sum = sum(
        normalized.get(dim, 0) * weights.get(dim, 0)
        for dim in weights
    )
    # Cap the pre-modifier weighted score at 97. LLMs tend to award very high
    # per-dimension scores for strong resumes; combined with positive modifiers
    # (achievements, fit, trajectory) this trivially produces 100 even when gaps
    # and improvements exist. No real-world resume is dimensionally perfect.
    final_score = min(round(weighted_sum / total_weight, 1), 97.0)

    base_result["section_scores"] = raw_scores
    base_result["section_scores_normalized"] = normalized
    base_result["final_score"] = final_score

    # Skill lists: extraction is authoritative when available (it drove the score).
    # Fall back to unified's lists only when extraction was not run.
    if extraction_result is not None:
        # base_result already has extraction matched/missing set in score_resume_enhanced;
        # preserve them so the displayed skills match what drove the score.
        matched_for_ratio = base_result.get("matched_skills") or []
        missing_for_ratio = base_result.get("missing_skills") or []
    else:
        base_result["matched_skills"] = list(unified.matched_skills)
        base_result["missing_skills"] = list(unified.missing_skills)
        matched_for_ratio = unified.matched_skills
        missing_for_ratio = unified.missing_skills
    total_req = max(len(matched_for_ratio) + len(missing_for_ratio), 1)
    base_result["skill_match_ratio"] = f"{len(matched_for_ratio)}/{total_req}"

    explanations = dict(base_result.get("section_explanations") or {})
    ux = unified.section_explanations or {}
    for dim_key in ("skills", "similarity", "experience", "education", "projects", "certifications"):
        if ux.get(dim_key):
            explanations[dim_key] = ux[dim_key]
    base_result["section_explanations"] = explanations

    sorted_dims = sorted(normalized, key=lambda d: normalized[d])
    top_factors = []
    if sorted_dims:
        top_factors.append(f"Strongest: {sorted_dims[-1]} ({normalized[sorted_dims[-1]]:.0f})")
        top_factors.append(f"Lowest: {sorted_dims[0]} ({normalized[sorted_dims[0]]:.0f})")
    if len(sorted_dims) > 1 and normalized[sorted_dims[1]] < 70:
        top_factors.append(f"Also low: {sorted_dims[1]} ({normalized[sorted_dims[1]]:.0f})")
    base_result["top_factors"] = top_factors

    # Skip the deep enricher call for weak candidates — saves one full LLM round-trip
    # (the second of two per resume) for roughly half the batch in typical use.
    # ENRICHER_SCORE_THRESHOLD=0 disables this early exit.
    if ENRICHER_SCORE_THRESHOLD > 0 and base_result["final_score"] < ENRICHER_SCORE_THRESHOLD:
        logger.info(
            "Skipping deep enricher call for score %.1f < threshold %.1f",
            base_result["final_score"],
            ENRICHER_SCORE_THRESHOLD,
        )
        deep = None
        base_result["deep_analysis_skipped"] = True
    else:
        # Single consolidated LLM call replacing the previous 5-enricher asyncio.gather.
        # On local Ollama this reduces model queue occupancy from 5 concurrent slots to 1,
        # cutting per-resume wall-clock time by ~4×.
        inferred_names = [s.skill for s in evidence_valid] if evidence_valid else None
        deep = await analyze_resume_deep(
            resume_text, jd_text, role,
            inferred_skill_names=inferred_names,
            education_details=base_result.get("education_details") or [],
            cert_details=base_result.get("cert_details") or [],
            experience_metadata=base_result.get("experience_metadata") or {},
        )
        base_result["deep_analysis_skipped"] = False

    if deep is not None:
        experience_analysis = deep.experience
        achievement_analysis = deep.achievements
        career_trajectory = deep.trajectory
        candidate_fit = deep.fit
        red_flags = deep.red_flags
    else:
        logger.warning("Deep analysis returned None — enricher data unavailable")
        experience_analysis = None
        achievement_analysis = None
        career_trajectory = None
        candidate_fit = None
        red_flags = None

    final_score = base_result["final_score"]
    modifier_total = 0.0
    impact_parts: list[str] = []

    if achievement_analysis:
        ach_bonus = get_achievement_bonus(achievement_analysis)
        modifier_total += ach_bonus
        if ach_bonus != 0:
            impact_parts.append(f"achievements {ach_bonus:+.0f}")

    if candidate_fit:
        fit_mod = get_fit_modifier(candidate_fit)
        modifier_total += fit_mod
        if fit_mod != 0:
            impact_parts.append(f"fit {fit_mod:+.0f}")

    if red_flags:
        red_flag_penalty = get_red_flag_penalty(red_flags)
        modifier_total += red_flag_penalty
        if red_flag_penalty != 0:
            risk = getattr(red_flags, "risk_level", "medium")
            impact_parts.append(f"red flags {red_flag_penalty:+.0f} (risk: {risk})")
            focus = getattr(red_flags, "interview_focus_areas", [])[:3]
            if focus:
                impact_parts.append(f"focus: {', '.join(focus)}")

    # Experience-trajectory bonus: ascending careers in relevant domain get a boost.
    # Validate career_progression against known enum values to avoid silent skip
    # when LLM returns an unexpected string (e.g. "upward" instead of "ascending").
    _VALID_PROGRESSIONS = {"ascending", "stable", "mixed", "descending", "lateral"}
    if experience_analysis:
        prog_raw = getattr(experience_analysis, "career_progression", "mixed")
        prog = prog_raw if prog_raw in _VALID_PROGRESSIONS else "mixed"
        if prog_raw not in _VALID_PROGRESSIONS:
            logger.warning(
                "Unexpected career_progression value %r from LLM — defaulting to 'mixed'",
                prog_raw,
            )
        domain = getattr(experience_analysis, "domain_match", "different")
        if prog == "ascending" and domain in ("exact", "related"):
            trajectory_bonus = 3.0
            modifier_total += trajectory_bonus
            impact_parts.append(f"career trajectory {trajectory_bonus:+.0f}")
        elif prog == "descending":
            trajectory_penalty = -2.0
            modifier_total += trajectory_penalty
            impact_parts.append(f"career trajectory {trajectory_penalty:+.0f}")

    # Clamp to [-15, +10] as documented in config.py scoring reference comment
    modifier_total = max(-15.0, min(10.0, modifier_total))
    final_score += modifier_total

    if impact_parts:
        base_result["score_impact_summary"] = (
            f"Modifiers ({modifier_total:+.0f}): {'; '.join(impact_parts)}."
        )

    # Hard ceiling at 99: a score of 100 implies a perfect resume with nothing
    # to improve, but the system simultaneously surfaces gaps, missing skills,
    # and suggestions. Cap at 99 so the displayed score is always honest.
    final_score = max(0, min(99, round(final_score, 1)))

    # Always update experience_metadata when LLM enricher ran; if enricher was skipped,
    # keep the regex-derived metadata that score_resume() populated (Bug 4 fix).
    if experience_analysis:
        llm_meta = get_experience_metadata(experience_analysis)
        base_result["experience_metadata"] = llm_meta
        base_result["match_type"] = llm_meta.get("match_type", "exact")

    base_result["final_score"] = final_score
    base_result["semantic_skills"] = {
        "matched": list(unified.matched_skills),
        "missing": list(unified.missing_skills),
        "partial": [
            {
                "required": t,
                "found": "",
                "similarity": 0.0,
                "note": "transferable/alternative",
            }
            for t in unified.transferable_skills
        ],
        "extra": base_result.get("extra_skills", []),
        "proficiency_levels": {},
        "score": raw_scores["skills"],
        "method": "ai_primary_unified",
    }
    base_result["experience_analysis"] = (
        experience_analysis.model_dump()
        if experience_analysis and hasattr(experience_analysis, "model_dump")
        else None
    )
    base_result["achievement_analysis"] = (
        achievement_analysis.model_dump()
        if achievement_analysis and hasattr(achievement_analysis, "model_dump")
        else None
    )
    base_result["career_trajectory"] = (
        career_trajectory.model_dump()
        if career_trajectory and hasattr(career_trajectory, "model_dump")
        else None
    )
    base_result["candidate_fit"] = (
        candidate_fit.model_dump()
        if candidate_fit and hasattr(candidate_fit, "model_dump")
        else None
    )
    base_result["red_flags"] = (
        red_flags.model_dump()
        if red_flags and hasattr(red_flags, "model_dump")
        else None
    )
    base_result["llm_holistic_score"] = None
    base_result["unified_resume_score"] = unified.model_dump()
    base_result["scoring_method"] = "ai_primary_unified"

    decision, color = _hiring_decision(final_score)
    base_result["decision"] = decision
    base_result["decision_color"] = color

    return base_result



def score_resume(
    resume_text: str,
    jd_text: str,
    role: str | None = None,
    filename: str = "unknown",
) -> dict[str, Any]:
    """Score a single resume against a JD.

    Role is optional and used for advisory context only.
    Scoring uses unified weights regardless of role selection.
    When AI_PRIMARY_SCORING is enabled, similarity is set to 0 as a placeholder
    because the unified LLM call immediately overrides all dimension scores.
    """
    skill_result = match_skills(resume_text, jd_text, role)
    # Skip the blocking embedding call when the AI unified scorer will override it anyway
    if AI_PRIMARY_SCORING and ai_enabled():
        similarity, sim_method = 0.0, "pending_ai"
    else:
        similarity, sim_method = calculate_similarity(resume_text, jd_text)
    experience, exp_metadata = score_experience(resume_text, jd_text, return_metadata=True)
    education = score_education(resume_text, jd_text)
    education_details = extract_education_details(resume_text)
    projects = score_projects(resume_text, jd_text)
    certifications = score_certifications(resume_text, jd_text)
    cert_details = extract_certifications_details(resume_text)

    # Raw section scores. "skills" dimension uses regex match_skills.
    # AI-primary path overrides all dimensions via the unified LLM scorer.
    raw_scores = {
        "skills": skill_result["score"],
        "similarity": similarity,
        "experience": experience,
        "education": education,
        "projects": projects,
        "certifications": certifications,
    }

    # Equivalent experience: "degree or N years equivalent" — floor education when experience >= N
    equiv_years = _extract_equivalent_years_from_jd(jd_text)
    if equiv_years is not None and exp_metadata.get("candidate_years", 0) >= equiv_years:
        raw_scores["education"] = max(raw_scores["education"], 70.0)

    # Apply normalization to prevent dimension bias
    normalized = normalize_scores(raw_scores)

    weights = get_role_weights(role)
    total_weight = sum(weights.values()) or 1.0

    # Compute final score using normalized values (all dimensions in weights)
    weighted_sum = sum(
        normalized.get(dim, 0) * weights.get(dim, 0)
        for dim in weights
    )
    final_score = round(weighted_sum / total_weight, 1)

    decision, color = _hiring_decision(final_score)

    # Explainability: per-dimension explanation and top factors
    section_explanations = {
        "skills": f"{skill_result['matched_count']}/{skill_result['total_required']} JD requirements matched; missing: {', '.join(skill_result['missing'][:5])}{'...' if len(skill_result['missing']) > 5 else ''}",
        "experience": f"{exp_metadata.get('candidate_years', 0)} years vs {exp_metadata.get('required_years') or 'not specified'} required; {exp_metadata.get('match_type', 'exact')} seniority match",
        "education": (
            "; ".join(
                f"{d['degree']}{' in ' + d['field'] if d['field'] else ''}"
                f"{' — ' + d['institution'] if d['institution'] else ''}"
                f"{' (' + d['score_label'] + ': ' + d['raw_score'] + ')' if d['raw_score'] else ''}"
                for d in education_details
            ) or f"Score {raw_scores['education']:.0f}: degree/field match to JD requirements"
        ),
        "projects": f"Raw score {raw_scores['projects']:.0f} (JD keyword overlap and project section quality)",
        "similarity": f"Text similarity {raw_scores['similarity']:.0f} ({sim_method})",
        "certifications": f"Certification match: {raw_scores['certifications']:.0f}/100",
    }
    sorted_dims = sorted(normalized, key=lambda d: normalized[d])
    top_factors = []
    if sorted_dims:
        top_factors.append(f"Strongest: {sorted_dims[-1]} ({normalized[sorted_dims[-1]]:.0f})")
        top_factors.append(f"Lowest: {sorted_dims[0]} ({normalized[sorted_dims[0]]:.0f})")
    if len(sorted_dims) > 1 and normalized[sorted_dims[1]] < 70:
        top_factors.append(f"Also low: {sorted_dims[1]} ({normalized[sorted_dims[1]]:.0f})")
    score_impact_summary = None  # Set in enhanced path when red flags apply

    return {
        "candidate": filename,
        "final_score": final_score,
        "decision": decision,
        "decision_color": color,
        "matched_skills": skill_result["matched"],
        "missing_skills": skill_result["missing"],
        "extra_skills": skill_result["extra"],
        "section_scores": raw_scores,  # Keep raw scores for display
        "section_scores_normalized": normalized,  # Include normalized for transparency
        "section_explanations": section_explanations,
        "top_factors": top_factors,
        "score_impact_summary": score_impact_summary,
        "similarity_method": sim_method,
        "weights": weights,
        "skill_match_ratio": f"{skill_result['matched_count']}/{skill_result['total_required']}",
        # Experience metadata for UI indicators
        "match_type": exp_metadata.get("match_type", "exact"),
        "experience_metadata": exp_metadata,
        # Education details: parsed degree entries with institution and GPA
        "education_details": education_details,
        # Cert details: list of cert names parsed from full (untruncated) resume
        "cert_details": cert_details,
        # Placeholder for AI insights (populated async later)
        "ai_insight": None,
        # Populated only in analyze_single_resume (Individual Resume Review)
        "improvement_suggestions": None,
        # Phase 3: Placeholders for LLM-powered analysis
        "semantic_skills": None,
        "experience_analysis": None,
        "achievement_analysis": None,
        "career_trajectory": None,
        "candidate_fit": None,
        "red_flags": None,
        "llm_holistic_score": None,
        # Store raw text for AI analysis
        "_resume_text": resume_text,
        # Stored for _fallback_extraction() when LLM extraction fails
        "_skill_result": skill_result,
    }


async def score_resume_enhanced(
    resume_text: str,
    jd_text: str,
    role: str | None = None,
    filename: str = "unknown",
    jd_skills_preextracted: list[JDRequiredSkill] | None = None,
) -> dict[str, Any]:
    """AI-only enhanced scoring. Requires AI to be available."""
    if not ai_enabled():
        raise RuntimeError("AI scoring unavailable: Ollama not running or no API key configured")

    # Offload CPU-bound regex/section scoring to a thread so the event loop
    # remains free to dispatch other coroutines while this resume is prepared.
    base_result = await asyncio.to_thread(score_resume, resume_text, jd_text, role, filename)

    # --- LLM skill extraction (Call 1 of two-call architecture) ---
    # Replaces the regex grounding block passed to the unified scorer.
    extraction_result: LLMSkillExtraction | None = None
    try:
        extraction_result = await extract_skills_llm(resume_text, jd_text)
    except Exception as exc:
        logger.error("extract_skills_llm raised unexpectedly: %s — using regex fallback", exc)
    if extraction_result is None:
        logger.warning(
            "LLM skill extraction failed for %s — using regex fallback",
            filename,
        )
        skill_result = base_result.get("_skill_result") or {
            "matched": list(base_result.get("matched_skills", [])),
            "missing": list(base_result.get("missing_skills", [])),
            "score": float(base_result.get("section_scores", {}).get("skills", 50.0)),
        }
        extraction_result = _fallback_extraction(skill_result)
        base_result["scoring_method"] = "regex_fallback"
    else:
        base_result["scoring_method"] = "llm_extraction"

    # Populate matched/missing in base_result from extraction (overrides regex output)
    base_result["matched_skills"] = [m.skill for m in extraction_result.matched]
    base_result["missing_skills"] = [m.skill for m in extraction_result.missing]
    total_req = max(
        len(extraction_result.matched) + len(extraction_result.missing), 1
    )
    base_result["skill_match_ratio"] = f"{len(extraction_result.matched)}/{total_req}"
    base_result["section_scores"]["skills"] = extraction_result.skills_score

    unified_result: UnifiedResumeScore | None = None
    if AI_PRIMARY_SCORING:
        try:
            unified_result = await analyze_resume_unified(
                resume_text,
                jd_text,
                extraction_result=extraction_result,
                education_details=base_result.get("education_details") or [],
                cert_details=base_result.get("cert_details") or [],
                experience_metadata=base_result.get("experience_metadata") or {},
            )
        except Exception as exc:
            logger.exception("Unified AI scoring failed")
            raise RuntimeError(f"Unified AI scoring failed: {exc}") from exc

    if unified_result is not None:
        try:
            return await _apply_enhanced_unified(
                base_result, unified_result, resume_text, jd_text, role,
                extraction_result=extraction_result,
            )
        except Exception as exc:
            logger.exception("Unified enhanced pipeline failed")
            raise RuntimeError(f"Unified enhanced pipeline failed: {exc}") from exc

    logger.warning(
        "Unified AI scoring returned None for %s — falling back to regex-only scores",
        filename,
    )
    base_result["scoring_method"] = "regex_fallback"
    base_result.pop("_resume_text", None)
    base_result.pop("_skill_result", None)
    return base_result


async def analyze_batch(
    resumes: list[dict[str, str]],
    jd_text: str,
    role: str | None = None,
    use_enhanced_scoring: bool = True,
) -> dict[str, Any]:
    """Analyze a batch of resumes with optional AI insights and suggestions.

    Role is optional and used for advisory context only.
    Scoring uses unified weights regardless of role selection.

    Args:
        resumes: List of dicts with 'text' and 'filename' keys
        jd_text: The job description text
        role: Optional role name for context
        use_enhanced_scoring: If True and AI enabled, use LLM-enhanced scoring
    """
    # Duplicate detection: hash normalised text and track which files share content.
    # Duplicates reuse the first copy's LLM result — saves 2-3 Ollama calls per dupe.
    def _normalize_for_hash(t: str) -> str:
        return re.sub(r"\s+", " ", (t or "").lower().strip())

    hash_to_filenames: dict[str, list[str]] = {}
    first_resume_for_hash: dict[str, str] = {}  # hash → filename of canonical copy
    for r in resumes:
        h = hashlib.sha256(_normalize_for_hash(r["text"]).encode("utf-8")).hexdigest()
        hash_to_filenames.setdefault(h, []).append(r["filename"])
        first_resume_for_hash.setdefault(h, r["filename"])

    duplicate_of: dict[str, list[str]] = {}
    for filenames in hash_to_filenames.values():
        if len(filenames) > 1:
            for f in filenames:
                duplicate_of[f] = [x for x in filenames if x != f]

    # Map filename → hash for quick lookup inside _score_then_insight
    filename_to_hash: dict[str, str] = {
        r["filename"]: hashlib.sha256(_normalize_for_hash(r["text"]).encode("utf-8")).hexdigest()
        for r in resumes
    }

    # Phase 1: AI-only scoring — requires Ollama/OpenAI to be available
    if not ai_enabled():
        raise RuntimeError(
            "AI scoring is required but unavailable. "
            "Start Ollama or configure OPENAI_API_KEY before uploading resumes."
        )

    jd_skills_preextracted: list[JDRequiredSkill] | None = None
    if resumes and ENABLE_SEMANTIC_SKILLS and not AI_PRIMARY_SCORING:
        try:
            jd_skills_preextracted = await extract_jd_skills_llm(jd_text)
        except Exception:
            logger.exception("Batch JD skill extraction failed; continuing without pre-extraction")
            jd_skills_preextracted = None

    # Cache of completed results keyed by content hash for duplicate reuse
    _result_cache: dict[str, dict[str, Any]] = {}

    async def _score_then_insight(r: dict[str, Any]) -> dict[str, Any]:
        """Score one resume then immediately get its AI insight — no waiting for the batch."""
        h = filename_to_hash[r["filename"]]
        canonical = first_resume_for_hash[h]

        # If this is a duplicate of an already-completed (or in-progress) resume,
        # wait for the canonical result and return a shallow copy with the new filename.
        if canonical != r["filename"]:
            max_wait_sec = 300
            waited = 0.0
            while h not in _result_cache and waited < max_wait_sec:
                await asyncio.sleep(0.1)
                waited += 0.1
            if h not in _result_cache:
                logger.warning(
                    "Timed out waiting for canonical result of %s; scoring %s independently",
                    canonical, r["filename"],
                )
            else:
                orig = _result_cache[h]
                dup_result = dict(orig)
                dup_result["candidate"] = r["filename"]
                logger.debug("Reusing result from %s for duplicate %s", canonical, r["filename"])
                return dup_result

        try:
            result = await score_resume_enhanced(
                r["text"],
                jd_text,
                role,
                r["filename"],
                jd_skills_preextracted=jd_skills_preextracted,
            )
        except Exception as exc:
            logger.error("AI scoring failed for %s: %s", r["filename"], exc)
            err = _error_candidate_result(r["filename"], role, str(exc))
            _result_cache[h] = err
            return err

        # Insight call removed from batch mode: per-resume analyze_candidate() adds
        # one full LLM round-trip per resume on top of the two already done
        # (unified score + deep enricher). On 8GB RAM with serial inference this adds
        # ~30-60s per resume, making 50-resume batches infeasible.
        # The batch executive summary (generate_batch_summary) covers the same need
        # at the cohort level. Individual insight is available in /review (single mode).

        _result_cache[h] = result
        return result

    pipeline_tasks = [_score_then_insight(r) for r in resumes]
    results_raw = await _gather_with_concurrency_cap(
        pipeline_tasks,
        LLM_MAX_CONCURRENT_RESUMES,
        return_exceptions=True,
    )

    results = []
    for i, result in enumerate(results_raw):
        if isinstance(result, Exception):
            logger.error("Pipeline failed for %s: %s", resumes[i]["filename"], result)
            results.append(_error_candidate_result(resumes[i]["filename"], role, str(result)))
        else:
            results.append(result)

    ranked = sorted(results, key=lambda x: x["final_score"], reverse=True)

    # Phase 2: Executive summary + comparative ranking (per-resume insights already done above)
    ai_summary = None
    comparative_analysis = None

    if ai_enabled():
        try:
            if len(ranked) >= 2:
                sum_res, comp_res = await asyncio.gather(
                    generate_batch_summary(jd_text, ranked),
                    compare_candidates(ranked, jd_text),
                    return_exceptions=True,
                )
            else:
                sum_res = await generate_batch_summary(jd_text, ranked)
                comp_res = None

            if isinstance(sum_res, Exception):
                logger.warning("Batch summary failed: %s", sum_res)
                sum_res = None
            if isinstance(comp_res, Exception):
                logger.warning("Comparative analysis failed: %s", comp_res)
                comp_res = None

            if sum_res is not None and hasattr(sum_res, "model_dump"):
                ai_summary = sum_res.model_dump()
            if comp_res is not None and hasattr(comp_res, "model_dump"):
                comparative_analysis = comp_res.model_dump()

        except Exception:
            logger.exception("AI batch summary/comparison failed (non-fatal)")

    # Attach duplicate flag and clean up internal fields
    for c in ranked:
        dup = duplicate_of.get(c["candidate"], [])
        if dup:
            c["duplicate_of"] = dup  # "Possible duplicate of [filename]"
        c.pop("_resume_text", None)
        c.pop("_skill_result", None)

    return {
        "job_role": role or "General",
        "total_resumes": len(resumes),
        "all_candidates": ranked,
        "ai_enabled": ai_enabled(),
        "ai_summary": ai_summary,
        "comparative_analysis": comparative_analysis,
    }


async def analyze_single_resume(
    resume_text: str,
    jd_text: str | None = None,
    role: str | None = None,
    filename: str = "resume",
) -> dict[str, Any]:
    """Analyze a single resume for the Individual Resume Analyzer.

    Unlike analyze_batch (recruiter view), this focuses on giving
    the job seeker personal feedback. JD is optional — without it,
    we provide general resume quality feedback.
    """
    # Use a generic JD placeholder if none provided
    has_jd = bool(jd_text and jd_text.strip())
    if not has_jd:
        # General quality assessment — use a broad "any role" JD
        jd_text = (
            "Looking for a skilled professional with strong technical abilities, "
            "relevant work experience, solid educational background, and "
            "demonstrated project experience. Must have clear communication skills "
            "and a track record of delivering results."
        )

    # AI-only scoring — score_resume_enhanced raises if AI unavailable
    result = await score_resume_enhanced(resume_text, jd_text, role, filename)

    # Run AI insights
    result["suggestions_unavailable"] = False
    result["suggestions_unavailable_reason"] = None
    try:
        insight_task = analyze_candidate(resume_text, jd_text, result)
        suggestion_task = generate_resume_suggestions(resume_text, jd_text, result)

        insight, suggestions = await asyncio.gather(
            insight_task, suggestion_task, return_exceptions=True,
        )

        if not isinstance(insight, Exception) and insight:
            result["ai_insight"] = insight.model_dump()
        elif isinstance(insight, Exception):
            logger.warning("AI insight failed for single resume: %s", insight)

        if not isinstance(suggestions, Exception) and suggestions:
            result["improvement_suggestions"] = suggestions.model_dump()
        elif FEATURE_IMPROVEMENT_SUGGESTIONS:
            result["suggestions_unavailable"] = True
            if isinstance(suggestions, Exception):
                result["suggestions_unavailable_reason"] = (
                    "Suggestions could not be generated due to an error. Try again in a moment."
                )
                logger.warning("Improvement suggestions failed: %s", suggestions)
            else:
                result["suggestions_unavailable_reason"] = (
                    "The AI did not return usable structured suggestions after retries. "
                    "Try again, or check Ollama logs / raise OLLAMA_NUM_PREDICT_JSON_HEAVY in app/config.py."
                )
    except Exception:
        logger.exception("AI analysis failed for single resume (non-fatal)")
        if FEATURE_IMPROVEMENT_SUGGESTIONS:
            result["suggestions_unavailable"] = True
            result["suggestions_unavailable_reason"] = (
                "AI analysis encountered an error. Try again in a moment."
            )

    # Clean up internal fields
    result.pop("_resume_text", None)
    result.pop("_skill_result", None)

    return result
