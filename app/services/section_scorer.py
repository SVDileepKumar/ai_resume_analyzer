"""JD-aware section scoring for experience, education, and projects.

Every score is computed RELATIVE to what the Job Description asks for.
A PhD is only valuable if the JD values it. 10 years experience is only
valuable if the JD asks for senior-level candidates.
"""

from __future__ import annotations

import re
from functools import lru_cache

# Seniority level ordering (higher = more senior)
_SENIORITY_MAP = {"unknown": 0, "intern": 1, "junior": 2, "mid": 3, "senior": 4}


# ---------------------------------------------------------------------------
# JD Requirement Extraction
# ---------------------------------------------------------------------------

@lru_cache(maxsize=32)
def _extract_required_years(jd_text: str) -> int | None:
    """Extract minimum years of experience required from the JD."""
    jd_lower = jd_text.lower()
    patterns = [
        r"(\d{1,2})\+?\s*(?:years?|yrs?)\.?\s*(?:of)?\s*(?:experience|exp)",
        r"(?:at least|minimum|min)\s*(\d{1,2})\+?\s*(?:years?|yrs?)",
        r"(\d{1,2})\+?\s*(?:years?|yrs?)\.?\s*(?:in|of|working|relevant)",
    ]
    years = []
    for p in patterns:
        years.extend(int(m) for m in re.findall(p, jd_lower))
    return max(years) if years else None


@lru_cache(maxsize=32)
def _extract_seniority_level(jd_text: str) -> str:
    """Detect the seniority level the JD is hiring for."""
    jd_lower = jd_text.lower()
    if re.search(r"\b(?:senior|sr\.?|lead|staff|principal|architect)\b", jd_lower):
        return "senior"
    if re.search(r"\b(?:junior|jr\.?|entry.?level|graduate|new grad|fresher)\b", jd_lower):
        return "junior"
    if re.search(r"\b(?:intern|internship|trainee|apprentice|co.?op)\b", jd_lower):
        return "intern"
    return "mid"  # default assumption


@lru_cache(maxsize=32)
def _extract_required_education(jd_text: str) -> str:
    """Detect the education level the JD requires.

    Order matters: check PhD first, then masters, then bachelors.
    Each pattern must be specific enough to avoid false positives like
    "Scrum Master" or "master data management" triggering masters.
    """
    jd_lower = jd_text.lower()
    if re.search(r"\b(?:ph\.?d|doctorate|doctoral)\b", jd_lower):
        return "phd"
    # Masters: require degree-like context (not just bare "master")
    masters_patterns = [
        r"\bmaster(?:'?s)?\s+(?:degree|of|in)\b",
        r"\bmaster(?:'?s)?\s+(?:program|level|qualification)\b",
        r"\bmba\b",
        r"\bm\.?\s*s\.?\s+(?:in|degree|from)\b",
        r"\bm\.?\s*tech\b", r"\bm\.?\s*sc\b", r"\bm\.?\s*eng\b",
        r"\bpostgraduate\b",
        r"\bmaster(?:'?s)?\s+degree\b",
    ]
    if any(re.search(p, jd_lower) for p in masters_patterns):
        return "masters"
    # Bachelors: specific patterns to avoid matching bare "degree"
    bachelors_patterns = [
        r"\bbachelor(?:'?s)?\s+(?:degree|of|in)\b",
        r"\bb\.?\s*s\.?\s+(?:in|degree|from)\b",
        r"\bb\.?\s*tech\b", r"\bb\.?\s*sc\b", r"\bb\.?\s*eng\b",
        r"\bundergraduate\s+degree\b",
        r"\bbachelor\s+(?:degree|of)\b",
        # "degree in X" or "degree required" — generic but usually means bachelors
        r"\bdegree\s+(?:in|required|preferred|from)\b",
    ]
    if any(re.search(p, jd_lower) for p in bachelors_patterns):
        return "bachelors"
    if re.search(r"\b(?:diploma|associate(?:'?s)?\s+degree|certification)\b", jd_lower):
        return "diploma"
    # UK honours: 2:1, 2:2, First, upper/lower second
    if re.search(r"\b(?:first\s+class|2:1|2:2|upper\s+second|lower\s+second|2\.1|2\.2)\b", jd_lower):
        return "bachelors"
    # EU-style degree names (often equivalent to bachelors/masters)
    if re.search(r"\b(?:licentiate|diplom|magister|maîtrise|laurea|ingenieur)\b", jd_lower):
        return "bachelors"
    return "none"  # JD doesn't specify education


@lru_cache(maxsize=32)
def _extract_equivalent_years_from_jd(jd_text: str) -> int | None:
    """Extract 'degree or N years equivalent experience' from JD. Returns N or None."""
    jd_lower = jd_text.lower()
    patterns = [
        r"(?:bachelor'?s?|degree|master'?s?)\s+(?:or|/)\s*(\d{1,2})\s*(?:years?|yrs?)\s*(?:equivalent|experience)",
        r"(\d{1,2})\s*(?:years?|yrs?)\s*(?:equivalent|of\s+experience)\s*(?:or|in\s+lieu\s+of)\s*(?:a\s+)?degree",
        r"(?:or|/)\s*(\d{1,2})\+?\s*(?:years?|yrs?)\s*(?:equivalent|experience)",
    ]
    for p in patterns:
        m = re.search(p, jd_lower)
        if m:
            return int(m.group(1))
    return None


@lru_cache(maxsize=32)
def _extract_required_field_from_jd(jd_text: str) -> list[str]:
    """Extract required field(s) of study from JD (e.g. 'CS or equivalent', 'business or technical'). Returns normalized list of field keywords."""
    jd_lower = jd_text.lower()
    fields: list[str] = []
    # "degree in X", "CS or equivalent", "computer science", "business or technical"
    if re.search(r"\b(?:computer\s+science|cs|software\s+engineering|it)\b", jd_lower):
        fields.append("computer science")
    if re.search(r"\b(?:business|mba|finance|economics)\b", jd_lower) and re.search(r"\b(?:degree|bachelor|master)\b", jd_lower):
        fields.append("business")
    if re.search(r"\b(?:math(?:ematics)?|physics|engineering|technical)\b", jd_lower) and re.search(r"\b(?:degree|bachelor|master)\b", jd_lower):
        fields.extend(["math", "physics", "engineering"])
    if re.search(r"\b(?:data\s+science|statistics)\b", jd_lower):
        fields.append("data science")
    return list(dict.fromkeys(fields))  # preserve order, dedupe


# ---------------------------------------------------------------------------
# Experience Scoring (JD-aware)
# ---------------------------------------------------------------------------

# Seniority detection: we look for seniority QUALIFIERS as standalone words
# near role titles, rather than trying to match exact title patterns.
# This handles "Senior Software Engineer", "Lead Data Scientist",
# "Staff ML Engineer" etc. without needing to enumerate every combination.
_SENIOR_KEYWORDS = re.compile(
    r"\b(?:senior|sr\.?|lead|staff|principal|director|vp|head\s+of|chief|architect)\b",
    re.IGNORECASE,
)
_JUNIOR_KEYWORDS = re.compile(
    r"\b(?:junior|jr\.?|associate|entry[\s-]?level)\b",
    re.IGNORECASE,
)
_INTERN_KEYWORDS = re.compile(
    r"\b(?:intern|internship|trainee|apprentice|co[\s-]?op)\b",
    re.IGNORECASE,
)
_ROLE_NOUNS = re.compile(
    r"\b(?:engineer|developer|manager|analyst|scientist|designer|consultant|architect)\b",
    re.IGNORECASE,
)


def _years_from_date_ranges(text: str) -> float:
    """Infer total years from date ranges with overlap detection.

    Merges overlapping intervals (e.g. two concurrent roles) so the same
    calendar period is never counted twice. Returns approximate total span.
    """
    import datetime
    text_lower = text.lower()
    now_year = datetime.date.today().year

    def two_digit_to_year(g: str) -> int:
        n = int(g)
        return 2000 + n if n < 50 else 1900 + n

    intervals: list[tuple[int, int]] = []

    pattern1 = re.compile(
        r"(?:19|20)(\d{2})\s*[-\u2013]\s*(?:(?:19|20)(\d{2})|present|current|now)",
        re.IGNORECASE,
    )
    for m in pattern1.finditer(text_lower):
        start_year = two_digit_to_year(m.group(1))
        if m.lastindex >= 2 and m.group(2) is not None:
            end_year = two_digit_to_year(m.group(2))
        else:
            end_year = now_year
        if end_year >= start_year:
            intervals.append((start_year, end_year))

    pattern2 = re.compile(
        r"(?:(\d{1,2})[/\-.](\d{1,2})[/\-.](\d{4})|(\d{1,2})[/\-.](\d{4}))\s*[-\u2013]\s*(?:(?:(\d{1,2})[/\-.](\d{1,2})[/\-.](\d{4})|(\d{1,2})[/\-.](\d{4}))|present|current|now)",
        re.IGNORECASE,
    )
    for m in pattern2.finditer(text_lower):
        if m.group(3):
            start_year = int(m.group(3))
        elif m.group(5):
            start_year = int(m.group(5))
        else:
            continue
        if m.group(8):
            end_year = int(m.group(8))
        elif m.group(10):
            end_year = int(m.group(10))
        else:
            end_year = now_year
        if end_year >= start_year:
            intervals.append((start_year, end_year))

    if not intervals:
        return 0.0

    # Merge overlapping intervals to avoid double-counting concurrent roles
    intervals.sort()
    merged: list[tuple[int, int]] = [intervals[0]]
    for start, end in intervals[1:]:
        prev_start, prev_end = merged[-1]
        if start <= prev_end:
            merged[-1] = (prev_start, max(prev_end, end))
        else:
            merged.append((start, end))

    return sum(end - start for start, end in merged)


_EXPERIENCE_CONTEXT = (
    r"(?:experience|exp|professional|industry|hands[\s-]?on|relevant|"
    r"software|development|engineering|programming|working\s+(?:in|with|on|as))"
)


def _extract_candidate_years(text: str) -> tuple[int, bool]:
    """Extract max years of experience from resume text.

    Returns:
        (years_found, was_explicitly_stated) tuple.
    When no explicit phrase found, infers from date ranges (tenure) so career changers aren't penalized.
    """
    text_lower = text.lower()
    patterns = [
        rf"(\d{{1,2}})\+?\s*(?:years?|yrs?)\.?\s*(?:of)?\s*{_EXPERIENCE_CONTEXT}",
        rf"{_EXPERIENCE_CONTEXT}\s*(?:of|:)?\s*(\d{{1,2}})\+?\s*(?:years?|yrs?)",
        r"(\d{1,2})\+?\s*(?:years?|yrs?)\.?\s*(?:of)?\s*(?:experience|exp)",
        r"experience\s*(?:of|:)?\s*(\d{1,2})\+?\s*(?:years?|yrs?)",
    ]
    years = []
    for p in patterns:
        years.extend(int(m) for m in re.findall(p, text_lower))
    if years:
        return (max(years), True)
    # Infer from tenure when no explicit "X years" phrase
    tenure_years = _years_from_date_ranges(text)
    if tenure_years > 0:
        return (min(int(round(tenure_years)), 30), False)
    return (0, False)


# Seniority-to-years inference map (used when no explicit years mentioned)
_SENIORITY_YEARS_INFERENCE = {
    "senior": 8,    # Senior typically means 8+ years
    "mid": 4,       # Mid-level typically means 4-6 years
    "junior": 2,    # Junior typically means 1-3 years
    "intern": 0,    # Intern has no experience
    "unknown": 3,   # Default assumption for unknown seniority
}


def _detect_candidate_seniority(text: str) -> str:
    """Detect the highest seniority level in the resume.

    Seniority QUALIFIERS (senior, junior, intern) always take priority
    over generic role nouns. Among qualifiers, the highest wins.
    Falls back to 'mid' only when role nouns exist without any qualifier.
    """
    # Collect all qualifying levels found in text
    found_levels: list[int] = []
    if _SENIOR_KEYWORDS.search(text):
        found_levels.append(_SENIORITY_MAP["senior"])
    if _JUNIOR_KEYWORDS.search(text):
        found_levels.append(_SENIORITY_MAP["junior"])
    if _INTERN_KEYWORDS.search(text):
        found_levels.append(_SENIORITY_MAP["intern"])

    if found_levels:
        # Return the HIGHEST qualifier found
        best = max(found_levels)
        return next(k for k, v in _SENIORITY_MAP.items() if v == best)

    # No qualifiers — fall back to mid if they have role nouns
    if _ROLE_NOUNS.search(text):
        return "mid"
    return "unknown"


def score_experience(
    resume_text: str,
    jd_text: str,
    return_metadata: bool = False,
) -> float | tuple[float, dict]:
    """Score experience FIT against what the JD asks for (0-100).

    Args:
        resume_text: The candidate's resume text
        jd_text: The job description text
        return_metadata: If True, return (score, metadata) tuple with match_type info

    Returns:
        score (0-100) or (score, metadata) tuple if return_metadata=True

    Scoring logic:
    - If JD asks for 5+ years and resume has 7 → high score
    - If JD asks for senior and resume has senior roles → high score
    - If JD asks for junior but resume is senior → still decent (overqualified)
    - If JD asks for senior but resume is intern → low score

    When no explicit years are mentioned in the resume, we infer years from
    the seniority level (e.g., "Senior Engineer" → ~8 years) with reduced
    confidence weighting.
    """
    # What the JD wants
    required_years = _extract_required_years(jd_text)
    required_seniority = _extract_seniority_level(jd_text)

    # What the candidate has
    candidate_years, years_explicit = _extract_candidate_years(resume_text)
    candidate_seniority = _detect_candidate_seniority(resume_text)

    # Infer years from seniority if not explicitly stated
    years_inferred = False
    if not years_explicit and candidate_seniority != "unknown":
        candidate_years = _SENIORITY_YEARS_INFERENCE.get(candidate_seniority, 3)
        years_inferred = True

    # Date range evidence (tenure)
    date_ranges = re.findall(
        r"(?:19|20)\d{2}\s*[-\u2013]\s*(?:(?:19|20)\d{2}|present|current|now)",
        resume_text.lower(),
    )

    score = 0.0

    # --- Years match (up to 40 pts) ---
    if required_years is not None:
        if candidate_years >= required_years:
            pts = 40.0 if years_explicit else 32.0
            score += pts
        elif candidate_years > 0:
            ratio = candidate_years / required_years
            base_pts = 40.0 if years_explicit else 32.0
            score += round(base_pts * min(ratio, 1.0), 1)
    else:
        multiplier = 5 if years_explicit else 4
        score += min(candidate_years * multiplier, 30)

    # --- Seniority match (up to 30 pts) ---
    required_level = _SENIORITY_MAP.get(required_seniority, 2)
    candidate_level = _SENIORITY_MAP.get(candidate_seniority, 0)

    if candidate_level >= required_level:
        score += 30.0
    elif candidate_level > 0:
        ratio = candidate_level / max(required_level, 1)
        score += round(30.0 * ratio, 1)

    # --- Tenure evidence (up to 15 pts) ---
    score += min(len(date_ranges) * 4, 15)

    # --- Recency bonus (up to 15 pts): reward current/recent employment ---
    import datetime
    now_year = datetime.date.today().year
    has_current = bool(re.search(
        r"(?:present|current|now)\b",
        resume_text.lower(),
    ))
    has_recent = bool(re.search(
        rf"\b(?:{now_year}|{now_year - 1})\b",
        resume_text,
    ))
    if has_current:
        score += 15.0
    elif has_recent:
        score += 10.0
    elif date_ranges:
        score += 5.0

    final_score = min(round(score, 1), 100.0)

    if return_metadata:
        # Determine match type for UI display
        if candidate_level > required_level + 1:
            match_type = "overqualified"
        elif candidate_level < required_level - 1:
            match_type = "underqualified"
        else:
            match_type = "exact"

        metadata = {
            "match_type": match_type,
            "candidate_years": candidate_years,
            "years_explicit": years_explicit,
            "years_inferred": years_inferred,
            "required_years": required_years,
            "candidate_seniority": candidate_seniority,
            "required_seniority": required_seniority,
        }
        return final_score, metadata

    return final_score


# ---------------------------------------------------------------------------
# Education Scoring (JD-aware)
# ---------------------------------------------------------------------------

_DEGREE_LEVELS = {"none": 0, "diploma": 1, "bachelors": 2, "masters": 3, "phd": 4}


def _detect_candidate_education(text: str) -> str:
    """Detect highest education level in resume."""
    text_lower = text.lower()

    if re.search(r"\b(?:ph\.?d|doctorate|doctoral)\b", text_lower):
        return "phd"

    masters_patterns = [
        r"\bmaster(?:'s|s)?\s+(?:degree|of|in)\b",
        r"\bm\.?\s*s\.?\s+(?:in|degree)\b",
        r"\bm\.?\s*s\.?\b(?!\w)",
        r"\bm\.?\s*tech\b", r"\bm\.?\s*sc\b", r"\bmba\b",
        r"\bm\.?\s*eng\b", r"\bm\.?\s*a\.?\s+(?:in|degree)\b",
        r"\bpost\s*graduate|postgraduate\b",
    ]
    if any(re.search(p, text_lower) for p in masters_patterns):
        return "masters"

    bachelors_patterns = [
        r"\bbachelor(?:'s|s)?\s+(?:degree|of|in)\b",
        r"\bb\.?\s*s\.?\s+(?:in|degree)\b",
        r"\bb\.?\s*tech\b", r"\bb\.?\s*sc\b", r"\bb\.?\s*eng\b",
        r"\bb\.?\s*a\.?\s+(?:in|degree)\b",
        r"\bundergraduate\s+degree\b",
        r"\bbachelor\s+of\s+technology\b",
    ]
    if any(re.search(p, text_lower) for p in bachelors_patterns):
        return "bachelors"

    if re.search(r"\b(?:diploma|associate(?:'s)?\s+degree|certification)\b", text_lower):
        return "diploma"
    # UK honours (2:1, First class, etc.) — treat as bachelors-level
    if re.search(r"\b(?:first\s+class|2:1|2:2|upper\s+second|lower\s+second|2\.1|2\.2)\b", text_lower):
        return "bachelors"
    # EU-style degree names
    if re.search(r"\b(?:licentiate|diplom|magister|maîtrise|laurea|ingenieur)\b", text_lower):
        return "bachelors"

    return "none"


# Matches "CGPA: 8.84/10", "GPA: 3.9/4.0", "Score: 981/1000", "8.84/10.0", etc.
# Labelled score: "CGPA: 8.84/10.0", "Score: 981/1000", "GPA 3.9/4"
_GPA_LABELLED_PATTERN = re.compile(
    r"(cgpa|gpa|score|marks?|percentage|grade)\s*:?\s*(\d{1,3}(?:\.\d{1,2})?)\s*/\s*(\d{1,4}(?:\.\d{1,2})?)",
    re.IGNORECASE,
)

# Bare fraction: "8.84/10.0", "3.9/4" — only trusted for known denominators
_BARE_GPA_PATTERN = re.compile(
    r"\b(\d{1,2}\.\d{1,2})\s*/\s*(\d{1,2}(?:\.\d)?)\b",
)

# Degree titles with optional specialisation ("in X")
_DEGREE_TITLE_PATTERN = re.compile(
    r"\b(b\.?\s*tech|b\.?\s*e\.?|b\.?\s*sc\.?|b\.?\s*s\.?|bachelor\s+of\s+\w+(?:\s+\w+)?"
    r"|m\.?\s*tech|m\.?\s*e\.?|m\.?\s*sc\.?|m\.?\s*s\.?|master\s+of\s+\w+(?:\s+\w+)?"
    r"|mba|ph\.?d\.?|doctorate"
    r"|intermediate|10\+2|hsc|ssc|secondary|senior\s+secondary|class\s+(?:x{1,2}|\d{1,2})"
    r")\b(?:\s+in\s+([A-Za-z &/()+]+))?",
    re.IGNORECASE,
)

# Institution: word before a known suffix (NIT, IIT, university, college, etc.)
# Anchored to stop at newline so we don't grab too much
_INSTITUTION_PATTERN = re.compile(
    r"([A-Z][A-Za-z\s&,.'()-]{2,50}"
    r"(?:university|institute|college|school|academy|iit|nit|iiit|bits|iim|iiser"
    r"|institution|polytechnic|faculty|campus)(?:\s+[A-Za-z,.'()-]{1,30})?)",
    re.IGNORECASE,
)


def _normalise_gpa(value: float, out_of: float) -> float | None:
    """Convert raw score to a 0-10 scale for uniform comparison."""
    if out_of <= 0:
        return None
    if out_of <= 10:
        return round(value / out_of * 10, 2)
    if out_of <= 100:
        return round(value / out_of * 10, 2)
    if out_of <= 1000:
        return round(value / out_of * 10, 2)
    return None


def extract_education_details(text: str) -> list[dict]:
    """Parse education entries from resume text.

    Returns a list of dicts, one per education entry detected, each with:
        degree       – detected degree type (e.g. "B.Tech")
        field        – field of study if mentioned (e.g. "Computer Science & Engineering")
        institution  – institution name if detectable
        raw_score    – original score string (e.g. "8.84/10.0" or "981/1000")
        gpa_10       – score normalised to /10 scale (float), or None
        score_label  – "CGPA", "GPA", "Score", "Marks", or "Percentage"

    Entries are ordered as they appear in the resume.
    Falls back gracefully: missing fields are None / empty string.
    """
    entries: list[dict] = []
    lines = text.splitlines()

    # Build a line-indexed list of (line_no, degree_match) for anchoring
    degree_hits: list[tuple[int, re.Match]] = []
    for i, line in enumerate(lines):
        m = _DEGREE_TITLE_PATTERN.search(line)
        if m:
            degree_hits.append((i, m))

    for idx in range(len(degree_hits)):
        line_no, deg_match = degree_hits[idx]
        degree_str = deg_match.group(1).strip()
        field_str = (deg_match.group(2) or "").strip()

        # Search window: this line + next 5 lines for institution and score
        window_start = line_no
        window_end = min(line_no + 6, len(lines))
        # Don't bleed into the next degree block
        if idx + 1 < len(degree_hits):
            window_end = min(window_end, degree_hits[idx + 1][0])
        window_text = "\n".join(lines[window_start:window_end])

        # Institution detection — search line-by-line to avoid grabbing preamble
        institution = ""
        for win_line in lines[window_start:window_end]:
            inst_match = _INSTITUTION_PATTERN.search(win_line)
            if inst_match:
                candidate_inst = re.sub(r"\s+", " ", inst_match.group(1).strip())
                # If this line also has a degree token before the institution,
                # trim everything up to and including the last comma before the match
                deg_in_line = _DEGREE_TITLE_PATTERN.match(win_line.strip())
                if deg_in_line:
                    # Take only the portion after the last comma on the line
                    comma_parts = win_line.split(",")
                    if len(comma_parts) > 1:
                        trailing = comma_parts[-1].strip()
                        if trailing and len(trailing) > 3:
                            candidate_inst = trailing
                if candidate_inst:
                    institution = candidate_inst
                    break

        # Score detection — prefer labelled match, then bare fraction
        raw_score = ""
        gpa_10: float | None = None
        score_label = ""

        labelled = _GPA_LABELLED_PATTERN.search(window_text)
        if labelled:
            try:
                lbl_raw = labelled.group(1).strip().upper()
                val = float(labelled.group(2))
                out_of = float(labelled.group(3))
                gpa_10 = _normalise_gpa(val, out_of)
                raw_score = f"{labelled.group(2)}/{labelled.group(3)}"
                _label_map = {
                    "CGPA": "CGPA", "GPA": "GPA",
                    "SCORE": "Score", "MARKS": "Marks",
                    "MARK": "Marks", "PERCENTAGE": "Percentage",
                    "GRADE": "Grade",
                }
                score_label = _label_map.get(lbl_raw, "Score")
            except ValueError:
                pass
        else:
            bare = _BARE_GPA_PATTERN.search(window_text)
            if bare:
                try:
                    val = float(bare.group(1))
                    out_of = float(bare.group(2))
                    if out_of in (4.0, 5.0, 10.0):
                        gpa_10 = _normalise_gpa(val, out_of)
                        raw_score = f"{bare.group(1)}/{bare.group(2)}"
                        score_label = "GPA"
                except ValueError:
                    pass

        entries.append({
            "degree": degree_str,
            "field": field_str,
            "institution": institution,
            "raw_score": raw_score,
            "gpa_10": gpa_10,
            "score_label": score_label,
        })

    return entries


def score_education(resume_text: str, jd_text: str) -> float:
    """Score education FIT against what the JD asks for (0-100).

    - If JD requires bachelors and resume has masters → 100
    - If JD requires masters and resume has bachelors → partial credit
    - If JD doesn't mention education → generous baseline
    """
    required = _extract_required_education(jd_text)
    candidate = _detect_candidate_education(resume_text)

    required_level = _DEGREE_LEVELS.get(required, 0)
    candidate_level = _DEGREE_LEVELS.get(candidate, 0)

    text_lower = resume_text.lower()
    score = 0.0

    if required_level == 0:
        # JD doesn't specify education → give baseline credit
        score = min(candidate_level * 20, 70)
    elif candidate_level >= required_level:
        score = 85.0  # meets or exceeds
    elif candidate_level > 0:
        ratio = candidate_level / required_level
        score = round(85.0 * ratio, 1)

    # Bonus: relevant field of study (fixed list + JD-required field match)
    jd_fields = _extract_required_field_from_jd(jd_text)
    resume_has_field = bool(re.search(
        r"\b(?:computer science|software engineering|information technology"
        r"|data science|electrical engineering|mathematics|statistics"
        r"|physics|business|economics|engineering)\b",
        text_lower,
    ))
    if jd_fields:
        resume_matches_jd_field = any(
            any(f in text_lower for f in (field.split() if " " in field else [field]))
            for field in jd_fields
        )
        if resume_matches_jd_field:
            score = min(score + 10, 100.0)
    elif resume_has_field:
        score = min(score + 10, 100.0)

    # Bonus: honors / strong GPA — reward explicitly high scores
    edu_details = extract_education_details(resume_text)
    has_strong_gpa = any(
        d["gpa_10"] is not None and d["gpa_10"] >= 8.0
        for d in edu_details
    )
    if has_strong_gpa:
        score = min(score + 8, 100.0)
    elif re.search(r"\b(?:gpa|cgpa|distinction|honors?|cum laude|summa)\b", text_lower):
        score = min(score + 5, 100.0)

    return round(score, 1)


# ---------------------------------------------------------------------------
# Project Scoring (JD-aware)
# ---------------------------------------------------------------------------

_PROJECT_SECTION = re.compile(
    r"(?:^|\n)\s*(?:projects?|personal projects?|side projects?"
    r"|key projects?|academic projects?|portfolio)\s*[:\n\-|]",
    re.IGNORECASE,
)
_NON_PROJECT_SECTION = re.compile(
    r"(?:^|\n)\s*(?:experience|work\s+experience|employment|professional\s+experience"
    r"|education|skills|certifications?|awards?|publications?|references?)\s*[:\n\-|]",
    re.IGNORECASE,
)


def _extract_project_section(text: str) -> str:
    """Isolate the 'Projects' section from the full resume text."""
    proj_match = _PROJECT_SECTION.search(text)
    if not proj_match:
        return text

    start = proj_match.end()
    next_section = _NON_PROJECT_SECTION.search(text, pos=start)
    end = next_section.start() if next_section else len(text)

    section_text = text[start:end].strip()
    return section_text if len(section_text) > 30 else text


_ACTION_VERBS_PATTERN = re.compile(
    r"\b(?:built|developed|created|implemented|designed|deployed"
    r"|architected|integrated|automated|optimized|engineered"
    r"|delivered|launched|migrated|refactored|modernized|scaled)\b",
    re.IGNORECASE,
)

# Detects named deliverables / products inside experience bullets (e.g. "• Wolf (Online Facilitator):")
_NAMED_DELIVERABLE_PATTERN = re.compile(
    r"(?:^|\n)\s*[\u2022\-\*\u25cf]\s*[A-Z][A-Za-z0-9\s\-]{2,40}(?:\([^)]{2,40}\))?:",
)


def score_projects(resume_text: str, jd_text: str) -> float:
    """Score project RELEVANCE to the JD (0-100).

    Checks the explicit Projects section first. When absent, detects
    project-like content embedded in the Experience section — named
    deliverables, action-verb bullet points, and tech stack mentions —
    and awards partial credit rather than a flat penalty. Senior engineers
    at large companies routinely describe shipped products inside
    experience bullets rather than a separate Projects section.
    """
    has_section = bool(_PROJECT_SECTION.search(resume_text))
    project_text = _extract_project_section(resume_text) if has_section else ""
    proj_lower = project_text.lower()
    jd_lower = jd_text.lower()
    full_lower = resume_text.lower()

    # --- Detect implicit project content in Experience when no Projects section ---
    # Named deliverables (e.g. "• Wolf (Online Facilitator):") signal real shipped work.
    has_embedded_projects = (
        not has_section
        and bool(_NAMED_DELIVERABLE_PATTERN.search(resume_text))
        and len(_ACTION_VERBS_PATTERN.findall(resume_text)) >= 4
    )

    # --- Has project section? (up to 10 pts) ---
    # Embedded projects get partial credit (6 pts) vs explicit section (10 pts).
    if has_section:
        section_score = 10
    elif has_embedded_projects:
        section_score = 6
    else:
        section_score = 0

    # --- Project bullet quality (up to 20 pts) ---
    if has_section:
        bullets = re.findall(r"(?:^|\n)\s*[\u2022\-\*\u25cf]\s*.{20,}", proj_lower)
        bullet_score = min(len(bullets) * 4, 20)
    elif has_embedded_projects:
        # Count action-verb bullets in the full resume (experience bullets)
        exp_bullets = re.findall(
            r"(?:^|\n)\s*[\u2022\-\*\u25cf]\s*.{20,}", full_lower
        )
        bullet_score = min(len(exp_bullets) * 2, 14)  # capped lower than explicit section
    else:
        bullet_score = 0

    # --- Action verbs in project section (up to 15 pts) ---
    if has_section:
        action_verbs = _ACTION_VERBS_PATTERN.findall(proj_lower)
        verb_score = min(len(action_verbs) * 3, 15)
    elif has_embedded_projects:
        action_verbs = _ACTION_VERBS_PATTERN.findall(full_lower)
        verb_score = min(len(action_verbs) * 2, 10)  # capped lower than explicit section
    else:
        verb_score = 0

    # --- JD keyword overlap ---
    # Explicit section → up to 40 pts; embedded projects → up to 32 pts; no projects → up to 20 pts.
    jd_keywords = set(re.findall(r"\b[a-z][a-z0-9+#.]{1,}\b", jd_lower))
    _stopwords = {
        "the", "and", "for", "with", "that", "this", "are", "was",
        "will", "from", "have", "has", "been", "our", "your", "you",
        "not", "but", "all", "can", "had", "her", "one", "who",
        "their", "there", "what", "about", "which", "when", "make",
        "like", "than", "each", "other", "into", "more", "some",
        "should", "would", "could", "must", "also", "such", "work",
        "using", "used", "including", "able", "strong", "good",
        "experience", "required", "preferred", "looking", "team",
    }
    jd_keywords -= _stopwords

    if jd_keywords:
        if has_section:
            search_text, max_relevance = proj_lower, 40
        elif has_embedded_projects:
            search_text, max_relevance = full_lower, 32
        else:
            search_text, max_relevance = full_lower, 20
        text_keywords = set(re.findall(r"\b[a-z][a-z0-9+#.]{1,}\b", search_text))
        overlap = text_keywords & jd_keywords
        relevance_ratio = len(overlap) / len(jd_keywords)
        relevance_score = round(
            max_relevance * min(relevance_ratio * 2.5, 1.0), 1,
        )
    else:
        relevance_score = 10 if has_section else (7 if has_embedded_projects else 5)

    # --- Portfolio/GitHub links (up to 15 pts) ---
    has_links = bool(re.search(
        r"(?:github\.com|gitlab\.com|bitbucket|portfolio|demo|heroku|vercel|netlify)",
        full_lower,
    ))
    link_score = 15 if has_links else 0

    return min(
        round(section_score + bullet_score + verb_score + relevance_score + link_score, 1),
        100.0,
    )


# ---------------------------------------------------------------------------
# Certifications Scoring (JD-aware)
# ---------------------------------------------------------------------------

# Common certification patterns (JD and resume). Normalized to lowercase for matching.
_CERT_PATTERNS = re.compile(
    r"\b(?:"
    r"pmp|pmbok|aws\s+certified|azure\s+certified|gcp\s+certified|"
    r"cissp|cism|comptia\s+a\+|comptia\s+network\+|"
    r"scrum\s+master|csam|csap|splunk\s+certified|"
    r"salesforce\s+certified|google\s+cloud\s+certified|"
    r"certified\s+[a-z\s]+(?:professional|associate|specialty)?|"
    r"[a-z]+\s+certification|certification\s+in\s+[a-z\s]+"
    r")\b",
    re.IGNORECASE,
)


@lru_cache(maxsize=128)
def _extract_certs(text: str) -> frozenset[str]:
    """Extract certification-like phrases from text. Returns normalized frozenset (cached)."""
    found: set[str] = set()
    for m in _CERT_PATTERNS.finditer(text.lower()):
        found.add(re.sub(r"\s+", " ", m.group(0).strip()))
    return frozenset(found)


def score_certifications(resume_text: str, jd_text: str) -> float:
    """Score certification match (0-100).

    When JD does not mention certs, return a neutral midpoint (50) so this
    dimension neither inflates nor penalizes the final score.
    """
    jd_certs = _extract_certs(jd_text)
    if not jd_certs:
        return 50.0
    resume_certs = _extract_certs(resume_text)
    matched = sum(1 for jc in jd_certs if any(jc in rc or rc in jc for rc in resume_certs))
    return round((matched / len(jd_certs)) * 100, 1) if jd_certs else 50.0


def extract_certifications_details(text: str) -> list[str]:
    """Return a deduplicated list of certification names found in the full resume text.

    Uses the same regex as ``_extract_certs`` but returns a plain sorted list
    suitable for LLM grounding prompts.  Operates on the full (untruncated) text
    so certs that appear after the LLM's character limit are still captured.
    """
    return sorted(_extract_certs(text))


# ---------------------------------------------------------------------------
# Score Normalization (prevents dimension bias)
# ---------------------------------------------------------------------------

# Default observed score bounds. Override via config DIMENSION_BOUNDS_OVERRIDE
# (e.g. "similarity:15,95" to change similarity bounds). Bounds map raw scores to 0-100.
_DIMENSION_BOUNDS_DEFAULT = {
    "skills": (0, 100),
    "similarity": (20, 90),   # Semantic/TF-IDF blend rarely exceeds ~90
    "experience": (10, 100),
    "education": (0, 100),
    "projects": (15, 90),
    "certifications": (0, 100),
}


def _get_dimension_bounds() -> dict[str, tuple[float, float]]:
    """Return dimension bounds, applying optional env override."""
    from app.config import DIMENSION_BOUNDS_OVERRIDE

    bounds = dict(_DIMENSION_BOUNDS_DEFAULT)
    if not DIMENSION_BOUNDS_OVERRIDE or not DIMENSION_BOUNDS_OVERRIDE.strip():
        return bounds
    # Parse "dim:low,high" pairs separated by semicolon
    for part in DIMENSION_BOUNDS_OVERRIDE.strip().split(";"):
        part = part.strip()
        if ":" not in part:
            continue
        dim, rest = part.split(":", 1)
        dim = dim.strip().lower()
        if dim not in bounds:
            continue
        try:
            low_str, high_str = rest.split(",", 1)
            low, high = float(low_str.strip()), float(high_str.strip())
            if low <= high:
                bounds[dim] = (low, high)
        except ValueError:
            continue
    return bounds


_DIMENSION_BOUNDS_AI = {
    "skills": (0, 100),
    "similarity": (0, 100),
    "experience": (0, 100),
    "education": (0, 100),
    "projects": (0, 100),
    "certifications": (0, 100),
}


def normalize_scores(
    scores: dict[str, float],
    enabled: bool = True,
    *,
    ai_sourced: bool = False,
) -> dict[str, float]:
    """Normalize section scores to a comparable 0-100 scale.

    Different sections have different natural score ranges:
    - Skills can easily hit 100% if all keywords match
    - Similarity (semantic) rarely exceeds 85-90%
    - Projects rarely exceeds 85% even with great projects

    This function maps each section's natural range to 0-100 so that
    all dimensions are comparable when computing the weighted final score.

    Args:
        scores: Dict of dimension name → raw score (0-100)
        enabled: If False, returns scores unchanged (for feature flag)
        ai_sourced: If True, use wider bounds since LLM already returns
                    calibrated 0-100 scores (avoids double-normalization)

    Returns:
        Dict of dimension name → normalized score (0-100)
    """
    from app.config import FEATURE_SCORE_NORMALIZATION

    if not enabled or not FEATURE_SCORE_NORMALIZATION:
        return scores.copy()

    bounds_map = _DIMENSION_BOUNDS_AI if ai_sourced else _get_dimension_bounds()
    normalized = {}
    for dim, score in scores.items():
        bounds = bounds_map.get(dim, (0, 100))
        low, high = bounds

        # Clamp to observed bounds
        clamped = max(low, min(score, high))

        # Map [low, high] → [0, 100]
        if high > low:
            normalized[dim] = round(((clamped - low) / (high - low)) * 100, 1)
        else:
            normalized[dim] = clamped

    return normalized
