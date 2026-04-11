"""Shared text normalization for resume/JD scoring (e.g. boilerplate stripping)."""

import re

# Common resume boilerplate phrases to strip so they don't dilute keyword density
_BOILERPLATE_PHRASES = [
    "references available upon request",
    "references furnished upon request",
    "confidential",
    "strictly confidential",
    "do not distribute",
    "all rights reserved",
    "page 1 of",
    "page 2 of",
    "continued on next page",
]


def strip_boilerplate(text: str) -> str:
    """Remove common boilerplate phrases from resume text; replace with space to avoid joining words."""
    if not text or not text.strip():
        return text
    out = text
    for phrase in _BOILERPLATE_PHRASES:
        # Case-insensitive replace; use space so "confidential" doesn't glue words
        while phrase.lower() in out.lower():
            start = out.lower().index(phrase.lower())
            out = out[:start] + " " + out[start + len(phrase) :]
    # Collapse multiple spaces/newlines that might result
    out = re.sub(r"[ \t]+", " ", out)
    out = re.sub(r"\n\s*\n\s*\n+", "\n\n", out)
    return out.strip()
