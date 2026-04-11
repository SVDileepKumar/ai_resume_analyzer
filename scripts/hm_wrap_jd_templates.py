#!/usr/bin/env python3
"""One-off builder: wrap existing jd_templates.json bodies with hiring-manager framing."""

from __future__ import annotations

import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "app" / "data" / "jd_templates.json"
OUT = ROOT / "app" / "data" / "jd_templates.json"

DISC = (
    "(Sample job description for demo only — not affiliated with any employer; language is illustrative for ATS scoring demos.)"
)

HM_OPEN: dict[str, str] = {
    "software engineer": "I’m hiring someone who can own services end to end: design, delivery, and a fair share of on-call. I’ll prioritize judgment, reliability, and how you help the team get better.",
    "frontend developer": "I want a frontend engineer who treats accessibility and performance as product features, not polish at the end. Show me how you partner with design and backend to ship fast without hiding debt.",
    "backend developer": "I need a backend engineer who ships APIs and data paths I can trust in production—clear contracts, observability, and calm incident habits matter as much as raw coding speed.",
    "full stack developer": "I’m looking for someone who can carry a feature across UI, API, and data with pragmatism. I care about end-to-end ownership and honest tradeoffs when scope or time is tight.",
    "data analyst": "I hire analysts who make metrics trustworthy and explainable. I’ll push on definitions, edge cases, and how you influence decisions—not just how pretty the chart is.",
    "data scientist": "I want a scientist who connects models to business decisions: measurement, ethics, and what happens when the model is wrong. Partnering with engineering on production reality is non-negotiable.",
    "devops engineer": "I’m building a platform team that removes toil and keeps SLOs honest. I’ll look for depth in Linux, K8s, and cloud—and for how you teach other teams to use the paved path.",
    "ml engineer": "I need ML engineers who treat serving, monitoring, and rollback as first-class. I’ll ask how you work with scientists and how you keep costs and latency under control.",
    "qa engineer": "I value QA who blend risk-based judgment with automation that earns its maintenance. I’ll notice how you work with devs before bugs hit customers, not just how you file tickets.",
    "product manager": "I hire PMs who write crisply, decide with data, and earn trust from engineering. I’ll probe how you say no, how you handle ambiguity, and how you connect roadmap to revenue or retention.",
    "data engineer": "I want data engineers who care about lineage, SLAs, and cost as much as pipelines. I’ll prioritize clarity in data contracts and how you partner with analysts and ML.",
    "mobile developer": "I’m hiring for store-quality releases: performance, stability, and sensible release hygiene. Show me how you profile, how you handle rollbacks, and how you work with backend on APIs.",
    "sdet": "I’m looking for an SDET who codes like an engineer and thinks like a quality partner. Framework design, CI gates, and killing flakes before they rot are what I’ll dig into.",
    "test lead": "I need a test lead who can plan, estimate, and communicate upward without drama. Traceability, risk-based coverage, and developing the team matter as much as your own hands-on skills.",
    "performance test engineer": "I hire performance engineers who translate NFRs into scenarios and tell a credible story to architects and SREs. I’ll ask how you isolate bottlenecks and how you document results.",
    "sap consultant": "I’m staffing SAP work that will be scrutinized in workshops and UAT. I’ll prioritize structured documentation, stakeholder patience, and clean handoffs between offshore and onsite.",
    "servicenow developer": "I want a ServiceNow developer who can translate ITIL intent into maintainable flows and integrations. I’ll look for disciplined migrations, testing around upgrades, and clear runbooks.",
    "salesforce developer": "I need Salesforce devs who respect the security model and can pair with BAs under delivery pressure. I’ll ask about deployments, reviews, and how you keep tech debt visible.",
    "snowflake engineer": "I’m hiring for Snowflake-centric delivery with cost and governance in mind. I’ll prioritize SQL depth, dbt or equivalent discipline, and how you work with BI consumers.",
    "business analyst": "I hire BAs who turn fuzzy asks into testable requirements and keep traceability honest. Facilitation, writing, and calm UAT coordination are what I’ll evaluate hardest.",
    "hr business partner": "I want an HRBP who partners with managers on hard people decisions with empathy and policy rigor. I’ll look for judgment, discretion, and how you scale yourself across many leaders.",
    "financial analyst": "I need an analyst who builds models leadership can trust and explains variance without hand-waving. I’ll probe Excel discipline, business partnership, and audit readiness.",
    "technical recruiter": "I hire recruiters who earn hiring-manager trust with prep, calibration, and candidate experience. I’ll notice funnel hygiene, structured notes, and how you represent our bar fairly.",
    "customer success manager": "I’m looking for a CSM who drives adoption and renewal with data, not hope. I’ll prioritize QBR quality, cross-functional triage, and how you align with sales without dropping the customer.",
}

SHORTLIST = """What will make you stand out on my shortlist
- Concrete impact: scope, constraints, metrics, and what you learned when things went wrong
- Artifacts you can discuss (doc, dashboard, test plan, design note, model card—whatever fits your craft)
- How you communicate risk and tradeoffs before they become surprises
- How you collaborate across roles—not only your lane
- Curiosity about our users, our constraints, and how we measure success"""

INTERVIEW = """Interview process
Recruiter or scheduling screen → skills or practical conversation → panel or deep-dive with peers → hiring manager discussion → offer (steps may vary by level and location)."""

WORK_WITH = """How you'll work with me and the team
- We align on outcomes; I expect crisp async updates and early flags when plans change.
- You'll partner with peers across disciplines; I value ownership of handoffs, not heroics in a silo."""


def strip_header_line(blob: str) -> tuple[str, str]:
    """Return (title_line, remainder_after_first_paragraph)."""
    blob = blob.strip()
    first_break = blob.find("\n\n")
    if first_break == -1:
        return blob, ""
    head = blob[:first_break].strip()
    rest = blob[first_break:].strip()
    return head, rest


def title_only(head: str) -> str:
    """Remove trailing parenthetical disclaimer from first line if present."""
    head = head.strip()
    m = re.match(r"^(.+?)\s*\(Sample job description", head, re.I)
    if m:
        return m.group(1).strip()
    return head


def main() -> None:
    data = json.loads(SRC.read_text(encoding="utf-8"))
    out: dict[str, str] = {}
    for key in sorted(data.keys()):
        head, body = strip_header_line(data[key])
        title = title_only(head)
        hm = HM_OPEN.get(key)
        if not hm:
            raise SystemExit(f"Missing HM_OPEN for role: {key}")
        # Drop redundant opening "Posting style" label by keeping full body (includes duties and must/nice)
        out[key] = (
            f"{title}\n\n{DISC}\n\n"
            f"A note from your hiring manager\n{hm}\n\n"
            f"{WORK_WITH}\n\n"
            f"The role in practice\n{body}\n\n"
            f"{SHORTLIST}\n\n"
            f"{INTERVIEW}"
        )
    OUT.write_text(json.dumps(out, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print("Wrote", len(out), "templates to", OUT)


if __name__ == "__main__":
    main()
