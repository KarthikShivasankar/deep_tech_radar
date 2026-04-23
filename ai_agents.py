"""
ai_agents.py — AI Research Assistant powered by OpenAI GPT.

Five streaming agents that analyse the group's skill radar data and surface
actionable insights: team synergies, project ideas, funding proposals, skill
gaps, and a competence development roadmap.

Model: configured via OPENAI_MODEL in config.py / .env (default: gpt-4o-mini).
       Set OPENAI_MODEL=gpt-5-nano once that model ID is available.
"""

from __future__ import annotations

import logging
from typing import Iterator

import pandas as pd

from config import (
    DEFAULT_TECH_AREAS,
    DIMENSIONS,
    OPENAI_API_KEY,
    OPENAI_MODEL,
    OPENAI_TIMEOUT,
    TW_QUADRANTS,
    TW_RINGS,
    TW_RING_THRESHOLDS,
)

log = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# OpenAI client & health check
# ─────────────────────────────────────────────────────────────────────────────

def _get_client():
    from openai import OpenAI
    return OpenAI(api_key=OPENAI_API_KEY)


def check_openai() -> tuple[bool, str]:
    """Verify OpenAI API key and model accessibility."""
    if not OPENAI_API_KEY:
        return False, "OPENAI_API_KEY not set in .env"
    try:
        client = _get_client()
        client.models.retrieve(OPENAI_MODEL)
        return True, f"OpenAI ready — using {OPENAI_MODEL}"
    except Exception as exc:
        return False, str(exc)


# ─────────────────────────────────────────────────────────────────────────────
# Context builder
# ─────────────────────────────────────────────────────────────────────────────

def _ring_name(expertise, contribute) -> str:
    try:
        avg = (float(expertise) + float(contribute)) / 2
    except (TypeError, ValueError):
        return TW_RINGS[-1]["name"]
    for idx, threshold in enumerate(TW_RING_THRESHOLDS):
        if avg >= threshold:
            return TW_RINGS[idx]["name"]
    return TW_RINGS[-1]["name"]


def _build_radar_context(df: pd.DataFrame, vision_data: list[dict]) -> str:
    sections: list[str] = []

    n_people = df["researcher"].nunique() if not df.empty else 0
    sections.append(f"**Group:** {n_people} researchers rated deep tech areas.\n")

    if not df.empty:
        area_stats: dict[str, dict] = {}
        for area in df["tech_area"].unique():
            sub = df[df["tech_area"] == area]
            quadrant = "Other"
            for q_name, q_data in TW_QUADRANTS.items():
                if area in q_data["areas"]:
                    quadrant = q_name
                    break
            area_stats[area] = {
                "quadrant":   quadrant,
                "interest":   round(sub["interest"].mean(), 1) if "interest" in sub else "–",
                "expertise":  round(sub["expertise"].mean(), 1) if "expertise" in sub else "–",
                "contribute": round(sub["contribute"].mean(), 1) if "contribute" in sub else "–",
                "ring":       _ring_name(
                    sub["expertise"].mean() if "expertise" in sub else 3,
                    sub["contribute"].mean() if "contribute" in sub else 3,
                ),
                "n": len(sub),
            }

        rows = ["Area | Quadrant | Ring | Interest | Expertise | Contribute | N",
                "---|---|---|---|---|---|---"]
        for area, s in area_stats.items():
            rows.append(
                f"{area} | {s['quadrant']} | {s['ring']} | "
                f"{s['interest']} | {s['expertise']} | {s['contribute']} | {s['n']}"
            )
        sections.append("**Tech Area Scores (group averages):**\n" + "\n".join(rows) + "\n")

        researcher_rows = ["Researcher | Top contribute areas (score)", "---|---"]
        for researcher in sorted(df["researcher"].unique()):
            sub_r = df[df["researcher"] == researcher].sort_values("contribute", ascending=False)
            top   = ", ".join(
                f"{row['tech_area']} ({row['contribute']})"
                for _, row in sub_r.head(3).iterrows()
                if row.get("contribute") is not None
            )
            researcher_rows.append(f"{researcher} | {top}")
        sections.append("**Researcher priorities:**\n" + "\n".join(researcher_rows) + "\n")

    if vision_data:
        excerpts = []
        for entry in vision_data[:8]:
            name = entry.get("researcher", "")
            defn = (entry.get("definition") or "")[:120]
            expl = (entry.get("explore") or "")[:80]
            if defn or expl:
                excerpts.append(f"- **{name}**: {defn}" + (f" | Wants to explore: {expl}" if expl else ""))
        if excerpts:
            sections.append("**Deep tech vision excerpts:**\n" + "\n".join(excerpts))

    return "\n\n".join(sections)


# ─────────────────────────────────────────────────────────────────────────────
# System prompts
# ─────────────────────────────────────────────────────────────────────────────

_SHARED_CONTEXT = (
    "You are a research strategy advisor for the SINTEF Trustworthy Green IoT Software Research Group "
    "in Norway. The group works on trustworthy AI, IoT/edge computing, cybersecurity, privacy engineering, "
    "and sustainable computing. They want to identify project ideas suitable for EU Horizon Europe or "
    "Norwegian Research Council (NRC) funding. Be specific, actionable, and ground your advice in the "
    "radar data provided. Format with Markdown headers and bullet points."
)

_SYNERGY_PROMPT = _SHARED_CONTEXT + """

Your task: TEAM SYNERGY ANALYSIS.
Identify 3–5 high-potential collaboration pairs or small teams based on:
- Complementary skills (one person's 'Lead' covers another's 'Watch')
- Shared 'Contribute' intent in adjacent areas
- Vision text alignment

For each pairing, output:
### [Name A] + [Name B] (+ optional others)
**Shared strength:** …
**Complementary gap:** …
**Suggested collaboration topic:** …
"""

_IDEAS_PROMPT = _SHARED_CONTEXT + """

Your task: PROJECT IDEA GENERATION.
Generate 5 innovative but fundable research project ideas that leverage the group's collective strengths.
Prioritise areas where the group has high Contribute intent and reasonable Expertise.
Consider EU Horizon Europe (especially Trustworthy AI, Green Deal, Cybersecurity) and NRC programs.

For each idea:
### Idea N: [Title]
**One-line pitch:** …
**Deep tech areas involved:** …
**Group fit:** (which researchers are well-placed to contribute)
**Potential funding stream:** …
**Why now:** (why this is timely in 2025–2026)
"""

_PROPOSAL_PROMPT = _SHARED_CONTEXT + """

Your task: PROJECT PROPOSAL DRAFT.
Write a structured 1-page research project proposal for the idea the user specifies.
Use the researcher profiles from the radar data to populate the team section.

Structure:
## [Title]
**Executive Summary** (100–120 words)
**Problem Statement**
**Research Objectives** (3–4 bullet points)
**Methodology** (brief, 3–4 sentences)
**Expected Outcomes & Impact**
**Team** (name relevant researchers from the radar data with their roles)
**Indicative Timeline** (3 phases over 36 months)
**Target Funding Programme**
"""

_GAP_PROMPT = _SHARED_CONTEXT + """

Your task: SKILL GAP ANALYSIS.
Analyse the radar data and identify:

### 1. Interest–Expertise Gaps
Areas where group Interest significantly exceeds Expertise (training/hiring opportunity).

### 2. Coverage Gaps
Deep tech areas relevant to the group's projects that are in the 'Watch' ring across the board.

### 3. Single-Point-of-Failure Risks
Areas where only one or two people have 'Lead' or 'Contribute' status — bus-factor risk.

### 4. Recommended Actions
Concrete steps: who should upskill in what, what external expertise to seek, what to deprioritise.
"""

_ROADMAP_PROMPT = _SHARED_CONTEXT + """

Your task: 12-MONTH COMPETENCE ROADMAP.
Design a realistic roadmap for moving the group's collective skills up one ring in priority areas.

Structure:
### Priority Areas (next 6 months)
Which 2–3 areas to focus on first, and why.

### Month 1–3: Foundation
Specific actions (workshops, reading groups, small experiments).

### Month 4–6: Applied Practice
Projects, collaborations, or tools to deepen skills.

### Month 7–12: Consolidation & Leadership
How to reach 'Contribute' or 'Lead' ring in target areas.

### Success Metrics
How to know the roadmap is working.
"""

_AGENT_PROMPTS: dict[str, str] = {
    "Team Synergies":     _SYNERGY_PROMPT,
    "Project Ideas":      _IDEAS_PROMPT,
    "Project Proposal":   _PROPOSAL_PROMPT,
    "Skill Gap Analysis": _GAP_PROMPT,
    "Competence Roadmap": _ROADMAP_PROMPT,
}


# ─────────────────────────────────────────────────────────────────────────────
# Generic streaming runner
# ─────────────────────────────────────────────────────────────────────────────

def _stream(system_prompt: str, user_message: str) -> Iterator[str]:
    """Stream a response from OpenAI. Yields accumulated text each chunk."""
    client      = _get_client()
    accumulated = ""
    try:
        stream = client.chat.completions.create(
            model=OPENAI_MODEL,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user",   "content": user_message},
            ],
            stream=True,
            timeout=OPENAI_TIMEOUT,
        )
        for chunk in stream:
            delta = (chunk.choices[0].delta.content or "") if chunk.choices else ""
            accumulated += delta
            yield accumulated
    except Exception as exc:
        log.error("OpenAI stream error: %s", exc)
        yield accumulated + f"\n\n⚠ **Error:** {exc}"


# ─────────────────────────────────────────────────────────────────────────────
# Public agent functions
# ─────────────────────────────────────────────────────────────────────────────

def stream_synergies(df: pd.DataFrame, vision_data: list[dict], extra: str = "") -> Iterator[str]:
    context  = _build_radar_context(df, vision_data)
    user_msg = f"Radar data:\n{context}"
    if extra:
        user_msg += f"\n\nAdditional context: {extra}"
    yield from _stream(_SYNERGY_PROMPT, user_msg)


def stream_project_ideas(df: pd.DataFrame, vision_data: list[dict], extra: str = "") -> Iterator[str]:
    context  = _build_radar_context(df, vision_data)
    user_msg = f"Radar data:\n{context}"
    if extra:
        user_msg += f"\n\nFocus on: {extra}"
    yield from _stream(_IDEAS_PROMPT, user_msg)


def stream_proposal(df: pd.DataFrame, vision_data: list[dict], extra: str = "") -> Iterator[str]:
    context  = _build_radar_context(df, vision_data)
    idea     = extra.strip() or "the most promising project idea based on the radar data"
    user_msg = f"Write a proposal for: **{idea}**\n\nTeam radar data:\n{context}"
    yield from _stream(_PROPOSAL_PROMPT, user_msg)


def stream_gap_analysis(df: pd.DataFrame, vision_data: list[dict], extra: str = "") -> Iterator[str]:
    context  = _build_radar_context(df, vision_data)
    user_msg = f"Radar data:\n{context}"
    if extra:
        user_msg += f"\n\nFocus on: {extra}"
    yield from _stream(_GAP_PROMPT, user_msg)


def stream_roadmap(df: pd.DataFrame, vision_data: list[dict], extra: str = "") -> Iterator[str]:
    context  = _build_radar_context(df, vision_data)
    user_msg = f"Radar data:\n{context}"
    if extra:
        user_msg += f"\n\nConstraints: {extra}"
    yield from _stream(_ROADMAP_PROMPT, user_msg)


AGENT_FUNCTIONS: dict[str, object] = {
    "Team Synergies":     stream_synergies,
    "Project Ideas":      stream_project_ideas,
    "Project Proposal":   stream_proposal,
    "Skill Gap Analysis": stream_gap_analysis,
    "Competence Roadmap": stream_roadmap,
}
