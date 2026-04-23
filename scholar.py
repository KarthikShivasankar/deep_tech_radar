"""
scholar.py — Research profile lookup via Semantic Scholar and Google Scholar.

Lookup priority:
  1. Google Scholar — if a profile URL is supplied
  2. Semantic Scholar — by author name (rich: h-index, citations, recency signals)
  3. Manual fallback — empty tags, user fills via text description

New functions:
  build_scholar_url_mapping() — auto-detects URL→name mapping at startup
  auto_rate_from_tags()       — tags → {area: {dim: 1-5}} ratings
  auto_rate_from_rich_data()  — richer inference using recency signals
  infer_ratings_from_text()   — free-text → ratings via keyword matching
"""

import logging
import time
from functools import lru_cache
from typing import Optional

import requests

from config import (
    SS_API_KEY, SS_MAX_PAPERS, SS_GENERIC_TAGS,
    DEFAULT_TECH_AREAS, SCHOLAR_TAG_TO_AREA, SCHOLAR_URLS_RAW,
    TEAM_MEMBERS, SLIDER_DEFAULT, CURRENT_YEAR, RECENT_YEARS,
)

log = logging.getLogger(__name__)

_SS_BASE    = "https://api.semanticscholar.org/graph/v1"
_SS_HEADERS = {"x-api-key": SS_API_KEY} if SS_API_KEY else {}


# ─────────────────────────────────────────────────────────────────────────────
# Internal Semantic Scholar helpers
# ─────────────────────────────────────────────────────────────────────────────

def _ss_get(path: str, params: dict, retries: int = 1) -> Optional[dict]:
    url = f"{_SS_BASE}{path}"
    for attempt in range(retries + 1):
        try:
            resp = requests.get(url, params=params, headers=_SS_HEADERS, timeout=12)
            if resp.status_code == 429:
                wait = int(resp.headers.get("Retry-After", 8))
                log.warning("Semantic Scholar rate limit — waiting %ds", wait)
                time.sleep(wait)
                continue
            resp.raise_for_status()
            return resp.json()
        except requests.exceptions.Timeout:
            log.warning("Semantic Scholar timeout on %s (attempt %d)", path, attempt + 1)
        except requests.exceptions.RequestException as exc:
            log.warning("Semantic Scholar request error: %s", exc)
            break
    return None


@lru_cache(maxsize=128)
def _search_author_id(name: str) -> Optional[str]:
    data = _ss_get("/author/search", {"query": name, "fields": "authorId,name", "limit": 5})
    if not data or not data.get("data"):
        return None
    return data["data"][0].get("authorId")


@lru_cache(maxsize=128)
def _get_author_details(author_id: str) -> dict:
    """Fetch h-index, citation count, paper count for an author."""
    data = _ss_get(
        f"/author/{author_id}",
        {"fields": "name,hIndex,citationCount,paperCount,affiliations"},
    )
    return data or {}


@lru_cache(maxsize=128)
def _get_papers_rich(author_id: str) -> list[dict]:
    """Fetch papers with title, abstract, year, topics, citations."""
    time.sleep(0.4)
    data = _ss_get(
        f"/author/{author_id}/papers",
        {
            "fields": "title,abstract,year,fieldsOfStudy,topics,citationCount",
            "limit": SS_MAX_PAPERS,
        },
    )
    return (data or {}).get("data", [])


def _extract_tags(papers: list[dict], top_n: int = 20) -> list[str]:
    from collections import Counter
    counts: Counter = Counter()
    for paper in papers:
        for field in paper.get("fieldsOfStudy") or []:
            if isinstance(field, dict):
                field = field.get("category", "")
            if field and field not in SS_GENERIC_TAGS:
                counts[str(field)] += 1
        for topic in paper.get("topics") or []:
            label = topic.get("topic", "") if isinstance(topic, dict) else str(topic)
            if label and label not in SS_GENERIC_TAGS:
                counts[label] += 1
    return [tag for tag, _ in counts.most_common(top_n)]


def _extract_recent_tags(papers: list[dict], top_n: int = 20) -> list[str]:
    """Tags from papers published in the last RECENT_YEARS years."""
    cutoff = CURRENT_YEAR - RECENT_YEARS
    recent = [p for p in papers if (p.get("year") or 0) >= cutoff]
    return _extract_tags(recent, top_n)


def _count_area_hits(papers: list[dict]) -> dict[str, int]:
    """Count how many papers map to each DEFAULT_TECH_AREA via keyword matching."""
    hits: dict[str, int] = {a: 0 for a in DEFAULT_TECH_AREAS}
    for paper in papers:
        matched_areas: set[str] = set()
        fields = []
        for f in paper.get("fieldsOfStudy") or []:
            fields.append(f.get("category", "") if isinstance(f, dict) else str(f))
        for t in paper.get("topics") or []:
            fields.append(t.get("topic", "") if isinstance(t, dict) else str(t))
        title = (paper.get("title") or "").lower()
        abstract = (paper.get("abstract") or "").lower()
        text = title + " " + abstract

        for keyword, area in SCHOLAR_TAG_TO_AREA.items():
            if area in matched_areas:
                continue
            for f in fields:
                if keyword in f.lower():
                    matched_areas.add(area)
                    break
            if area not in matched_areas and keyword in text:
                matched_areas.add(area)

        for area in matched_areas:
            hits[area] += 1
    return hits


# ─────────────────────────────────────────────────────────────────────────────
# Rating inference
# ─────────────────────────────────────────────────────────────────────────────

def _papers_to_expertise(n: int) -> int:
    if n == 0:   return SLIDER_DEFAULT
    if n <= 1:   return 2
    if n <= 4:   return 3
    if n <= 12:  return 4
    return 5


def auto_rate_from_tags(tags: list[str]) -> dict[str, dict[str, int]]:
    """
    Map Scholar tags to DEFAULT_TECH_AREAS and produce 1-5 ratings.
    All three dimensions derived from tag frequency per area.
    Areas with no matching tags get SLIDER_DEFAULT (3) for all dims.
    """
    area_hits: dict[str, int] = {a: 0 for a in DEFAULT_TECH_AREAS}
    for tag in tags:
        tag_lower = tag.lower().strip()
        for keyword, area in SCHOLAR_TAG_TO_AREA.items():
            if keyword in tag_lower:
                area_hits[area] += 1
                break

    ratings: dict[str, dict[str, int]] = {}
    for area in DEFAULT_TECH_AREAS:
        n = area_hits[area]
        if n == 0:
            ratings[area] = {d: SLIDER_DEFAULT for d in ("interest", "expertise", "contribute")}
        elif n == 1:
            ratings[area] = {"interest": 2, "expertise": 2, "contribute": 1}
        elif n <= 4:
            ratings[area] = {"interest": 3, "expertise": 3, "contribute": 2}
        elif n <= 12:
            ratings[area] = {"interest": 4, "expertise": 4, "contribute": 3}
        else:
            ratings[area] = {"interest": 5, "expertise": 5, "contribute": 4}
    return ratings


def auto_rate_from_rich_data(rich: dict) -> dict[str, dict[str, int]]:
    """
    Infer ratings using career-total AND recent paper counts.
    - expertise  ← total paper count in area (career depth)
    - interest   ← recent paper count last 3 years (active pursuit)
    - contribute ← max(expertise, interest) - 1, min 1
    """
    total   = rich.get("paper_area_counts", {})
    recent  = rich.get("recent_area_counts", {})

    ratings: dict[str, dict[str, int]] = {}
    for area in DEFAULT_TECH_AREAS:
        exp  = _papers_to_expertise(total.get(area, 0))
        intr = _papers_to_expertise(recent.get(area, 0))
        if total.get(area, 0) == 0 and recent.get(area, 0) == 0:
            ratings[area] = {d: SLIDER_DEFAULT for d in ("interest", "expertise", "contribute")}
        else:
            cont = max(1, min(5, max(exp, intr) - 1))
            ratings[area] = {"interest": intr, "expertise": exp, "contribute": cont}
    return ratings


def infer_ratings_from_text(text: str) -> dict[str, dict[str, int]]:
    """
    Infer ratings from free-text research description via keyword matching.
    Interest is set 1pt higher than expertise (text implies aspiration).
    """
    text_lower = text.lower()
    area_hits: dict[str, int] = {a: 0 for a in DEFAULT_TECH_AREAS}
    for keyword, area in SCHOLAR_TAG_TO_AREA.items():
        if keyword in text_lower:
            area_hits[area] += 1

    ratings: dict[str, dict[str, int]] = {}
    for area in DEFAULT_TECH_AREAS:
        n = area_hits[area]
        if n == 0:
            ratings[area] = {d: SLIDER_DEFAULT for d in ("interest", "expertise", "contribute")}
        elif n == 1:
            ratings[area] = {"interest": 3, "expertise": 2, "contribute": 2}
        elif n <= 3:
            ratings[area] = {"interest": 4, "expertise": 3, "contribute": 3}
        else:
            ratings[area] = {"interest": 5, "expertise": 4, "contribute": 4}
    return ratings


# ─────────────────────────────────────────────────────────────────────────────
# Public: Rich Semantic Scholar lookup
# ─────────────────────────────────────────────────────────────────────────────

def fetch_rich_scholar_data(author_id: str) -> dict:
    """
    Fetch enriched profile: h-index, citations, recency signals, per-area counts.

    Returns:
        {
            tags, recent_tags, paper_count, h_index, citation_count,
            paper_area_counts, recent_area_counts, top_papers, error
        }
    """
    papers  = _get_papers_rich(author_id)
    details = _get_author_details(author_id)

    cutoff  = CURRENT_YEAR - RECENT_YEARS
    recent_papers = [p for p in papers if (p.get("year") or 0) >= cutoff]

    all_hits    = _count_area_hits(papers)
    recent_hits = _count_area_hits(recent_papers)

    # Top 5 by citation count
    top_papers = sorted(
        [p for p in papers if p.get("citationCount")],
        key=lambda p: p.get("citationCount", 0),
        reverse=True,
    )[:5]
    top_papers_clean = [
        {"title": p.get("title", ""), "year": p.get("year", ""), "citations": p.get("citationCount", 0)}
        for p in top_papers
    ]

    return {
        "tags":               _extract_tags(papers, top_n=20),
        "recent_tags":        _extract_tags(recent_papers, top_n=10),
        "paper_count":        len(papers),
        "h_index":            details.get("hIndex", 0) or 0,
        "citation_count":     details.get("citationCount", 0) or 0,
        "paper_area_counts":  all_hits,
        "recent_area_counts": recent_hits,
        "top_papers":         top_papers_clean,
        "error":              None,
    }


def fetch_from_semantic_scholar(name: str) -> dict:
    """Full Semantic Scholar lookup for an author by name (rich version)."""
    author_id = _search_author_id(name)
    if not author_id:
        return {
            "tags": [], "paper_count": 0, "author_id": None,
            "h_index": 0, "citation_count": 0,
            "error": f"'{name}' not found on Semantic Scholar",
        }
    rich = fetch_rich_scholar_data(author_id)
    rich["author_id"] = author_id
    return rich


# ─────────────────────────────────────────────────────────────────────────────
# Public: Google Scholar
# ─────────────────────────────────────────────────────────────────────────────

def _parse_gs_user_id(url: str) -> Optional[str]:
    if "user=" not in url:
        return None
    return url.split("user=")[-1].split("&")[0].strip()


def fetch_from_google_scholar(profile_url: str) -> dict:
    """Fetch research interests from a Google Scholar profile URL."""
    try:
        from scholarly import scholarly as _scholarly
    except ImportError:
        return {"tags": [], "citations": 0, "name": "", "error": "scholarly not installed"}

    user_id = _parse_gs_user_id(profile_url)
    if not user_id:
        return {"tags": [], "citations": 0, "name": "", "error": "Could not parse user ID"}

    try:
        author = _scholarly.search_author_id(user_id)
        author = _scholarly.fill(author, sections=["basics", "indices", "interests"])
        interests = author.get("interests") or []
        tags      = [str(i).strip() for i in interests if i]
        return {
            "tags":      tags,
            "citations": int(author.get("citedby") or 0),
            "name":      author.get("name", ""),
            "error":     None,
        }
    except Exception as exc:
        log.warning("Google Scholar fetch failed for %s: %s", profile_url, exc)
        return {"tags": [], "citations": 0, "name": "", "error": str(exc)}


# ─────────────────────────────────────────────────────────────────────────────
# Startup: auto-detect URL → name mapping
# ─────────────────────────────────────────────────────────────────────────────

@lru_cache(maxsize=1)
def build_scholar_url_mapping() -> dict[str, str]:
    """
    Fetch each Google Scholar profile in SCHOLAR_URLS_RAW, read the researcher
    name, fuzzy-match to TEAM_MEMBERS, and return the complete mapping.

    Cached — runs only once per process.
    Returns: {team_member_name: google_scholar_url}  (empty str for unmatched)
    """
    import difflib
    from urllib.parse import urlparse, parse_qs

    try:
        from scholarly import scholarly as _scholarly
    except ImportError:
        log.warning("scholarly not installed — Google Scholar URL mapping skipped")
        return {name: "" for name in TEAM_MEMBERS}

    mapping = {name: "" for name in TEAM_MEMBERS}
    for url in SCHOLAR_URLS_RAW:
        try:
            params  = parse_qs(urlparse(url).query)
            user_id = params.get("user", [None])[0]
            if not user_id:
                continue
            author       = _scholarly.search_author_id(user_id)
            fetched_name = author.get("name", "")
            matches      = difflib.get_close_matches(fetched_name, TEAM_MEMBERS, n=1, cutoff=0.5)
            if matches:
                mapping[matches[0]] = url
                log.info("Scholar URL mapped: %s → %s", matches[0], url)
            else:
                log.warning("Scholar name '%s' did not match any team member", fetched_name)
            time.sleep(1.0)   # be polite to Google Scholar
        except Exception as exc:
            log.warning("Failed to fetch Scholar profile %s: %s", url, exc)
    return mapping


# ─────────────────────────────────────────────────────────────────────────────
# Public: Unified entry point
# ─────────────────────────────────────────────────────────────────────────────

def lookup_researcher(
    name: str,
    google_scholar_url: Optional[str] = None,
) -> dict:
    """
    Unified researcher lookup with automatic fallback chain.

    Priority:
      1. Google Scholar (if URL provided) — interests tags
      2. Semantic Scholar (rich data) — h-index, recency, per-area counts
      3. Returns empty tags for manual text entry

    Returns:
        {
            tags, source, paper_count, h_index, citation_count,
            rich_data (full rich dict or None), extra, error
        }
    """
    # ── Try Google Scholar first ──────────────────────────────────────────
    if google_scholar_url and google_scholar_url.strip():
        gs = fetch_from_google_scholar(google_scholar_url.strip())
        if gs["tags"]:
            return {
                "tags":        gs["tags"],
                "source":      "google_scholar",
                "paper_count": 0,
                "h_index":     0,
                "citation_count": gs.get("citations", 0),
                "rich_data":   None,
                "extra":       {"citations": gs.get("citations", 0)},
                "error":       gs.get("error"),
            }
        log.info("Google Scholar returned no tags for %s, falling back to Semantic Scholar", name)

    # ── Try Semantic Scholar (rich) ───────────────────────────────────────
    ss = fetch_from_semantic_scholar(name)
    if ss.get("tags"):
        return {
            "tags":           ss["tags"],
            "source":         "semantic_scholar",
            "paper_count":    ss.get("paper_count", 0),
            "h_index":        ss.get("h_index", 0),
            "citation_count": ss.get("citation_count", 0),
            "rich_data":      ss,
            "extra":          {"author_id": ss.get("author_id")},
            "error":          ss.get("error"),
        }

    # ── Manual fallback ───────────────────────────────────────────────────
    return {
        "tags":           [],
        "source":         "manual",
        "paper_count":    0,
        "h_index":        0,
        "citation_count": 0,
        "rich_data":      None,
        "extra":          {},
        "error":          ss.get("error") or "No tags found — please describe your research below",
    }
