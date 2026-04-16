"""
scholar.py — Research profile lookup via Semantic Scholar and Google Scholar.

Lookup priority:
  1. Google Scholar — if a profile URL is supplied
  2. Semantic Scholar — by author name (free REST API, no key needed for basic use)
  3. Manual fallback — empty tags, user fills in by hand

All external calls are wrapped in try/except and fail gracefully.
Results are cached in-process with functools.lru_cache to avoid redundant
API calls within a session.
"""

import logging
import time
from functools import lru_cache
from typing import Optional

import requests

from config import SS_API_KEY, SS_MAX_PAPERS, SS_GENERIC_TAGS

log = logging.getLogger(__name__)

_SS_BASE    = "https://api.semanticscholar.org/graph/v1"
_SS_HEADERS = {"x-api-key": SS_API_KEY} if SS_API_KEY else {}


# ─────────────────────────────────────────────────────────────────────────────
# Internal Semantic Scholar helpers
# ─────────────────────────────────────────────────────────────────────────────

def _ss_get(path: str, params: dict, retries: int = 1) -> Optional[dict]:
    """
    GET a Semantic Scholar endpoint with timeout and one rate-limit retry.

    Returns the parsed JSON dict, or None on any failure.
    """
    url = f"{_SS_BASE}{path}"
    for attempt in range(retries + 1):
        try:
            resp = requests.get(url, params=params, headers=_SS_HEADERS, timeout=12)
            if resp.status_code == 429:
                wait = int(resp.headers.get("Retry-After", 8))
                log.warning("Semantic Scholar rate limit — waiting %ds (attempt %d)", wait, attempt + 1)
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
    """Return the Semantic Scholar authorId for the best name match."""
    data = _ss_get(
        "/author/search",
        {"query": name, "fields": "authorId,name", "limit": 5},
    )
    if not data or not data.get("data"):
        return None
    # Use the first result (Semantic Scholar ranks by relevance)
    return data["data"][0].get("authorId")


@lru_cache(maxsize=128)
def _get_papers(author_id: str) -> list[dict]:
    """Fetch papers for a Semantic Scholar authorId."""
    time.sleep(0.4)   # polite pause between chained calls
    data = _ss_get(
        f"/author/{author_id}/papers",
        {
            "fields": "title,year,fieldsOfStudy,publicationTypes,externalIds",
            "limit": SS_MAX_PAPERS,
        },
    )
    return (data or {}).get("data", [])


def _extract_tags(papers: list[dict], top_n: int = 15) -> list[str]:
    """
    Flatten and rank fieldsOfStudy tags across all papers.

    Filters out generic tags defined in config.SS_GENERIC_TAGS.
    Returns the top_n tags by frequency.
    """
    from collections import Counter
    counts: Counter = Counter()
    for paper in papers:
        for field in paper.get("fieldsOfStudy") or []:
            # fieldsOfStudy can be a string or {"category": "...", "source": "..."}
            if isinstance(field, dict):
                field = field.get("category", "")
            if field and field not in SS_GENERIC_TAGS:
                counts[str(field)] += 1
    return [tag for tag, _ in counts.most_common(top_n)]


# ─────────────────────────────────────────────────────────────────────────────
# Public: Semantic Scholar
# ─────────────────────────────────────────────────────────────────────────────

def fetch_from_semantic_scholar(name: str) -> dict:
    """
    Full Semantic Scholar lookup for an author by name.

    Returns:
        {
            "tags":        list[str],   # research area tags
            "paper_count": int,
            "author_id":   str | None,
            "error":       str | None,
        }
    """
    author_id = _search_author_id(name)
    if not author_id:
        return {
            "tags": [],
            "paper_count": 0,
            "author_id": None,
            "error": f"'{name}' not found on Semantic Scholar",
        }
    papers = _get_papers(author_id)
    tags   = _extract_tags(papers)
    return {
        "tags":        tags,
        "paper_count": len(papers),
        "author_id":   author_id,
        "error":       None if tags else "No field tags found in papers",
    }


# ─────────────────────────────────────────────────────────────────────────────
# Public: Google Scholar (scholarly)
# ─────────────────────────────────────────────────────────────────────────────

def _parse_gs_user_id(url: str) -> Optional[str]:
    """Extract the user= parameter from a Google Scholar profile URL."""
    if "user=" not in url:
        return None
    return url.split("user=")[-1].split("&")[0].strip()


def fetch_from_google_scholar(profile_url: str) -> dict:
    """
    Fetch research interests from a Google Scholar profile URL.

    Uses the `scholarly` library. Google Scholar aggressively blocks
    scrapers, so this always falls back gracefully on failure.

    Returns:
        {
            "tags":      list[str],
            "citations": int,
            "error":     str | None,
        }
    """
    try:
        from scholarly import scholarly as _scholarly   # type: ignore[import]
    except ImportError:
        return {"tags": [], "citations": 0, "error": "scholarly library not installed"}

    user_id = _parse_gs_user_id(profile_url)
    if not user_id:
        return {"tags": [], "citations": 0, "error": "Could not parse user ID from Google Scholar URL"}

    try:
        author = _scholarly.search_author_id(user_id)
        author = _scholarly.fill(author, sections=["basics", "indices", "interests"])
        interests = author.get("interests") or []
        tags      = [str(i).strip() for i in interests if i]
        return {
            "tags":      tags,
            "citations": int(author.get("citedby") or 0),
            "error":     None,
        }
    except Exception as exc:
        log.warning("Google Scholar fetch failed for %s: %s", profile_url, exc)
        return {"tags": [], "citations": 0, "error": str(exc)}


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
      1. Google Scholar — if profile URL is non-empty and valid
      2. Semantic Scholar — by author name
      3. Returns empty tags for manual entry

    Returns:
        {
            "tags":        list[str],
            "source":      "google_scholar" | "semantic_scholar" | "manual",
            "paper_count": int,
            "extra":       dict,          # source-specific metadata
            "error":       str | None,    # human-readable note (non-fatal)
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
                "extra":       {"citations": gs.get("citations", 0)},
                "error":       gs.get("error"),
            }
        log.info("Google Scholar returned no tags, falling back to Semantic Scholar")

    # ── Try Semantic Scholar ──────────────────────────────────────────────
    ss = fetch_from_semantic_scholar(name)
    if ss["tags"]:
        return {
            "tags":        ss["tags"],
            "source":      "semantic_scholar",
            "paper_count": ss["paper_count"],
            "extra":       {"author_id": ss.get("author_id")},
            "error":       ss.get("error"),
        }

    # ── Manual fallback ───────────────────────────────────────────────────
    return {
        "tags":        [],
        "source":      "manual",
        "paper_count": 0,
        "extra":       {},
        "error":       ss.get("error") or "No tags found — please add manually",
    }
