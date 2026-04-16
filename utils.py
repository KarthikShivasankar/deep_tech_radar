"""
utils.py — Record assembly, validation, and schema helpers.

All functions are pure (no I/O). They transform data between the Gradio
UI representation (flat slider values, list of area names) and the flat
column format used by the HuggingFace dataset.
"""

import json
import logging
from datetime import datetime, timezone
from typing import Any, Optional

from config import DEFAULT_TECH_AREAS, DIMENSIONS, SLIDER_MIN, SLIDER_MAX, MAX_AREAS

log = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# Timestamps
# ─────────────────────────────────────────────────────────────────────────────

def now_iso() -> str:
    """Return current UTC datetime as ISO 8601 string."""
    return datetime.now(timezone.utc).isoformat()


# ─────────────────────────────────────────────────────────────────────────────
# Column flattening / parsing
# ─────────────────────────────────────────────────────────────────────────────

def flatten_ratings_to_columns(
    ratings: dict[str, dict[str, Any]],
    tech_areas: list[str],
) -> dict[str, Any]:
    """
    Convert nested ratings dict to a flat column dict with index-based keys.

    Input:  {"AI/ML & Trustworthy AI": {"interest": 4, "expertise": 3, ...}, ...}
    Output: {"area_0_name": "AI/ML & Trustworthy AI",
             "area_0_interest": 4, "area_0_expertise": 3, ...,
             "area_1_name": ..., ...}

    Pads unused slots up to MAX_AREAS with None values so the schema
    stays stable regardless of how many areas each person rates.
    """
    result: dict[str, Any] = {}
    for i in range(MAX_AREAS):
        prefix = f"area_{i}"
        if i < len(tech_areas):
            area = tech_areas[i]
            area_ratings = ratings.get(area) or {}
            result[f"{prefix}_name"] = area
            for dim in DIMENSIONS:
                val = area_ratings.get(dim)
                result[f"{prefix}_{dim}"] = int(val) if val is not None else None
        else:
            result[f"{prefix}_name"] = None
            for dim in DIMENSIONS:
                result[f"{prefix}_{dim}"] = None
    return result


def parse_ratings_from_record(
    record: dict,
    tech_areas: list[str],
) -> dict[str, dict[str, int]]:
    """
    Inverse of flatten_ratings_to_columns.

    Reconstructs {area_name: {dim_key: value}} from a flat HF dataset row.
    Used when restoring a saved session to pre-populate UI sliders.

    Missing or null values default to SLIDER_DEFAULT (3).
    """
    from config import SLIDER_DEFAULT
    ratings: dict = {}
    for i, area in enumerate(tech_areas):
        if i >= MAX_AREAS or not area:
            break
        dim_values: dict[str, int] = {}
        for dim in DIMENSIONS:
            raw = record.get(f"area_{i}_{dim}")
            try:
                dim_values[dim] = int(raw) if raw is not None else SLIDER_DEFAULT
            except (TypeError, ValueError):
                dim_values[dim] = SLIDER_DEFAULT
        ratings[area] = dim_values
    return ratings


def build_active_tech_areas(areas_list: list[Optional[str]]) -> list[str]:
    """
    Return only the non-None entries from the full 15-slot areas list.

    Used everywhere we need the ordered list of areas actually rated.
    """
    return [a for a in areas_list if a]


# ─────────────────────────────────────────────────────────────────────────────
# Record assembly
# ─────────────────────────────────────────────────────────────────────────────

def build_submission_record(
    name: str,
    scholar_tags: list[str],
    scholar_source: str,
    ratings: dict[str, dict[str, Any]],
    tech_areas: list[str],
    vision_definition: str,
    vision_examples: str,
    vision_explore: str,
    submitted_at: Optional[str] = None,
) -> dict:
    """
    Assemble a complete, flat record ready for HF dataset storage.

    Args:
        name:              Researcher full name (stripped).
        scholar_tags:      Tags from Scholar API or manual input.
        scholar_source:    "google_scholar" | "semantic_scholar" | "manual".
        ratings:           {tech_area: {dim_key: slider_value (1-5)}}.
        tech_areas:        Ordered list of active area names for this submission.
        vision_*:          Free-text answers from the Deep Tech Vision tab.
        submitted_at:      Override ISO timestamp (leave None for current time).

    Returns:
        Flat dict matching the HF dataset schema.
    """
    flat = flatten_ratings_to_columns(ratings, tech_areas)
    return {
        "researcher_name": name.strip(),
        "submitted_at":    submitted_at or now_iso(),
        "scholar_tags":    scholar_tags or [],
        "scholar_source":  scholar_source or "manual",
        "tech_areas_used": tech_areas,
        **flat,
        "vision_definition": (vision_definition or "").strip(),
        "vision_examples":   (vision_examples or "").strip(),
        "vision_explore":    (vision_explore or "").strip(),
    }


# ─────────────────────────────────────────────────────────────────────────────
# Validation
# ─────────────────────────────────────────────────────────────────────────────

def validate_record(record: dict) -> tuple[bool, list[str]]:
    """
    Validate a submission record before it is saved.

    Returns:
        (is_valid, errors_list)  — errors_list is empty when is_valid is True.
    """
    errors: list[str] = []

    if not str(record.get("researcher_name", "")).strip():
        errors.append("Researcher name is required.")

    areas = record.get("tech_areas_used") or []
    if not areas:
        errors.append("At least one tech area must be rated.")

    for i, area in enumerate(areas):
        if i >= MAX_AREAS:
            break
        for dim in DIMENSIONS:
            val = record.get(f"area_{i}_{dim}")
            if val is not None:
                try:
                    iv = int(val)
                    if not (SLIDER_MIN <= iv <= SLIDER_MAX):
                        errors.append(
                            f"'{area}' {dim} value {val} is outside [{SLIDER_MIN}, {SLIDER_MAX}]."
                        )
                except (TypeError, ValueError):
                    errors.append(f"'{area}' {dim} has non-integer value: {val!r}.")

    return (len(errors) == 0, errors)


# ─────────────────────────────────────────────────────────────────────────────
# Serialisation helpers
# ─────────────────────────────────────────────────────────────────────────────

def record_to_json_preview(record: dict) -> str:
    """
    Serialise a submission record to pretty-printed JSON for the preview pane.

    Handles numpy/pandas types gracefully via a custom default encoder.
    """
    def _default(obj: Any) -> Any:
        if hasattr(obj, "tolist"):       # numpy arrays
            return obj.tolist()
        if hasattr(obj, "item"):         # numpy scalars
            return obj.item()
        return str(obj)

    return json.dumps(record, indent=2, ensure_ascii=False, default=_default)


def ratings_from_sliders(
    areas_list: list[Optional[str]],
    interest_vals: list[int],
    expertise_vals: list[int],
    contribute_vals: list[int],
) -> dict[str, dict[str, int]]:
    """
    Convert parallel slider-value lists into the nested ratings dict.

    This is called both for the live preview and for generating the
    submission record from the slider state.

    Args:
        areas_list:      15-slot list of area names (None = unused custom slot).
        interest_vals:   15 interest slider values.
        expertise_vals:  15 expertise slider values.
        contribute_vals: 15 contribute slider values.

    Returns:
        {area_name: {"interest": v, "expertise": v, "contribute": v}}
        Only non-None areas are included.
    """
    ratings: dict = {}
    dim_lists = {
        "interest":   interest_vals,
        "expertise":  expertise_vals,
        "contribute": contribute_vals,
    }
    for i, area in enumerate(areas_list):
        if not area or i >= len(interest_vals):
            continue
        ratings[area] = {dim: (dim_lists[dim][i] if i < len(dim_lists[dim]) else 3)
                         for dim in DIMENSIONS}
    return ratings
