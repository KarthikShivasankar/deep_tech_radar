"""
storage.py — HuggingFace dataset read/write with session persistence.

Dataset layout:
  - One row per researcher (researcher_name is the upsert key).
  - Flat Parquet format: index-based columns (area_0_name, area_0_interest, …)
    allow custom areas without schema conflicts.
  - List columns (scholar_tags, tech_areas_used) are JSON-serialised to strings
    before push and deserialised on load — the safest Parquet-compatible approach.

Session retrieval:
  - load_researcher_session(name) reads the most recent row for that name.
  - The app uses this to pre-populate all form fields when a user selects
    their name from the dropdown — avoiding re-entry of previous answers.
"""

import json
import logging
import tempfile
from typing import Optional

import pandas as pd

from config import (
    HF_TOKEN, HF_USERNAME, HF_DATASET_NAME,
    DIMENSIONS, MAX_AREAS, DEFAULT_TECH_AREAS, SLIDER_DEFAULT,
)

log = logging.getLogger(__name__)

# List-type columns that are serialised as JSON strings in Parquet
_LIST_COLS = ("scholar_tags", "tech_areas_used")


# ─────────────────────────────────────────────────────────────────────────────
# Repo ID
# ─────────────────────────────────────────────────────────────────────────────

def get_repo_id() -> str:
    """Full HuggingFace dataset repo id, e.g. 'myorg/deep-tech-radar'."""
    if HF_USERNAME:
        return f"{HF_USERNAME}/{HF_DATASET_NAME}"
    return HF_DATASET_NAME


# ─────────────────────────────────────────────────────────────────────────────
# Authentication
# ─────────────────────────────────────────────────────────────────────────────

def authenticate_hf() -> bool:
    """
    Login to HuggingFace Hub using HF_TOKEN from environment.

    Returns True on success, False if the token is missing or invalid.
    Logs a warning but never raises — the app runs in read-only mode without a token.
    """
    if not HF_TOKEN:
        log.warning("HF_TOKEN not configured — dataset save/load disabled.")
        return False
    try:
        from huggingface_hub import login
        login(token=HF_TOKEN, add_to_git_credential=False)
        log.info("HuggingFace authentication successful (repo: %s).", get_repo_id())
        return True
    except Exception as exc:
        log.error("HuggingFace authentication failed: %s", exc)
        return False


# ─────────────────────────────────────────────────────────────────────────────
# Schema helpers
# ─────────────────────────────────────────────────────────────────────────────

def _build_schema_columns() -> list[str]:
    """Return the full ordered list of column names for the dataset schema."""
    cols = [
        "researcher_name", "submitted_at",
        "scholar_tags", "scholar_source", "tech_areas_used",
    ]
    for i in range(MAX_AREAS):
        cols.append(f"area_{i}_name")
        for dim in DIMENSIONS:
            cols.append(f"area_{i}_{dim}")
    cols += ["vision_definition", "vision_examples", "vision_explore"]
    return cols


def _empty_dataframe() -> pd.DataFrame:
    """Return a typed empty DataFrame matching the dataset schema."""
    return pd.DataFrame(columns=_build_schema_columns())


# ─────────────────────────────────────────────────────────────────────────────
# Deserialisation
# ─────────────────────────────────────────────────────────────────────────────

def _deserialise_row(row_dict: dict) -> dict:
    """
    Convert a raw HF dataset row into a clean Python dict.

    - JSON-decodes list columns that were stored as strings.
    - Replaces float NaN (pandas null) with Python None.
    - Converts integer slider columns from float64 to int where possible.
    """
    result = {}
    for k, v in row_dict.items():
        # NaN → None
        if isinstance(v, float) and pd.isna(v):
            result[k] = None
            continue
        # JSON-decode list columns
        if k in _LIST_COLS:
            if isinstance(v, str):
                try:
                    v = json.loads(v)
                except (json.JSONDecodeError, ValueError):
                    v = []
            result[k] = v or []
            continue
        # Float slider values → int
        if isinstance(v, float) and v == int(v):
            result[k] = int(v)
            continue
        result[k] = v
    return result


def _serialise_record(record: dict) -> dict:
    """
    Prepare a record for Parquet storage.

    - JSON-encodes list columns.
    - Keeps everything else as-is.
    """
    out = dict(record)
    for col in _LIST_COLS:
        val = out.get(col, [])
        if isinstance(val, list):
            out[col] = json.dumps(val)
    return out


# ─────────────────────────────────────────────────────────────────────────────
# Load
# ─────────────────────────────────────────────────────────────────────────────

def load_existing_submissions() -> pd.DataFrame:
    """
    Load all submissions from the HF dataset into a DataFrame.

    Returns an empty DataFrame (correct schema, no rows) on any error —
    so callers never need to guard against None.
    """
    if not HF_TOKEN:
        return _empty_dataframe()
    try:
        from datasets import load_dataset   # type: ignore[import]
        ds  = load_dataset(get_repo_id(), split="train", token=HF_TOKEN)
        df  = ds.to_pandas()
        # Deserialise list columns that were stored as JSON strings
        for col in _LIST_COLS:
            if col in df.columns:
                df[col] = df[col].apply(
                    lambda v: json.loads(v)
                    if isinstance(v, str) else (v if v is not None else [])
                )
        return df
    except Exception as exc:
        log.warning("Could not load HF dataset (%s): %s", get_repo_id(), exc)
        return _empty_dataframe()


def load_researcher_session(name: str) -> Optional[dict]:
    """
    Load a single researcher's most recent submission as a plain Python dict.

    Returns None if the dataset is empty or no row exists for this name.
    The returned dict is ready to be consumed by parse_ratings_from_record()
    and the Gradio session-restore logic in app.py.
    """
    df = load_existing_submissions()
    if df.empty:
        return None
    matches = df[df["researcher_name"] == name]
    if matches.empty:
        return None
    # Use the last row (most recent, if multiple exist due to partial uploads)
    raw = matches.iloc[-1].to_dict()
    return _deserialise_row(raw)


# ─────────────────────────────────────────────────────────────────────────────
# Write
# ─────────────────────────────────────────────────────────────────────────────

def upsert_submission(record: dict) -> tuple[bool, str]:
    """
    Insert or replace a researcher's submission in the HF dataset.

    If a row already exists for researcher_name, it is removed and the
    new record is appended. The updated dataset is then pushed to Hub.

    Returns:
        (success: bool, message: str)   — message is user-displayable.
    """
    if not HF_TOKEN:
        return False, "HF_TOKEN not configured — submission was not saved to HuggingFace."

    name = str(record.get("researcher_name", "")).strip()
    if not name:
        return False, "Cannot save: researcher name is empty."

    try:
        # Load current dataset
        df = load_existing_submissions()

        # Drop any existing rows for this researcher
        if not df.empty and "researcher_name" in df.columns:
            df = df[df["researcher_name"] != name].reset_index(drop=True)

        # Serialise and append
        serialised = _serialise_record(record)
        new_row_df  = pd.DataFrame([serialised])

        # Align columns so concat doesn't create NaN columns
        for col in _build_schema_columns():
            if col not in new_row_df.columns:
                new_row_df[col] = None
            if col not in df.columns:
                df[col] = None

        combined = pd.concat([df, new_row_df[_build_schema_columns()]], ignore_index=True)

        # Push to Hub
        from datasets import Dataset   # type: ignore[import]
        ds = Dataset.from_pandas(combined, preserve_index=False)
        ds.push_to_hub(get_repo_id(), token=HF_TOKEN, private=True)

        log.info("Submission saved for '%s' → %s", name, get_repo_id())
        return True, f"Submission saved successfully for {name}."

    except Exception as exc:
        log.error("upsert_submission failed: %s", exc)
        return False, f"Error saving submission: {exc}"


# ─────────────────────────────────────────────────────────────────────────────
# Analytics transforms
# ─────────────────────────────────────────────────────────────────────────────

def get_all_radar_data() -> pd.DataFrame:
    """
    Return all submissions as a long-form DataFrame for visualisation.

    Output columns: researcher, tech_area, <dim_key for each DIMENSION>,
                    submitted_at

    One row per (researcher, tech_area) — used by all dashboard charts.
    Areas with no rated dimensions are skipped.
    """
    df = load_existing_submissions()
    empty = pd.DataFrame(
        columns=["researcher", "tech_area"] + list(DIMENSIONS.keys()) + ["submitted_at"]
    )
    if df.empty:
        return empty

    records = []
    for _, row in df.iterrows():
        areas = row.get("tech_areas_used") or []
        if isinstance(areas, str):
            try:
                areas = json.loads(areas)
            except Exception:
                areas = list(DEFAULT_TECH_AREAS)

        for i, area in enumerate(areas):
            if not area or i >= MAX_AREAS:
                continue
            dim_vals = {}
            for dim in DIMENSIONS:
                raw = row.get(f"area_{i}_{dim}")
                try:
                    dim_vals[dim] = int(raw) if raw is not None and not pd.isna(raw) else None
                except (TypeError, ValueError):
                    dim_vals[dim] = None

            if any(v is not None for v in dim_vals.values()):
                records.append({
                    "researcher":  row.get("researcher_name", ""),
                    "tech_area":   area,
                    **dim_vals,
                    "submitted_at": row.get("submitted_at", ""),
                })

    return pd.DataFrame(records) if records else empty


def get_all_vision_data() -> list[dict]:
    """
    Return all vision responses as a list of dicts.

    Used by the 'Deep Tech Voices' dashboard tab.
    Rows with all-empty vision fields are excluded.
    """
    df = load_existing_submissions()
    if df.empty:
        return []

    rows = []
    for _, row in df.iterrows():
        entry = {
            "researcher":   str(row.get("researcher_name") or ""),
            "definition":   str(row.get("vision_definition") or ""),
            "examples":     str(row.get("vision_examples") or ""),
            "explore":      str(row.get("vision_explore") or ""),
            "submitted_at": str(row.get("submitted_at") or ""),
        }
        if any(entry[k] for k in ("definition", "examples", "explore")):
            rows.append(entry)
    return rows


def get_submission_summary() -> dict:
    """
    Return a lightweight summary dict for the dashboard status bar.

    Keys: total_submissions, researchers_list, last_updated.
    """
    df = load_existing_submissions()
    if df.empty:
        return {"total_submissions": 0, "researchers_list": [], "last_updated": "—"}

    names = sorted(df["researcher_name"].dropna().unique().tolist())
    last  = ""
    if "submitted_at" in df.columns:
        ts = df["submitted_at"].dropna()
        last = ts.max() if not ts.empty else ""

    return {
        "total_submissions": len(df),
        "researchers_list":  names,
        "last_updated":      last[:10] if last else "—",
    }


# ─────────────────────────────────────────────────────────────────────────────
# Export
# ─────────────────────────────────────────────────────────────────────────────

def export_to_csv() -> str:
    """
    Export the full dataset as a CSV to a temp file.

    Returns the file path string (for gr.File download component),
    or an empty string if no data or an error occurs.
    """
    try:
        df = load_existing_submissions()
        if df.empty:
            return ""
        tmp = tempfile.NamedTemporaryFile(
            suffix=".csv", delete=False, prefix="deep_tech_radar_"
        )
        df.to_csv(tmp.name, index=False, encoding="utf-8")
        log.info("CSV exported to %s", tmp.name)
        return tmp.name
    except Exception as exc:
        log.error("CSV export failed: %s", exc)
        return ""
