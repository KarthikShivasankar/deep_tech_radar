"""
config.py — Central configuration for the Deep Tech Skill Radar app.

To customise the app without touching any other file:
  - Add/remove team members in TEAM_MEMBERS
  - Add/remove default tech areas in DEFAULT_TECH_AREAS
  - Add a new rating dimension in DIMENSIONS (then add a slider in app.py
    and a column in storage.py's schema — see the EXTENSION NOTES below)
  - Change colour scheme in DIMENSION_COLORS
  - Adjust max custom areas in MAX_CUSTOM_AREAS
"""

import os
from dotenv import load_dotenv

load_dotenv()

# ── Team ─────────────────────────────────────────────────────────────────────

TEAM_MEMBERS: list[str] = [
    "Sagar Sen",
    "Merve Astekin",
    "Rustem Dautov",
    "Gencer Erdogan",
    "Arda Goknil",
    "Erik Johannes Husom",
    "Phu Nguyen",
    "Karthik Shivashankar",
    "Hui Song",
    "Shukun Tokas",
    "Simeon Tverdal",
    "Adela Nedisan Videsjorden",
    "Sondre Sigstad Wikberg",
]

# ── Deep Tech Areas ───────────────────────────────────────────────────────────
# Default areas shown to all users. Each user can append up to
# MAX_CUSTOM_AREAS more via the UI, stored per-submission.

DEFAULT_TECH_AREAS: list[str] = [
    "AI/ML & Trustworthy AI",
    "IoT & Edge Computing",
    "Cybersecurity",
    "Privacy Engineering",
    "Green/Sustainable Computing",
    "Digital Twins",
    "Software Engineering",
    "Cloud & Distributed Systems",
    "Self-Adaptive Systems",
    "Technical Debt & Quality",
]

MAX_CUSTOM_AREAS: int = 5          # extra slots a user can add per submission
MAX_AREAS: int = len(DEFAULT_TECH_AREAS) + MAX_CUSTOM_AREAS   # 15 total

# ── Rating Dimensions ─────────────────────────────────────────────────────────
# Maps internal key → display label.
# EXTENSION NOTE: to add "Readiness" as a 4th dimension:
#   1. Add  "readiness": "Readiness Level"  here
#   2. Add  "readiness": ("rgba(200,0,120,0.15)", "rgb(200,0,120)")  in DIMENSION_COLORS
#   3. Add a readiness slider per accordion in app.py
#   4. The storage schema auto-expands via DIMENSIONS iteration

DIMENSIONS: dict[str, str] = {
    "interest":   "Interest Level",
    "expertise":  "Current Technical Expertise",
    "contribute": "Desire to Contribute",
}

DIMENSION_COLORS: dict[str, tuple[str, str]] = {
    # key: (fill_rgba, solid_rgb)
    "interest":   ("rgba(0, 120, 200, 0.18)",  "rgb(0, 120, 200)"),
    "expertise":  ("rgba(0, 180, 80,  0.18)",  "rgb(0, 180, 80)"),
    "contribute": ("rgba(220, 120, 0, 0.18)",  "rgb(220, 120, 0)"),
}

# ── Sliders ───────────────────────────────────────────────────────────────────

SLIDER_MIN:     int = 1
SLIDER_MAX:     int = 5
SLIDER_DEFAULT: int = 3   # pre-filled value before user interacts

# ── HuggingFace ───────────────────────────────────────────────────────────────

HF_TOKEN:        str | None = os.getenv("HF_TOKEN")
HF_USERNAME:     str        = os.getenv("HF_USERNAME", "")
HF_DATASET_NAME: str        = os.getenv("DATASET_NAME", "deep-tech-radar")

# ── Semantic Scholar ──────────────────────────────────────────────────────────

SS_API_KEY:      str | None = os.getenv("SS_API_KEY")   # optional — raises rate limit
SS_MAX_PAPERS:   int        = 50

# Tags filtered out from Semantic Scholar results (too broad to be useful)
SS_GENERIC_TAGS: frozenset[str] = frozenset({
    "Computer Science", "Mathematics", "Physics", "Engineering",
    "Biology", "Medicine", "Chemistry", "Economics", "Psychology",
    "Sociology", "Philosophy", "History", "Art", "Literature",
    "Environmental Science", "Business", "Political Science",
})

# ── UI ────────────────────────────────────────────────────────────────────────

APP_TITLE:    str = "SINTEF Deep Tech & Skill Radar"
APP_SUBTITLE: str = (
    "Map your research interests, expertise, and contribution goals "
    "across deep tech areas — visualised as a group skill radar."
)

# ── ThoughtWorks-style Radar ──────────────────────────────────────────────────
# The 10 DEFAULT_TECH_AREAS are distributed across 4 quadrants (90° each).
# Angle convention: standard math — 0° = 3 o'clock, counter-clockwise.
#   Q1 upper-right 0–90°, Q2 upper-left 90–180°,
#   Q3 lower-left 180–270°, Q4 lower-right 270–360°
#
# To add a new default area: add it to DEFAULT_TECH_AREAS above AND to the
# relevant quadrant's "areas" list here.  Custom (user-added) areas not listed
# in any quadrant appear in a separate panel below the chart.

TW_QUADRANTS: dict = {
    "AI & Intelligence": {
        "areas":       ["AI/ML & Trustworthy AI", "Digital Twins"],
        "angle_start": 0,
        "angle_end":   90,
        "fill":        "rgba(52, 152, 219, 0.10)",
        "color":       "rgb(52, 152, 219)",    # blue
        "label_angle": 45,
    },
    "Infrastructure & Systems": {
        "areas":       ["IoT & Edge Computing", "Cloud & Distributed Systems", "Self-Adaptive Systems"],
        "angle_start": 90,
        "angle_end":   180,
        "fill":        "rgba(46, 204, 113, 0.10)",
        "color":       "rgb(46, 204, 113)",    # green
        "label_angle": 135,
    },
    "Security & Privacy": {
        "areas":       ["Cybersecurity", "Privacy Engineering"],
        "angle_start": 180,
        "angle_end":   270,
        "fill":        "rgba(231, 76, 60, 0.10)",
        "color":       "rgb(231, 76, 60)",     # red
        "label_angle": 225,
    },
    "Engineering Practices": {
        "areas":       ["Software Engineering", "Green/Sustainable Computing", "Technical Debt & Quality"],
        "angle_start": 270,
        "angle_end":   360,
        "fill":        "rgba(243, 156, 18, 0.10)",
        "color":       "rgb(243, 156, 18)",    # orange
        "label_angle": 315,
    },
}

# Rings inner → outer.  Lead = centre = highest skill/readiness.
TW_RINGS: list[dict] = [
    {"name": "Lead",       "description": "Expert — ready to lead projects",    "r_inner": 0.00, "r_outer": 0.25, "ring_color": "rgba(44,62,80,0.85)"},
    {"name": "Contribute", "description": "Proficient — contributing actively", "r_inner": 0.25, "r_outer": 0.50, "ring_color": "rgba(52,152,219,0.85)"},
    {"name": "Grow",       "description": "Learning — developing skills",       "r_inner": 0.50, "r_outer": 0.75, "ring_color": "rgba(39,174,96,0.85)"},
    {"name": "Watch",      "description": "Aware — monitoring the area",        "r_inner": 0.75, "r_outer": 1.00, "ring_color": "rgba(189,195,199,0.85)"},
]

# Thresholds for avg(expertise, contribute) → ring index (0=Lead … 3=Watch)
TW_RING_THRESHOLDS: list[float] = [4.0, 3.0, 2.0]

# ── Ollama / AI Agents ────────────────────────────────────────────────────────
# The app uses the OpenAI-compatible Ollama REST API.
# Start Ollama:  ollama serve
# Pull model:    ollama pull gemma3   (or gemma4 if available)
# Override via environment variables.

OLLAMA_BASE_URL: str = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434/v1")
OLLAMA_MODEL:    str = os.getenv("OLLAMA_MODEL",    "gemma3:latest")
OLLAMA_TIMEOUT:  int = 120   # seconds for streaming response
