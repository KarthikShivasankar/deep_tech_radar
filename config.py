"""
config.py — Central configuration for the Deep Tech Skill Radar app.
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

# Raw list of 12 provided Google Scholar URLs.
# Mapping to team member names is done at runtime by scholar.build_scholar_url_mapping().
SCHOLAR_URLS_RAW: list[str] = [
    "https://scholar.google.com/citations?user=OrRAiLoAAAAJ&hl=en",
    "https://scholar.google.com/citations?user=aBqoj50AAAAJ&hl=en",
    "https://scholar.google.com/citations?user=AsUXppAAAAAJ&hl=en",
    "https://scholar.google.com/citations?user=lT6h24IAAAAJ&hl=en",
    "https://scholar.google.com/citations?user=Nm4OBGYAAAAJ&hl=en",
    "https://scholar.google.com/citations?user=2aWHQCAAAAAJ&hl=en",
    "https://scholar.google.com/citations?user=FOLGI1MAAAAJ&hl=en",
    "https://scholar.google.com/citations?user=n8l-leAAAAAJ&hl=no",
    "https://scholar.google.com/citations?user=zD4LcjsAAAAJ&hl=en",
    "https://scholar.google.com/citations?view_op=list_works&hl=en&user=o39iDR0AAAAJ",
    "https://scholar.google.com/citations?user=y3n7pxkAAAAJ&hl=en",
    "https://scholar.google.com/citations?user=sdZqJXMAAAAJ&hl=en",
]

# ── Deep Tech Areas ───────────────────────────────────────────────────────────

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

MAX_CUSTOM_AREAS: int = 5
MAX_AREAS: int = len(DEFAULT_TECH_AREAS) + MAX_CUSTOM_AREAS

# ── Rating Dimensions ─────────────────────────────────────────────────────────

DIMENSIONS: dict[str, str] = {
    "interest":   "Interest Level",
    "expertise":  "Current Technical Expertise",
    "contribute": "Desire to Contribute",
}

DIMENSION_COLORS: dict[str, tuple[str, str]] = {
    "interest":   ("rgba(33, 150, 243, 0.18)",  "rgb(33, 150, 243)"),
    "expertise":  ("rgba(76, 175, 80,  0.18)",  "rgb(76, 175, 80)"),
    "contribute": ("rgba(255, 152, 0, 0.18)",   "rgb(255, 152, 0)"),
}

# ── Sliders ───────────────────────────────────────────────────────────────────

SLIDER_MIN:     int = 1
SLIDER_MAX:     int = 5
SLIDER_DEFAULT: int = 3

# ── HuggingFace ───────────────────────────────────────────────────────────────

HF_TOKEN:        str | None = os.getenv("HF_TOKEN")
HF_USERNAME:     str        = os.getenv("HF_USERNAME", "")
HF_DATASET_NAME: str        = os.getenv("DATASET_NAME", "deep-tech-radar")

# ── Semantic Scholar ──────────────────────────────────────────────────────────

SS_API_KEY:      str | None = os.getenv("SS_API_KEY")
SS_MAX_PAPERS:   int        = 50
CURRENT_YEAR:    int        = 2025
RECENT_YEARS:    int        = 3   # papers from CURRENT_YEAR-RECENT_YEARS onwards = "recent"

SS_GENERIC_TAGS: frozenset[str] = frozenset({
    "Computer Science", "Mathematics", "Physics", "Engineering",
    "Biology", "Medicine", "Chemistry", "Economics", "Psychology",
    "Sociology", "Philosophy", "History", "Art", "Literature",
    "Environmental Science", "Business", "Political Science",
})

# ── Scholar Tag → Tech Area Mapping ──────────────────────────────────────────
# Keys are lowercase substrings; matching: any(kw in tag.lower() for kw in map)

SCHOLAR_TAG_TO_AREA: dict[str, str] = {
    # AI/ML & Trustworthy AI
    "machine learning":              "AI/ML & Trustworthy AI",
    "deep learning":                 "AI/ML & Trustworthy AI",
    "neural network":                "AI/ML & Trustworthy AI",
    "artificial intelligence":       "AI/ML & Trustworthy AI",
    "large language model":          "AI/ML & Trustworthy AI",
    "natural language processing":   "AI/ML & Trustworthy AI",
    "nlp":                           "AI/ML & Trustworthy AI",
    "computer vision":               "AI/ML & Trustworthy AI",
    "reinforcement learning":        "AI/ML & Trustworthy AI",
    "federated learning":            "AI/ML & Trustworthy AI",
    "explainability":                "AI/ML & Trustworthy AI",
    "explainable ai":                "AI/ML & Trustworthy AI",
    "xai":                           "AI/ML & Trustworthy AI",
    "fairness":                      "AI/ML & Trustworthy AI",
    "trustworthy ai":                "AI/ML & Trustworthy AI",
    "responsible ai":                "AI/ML & Trustworthy AI",
    "ai ethics":                     "AI/ML & Trustworthy AI",
    "transfer learning":             "AI/ML & Trustworthy AI",
    "generative model":              "AI/ML & Trustworthy AI",
    "llm":                           "AI/ML & Trustworthy AI",
    "transformer":                   "AI/ML & Trustworthy AI",
    "anomaly detection":             "AI/ML & Trustworthy AI",
    "predictive model":              "AI/ML & Trustworthy AI",
    "data science":                  "AI/ML & Trustworthy AI",
    "pattern recognition":           "AI/ML & Trustworthy AI",
    "classification":                "AI/ML & Trustworthy AI",
    "regression":                    "AI/ML & Trustworthy AI",

    # IoT & Edge Computing
    "internet of things":            "IoT & Edge Computing",
    "iot":                           "IoT & Edge Computing",
    "edge computing":                "IoT & Edge Computing",
    "fog computing":                 "IoT & Edge Computing",
    "embedded system":               "IoT & Edge Computing",
    "sensor network":                "IoT & Edge Computing",
    "wireless sensor":               "IoT & Edge Computing",
    "mqtt":                          "IoT & Edge Computing",
    "smart device":                  "IoT & Edge Computing",
    "industrial iot":                "IoT & Edge Computing",
    "iiot":                          "IoT & Edge Computing",
    "edge intelligence":             "IoT & Edge Computing",
    "cyber-physical":                "IoT & Edge Computing",
    "real-time system":              "IoT & Edge Computing",
    "smart manufacturing":           "IoT & Edge Computing",
    "pervasive computing":           "IoT & Edge Computing",
    "ubiquitous computing":          "IoT & Edge Computing",
    "wearable":                      "IoT & Edge Computing",

    # Cybersecurity
    "cybersecurity":                 "Cybersecurity",
    "cyber security":                "Cybersecurity",
    "information security":          "Cybersecurity",
    "network security":              "Cybersecurity",
    "intrusion detection":           "Cybersecurity",
    "vulnerability":                 "Cybersecurity",
    "malware":                       "Cybersecurity",
    "threat detection":              "Cybersecurity",
    "access control":                "Cybersecurity",
    "authentication":                "Cybersecurity",
    "encryption":                    "Cybersecurity",
    "cryptography":                  "Cybersecurity",
    "security testing":              "Cybersecurity",
    "penetration testing":           "Cybersecurity",
    "zero trust":                    "Cybersecurity",
    "attack detection":              "Cybersecurity",
    "supply chain security":         "Cybersecurity",
    "secure coding":                 "Cybersecurity",

    # Privacy Engineering
    "privacy":                       "Privacy Engineering",
    "data privacy":                  "Privacy Engineering",
    "privacy by design":             "Privacy Engineering",
    "differential privacy":          "Privacy Engineering",
    "anonymization":                 "Privacy Engineering",
    "anonymisation":                 "Privacy Engineering",
    "gdpr":                          "Privacy Engineering",
    "data protection":               "Privacy Engineering",
    "privacy-preserving":            "Privacy Engineering",
    "consent management":            "Privacy Engineering",
    "k-anonymity":                   "Privacy Engineering",
    "secure multiparty computation": "Privacy Engineering",
    "personal data":                 "Privacy Engineering",

    # Green/Sustainable Computing
    "green computing":               "Green/Sustainable Computing",
    "sustainable computing":         "Green/Sustainable Computing",
    "energy efficiency":             "Green/Sustainable Computing",
    "energy consumption":            "Green/Sustainable Computing",
    "carbon footprint":              "Green/Sustainable Computing",
    "green software":                "Green/Sustainable Computing",
    "sustainability":                "Green/Sustainable Computing",
    "low-power":                     "Green/Sustainable Computing",
    "environmental impact":          "Green/Sustainable Computing",
    "net zero":                      "Green/Sustainable Computing",
    "carbon-aware":                  "Green/Sustainable Computing",
    "renewable energy":              "Green/Sustainable Computing",

    # Digital Twins
    "digital twin":                  "Digital Twins",
    "model-driven engineering":      "Digital Twins",
    "simulation":                    "Digital Twins",
    "virtual model":                 "Digital Twins",
    "predictive maintenance":        "Digital Twins",
    "mde":                           "Digital Twins",
    "model synchronization":         "Digital Twins",
    "runtime model":                 "Digital Twins",
    "system modeling":               "Digital Twins",
    "ocl":                           "Digital Twins",

    # Software Engineering
    "software engineering":          "Software Engineering",
    "software development":          "Software Engineering",
    "agile":                         "Software Engineering",
    "devops":                        "Software Engineering",
    "continuous integration":        "Software Engineering",
    "software testing":              "Software Engineering",
    "test automation":               "Software Engineering",
    "software architecture":         "Software Engineering",
    "software design":               "Software Engineering",
    "requirements engineering":      "Software Engineering",
    "model-based testing":           "Software Engineering",
    "refactoring":                   "Software Engineering",
    "program analysis":              "Software Engineering",
    "static analysis":               "Software Engineering",
    "software maintenance":          "Software Engineering",
    "code review":                   "Software Engineering",
    "api design":                    "Software Engineering",

    # Cloud & Distributed Systems
    "cloud computing":               "Cloud & Distributed Systems",
    "distributed system":            "Cloud & Distributed Systems",
    "serverless":                    "Cloud & Distributed Systems",
    "kubernetes":                    "Cloud & Distributed Systems",
    "container":                     "Cloud & Distributed Systems",
    "service mesh":                  "Cloud & Distributed Systems",
    "stream processing":             "Cloud & Distributed Systems",
    "fault tolerance":               "Cloud & Distributed Systems",
    "scalability":                   "Cloud & Distributed Systems",
    "distributed computing":         "Cloud & Distributed Systems",
    "microservice":                  "Cloud & Distributed Systems",
    "cloud-native":                  "Cloud & Distributed Systems",
    "consensus algorithm":           "Cloud & Distributed Systems",
    "service-oriented":              "Cloud & Distributed Systems",

    # Self-Adaptive Systems
    "self-adaptive":                 "Self-Adaptive Systems",
    "self-healing":                  "Self-Adaptive Systems",
    "self-organizing":               "Self-Adaptive Systems",
    "autonomic computing":           "Self-Adaptive Systems",
    "feedback control":              "Self-Adaptive Systems",
    "runtime adaptation":            "Self-Adaptive Systems",
    "context-aware":                 "Self-Adaptive Systems",
    "adaptive system":               "Self-Adaptive Systems",
    "mape-k":                        "Self-Adaptive Systems",
    "dynamic reconfiguration":       "Self-Adaptive Systems",

    # Technical Debt & Quality
    "technical debt":                "Technical Debt & Quality",
    "code smell":                    "Technical Debt & Quality",
    "software quality":              "Technical Debt & Quality",
    "maintainability":               "Technical Debt & Quality",
    "software metrics":              "Technical Debt & Quality",
    "software evolution":            "Technical Debt & Quality",
    "legacy system":                 "Technical Debt & Quality",
    "architecture erosion":          "Technical Debt & Quality",
    "defect prediction":             "Technical Debt & Quality",
    "code coverage":                 "Technical Debt & Quality",
    "software reliability":          "Technical Debt & Quality",
}

# ── OpenAI ────────────────────────────────────────────────────────────────────

OPENAI_API_KEY: str | None = os.getenv("OPENAI_API_KEY")
OPENAI_MODEL:   str        = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
OPENAI_TIMEOUT: int        = 120

# ── UI ────────────────────────────────────────────────────────────────────────

APP_TITLE:    str = "SINTEF Deep Tech & Skill Radar"
APP_SUBTITLE: str = (
    "Map your research interests, expertise, and contribution goals "
    "across deep tech areas — visualised as a group skill radar."
)

# ── ThoughtWorks-style Radar ──────────────────────────────────────────────────

TW_QUADRANTS: dict = {
    "AI & Intelligence": {
        "areas":       ["AI/ML & Trustworthy AI", "Digital Twins"],
        "angle_start": 0,
        "angle_end":   90,
        "fill":        "rgba(52, 152, 219, 0.10)",
        "color":       "rgb(52, 152, 219)",
        "label_angle": 45,
    },
    "Infrastructure & Systems": {
        "areas":       ["IoT & Edge Computing", "Cloud & Distributed Systems", "Self-Adaptive Systems"],
        "angle_start": 90,
        "angle_end":   180,
        "fill":        "rgba(46, 204, 113, 0.10)",
        "color":       "rgb(46, 204, 113)",
        "label_angle": 135,
    },
    "Security & Privacy": {
        "areas":       ["Cybersecurity", "Privacy Engineering"],
        "angle_start": 180,
        "angle_end":   270,
        "fill":        "rgba(231, 76, 60, 0.10)",
        "color":       "rgb(231, 76, 60)",
        "label_angle": 225,
    },
    "Engineering Practices": {
        "areas":       ["Software Engineering", "Green/Sustainable Computing", "Technical Debt & Quality"],
        "angle_start": 270,
        "angle_end":   360,
        "fill":        "rgba(243, 156, 18, 0.10)",
        "color":       "rgb(243, 156, 18)",
        "label_angle": 315,
    },
}

TW_RINGS: list[dict] = [
    {"name": "Lead",       "description": "Expert — ready to lead projects",    "r_inner": 0.00, "r_outer": 0.25, "ring_color": "rgba(44,62,80,0.85)"},
    {"name": "Contribute", "description": "Proficient — contributing actively", "r_inner": 0.25, "r_outer": 0.50, "ring_color": "rgba(52,152,219,0.85)"},
    {"name": "Grow",       "description": "Learning — developing skills",       "r_inner": 0.50, "r_outer": 0.75, "ring_color": "rgba(39,174,96,0.85)"},
    {"name": "Watch",      "description": "Aware — monitoring the area",        "r_inner": 0.75, "r_outer": 1.00, "ring_color": "rgba(189,195,199,0.85)"},
]

TW_RING_THRESHOLDS: list[float] = [4.0, 3.0, 2.0]

# ── Ollama (kept for backward compat, unused by new AI tab) ──────────────────
OLLAMA_BASE_URL: str = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434/v1")
OLLAMA_MODEL:    str = os.getenv("OLLAMA_MODEL",    "gemma3:latest")
OLLAMA_TIMEOUT:  int = 120
