# Deep Tech & Skill Radar

Interactive skill-mapping and visualisation tool for the
[SINTEF Trustworthy Green IoT Software Research Group](https://www.sintef.no/en/digital/departments/department-of-sustainable-communication-technologies/trustworthy-green-iot-software-research-group/).

Researchers rate their **interest**, **expertise**, and **contribution intent** across deep tech areas.
The app auto-fetches research tags from Semantic Scholar / Google Scholar, stores everything in a
private HuggingFace dataset, and renders group dashboards that update as people submit.

---

## Features

| Feature | Detail |
|---|---|
| **Session restore** | Selecting your name pre-fills all previous answers from the dataset |
| **Scholar lookup** | Auto-tags via Semantic Scholar (by name) or Google Scholar (by profile URL) |
| **Live radar preview** | Radar chart updates on every slider move |
| **Custom tech areas** | Up to 5 extra areas per person on top of 10 defaults |
| **7 dashboard charts** | Group radar, heatmap, contribution bar, capability bubble, individual radar, overlay, Deep Tech Voices |
| **Upsert storage** | Re-submitting replaces the previous row — no duplicates |
| **CSV export** | Download the full dataset as CSV from the dashboard |
| **HF Spaces ready** | One `uv run` locally, or push straight to Hugging Face Spaces |

---

## Prerequisites

Install **uv** once on your machine:

```bash
# macOS / Linux
curl -LsSf https://astral.sh/uv/install.sh | sh

# Windows (PowerShell)
powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
```

> uv manages Python versions and virtual environments for you — no separate `python -m venv` step needed.

---

## Quick start

```bash
# 1. Clone / navigate to the project
cd "Deep tech and skill radar"

# 2. Install all dependencies (creates .venv automatically)
uv sync

# 3. Configure credentials
cp .env.example .env
#    Open .env and fill in HF_TOKEN and HF_USERNAME

# 4. Run the app
uv run python app.py
#    → open http://localhost:7860

# Public tunnel (temporary shareable URL)
uv run python app.py --share
```

That's it. `uv sync` reads `pyproject.toml`, resolves the lock file, creates `.venv`, and installs everything. No `pip`, no `conda`, no manual venv activation required.

---

## HuggingFace setup

The app stores submissions in a **private** HuggingFace dataset.

1. Create a HuggingFace account at [huggingface.co](https://huggingface.co)
2. Go to **Settings → Access Tokens → New token** — set Role to **Write**
3. Copy the token into `.env`:

```env
HF_TOKEN=hf_your_write_token_here
HF_USERNAME=your-hf-username-or-org
DATASET_NAME=deep-tech-radar          # repo created automatically on first submit
SS_API_KEY=                           # optional — raises Semantic Scholar rate limit
```

The dataset at `HF_USERNAME/DATASET_NAME` is created automatically the first time someone submits.

---

## App walkthrough

### Tab 1 · Profile & Scholar Lookup
- Select your name from the dropdown (or type a custom name)
- **Selecting a name auto-restores your previous submission** — sliders, tags, and vision text are all pre-filled
- Optionally paste your Google Scholar profile URL, then click **Lookup Research Profile**
- The app queries Semantic Scholar (by name) and/or Google Scholar (by URL) and populates research tags
- Uncheck tags you want to exclude; add custom ones manually

### Tab 2 · Skill Radar Assessment
- Rate each deep tech area on three sliders (1 = low, 5 = high):
  - **Interest Level** — how interested are you?
  - **Current Expertise** — how deep is your technical knowledge?
  - **Desire to Contribute** — how much do you want to work in this area?
- The radar chart on the right updates live as you move sliders
- Click **Add area** to add a custom tech area (up to 5 extra)

### Tab 3 · Deep Tech Vision
Three open-ended questions:
- What do you think deep tech is?
- Give examples from your work or industry
- Which area do you most want to explore next?

Responses appear in the **Deep Tech Voices** dashboard panel.

### Tab 4 · Submit & Preview
- Click **Generate Preview** to see your full submission as JSON
- Review, then click **Submit to Dataset** to save
- Re-submitting replaces your previous entry (upsert, no duplicates)

### Tab 5 · Group Dashboard
Click **Refresh Dashboard** to load live data. Switch between:

| Sub-tab | What it shows |
|---|---|
| Group Radar | Mean ± std deviation for the selected dimension |
| Heatmap | Researcher × tech area grid, coloured by dimension score |
| Contribution Priorities | Horizontal bar — total group contribution score per area |
| Capability Map | Bubble chart: interest vs expertise, bubble size = contribution |
| Individual Comparison | Radar + grouped bar for one researcher across all dimensions |
| Overlay Comparison | Multiple researchers on a single radar for side-by-side comparison |
| Deep Tech Voices | All open-ended vision responses |

Use the **Dimension** dropdown to switch between Interest, Expertise, and Contribution views across all group charts.

### Tab 6 · Technology Radar

Inspired by [ThoughtWorks Technology Radar](https://www.thoughtworks.com/radar) and the [Skill Radar adaptation](https://lihsmi.ch/learning/2015/04/25/skill-radar-technology-radar.html).

The tab combines **both** radar concepts in one view:

| View mode | What it shows | Inspired by |
|---|---|---|
| **Group** | One blip per tech area, positioned by the group's collective expertise & contribution intent | *Technology Radar* — tracks adoption readiness of technologies |
| **Individual** | One blip per tech area for a selected researcher, positioned by their personal scores | *Skill Radar* — tracks individual proficiency |

**Rings** (centre = highest skill/readiness):

| Ring | Meaning |
|---|---|
| **Lead** | Expert — ready to lead projects in this area |
| **Contribute** | Proficient — can contribute actively |
| **Grow** | Learning — currently developing skills |
| **Watch** | Aware — monitoring this area |

**Quadrants** group tech areas by domain:
- **AI & Intelligence** — AI/ML, Digital Twins
- **Infrastructure & Systems** — IoT/Edge, Cloud, Self-Adaptive Systems
- **Security & Privacy** — Cybersecurity, Privacy Engineering
- **Engineering Practices** — Software Engineering, Green Computing, Technical Debt

The numbered blip legend below the chart names every blip and shows its scores.

### Tab 7 · AI Research Assistant

Five agents powered by Ollama (local LLM — no API costs, fully private):

| Agent | What it produces |
|---|---|
| **Team Synergies** | Collaboration pairs/teams based on complementary radar scores |
| **Project Ideas** | 5 fundable ideas aligned with EU Horizon / NRC programs |
| **Project Proposal** | Structured 1-page proposal (give an idea title in the context box) |
| **Skill Gap Analysis** | Interest–expertise gaps, bus-factor risks, recommended actions |
| **Competence Roadmap** | 12-month plan for moving the group up a ring in priority areas |

**Setup:**
```bash
ollama serve                  # start the local API (keep this running)
ollama pull gemma3            # download the model (~5 GB, once)
# or: ollama pull gemma4      # if available — check with: ollama list
```

Change the model by setting `OLLAMA_MODEL=<model>` in `.env`.  Any model available via `ollama list` works.

---

## Project structure

```
.
├── app.py              # Gradio Blocks UI — tabs, event wiring, handlers
├── config.py           # All constants: team, areas, TW quadrants, Ollama config
├── scholar.py          # Semantic Scholar REST API + scholarly (Google Scholar)
├── storage.py          # HuggingFace dataset: load, upsert, session restore, CSV export
├── charts.py           # Plotly chart builders (radar, heatmap, bar, bubble, overlay…)
├── radar_viz.py        # ThoughtWorks-style Technology + Skill Radar (Plotly)
├── ai_agents.py        # Ollama AI agents: synergies, ideas, proposals, gaps, roadmap
├── utils.py            # Record assembly, validation, column flattening / parsing
├── pyproject.toml      # Project metadata and dependencies (uv reads this)
├── uv.lock             # Pinned dependency tree — commit this file
├── .env.example        # Credential template — copy to .env
├── .gitignore
└── README.md
```

---

## Customisation

All customisation is done in **`config.py`** — no other file needs changing for common adjustments.

| What to change | Where |
|---|---|
| Add / remove team members | `TEAM_MEMBERS` list in `config.py` |
| Add / remove default tech areas | `DEFAULT_TECH_AREAS` + matching `TW_QUADRANTS` entry in `config.py` |
| Change max custom areas per person | `MAX_CUSTOM_AREAS` in `config.py` |
| Change slider scale (default 1–5) | `SLIDER_MIN`, `SLIDER_MAX` in `config.py` |
| Change radar colours | `DIMENSION_COLORS` and `TW_QUADRANTS[…]["color"]` in `config.py` |
| Move an area to a different quadrant | Edit `TW_QUADRANTS[…]["areas"]` in `config.py` |
| Change AI model | Set `OLLAMA_MODEL=<model>` in `.env` or `OLLAMA_MODEL` in `config.py` |
| Add a new AI agent | Add a `stream_<name>` function in `ai_agents.py` and register it in `AGENT_FUNCTIONS` |
| Add a new chart | New `build_*` function in `charts.py`, called from `app.py` |
| Add a 4th rating dimension | See `EXTENSION NOTE` comment in `config.py` and `DIMENSIONS` dict |

---

## Adding a dependency

```bash
uv add some-package          # adds to pyproject.toml and updates uv.lock
uv add "some-package>=2.0"   # with version constraint
uv remove some-package       # remove a dependency
```

---

## Deploying to Hugging Face Spaces

1. Create a new Space at [huggingface.co/new-space](https://huggingface.co/new-space)
   - **SDK**: Gradio
   - **Visibility**: Private (recommended for internal use)
   - **Hardware**: CPU Basic (free tier is sufficient)

2. Add Secrets in the Space **Settings** tab:
   ```
   HF_TOKEN      →  your write token
   HF_USERNAME   →  your HF username or org
   DATASET_NAME  →  deep-tech-radar
   ```

3. Push to the Space repository:
   ```bash
   git remote add space https://huggingface.co/spaces/YOUR_ORG/deep-tech-radar-app
   git push space main
   ```

   Hugging Face Spaces automatically detects `app.py` as the Gradio entry point.

> The private HF dataset and the Space can share the same `HF_TOKEN` — the app reads and writes the dataset on behalf of the Space.

---

## Semantic Scholar rate limits

Without an API key the public endpoint allows ~100 requests per 5 minutes — sufficient for a small team.
For higher throughput, request a free key at [semanticscholar.org/product/api](https://www.semanticscholar.org/product/api) and add it to `.env` as `SS_API_KEY`.

---

## Tech stack

| Library | Role |
|---|---|
| [Gradio](https://www.gradio.app) | Web UI |
| [Plotly](https://plotly.com/python/) | Interactive charts |
| [HuggingFace datasets](https://huggingface.co/docs/datasets) | Dataset storage (Parquet on HF Hub) |
| [Semantic Scholar API](https://api.semanticscholar.org) | Auto-tag extraction from papers |
| [scholarly](https://scholarly.readthedocs.io) | Google Scholar profile scraping |
| [uv](https://docs.astral.sh/uv/) | Package and environment management |
