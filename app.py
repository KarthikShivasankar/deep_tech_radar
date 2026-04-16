"""
app.py — Deep Tech & Skill Radar (Gradio app).

Tabs:
  1. Profile & Scholar Lookup   — select name, fetch or enter research tags
  2. Skill Radar Assessment     — rate each tech area on 3 dimensions; live radar
  3. Deep Tech Vision           — free-text views on what "deep tech" means
  4. Submit & Preview           — JSON preview + submit to HuggingFace dataset
  5. Group Dashboard            — aggregate charts; individual comparison; export
  6. Technology Radar           — ThoughtWorks-style quadrant+ring radar
                                   Group view = Technology Radar (collective adoption)
                                   Individual view = Skill Radar (personal proficiency)
  7. AI Research Assistant      — Ollama-powered agents: synergies, ideas, proposals

Session persistence:
  Selecting a name auto-loads any previous submission from the HF dataset,
  pre-populating all sliders, tags, and vision text.  A returning user only
  needs to adjust what has changed.

Run locally:
    python app.py                     # http://localhost:7860
    python app.py --share             # temporary public URL
"""

from __future__ import annotations

import json
import logging
import sys
from typing import Optional

import gradio as gr

import ai_agents
import charts
import radar_viz
import scholar
import storage
from config import (
    APP_SUBTITLE,
    APP_TITLE,
    DEFAULT_TECH_AREAS,
    DIMENSIONS,
    MAX_AREAS,
    MAX_CUSTOM_AREAS,
    OLLAMA_MODEL,
    SLIDER_DEFAULT,
    SLIDER_MAX,
    SLIDER_MIN,
    TEAM_MEMBERS,
    TW_QUADRANTS,
)
from utils import (
    build_active_tech_areas,
    build_submission_record,
    ratings_from_sliders,
    record_to_json_preview,
    validate_record,
)

logging.basicConfig(level=logging.INFO, format="%(levelname)s  %(name)s  %(message)s")
log = logging.getLogger(__name__)

# ── Constants ─────────────────────────────────────────────────────────────────

_N_DEFAULT = len(DEFAULT_TECH_AREAS)
_DROPDOWN_CHOICES = ["— Select your name —"] + sorted(TEAM_MEMBERS) + ["Other (type below)"]
_DIM_CHOICES      = list(DIMENSIONS.keys())
_DIM_LABELS       = list(DIMENSIONS.values())


# ─────────────────────────────────────────────────────────────────────────────
# Event handlers
# ─────────────────────────────────────────────────────────────────────────────

def _safe_name(raw: str) -> str:
    """Strip whitespace; return empty string for placeholder selections."""
    if not raw or raw.startswith("—"):
        return ""
    return raw.strip()


def on_researcher_selected(name_raw: str, custom_name: str):
    """
    Triggered when the researcher dropdown changes.

    Loads any existing session from the HF dataset and returns Gradio
    update objects to pre-populate all form components.

    Returns a tuple of 66 values (must match the outputs list below).
    """
    name = _safe_name(name_raw) or _safe_name(custom_name)
    if not name:
        return _default_session_outputs()

    session = storage.load_researcher_session(name)
    if session:
        return _restore_session_outputs(name, session)
    return _default_session_outputs(name=name, msg=f"No previous session for '{name}' — starting fresh.")


def _default_session_outputs(name: str = "", msg: str = ""):
    """
    Return default (blank) output tuple for a new researcher.
    """
    areas        = list(DEFAULT_TECH_AREAS) + [None] * MAX_CUSTOM_AREAS
    interest     = [SLIDER_DEFAULT] * MAX_AREAS
    expertise    = [SLIDER_DEFAULT] * MAX_AREAS
    contribute   = [SLIDER_DEFAULT] * MAX_AREAS
    acc_updates  = [
        gr.update(
            label=DEFAULT_TECH_AREAS[i] if i < _N_DEFAULT else f"Custom Area {i - _N_DEFAULT + 1}",
            visible=(i < _N_DEFAULT),
        )
        for i in range(MAX_AREAS)
    ]
    status = msg or ("Enter your name below and click 'Lookup' to fetch research tags."
                     if not name else "")
    return (
        areas,                                          # → current_areas_state
        "manual",                                       # → scholar_source_state
        status,                                         # → session_status_msg
        gr.update(choices=[], value=[]),                # → tags_cg
        [], "",                                         # → tag_choices_state, tag_input
        "", "", "",                                     # → vision_def, vision_ex, vision_exp
        *acc_updates,                                   # → *area_accordions  (15)
        *[gr.update(value=v) for v in interest],        # → *interest_sliders (15)
        *[gr.update(value=v) for v in expertise],       # → *expertise_sliders(15)
        *[gr.update(value=v) for v in contribute],      # → *contribute_sliders(15)
    )


def _restore_session_outputs(name: str, session: dict):
    """
    Build output tuple from a loaded session dict.
    """
    raw_areas = session.get("tech_areas_used") or list(DEFAULT_TECH_AREAS)
    if isinstance(raw_areas, str):
        try:
            raw_areas = json.loads(raw_areas)
        except Exception:
            raw_areas = list(DEFAULT_TECH_AREAS)

    # Fill to MAX_AREAS slots
    areas: list[Optional[str]] = (list(raw_areas) + [None] * MAX_AREAS)[:MAX_AREAS]

    interest   = [SLIDER_DEFAULT] * MAX_AREAS
    expertise  = [SLIDER_DEFAULT] * MAX_AREAS
    contribute = [SLIDER_DEFAULT] * MAX_AREAS

    acc_updates = []
    for i in range(MAX_AREAS):
        area = areas[i]
        if area:
            interest[i]   = int(session.get(f"area_{i}_interest")  or SLIDER_DEFAULT)
            expertise[i]  = int(session.get(f"area_{i}_expertise") or SLIDER_DEFAULT)
            contribute[i] = int(session.get(f"area_{i}_contribute") or SLIDER_DEFAULT)
            acc_updates.append(gr.update(label=area, visible=True))
        else:
            label = (DEFAULT_TECH_AREAS[i] if i < _N_DEFAULT
                     else f"Custom Area {i - _N_DEFAULT + 1}")
            acc_updates.append(gr.update(label=label, visible=(i < _N_DEFAULT)))

    tags = session.get("scholar_tags") or []
    if isinstance(tags, str):
        try:
            tags = json.loads(tags)
        except Exception:
            tags = []

    source  = session.get("scholar_source", "manual")
    date    = str(session.get("submitted_at", ""))[:10]
    status  = f"Session restored for '{name}' (last saved: {date}). Review and update below."

    return (
        areas,                                          # → current_areas_state
        source,                                         # → scholar_source_state
        status,                                         # → session_status_msg
        gr.update(choices=tags, value=tags),            # → tags_cg
        tags, "",                                       # → tag_choices_state, tag_input
        session.get("vision_definition") or "",         # → vision_def
        session.get("vision_examples")   or "",         # → vision_ex
        session.get("vision_explore")    or "",         # → vision_exp
        *acc_updates,                                   # → *area_accordions  (15)
        *[gr.update(value=v) for v in interest],        # → *interest_sliders (15)
        *[gr.update(value=v) for v in expertise],       # → *expertise_sliders(15)
        *[gr.update(value=v) for v in contribute],      # → *contribute_sliders(15)
    )


def on_scholar_lookup(name_raw: str, custom_name: str, scholar_url: str):
    """
    Triggered by 'Lookup Research Profile' button.

    Queries Semantic Scholar and/or Google Scholar and returns tag choices.
    """
    name = _safe_name(name_raw) or _safe_name(custom_name)
    if not name:
        return "manual", [], gr.update(choices=[], value=[]), "Please enter your name first.", "–"

    result = scholar.lookup_researcher(name, scholar_url or "")
    tags   = result["tags"]
    source = result["source"]
    count  = result.get("paper_count", 0)
    error  = result.get("error", "")

    status = f"Found {len(tags)} tag(s) via {source.replace('_', ' ')}"
    if count:
        status += f"  •  {count} papers indexed"
    if error:
        status += f"  •  Note: {error}"

    return (
        source,                                        # → scholar_source_state
        tags,                                          # → tag_choices_state
        gr.update(choices=tags, value=tags),           # → tags_cg
        status,                                        # → scholar_status_msg
        str(count) if count else "–",                  # → paper_count_txt
    )


def on_add_manual_tag(tag_text: str, current_choices: list):
    """Add a manually typed tag to the CheckboxGroup."""
    tag = (tag_text or "").strip()
    if not tag:
        return current_choices, gr.update(), ""
    new_choices = current_choices + [tag] if tag not in current_choices else current_choices
    return new_choices, gr.update(choices=new_choices, value=new_choices), ""


def on_add_custom_area(area_name: str, current_areas: list):
    """
    Triggered by 'Add Custom Area' button.

    Finds the next empty custom slot (index >= _N_DEFAULT) and activates it.
    Returns updated state + accordion updates.
    """
    area_name = (area_name or "").strip()
    if not area_name:
        return (current_areas, "Please enter an area name.", "") + tuple(gr.update() for _ in range(MAX_AREAS))

    new_areas = list(current_areas)
    acc_updates = [gr.update() for _ in range(MAX_AREAS)]
    added = False

    for i in range(_N_DEFAULT, MAX_AREAS):
        if new_areas[i] is None:
            new_areas[i] = area_name
            acc_updates[i] = gr.update(label=area_name, visible=True)
            added = True
            break

    msg = f"Added '{area_name}' to your assessment." if added else \
          f"All {MAX_CUSTOM_AREAS} custom slots are filled."

    return (new_areas, msg, "") + tuple(acc_updates)


def on_remove_custom_area(current_areas: list):
    """Remove the last non-None custom area slot."""
    new_areas   = list(current_areas)
    acc_updates = [gr.update() for _ in range(MAX_AREAS)]

    for i in range(MAX_AREAS - 1, _N_DEFAULT - 1, -1):
        if new_areas[i] is not None:
            removed = new_areas[i]
            new_areas[i] = None
            acc_updates[i] = gr.update(
                label=f"Custom Area {i - _N_DEFAULT + 1}",
                visible=False,
            )
            return (new_areas, f"Removed '{removed}'.", *acc_updates)

    return (new_areas, "No custom areas to remove.", *acc_updates)


def on_slider_change(current_areas: list, *slider_vals):
    """
    Triggered by any slider change — rebuilds the live radar preview.

    slider_vals order: [interest_0..14, expertise_0..14, contribute_0..14]
    """
    n = MAX_AREAS
    interest_vals   = list(slider_vals[:n])
    expertise_vals  = list(slider_vals[n:2 * n])
    contribute_vals = list(slider_vals[2 * n:])

    ratings    = ratings_from_sliders(current_areas, interest_vals, expertise_vals, contribute_vals)
    active     = build_active_tech_areas(current_areas)
    active_rat = {a: ratings[a] for a in active if a in ratings}

    return charts.build_realtime_radar_preview(active, active_rat)


def on_generate_preview(
    name_raw: str, custom_name: str,
    tags_value: list,
    scholar_source: str,
    current_areas: list,
    vision_def: str, vision_ex: str, vision_exp: str,
    *slider_vals,
) -> tuple[str, str]:
    """Build JSON preview of the submission record."""
    name = _safe_name(name_raw) or _safe_name(custom_name)
    if not name:
        return "", "Please select or enter your name first."

    n = MAX_AREAS
    ratings = ratings_from_sliders(
        current_areas,
        list(slider_vals[:n]),
        list(slider_vals[n:2 * n]),
        list(slider_vals[2 * n:]),
    )
    active = build_active_tech_areas(current_areas)

    record = build_submission_record(
        name=name,
        scholar_tags=tags_value or [],
        scholar_source=scholar_source or "manual",
        ratings=ratings,
        tech_areas=active,
        vision_definition=vision_def,
        vision_examples=vision_ex,
        vision_explore=vision_exp,
    )

    valid, errors = validate_record(record)
    if not valid:
        return "", "Validation errors:\n• " + "\n• ".join(errors)

    return record_to_json_preview(record), "Preview generated. Review and click Submit."


def on_submit(preview_json: str) -> str:
    """Parse the preview JSON and save to HuggingFace dataset."""
    if not preview_json or not preview_json.strip():
        return "Nothing to submit — generate a preview first."
    try:
        record = json.loads(preview_json)
    except json.JSONDecodeError as exc:
        return f"Preview JSON is malformed: {exc}"

    valid, errors = validate_record(record)
    if not valid:
        return "Cannot submit — validation errors:\n• " + "\n• ".join(errors)

    success, message = storage.upsert_submission(record)
    return ("✓ " if success else "✗ ") + message


def on_dashboard_refresh(dimension: str):
    """Reload all dashboard charts from the HF dataset."""
    df          = storage.get_all_radar_data()
    vision_data = storage.get_all_vision_data()
    summary     = storage.get_submission_summary()

    agg_radar   = charts.build_aggregate_radar(df, dimension)
    heatmap     = charts.build_heatmap(df, dimension)
    contrib_bar = charts.build_contribution_bar(df)
    bubble      = charts.build_interest_vs_expertise_bubble(df)
    voices_md   = charts.build_text_display(vision_data)

    names = summary.get("researchers_list") or []
    n     = summary.get("total_submissions", 0)
    last  = summary.get("last_updated", "—")
    status_md = (
        f"**{n} submission(s)**   •   Last updated: {last}"
        if n else "_No submissions yet._"
    )

    return (
        agg_radar, heatmap, contrib_bar, bubble,
        voices_md, status_md,
        gr.update(choices=names, value=names[0] if names else None),  # → individual researcher dropdown
    )


def on_individual_researcher_change(
    researcher_name: Optional[str],
    dimension: str,
) -> tuple:
    """Update individual radar and dimension comparison when researcher/dim changes."""
    if not researcher_name:
        placeholder = charts.empty_figure("Select a researcher above")
        return placeholder, placeholder

    df = storage.get_all_radar_data()
    if df.empty:
        placeholder = charts.empty_figure("No data yet")
        return placeholder, placeholder

    sub = df[df["researcher"] == researcher_name]
    if sub.empty:
        placeholder = charts.empty_figure(f"No data for {researcher_name}")
        return placeholder, placeholder

    # Rebuild ratings dict from long-form data
    tech_areas = sub["tech_area"].tolist()
    ratings    = {
        row["tech_area"]: {dim: row.get(dim) for dim in DIMENSIONS}
        for _, row in sub.iterrows()
    }

    radar  = charts.build_individual_radar(researcher_name, tech_areas, ratings, dimension or None)
    bars   = charts.build_dimension_comparison(researcher_name, tech_areas, ratings)
    return radar, bars


def on_overlay_update(
    selected_researchers: list,
    dimension: str,
) -> go.Figure:
    """Rebuild the researcher overlay radar when selection changes."""
    df = storage.get_all_radar_data()
    return charts.build_overlay_radar(
        df,
        dimension=dimension or "interest",
        researchers=selected_researchers or None,
    )


def on_export_csv() -> str:
    """Export the full dataset to a temp CSV file for download."""
    path = storage.export_to_csv()
    if not path:
        return None   # gr.File handles None gracefully
    return path


# ─────────────────────────────────────────────────────────────────────────────
# Tab 6 — Technology Radar handlers
# ─────────────────────────────────────────────────────────────────────────────

def _list_custom_areas(df) -> list[str]:
    """Return tech areas in the data that are not assigned to any TW quadrant."""
    return radar_viz.get_custom_areas(df)


def on_tw_view_change(view: str):
    """Show/hide the researcher dropdown based on Group vs Individual view."""
    return gr.update(visible=(view == "Individual"))


def on_tw_radar_refresh(view: str, researcher: Optional[str], dimension: str):
    """
    Reload the TW radar and blip legend.
    - Group view → Technology Radar (collective adoption/readiness per area)
    - Individual view → Skill Radar (personal proficiency for selected researcher)
    """
    df      = storage.get_all_radar_data()
    dim     = dimension or "contribute"
    fig     = radar_viz.build_tw_radar(df, view.lower(), researcher or None, dim)
    legend  = radar_viz.build_blip_legend(fig, view.lower())
    custom  = _list_custom_areas(df)
    custom_md = (
        "**Custom areas** (not yet assigned to a quadrant): " + ", ".join(custom)
        if custom else ""
    )
    summary = storage.get_submission_summary()
    names   = summary.get("researchers_list", [])
    return fig, legend, custom_md, gr.update(choices=names)


# ─────────────────────────────────────────────────────────────────────────────
# Tab 7 — AI Research Assistant handlers
# ─────────────────────────────────────────────────────────────────────────────

def stream_agent_response(agent_name: str, extra: str):
    """
    Stream an AI agent response into gr.Markdown.

    Yields (ollama_status, ai_output) tuples.  The status clears on success;
    shows an error message if Ollama is unreachable.
    """
    ok, msg = ai_agents.check_ollama()
    if not ok:
        yield (
            f"⚠ **Ollama not reachable:** {msg}\n\n"
            "**To start Ollama:**\n```\nollama serve\n```\n"
            f"**To pull the model:**\n```\nollama pull {OLLAMA_MODEL}\n```",
            "",
        )
        return

    df          = storage.get_all_radar_data()
    vision_data = storage.get_all_vision_data()

    fn = ai_agents.AGENT_FUNCTIONS.get(agent_name, ai_agents.stream_synergies)
    for text in fn(df, vision_data, extra or ""):   # type: ignore[operator]
        yield "", text


# ─────────────────────────────────────────────────────────────────────────────
# App layout
# ─────────────────────────────────────────────────────────────────────────────

def build_app() -> gr.Blocks:   # noqa: C901 — intentionally long UI builder
    theme = gr.themes.Soft(
        primary_hue="blue",
        secondary_hue="teal",
        neutral_hue="slate",
        font=[gr.themes.GoogleFont("Inter"), "sans-serif"],
    )

    with gr.Blocks(title=APP_TITLE, theme=theme, css=_CSS) as demo:

        # ── Shared state ─────────────────────────────────────────────────
        # current_areas_state: 15-slot list, None for unused custom slots
        current_areas_state  = gr.State(list(DEFAULT_TECH_AREAS) + [None] * MAX_CUSTOM_AREAS)
        scholar_source_state = gr.State("manual")
        tag_choices_state    = gr.State([])   # all available tag choices

        # ── Header ───────────────────────────────────────────────────────
        gr.Markdown(f"# {APP_TITLE}\n{APP_SUBTITLE}", elem_classes=["app-header"])

        with gr.Tabs() as tabs:

            # ═══════════════════════════════════════════════════════════════
            # TAB 1 — Profile & Scholar Lookup
            # ═══════════════════════════════════════════════════════════════
            with gr.Tab("1 · Profile & Scholar Lookup", id="tab_profile"):

                gr.Markdown(
                    "Select your name to **auto-restore a previous session**, "
                    "then click **Lookup** to fetch research tags from Semantic Scholar "
                    "or Google Scholar.  You can also add tags manually."
                )

                with gr.Row():
                    name_dropdown = gr.Dropdown(
                        choices=_DROPDOWN_CHOICES,
                        value=_DROPDOWN_CHOICES[0],
                        label="Select Researcher",
                        allow_custom_value=False,
                        scale=2,
                    )
                    custom_name_tb = gr.Textbox(
                        label="Or type a name (if not listed above)",
                        placeholder="e.g. Jane Smith",
                        scale=2,
                    )

                session_status_msg = gr.Markdown("_Select your name to begin._", elem_classes=["status-info"])

                with gr.Row():
                    scholar_url_tb = gr.Textbox(
                        label="Google Scholar Profile URL  (optional)",
                        placeholder="https://scholar.google.com/citations?user=XXXX",
                        scale=4,
                    )
                    lookup_btn = gr.Button("Lookup Research Profile", variant="primary", scale=1)

                with gr.Row():
                    scholar_status_msg = gr.Textbox(
                        label="Lookup Status", interactive=False, scale=4
                    )
                    paper_count_txt = gr.Textbox(
                        label="Papers found", interactive=False, scale=1, value="–"
                    )

                gr.Markdown("#### Research Tags")
                gr.Markdown(
                    "Auto-fetched tags appear below — uncheck any you want to exclude, "
                    "or add your own using the field at the right."
                )
                with gr.Row():
                    tags_cg = gr.CheckboxGroup(
                        label="Included tags",
                        choices=[],
                        value=[],
                        interactive=True,
                        scale=4,
                    )
                    with gr.Column(scale=1):
                        tag_input  = gr.Textbox(label="Add custom tag", placeholder="e.g. Federated Learning")
                        add_tag_btn = gr.Button("Add tag", size="sm")

            # ═══════════════════════════════════════════════════════════════
            # TAB 2 — Skill Radar Assessment
            # ═══════════════════════════════════════════════════════════════
            with gr.Tab("2 · Skill Radar Assessment", id="tab_radar"):

                gr.Markdown(
                    "Rate each deep tech area on three dimensions.  "
                    "The radar chart on the right updates as you move the sliders."
                )

                with gr.Row():

                    # ── Left column: sliders ──────────────────────────────
                    with gr.Column(scale=3):

                        # Custom area controls
                        with gr.Row(equal_height=True):
                            custom_area_input = gr.Textbox(
                                label="Add custom tech area",
                                placeholder="e.g. Quantum Computing",
                                scale=3,
                            )
                            add_area_btn    = gr.Button("Add area", variant="secondary", scale=1)
                            remove_area_btn = gr.Button("Remove last", variant="stop", scale=1)

                        area_add_status = gr.Markdown("", elem_classes=["status-info"])

                        # Pre-render all 15 accordion groups
                        area_accordions  : list[gr.Accordion] = []
                        interest_sliders : list[gr.Slider]    = []
                        expertise_sliders: list[gr.Slider]    = []
                        contribute_sliders: list[gr.Slider]   = []

                        for i in range(MAX_AREAS):
                            is_default = i < _N_DEFAULT
                            label      = DEFAULT_TECH_AREAS[i] if is_default else f"Custom Area {i - _N_DEFAULT + 1}"

                            with gr.Accordion(
                                label=label,
                                open=is_default,
                                visible=is_default,
                            ) as acc:
                                int_s = gr.Slider(
                                    SLIDER_MIN, SLIDER_MAX,
                                    value=SLIDER_DEFAULT, step=1,
                                    label="Interest Level  (1 = none  →  5 = very high)",
                                    info="How interested are you in this area?",
                                )
                                exp_s = gr.Slider(
                                    SLIDER_MIN, SLIDER_MAX,
                                    value=SLIDER_DEFAULT, step=1,
                                    label="Current Expertise  (1 = novice  →  5 = expert)",
                                    info="What is your current technical depth?",
                                )
                                con_s = gr.Slider(
                                    SLIDER_MIN, SLIDER_MAX,
                                    value=SLIDER_DEFAULT, step=1,
                                    label="Desire to Contribute  (1 = unlikely  →  5 = strongly)",
                                    info="How much do you want to work in this area?",
                                )

                            area_accordions.append(acc)
                            interest_sliders.append(int_s)
                            expertise_sliders.append(exp_s)
                            contribute_sliders.append(con_s)

                    # ── Right column: live radar ──────────────────────────
                    with gr.Column(scale=2):
                        radar_preview = gr.Plot(
                            value=charts.build_realtime_radar_preview(
                                DEFAULT_TECH_AREAS,
                                {a: {d: SLIDER_DEFAULT for d in DIMENSIONS} for a in DEFAULT_TECH_AREAS},
                            ),
                            label="Live Radar Preview",
                            show_label=True,
                        )
                        gr.Markdown(
                            "<small>The chart shows **all three dimensions** simultaneously. "
                            "Blue = Interest, Green = Expertise, Orange = Contribute.</small>",
                            elem_classes=["legend-note"],
                        )

            # ═══════════════════════════════════════════════════════════════
            # TAB 3 — Deep Tech Vision
            # ═══════════════════════════════════════════════════════════════
            with gr.Tab("3 · Deep Tech Vision", id="tab_vision"):

                gr.Markdown(
                    "Share your personal perspective on deep tech.  "
                    "These open-ended responses are displayed on the group dashboard "
                    "and help the team align on shared goals."
                )

                vision_def_tb = gr.Textbox(
                    label="What do you think deep tech is?",
                    lines=5, max_lines=20,
                    placeholder=(
                        "In my view, deep tech refers to technologies grounded in substantial "
                        "scientific or engineering advances that take years to mature…"
                    ),
                )
                vision_ex_tb = gr.Textbox(
                    label="Give examples of deep tech from your work or industry",
                    lines=5, max_lines=20,
                    placeholder=(
                        "E.g. AI-assisted code analysis for technical debt, "
                        "privacy-preserving federated learning at the edge…"
                    ),
                )
                vision_exp_tb = gr.Textbox(
                    label="Which deep tech area do you most want to explore next?",
                    lines=3, max_lines=10,
                    placeholder="E.g. LLM-driven digital twins for smart manufacturing…",
                )

            # ═══════════════════════════════════════════════════════════════
            # TAB 4 — Submit & Preview
            # ═══════════════════════════════════════════════════════════════
            with gr.Tab("4 · Submit & Preview", id="tab_submit"):

                gr.Markdown(
                    "Click **Generate Preview** to assemble your submission as JSON. "
                    "Review the data, then click **Submit** to save it to the "
                    "shared HuggingFace dataset.  Submitting again will overwrite "
                    "your previous entry (upsert)."
                )

                with gr.Row():
                    preview_btn = gr.Button("Generate Preview", variant="secondary", scale=1)
                    submit_btn  = gr.Button("Submit to Dataset", variant="primary",   scale=1)

                preview_status = gr.Markdown("", elem_classes=["status-info"])

                preview_code = gr.Code(
                    label="Submission Preview (JSON)",
                    language="json",
                    interactive=False,
                    lines=30,
                )

                submit_status = gr.Markdown("", elem_classes=["status-info"])

            # ═══════════════════════════════════════════════════════════════
            # TAB 5 — Group Dashboard
            # ═══════════════════════════════════════════════════════════════
            with gr.Tab("5 · Group Dashboard", id="tab_dashboard"):

                with gr.Row(equal_height=True):
                    refresh_btn = gr.Button("Refresh Dashboard", variant="primary", scale=1)
                    dim_dropdown = gr.Dropdown(
                        choices=list(zip(_DIM_LABELS, _DIM_CHOICES)),
                        value="interest",
                        label="Dimension",
                        scale=1,
                    )
                    export_btn = gr.Button("Export CSV", variant="secondary", scale=1)
                    export_file = gr.File(label="Download CSV", visible=True, scale=1)

                dashboard_status = gr.Markdown("_Click Refresh to load data._")

                with gr.Tabs() as dash_tabs:

                    # ── Sub-tab A: Group Radar ────────────────────────────
                    with gr.Tab("Group Radar"):
                        agg_radar_plot = gr.Plot(
                            value=charts.empty_figure("Click 'Refresh Dashboard' to load"),
                            label="Group Aggregate Radar",
                        )

                    # ── Sub-tab B: Heatmap ────────────────────────────────
                    with gr.Tab("Heatmap"):
                        heatmap_plot = gr.Plot(
                            value=charts.empty_figure("Click 'Refresh Dashboard' to load"),
                            label="Researcher × Area Heatmap",
                        )

                    # ── Sub-tab C: Contribution Priorities ───────────────
                    with gr.Tab("Contribution Priorities"):
                        contrib_bar_plot = gr.Plot(
                            value=charts.empty_figure("Click 'Refresh Dashboard' to load"),
                            label="Most Wanted Contribution Areas",
                        )

                    # ── Sub-tab D: Capability Map ─────────────────────────
                    with gr.Tab("Capability Map"):
                        bubble_plot = gr.Plot(
                            value=charts.empty_figure("Click 'Refresh Dashboard' to load"),
                            label="Interest vs Expertise Bubble Chart",
                        )
                        gr.Markdown(
                            "<small>**Bubble size** = desire to contribute.  "
                            "Areas in the **top-left** quadrant have high expertise but low interest; "
                            "**bottom-right** = high interest but skill gap.</small>",
                            elem_classes=["legend-note"],
                        )

                    # ── Sub-tab E: Individual Comparison ─────────────────
                    with gr.Tab("Individual Comparison"):
                        with gr.Row():
                            individual_dd = gr.Dropdown(
                                choices=[],
                                label="Researcher",
                                scale=2,
                            )
                            gr.Markdown("", scale=3)   # spacer

                        with gr.Row():
                            ind_radar_plot = gr.Plot(
                                value=charts.empty_figure("Select a researcher above"),
                                label="Individual Radar",
                            )
                            ind_bar_plot = gr.Plot(
                                value=charts.empty_figure("Select a researcher above"),
                                label="Dimension Comparison",
                            )

                    # ── Sub-tab F: Overlay Comparison ─────────────────────
                    with gr.Tab("Overlay Comparison"):
                        overlay_dd = gr.Dropdown(
                            choices=[],
                            multiselect=True,
                            label="Select researchers to overlay",
                        )
                        overlay_plot = gr.Plot(
                            value=charts.empty_figure("Select researchers above"),
                            label="Multi-Researcher Overlay Radar",
                        )

                    # ── Sub-tab G: Deep Tech Voices ───────────────────────
                    with gr.Tab("Deep Tech Voices"):
                        voices_md = gr.Markdown("_No responses yet._")

            # ═══════════════════════════════════════════════════════════════
            # TAB 6 — Technology Radar  (ThoughtWorks-style)
            # ═══════════════════════════════════════════════════════════════
            with gr.Tab("6 · Technology Radar", id="tab_tw"):

                gr.Markdown(
                    "**Dual radar:** the **Group** view is a *Technology Radar* — it shows "
                    "where the team collectively sits on each deep tech area (adoption readiness). "
                    "The **Individual** view is a *Skill Radar* — it maps a single person's "
                    "proficiency across all areas.  "
                    "Ring placement is driven by the average of *Expertise* and *Contribute* scores.\n\n"
                    "_Inspired by [ThoughtWorks Technology Radar](https://www.thoughtworks.com/radar) "
                    "and the [Skill Radar adaptation](https://lihsmi.ch/learning/2015/04/25/skill-radar-technology-radar.html)._"
                )

                with gr.Row(equal_height=True):
                    tw_view_radio    = gr.Radio(
                        choices=["Group", "Individual"],
                        value="Group",
                        label="View mode",
                        info="Group = Technology Radar  |  Individual = Skill Radar",
                        scale=2,
                    )
                    tw_researcher_dd = gr.Dropdown(
                        choices=[],
                        label="Researcher (Individual view)",
                        visible=False,
                        scale=2,
                    )
                    tw_dim_dd = gr.Dropdown(
                        choices=list(zip(_DIM_LABELS, _DIM_CHOICES)),
                        value="contribute",
                        label="Ring dimension",
                        info="Which score drives ring placement",
                        scale=2,
                    )
                    tw_refresh_btn = gr.Button("Refresh Radar", variant="primary", scale=1)

                tw_radar_plot = gr.Plot(
                    value=radar_viz.empty_tw_figure(),
                    label="Technology / Skill Radar",
                )

                tw_blip_legend_md = gr.Markdown(
                    "_Blip legend appears here after refresh._",
                    label="Blip index",
                )
                tw_custom_areas_md = gr.Markdown("")
                gr.Markdown(radar_viz.build_tw_legend_table())

            # ═══════════════════════════════════════════════════════════════
            # TAB 7 — AI Research Assistant
            # ═══════════════════════════════════════════════════════════════
            with gr.Tab("7 · AI Research Assistant", id="tab_ai"):

                gr.Markdown(
                    f"Powered by **Ollama** (`{OLLAMA_MODEL}`).  "
                    "The agent reads the live radar data and vision responses, then generates "
                    "actionable insights — synergies, project ideas, funding proposals, skill gaps, "
                    "or a competence roadmap.\n\n"
                    "**Before using:** run `ollama serve` and `ollama pull gemma3` in a terminal."
                )

                with gr.Row(equal_height=True):
                    agent_dd = gr.Dropdown(
                        choices=list(ai_agents.AGENT_FUNCTIONS.keys()),
                        value="Team Synergies",
                        label="Agent",
                        scale=3,
                    )
                    ai_run_btn = gr.Button("Run Agent", variant="primary", scale=1)
                    ai_clr_btn = gr.Button("Clear", variant="secondary", scale=1)

                with gr.Accordion("Agent descriptions", open=False):
                    gr.Markdown(
                        "| Agent | What it does |\n"
                        "|---|---|\n"
                        "| **Team Synergies** | Identifies researcher pairs/teams with complementary strengths |\n"
                        "| **Project Ideas** | Generates 5 fundable ideas (EU Horizon / NRC framing) |\n"
                        "| **Project Proposal** | Writes a structured 1-page proposal — paste an idea title below |\n"
                        "| **Skill Gap Analysis** | Finds interest–expertise gaps and bus-factor risks |\n"
                        "| **Competence Roadmap** | 12-month plan for moving skills up a ring |"
                    )

                extra_input = gr.Textbox(
                    label="Additional context  (optional)",
                    placeholder=(
                        'For "Project Ideas": specify focus areas.  '
                        'For "Project Proposal": paste the idea title here.'
                    ),
                    lines=2,
                )

                ollama_status = gr.Markdown("", elem_classes=["status-info"])
                ai_output     = gr.Markdown(
                    "_Select an agent and click **Run Agent** to see insights._"
                )

        # ─────────────────────────────────────────────────────────────────
        # Event wiring
        # ─────────────────────────────────────────────────────────────────

        # ── Shared: all session-restore outputs ──────────────────────────
        # Must match _default_session_outputs() / _restore_session_outputs() tuples:
        # 1  current_areas_state
        # 2  scholar_source_state
        # 3  session_status_msg
        # 4  tags_cg
        # 5  tag_choices_state
        # 6  tag_input              (clear on restore)
        # 7  vision_def_tb
        # 8  vision_ex_tb
        # 9  vision_exp_tb
        # 10-24  area_accordions    (15)
        # 25-39  interest_sliders   (15)
        # 40-54  expertise_sliders  (15)
        # 55-69  contribute_sliders (15)

        _session_outputs = [
            current_areas_state,
            scholar_source_state,
            session_status_msg,
            tags_cg,
            tag_choices_state,
            tag_input,
            vision_def_tb, vision_ex_tb, vision_exp_tb,
            *area_accordions,
            *interest_sliders,
            *expertise_sliders,
            *contribute_sliders,
        ]

        # Researcher selection → restore session
        name_dropdown.change(
            fn=on_researcher_selected,
            inputs=[name_dropdown, custom_name_tb],
            outputs=_session_outputs,
        )

        # ── Tab 1: Scholar lookup ─────────────────────────────────────────
        lookup_btn.click(
            fn=on_scholar_lookup,
            inputs=[name_dropdown, custom_name_tb, scholar_url_tb],
            outputs=[scholar_source_state, tag_choices_state, tags_cg,
                     scholar_status_msg, paper_count_txt],
        )

        add_tag_btn.click(
            fn=on_add_manual_tag,
            inputs=[tag_input, tag_choices_state],
            outputs=[tag_choices_state, tags_cg, tag_input],
        )

        # ── Tab 2: Custom area management ────────────────────────────────
        add_area_btn.click(
            fn=on_add_custom_area,
            inputs=[custom_area_input, current_areas_state],
            outputs=[current_areas_state, area_add_status, custom_area_input,
                     *area_accordions],
        )

        remove_area_btn.click(
            fn=on_remove_custom_area,
            inputs=[current_areas_state],
            outputs=[current_areas_state, area_add_status, *area_accordions],
        )

        # ── Tab 2: Live radar (fires on every slider change) ─────────────
        all_sliders = interest_sliders + expertise_sliders + contribute_sliders
        gr.on(
            triggers=[s.change for s in all_sliders],
            fn=on_slider_change,
            inputs=[current_areas_state] + all_sliders,
            outputs=[radar_preview],
        )

        # ── Tab 4: Preview and submit ─────────────────────────────────────
        _preview_inputs = [
            name_dropdown, custom_name_tb,
            tags_cg, scholar_source_state,
            current_areas_state,
            vision_def_tb, vision_ex_tb, vision_exp_tb,
            *interest_sliders, *expertise_sliders, *contribute_sliders,
        ]

        preview_btn.click(
            fn=on_generate_preview,
            inputs=_preview_inputs,
            outputs=[preview_code, preview_status],
        )

        submit_btn.click(
            fn=on_submit,
            inputs=[preview_code],
            outputs=[submit_status],
        )

        # ── Tab 5: Dashboard refresh ─────────────────────────────────────
        refresh_btn.click(
            fn=on_dashboard_refresh,
            inputs=[dim_dropdown],
            outputs=[
                agg_radar_plot, heatmap_plot, contrib_bar_plot, bubble_plot,
                voices_md, dashboard_status,
                individual_dd,
            ],
        )

        # Dimension selector changes group-level charts
        dim_dropdown.change(
            fn=on_dashboard_refresh,
            inputs=[dim_dropdown],
            outputs=[
                agg_radar_plot, heatmap_plot, contrib_bar_plot, bubble_plot,
                voices_md, dashboard_status,
                individual_dd,
            ],
        )

        # Individual researcher charts
        individual_dd.change(
            fn=on_individual_researcher_change,
            inputs=[individual_dd, dim_dropdown],
            outputs=[ind_radar_plot, ind_bar_plot],
        )

        # Overlay radar
        overlay_dd.change(
            fn=on_overlay_update,
            inputs=[overlay_dd, dim_dropdown],
            outputs=[overlay_plot],
        )

        # Export CSV
        export_btn.click(
            fn=on_export_csv,
            inputs=[],
            outputs=[export_file],
        )

        # ── Tab 6: Technology / Skill Radar ───────────────────────────────
        tw_refresh_btn.click(
            fn=on_tw_radar_refresh,
            inputs=[tw_view_radio, tw_researcher_dd, tw_dim_dd],
            outputs=[tw_radar_plot, tw_blip_legend_md, tw_custom_areas_md, tw_researcher_dd],
        )

        tw_view_radio.change(
            fn=on_tw_view_change,
            inputs=[tw_view_radio],
            outputs=[tw_researcher_dd],
        )

        # Re-render when researcher or dimension changes (Individual view)
        tw_researcher_dd.change(
            fn=on_tw_radar_refresh,
            inputs=[tw_view_radio, tw_researcher_dd, tw_dim_dd],
            outputs=[tw_radar_plot, tw_blip_legend_md, tw_custom_areas_md, tw_researcher_dd],
        )

        tw_dim_dd.change(
            fn=on_tw_radar_refresh,
            inputs=[tw_view_radio, tw_researcher_dd, tw_dim_dd],
            outputs=[tw_radar_plot, tw_blip_legend_md, tw_custom_areas_md, tw_researcher_dd],
        )

        # ── Tab 7: AI Research Assistant ──────────────────────────────────
        ai_run_btn.click(
            fn=stream_agent_response,
            inputs=[agent_dd, extra_input],
            outputs=[ollama_status, ai_output],
        )

        ai_clr_btn.click(
            fn=lambda: ("", "_Select an agent and click **Run Agent** to see insights._"),
            inputs=[],
            outputs=[ollama_status, ai_output],
        )

    return demo


# ─────────────────────────────────────────────────────────────────────────────
# Styling
# ─────────────────────────────────────────────────────────────────────────────

_CSS = """
.app-header {
    text-align: center;
    margin-bottom: 0.5rem;
}
.app-header h1 {
    font-size: 1.9rem;
    color: #1a3a5c;
}
.status-info {
    font-size: 0.88rem;
    color: #555;
    padding: 0.2rem 0;
    min-height: 1.2rem;
}
.legend-note {
    font-size: 0.82rem;
    color: #666;
    padding-top: 0.4rem;
}
"""


# ─────────────────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    share_flag = "--share" in sys.argv

    # Authenticate with HuggingFace (non-fatal if token missing)
    storage.authenticate_hf()

    app = build_app()
    app.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=share_flag,
        show_error=True,
    )
