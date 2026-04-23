"""
app.py — Deep Tech & Skill Radar (Gradio app) — Simplified 3-Click UX.

Tabs:
  1. My Radar           — select name → auto-fetch Scholar → radar preview → submit
  2. Group Dashboard    — aggregate charts; individual comparison; export
  3. Technology Radar   — ThoughtWorks-style quadrant+ring radar
  4. AI Insights        — OpenAI-powered agents: synergies, ideas, proposals

User flow (Scholar path):
  1. Select name from dropdown  →  Scholar data fetched automatically
  2. (Optional) Open "Fine-tune" accordion and adjust sliders
  3. Click "Confirm & Submit"

User flow (no Scholar):
  1. Select name  →  text form appears
  2. Type research description
  3. Click "Auto-generate"  →  radar preview
  4. Click "Confirm & Submit"

Run locally:
    python app.py          # http://localhost:7860
    python app.py --share  # temporary public URL
"""

from __future__ import annotations

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
    OLLAMA_MODEL,
    OPENAI_MODEL,
    SLIDER_DEFAULT,
    SLIDER_MAX,
    SLIDER_MIN,
    TEAM_MEMBERS,
    TW_QUADRANTS,
)
from utils import (
    build_submission_record,
    parse_ratings_from_record,
    validate_record,
)

logging.basicConfig(level=logging.INFO, format="%(levelname)s  %(name)s  %(message)s")
log = logging.getLogger(__name__)

_DROPDOWN_CHOICES = ["— Select your name —"] + sorted(TEAM_MEMBERS) + ["Other (type below)"]
_N_AREAS = len(DEFAULT_TECH_AREAS)


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _safe_name(raw: str) -> str:
    if not raw or raw.startswith("—"):
        return ""
    return raw.strip()


def _default_ratings() -> dict[str, dict[str, int]]:
    return {a: {d: SLIDER_DEFAULT for d in DIMENSIONS} for a in DEFAULT_TECH_AREAS}


def _ratings_to_slider_updates(ratings: dict) -> list:
    """Convert ratings dict to 30 gr.update(value=v) objects: interest×10, expertise×10, contribute×10."""
    updates = []
    for dim in ("interest", "expertise", "contribute"):
        for area in DEFAULT_TECH_AREAS:
            v = ratings.get(area, {}).get(dim, SLIDER_DEFAULT)
            updates.append(gr.update(value=int(v)))
    return updates


# ─────────────────────────────────────────────────────────────────────────────
# Tab 1 event handlers
# ─────────────────────────────────────────────────────────────────────────────

def on_name_selected(name_raw: str, url_map: dict):
    """
    Fires immediately when researcher dropdown changes.
    Auto-fetches Scholar data and populates the radar.
    Returns 37 values: 7 main + 30 slider updates.
    """
    name = _safe_name(name_raw)
    if not name:
        empty_fig = charts.empty_figure("Select your name to see your radar")
        return (
            "_Select your name above to begin._",
            _default_ratings(), "manual", [], empty_fig,
            gr.update(visible=False),   # no_scholar_section
            gr.update(visible=False),   # fine_tune_accordion
            *_ratings_to_slider_updates(_default_ratings()),
        )

    # Try restoring a previous session first
    session = storage.load_researcher_session(name)
    if session:
        ratings  = parse_ratings_from_record(session, DEFAULT_TECH_AREAS)
        date     = str(session.get("submitted_at", ""))[:10]
        tags     = session.get("scholar_tags") or []
        source   = session.get("scholar_source", "manual")
        status   = (
            f"**Previous submission restored** ({date}).  "
            f"Scholar data re-fetched in the background.  "
            f"Click **Confirm & Submit** to save any changes."
        )
        fig = charts.build_realtime_radar_preview(DEFAULT_TECH_AREAS, ratings)
        return (
            status, ratings, source, tags, fig,
            gr.update(visible=False),   # no_scholar_section
            gr.update(visible=True),    # fine_tune_accordion
            *_ratings_to_slider_updates(ratings),
        )

    # No session — fetch from Scholar
    scholar_url = (url_map or {}).get(name, "")
    status_msg  = f"Fetching research profile for **{name}**…"

    result = scholar.lookup_researcher(name, scholar_url)
    tags   = result.get("tags", [])
    source = result.get("source", "manual")

    if tags:
        # Use rich data if available (Semantic Scholar)
        rich = result.get("rich_data")
        if rich and rich.get("paper_area_counts"):
            ratings = scholar.auto_rate_from_rich_data(rich)
        else:
            ratings = scholar.auto_rate_from_tags(tags)

        pc = result.get("paper_count", 0)
        hi = result.get("h_index", 0)
        ci = result.get("citation_count", 0)
        src_label = source.replace("_", " ").title()

        metrics = []
        if pc:  metrics.append(f"{pc} papers")
        if hi:  metrics.append(f"h-index {hi}")
        if ci:  metrics.append(f"{ci} citations")

        status_msg = (
            f"**Auto-generated from {src_label}**"
            + (f" — {', '.join(metrics)}" if metrics else "")
            + ".  Adjust sliders below if needed, then click **Confirm & Submit**."
        )
        has_scholar = True
    else:
        ratings    = _default_ratings()
        status_msg = (
            f"No Scholar profile found for **{name}**.  "
            f"Describe your research interests below and click **Auto-generate**."
        )
        has_scholar = False

    fig = charts.build_realtime_radar_preview(DEFAULT_TECH_AREAS, ratings)
    return (
        status_msg, ratings, source, tags, fig,
        gr.update(visible=not has_scholar),   # no_scholar_section
        gr.update(visible=has_scholar),       # fine_tune_accordion
        *_ratings_to_slider_updates(ratings),
    )


def on_infer_from_text(text: str):
    """Auto-generate ratings from free-text research description."""
    if not text.strip():
        return (
            _default_ratings(),
            charts.empty_figure("Enter your research description first"),
            "Please enter some text about your research interests.",
            gr.update(visible=False),
            *_ratings_to_slider_updates(_default_ratings()),
        )
    ratings  = scholar.infer_ratings_from_text(text)
    fig      = charts.build_realtime_radar_preview(DEFAULT_TECH_AREAS, ratings)
    status   = "**Radar generated from your description.** Adjust sliders if needed, then click **Confirm & Submit**."
    return (
        ratings, fig, status,
        gr.update(visible=True),   # show fine_tune_accordion
        *_ratings_to_slider_updates(ratings),
    )


def on_reinfer_from_scholar(tags: list, url_map: dict, name_raw: str):
    """Reset ratings to Scholar-inferred values (discards manual slider changes)."""
    name = _safe_name(name_raw)
    scholar_url = (url_map or {}).get(name, "")
    result  = scholar.lookup_researcher(name, scholar_url) if name else {}
    fresh_tags = result.get("tags", tags or [])
    rich = result.get("rich_data") if result else None
    if rich and rich.get("paper_area_counts"):
        ratings = scholar.auto_rate_from_rich_data(rich)
    elif fresh_tags:
        ratings = scholar.auto_rate_from_tags(fresh_tags)
    else:
        ratings = _default_ratings()
    fig = charts.build_realtime_radar_preview(DEFAULT_TECH_AREAS, ratings)
    return ratings, fig, *_ratings_to_slider_updates(ratings)


def on_slider_change(current_ratings: dict, *slider_vals):
    """Rebuild ratings from live slider positions and refresh radar preview."""
    n = _N_AREAS
    interest_vals   = list(slider_vals[:n])
    expertise_vals  = list(slider_vals[n:2 * n])
    contribute_vals = list(slider_vals[2 * n:3 * n])

    new_ratings: dict[str, dict[str, int]] = {}
    for i, area in enumerate(DEFAULT_TECH_AREAS):
        new_ratings[area] = {
            "interest":   int(interest_vals[i])   if i < len(interest_vals)   else SLIDER_DEFAULT,
            "expertise":  int(expertise_vals[i])  if i < len(expertise_vals)  else SLIDER_DEFAULT,
            "contribute": int(contribute_vals[i]) if i < len(contribute_vals) else SLIDER_DEFAULT,
        }
    fig = charts.build_realtime_radar_preview(DEFAULT_TECH_AREAS, new_ratings)
    return new_ratings, fig


def on_confirm_submit(
    name_raw: str, scholar_tags: list, scholar_source: str,
    current_ratings: dict,
    vision_def: str, vision_ex: str, vision_exp: str,
):
    """Build full record and save to HuggingFace dataset."""
    name = _safe_name(name_raw)
    if not name:
        return "**Please select your name first.**"
    if not current_ratings:
        return "**No ratings found — please select your name and let the radar load.**"

    record = build_submission_record(
        name=name,
        scholar_tags=scholar_tags or [],
        scholar_source=scholar_source or "manual",
        ratings=current_ratings,
        tech_areas=DEFAULT_TECH_AREAS,
        vision_definition=vision_def or "",
        vision_examples=vision_ex or "",
        vision_explore=vision_exp or "",
    )

    valid, errors = validate_record(record)
    if not valid:
        return "**Validation errors:**\n• " + "\n• ".join(errors)

    success, message = storage.upsert_submission(record)
    return ("**✓ " if success else "**✗ ") + message + "**"


# ─────────────────────────────────────────────────────────────────────────────
# Tab 2 event handlers (Group Dashboard — unchanged)
# ─────────────────────────────────────────────────────────────────────────────

def on_dashboard_refresh(dimension: str):
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
        gr.update(choices=names, value=names[0] if names else None),
    )


def on_individual_researcher_change(researcher_name: Optional[str], dimension: str):
    if not researcher_name:
        placeholder = charts.empty_figure("Select a researcher above")
        return placeholder, placeholder

    df = storage.get_all_radar_data()
    if df.empty:
        return charts.empty_figure("No data yet"), charts.empty_figure("No data yet")

    sub = df[df["researcher"] == researcher_name]
    if sub.empty:
        return charts.empty_figure(f"No data for {researcher_name}"), charts.empty_figure("")

    ratings = {
        row["tech_area"]: {dim: row.get(dim) for dim in DIMENSIONS}
        for _, row in sub.iterrows()
    }
    tech_areas = sub["tech_area"].tolist()
    radar = charts.build_individual_radar(researcher_name, tech_areas, ratings, dimension or None)
    bars  = charts.build_dimension_comparison(researcher_name, tech_areas, ratings)
    return radar, bars


def on_overlay_update(selected_researchers: list, dimension: str):
    df = storage.get_all_radar_data()
    return charts.build_overlay_radar(df, dimension=dimension or "interest", researchers=selected_researchers or None)


def on_export_csv():
    path = storage.export_to_csv()
    return path if path else None


# ─────────────────────────────────────────────────────────────────────────────
# Tab 3 event handlers (Technology Radar — unchanged)
# ─────────────────────────────────────────────────────────────────────────────

def on_tw_view_change(view: str):
    return gr.update(visible=(view == "Individual"))


def on_tw_radar_refresh(view: str, researcher: Optional[str], dimension: str):
    df      = storage.get_all_radar_data()
    dim     = dimension or "contribute"
    fig     = radar_viz.build_tw_radar(df, view.lower(), researcher or None, dim)
    legend  = radar_viz.build_blip_legend(fig, view.lower())
    custom  = radar_viz.get_custom_areas(df)
    custom_md = (
        "**Custom areas** (not yet assigned to a quadrant): " + ", ".join(custom)
        if custom else ""
    )
    summary = storage.get_submission_summary()
    names   = summary.get("researchers_list", [])
    return fig, legend, custom_md, gr.update(choices=names)


# ─────────────────────────────────────────────────────────────────────────────
# Tab 4 event handlers (AI Insights)
# ─────────────────────────────────────────────────────────────────────────────

def stream_agent_response(agent_name: str, extra: str):
    """Stream OpenAI response into gr.Markdown."""
    ok, msg = ai_agents.check_openai()
    if not ok:
        yield f"⚠ **OpenAI not configured:** {msg}\n\nSet `OPENAI_API_KEY` in `.env`.", ""
        return

    df          = storage.get_all_radar_data()
    vision_data = storage.get_all_vision_data()
    fn          = ai_agents.AGENT_FUNCTIONS.get(agent_name, ai_agents.stream_synergies)
    for text in fn(df, vision_data, extra or ""):
        yield "", text


# ─────────────────────────────────────────────────────────────────────────────
# App layout
# ─────────────────────────────────────────────────────────────────────────────

def build_app(url_map: dict) -> gr.Blocks:

    with gr.Blocks(title=APP_TITLE) as demo:

        # ── Shared state ─────────────────────────────────────────────────────
        current_ratings_state = gr.State(_default_ratings())
        scholar_source_state  = gr.State("manual")
        scholar_tags_state    = gr.State([])
        scholar_url_map_state = gr.State(url_map)

        # ── Header ────────────────────────────────────────────────────────────
        gr.Markdown(f"# {APP_TITLE}\n{APP_SUBTITLE}", elem_classes=["app-header"])

        with gr.Tabs():

            # ═══════════════════════════════════════════════════════════════
            # TAB 1 — My Radar (the main submission tab)
            # ═══════════════════════════════════════════════════════════════
            with gr.Tab("1 · My Radar", id="tab_myradar"):

                gr.Markdown(
                    "Select your name — your radar is **auto-generated from your Scholar profile**.  "
                    "Fine-tune the sliders if needed, then click **Confirm & Submit**."
                )

                with gr.Row():
                    name_dropdown = gr.Dropdown(
                        choices=_DROPDOWN_CHOICES,
                        value=_DROPDOWN_CHOICES[0],
                        label="Select Researcher",
                        allow_custom_value=True,
                        scale=3,
                    )

                fetch_status_md = gr.Markdown(
                    "_Select your name above to begin._",
                    elem_classes=["status-info"],
                )

                # ── Live radar preview ────────────────────────────────────
                radar_preview = gr.Plot(
                    value=charts.empty_figure("Select your name to see your radar"),
                    label="Your Skill Radar",
                    elem_classes=["radar-container"],
                )

                # ── No-Scholar fallback (hidden until needed) ─────────────
                with gr.Row(visible=False) as no_scholar_section:
                    with gr.Column():
                        gr.Markdown(
                            "**No Scholar profile found.**  "
                            "Describe your research interests and past work below — "
                            "the radar will be auto-generated from your description."
                        )
                        research_text_tb = gr.Textbox(
                            label="Your research interests & past work",
                            placeholder=(
                                "e.g. I work on machine learning for IoT security, "
                                "specifically federated learning for anomaly detection on edge devices. "
                                "I have a background in software engineering and privacy engineering."
                            ),
                            lines=6,
                        )
                        with gr.Row():
                            infer_text_btn       = gr.Button("Auto-generate radar from description", variant="primary")
                            text_infer_status_md = gr.Markdown("", elem_classes=["status-info"])

                # ── Fine-tune accordion (hidden until Scholar data loads) ──
                with gr.Accordion("Fine-tune ratings  (expand to adjust)", open=False, visible=False) as fine_tune_accordion:

                    with gr.Row():
                        reinfer_btn = gr.Button(
                            "Re-infer from Scholar data",
                            variant="secondary", size="sm",
                        )
                        gr.Markdown(
                            "<small>Sliders are pre-filled from your Scholar profile.  "
                            "Open an area below to override.</small>",
                            elem_classes=["legend-note"],
                        )

                    area_accordions:   list[gr.Accordion] = []
                    interest_sliders:  list[gr.Slider]    = []
                    expertise_sliders: list[gr.Slider]    = []
                    contribute_sliders: list[gr.Slider]   = []

                    for i, area in enumerate(DEFAULT_TECH_AREAS):
                        with gr.Accordion(label=area, open=False) as acc:
                            int_s = gr.Slider(SLIDER_MIN, SLIDER_MAX, value=SLIDER_DEFAULT, step=1,
                                              label="Interest", info="1 = none  →  5 = very high")
                            exp_s = gr.Slider(SLIDER_MIN, SLIDER_MAX, value=SLIDER_DEFAULT, step=1,
                                              label="Expertise", info="1 = novice  →  5 = expert")
                            con_s = gr.Slider(SLIDER_MIN, SLIDER_MAX, value=SLIDER_DEFAULT, step=1,
                                              label="Contribute", info="1 = unlikely  →  5 = strongly")
                        area_accordions.append(acc)
                        interest_sliders.append(int_s)
                        expertise_sliders.append(exp_s)
                        contribute_sliders.append(con_s)

                # ── Optional vision context ───────────────────────────────
                with gr.Accordion("Add research vision context  (optional)", open=False):
                    vision_def_tb = gr.Textbox(
                        label="What do you think deep tech is?",
                        lines=3, max_lines=10,
                        placeholder="In my view, deep tech refers to…",
                    )
                    vision_ex_tb = gr.Textbox(
                        label="Examples of deep tech from your work",
                        lines=3, max_lines=10,
                        placeholder="e.g. AI-assisted code analysis, federated learning at the edge…",
                    )
                    vision_exp_tb = gr.Textbox(
                        label="Which deep tech area do you most want to explore next?",
                        lines=2, max_lines=6,
                        placeholder="e.g. LLM-driven digital twins…",
                    )

                # ── Submit ────────────────────────────────────────────────
                submit_btn = gr.Button(
                    "Confirm & Submit",
                    variant="primary",
                    elem_classes=["submit-btn-large"],
                )
                submit_status_md = gr.Markdown("", elem_classes=["status-info"])

            # ═══════════════════════════════════════════════════════════════
            # TAB 2 — Group Dashboard
            # ═══════════════════════════════════════════════════════════════
            with gr.Tab("2 · Group Dashboard", id="tab_dashboard"):

                with gr.Row(equal_height=True):
                    refresh_btn  = gr.Button("Refresh Dashboard", variant="primary", scale=1)
                    _DIM_CHOICES = list(DIMENSIONS.keys())
                    _DIM_LABELS  = list(DIMENSIONS.values())
                    dim_dropdown = gr.Dropdown(
                        choices=list(zip(_DIM_LABELS, _DIM_CHOICES)),
                        value="interest",
                        label="Dimension",
                        scale=1,
                    )
                    export_btn  = gr.Button("Export CSV", variant="secondary", scale=1)
                    export_file = gr.File(label="Download CSV", visible=True, scale=1)

                dashboard_status = gr.Markdown("_Click Refresh to load data._")

                with gr.Tabs():
                    with gr.Tab("Group Radar"):
                        agg_radar_plot = gr.Plot(
                            value=charts.empty_figure("Click 'Refresh Dashboard' to load"),
                            label="Group Aggregate Radar",
                        )
                    with gr.Tab("Heatmap"):
                        heatmap_plot = gr.Plot(
                            value=charts.empty_figure("Click 'Refresh Dashboard' to load"),
                            label="Researcher × Area Heatmap",
                        )
                    with gr.Tab("Contribution Priorities"):
                        contrib_bar_plot = gr.Plot(
                            value=charts.empty_figure("Click 'Refresh Dashboard' to load"),
                            label="Most Wanted Contribution Areas",
                        )
                    with gr.Tab("Capability Map"):
                        bubble_plot = gr.Plot(
                            value=charts.empty_figure("Click 'Refresh Dashboard' to load"),
                            label="Interest vs Expertise Bubble Chart",
                        )
                    with gr.Tab("Individual Comparison"):
                        with gr.Row():
                            individual_dd = gr.Dropdown(choices=[], label="Researcher", scale=2)
                        with gr.Row():
                            ind_radar_plot = gr.Plot(
                                value=charts.empty_figure("Select a researcher above"),
                                label="Individual Radar",
                            )
                            ind_bar_plot = gr.Plot(
                                value=charts.empty_figure("Select a researcher above"),
                                label="Dimension Comparison",
                            )
                    with gr.Tab("Overlay Comparison"):
                        overlay_dd = gr.Dropdown(
                            choices=[], multiselect=True,
                            label="Select researchers to overlay",
                        )
                        overlay_plot = gr.Plot(
                            value=charts.empty_figure("Select researchers above"),
                            label="Multi-Researcher Overlay Radar",
                        )
                    with gr.Tab("Deep Tech Voices"):
                        voices_md = gr.Markdown("_No responses yet._")

            # ═══════════════════════════════════════════════════════════════
            # TAB 3 — Technology Radar
            # ═══════════════════════════════════════════════════════════════
            with gr.Tab("3 · Technology Radar", id="tab_tw"):

                gr.Markdown(
                    "**Group** view = Technology Radar (collective adoption readiness).  "
                    "**Individual** view = Skill Radar (personal proficiency).  "
                    "Ring placement driven by avg(Expertise + Contribute)."
                )

                with gr.Row(equal_height=True):
                    tw_view_radio = gr.Radio(
                        choices=["Group", "Individual"],
                        value="Group",
                        label="View mode",
                        scale=2,
                    )
                    tw_researcher_dd = gr.Dropdown(
                        choices=[], label="Researcher (Individual view)",
                        visible=False, scale=2,
                    )
                    tw_dim_dd = gr.Dropdown(
                        choices=list(zip(_DIM_LABELS, _DIM_CHOICES)),
                        value="contribute",
                        label="Ring dimension",
                        scale=2,
                    )
                    tw_refresh_btn = gr.Button("Refresh Radar", variant="primary", scale=1)

                tw_radar_plot = gr.Plot(
                    value=radar_viz.empty_tw_figure(),
                    label="Technology / Skill Radar",
                )
                tw_blip_legend_md = gr.Markdown("_Blip legend appears here after refresh._")
                tw_custom_areas_md = gr.Markdown("")
                gr.Markdown(radar_viz.build_tw_legend_table())

            # ═══════════════════════════════════════════════════════════════
            # TAB 4 — AI Insights
            # ═══════════════════════════════════════════════════════════════
            with gr.Tab("4 · AI Insights", id="tab_ai"):

                gr.Markdown(
                    f"Powered by **OpenAI** (`{OPENAI_MODEL}`).  "
                    "The agent reads the live radar data and vision responses, then generates "
                    "actionable insights — synergies, project ideas, funding proposals, "
                    "skill gaps, or a competence roadmap."
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
                        'For "Project Proposal": paste the idea title here.  '
                        'For "Project Ideas": specify focus areas.'
                    ),
                    lines=2,
                )

                openai_status = gr.Markdown("", elem_classes=["status-info"])
                ai_output     = gr.Markdown(
                    "_Select an agent and click **Run Agent** to see insights._"
                )

        # ─────────────────────────────────────────────────────────────────
        # Event wiring
        # ─────────────────────────────────────────────────────────────────

        all_sliders    = interest_sliders + expertise_sliders + contribute_sliders
        _name_outputs  = [
            fetch_status_md,
            current_ratings_state, scholar_source_state, scholar_tags_state,
            radar_preview,
            no_scholar_section,
            fine_tune_accordion,
            *interest_sliders, *expertise_sliders, *contribute_sliders,
        ]

        # Tab 1: name selection → auto-fetch
        name_dropdown.change(
            fn=on_name_selected,
            inputs=[name_dropdown, scholar_url_map_state],
            outputs=_name_outputs,
        )

        # Tab 1: text-based inference (no-Scholar path)
        _text_infer_outputs = [
            current_ratings_state, radar_preview, text_infer_status_md,
            fine_tune_accordion,
            *interest_sliders, *expertise_sliders, *contribute_sliders,
        ]
        infer_text_btn.click(
            fn=on_infer_from_text,
            inputs=[research_text_tb],
            outputs=_text_infer_outputs,
        )

        # Tab 1: re-infer from Scholar (reset sliders)
        _reinfer_outputs = [
            current_ratings_state, radar_preview,
            *interest_sliders, *expertise_sliders, *contribute_sliders,
        ]
        reinfer_btn.click(
            fn=on_reinfer_from_scholar,
            inputs=[scholar_tags_state, scholar_url_map_state, name_dropdown],
            outputs=_reinfer_outputs,
        )

        # Tab 1: live radar updates on any slider change
        gr.on(
            triggers=[s.change for s in all_sliders],
            fn=on_slider_change,
            inputs=[current_ratings_state] + all_sliders,
            outputs=[current_ratings_state, radar_preview],
        )

        # Tab 1: submit
        submit_btn.click(
            fn=on_confirm_submit,
            inputs=[
                name_dropdown, scholar_tags_state, scholar_source_state,
                current_ratings_state,
                vision_def_tb, vision_ex_tb, vision_exp_tb,
            ],
            outputs=[submit_status_md],
        )

        # Tab 2: dashboard refresh
        _dash_outputs = [
            agg_radar_plot, heatmap_plot, contrib_bar_plot, bubble_plot,
            voices_md, dashboard_status, individual_dd,
        ]
        refresh_btn.click(fn=on_dashboard_refresh, inputs=[dim_dropdown], outputs=_dash_outputs)
        dim_dropdown.change(fn=on_dashboard_refresh, inputs=[dim_dropdown], outputs=_dash_outputs)
        individual_dd.change(
            fn=on_individual_researcher_change,
            inputs=[individual_dd, dim_dropdown],
            outputs=[ind_radar_plot, ind_bar_plot],
        )
        overlay_dd.change(
            fn=on_overlay_update,
            inputs=[overlay_dd, dim_dropdown],
            outputs=[overlay_plot],
        )
        export_btn.click(fn=on_export_csv, inputs=[], outputs=[export_file])

        # Tab 3: TW radar
        tw_refresh_btn.click(
            fn=on_tw_radar_refresh,
            inputs=[tw_view_radio, tw_researcher_dd, tw_dim_dd],
            outputs=[tw_radar_plot, tw_blip_legend_md, tw_custom_areas_md, tw_researcher_dd],
        )
        tw_view_radio.change(fn=on_tw_view_change, inputs=[tw_view_radio], outputs=[tw_researcher_dd])
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

        # Tab 4: AI Insights
        ai_run_btn.click(
            fn=stream_agent_response,
            inputs=[agent_dd, extra_input],
            outputs=[openai_status, ai_output],
        )
        ai_clr_btn.click(
            fn=lambda: ("", "_Select an agent and click **Run Agent** to see insights._"),
            inputs=[],
            outputs=[openai_status, ai_output],
        )

    return demo


# ─────────────────────────────────────────────────────────────────────────────
# CSS
# ─────────────────────────────────────────────────────────────────────────────

_CSS = """
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');

body, .gradio-container {
    font-family: 'Inter', system-ui, sans-serif !important;
}
.app-header {
    text-align: center;
    margin-bottom: 0.6rem;
}
.app-header h1 {
    font-size: 2rem;
    font-weight: 700;
    color: #1a2332;
    letter-spacing: -0.02em;
}
.app-header p {
    color: #555;
    font-size: 0.95rem;
}
.status-info {
    font-size: 0.88rem;
    color: #444;
    padding: 0.3rem 0;
    min-height: 1.4rem;
}
.legend-note {
    font-size: 0.82rem;
    color: #666;
    padding-top: 0.3rem;
}
.radar-container {
    border: 1px solid #e8edf3;
    border-radius: 12px;
    padding: 6px;
    background: rgba(248, 250, 252, 0.6);
}
.submit-btn-large button {
    font-size: 1.1rem !important;
    font-weight: 600 !important;
    padding: 0.75rem 2.5rem !important;
    border-radius: 10px !important;
    letter-spacing: 0.01em;
    margin-top: 0.8rem;
}
"""


# ─────────────────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    share_flag = "--share" in sys.argv

    storage.authenticate_hf()

    log.info("Building Google Scholar URL → team member mapping (this may take ~30s)…")
    url_map = scholar.build_scholar_url_mapping()
    log.info("Scholar URL mapping complete: %d entries mapped", sum(1 for v in url_map.values() if v))

    app = build_app(url_map)
    app.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=share_flag,
        show_error=True,
        theme=gr.themes.Soft(
            primary_hue="blue",
            secondary_hue="teal",
            neutral_hue="slate",
            font=[gr.themes.GoogleFont("Inter"), "sans-serif"],
        ),
        css=_CSS,
    )
