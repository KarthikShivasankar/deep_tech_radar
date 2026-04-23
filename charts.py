"""
charts.py — All Plotly chart builders for the Skill Radar dashboard.

Every function is pure (no I/O, no side effects).  They accept either a
pandas DataFrame or structured dicts and return a plotly Figure ready for
gr.Plot.  The empty_figure() placeholder is used whenever data is absent
so the UI never shows a broken chart.

Extension note:
  - To add a new chart, define a new build_* function here and call it
    from app.py — no other files need changing.
  - To change colour scales, edit DIMENSION_COLORS in config.py.
"""

import logging
from typing import Optional

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

from config import (
    DEFAULT_TECH_AREAS,
    DIMENSION_COLORS,
    DIMENSIONS,
    SLIDER_DEFAULT,
    SLIDER_MAX,
    SLIDER_MIN,
)

log = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# Shared utilities
# ─────────────────────────────────────────────────────────────────────────────

def empty_figure(message: str = "No data yet") -> go.Figure:
    """Return a blank placeholder Figure with a centred message."""
    fig = go.Figure()
    fig.add_annotation(
        text=message,
        xref="paper", yref="paper",
        x=0.5, y=0.5,
        showarrow=False,
        font=dict(size=16, color="#888"),
    )
    fig.update_layout(
        xaxis_visible=False,
        yaxis_visible=False,
        paper_bgcolor="rgba(0,0,0,0)",
        margin=dict(t=40, b=40, l=40, r=40),
    )
    return fig


def _close_polygon(
    values: list[float],
    labels: list[str],
) -> tuple[list[float], list[str]]:
    """Append the first element to close a radar polygon."""
    return values + [values[0]], labels + [labels[0]]


_FONT = dict(family="Inter, system-ui, sans-serif", size=13, color="#1a2332")


def _radar_layout(title: str, height: int = 480) -> dict:
    """Shared layout dict for all radar/spider charts."""
    return dict(
        title=dict(text=title, font=dict(family="Inter, sans-serif", size=16, color="#1a2332"), x=0.5),
        font=_FONT,
        polar=dict(
            radialaxis=dict(
                visible=True,
                range=[SLIDER_MIN - 0.3, SLIDER_MAX + 0.3],
                tickvals=list(range(SLIDER_MIN, SLIDER_MAX + 1)),
                ticktext=["1", "2", "3", "4", "5"],
                tickfont=dict(size=9, color="#888"),
                gridcolor="rgba(0,0,0,0.07)",
                linecolor="rgba(0,0,0,0.12)",
            ),
            angularaxis=dict(
                tickfont=dict(size=11, color="#333"),
                linecolor="rgba(0,0,0,0.12)",
                gridcolor="rgba(0,0,0,0.07)",
            ),
            bgcolor="rgba(248,250,252,0.8)",
        ),
        showlegend=True,
        legend=dict(
            orientation="h",
            yanchor="bottom", y=-0.30,
            xanchor="center", x=0.5,
            font=dict(size=12),
            bgcolor="rgba(255,255,255,0.8)",
            bordercolor="rgba(0,0,0,0.1)",
            borderwidth=1,
        ),
        height=height,
        margin=dict(t=72, b=100, l=72, r=72),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
    )


# ─────────────────────────────────────────────────────────────────────────────
# Individual researcher radar
# ─────────────────────────────────────────────────────────────────────────────

def build_individual_radar(
    researcher_name: str,
    tech_areas: list[str],
    ratings: dict[str, dict[str, int]],
    dimension: Optional[str] = None,
) -> go.Figure:
    """
    Radar chart for a single researcher.

    Args:
        researcher_name:  Display name for the chart title.
        tech_areas:       Ordered list of active area names (spoke labels).
        ratings:          {area: {dim_key: value (1-5)}}.
        dimension:        If None, all dimensions are overlaid.
                          If a DIMENSIONS key, only that trace is shown.

    Returns:
        Plotly Figure with Scatterpolar trace(s).
    """
    if not tech_areas or not ratings:
        return empty_figure(f"No data for {researcher_name}")

    dims_to_plot = list(DIMENSIONS.keys()) if dimension is None else [dimension]
    fig = go.Figure()

    for dim in dims_to_plot:
        values = [float(ratings.get(a, {}).get(dim) or 0) for a in tech_areas]
        vals_c, areas_c = _close_polygon(values, list(tech_areas))
        fill, line = DIMENSION_COLORS.get(dim, ("rgba(128,128,128,0.15)", "rgb(128,128,128)"))
        fig.add_trace(go.Scatterpolar(
            r=vals_c,
            theta=areas_c,
            fill="toself",
            fillcolor=fill,
            line=dict(color=line, width=2.5),
            name=DIMENSIONS.get(dim, dim),
            hovertemplate="%{theta}: <b>%{r}</b><extra></extra>",
        ))

    dim_label = DIMENSIONS.get(dimension, "All Dimensions") if dimension else "All Dimensions"
    fig.update_layout(**_radar_layout(f"{researcher_name} — {dim_label}"))
    return fig


# ─────────────────────────────────────────────────────────────────────────────
# Live preview radar (updates on every slider change)
# ─────────────────────────────────────────────────────────────────────────────

def build_realtime_radar_preview(
    tech_areas: list[str],
    slider_values: dict[str, dict[str, int]],
) -> go.Figure:
    """
    Lightweight radar for the live preview pane in Tab 2.

    Shows all dimensions as overlapping semi-transparent traces.
    Optimised for fast rendering — no external calls.

    Args:
        tech_areas:    Active (non-None) area names in order.
        slider_values: {area: {dim_key: value}} — same as the ratings dict.
    """
    if not tech_areas:
        return empty_figure("Rate at least one area to see the radar preview")

    fig = go.Figure()
    for dim in DIMENSIONS:
        values = [float(slider_values.get(a, {}).get(dim, SLIDER_DEFAULT)) for a in tech_areas]
        vals_c, areas_c = _close_polygon(values, list(tech_areas))
        fill, line = DIMENSION_COLORS[dim]
        fig.add_trace(go.Scatterpolar(
            r=vals_c,
            theta=areas_c,
            fill="toself",
            fillcolor=fill,
            line=dict(color=line, width=2),
            name=DIMENSIONS[dim],
            hovertemplate="%{theta}: <b>%{r}</b><extra></extra>",
        ))

    fig.update_layout(**_radar_layout("Your Skill Radar (live preview)", height=400))
    return fig


# ─────────────────────────────────────────────────────────────────────────────
# Group aggregate radar
# ─────────────────────────────────────────────────────────────────────────────

def build_aggregate_radar(
    df: pd.DataFrame,
    dimension: str = "interest",
    tech_areas: Optional[list[str]] = None,
) -> go.Figure:
    """
    Aggregate radar: group mean + shaded ±1 std deviation band.

    Args:
        df:          Output of storage.get_all_radar_data().
        dimension:   DIMENSIONS key to aggregate.
        tech_areas:  Ordered area list (defaults to DEFAULT_TECH_AREAS).
    """
    if df.empty or dimension not in df.columns:
        return empty_figure("No group data yet")

    areas = tech_areas or DEFAULT_TECH_AREAS
    group   = df.groupby("tech_area")[dimension]
    means   = group.mean().reindex(areas).fillna(0)
    stds    = group.std().reindex(areas).fillna(0)
    upper   = (means + stds).clip(SLIDER_MIN, SLIDER_MAX)
    lower   = (means - stds).clip(SLIDER_MIN, SLIDER_MAX)

    fill, line = DIMENSION_COLORS.get(dimension, ("rgba(128,128,128,0.15)", "rgb(128,128,128)"))
    band_fill  = fill.replace("0.18", "0.07")

    fig = go.Figure()

    # ── Std-dev band ──────────────────────────────────────────────────────
    upper_v, upper_a = _close_polygon(upper.tolist(), areas)
    lower_v, lower_a = _close_polygon(lower.tolist(), areas)
    fig.add_trace(go.Scatterpolar(
        r=upper_v, theta=upper_a,
        mode="lines", line=dict(color="rgba(0,0,0,0)"),
        showlegend=False, hoverinfo="skip",
    ))
    fig.add_trace(go.Scatterpolar(
        r=lower_v, theta=lower_a,
        fill="tonext", fillcolor=band_fill,
        mode="lines", line=dict(color="rgba(0,0,0,0)"),
        name="±1 std dev",
        hoverinfo="skip",
    ))

    # ── Mean line ─────────────────────────────────────────────────────────
    mean_v, mean_a = _close_polygon(means.tolist(), areas)
    fig.add_trace(go.Scatterpolar(
        r=mean_v, theta=mean_a,
        fill="toself", fillcolor=fill,
        line=dict(color=line, width=3),
        name=f"Group mean — {DIMENSIONS.get(dimension, dimension)}",
        hovertemplate="%{theta}: <b>%{r:.2f}</b><extra></extra>",
    ))

    n = df["researcher"].nunique()
    fig.update_layout(**_radar_layout(
        f"Group Aggregate — {DIMENSIONS.get(dimension, dimension)}"
        f"  (n = {n})"
    ))
    return fig


# ─────────────────────────────────────────────────────────────────────────────
# Multi-researcher overlay radar
# ─────────────────────────────────────────────────────────────────────────────

def build_overlay_radar(
    df: pd.DataFrame,
    dimension: str = "interest",
    researchers: Optional[list[str]] = None,
    tech_areas: Optional[list[str]] = None,
) -> go.Figure:
    """
    Overlay individual researcher traces on a single radar (one dim).

    Useful for comparing a subset of people side-by-side.

    Args:
        df:           Output of storage.get_all_radar_data().
        dimension:    DIMENSIONS key.
        researchers:  Subset of researcher names to include (None = all).
        tech_areas:   Area order (defaults to DEFAULT_TECH_AREAS).
    """
    if df.empty or dimension not in df.columns:
        return empty_figure("No data to compare")

    areas = tech_areas or DEFAULT_TECH_AREAS
    names = researchers or sorted(df["researcher"].unique().tolist())
    palette = px.colors.qualitative.Plotly

    fig = go.Figure()
    for idx, name in enumerate(names):
        sub = df[df["researcher"] == name].set_index("tech_area")
        values = [float(sub.loc[a, dimension]) if a in sub.index else 0.0 for a in areas]
        vals_c, areas_c = _close_polygon(values, list(areas))
        color = palette[idx % len(palette)]
        fig.add_trace(go.Scatterpolar(
            r=vals_c, theta=areas_c,
            fill="toself",
            fillcolor=color.replace("rgb", "rgba").replace(")", ", 0.08)"),
            line=dict(color=color, width=2),
            name=name,
            hovertemplate=f"<b>{name}</b><br>%{{theta}}: %{{r}}<extra></extra>",
        ))

    fig.update_layout(**_radar_layout(
        f"Researcher Comparison — {DIMENSIONS.get(dimension, dimension)}"
    ))
    return fig


# ─────────────────────────────────────────────────────────────────────────────
# Heatmap
# ─────────────────────────────────────────────────────────────────────────────

def build_heatmap(
    df: pd.DataFrame,
    dimension: str = "interest",
) -> go.Figure:
    """
    Researcher × tech-area heatmap coloured by one dimension.

    Rows are sorted alphabetically; columns follow DEFAULT_TECH_AREAS order
    with any custom areas appended at the right.
    """
    if df.empty or dimension not in df.columns:
        return empty_figure("No data for heatmap")

    try:
        pivot = df.pivot_table(
            index="researcher",
            columns="tech_area",
            values=dimension,
            aggfunc="mean",
        )
        # Order columns: default areas first, then any extras
        ordered_cols = (
            [a for a in DEFAULT_TECH_AREAS if a in pivot.columns]
            + [a for a in pivot.columns if a not in DEFAULT_TECH_AREAS]
        )
        pivot = pivot.reindex(columns=ordered_cols)

        text_arr = pivot.round(1).astype(str).replace("nan", "–")

        fig = go.Figure(go.Heatmap(
            z=pivot.values,
            x=pivot.columns.tolist(),
            y=pivot.index.tolist(),
            colorscale="RdYlGn",
            zmin=SLIDER_MIN,
            zmax=SLIDER_MAX,
            text=text_arr.values,
            texttemplate="<b>%{text}</b>",
            textfont=dict(size=11),
            hovertemplate="<b>%{y}</b><br>%{x}<br>Score: <b>%{z:.1f}</b><extra></extra>",
            colorbar=dict(
                title=dict(text=DIMENSIONS.get(dimension, dimension), side="right", font=dict(size=12)),
                thickness=16,
                len=0.8,
                tickfont=dict(size=10),
            ),
        ))

        n_researchers = len(pivot.index)
        fig.update_layout(
            title=dict(
                text=f"Heatmap — {DIMENSIONS.get(dimension, dimension)}",
                font=dict(family="Inter, sans-serif", size=16, color="#1a2332"), x=0.5,
            ),
            font=_FONT,
            height=max(380, 54 * n_researchers + 160),
            xaxis=dict(tickangle=-40, tickfont=dict(size=10), side="bottom"),
            yaxis=dict(tickfont=dict(size=11), autorange="reversed"),
            margin=dict(t=70, b=170, l=190, r=80),
            paper_bgcolor="rgba(0,0,0,0)",
        )
        return fig

    except Exception as exc:
        log.error("Heatmap build error: %s", exc)
        return empty_figure(f"Chart error: {exc}")


# ─────────────────────────────────────────────────────────────────────────────
# Contribution priority bar
# ─────────────────────────────────────────────────────────────────────────────

def build_contribution_bar(df: pd.DataFrame) -> go.Figure:
    """
    Horizontal bar: total group 'contribute' score per tech area, sorted desc.

    The bar length represents collective desire to contribute to each area —
    a useful signal for research priority-setting.
    """
    if df.empty or "contribute" not in df.columns:
        return empty_figure("No contribution data yet")

    totals = (
        df.groupby("tech_area")["contribute"]
        .sum()
        .dropna()
        .sort_values(ascending=True)
    )
    if totals.empty:
        return empty_figure("No contribution scores found")

    fig = px.bar(
        x=totals.values,
        y=totals.index,
        orientation="h",
        color=totals.values,
        color_continuous_scale="Teal",
        labels={"x": "Total Group Contribution Score", "y": ""},
        title="Most Wanted Contribution Areas",
    )
    fig.update_layout(
        coloraxis_showscale=False,
        height=max(350, 46 * len(totals) + 100),
        margin=dict(t=60, b=60, l=210, r=60),
        yaxis=dict(tickfont=dict(size=11)),
        xaxis=dict(tickfont=dict(size=10)),
        title=dict(font=dict(size=15), x=0.5),
        paper_bgcolor="rgba(0,0,0,0)",
    )
    return fig


# ─────────────────────────────────────────────────────────────────────────────
# Per-researcher dimension comparison bar
# ─────────────────────────────────────────────────────────────────────────────

def build_dimension_comparison(
    researcher_name: str,
    tech_areas: list[str],
    ratings: dict[str, dict[str, int]],
) -> go.Figure:
    """
    Grouped horizontal bar: one researcher, all dimensions, all tech areas.

    Shows gaps between interest and expertise, or areas where contribute
    intent exceeds current skills — useful for individual coaching.
    """
    if not tech_areas or not ratings:
        return empty_figure(f"No data for {researcher_name}")

    records = []
    for area in tech_areas:
        for dim, label in DIMENSIONS.items():
            records.append({
                "Tech Area": area,
                "Dimension": label,
                "Score":     float(ratings.get(area, {}).get(dim) or 0),
            })

    plot_df    = pd.DataFrame(records)
    dim_colors = [DIMENSION_COLORS[d][1] for d in DIMENSIONS]

    fig = px.bar(
        plot_df,
        x="Score", y="Tech Area",
        color="Dimension",
        orientation="h",
        barmode="group",
        color_discrete_sequence=dim_colors,
        title=f"{researcher_name} — All Dimensions",
        range_x=[0, SLIDER_MAX + 0.3],
    )
    fig.update_layout(
        height=max(400, 62 * len(tech_areas) + 120),
        margin=dict(t=70, b=60, l=210, r=60),
        legend=dict(orientation="h", yanchor="bottom", y=-0.22, xanchor="center", x=0.5),
        title=dict(font=dict(size=15), x=0.5),
        paper_bgcolor="rgba(0,0,0,0)",
    )
    return fig


# ─────────────────────────────────────────────────────────────────────────────
# Bubble chart: group interest vs. expertise per tech area
# ─────────────────────────────────────────────────────────────────────────────

def build_interest_vs_expertise_bubble(df: pd.DataFrame) -> go.Figure:
    """
    Scatter/bubble chart: group-mean interest (x) vs. expertise (y) per area.

    Bubble size encodes group-mean contribution desire.
    Quadrants reveal capability gaps (high interest, low expertise) and
    hidden strengths (high expertise, low interest).
    """
    if df.empty:
        return empty_figure("No data for bubble chart")

    required = {"interest", "expertise", "contribute"}
    if not required.issubset(df.columns):
        return empty_figure("Missing dimension columns")

    agg = (
        df.groupby("tech_area")[list(required)]
        .mean()
        .reset_index()
        .dropna()
    )
    if agg.empty:
        return empty_figure("No aggregated data")

    fig = px.scatter(
        agg,
        x="interest",
        y="expertise",
        size="contribute",
        text="tech_area",
        size_max=55,
        color="contribute",
        color_continuous_scale="Teal",
        labels={
            "interest":   DIMENSIONS.get("interest",   "Interest"),
            "expertise":  DIMENSIONS.get("expertise",  "Expertise"),
            "contribute": DIMENSIONS.get("contribute", "Contribute"),
        },
        title="Group Capability Map (Interest vs Expertise)",
        range_x=[SLIDER_MIN - 0.3, SLIDER_MAX + 0.3],
        range_y=[SLIDER_MIN - 0.3, SLIDER_MAX + 0.3],
    )
    # Quadrant reference lines at the midpoint
    mid = (SLIDER_MIN + SLIDER_MAX) / 2
    fig.add_hline(y=mid, line_dash="dot", line_color="#ccc", opacity=0.7)
    fig.add_vline(x=mid, line_dash="dot", line_color="#ccc", opacity=0.7)

    mid = (SLIDER_MIN + SLIDER_MAX) / 2
    # Quadrant labels
    for (tx, ty, label) in [
        (SLIDER_MIN + 0.2, SLIDER_MAX - 0.15, "High Interest<br>Low Expertise"),
        (SLIDER_MAX - 0.15, SLIDER_MAX - 0.15, "High Interest<br>High Expertise"),
        (SLIDER_MIN + 0.2, SLIDER_MIN + 0.15, "Low Interest<br>Low Expertise"),
        (SLIDER_MAX - 0.15, SLIDER_MIN + 0.15, "Low Interest<br>High Expertise"),
    ]:
        fig.add_annotation(
            x=tx, y=ty, text=f"<i style='color:#aaa;font-size:9px'>{label}</i>",
            showarrow=False, xref="x", yref="y",
            align="center", font=dict(size=9, color="#aaa"),
        )

    fig.update_traces(textposition="top center", textfont_size=9, marker=dict(opacity=0.85, line=dict(width=1.5, color="#fff")))
    fig.update_layout(
        font=_FONT,
        height=520,
        margin=dict(t=70, b=60, l=80, r=60),
        title=dict(font=dict(family="Inter, sans-serif", size=16, color="#1a2332"), x=0.5),
        coloraxis_colorbar=dict(
            title=dict(text=DIMENSIONS.get("contribute", "Contribute"), font=dict(size=12)),
            thickness=16,
        ),
        paper_bgcolor="rgba(0,0,0,0)",
    )
    return fig


# ─────────────────────────────────────────────────────────────────────────────
# Deep Tech Voices text formatter
# ─────────────────────────────────────────────────────────────────────────────

def build_text_display(vision_data: list[dict]) -> str:
    """
    Format all deep tech vision responses as Markdown for gr.Markdown.

    Args:
        vision_data:  List of {researcher, definition, examples, explore}.

    Returns:
        Markdown string — empty placeholder if no data.
    """
    if not vision_data:
        return "_No vision responses submitted yet. Be the first!_"

    parts: list[str] = []
    for entry in vision_data:
        name = entry.get("researcher") or "Unknown"
        date = (entry.get("submitted_at") or "")[:10]
        header = f"### {name}"
        if date:
            header += f"  <small style='color:#888'>({date})</small>"
        parts.append(header)

        if entry.get("definition"):
            parts.append(f"\n**What is deep tech?**\n> {entry['definition']}\n")
        if entry.get("examples"):
            parts.append(f"**Examples from their work:**\n> {entry['examples']}\n")
        if entry.get("explore"):
            parts.append(f"**Area they want to explore most:**\n> {entry['explore']}\n")
        parts.append("---")

    return "\n".join(parts)
