"""
radar_viz.py — ThoughtWorks-style Technology & Skill Radar (Plotly implementation).

Layout mirrors the classic TW radar:
  • Four coloured quadrant wedges (AI & Intelligence, Infrastructure & Systems,
    Security & Privacy, Engineering Practices)
  • Four concentric rings (Lead → Contribute → Grow → Watch, centre outward)
  • Blips = dots with number labels; a legend table below names each blip

Ring placement is driven by avg(expertise, contribute) scores from the HF dataset.
Marker size encodes group mean interest level.

Inspiration:
  https://lihsmi.ch/learning/2015/04/25/skill-radar-technology-radar.html
  https://github.com/thoughtworks/build-your-own-radar
"""

from __future__ import annotations

import math
from typing import Optional

import numpy as np
import pandas as pd
import plotly.graph_objects as go

from config import (
    DEFAULT_TECH_AREAS,
    TW_QUADRANTS,
    TW_RING_THRESHOLDS,
    TW_RINGS,
)

# ── Geometry constants ────────────────────────────────────────────────────────

_AXIS_RANGE   = 1.28          # axis limits ± this value
_LABEL_R      = 1.12          # quadrant title radius (just outside outer ring)
_RING_LABEL_R = [0.125, 0.375, 0.625, 0.875]   # midpoints of each ring band
_N_ARC        = 90            # points per arc for smooth wedges


# ─────────────────────────────────────────────────────────────────────────────
# Score → ring mapping
# ─────────────────────────────────────────────────────────────────────────────

def score_to_ring_idx(expertise: float, contribute: float) -> int:
    """
    Map avg(expertise, contribute) to a TW ring index.

    Returns:
        0 = Lead (centre), 1 = Contribute, 2 = Grow, 3 = Watch (outer)
    """
    try:
        avg = (float(expertise) + float(contribute)) / 2
    except (TypeError, ValueError):
        return 3   # NaN / missing → Watch
    for idx, threshold in enumerate(TW_RING_THRESHOLDS):
        if avg >= threshold:
            return idx
    return 3


# ─────────────────────────────────────────────────────────────────────────────
# Geometry helpers
# ─────────────────────────────────────────────────────────────────────────────

def _deg2rad(deg: float) -> float:
    return deg * math.pi / 180


def _wedge_xy(
    r_inner: float,
    r_outer: float,
    theta_start_deg: float,
    theta_end_deg: float,
    n: int = _N_ARC,
) -> tuple[list[float], list[float]]:
    """
    Cartesian (x, y) for a filled annular wedge.
    Outer arc CW + inner arc CCW traces a closed polygon.
    """
    t_out = np.linspace(_deg2rad(theta_start_deg), _deg2rad(theta_end_deg), n)
    t_in  = np.linspace(_deg2rad(theta_end_deg),   _deg2rad(theta_start_deg), n)
    x = np.concatenate([r_outer * np.cos(t_out), r_inner * np.cos(t_in)])
    y = np.concatenate([r_outer * np.sin(t_out), r_inner * np.sin(t_in)])
    return x.tolist(), y.tolist()


def _circle_xy(r: float, n: int = 200) -> tuple[list[float], list[float]]:
    """Full circle of radius r."""
    t = np.linspace(0, 2 * math.pi, n)
    return (r * np.cos(t)).tolist(), (r * np.sin(t)).tolist()


def _blip_xy(
    q_start_deg: float,
    q_end_deg: float,
    area_idx: int,
    n_areas: int,
    ring_idx: int,
    jitter_seed: int = 0,
) -> tuple[float, float]:
    """
    (x, y) for a blip inside a quadrant/ring cell.

    Angular position: evenly space n_areas slots across the quadrant arc;
    area_idx selects slot (idx + 1).
    Radial position: midpoint of the ring band.
    jitter_seed ≠ 0 adds small reproducible noise to separate individual blips.
    """
    slot_deg = (q_end_deg - q_start_deg) / (n_areas + 1)
    angle_deg = q_start_deg + (area_idx + 1) * slot_deg

    ring    = TW_RINGS[ring_idx]
    r_mid   = (ring["r_inner"] + ring["r_outer"]) / 2

    if jitter_seed != 0:
        rng       = np.random.default_rng(jitter_seed)
        angle_deg += float(rng.uniform(-4, 4))
        r_mid     += float(rng.uniform(-0.04, 0.04))

    angle_rad = _deg2rad(angle_deg)
    return r_mid * math.cos(angle_rad), r_mid * math.sin(angle_rad)


# ─────────────────────────────────────────────────────────────────────────────
# Quadrant / area utilities
# ─────────────────────────────────────────────────────────────────────────────

def _area_to_quadrant(area: str) -> Optional[str]:
    """Return the quadrant name for a tech area, or None if custom/unassigned."""
    for q_name, q_data in TW_QUADRANTS.items():
        if area in q_data["areas"]:
            return q_name
    return None


def get_custom_areas(df: pd.DataFrame) -> list[str]:
    """
    Return tech areas present in the radar data that are not in any TW quadrant.
    These appear in a panel below the chart rather than as blips.
    """
    if df.empty or "tech_area" not in df.columns:
        return []
    all_areas    = df["tech_area"].unique().tolist()
    known_areas  = {a for q in TW_QUADRANTS.values() for a in q["areas"]}
    return sorted(a for a in all_areas if a not in known_areas)


# ─────────────────────────────────────────────────────────────────────────────
# Figure scaffold (empty radar with quadrant/ring structure)
# ─────────────────────────────────────────────────────────────────────────────

def _add_scaffold(fig: go.Figure) -> None:
    """
    Draw the static radar structure:
      • Coloured quadrant wedge fills
      • Gray ring boundary circles
      • Quadrant divider lines
      • Ring labels (inner edge of each band, at top of chart)
      • Quadrant title labels (outside outer ring)
    """
    # ── Quadrant coloured fills ───────────────────────────────────────────
    for q_name, q in TW_QUADRANTS.items():
        x, y = _wedge_xy(0, 1.0, q["angle_start"], q["angle_end"])
        fig.add_trace(go.Scatter(
            x=x, y=y,
            fill="toself",
            fillcolor=q["fill"],
            line=dict(width=0),
            mode="lines",
            hoverinfo="skip",
            showlegend=False,
            name=f"_q_{q_name}",
        ))

    # ── Ring boundary circles ─────────────────────────────────────────────
    for ring in TW_RINGS:
        r = ring["r_outer"]
        cx, cy = _circle_xy(r)
        fig.add_trace(go.Scatter(
            x=cx, y=cy,
            mode="lines",
            line=dict(color="#cccccc", width=1),
            hoverinfo="skip",
            showlegend=False,
            name=f"_ring_{ring['name']}",
        ))

    # ── Quadrant divider lines ────────────────────────────────────────────
    for angle_deg in [0, 90, 180, 270]:
        angle_rad = _deg2rad(angle_deg)
        fig.add_shape(
            type="line",
            x0=0, y0=0,
            x1=math.cos(angle_rad), y1=math.sin(angle_rad),
            line=dict(color="#cccccc", width=1.5, dash="dot"),
        )

    # ── Ring labels (displayed at 90° / top of chart) ─────────────────────
    for ring, r_mid in zip(TW_RINGS, _RING_LABEL_R):
        fig.add_annotation(
            x=0,
            y=r_mid,
            text=f"<b>{ring['name']}</b>",
            showarrow=False,
            font=dict(size=9, color="#777"),
            xanchor="center",
            yanchor="middle",
            bgcolor="rgba(255,255,255,0.7)",
        )

    # ── Quadrant title labels ─────────────────────────────────────────────
    for q_name, q in TW_QUADRANTS.items():
        angle_rad = _deg2rad(q["label_angle"])
        fig.add_annotation(
            x=_LABEL_R * math.cos(angle_rad),
            y=_LABEL_R * math.sin(angle_rad),
            text=f"<b>{q_name}</b>",
            showarrow=False,
            font=dict(size=11, color=q["color"]),
            xanchor="center",
            yanchor="middle",
        )


def _base_layout() -> dict:
    return dict(
        xaxis=dict(range=[-_AXIS_RANGE, _AXIS_RANGE], visible=False, scaleanchor="y", scaleratio=1),
        yaxis=dict(range=[-_AXIS_RANGE, _AXIS_RANGE], visible=False),
        height=620,
        margin=dict(t=40, b=20, l=20, r=20),
        paper_bgcolor="white",
        plot_bgcolor="white",
        showlegend=True,
        legend=dict(
            orientation="v",
            x=1.02, y=1,
            xanchor="left", yanchor="top",
            font=dict(size=10),
            title=dict(text="<b>Ring</b>", font=dict(size=11)),
        ),
    )


def empty_tw_figure(message: str = "Click 'Refresh' to load the Technology Radar") -> go.Figure:
    """Radar scaffold with quadrants/rings but no blips — shown before data loads."""
    fig = go.Figure()
    _add_scaffold(fig)
    fig.add_annotation(
        x=0, y=0,
        text=message,
        showarrow=False,
        font=dict(size=13, color="#999"),
        xanchor="center",
        yanchor="middle",
    )
    fig.update_layout(**_base_layout())
    return fig


# ─────────────────────────────────────────────────────────────────────────────
# Main TW radar builder
# ─────────────────────────────────────────────────────────────────────────────

def build_tw_radar(
    df: pd.DataFrame,
    view: str = "group",
    researcher: Optional[str] = None,
    dimension: str = "contribute",
) -> go.Figure:
    """
    Build the ThoughtWorks-style radar figure.

    Group view (default):
      • One numbered blip per tech area (only areas with data)
      • Ring = avg(expertise, contribute) across all researchers
      • Marker size ∝ group mean interest (14–32 px)
      • Marker color = quadrant color
      • Hover: area name, ring, mean scores, researcher count

    Individual view:
      • One blip per tech area for the selected researcher
      • Ring = that researcher's avg(expertise, contribute)
      • Marker color = ring color (Lead=dark … Watch=light)
      • Hover: area, personal scores, ring

    Returns a Plotly Figure ready for gr.Plot.
    """
    fig = go.Figure()
    _add_scaffold(fig)

    if df.empty:
        fig.add_annotation(
            x=0, y=0, text="No submissions yet — submit data via Tab 4",
            showarrow=False, font=dict(size=13, color="#999"),
            xanchor="center", yanchor="middle",
        )
        fig.update_layout(**_base_layout())
        return fig

    # Add ring legend traces (invisible, just for the legend)
    for ring in TW_RINGS:
        fig.add_trace(go.Scatter(
            x=[None], y=[None],
            mode="markers",
            marker=dict(size=12, color=ring["ring_color"], symbol="circle"),
            name=f"{ring['name']} — {ring['description']}",
            showlegend=True,
        ))

    if view == "individual":
        return _build_individual_view(fig, df, researcher, dimension)
    else:
        return _build_group_view(fig, df, dimension)


def _build_group_view(fig: go.Figure, df: pd.DataFrame, dimension: str) -> go.Figure:
    """Aggregate view: one blip per tech area, averaged across all researchers."""
    blip_num    = 1
    legend_rows = []   # for the blip legend table

    for q_name, q in TW_QUADRANTS.items():
        areas_with_data = [a for a in q["areas"] if a in df["tech_area"].values]
        n               = len(areas_with_data)
        if n == 0:
            continue

        bx, by, btext, bhover, bsize, bcolor = [], [], [], [], [], []

        for idx, area in enumerate(areas_with_data):
            sub     = df[df["tech_area"] == area]
            exp_avg = sub["expertise"].mean()
            con_avg = sub["contribute"].mean()
            int_avg = sub["interest"].mean()
            n_res   = len(sub)

            ring_idx  = score_to_ring_idx(exp_avg, con_avg)
            ring_name = TW_RINGS[ring_idx]["name"]

            x, y = _blip_xy(q["angle_start"], q["angle_end"], idx, n, ring_idx)
            size  = 14 + int(min(int_avg or 3, 5) / 5 * 18)   # 14–32 px

            bx.append(x);  by.append(y)
            btext.append(str(blip_num))
            bhover.append(
                f"<b>{blip_num}. {area}</b><br>"
                f"Quadrant: {q_name}<br>"
                f"Ring: <b>{ring_name}</b><br>"
                f"Expertise: {exp_avg:.1f}  |  Contribute: {con_avg:.1f}  |  Interest: {int_avg:.1f}<br>"
                f"Researchers: {n_res}"
            )
            bsize.append(size)
            bcolor.append(q["color"])
            legend_rows.append({
                "num": blip_num,
                "area": area,
                "quadrant": q_name,
                "ring": ring_name,
                "exp": f"{exp_avg:.1f}",
                "con": f"{con_avg:.1f}",
                "int": f"{int_avg:.1f}",
                "n": n_res,
            })
            blip_num += 1

        fig.add_trace(go.Scatter(
            x=bx, y=by,
            mode="markers+text",
            marker=dict(size=bsize, color=bcolor, line=dict(color="white", width=1.5)),
            text=btext,
            textposition="middle center",
            textfont=dict(size=9, color="white", family="Arial Black"),
            hovertext=bhover,
            hoverinfo="text",
            showlegend=False,
            name=q_name,
        ))

    fig.update_layout(**_base_layout(), title=dict(text="Technology Radar — Group View", x=0.5, font=dict(size=16)))
    # Attach legend rows for the table below the chart
    fig._blip_legend = legend_rows   # type: ignore[attr-defined]
    return fig


def _build_individual_view(
    fig: go.Figure,
    df: pd.DataFrame,
    researcher: Optional[str],
    dimension: str,
) -> go.Figure:
    """Individual view: blips coloured by ring, sized uniformly."""
    if not researcher:
        fig.add_annotation(
            x=0, y=0, text="Select a researcher in the dropdown above",
            showarrow=False, font=dict(size=13, color="#999"),
        )
        fig.update_layout(**_base_layout())
        return fig

    sub_all = df[df["researcher"] == researcher]
    if sub_all.empty:
        fig.add_annotation(
            x=0, y=0, text=f"No data found for {researcher}",
            showarrow=False, font=dict(size=13, color="#999"),
        )
        fig.update_layout(**_base_layout())
        return fig

    blip_num    = 1
    legend_rows = []

    for q_name, q in TW_QUADRANTS.items():
        areas_with_data = [a for a in q["areas"] if a in sub_all["tech_area"].values]
        n               = len(areas_with_data)
        if n == 0:
            continue

        bx, by, btext, bhover, bcolor = [], [], [], [], []

        for idx, area in enumerate(areas_with_data):
            row      = sub_all[sub_all["tech_area"] == area].iloc[0]
            exp_val  = row.get("expertise")  or 3
            con_val  = row.get("contribute") or 3
            int_val  = row.get("interest")   or 3

            ring_idx   = score_to_ring_idx(exp_val, con_val)
            ring_name  = TW_RINGS[ring_idx]["name"]
            ring_color = TW_RINGS[ring_idx]["ring_color"]

            x, y = _blip_xy(q["angle_start"], q["angle_end"], idx, n, ring_idx, jitter_seed=blip_num)

            bx.append(x);  by.append(y)
            btext.append(str(blip_num))
            bhover.append(
                f"<b>{blip_num}. {area}</b><br>"
                f"Ring: <b>{ring_name}</b><br>"
                f"Expertise: {exp_val}  |  Contribute: {con_val}  |  Interest: {int_val}"
            )
            bcolor.append(ring_color)
            legend_rows.append({
                "num": blip_num, "area": area, "quadrant": q_name,
                "ring": ring_name, "exp": exp_val, "con": con_val, "int": int_val,
            })
            blip_num += 1

        fig.add_trace(go.Scatter(
            x=bx, y=by,
            mode="markers+text",
            marker=dict(size=22, color=bcolor, line=dict(color="white", width=1.5)),
            text=btext,
            textposition="middle center",
            textfont=dict(size=9, color="white", family="Arial Black"),
            hovertext=bhover,
            hoverinfo="text",
            showlegend=False,
            name=q_name,
        ))

    title = f"Technology Radar — {researcher}"
    fig.update_layout(**_base_layout(), title=dict(text=title, x=0.5, font=dict(size=16)))
    fig._blip_legend = legend_rows  # type: ignore[attr-defined]
    return fig


# ─────────────────────────────────────────────────────────────────────────────
# Blip legend table (rendered below the chart as Markdown)
# ─────────────────────────────────────────────────────────────────────────────

def build_blip_legend(fig: go.Figure, view: str = "group") -> str:
    """
    Extract blip metadata attached to the figure and render as a Markdown table.
    Returns empty string if no blips are present.
    """
    rows = getattr(fig, "_blip_legend", [])
    if not rows:
        return ""

    if view == "group":
        header = "| # | Tech Area | Quadrant | Ring | Expertise | Contribute | Interest | Researchers |"
        sep    = "|---|---|---|---|---|---|---|---|"
        lines  = [header, sep]
        for r in rows:
            lines.append(
                f"| **{r['num']}** | {r['area']} | {r['quadrant']} | **{r['ring']}** "
                f"| {r['exp']} | {r['con']} | {r['int']} | {r['n']} |"
            )
    else:
        header = "| # | Tech Area | Quadrant | Ring | Expertise | Contribute | Interest |"
        sep    = "|---|---|---|---|---|---|---|"
        lines  = [header, sep]
        for r in rows:
            lines.append(
                f"| **{r['num']}** | {r['area']} | {r['quadrant']} | **{r['ring']}** "
                f"| {r['exp']} | {r['con']} | {r['int']} |"
            )

    return "\n".join(lines)


# ─────────────────────────────────────────────────────────────────────────────
# Static legend / explanation (always visible below chart)
# ─────────────────────────────────────────────────────────────────────────────

def build_tw_legend_table() -> str:
    """
    Markdown explanation of rings and quadrants — rendered below the TW radar.
    Mirrors the lihsmi.ch skill radar explanation style.
    """
    ring_rows = "\n".join(
        f"| **{r['name']}** | {r['description']} |"
        for r in TW_RINGS
    )
    q_rows = "\n".join(
        f"| **{q}** | {', '.join(data['areas'])} |"
        for q, data in TW_QUADRANTS.items()
    )

    return f"""
---
#### How to read this radar

The radar places each deep tech area as a **blip** at the intersection of a **quadrant** (domain)
and a **ring** (readiness level).  Ring placement is driven by the average of *Expertise* and
*Desire to Contribute* scores across the team.

| Ring | Meaning |
|---|---|
{ring_rows}

| Quadrant | Tech Areas |
|---|---|
{q_rows}

**Marker size** (group view) scales with the group's mean **Interest** score for that area —
larger blips signal stronger collective curiosity.
Custom areas added by individual researchers appear in the panel above if they are not
assigned to a quadrant yet.
"""
