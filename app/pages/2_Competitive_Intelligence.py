"""
2_Competitive_Intelligence.py — DecisionLENS Competitive Intelligence page.

Sections:
  1. KPI cards — total trials, active, enrollment pressure, competition intensity
  2. Choropleth world map of competing trial sites
  3. Gantt timeline of competing trials coloured by phase
  4. Sponsor market-share treemap
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).parent.parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from dotenv import load_dotenv
load_dotenv(ROOT / ".env")

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

st.set_page_config(
    page_title="Competitive Intelligence — DecisionLENS",
    page_icon="🗺️",
    layout="wide",
)

from app.components._theme import apply_theme
from app.components.sidebar import render_sidebar

apply_theme()
state = render_sidebar()

# ---------------------------------------------------------------------------
# Cached loaders
# ---------------------------------------------------------------------------

@st.cache_resource(show_spinner="Loading competitive analyzer…")
def _get_analyzer(data_dir: str):
    from src.competitive_intel import CompetitiveAnalyzer
    return CompetitiveAnalyzer(data_dir)


@st.cache_data(show_spinner="Loading landscape…", ttl=300)
def _load_landscape(data_dir: str, condition: str) -> dict:
    analyzer = _get_analyzer(data_dir)
    return analyzer.get_landscape(condition)


# ---------------------------------------------------------------------------
# Page header
# ---------------------------------------------------------------------------
st.markdown("## 🗺️ Competitive Intelligence")
st.markdown(
    "Explore the competitive landscape for any condition: site distribution, "
    "trial timelines, and sponsor activity."
)
st.markdown("<hr>", unsafe_allow_html=True)

# ---------------------------------------------------------------------------
# Condition selector
# ---------------------------------------------------------------------------
col_cond, col_date = st.columns([2, 2])

with col_cond:
    condition = st.text_input(
        "Condition / Indication",
        value=state["condition_query"] or "Non-Small Cell Lung Cancer",
        help="Case-insensitive substring search against AACT conditions.",
    )

with col_date:
    st.markdown("*Use the sidebar date range and phase filters to refine results.*")

if not condition.strip():
    st.warning("Enter a condition to analyse.", icon="⚠️")
    st.stop()

# ---------------------------------------------------------------------------
# Load landscape data
# ---------------------------------------------------------------------------
with st.spinner(f"Analysing landscape for '{condition}'…"):
    landscape = _load_landscape(state["data_dir"], condition)

if landscape.get("total_trials", 0) == 0:
    st.error(
        f"No trials found for **'{condition}'**.  Try a different search term.",
        icon="🔍",
    )
    st.stop()

# ---------------------------------------------------------------------------
# Section 1 — KPI cards
# ---------------------------------------------------------------------------
st.markdown(
    "<div class='section-header'>Landscape Overview</div>",
    unsafe_allow_html=True,
)

k1, k2, k3, k4, k5 = st.columns(5)

k1.metric("Total Trials",       f"{landscape['total_trials']:,}")
k2.metric("Active / Recruiting",
          f"{landscape['active_trials']:,}",
          delta=f"{landscape['recruiting_trials']:,} actively recruiting")
k3.metric("Completed Trials",   f"{landscape['completed_trials']:,}")
k4.metric("Terminated Trials",  f"{landscape['terminated_trials']:,}")

intensity_pct = f"{landscape['competition_intensity']:.0%}"
k5.metric(
    "Competition Intensity",
    intensity_pct,
    help="Fraction of a 200-trial ceiling of active competing trials.",
)

st.markdown("<hr>", unsafe_allow_html=True)

# Temporal density mini-row
temp = landscape.get("temporal_density", {})
if temp:
    t1, t2, t3, _ = st.columns(4)
    t1.metric("Started last 6 mo",  f"{temp.get('last_6m',  0):,}")
    t2.metric("Started last 12 mo", f"{temp.get('last_12m', 0):,}")
    t3.metric("Started last 24 mo", f"{temp.get('last_24m', 0):,}")

total_enroll = landscape.get("total_competing_enrollment", 0)
if total_enroll:
    st.info(
        f"Active competing trials target a combined **{total_enroll:,} patients**.",
        icon="👥",
    )

st.markdown("<hr>", unsafe_allow_html=True)

# ---------------------------------------------------------------------------
# Section 2 — Choropleth map
# ---------------------------------------------------------------------------
st.markdown(
    "<div class='section-header'>Global Site Distribution</div>",
    unsafe_allow_html=True,
)

analyzer = _get_analyzer(state["data_dir"])
fig_map  = analyzer.plot_competition_map(condition)
st.plotly_chart(fig_map, use_container_width=True)

st.markdown("<hr>", unsafe_allow_html=True)

# ---------------------------------------------------------------------------
# Section 3 — Competition timeline
# ---------------------------------------------------------------------------
st.markdown(
    "<div class='section-header'>Competing Trial Timeline</div>",
    unsafe_allow_html=True,
)
fig_timeline = analyzer.plot_competition_timeline(condition)
st.plotly_chart(fig_timeline, use_container_width=True)
st.caption("Showing interventional trials with defined phase only (Phase 1–4).")

st.markdown("<hr>", unsafe_allow_html=True)

# ---------------------------------------------------------------------------
# Section 4 — Sponsor treemap
# ---------------------------------------------------------------------------
st.markdown(
    "<div class='section-header'>Sponsor Market Share (Active Trials)</div>",
    unsafe_allow_html=True,
)

top_sponsors = landscape.get("top_sponsors", [])
if top_sponsors:
    sp_df = pd.DataFrame(top_sponsors, columns=["sponsor", "n_trials"])
    fig_tree = px.treemap(
        sp_df,
        path=["sponsor"],
        values="n_trials",
        color="n_trials",
        color_continuous_scale=[[0, "#caf0f8"], [0.5, "#0096c7"], [1, "#023e8a"]],
        title=f"Top Sponsors — {condition}",
    )
    fig_tree.update_traces(
        hovertemplate="<b>%{label}</b><br>Trials: %{value}<extra></extra>",
        textinfo="label+value",
    )
    fig_tree.update_layout(
        height=380, template="plotly_white",
        margin=dict(l=10, r=10, t=50, b=10),
        coloraxis_showscale=False,
    )
    st.plotly_chart(fig_tree, use_container_width=True)
else:
    st.info("No sponsor data available for active trials in this condition.")

# ---------------------------------------------------------------------------
# Phase breakdown table
# ---------------------------------------------------------------------------
phase_breakdown = landscape.get("phase_breakdown", {})
if phase_breakdown:
    with st.expander("Phase breakdown (active trials)", expanded=False):
        pb_df = (
            pd.DataFrame.from_dict(phase_breakdown, orient="index", columns=["Active Trials"])
            .reset_index()
            .rename(columns={"index": "Phase"})
            .sort_values("Active Trials", ascending=False)
        )
        st.dataframe(pb_df, use_container_width=True, hide_index=True)

# ---------------------------------------------------------------------------
# Top countries table
# ---------------------------------------------------------------------------
top_countries = landscape.get("top_countries", [])
if top_countries:
    with st.expander("Top countries by site count", expanded=False):
        tc_df = pd.DataFrame(top_countries, columns=["Country", "Sites"])
        st.dataframe(tc_df, use_container_width=True, hide_index=True)
