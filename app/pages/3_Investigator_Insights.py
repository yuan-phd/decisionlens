"""
3_Investigator_Insights.py — DecisionLENS Investigator & Site Insights page.

Sections:
  1. Top-performing sites table (sortable)
  2. Country performance heatmap
  3. Site co-participation network graph
  4. Site recommendation engine
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
import streamlit as st

st.set_page_config(
    page_title="Investigator Insights — DecisionLENS",
    page_icon="🏥",
    layout="wide",
)

from app.components._theme import apply_theme
from app.components.sidebar import render_sidebar
from app.components.charts import site_heatmap

apply_theme()
state = render_sidebar()

# ---------------------------------------------------------------------------
# Cached loader
# ---------------------------------------------------------------------------

@st.cache_resource(show_spinner="Loading investigator analyzer…")
def _get_analyzer(data_dir: str):
    from src.investigator_insights import InvestigatorAnalyzer
    return InvestigatorAnalyzer(data_dir)


# ---------------------------------------------------------------------------
# Page header
# ---------------------------------------------------------------------------
st.markdown("## 🏥 Investigator & Site Insights")
st.markdown(
    "Rank trial sites by historical performance, visualise country-level "
    "enrollment efficiency, and get data-driven site allocation recommendations."
)
st.markdown("<hr>", unsafe_allow_html=True)

# ---------------------------------------------------------------------------
# Filters row
# ---------------------------------------------------------------------------
filter_col1, filter_col2, filter_col3 = st.columns(3)

with filter_col1:
    condition = st.text_input(
        "Condition / Indication",
        value=state["condition_query"] or "Non-Small Cell Lung Cancer",
        help="Case-insensitive substring search.",
    )

with filter_col2:
    n_top_sites = st.slider(
        "Top N sites to display", min_value=5, max_value=50, value=20, step=5
    )

with filter_col3:
    country_filter = st.text_input(
        "Filter by country (optional)",
        value="",
        placeholder="e.g. United States",
    )

st.markdown("<hr>", unsafe_allow_html=True)

analyzer = _get_analyzer(state["data_dir"])

# ---------------------------------------------------------------------------
# Section 1 — Top sites table
# ---------------------------------------------------------------------------
st.markdown(
    "<div class='section-header'>Top Investigator Sites</div>",
    unsafe_allow_html=True,
)

with st.spinner(f"Loading top sites for '{condition}'…"):
    try:
        top_sites_df = analyzer.get_top_sites(condition, n=n_top_sites)
    except Exception as exc:
        top_sites_df = pd.DataFrame()
        st.error(f"Error loading sites: {exc}", icon="❌")

if top_sites_df.empty:
    st.warning(
        f"No named investigator sites found for **'{condition}'**.  "
        "Try a broader search term.",
        icon="⚠️",
    )
else:
    # Apply country filter
    if country_filter.strip():
        top_sites_df = top_sites_df[
            top_sites_df["country"].str.contains(
                country_filter.strip(), case=False, na=False
            )
        ]

    display_cols = [c for c in [
        "name", "city", "country", "n_trials", "completion_rate",
        "n_completed", "n_terminated", "avg_enrollment",
    ] if c in top_sites_df.columns]

    rename_map = {
        "name":             "Site Name",
        "city":             "City",
        "country":          "Country",
        "n_trials":         "Trials",
        "completion_rate":  "Completion Rate",
        "n_completed":      "Completed",
        "n_terminated":     "Terminated",
        "avg_enrollment":   "Avg Enrollment",
    }

    show_df = top_sites_df[display_cols].rename(columns=rename_map).reset_index(drop=True)

    # Format completion rate as percentage
    if "Completion Rate" in show_df.columns:
        show_df["Completion Rate"] = show_df["Completion Rate"].map(
            lambda x: f"{x:.1%}" if pd.notna(x) else "—"
        )

    st.dataframe(show_df, use_container_width=True, height=400)
    st.caption(
        f"{len(show_df)} sites shown.  "
        "Sites are ranked by trial volume then completion rate.  "
        "Generic sponsor placeholder names are excluded."
    )

st.markdown("<hr>", unsafe_allow_html=True)

# ---------------------------------------------------------------------------
# Section 2 — Country performance heatmap
# ---------------------------------------------------------------------------
st.markdown(
    "<div class='section-header'>Country-Level Enrollment Performance</div>",
    unsafe_allow_html=True,
)

with st.spinner("Computing country performance…"):
    try:
        country_perf_df = analyzer.get_country_performance(
            condition if condition.strip() else None
        )
    except Exception as exc:
        country_perf_df = pd.DataFrame()
        st.error(f"Error computing country performance: {exc}", icon="❌")

if country_perf_df.empty:
    st.info("No country performance data available for this condition.")
else:
    hm_metrics = [c for c in [
        "n_trials", "completion_rate", "avg_enrollment",
        "avg_duration_days", "trials_per_year",
    ] if c in country_perf_df.columns]

    fig_hm = site_heatmap(
        country_perf_df,
        country_col="country",
        metrics=hm_metrics,
        title=f"Country Performance — {condition or 'All Conditions'}",
    )
    st.plotly_chart(fig_hm, use_container_width=True)
    st.caption(
        "Values are z-score normalised per column so colours are comparable "
        "across metrics with different units. Raw values are shown as annotations."
    )

st.markdown("<hr>", unsafe_allow_html=True)

# ---------------------------------------------------------------------------
# Section 3 — Site co-participation network
# ---------------------------------------------------------------------------
st.markdown(
    "<div class='section-header'>Site Co-participation Network</div>",
    unsafe_allow_html=True,
)

with st.expander("Show network graph (may be slow for large conditions)", expanded=False):
    with st.spinner("Building site network…"):
        try:
            fig_network = analyzer.plot_site_network(condition, n_sites=25)
            st.plotly_chart(fig_network, use_container_width=True)
            st.caption(
                "Nodes = investigator sites (sized by trial count, coloured by country).  "
                "Edges connect sites that co-participated in ≥ 1 trial."
            )
        except Exception as exc:
            st.info(f"Network graph unavailable: {exc}", icon="ℹ️")

st.markdown("<hr>", unsafe_allow_html=True)

# ---------------------------------------------------------------------------
# Section 4 — Site recommendation engine
# ---------------------------------------------------------------------------
st.markdown(
    "<div class='section-header'>Site Allocation Recommendation</div>",
    unsafe_allow_html=True,
)

rec_col1, rec_col2, rec_col3 = st.columns(3)

with rec_col1:
    target_enrollment_rec = st.number_input(
        "Target enrollment (# patients)",
        min_value=50, max_value=50_000, value=500, step=50, key="rec_enroll",
    )
with rec_col2:
    n_countries_rec = st.number_input(
        "Number of countries",
        min_value=1, max_value=30, value=5, step=1, key="rec_countries",
    )
with rec_col3:
    st.markdown("<br>", unsafe_allow_html=True)
    rec_btn = st.button(
        "📍 Get Recommendations",
        use_container_width=True,
        type="primary",
    )

if rec_btn:
    with st.spinner("Computing site allocation…"):
        try:
            rec_df = analyzer.recommend_sites(
                condition,
                target_enrollment=int(target_enrollment_rec),
                n_countries=int(n_countries_rec),
            )
        except Exception as exc:
            rec_df = pd.DataFrame()
            st.error(f"Recommendation error: {exc}", icon="❌")

    if rec_df.empty:
        st.warning(
            "Not enough historical data to generate recommendations for this condition.  "
            "Try a broader condition search.",
            icon="⚠️",
        )
    else:
        st.success(
            f"Recommended allocation for **{target_enrollment_rec:,} patients** "
            f"across **{n_countries_rec}** countries:",
            icon="✅",
        )

        rec_display_cols = [c for c in [
            "country", "recommended_sites", "recommended_patients",
            "completion_rate", "rationale",
        ] if c in rec_df.columns]

        rec_rename = {
            "country":               "Country",
            "recommended_sites":     "Recommended Sites",
            "recommended_patients":  "Target Patients",
            "completion_rate":       "Historical Completion Rate",
            "rationale":             "Rationale",
        }

        rec_show = rec_df[rec_display_cols].rename(columns=rec_rename).reset_index(drop=True)
        if "Historical Completion Rate" in rec_show.columns:
            rec_show["Historical Completion Rate"] = rec_show["Historical Completion Rate"].map(
                lambda x: f"{x:.1%}" if pd.notna(x) else "—"
            )
        st.dataframe(rec_show, use_container_width=True)

# ---------------------------------------------------------------------------
# Methodology expander
# ---------------------------------------------------------------------------
with st.expander("⚙️ Methodology", expanded=False):
    st.markdown(
        """
        **Site ranking**: Sites are ranked by total trial count for the condition,
        then by completion rate.  Generic sponsor placeholder names
        (e.g., "Research Site", "GSK Investigational Site") are excluded.

        **Country heatmap**: Metrics are z-score normalised per column so
        colours reflect relative standing, not absolute values.
        Completion rate = completed / (completed + terminated) trials.

        **Network graph**: Edges connect sites that co-participated in at least
        one trial.  Edge width scales with the number of shared trials.

        **Site recommendation**: Countries are ranked by a composite performance
        score (completion rate × log₁⁺(completed trials)).  The enrollment
        target is distributed proportionally to performance score, subject to
        a minimum of 1 site per selected country.
        """
    )
