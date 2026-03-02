"""
4_Eligibility_Analyzer.py — DecisionLENS AI Eligibility Analyzer page.

Powered by Groq API (llama-3.3-70b-versatile).  Falls back to realistic
mock responses when GROQ_API_KEY is not set.

Sections:
  1. Criteria input (text area) + Analyze button + Example button
  2. Risk factor table with colour-coded severity badges
  3. Simplification suggestions (expandable cards)
  4. Population impact visualization
  5. Generate Executive Briefing (downloads as text)
  6. Protocol comparison tool (side-by-side criteria comparison)
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
import plotly.graph_objects as go
import streamlit as st

st.set_page_config(
    page_title="Eligibility Analyzer — DecisionLENS",
    page_icon="🤖",
    layout="wide",
)

from app.components._theme import apply_theme
from app.components.sidebar import render_sidebar

apply_theme()
state = render_sidebar()

# ---------------------------------------------------------------------------
# Example criteria (pre-loaded for demo)
# ---------------------------------------------------------------------------
_EXAMPLE_CRITERIA: str = """\
INCLUSION CRITERIA:
1. Adults aged 18–75 years.
2. Histologically confirmed non-small cell lung cancer (NSCLC), stage IIIB or IV.
3. ECOG performance status 0–1.
4. No prior systemic anti-cancer therapy for advanced disease (≤ 2 prior lines allowed if chemotherapy-naive for this indication).
5. Measurable disease per RECIST v1.1.
6. Adequate bone marrow, renal, and hepatic function (ANC ≥ 1.5 × 10⁹/L, platelets ≥ 100 × 10⁹/L, eGFR ≥ 60 mL/min/1.73 m²).

EXCLUSION CRITERIA:
1. Prior treatment with any anti-PD-1, anti-PD-L1, or anti-CTLA-4 antibody.
2. Known active brain metastases (treated, stable brain metastases are excluded).
3. Active or prior documented autoimmune disease requiring systemic treatment within 2 years.
4. Systemic corticosteroid therapy > 10 mg/day prednisone equivalent within 14 days of first dose.
5. Active hepatitis B or C infection.
6. Any prior malignancy (other than adequately treated basal cell carcinoma or cervical cancer in situ) within 3 years.\
"""

_SEVERITY_COLOR: dict[str, str] = {
    "high":   "#d32f2f",
    "medium": "#f9a825",
    "low":    "#388e3c",
}
_SEVERITY_BG: dict[str, str] = {
    "high":   "#fdecea",
    "medium": "#fff9e6",
    "low":    "#e8f5e9",
}

# ---------------------------------------------------------------------------
# Cached EligibilityAnalyzer
# ---------------------------------------------------------------------------

@st.cache_resource(show_spinner=False)
def _get_analyzer():
    from src.genai_utils import EligibilityAnalyzer
    return EligibilityAnalyzer()


analyzer = _get_analyzer()

# ---------------------------------------------------------------------------
# Page header
# ---------------------------------------------------------------------------
st.markdown("## 🤖 Eligibility Criteria Analyzer")

if analyzer.is_demo_mode:
    st.info(
        "**Demo mode** — GROQ_API_KEY is not set.  "
        "All analysis results below are pre-written mock responses.  "
        "Add `GROQ_API_KEY` to your `.env` file to enable live Groq LLM analysis.",
        icon="ℹ️",
    )
else:
    st.success(
        f"Live mode — using **{analyzer.model}** via Groq API.",
        icon="✅",
    )

st.markdown(
    "Paste eligibility criteria (inclusion + exclusion) to identify enrollment "
    "risk factors, get simplification suggestions, and generate a stakeholder briefing."
)
st.markdown("<hr>", unsafe_allow_html=True)

# ---------------------------------------------------------------------------
# Session state initialisation
# ---------------------------------------------------------------------------
if "criteria_text" not in st.session_state:
    st.session_state["criteria_text"] = ""
if "analysis_result" not in st.session_state:
    st.session_state["analysis_result"] = None
if "briefing_text" not in st.session_state:
    st.session_state["briefing_text"] = None

# ---------------------------------------------------------------------------
# Input area
# ---------------------------------------------------------------------------
st.markdown(
    "<div class='section-header'>Eligibility Criteria Input</div>",
    unsafe_allow_html=True,
)

btn_col1, btn_col2, btn_col3, _ = st.columns([1, 1, 1, 3])

with btn_col1:
    if st.button("📋 Load Example", use_container_width=True):
        st.session_state["criteria_text"] = _EXAMPLE_CRITERIA
        st.session_state["analysis_result"] = None
        st.session_state["briefing_text"]   = None
        st.rerun()

with btn_col2:
    if st.button("🗑️ Clear", use_container_width=True):
        st.session_state["criteria_text"] = ""
        st.session_state["analysis_result"] = None
        st.session_state["briefing_text"]   = None
        st.rerun()

criteria_text: str = st.text_area(
    label="Paste eligibility criteria here",
    value=st.session_state["criteria_text"],
    height=260,
    placeholder=(
        "Paste the INCLUSION and EXCLUSION criteria from a clinical trial protocol.\n\n"
        "Click 'Load Example' above to pre-fill a sample NSCLC protocol."
    ),
    help="Supports free-form text. Both inclusion and exclusion criteria are analysed together.",
)
st.session_state["criteria_text"] = criteria_text

with btn_col3:
    analyze_btn = st.button(
        "🔍 Analyze",
        use_container_width=True,
        type="primary",
        disabled=not criteria_text.strip(),
    )

# ---------------------------------------------------------------------------
# Run analysis
# ---------------------------------------------------------------------------
if analyze_btn and criteria_text.strip():
    with st.spinner("Analyzing eligibility criteria…"):
        result = analyzer.analyze_criteria(criteria_text)
    st.session_state["analysis_result"] = result
    st.session_state["briefing_text"]   = None   # reset on new analysis

result = st.session_state.get("analysis_result")

if result is None:
    if not criteria_text.strip():
        st.info(
            "Paste eligibility criteria above and click **Analyze**, "
            "or click **Load Example** to try a sample NSCLC protocol.",
            icon="👆",
        )
    st.stop()

# ---------------------------------------------------------------------------
# Section 2 — Risk factor table
# ---------------------------------------------------------------------------
st.markdown("<hr>", unsafe_allow_html=True)
st.markdown(
    "<div class='section-header'>Enrollment Risk Factors</div>",
    unsafe_allow_html=True,
)

risk_factors   = result.get("risk_factors", [])
severity_scores = result.get("severity_scores", [])
suggestions     = result.get("simplification_suggestions", [])
impacts         = result.get("estimated_population_impact", [])

n_factors = len(risk_factors)

if n_factors == 0:
    st.success("No significant enrollment risk factors identified.", icon="✅")
else:
    # Build colour-coded table
    rows = []
    for i in range(n_factors):
        sev   = severity_scores[i] if i < len(severity_scores) else "medium"
        color = _SEVERITY_COLOR.get(sev, "#999")
        bg    = _SEVERITY_BG.get(sev, "#fff")
        rows.append({
            "#":            i + 1,
            "Risk Factor":  risk_factors[i] if i < len(risk_factors) else "",
            "Severity":     sev.upper(),
            "Population Impact": impacts[i] if i < len(impacts) else "",
        })

    # Severity counts as metrics
    m1, m2, m3 = st.columns(3)
    m1.metric("🔴 High Risk",   sum(1 for s in severity_scores if s == "high"))
    m2.metric("🟡 Medium Risk", sum(1 for s in severity_scores if s == "medium"))
    m3.metric("🟢 Low Risk",    sum(1 for s in severity_scores if s == "low"))

    st.markdown("<br>", unsafe_allow_html=True)
    risk_df = pd.DataFrame(rows)
    st.dataframe(
        risk_df,
        use_container_width=True,
        height=min(60 + 38 * n_factors, 500),
        hide_index=True,
    )

st.markdown("<hr>", unsafe_allow_html=True)

# ---------------------------------------------------------------------------
# Section 3 — Simplification suggestions
# ---------------------------------------------------------------------------
st.markdown(
    "<div class='section-header'>Simplification Suggestions</div>",
    unsafe_allow_html=True,
)

for i in range(n_factors):
    rf  = risk_factors[i]  if i < len(risk_factors)  else f"Factor {i+1}"
    sev = severity_scores[i] if i < len(severity_scores) else "medium"
    sug = suggestions[i]   if i < len(suggestions)   else "—"
    imp = impacts[i]       if i < len(impacts)        else "—"

    color  = _SEVERITY_COLOR.get(sev, "#999")
    badge  = f"<span style='background:{color};color:#fff;border-radius:4px;padding:2px 8px;font-size:0.75rem;'>{sev.upper()}</span>"

    with st.expander(f"{i+1}. {rf}", expanded=(sev == "high")):
        st.markdown(
            f"{badge}&nbsp;&nbsp;<b>Suggested change:</b> {sug}",
            unsafe_allow_html=True,
        )
        st.markdown(f"**Population impact:** {imp}")

st.markdown("<hr>", unsafe_allow_html=True)

# ---------------------------------------------------------------------------
# Section 4 — Population impact visualization
# ---------------------------------------------------------------------------
st.markdown(
    "<div class='section-header'>Estimated Population Impact</div>",
    unsafe_allow_html=True,
)

sev_order   = ["high", "medium", "low"]
sev_counts  = {s: severity_scores.count(s) for s in sev_order}
sev_colors  = [_SEVERITY_COLOR[s] for s in sev_order]

fig_impact = go.Figure()

fig_impact.add_trace(go.Bar(
    x=[s.title() for s in sev_order],
    y=[sev_counts[s] for s in sev_order],
    marker_color=sev_colors,
    text=[sev_counts[s] for s in sev_order],
    textposition="outside",
    hovertemplate="<b>%{x}</b><br>%{y} criteria<extra></extra>",
))

fig_impact.update_layout(
    title="Risk Factor Distribution by Severity",
    xaxis_title="Severity Level",
    yaxis_title="Number of Criteria",
    height=300, template="plotly_white",
    margin=dict(l=10, r=10, t=50, b=40),
    showlegend=False,
)
st.plotly_chart(fig_impact, use_container_width=True)

st.markdown("<hr>", unsafe_allow_html=True)

# ---------------------------------------------------------------------------
# Section 5 — Executive briefing
# ---------------------------------------------------------------------------
st.markdown(
    "<div class='section-header'>Executive Briefing</div>",
    unsafe_allow_html=True,
)

st.markdown(
    "Generate a 1-page briefing for a VP of Clinical Operations, synthesising "
    "this eligibility analysis with enrollment forecast and competitive data."
)

brief_col1, brief_col2 = st.columns([2, 1])

with brief_col2:
    generate_btn = st.button(
        "📝 Generate Briefing",
        use_container_width=True,
        type="primary",
    )

if generate_btn:
    with st.spinner("Generating executive briefing via Groq…"):
        prediction_stub = {
            "n_risk_factors":    n_factors,
            "high_risk_count":   sev_counts.get("high", 0),
            "medium_risk_count": sev_counts.get("medium", 0),
            "condition":         state["condition_query"] or "unspecified condition",
            "note": "Enrollment prediction not yet run — open Page 1 to configure a trial.",
        }
        competition_stub = {
            "note": (
                "Competitive landscape not yet loaded — "
                "open Page 2 for condition-specific data."
            )
        }
        site_stub = {
            "note": (
                "Site data not yet loaded — "
                "open Page 3 for top-site recommendations."
            )
        }
        briefing = analyzer.generate_executive_briefing(
            prediction_stub, competition_stub, site_stub
        )
    st.session_state["briefing_text"] = briefing

briefing_text = st.session_state.get("briefing_text")
if briefing_text:
    with brief_col1:
        st.text_area(
            "Executive Briefing",
            value=briefing_text,
            height=350,
            disabled=True,
            label_visibility="collapsed",
        )
    st.download_button(
        label="⬇️ Download Briefing (.txt)",
        data=briefing_text,
        file_name="decisionlens_executive_briefing.txt",
        mime="text/plain",
        use_container_width=False,
    )

st.markdown("<hr>", unsafe_allow_html=True)

# ---------------------------------------------------------------------------
# Section 6 — Protocol comparison
# ---------------------------------------------------------------------------
st.markdown(
    "<div class='section-header'>Protocol Comparison</div>",
    unsafe_allow_html=True,
)

with st.expander("Compare two eligibility criteria sets", expanded=False):
    comp_col_a, comp_col_b = st.columns(2)

    with comp_col_a:
        criteria_a = st.text_area(
            "Protocol A criteria",
            value=criteria_text,
            height=200,
            help="Protocol A — defaults to the criteria entered above.",
        )
    with comp_col_b:
        criteria_b = st.text_area(
            "Protocol B criteria",
            value="",
            height=200,
            placeholder="Paste Protocol B eligibility criteria here…",
        )

    compare_btn = st.button(
        "⚖️ Compare Protocols",
        type="primary",
        disabled=not (criteria_a.strip() and criteria_b.strip()),
    )

    if compare_btn:
        with st.spinner("Comparing protocols…"):
            cmp_result = analyzer.compare_criteria(criteria_a, criteria_b)

        st.markdown("**Overall Assessment**")
        st.info(cmp_result.get("overall_assessment", "No assessment available."))

        diffs = cmp_result.get("differences", [])
        impacts_cmp = cmp_result.get("enrollment_impact", [])

        if diffs:
            cmp_rows = []
            for i, d in enumerate(diffs):
                impact = impacts_cmp[i] if i < len(impacts_cmp) else "neutral"
                impact_label = {
                    "favorable_A": "✅ Favors A",
                    "favorable_B": "✅ Favors B",
                    "neutral":     "⬡ Neutral",
                }.get(impact, impact)
                cmp_rows.append({
                    "Criterion":       d.get("criterion", ""),
                    "Protocol A":      d.get("protocol_a", ""),
                    "Protocol B":      d.get("protocol_b", ""),
                    "More Restrictive":d.get("more_restrictive", ""),
                    "Enrollment Impact":impact_label,
                })
            st.dataframe(
                pd.DataFrame(cmp_rows),
                use_container_width=True,
                hide_index=True,
            )

        recs = cmp_result.get("recommendations", [])
        if recs:
            st.markdown("**Recommendations to improve the less favorable protocol:**")
            for r in recs:
                st.markdown(f"- {r}")

# ---------------------------------------------------------------------------
# Methodology expander
# ---------------------------------------------------------------------------
with st.expander("⚙️ Methodology", expanded=False):
    st.markdown(
        f"""
        **AI model:** `{analyzer.model}` via Groq API
        (fast inference, free tier available).

        **Structured analysis** (`analyze_criteria`): The model receives the
        eligibility criteria with a system prompt instructing it to act as a
        clinical enrollment optimization expert.  It returns a JSON object with
        four parallel lists: risk factors, severity scores, simplification
        suggestions, and population impact estimates.

        **Executive briefing** (`generate_executive_briefing`): Uses streaming
        to generate a 300–400 word formatted briefing in four sections:
        Forecast Summary, Risk Assessment, Competitive Context, Recommended Actions.

        **Protocol comparison** (`compare_criteria`): Returns a structured
        JSON diff of the two criteria sets with per-criterion enrollment impact
        assessment and overall recommendations.

        **Demo mode:** When `GROQ_API_KEY` is not set, all methods return
        realistic pre-written mock responses so the app is fully usable without
        a live API key.

        **Caching:** API responses are cached in memory and on disk at
        `.cache/genai/` using SHA-256 keys, so repeated identical calls
        do not consume API quota.
        """
    )
