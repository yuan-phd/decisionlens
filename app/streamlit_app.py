"""
streamlit_app.py — DecisionLENS home page and app entry point.

Run with::

    streamlit run app/streamlit_app.py

Navigation is handled by Streamlit's built-in multi-page support.
The four analysis pages live in ``app/pages/``.
"""

from __future__ import annotations

import logging
import subprocess
import sys
from pathlib import Path

# ---------------------------------------------------------------------------
# Path setup — ensure project root is on sys.path so `src.*` imports work
# ---------------------------------------------------------------------------
ROOT = Path(__file__).parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# ---------------------------------------------------------------------------
# .env loading (must happen before any src import)
# ---------------------------------------------------------------------------
from dotenv import load_dotenv
load_dotenv(ROOT / ".env")

import streamlit as st

log = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

# ---------------------------------------------------------------------------
# Page config — must be the first Streamlit call
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="DecisionLENS",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={
        "About": (
            "**DecisionLENS** — AI-augmented clinical trial enrollment forecasting.\n\n"
            "Data source: AACT / ClinicalTrials.gov\n"
            "Models: XGBoost + Cox PH | AI: Groq Llama 3.3 70B"
        ),
    },
)

# ---------------------------------------------------------------------------
# Data initialisation — runs once per server process (cached by Streamlit)
# ---------------------------------------------------------------------------

@st.cache_resource(show_spinner="Downloading data from HuggingFace… (first run only, ~2 min)")
def initialize_data() -> bool:
    """
    Ensure data/processed/ has the AACT parquet files.

    On first run (or when the processed/ directory is absent), calls
    ``setup_data.py --source=huggingface`` as a subprocess so the download
    log streams to the server terminal.  Returns immediately on subsequent
    runs thanks to ``@st.cache_resource``.
    """
    data_dir = ROOT / "data" / "processed"
    if (data_dir / "studies.parquet").exists():
        return True   # already present — nothing to do

    st.info(
        "Downloading AACT trial data from HuggingFace on first run. "
        "This takes about 2 minutes. The page will refresh automatically.",
        icon="⏬",
    )
    result = subprocess.run(
        [sys.executable, str(ROOT / "setup_data.py"), "--source=huggingface"],
        cwd=str(ROOT),
        check=False,
    )
    if result.returncode != 0:
        log.warning(
            "setup_data.py --source=huggingface exited with code %d — "
            "app will use synthetic fallback data.",
            result.returncode,
        )
    return True


initialize_data()

# ---------------------------------------------------------------------------
# Global CSS — clinical/pharma theme
# ---------------------------------------------------------------------------
st.markdown(
    """
    <style>
    /* ── Typography ── */
    html, body, [class*="css"] {
        font-family: "Inter", "Segoe UI", Helvetica, Arial, sans-serif;
    }

    /* ── Sidebar ── */
    section[data-testid="stSidebar"] {
        background: #f0f4f8;
        border-right: 1px solid #dde3ea;
    }

    /* ── Metric cards ── */
    [data-testid="stMetric"] {
        background: #ffffff;
        border: 1px solid #e3e8ef;
        border-radius: 10px;
        padding: 1rem 1.25rem;
        box-shadow: 0 1px 4px rgba(0,0,0,0.05);
    }
    [data-testid="stMetricLabel"] { font-size: 0.78rem; color: #6b7280; }
    [data-testid="stMetricValue"] { font-size: 1.75rem; color: #0077b6; font-weight: 700; }
    [data-testid="stMetricDelta"] { font-size: 0.78rem; }

    /* ── Nav cards ── */
    .nav-card {
        background: #ffffff;
        border: 1px solid #e3e8ef;
        border-radius: 12px;
        padding: 1.25rem 1.5rem;
        margin-bottom: 0.5rem;
        box-shadow: 0 1px 4px rgba(0,0,0,0.04);
        transition: box-shadow 0.15s;
    }
    .nav-card:hover { box-shadow: 0 4px 12px rgba(0,119,182,0.12); }
    .nav-card h4 { margin: 0 0 0.35rem 0; color: #0077b6; }
    .nav-card p  { margin: 0; font-size: 0.85rem; color: #6b7280; }

    /* ── Section headers ── */
    .section-header {
        font-size: 1.05rem;
        font-weight: 600;
        color: #1e293b;
        border-bottom: 2px solid #0077b6;
        padding-bottom: 0.3rem;
        margin: 1.5rem 0 0.75rem 0;
    }

    /* ── Divider ── */
    hr { border: none; border-top: 1px solid #e3e8ef; margin: 1.5rem 0; }

    /* ── Plotly chart wrapper ── */
    .js-plotly-plot { border-radius: 8px; }
    </style>
    """,
    unsafe_allow_html=True,
)

# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------
from app.components.sidebar import render_sidebar
state = render_sidebar()

# ---------------------------------------------------------------------------
# Data loader (cached) — used only for landing-page summary metrics
# ---------------------------------------------------------------------------

@st.cache_data(show_spinner=False)
def _load_summary_metrics(data_dir: str) -> dict:
    """
    Load lightweight summary metrics from the processed data directory.

    Falls back to illustrative placeholder values if data is not present.

    Args:
        data_dir: Path to the processed Parquet directory.

    Returns:
        dict with keys: n_trials, n_conditions, n_countries, model_auc,
        model_threshold, data_vintage.
    """
    try:
        import pandas as pd
        studies_path = Path(data_dir) / "studies.parquet"
        if not studies_path.exists():
            raise FileNotFoundError(studies_path)

        studies = pd.read_parquet(studies_path, columns=["nct_id", "overall_status"])
        n_trials = len(studies)

        cond_path = Path(data_dir) / "conditions.parquet"
        n_conditions = (
            pd.read_parquet(cond_path, columns=["downcase_name"])["downcase_name"]
            .nunique()
            if cond_path.exists()
            else "—"
        )

        country_path = Path(data_dir) / "countries.parquet"
        n_countries = (
            pd.read_parquet(country_path, columns=["name"])["name"].nunique()
            if country_path.exists()
            else "—"
        )

        return {
            "n_trials":       f"{n_trials:,}",
            "n_conditions":   f"{n_conditions:,}" if isinstance(n_conditions, int) else n_conditions,
            "n_countries":    f"{n_countries:,}"  if isinstance(n_countries, int)  else n_countries,
            "model_auc":      "0.787",
            "model_threshold": "0.93",
            "data_vintage":   "AACT synthetic snapshot",
        }
    except Exception:
        # Graceful fallback — always show something on the landing page
        return {
            "n_trials":        "10,000",
            "n_conditions":    "1,200+",
            "n_countries":     "89",
            "model_auc":       "0.787",
            "model_threshold": "0.93",
            "data_vintage":    "Synthetic demo data",
        }


metrics = _load_summary_metrics(state["data_dir"])

# ---------------------------------------------------------------------------
# Hero header
# ---------------------------------------------------------------------------
col_logo, col_title = st.columns([1, 9])
with col_logo:
    st.markdown(
        "<div style='font-size:3.5rem; margin-top:0.3rem;'>🔬</div>",
        unsafe_allow_html=True,
    )
with col_title:
    st.markdown(
        """
        <h1 style='margin:0; color:#0077b6; font-size:2.2rem; font-weight:800;'>
            DecisionLENS
        </h1>
        <p style='margin:0; font-size:1.0rem; color:#64748b;'>
            AI-augmented clinical trial enrollment forecasting &amp;
            decision support
        </p>
        """,
        unsafe_allow_html=True,
    )

st.markdown("<hr>", unsafe_allow_html=True)

# ---------------------------------------------------------------------------
# KPI metrics row
# ---------------------------------------------------------------------------
st.markdown(
    "<div class='section-header'>Dataset &amp; Model Overview</div>",
    unsafe_allow_html=True,
)

m1, m2, m3, m4, m5 = st.columns(5)

m1.metric(
    label="Trials Analyzed",
    value=metrics["n_trials"],
    help="Total interventional trials in the processed AACT snapshot.",
)
m2.metric(
    label="Distinct Conditions",
    value=metrics["n_conditions"],
    help="Unique condition/indication strings in the dataset.",
)
m3.metric(
    label="Countries Covered",
    value=metrics["n_countries"],
    help="Countries with at least one trial site.",
)
m4.metric(
    label="Classifier ROC-AUC",
    value=metrics["model_auc"],
    help=(
        "XGBoost enrollment-completion classifier ROC-AUC on held-out test set "
        "(5-fold CV on synthetic AACT snapshot, March 2026)."
    ),
)
m5.metric(
    label="Decision Threshold",
    value=metrics["model_threshold"],
    delta="+0.43 vs default",
    help=(
        "P(complete) threshold tuned in notebook 05 for maximum F1-macro. "
        "F1-macro=0.660, F1-Terminated=0.368 at 0.93 vs 0.564/0.170 at 0.5."
    ),
)

st.markdown("<hr>", unsafe_allow_html=True)

# ---------------------------------------------------------------------------
# Navigation guide
# ---------------------------------------------------------------------------
st.markdown(
    "<div class='section-header'>Navigate the Dashboard</div>",
    unsafe_allow_html=True,
)

nav_col1, nav_col2 = st.columns(2)

with nav_col1:
    st.markdown(
        """
        <div class="nav-card">
            <h4>📈 1 — Enrollment Forecast</h4>
            <p>
                Configure a hypothetical trial (phase, condition, sites,
                countries) and get an instant enrollment-completion probability,
                predicted duration, and XGBoost feature importances.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.markdown(
        """
        <div class="nav-card">
            <h4>🗺️ 2 — Competitive Intelligence</h4>
            <p>
                Explore the competitive landscape for any condition: choropleth
                site map, competing-trial Gantt chart, sponsor treemap, and
                recruitment-saturation score by country.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

with nav_col2:
    st.markdown(
        """
        <div class="nav-card">
            <h4>🏥 3 — Investigator &amp; Site Insights</h4>
            <p>
                Rank trial sites by historical completion rate, visualise
                country-level enrollment efficiency, and get a data-driven
                site-allocation recommendation for your trial.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.markdown(
        """
        <div class="nav-card">
            <h4>🤖 4 — Eligibility Analyzer</h4>
            <p>
                Paste inclusion/exclusion criteria and get an AI-powered
                (Groq Llama 3.3 70B) risk assessment with simplification
                suggestions, population-impact estimates, and a downloadable
                executive briefing.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

st.markdown("<hr>", unsafe_allow_html=True)

# ---------------------------------------------------------------------------
# Problem statement / context strip
# ---------------------------------------------------------------------------
st.markdown(
    "<div class='section-header'>Why DecisionLENS?</div>",
    unsafe_allow_html=True,
)

prob_col1, prob_col2, prob_col3 = st.columns(3)

with prob_col1:
    st.info(
        "**~37% of clinical trials** fail to meet enrollment targets, adding "
        "an average of 7–12 months to development timelines and $35M+ in "
        "opportunity costs.",
        icon="⚠️",
    )
with prob_col2:
    st.success(
        "**XGBoost + Cox PH models** trained on 10 K+ AACT trials identify "
        "structural risk factors (phase, geography, sponsor history) with "
        "ROC-AUC 0.787 — before enrollment begins.",
        icon="✅",
    )
with prob_col3:
    st.info(
        "**Groq LLM integration** turns eligibility criteria text and model "
        "outputs into plain-English stakeholder briefings in seconds, "
        "closing the gap between data science and clinical ops decision-making.",
        icon="🤖",
    )

# ---------------------------------------------------------------------------
# Technical architecture expander
# ---------------------------------------------------------------------------
with st.expander("⚙️  Technical Architecture", expanded=False):
    st.markdown(
        """
        ```
        AACT / ClinicalTrials.gov flat files
                │
                ▼
        setup_data.py  →  data/processed/*.parquet
                │
                ├──► TrialDataPipeline (src/data_pipeline.py)
                │         feature engineering, cleaning
                │
                ├──► EnrollmentForecaster (src/models.py)
                │         XGBoost classifier + regressor
                │         Cox Proportional Hazards (lifelines)
                │         XGBoost feature importances
                │
                ├──► CompetitiveAnalyzer (src/competitive_intel.py)
                │         landscape metrics, choropleth, Gantt
                │
                ├──► InvestigatorAnalyzer (src/investigator_insights.py)
                │         site ranking, country heatmap, allocation
                │
                └──► EligibilityAnalyzer (src/genai_utils.py)
                          Groq API — llama-3.3-70b-versatile
                          structured JSON + streaming briefing
                                │
                                ▼
                    Streamlit Dashboard (app/)
                    ├── Home (streamlit_app.py)
                    ├── 1_Enrollment_Forecast.py
                    ├── 2_Competitive_Intelligence.py
                    ├── 3_Investigator_Insights.py
                    └── 4_Eligibility_Analyzer.py
        ```

        **Stack:** Python 3.11 · XGBoost 2.x · lifelines · SHAP · Plotly ·
        Streamlit 1.3x · Groq SDK · AACT flat-file snapshots
        """,
        unsafe_allow_html=False,
    )

# ---------------------------------------------------------------------------
# Data source note
# ---------------------------------------------------------------------------
st.markdown("<hr>", unsafe_allow_html=True)
st.caption(
    f"**Data:** {metrics['data_vintage']} — "
    "Aggregate Analysis of ClinicalTrials.gov (AACT), "
    "provided by the Clinical Trials Transformation Initiative (CTTI).  "
    "| **Models trained:** March 2026  "
    "| **AI:** Groq `llama-3.3-70b-versatile`  "
    "| [AACT download](https://aact.ctti-clinicaltrials.org/pipe_files)"
)
