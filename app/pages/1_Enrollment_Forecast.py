"""
1_Enrollment_Forecast.py — DecisionLENS Enrollment Forecast page.

Configure a hypothetical trial and get:
  • P(enrollment completes) via the XGBoost classifier
  • Predicted enrollment duration via the XGBoost regressor
  • Predicted enrollment duration chart vs phase median
  • XGBoost built-in feature importances
  • Benchmark comparison vs historical trials in same phase
  • Table of 10 most similar historical trials
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path

ROOT = Path(__file__).parent.parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from dotenv import load_dotenv
load_dotenv(ROOT / ".env")

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

log = logging.getLogger(__name__)

st.set_page_config(
    page_title="Enrollment Forecast — DecisionLENS",
    page_icon="📈",
    layout="wide",
)

from app.components._theme import apply_theme
from app.components.sidebar import render_sidebar
from app.components.charts import risk_gauge

apply_theme()

# ---------------------------------------------------------------------------
# Constants — must match src/models.py feature lists exactly
# ---------------------------------------------------------------------------

PHASE_MAP: dict[str, float] = {
    "Phase 1":         1.0,
    "Phase 1/Phase 2": 1.5,
    "Phase 2":         2.0,
    "Phase 2/Phase 3": 2.5,
    "Phase 3":         3.0,
    "Phase 4":         4.0,
}

SPONSOR_TYPE_MAP: dict[str, str] = {
    "Industry":    "industry",
    "Government":  "government",
    "Academic":    "academic",
    "Other":       "other",
}

INTERVENTION_MODEL_OPTIONS: list[str] = [
    "Parallel Assignment",
    "Single Group Assignment",
    "Crossover Assignment",
    "Sequential Assignment",
    "Factorial Assignment",
]

MASKING_OPTIONS: list[str] = [
    "None (Open Label)",
    "Single",
    "Double",
    "Triple",
    "Quadruple",
]

CLF_THRESHOLD: float = 0.93   # mirrors src/models.py

# Human-readable labels for pipeline-prefixed feature names.
# Keys are the exact strings produced by sklearn's ColumnTransformer.
FEATURE_DISPLAY_NAMES: dict[str, str] = {
    "num__enrollment":                        "Target Enrollment Size",
    "num__n_facilities":                      "Number of Sites",
    "num__phase_numeric":                     "Trial Phase",
    "num__sponsor_historical_performance":    "Sponsor Track Record",
    "num__competing_trials_count":            "Competing Trials",
    "num__n_countries":                       "Number of Countries",
    "num__n_eligibility_criteria":            "Eligibility Criteria Count",
    "num__geographic_concentration":          "Geographic Concentration (HHI)",
    "num__condition_prevalence_proxy":        "Condition Prevalence",
    "num__enrollment_type_is_actual":         "Enrollment Type (Actual)",
    "bool__is_multicountry":                  "Multi-country Trial",
    "num__is_multicountry":                   "Multi-country Trial",
    "cat__sponsor_type_industry":             "Industry Sponsor",
    "cat__sponsor_type_government":           "Government Sponsor",
    "cat__sponsor_type_academic":             "Academic Sponsor",
    "cat__intervention_model_Parallel Assignment":    "Parallel Design",
    "cat__intervention_model_Crossover Assignment":   "Crossover Design",
    "cat__intervention_model_Single Group Assignment": "Single-arm Design",
    "cat__masking_Double":                    "Double-blind Masking",
    "cat__masking_Triple":                    "Triple-blind Masking",
    "cat__masking_None (Open Label)":         "Open Label",
}

# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------
state = render_sidebar()

# ---------------------------------------------------------------------------
# Cached loaders
# ---------------------------------------------------------------------------

@st.cache_resource(show_spinner="Loading model…")
def _load_model(model_path: str):
    """Load EnrollmentForecaster from disk via joblib (cached for the session)."""
    import joblib
    try:
        return joblib.load(model_path)
    except FileNotFoundError:
        log.warning("Model file not found: %s", model_path)
        return None
    except Exception as exc:
        log.warning("Model load failed (%s). Returning None.", exc)
        return None


@st.cache_data(show_spinner="Loading trial data…")
def _load_df(data_dir: str) -> pd.DataFrame:
    """Load the modeling DataFrame for benchmark comparison."""
    try:
        from src.data_pipeline import TrialDataPipeline
        pipeline = TrialDataPipeline()
        tables = pipeline.load_raw_data(data_dir)
        return pipeline.engineer_features(tables)
    except Exception as exc:
        log.warning("Data load failed (%s). Returning empty DataFrame.", exc)
        return pd.DataFrame()


MODEL_PATH = str(ROOT / "models" / "forecaster.joblib")
model = _load_model(MODEL_PATH)
df_full = _load_df(state["data_dir"])

# ---------------------------------------------------------------------------
# Page header
# ---------------------------------------------------------------------------
st.markdown("## 📈 Enrollment Forecast")
st.markdown(
    "Configure a hypothetical trial below.  The XGBoost classifier predicts "
    "enrollment-completion probability; the regressor predicts expected duration; "
    "and feature importances show the top risk drivers."
)
st.markdown("<hr>", unsafe_allow_html=True)

if model is None:
    st.warning(
        "No trained model found at `models/forecaster.joblib`.  "
        "Run notebook 03 to train and save the model first.",
        icon="⚠️",
    )

# ---------------------------------------------------------------------------
# Layout: left (inputs) | right (results)
# ---------------------------------------------------------------------------
col_in, col_out = st.columns([1, 2], gap="large")

# ── Input panel ──────────────────────────────────────────────────────────
with col_in:
    st.markdown(
        "<div class='section-header'>Trial Configuration</div>",
        unsafe_allow_html=True,
    )

    phase = st.selectbox("Phase", options=list(PHASE_MAP.keys()), index=2)

    condition_input = st.text_input(
        "Condition / Indication",
        value=state["condition_query"] or "Non-Small Cell Lung Cancer",
        help="Used only for benchmark filtering. Does not affect the model directly.",
    )

    target_enrollment = st.number_input(
        "Target enrollment (# patients)",
        min_value=10, max_value=50_000, value=300, step=10,
    )

    n_sites = st.slider("Number of sites", min_value=1, max_value=500, value=40)
    n_countries = st.slider("Number of countries", min_value=1, max_value=50, value=5)

    sponsor_type_label = st.radio(
        "Sponsor type",
        options=list(SPONSOR_TYPE_MAP.keys()),
        index=0,
        horizontal=True,
    )

    intervention_model_val = st.selectbox(
        "Study design (intervention model)",
        options=INTERVENTION_MODEL_OPTIONS,
        index=0,
    )

    masking_val = st.selectbox("Masking (blinding)", options=MASKING_OPTIONS, index=2)

    n_eligibility_criteria = st.slider(
        "Number of eligibility criteria", min_value=5, max_value=60, value=20
    )

    has_dmc = st.checkbox("Data Monitoring Committee (DMC)", value=True)

    st.markdown("<br>", unsafe_allow_html=True)
    run_btn = st.button("🔮 Run Forecast", use_container_width=True, type="primary")

# ── Build input row ───────────────────────────────────────────────────────

def _build_input_row() -> pd.DataFrame:
    """Construct a single-row DataFrame matching the model's expected features."""
    geographic_conc = round(1.0 / max(n_countries, 1), 4)
    sponsor_type_val = SPONSOR_TYPE_MAP[sponsor_type_label]

    return pd.DataFrame([{
        "phase_numeric":              PHASE_MAP[phase],
        "n_facilities":               n_sites,
        "n_countries":                n_countries,
        "n_eligibility_criteria":     n_eligibility_criteria,
        "geographic_concentration":   geographic_conc,
        "condition_prevalence_proxy": 50.0,           # dataset median
        "sponsor_historical_performance": 0.75,       # neutral baseline
        "competing_trials_count":     10,             # typical
        "enrollment":                 target_enrollment,
        "enrollment_type_is_actual":  0,
        "sponsor_type":               sponsor_type_val,
        "intervention_model":         intervention_model_val,
        "masking":                    masking_val,
        "is_multicountry":            int(n_countries > 1),
    }])


# ── Results panel ─────────────────────────────────────────────────────────
with col_out:
    if not run_btn:
        st.info(
            "Configure the trial parameters on the left and click **Run Forecast**.",
            icon="👈",
        )
    elif model is None:
        st.error("Model not available. See warning above.", icon="❌")
    else:
        input_row = _build_input_row()

        with st.spinner("Running prediction…"):
            try:
                preds = model.predict(input_row)
                p_completed = float(preds["p_completed"].iloc[0])
                pred_duration = float(preds["pred_duration_days"].iloc[0])
            except Exception as exc:
                st.error(f"Prediction error: {exc}")
                st.stop()

        # ── Metric summary row ──
        mc1, mc2, mc3 = st.columns(3)
        mc1.metric(
            "P(Enrollment Completes)",
            f"{p_completed:.1%}",
            delta=f"{'Above' if p_completed >= CLF_THRESHOLD else 'Below'} threshold",
        )
        mc2.metric(
            "Predicted Duration",
            f"{pred_duration/30.4:.1f} mo",
            help=f"{pred_duration:.0f} days",
        )
        mc3.metric(
            "Risk Label",
            "Low Risk" if p_completed >= CLF_THRESHOLD else
            "Moderate" if p_completed >= 0.50 else "High Risk",
        )

        st.markdown("<hr>", unsafe_allow_html=True)

        # ── Risk gauge ──
        st.markdown(
            "<div class='section-header'>Probability of Successful Completion</div>",
            unsafe_allow_html=True,
        )
        fig_gauge = risk_gauge(p_completed, threshold=CLF_THRESHOLD)
        st.plotly_chart(fig_gauge, use_container_width=True)

        # ── Duration prediction chart ──
        st.markdown(
            "<div class='section-header'>Predicted Enrollment Duration</div>",
            unsafe_allow_html=True,
        )
        try:
            pred_months = pred_duration / 30.4

            # Phase median from historical data (same phase, valid durations only)
            phase_val_num = PHASE_MAP[phase]
            phase_median_months: float | None = None
            if not df_full.empty:
                phase_hist = df_full[
                    df_full["phase_numeric"].eq(phase_val_num)
                    & df_full["enrollment_duration_days"].notna()
                    & (df_full["enrollment_duration_days"] > 0)
                ]["enrollment_duration_days"]
                if not phase_hist.empty:
                    phase_median_months = phase_hist.median() / 30.4

            y_max = max(pred_months, phase_median_months or 0) * 1.35 + 4

            fig_dur = go.Figure()
            fig_dur.add_trace(go.Bar(
                x=["This Trial"],
                y=[pred_months],
                marker_color="#0077b6",
                width=0.35,
                text=[f"{pred_months:.1f} mo"],
                textposition="outside",
                hovertemplate="Predicted: %{y:.1f} months<extra></extra>",
            ))
            if phase_median_months is not None:
                fig_dur.add_hline(
                    y=phase_median_months,
                    line_dash="dash",
                    line_color="#d32f2f",
                    annotation_text=f"{phase} median: {phase_median_months:.1f} mo",
                    annotation_position="right",
                )
            fig_dur.update_layout(
                title="Predicted Enrollment Duration (months)",
                yaxis_title="Enrollment Duration (months)",
                yaxis=dict(range=[0, y_max]),
                xaxis=dict(showticklabels=False),
                height=320,
                template="plotly_white",
                margin=dict(l=10, r=120, t=50, b=20),
                showlegend=False,
            )
            st.plotly_chart(fig_dur, use_container_width=True)
            st.caption("Survival Model C-index: 0.849 (Cox Proportional Hazards on labeled AACT trials)")
        except Exception as exc:
            st.info(f"Duration chart unavailable: {exc}", icon="ℹ️")

        # ── Feature importances (XGBoost built-in) ──
        st.markdown(
            "<div class='section-header'>Top Risk Drivers (Feature Importance)</div>",
            unsafe_allow_html=True,
        )
        try:
            fi = model.feature_importances
            fi_top = fi.nlargest(12, "clf_importance").sort_values("clf_importance").copy()
            fi_top["label"] = fi_top["feature"].map(
                lambda f: FEATURE_DISPLAY_NAMES.get(f, f.split("__", 1)[-1].replace("_", " ").title())
            )
            median_imp = fi_top["clf_importance"].median()
            fig_fi = go.Figure(go.Bar(
                x=fi_top["clf_importance"],
                y=fi_top["label"],
                orientation="h",
                marker_color=[
                    "#d32f2f" if v >= median_imp else "#0077b6"
                    for v in fi_top["clf_importance"]
                ],
                hovertemplate="%{y}: %{x:.4f}<extra></extra>",
            ))
            fig_fi.update_layout(
                xaxis_title="Feature importance (XGBoost gain)",
                yaxis=dict(autorange="reversed"),
                height=380,
                template="plotly_white",
                margin=dict(l=10, r=20, t=20, b=40),
            )
            st.plotly_chart(fig_fi, use_container_width=True)
        except Exception as exc:
            st.info(f"Feature importance unavailable: {exc}", icon="ℹ️")

# ---------------------------------------------------------------------------
# Bottom section: benchmark comparison + similar trials
# ---------------------------------------------------------------------------
if run_btn and model is not None and not df_full.empty:
    st.markdown("<hr>", unsafe_allow_html=True)
    st.markdown(
        "<div class='section-header'>Benchmark vs Historical Trials</div>",
        unsafe_allow_html=True,
    )

    bench_col, similar_col = st.columns([1, 2])

    # Filter to same phase
    phase_val = PHASE_MAP[phase]
    bench_df = df_full[
        df_full["phase_numeric"].eq(phase_val)
        & df_full["enrollment_duration_days"].notna()
        & (df_full["enrollment_duration_days"] > 0)
    ]

    with bench_col:
        if not bench_df.empty:
            fig_box = go.Figure()
            fig_box.add_trace(go.Box(
                y=bench_df["enrollment_duration_days"],
                name=phase,
                marker_color="#90e0ef",
                boxmean=True,
            ))
            fig_box.add_trace(go.Scatter(
                x=[phase], y=[pred_duration],
                mode="markers",
                marker=dict(color="#d32f2f", size=12, symbol="diamond"),
                name="This trial",
            ))
            fig_box.update_layout(
                title="Duration vs Historical (same phase)",
                yaxis_title="Enrollment duration (days)",
                height=360, template="plotly_white",
                margin=dict(l=10, r=10, t=50, b=40),
                showlegend=True,
            )
            st.plotly_chart(fig_box, use_container_width=True)
        else:
            st.info("No historical data available for benchmark comparison.")

    # 10 most similar trials (nearest by phase, n_countries, enrollment)
    with similar_col:
        if not bench_df.empty:
            candidates = bench_df.copy()
            candidates["_dist"] = (
                (candidates["n_countries"] - n_countries).abs() * 0.3
                + (candidates["n_facilities"] - n_sites).abs() * 0.01
                + (candidates["enrollment"] - target_enrollment).abs() / max(target_enrollment, 1)
            )
            top_10 = candidates.nsmallest(10, "_dist")

            display_cols = [c for c in [
                "nct_id", "phase_numeric", "n_facilities", "n_countries",
                "enrollment", "enrollment_duration_days", "enrollment_met_target",
            ] if c in top_10.columns]

            rename_map = {
                "nct_id": "NCT ID",
                "phase_numeric": "Phase",
                "n_facilities": "Sites",
                "n_countries": "Countries",
                "enrollment": "Enrollment",
                "enrollment_duration_days": "Duration (d)",
                "enrollment_met_target": "Met Target",
            }

            show_df = top_10[display_cols].rename(columns=rename_map).reset_index(drop=True)
            st.markdown("**10 Most Similar Historical Trials**")
            st.dataframe(show_df, use_container_width=True, height=360)

# ---------------------------------------------------------------------------
# Methodology expander
# ---------------------------------------------------------------------------
with st.expander("⚙️ Methodology", expanded=False):
    st.markdown(
        """
        **Classifier** (XGBoost): Predicts *P(trial completes enrollment)* from
        structural trial attributes.  Decision threshold = 0.93 (tuned in notebook 05
        to maximise F1-macro; ROC-AUC = 0.787 on synthetic AACT snapshot).

        **Regressor** (XGBoost): Predicts enrollment duration in days.
        Predictions are capped at the 99th-percentile of training durations
        to prevent outlier estimates.

        **Duration chart**: XGBoost regressor prediction (months) compared against
        the median enrollment duration for historical trials in the same phase.
        Cox Proportional Hazards C-index = 0.849 (survival model fit on labeled
        AACT trials with right-censoring for actively recruiting studies).

        **Feature importances** (XGBoost built-in gain): Shows which features
        most influence the classifier across training.  Higher gain = stronger
        discriminating power.  Red bars mark the top half by importance.

        **Inputs not in model**: "Condition" is used only for benchmark filtering.
        "Has DMC" is informational. Features without direct user inputs
        (e.g., `condition_prevalence_proxy`) use dataset-median defaults.
        """
    )
