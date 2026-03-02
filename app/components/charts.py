"""
charts.py — DecisionLENS reusable Plotly chart components.

All functions accept plain Python / pandas / numpy data structures and
return ``plotly.graph_objects.Figure`` objects, so they can be rendered
with ``st.plotly_chart(fig, use_container_width=True)`` in any page.

Available functions
-------------------
risk_gauge(p_completed, threshold)
    Bullet/gauge showing enrollment-completion probability vs threshold.

enrollment_curve(predictions_df, ...)
    Bar+line chart of predicted duration with risk-colour coding.

shap_waterfall(shap_values, feature_names, row_idx, ...)
    Horizontal waterfall for one row of SHAP values.

competition_map(country_counts_df, ...)
    Choropleth world map of competing active trial sites.

competition_timeline(trials_df, ...)
    Horizontal Gantt timeline of competing trials.

site_heatmap(country_performance_df, ...)
    Heatmap of country-level enrollment performance metrics.
"""

from __future__ import annotations

import logging
from typing import Optional, Sequence

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots

log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Shared style constants
# ---------------------------------------------------------------------------

#: Plotly template applied to all charts for visual consistency.
TEMPLATE: str = "plotly_white"

#: Brand colour palette — matches notebooks and src modules.
COLOUR_SUCCESS: str = "#0077b6"   # blue  — completed / low risk
COLOUR_WARNING: str = "#f9a825"   # amber — borderline
COLOUR_DANGER:  str = "#d32f2f"   # red   — high risk / terminated

#: Default figure height (px) used across all charts.
DEFAULT_HEIGHT: int = 420

#: Phase colour palette (mirrors competitive_intel.py / notebooks).
PHASE_COLORS: dict[str, str] = {
    "Phase 1":          "#caf0f8",
    "Phase 1/Phase 2":  "#90e0ef",
    "Phase 2":          "#00b4d8",
    "Phase 2/Phase 3":  "#0096c7",
    "Phase 3":          "#0077b6",
    "Phase 4":          "#023e8a",
    "N/A":              "#adb5bd",
}


# ---------------------------------------------------------------------------
# 1. risk_gauge
# ---------------------------------------------------------------------------

def risk_gauge(
    p_completed: float,
    threshold: float = 0.93,
    title: str = "Enrollment Completion Probability",
    height: int = 280,
) -> go.Figure:
    """
    Bullet-style gauge showing P(enrollment completes) vs the decision threshold.

    The needle colour transitions red → amber → blue as the probability crosses
    50% and 93% (the default operating threshold tuned in notebook 05).

    Args:
        p_completed: Scalar in [0, 1] — output of EnrollmentForecaster.predict().
        threshold:   Decision threshold above which a trial is predicted to
                     complete.  Defaults to CLF_THRESHOLD = 0.93.
        title:       Chart title displayed below the gauge arc.
        height:      Figure height in pixels.

    Returns:
        Plotly Figure containing a single indicator (gauge) trace.

    Example::

        fig = risk_gauge(0.87, threshold=0.93)
        st.plotly_chart(fig, use_container_width=True)
    """
    p = float(np.clip(p_completed, 0.0, 1.0))

    if p >= threshold:
        bar_colour  = COLOUR_SUCCESS
        risk_label  = "Low Risk"
    elif p >= 0.50:
        bar_colour  = COLOUR_WARNING
        risk_label  = "Moderate Risk"
    else:
        bar_colour  = COLOUR_DANGER
        risk_label  = "High Risk"

    fig = go.Figure(go.Indicator(
        mode="gauge+number+delta",
        value=round(p * 100, 1),
        number={"suffix": "%", "font": {"size": 36}},
        delta={
            "reference": threshold * 100,
            "suffix":    "%",
            "increasing": {"color": COLOUR_SUCCESS},
            "decreasing": {"color": COLOUR_DANGER},
        },
        title={"text": f"{title}<br><span style='font-size:13px'>{risk_label}</span>"},
        gauge={
            "axis": {
                "range": [0, 100],
                "ticksuffix": "%",
                "tickfont": {"size": 11},
            },
            "bar":  {"color": bar_colour, "thickness": 0.25},
            "steps": [
                {"range": [0, 50],          "color": "#fdecea"},
                {"range": [50, threshold * 100], "color": "#fff9e6"},
                {"range": [threshold * 100, 100], "color": "#e8f5e9"},
            ],
            "threshold": {
                "line": {"color": "#1a1a1a", "width": 3},
                "thickness": 0.85,
                "value": threshold * 100,
            },
        },
    ))

    fig.update_layout(
        height=height,
        margin=dict(l=20, r=20, t=60, b=10),
        template=TEMPLATE,
    )
    return fig


# ---------------------------------------------------------------------------
# 2. enrollment_curve
# ---------------------------------------------------------------------------

def enrollment_curve(
    predictions_df: pd.DataFrame,
    threshold: float = 0.93,
    max_rows: int = 50,
    height: int = DEFAULT_HEIGHT,
    title: str = "Predicted Enrollment Duration by Trial",
) -> go.Figure:
    """
    Horizontal bar chart of predicted enrollment durations, colour-coded by
    completion probability.

    Expects a DataFrame with at minimum these columns (output of
    ``EnrollmentForecaster.predict()``):

        - ``nct_id``              : str  — trial identifier (used as y-axis label)
        - ``pred_duration_days``  : float — predicted enrollment duration
        - ``p_completed``         : float — P(trial completes), range [0, 1]

    Optional columns used for hover tooltip if present:
        - ``phase``  : str
        - ``sponsor_type`` : str

    Args:
        predictions_df: Output of EnrollmentForecaster.predict() (with nct_id).
        threshold:      Decision threshold for colour boundary.
        max_rows:       Maximum number of trials to plot (sorted desc by duration).
        height:         Figure height in pixels.
        title:          Chart title.

    Returns:
        Plotly Figure with a horizontal bar trace and a secondary scatter trace
        showing P(complete) as circle markers on a right-hand y-axis.
    """
    df = predictions_df.copy()

    required = {"nct_id", "pred_duration_days", "p_completed"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"enrollment_curve: missing columns {missing}")

    df = (
        df.dropna(subset=["pred_duration_days"])
        .sort_values("pred_duration_days", ascending=False)
        .head(max_rows)
        .reset_index(drop=True)
    )

    colours = [
        COLOUR_SUCCESS if p >= threshold
        else COLOUR_WARNING if p >= 0.5
        else COLOUR_DANGER
        for p in df["p_completed"]
    ]

    # Hover text
    hover_parts = ["<b>%{y}</b>", "Duration: %{x:.0f} d"]
    if "phase" in df.columns:
        hover_parts.append("Phase: " + df["phase"].fillna("N/A").astype(str))
    hover_template = "<br>".join(hover_parts) + "<extra></extra>"

    fig = make_subplots(specs=[[{"secondary_y": True}]])

    fig.add_trace(
        go.Bar(
            x=df["pred_duration_days"],
            y=df["nct_id"],
            orientation="h",
            marker_color=colours,
            name="Pred. Duration (days)",
            hovertemplate=hover_template,
        ),
        secondary_y=False,
    )

    fig.add_trace(
        go.Scatter(
            x=df["p_completed"],
            y=df["nct_id"],
            mode="markers",
            marker=dict(
                symbol="circle",
                size=9,
                color=df["p_completed"],
                colorscale=[[0, COLOUR_DANGER], [0.5, COLOUR_WARNING], [1, COLOUR_SUCCESS]],
                cmin=0, cmax=1,
                colorbar=dict(
                    title="P(complete)",
                    len=0.6,
                    x=1.05,
                    tickformat=".0%",
                ),
                line=dict(color="#555", width=0.5),
            ),
            name="P(complete)",
            xaxis="x2",
            hovertemplate="<b>%{y}</b><br>P(complete): %{x:.2f}<extra></extra>",
        ),
        secondary_y=True,
    )

    fig.add_vline(
        x=threshold,
        line_dash="dot",
        line_color="#999",
        annotation_text=f"Threshold {threshold:.0%}",
        annotation_position="top right",
        secondary_y=True,
    )

    fig.update_layout(
        title=title,
        xaxis=dict(title="Predicted Duration (days)", side="bottom"),
        xaxis2=dict(title="P(complete)", range=[0, 1], tickformat=".0%", overlaying="x", side="top"),
        yaxis=dict(autorange="reversed"),
        height=height,
        template=TEMPLATE,
        margin=dict(l=10, r=80, t=60, b=40),
        legend=dict(orientation="h", y=-0.12),
        barmode="overlay",
    )
    return fig


# ---------------------------------------------------------------------------
# 3. shap_waterfall
# ---------------------------------------------------------------------------

def shap_waterfall(
    shap_values: np.ndarray | Sequence[float],
    feature_names: Sequence[str],
    expected_value: float = 0.0,
    row_idx: int = 0,
    max_features: int = 12,
    height: int = DEFAULT_HEIGHT,
    title: str = "SHAP Feature Contributions",
) -> go.Figure:
    """
    Horizontal waterfall chart of SHAP values for a single prediction row.

    Positive SHAP values (increase P(complete)) are blue; negative values
    (decrease P(complete)) are red.

    Args:
        shap_values:    2-D array of shape (n_samples, n_features) **or** a 1-D
                        array for a single row.  When 2-D, ``row_idx`` selects
                        the row.
        feature_names:  Sequence of feature name strings aligned with the last
                        axis of ``shap_values``.
        expected_value: SHAP base value (E[f(X)]); used to annotate the chart
                        baseline.
        row_idx:        Which sample row to display when ``shap_values`` is 2-D.
                        Ignored for 1-D input.
        max_features:   Maximum number of features to display (top by |SHAP|).
        height:         Figure height in pixels.
        title:          Chart title.

    Returns:
        Plotly Figure with a waterfall trace and annotations for base and
        final prediction values.

    Example::

        shap_info = model.explain(df.head(200))
        fig = shap_waterfall(
            shap_info["shap_values"],
            shap_info["feature_names"],
            expected_value=shap_info["expected_value"],
        )
    """
    sv = np.asarray(shap_values, dtype=float)
    if sv.ndim == 2:
        sv = sv[row_idx]

    names  = list(feature_names)
    if len(names) != len(sv):
        raise ValueError(
            f"shap_waterfall: len(feature_names)={len(names)} != "
            f"len(shap_values)={len(sv)}"
        )

    # Select top-N features by |SHAP|
    order = np.argsort(np.abs(sv))[::-1][:max_features]
    sv_sel    = sv[order]
    name_sel  = [names[i] for i in order]

    # Sort by SHAP value for a readable waterfall (largest positive at top)
    sort_idx   = np.argsort(sv_sel)
    sv_plot    = sv_sel[sort_idx]
    name_plot  = [name_sel[i] for i in sort_idx]

    colours = [
        COLOUR_SUCCESS if v >= 0 else COLOUR_DANGER
        for v in sv_plot
    ]

    fig = go.Figure(go.Bar(
        x=sv_plot,
        y=name_plot,
        orientation="h",
        marker_color=colours,
        hovertemplate="<b>%{y}</b><br>SHAP: %{x:+.4f}<extra></extra>",
        name="SHAP value",
    ))

    fig.add_vline(x=0, line_color="#333", line_width=1)

    fig.add_annotation(
        x=0, y=1.06, xref="x", yref="paper",
        text=f"Base value: {expected_value:+.3f}",
        showarrow=False, font=dict(size=11, color="#666"),
    )

    fig.update_layout(
        title=title,
        xaxis_title="SHAP value (impact on log-odds of completion)",
        yaxis_title="Feature",
        height=height,
        template=TEMPLATE,
        margin=dict(l=10, r=20, t=60, b=40),
    )
    return fig


# ---------------------------------------------------------------------------
# 4. competition_map
# ---------------------------------------------------------------------------

def competition_map(
    country_counts_df: pd.DataFrame,
    count_col: str = "n_active_trials",
    country_col: str = "country",
    color_label: str = "Active Competing Trials",
    height: int = DEFAULT_HEIGHT,
    title: str = "Active Competing Trial Sites by Country",
) -> go.Figure:
    """
    Choropleth world map of active competing clinical trials by country.

    Typically fed directly from ``CompetitiveAnalyzer.get_landscape()``
    or its underlying country-count table.

    Args:
        country_counts_df: DataFrame with at least two columns:
                           ``country_col`` (str country names) and
                           ``count_col`` (int/float counts).
        count_col:   Column containing the count to choropleth.
        country_col: Column containing country name strings (Plotly
                     ``locationmode='country names'``).
        color_label: Legend / colour-bar label.
        height:      Figure height in pixels.
        title:       Chart title.

    Returns:
        Plotly Figure with a choropleth trace.

    Example::

        landscape = analyzer.get_landscape("Breast Cancer")
        fig = competition_map(landscape["country_trial_counts"])
        st.plotly_chart(fig, use_container_width=True)
    """
    df = country_counts_df.copy().dropna(subset=[country_col, count_col])

    if df.empty:
        log.warning("competition_map: empty country_counts_df — returning blank figure.")
        fig = go.Figure()
        fig.update_layout(title=title, height=height, template=TEMPLATE)
        return fig

    df[count_col] = pd.to_numeric(df[count_col], errors="coerce").fillna(0)

    fig = go.Figure(go.Choropleth(
        locations=df[country_col],
        locationmode="country names",
        z=df[count_col],
        colorscale=[
            [0.0, "#e8f4fd"],
            [0.3, "#90e0ef"],
            [0.6, "#0096c7"],
            [1.0, "#023e8a"],
        ],
        colorbar=dict(title=color_label, len=0.6),
        hovertemplate=(
            "<b>%{location}</b><br>"
            f"{color_label}: " + "%{z}<extra></extra>"
        ),
        marker_line_color="#ffffff",
        marker_line_width=0.5,
    ))

    fig.update_layout(
        title=title,
        height=height,
        template=TEMPLATE,
        geo=dict(
            showframe=False,
            showcoastlines=True,
            coastlinecolor="#ccc",
            projection_type="natural earth",
            bgcolor="rgba(0,0,0,0)",
        ),
        margin=dict(l=0, r=0, t=50, b=0),
    )
    return fig


# ---------------------------------------------------------------------------
# 5. competition_timeline
# ---------------------------------------------------------------------------

def competition_timeline(
    trials_df: pd.DataFrame,
    start_col: str = "start_date",
    end_col: str   = "primary_completion_date",
    id_col: str    = "nct_id",
    phase_col: Optional[str] = "phase",
    status_col: Optional[str] = "overall_status",
    max_trials: int = 40,
    height: int = DEFAULT_HEIGHT,
    title: str = "Competing Trial Timeline",
) -> go.Figure:
    """
    Horizontal Gantt chart of competing clinical trials over time.

    Each bar represents one trial.  Bars are coloured by phase if
    ``phase_col`` is present, otherwise by status.  Trials with missing
    start or end dates are excluded automatically.

    Args:
        trials_df:  DataFrame containing at least ``id_col``, ``start_col``,
                    and ``end_col`` columns.  Dates may be strings
                    (ISO-8601), datetime objects, or pandas Timestamps.
        start_col:  Column name for trial start date.
        end_col:    Column name for trial end or primary-completion date.
        id_col:     Column name used as the y-axis label.
        phase_col:  Optional column for trial phase (used for colour coding).
        status_col: Optional column for overall status (used in hover text).
        max_trials: Maximum number of trials to display.  Trials with the
                    most recent start dates are shown if the DataFrame has
                    more than ``max_trials`` rows.
        height:     Figure height in pixels.
        title:      Chart title.

    Returns:
        Plotly Figure with one horizontal bar (Gantt-style) per trial.
    """
    df = trials_df.copy()

    for col in [start_col, end_col]:
        df[col] = pd.to_datetime(df[col], errors="coerce")

    df = df.dropna(subset=[start_col, end_col])
    df = df[df[end_col] > df[start_col]]

    if df.empty:
        log.warning("competition_timeline: no valid date rows — returning blank figure.")
        fig = go.Figure()
        fig.update_layout(title=title, height=height, template=TEMPLATE)
        return fig

    # Sort by start_date, keep most recent max_trials
    df = (
        df.sort_values(start_col, ascending=False)
        .head(max_trials)
        .sort_values(start_col, ascending=True)
        .reset_index(drop=True)
    )

    # Assign colours
    if phase_col and phase_col in df.columns:
        colour_map = PHASE_COLORS
        df["_colour"] = df[phase_col].map(colour_map).fillna(PHASE_COLORS["N/A"])
        colour_legend = "Phase"
    else:
        df["_colour"] = COLOUR_SUCCESS
        colour_legend = None

    # Build Gantt as a Bar chart (Plotly express timeline struggles with >20 categories)
    fig = go.Figure()

    for _, row in df.iterrows():
        duration_days = (row[end_col] - row[start_col]).days

        hover_text = (
            f"<b>{row[id_col]}</b><br>"
            f"Start: {row[start_col].date()}<br>"
            f"End:   {row[end_col].date()}<br>"
            f"Duration: {duration_days} days"
        )
        if phase_col and phase_col in df.columns:
            hover_text += f"<br>Phase: {row.get(phase_col, 'N/A')}"
        if status_col and status_col in df.columns:
            hover_text += f"<br>Status: {row.get(status_col, 'N/A')}"

        fig.add_trace(go.Bar(
            x=[row[end_col] - row[start_col]],
            y=[row[id_col]],
            base=[row[start_col]],
            orientation="h",
            marker_color=row["_colour"],
            showlegend=False,
            hovertemplate=hover_text + "<extra></extra>",
        ))

    # Add phase legend traces (invisible dummy bars)
    if phase_col and phase_col in df.columns:
        for phase, colour in PHASE_COLORS.items():
            if phase in df[phase_col].values:
                fig.add_trace(go.Bar(
                    x=[None], y=[None],
                    orientation="h",
                    marker_color=colour,
                    name=phase,
                    showlegend=True,
                ))

    fig.update_layout(
        title=title,
        xaxis=dict(type="date", title=""),
        yaxis=dict(title="", autorange="reversed"),
        height=height,
        template=TEMPLATE,
        barmode="stack",
        margin=dict(l=10, r=20, t=60, b=40),
        legend=dict(title=colour_legend, orientation="h", y=-0.15),
    )
    return fig


# ---------------------------------------------------------------------------
# 6. site_heatmap
# ---------------------------------------------------------------------------

def site_heatmap(
    country_performance_df: pd.DataFrame,
    country_col: str = "country",
    metrics: Optional[Sequence[str]] = None,
    height: int = DEFAULT_HEIGHT,
    title: str = "Country-Level Enrollment Performance",
) -> go.Figure:
    """
    Annotated heatmap of country-level enrollment performance metrics.

    Typically fed from ``InvestigatorAnalyzer.get_country_performance()``.

    Expected columns (all numeric except ``country_col``):
        - ``country``          : str
        - ``n_trials``         : int   — total trials in country
        - ``completion_rate``  : float — fraction of trials that completed
        - ``mean_duration_days`` : float — mean enrollment duration
        - ``n_sites``          : int   — distinct facility count

    Args:
        country_performance_df: DataFrame with one row per country and
                                numeric performance metric columns.
        country_col:  Column name for country labels (y-axis).
        metrics:      Which numeric columns to include as heatmap x-axis.
                      Defaults to all numeric columns except ``country_col``.
        height:       Figure height in pixels.
        title:        Chart title.

    Returns:
        Plotly Figure with a single annotated heatmap trace.

    Example::

        perf = ia.get_country_performance("Breast Cancer")
        fig  = site_heatmap(perf)
        st.plotly_chart(fig, use_container_width=True)
    """
    df = country_performance_df.copy().dropna(subset=[country_col])

    if metrics is None:
        metrics = [c for c in df.columns if c != country_col and pd.api.types.is_numeric_dtype(df[c])]

    if not metrics:
        log.warning("site_heatmap: no numeric metric columns found — returning blank figure.")
        fig = go.Figure()
        fig.update_layout(title=title, height=height, template=TEMPLATE)
        return fig

    # Keep top 25 countries by first metric (typically n_trials)
    sort_col = metrics[0]
    df = df.sort_values(sort_col, ascending=False).head(25).reset_index(drop=True)

    # Z-score normalise each column so colours are comparable across metrics
    z_matrix = np.zeros((len(df), len(metrics)))
    for j, col in enumerate(metrics):
        vals = df[col].fillna(0).astype(float).values
        std = vals.std()
        if std > 0:
            z_matrix[:, j] = (vals - vals.mean()) / std
        else:
            z_matrix[:, j] = 0.0

    # Raw values for annotation text
    text_matrix = df[metrics].fillna(0).round(1).astype(str).values

    # Pretty-print metric column headers
    def _fmt_col(c: str) -> str:
        return c.replace("_", " ").title()

    fig = go.Figure(go.Heatmap(
        z=z_matrix,
        x=[_fmt_col(m) for m in metrics],
        y=df[country_col].tolist(),
        text=text_matrix,
        texttemplate="%{text}",
        textfont={"size": 10},
        colorscale="RdYlBu",
        zmid=0,
        colorbar=dict(title="Z-score", len=0.6),
        hovertemplate=(
            "<b>%{y}</b><br>"
            "%{x}: %{text}<extra></extra>"
        ),
    ))

    fig.update_layout(
        title=title,
        xaxis=dict(title="", side="bottom"),
        yaxis=dict(title="Country", autorange="reversed"),
        height=max(height, len(df) * 18 + 120),
        template=TEMPLATE,
        margin=dict(l=10, r=60, t=60, b=60),
    )
    return fig
