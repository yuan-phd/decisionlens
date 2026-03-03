"""
competitive_intel.py — DecisionLENS competitive landscape intelligence.

Provides CompetitiveAnalyzer, a self-contained module that reads AACT parquet
flat files and exposes four public methods for clinical-trial competitive analysis:

  • get_landscape()                  — summary metrics for a condition
  • plot_competition_map()           — choropleth world map of competing sites
  • plot_competition_timeline()      — Gantt chart of competing trials over time
  • calculate_recruitment_saturation() — patient-recruitment saturation score

All table loads are lazy (on first access) and cached in-memory, so repeated
calls within the same session avoid redundant disk I/O.

Typical usage::

    from src.competitive_intel import CompetitiveAnalyzer

    analyzer  = CompetitiveAnalyzer("data/processed")
    landscape = analyzer.get_landscape("Breast Cancer")
    fig_map   = analyzer.plot_competition_map("Breast Cancer")
    fig_time  = analyzer.plot_competition_timeline("Breast Cancer")
    saturation = analyzer.calculate_recruitment_saturation("Breast Cancer", "United States")
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

#: Statuses treated as "currently active / competing for patients"
ACTIVE_STATUSES: frozenset[str] = frozenset({
    "RECRUITING",
    "ACTIVE_NOT_RECRUITING",
    "NOT_YET_RECRUITING",
    "ENROLLING_BY_INVITATION",
})

#: Subset of ACTIVE_STATUSES that are actively recruiting right now
RECRUITING_STATUSES: frozenset[str] = frozenset({
    "RECRUITING",
    "ENROLLING_BY_INVITATION",
})

#: Plotly default template
TEMPLATE: str = "plotly_white"

#: Clinical-phase colour palette (consistent with notebook 03)
PHASE_COLORS: dict[str, str] = {
    "Phase 1":          "#caf0f8",
    "Phase 1/Phase 2":  "#90e0ef",
    "Phase 2":          "#00b4d8",
    "Phase 2/Phase 3":  "#0096c7",
    "Phase 3":          "#0077b6",
    "Phase 4":          "#023e8a",
    "N/A":              "#adb5bd",
}

#: Plotly country-name patches for ``locationmode='country names'``
COUNTRY_NAME_MAP: dict[str, str] = {
    "Turkey (Türkiye)":                          "Turkey",
    "Korea, Republic of":                        "South Korea",
    "Iran, Islamic Republic of":                 "Iran",
    "Russian Federation":                        "Russia",
    "Czech Republic":                            "Czechia",
    "Viet Nam":                                  "Vietnam",
    "Syrian Arab Republic":                      "Syria",
    "Congo, The Democratic Republic of the":     "Democratic Republic of the Congo",
    "Tanzania, United Republic of":              "Tanzania",
    "Bolivia, Plurinational State of":           "Bolivia",
    "Venezuela, Bolivarian Republic of":         "Venezuela",
    "Moldova, Republic of":                      "Moldova",
    "Lao People's Democratic Republic":          "Laos",
    "Palestine, State of":                       "Palestine",
}


# ---------------------------------------------------------------------------
# CompetitiveAnalyzer
# ---------------------------------------------------------------------------


class CompetitiveAnalyzer:
    """
    Competitive landscape analyzer for clinical trial enrollment intelligence.

    Loads AACT parquet tables lazily on first access and caches each table
    in-memory for the lifetime of the instance.

    Condition matching uses case-insensitive substring search against the
    AACT ``conditions.downcase_name`` column, so both "breast cancer" and
    "Breast Cancer" produce the same results.

    Args:
        data_dir: Path to the directory containing AACT ``.parquet`` flat files.
                  Defaults to ``"data/processed"`` relative to the current
                  working directory.

    Example::

        analyzer  = CompetitiveAnalyzer("data/processed")
        landscape = analyzer.get_landscape("Breast Cancer")
        fig       = analyzer.plot_competition_map("Breast Cancer")
    """

    def __init__(self, data_dir: str | Path = "data/processed") -> None:
        self.data_dir = Path(data_dir)

    # ------------------------------------------------------------------
    # 1. get_landscape
    # ------------------------------------------------------------------

    def get_landscape(
        self,
        condition: str,
        therapeutic_area: Optional[str] = None,
    ) -> dict:
        """
        Return a comprehensive competitive landscape summary for a condition.

        Keys in the returned dict:

        ======================= ============================================
        Key                     Description
        ======================= ============================================
        condition               Query condition string
        total_trials            All historical trial count
        active_trials           Currently active / recruiting trials
        recruiting_trials       Actively recruiting now
        completed_trials        Trials with COMPLETED status
        terminated_trials       Trials with TERMINATED status
        total_competing_enrollment  Sum of enrollment targets (active trials)
        temporal_density        Dict: last_6m / last_12m / last_24m counts
        phase_breakdown         Dict of phase → count (active trials)
        top_countries           List of (country, site_count) tuples
        top_sponsors            List of (sponsor_name, trial_count) tuples
        competition_intensity   Float 0–1 (1 = highly saturated market)
        ======================= ============================================

        Args:
            condition: Condition name (partial, case-insensitive).
            therapeutic_area: Reserved for future therapeutic-area taxonomy
                              mapping; currently unused.

        Returns:
            Dictionary of landscape metrics.  Returns an empty-metrics dict
            (all zeros) when no trials are found for the condition.
        """
        trials = self._get_condition_trials(condition)
        if trials.empty:
            log.warning("get_landscape: no trials found for condition %r.", condition)
            return self._empty_landscape(condition)

        log.info("get_landscape: condition=%r  total=%d trials", condition, len(trials))

        status_norm = trials["overall_status"].str.upper().str.replace(" ", "_", regex=False)
        active_mask     = status_norm.isin(ACTIVE_STATUSES)
        recruiting_mask = status_norm.isin(RECRUITING_STATUSES)
        active_trials     = trials[active_mask]
        recruiting_trials = trials[recruiting_mask]

        # Temporal density
        today = pd.Timestamp.now()
        temporal: dict[str, int] = {}
        if "start_date" in trials.columns:
            for months, label in [(6, "last_6m"), (12, "last_12m"), (24, "last_24m")]:
                cutoff = today - pd.DateOffset(months=months)
                temporal[label] = int(
                    (trials["start_date"].notna() & (trials["start_date"] >= cutoff)).sum()
                )

        # Phase breakdown for active trials
        phase_breakdown: dict[str, int] = {}
        if "phase" in active_trials.columns and not active_trials.empty:
            phase_breakdown = (
                active_trials["phase"].fillna("N/A").value_counts().to_dict()
            )

        # Geographic and sponsor breakdowns
        nct_ids_all    = trials["nct_id"].tolist()
        nct_ids_active = active_trials["nct_id"].tolist()
        top_countries  = self._top_countries(nct_ids_all, n=10)
        top_sponsors   = self._top_sponsors(nct_ids_active, n=10)

        # Enrollment pressure
        competing_enrollment = 0
        if "enrollment" in active_trials.columns and not active_trials.empty:
            competing_enrollment = int(
                active_trials["enrollment"].sum(min_count=1) or 0
            )

        # Competition intensity: normalised by a 200-trial ceiling
        intensity = round(min(len(active_trials) / 200.0, 1.0), 3)

        return {
            "condition":                  condition,
            "total_trials":               len(trials),
            "active_trials":              len(active_trials),
            "recruiting_trials":          len(recruiting_trials),
            "completed_trials":           int(status_norm.eq("COMPLETED").sum()),
            "terminated_trials":          int(status_norm.eq("TERMINATED").sum()),
            "total_competing_enrollment": competing_enrollment,
            "temporal_density":           temporal,
            "phase_breakdown":            phase_breakdown,
            "top_countries":              top_countries,
            "top_sponsors":               top_sponsors,
            "competition_intensity":      intensity,
        }

    # ------------------------------------------------------------------
    # 2. plot_competition_map
    # ------------------------------------------------------------------

    def plot_competition_map(self, condition: str) -> go.Figure:
        """
        Choropleth world map showing number of competing trial sites per country.

        Site counts are aggregated from the AACT ``facilities`` table when
        available, falling back to the ``countries`` table.  Values are
        log-scaled on the colour axis for visual clarity.

        Args:
            condition: Condition name (partial, case-insensitive).

        Returns:
            Plotly Figure (``px.choropleth``).
        """
        trials = self._get_condition_trials(condition)
        if trials.empty:
            return self._empty_figure(f"No trials found for: {condition!r}")

        nct_ids = trials["nct_id"].tolist()

        # Prefer facility-level counts; fall back to countries table
        fac_sub = self._facilities[self._facilities["nct_id"].isin(nct_ids)]
        if not fac_sub.empty and "country" in fac_sub.columns:
            country_counts = (
                fac_sub.groupby("country")
                .size()
                .reset_index(name="site_count")
                .rename(columns={"country": "country_name"})
            )
        else:
            cntr_sub = self._countries[self._countries["nct_id"].isin(nct_ids)]
            country_counts = (
                cntr_sub.groupby("name")
                .size()
                .reset_index(name="site_count")
                .rename(columns={"name": "country_name"})
            )

        # Normalise names for Plotly
        country_counts["country_name"] = (
            country_counts["country_name"].replace(COUNTRY_NAME_MAP)
        )
        country_counts = country_counts[
            country_counts["country_name"].notna()
            & (country_counts["country_name"].str.strip() != "")
        ].copy()

        country_counts["log_sites"] = np.log1p(country_counts["site_count"])
        total_sites = int(country_counts["site_count"].sum())
        n_countries = len(country_counts)

        fig = px.choropleth(
            country_counts,
            locations="country_name",
            locationmode="country names",
            color="log_sites",
            hover_name="country_name",
            hover_data={"site_count": True, "log_sites": False},
            color_continuous_scale="Blues",
            labels={"log_sites": "log(sites+1)", "site_count": "Trial sites"},
            title=(
                f"Competitive Site Distribution — {condition}<br>"
                f"<sup>{total_sites:,} sites across {n_countries} countries</sup>"
            ),
        )
        fig.update_layout(
            geo=dict(
                showframe=False,
                showcoastlines=True,
                projection_type="natural earth",
            ),
            coloraxis_colorbar=dict(title="log(sites+1)", thickness=14),
            template=TEMPLATE,
            margin=dict(l=0, r=0, t=60, b=0),
            height=480,
        )
        return fig

    # ------------------------------------------------------------------
    # 3. plot_competition_timeline
    # ------------------------------------------------------------------

    def plot_competition_timeline(
        self,
        condition: str,
        max_trials: int = 40,
    ) -> go.Figure:
        """
        Gantt-style timeline of competing trials for a condition.

        Each horizontal bar spans from ``start_date`` to ``completion_date``
        (or today for ongoing trials).  Bars are coloured by clinical phase.
        Trials are sorted by enrollment size descending; only the top
        ``max_trials`` are shown.

        Args:
            condition: Condition name (partial, case-insensitive).
            max_trials: Maximum number of trial bars to display. Default 40.

        Returns:
            Plotly Figure (Gantt / ``px.timeline``).
        """
        trials = self._get_condition_trials(condition)
        if trials.empty:
            return self._empty_figure(f"No trials found for: {condition!r}")

        today = pd.Timestamp.now()

        plot_df = trials[trials["start_date"].notna()].copy()
        plot_df["end_date"] = plot_df["completion_date"].fillna(today)
        plot_df = plot_df[plot_df["end_date"] > plot_df["start_date"]].copy()

        # Top N by enrollment
        if "enrollment" in plot_df.columns:
            plot_df = plot_df.sort_values("enrollment", ascending=False)
        plot_df = plot_df.head(max_trials).sort_values("start_date").copy()

        if plot_df.empty:
            return self._empty_figure(
                f"No trials with valid dates for: {condition!r}"
            )

        # Y-axis label
        if "brief_title" in plot_df.columns:
            plot_df["label"] = (
                plot_df["brief_title"].fillna(plot_df["nct_id"]).str[:60]
            )
        else:
            plot_df["label"] = plot_df["nct_id"]

        plot_df["phase_clean"] = (
            plot_df["phase"].fillna("N/A")
            if "phase" in plot_df.columns
            else "N/A"
        )

        # Keep only trials with a defined phase; exclude N/A and null values
        _DEFINED_PHASES = {
            "Phase 1", "Phase 2", "Phase 3", "Phase 4",
            "Phase 1/Phase 2", "Phase 2/Phase 3", "Early Phase 1",
        }
        plot_df = plot_df[plot_df["phase_clean"].isin(_DEFINED_PHASES)].copy()
        if plot_df.empty:
            return self._empty_figure(
                f"No trials with defined phase and valid dates for: {condition!r}"
            )

        plot_df["status_clean"] = plot_df["overall_status"].fillna("Unknown")
        plot_df["enrollment_str"] = (
            plot_df["enrollment"].apply(
                lambda v: f"{int(v):,}" if pd.notna(v) else "N/A"
            )
            if "enrollment" in plot_df.columns
            else "N/A"
        )

        # Ensure datetime columns are proper Timestamps
        plot_df["start_date"] = pd.to_datetime(plot_df["start_date"])
        plot_df["end_date"]   = pd.to_datetime(plot_df["end_date"])

        fig = px.timeline(
            plot_df,
            x_start="start_date",
            x_end="end_date",
            y="label",
            color="phase_clean",
            color_discrete_map=PHASE_COLORS,
            hover_data={
                "start_date":     "|%Y-%m-%d",
                "end_date":       "|%Y-%m-%d",
                "status_clean":   True,
                "enrollment_str": True,
                "phase_clean":    False,
            },
            labels={
                "label":          "Trial",
                "phase_clean":    "Phase",
                "status_clean":   "Status",
                "enrollment_str": "Enrollment",
            },
            title=(
                f"Competing Trials Timeline — {condition}<br>"
                f"<sup>Top {len(plot_df)} trials by enrollment | Colored by phase</sup>"
            ),
        )
        fig.update_yaxes(autorange="reversed")
        fig.update_layout(
            xaxis_title="Date",
            yaxis_title="",
            legend_title_text="Phase",
            template=TEMPLATE,
            height=max(380, len(plot_df) * 22 + 120),
            margin=dict(l=10, r=10, t=70, b=40),
        )
        return fig

    # ------------------------------------------------------------------
    # 4. calculate_recruitment_saturation
    # ------------------------------------------------------------------

    def calculate_recruitment_saturation(
        self,
        condition: str,
        country: str,
    ) -> float:
        """
        Estimate patient-recruitment saturation for a condition–country pair.

        Saturation is the ratio of current active-trial enrollment demand to
        a historical-capacity baseline derived from completed trials:

        .. code-block:: text

            saturation = active_enrollment_targets /
                         (mean_completed_enrollment × completed_count / years_of_data)

        A value of ``1.0`` means the country is absorbing the same patient
        volume that completed trials historically required per year.
        Values above ``1.0`` indicate above-historical-peak pressure.

        Falls back to a simpler count-based ratio when fewer than 3 completed
        trials are available for the condition-country pair.

        Args:
            condition: Condition name (partial, case-insensitive).
            country: Country name (partial, case-insensitive).

        Returns:
            Saturation score (float, 0.0–∞ range, clipped display at 1.0).
            Returns ``0.0`` when no trials are found.
        """
        trials = self._get_condition_trials(condition)
        if trials.empty:
            log.warning("No trials for %r — saturation=0.0", condition)
            return 0.0

        # Filter to this country via the countries table
        cntr = self._countries
        country_mask = cntr["name"].str.contains(country, case=False, na=False)
        country_ncts = set(cntr.loc[country_mask, "nct_id"].tolist())
        country_trials = trials[trials["nct_id"].isin(country_ncts)].copy()

        if country_trials.empty:
            log.info("No trials in %r for condition %r → saturation=0.0", country, condition)
            return 0.0

        status_norm   = (
            country_trials["overall_status"]
            .str.upper()
            .str.replace(" ", "_", regex=False)
        )
        active_trials    = country_trials[status_norm.isin(ACTIVE_STATUSES)]
        completed_trials = country_trials[status_norm.eq("COMPLETED")]

        active_enrollment = float(
            active_trials["enrollment"].sum(min_count=1) or 0.0
            if "enrollment" in active_trials.columns
            else 0.0
        )

        if len(completed_trials) < 3 or active_enrollment == 0.0:
            # Insufficient history — count-based proxy
            saturation = min(
                len(active_trials) / max(len(completed_trials), 1),
                1.0,
            )
        else:
            historical_mean = float(completed_trials["enrollment"].mean())
            earliest = completed_trials["start_date"].dropna().min()
            years = max(
                (pd.Timestamp.now() - earliest).days / 365.25,
                1.0,
            )
            annual_capacity = historical_mean * len(completed_trials) / years
            saturation = active_enrollment / max(annual_capacity, 1.0)

        saturation = round(float(saturation), 4)
        log.info(
            "Saturation: condition=%r country=%r → %.4f "
            "(active=%d n=%d, completed=%d)",
            condition, country, saturation,
            len(active_trials), int(active_enrollment), len(completed_trials),
        )
        return saturation

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _get_condition_trials(self, condition: str) -> pd.DataFrame:
        """
        Return the studies DataFrame filtered to trials matching the condition.

        Matching is a case-insensitive substring search on
        ``conditions.downcase_name``, joining back to the studies table.

        Args:
            condition: Plain-text condition string.

        Returns:
            Studies DataFrame with only rows matching the condition.
            Empty DataFrame when no match is found.
        """
        cond_df  = self._conditions
        mask     = cond_df["downcase_name"].str.contains(
            condition.lower(), regex=False, na=False
        )
        nct_ids  = cond_df.loc[mask, "nct_id"].unique()

        if len(nct_ids) == 0:
            return pd.DataFrame()

        result = self._studies[self._studies["nct_id"].isin(nct_ids)].copy()
        log.info("_get_condition_trials: %r → %d studies", condition, len(result))
        return result

    def _top_countries(
        self, nct_ids: list[str], n: int = 10
    ) -> list[tuple[str, int]]:
        """
        Return the top-N countries by site count for a list of nct_ids.

        Uses the facilities table (most granular) when available.
        """
        fac_sub = self._facilities[self._facilities["nct_id"].isin(nct_ids)]
        if not fac_sub.empty and "country" in fac_sub.columns:
            counts = fac_sub["country"].value_counts().head(n)
        else:
            cntr_sub = self._countries[self._countries["nct_id"].isin(nct_ids)]
            counts = cntr_sub["name"].value_counts().head(n)
        return list(zip(counts.index.tolist(), counts.values.tolist()))

    def _top_sponsors(
        self, nct_ids: list[str], n: int = 10
    ) -> list[tuple[str, int]]:
        """
        Return the top-N lead sponsor names by trial count for a list of nct_ids.
        """
        sp_sub = self._sponsors[
            self._sponsors["nct_id"].isin(nct_ids)
            & (self._sponsors["lead_or_collaborator"] == "lead")
        ]
        counts = sp_sub["name"].value_counts().head(n)
        return list(zip(counts.index.tolist(), counts.values.tolist()))

    @staticmethod
    def _empty_landscape(condition: str) -> dict:
        """Return a zero-value landscape dict when no trials are found."""
        return {
            "condition":                  condition,
            "total_trials":               0,
            "active_trials":              0,
            "recruiting_trials":          0,
            "completed_trials":           0,
            "terminated_trials":          0,
            "total_competing_enrollment": 0,
            "temporal_density":           {"last_6m": 0, "last_12m": 0, "last_24m": 0},
            "phase_breakdown":            {},
            "top_countries":              [],
            "top_sponsors":               [],
            "competition_intensity":      0.0,
        }

    @staticmethod
    def _empty_figure(message: str) -> go.Figure:
        """Return a blank Plotly figure with a centred annotation message."""
        fig = go.Figure()
        fig.add_annotation(
            text=message,
            xref="paper", yref="paper",
            x=0.5, y=0.5,
            showarrow=False,
            font=dict(size=14, color="grey"),
        )
        fig.update_layout(template=TEMPLATE, height=400)
        return fig

    # ------------------------------------------------------------------
    # Lazy-loaded table properties (cached in instance __dict__)
    # ------------------------------------------------------------------

    @property
    def _studies(self) -> pd.DataFrame:
        """Interventional studies with parsed dates and numeric enrollment."""
        if "_studies_cache" not in self.__dict__:
            log.info("Loading studies.parquet …")
            df = pd.read_parquet(self.data_dir / "studies.parquet")
            if "study_type" in df.columns:
                df = df[
                    df["study_type"].str.strip().str.upper() == "INTERVENTIONAL"
                ].copy()
            for col in ("start_date", "completion_date"):
                if col in df.columns:
                    df[col] = pd.to_datetime(df[col], errors="coerce")
            if "enrollment" in df.columns:
                df["enrollment"] = pd.to_numeric(df["enrollment"], errors="coerce")
            self.__dict__["_studies_cache"] = df
            log.info("Loaded studies: %d interventional rows", len(df))
        return self.__dict__["_studies_cache"]

    @property
    def _conditions(self) -> pd.DataFrame:
        """Conditions table with guaranteed downcase_name column."""
        if "_conditions_cache" not in self.__dict__:
            log.info("Loading conditions.parquet …")
            df = pd.read_parquet(self.data_dir / "conditions.parquet")
            if "downcase_name" not in df.columns:
                df["downcase_name"] = df["name"].str.lower()
            self.__dict__["_conditions_cache"] = df
            log.info("Loaded conditions: %d rows", len(df))
        return self.__dict__["_conditions_cache"]

    @property
    def _countries(self) -> pd.DataFrame:
        """Countries table (one row per trial-country pair)."""
        if "_countries_cache" not in self.__dict__:
            log.info("Loading countries.parquet …")
            df = pd.read_parquet(self.data_dir / "countries.parquet")
            self.__dict__["_countries_cache"] = df
            log.info("Loaded countries: %d rows", len(df))
        return self.__dict__["_countries_cache"]

    @property
    def _facilities(self) -> pd.DataFrame:
        """Facilities table (one row per site)."""
        if "_facilities_cache" not in self.__dict__:
            log.info("Loading facilities.parquet …")
            df = pd.read_parquet(self.data_dir / "facilities.parquet")
            self.__dict__["_facilities_cache"] = df
            log.info("Loaded facilities: %d rows", len(df))
        return self.__dict__["_facilities_cache"]

    @property
    def _sponsors(self) -> pd.DataFrame:
        """Sponsors table (lead + collaborators)."""
        if "_sponsors_cache" not in self.__dict__:
            log.info("Loading sponsors.parquet …")
            df = pd.read_parquet(self.data_dir / "sponsors.parquet")
            self.__dict__["_sponsors_cache"] = df
            log.info("Loaded sponsors: %d rows", len(df))
        return self.__dict__["_sponsors_cache"]

    def __repr__(self) -> str:
        return f"CompetitiveAnalyzer(data_dir={str(self.data_dir)!r})"
