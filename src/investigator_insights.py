"""
investigator_insights.py — DecisionLENS investigator and site intelligence.

Provides InvestigatorAnalyzer, which reads AACT parquet flat files and exposes
four public methods for site-selection and investigator-performance analysis:

  • get_top_sites()          — rank facilities by trial volume and completion rate
  • get_country_performance() — country-level enrollment performance metrics
  • plot_site_network()       — co-participation network graph (Plotly)
  • recommend_sites()         — data-driven site-country allocation recommendation

All table loads are lazy (on first access) and cached in-memory.

Typical usage::

    from src.investigator_insights import InvestigatorAnalyzer

    ia = InvestigatorAnalyzer("data/processed")
    top = ia.get_top_sites("Breast Cancer", n=20)
    perf = ia.get_country_performance("Breast Cancer")
    fig = ia.plot_site_network("Breast Cancer")
    recs = ia.recommend_sites("Breast Cancer", target_enrollment=500, n_countries=5)
"""

from __future__ import annotations

import itertools
import logging
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import plotly.graph_objects as go

log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

#: Plotly default template (consistent with other modules)
TEMPLATE: str = "plotly_white"

#: Facility names that are generic sponsor placeholders — excluded from
#: site ranking and network analysis because they do not identify a real
#: physical investigator site.
GENERIC_SITE_NAMES: frozenset[str] = frozenset({
    "Research Site",
    "GSK Investigational Site",
    "Novartis Investigative Site",
    "Pfizer Investigational Site",
    "Novo Nordisk Investigational Site",
    "Investigational Site",
    "AstraZeneca Investigational Site",
    "Eli Lilly and Company",
    "Merck Sharp & Dohme LLC",
    "Sanofi-Aventis Investigational Site",
    "Roche Investigational Site",
    "Johnson & Johnson",
    "Boehringer Ingelheim Investigational Site",
    "Bristol-Myers Squibb",
    "Amgen Investigational Site",
    "Janssen Research & Development, LLC",
})

#: Completed / terminated statuses for computing completion rates
COMPLETED_STATUS:  str = "Completed"
TERMINATED_STATUS: str = "Terminated"

#: Minimum named-site trial count to include in network graph
MIN_SITE_TRIALS_NETWORK: int = 2

#: Colour palette for country-coded network nodes (cycles for >10 countries)
COUNTRY_COLORS: list[str] = [
    "#1a6fa8", "#e07b39", "#2a9d8f", "#e63946", "#8ecae6",
    "#f4a261", "#264653", "#a8dadc", "#6d6875", "#b5838d",
    "#457b9d", "#e9c46a", "#f3722c", "#43aa8b", "#577590",
]


# ---------------------------------------------------------------------------
# InvestigatorAnalyzer
# ---------------------------------------------------------------------------


class InvestigatorAnalyzer:
    """
    Site and investigator performance analyzer for clinical trial intelligence.

    Loads AACT parquet tables lazily on first access and caches each table
    in-memory for the lifetime of the instance.

    Site identity is derived from the ``(name, city, country)`` triple in the
    AACT facilities table.  Generic sponsor-assigned placeholder names (e.g.,
    "Research Site", "GSK Investigational Site") are excluded from ranking and
    network analysis because they do not uniquely identify a physical facility.

    Args:
        data_dir: Path to the directory containing AACT ``.parquet`` flat files.
                  Defaults to ``"data/processed"`` relative to the cwd.

    Example::

        ia   = InvestigatorAnalyzer("data/processed")
        top  = ia.get_top_sites("Non-Small Cell Lung Cancer", n=15)
        perf = ia.get_country_performance("Non-Small Cell Lung Cancer")
        fig  = ia.plot_site_network("Non-Small Cell Lung Cancer")
        recs = ia.recommend_sites("Non-Small Cell Lung Cancer", 800, 6)
    """

    def __init__(self, data_dir: str | Path = "data/processed") -> None:
        self.data_dir = Path(data_dir)

    # ------------------------------------------------------------------
    # 1. get_top_sites
    # ------------------------------------------------------------------

    def get_top_sites(
        self,
        condition: str,
        n: int = 20,
    ) -> pd.DataFrame:
        """
        Rank facilities by trial volume and completion rate for a condition.

        Generic sponsor-assigned site names are excluded.  Sites are ranked
        primarily by ``n_trials`` (descending), then ``completion_rate``
        (descending).

        Columns in the returned DataFrame:

        ================== ==================================================
        Column             Description
        ================== ==================================================
        name               Facility name
        city               City
        country            Country
        n_trials           Total number of condition-relevant trials at site
        n_completed        Trials with COMPLETED status
        n_terminated       Trials with TERMINATED status
        completion_rate    n_completed / (n_completed + n_terminated)
        avg_enrollment     Mean target enrollment across all condition trials
        latitude           Site latitude (NaN when unavailable)
        longitude          Site longitude (NaN when unavailable)
        ================== ==================================================

        Args:
            condition: Condition name (partial, case-insensitive).
            n: Number of top sites to return. Default 20.

        Returns:
            DataFrame of top sites, sorted by n_trials then completion_rate.
            Empty DataFrame when no named sites are found.
        """
        trials = self._get_condition_trials(condition)
        if trials.empty:
            log.warning("get_top_sites: no trials for condition %r.", condition)
            return pd.DataFrame()

        nct_ids = set(trials["nct_id"].tolist())
        fa = self._facilities
        fa_sub = fa[
            fa["nct_id"].isin(nct_ids)
            & ~fa["name"].isin(GENERIC_SITE_NAMES)
        ].copy()

        if fa_sub.empty:
            log.warning("get_top_sites: no named facilities found for %r.", condition)
            return pd.DataFrame()

        # Merge with study-level status and enrollment
        study_cols = ["nct_id", "overall_status", "enrollment"]
        study_sub  = trials[
            [c for c in study_cols if c in trials.columns]
        ].copy()
        fa_merged = fa_sub.merge(study_sub, on="nct_id", how="left")

        # Aggregate per site (name + city + country)
        group_cols = ["name", "city", "country"]

        def _agg(g: pd.DataFrame) -> pd.Series:
            n_t   = len(g["nct_id"].unique())
            n_c   = int((g["overall_status"] == COMPLETED_STATUS).sum())
            n_trm = int((g["overall_status"] == TERMINATED_STATUS).sum())
            denom = n_c + n_trm
            rate  = round(n_c / denom, 4) if denom > 0 else float("nan")
            avg_e = float(g["enrollment"].mean()) if "enrollment" in g else float("nan")
            lat   = g["latitude"].dropna().mean()  if "latitude"  in g else float("nan")
            lon   = g["longitude"].dropna().mean() if "longitude" in g else float("nan")
            return pd.Series({
                "n_trials":        n_t,
                "n_completed":     n_c,
                "n_terminated":    n_trm,
                "completion_rate": rate,
                "avg_enrollment":  round(avg_e, 1),
                "latitude":        round(lat, 5) if not np.isnan(lat) else float("nan"),
                "longitude":       round(lon, 5) if not np.isnan(lon) else float("nan"),
            })

        site_df = (
            fa_merged.groupby(group_cols, dropna=False)
            .apply(_agg, include_groups=False)
            .reset_index()
            .sort_values(["n_trials", "completion_rate"], ascending=[False, False])
            .head(n)
            .reset_index(drop=True)
        )

        log.info(
            "get_top_sites: condition=%r → %d sites (top %d shown)",
            condition, len(fa_merged.groupby(group_cols)), n,
        )
        return site_df

    # ------------------------------------------------------------------
    # 2. get_country_performance
    # ------------------------------------------------------------------

    def get_country_performance(
        self,
        condition: Optional[str] = None,
    ) -> pd.DataFrame:
        """
        Return country-level enrollment performance metrics.

        When ``condition`` is provided, metrics are restricted to trials
        matching that condition.  When ``None``, all interventional trials
        in the dataset are used.

        Columns in the returned DataFrame:

        =================== ================================================
        Column              Description
        =================== ================================================
        country             Country name
        n_trials            Total number of trials with a site in this country
        n_completed         Trials with COMPLETED status
        n_terminated        Trials with TERMINATED status
        completion_rate     n_completed / (n_completed + n_terminated)
        avg_enrollment      Mean target enrollment (all trials)
        avg_duration_days   Mean enrollment duration for completed trials
        trials_per_year     n_trials / years spanned by earliest–latest start
        performance_score   Composite score (completion_rate × log1p(n_completed))
        =================== ================================================

        Args:
            condition: Optional condition name filter (partial, case-insensitive).
                       Pass ``None`` to aggregate across all conditions.

        Returns:
            DataFrame sorted by ``performance_score`` descending.
        """
        if condition is not None:
            trials = self._get_condition_trials(condition)
            if trials.empty:
                log.warning("get_country_performance: no trials for %r.", condition)
                return pd.DataFrame()
        else:
            trials = self._studies

        # Join facilities (country) → studies (status, enrollment, dates)
        nct_ids = set(trials["nct_id"].tolist())
        fa_sub  = self._facilities[
            self._facilities["nct_id"].isin(nct_ids)
            & self._facilities["country"].notna()
        ][["nct_id", "country"]].drop_duplicates()

        study_cols = ["nct_id", "overall_status", "enrollment",
                      "start_date", "completion_date"]
        study_sub = trials[
            [c for c in study_cols if c in trials.columns]
        ].copy()

        merged = fa_sub.merge(study_sub, on="nct_id", how="left")

        if "start_date" in merged.columns:
            merged["start_date"] = pd.to_datetime(merged["start_date"], errors="coerce")
        if "completion_date" in merged.columns:
            merged["completion_date"] = pd.to_datetime(
                merged["completion_date"], errors="coerce"
            )

        def _country_agg(g: pd.DataFrame) -> pd.Series:
            n_t   = len(g["nct_id"].unique())
            n_c   = int((g["overall_status"] == COMPLETED_STATUS).sum())
            n_trm = int((g["overall_status"] == TERMINATED_STATUS).sum())
            denom = n_c + n_trm
            rate  = round(n_c / denom, 4) if denom > 0 else float("nan")
            avg_e = float(g["enrollment"].mean()) if "enrollment" in g.columns else float("nan")

            # Duration for completed trials only
            avg_dur = float("nan")
            if "start_date" in g.columns and "completion_date" in g.columns:
                comp_g = g[g["overall_status"] == COMPLETED_STATUS]
                if not comp_g.empty:
                    dur = (
                        comp_g["completion_date"] - comp_g["start_date"]
                    ).dt.days.dropna()
                    if not dur.empty:
                        avg_dur = round(float(dur.mean()), 1)

            # Trials per year
            tpy = float("nan")
            if "start_date" in g.columns and n_t > 0:
                dates = g["start_date"].dropna()
                if len(dates) >= 2:
                    span_years = max(
                        (dates.max() - dates.min()).days / 365.25, 1.0
                    )
                    tpy = round(n_t / span_years, 2)

            perf = round(rate * np.log1p(n_c), 4) if not np.isnan(rate) else 0.0

            return pd.Series({
                "n_trials":          n_t,
                "n_completed":       n_c,
                "n_terminated":      n_trm,
                "completion_rate":   rate,
                "avg_enrollment":    round(avg_e, 1),
                "avg_duration_days": avg_dur,
                "trials_per_year":   tpy,
                "performance_score": perf,
            })

        perf_df = (
            merged.groupby("country", dropna=True)
            .apply(_country_agg, include_groups=False)
            .reset_index()
            .sort_values("performance_score", ascending=False)
            .reset_index(drop=True)
        )

        log.info(
            "get_country_performance: condition=%r → %d countries",
            condition, len(perf_df),
        )
        return perf_df

    # ------------------------------------------------------------------
    # 3. plot_site_network
    # ------------------------------------------------------------------

    def plot_site_network(
        self,
        condition: str,
        n_sites: int = 30,
    ) -> go.Figure:
        """
        Network graph showing sites that co-participate in the same trials.

        Nodes represent the top ``n_sites`` named facilities (by trial count)
        for the condition.  An edge connects two sites that participated in at
        least one common trial; edge weight (line width) is proportional to the
        number of shared trials.

        Node size is proportional to ``n_trials``; nodes are coloured by
        country.  Positions are arranged using a circular layout grouped by
        country (same-country nodes are adjacent on the circle).

        Note: Generic sponsor placeholder names (e.g., "Research Site") are
        excluded because they do not identify a unique physical facility.

        Args:
            condition: Condition name (partial, case-insensitive).
            n_sites: Maximum number of top sites to include. Default 30.

        Returns:
            Plotly Figure (network graph using go.Scatter).
        """
        trials = self._get_condition_trials(condition)
        if trials.empty:
            return self._empty_figure(f"No trials found for: {condition!r}")

        nct_ids = set(trials["nct_id"].tolist())
        fa = self._facilities

        # Named facilities for this condition
        fa_sub = fa[
            fa["nct_id"].isin(nct_ids)
            & ~fa["name"].isin(GENERIC_SITE_NAMES)
        ][["nct_id", "name", "city", "country"]].copy()

        if fa_sub.empty:
            return self._empty_figure(f"No named sites for: {condition!r}")

        # Site key and trial counts
        fa_sub["site_key"] = (
            fa_sub["name"].fillna("") + "|"
            + fa_sub["city"].fillna("") + "|"
            + fa_sub["country"].fillna("")
        )

        site_trial_counts = (
            fa_sub.groupby("site_key")["nct_id"]
            .nunique()
            .sort_values(ascending=False)
        )
        top_sites = site_trial_counts[
            site_trial_counts >= MIN_SITE_TRIALS_NETWORK
        ].head(n_sites)

        if len(top_sites) < 2:
            return self._empty_figure(
                f"Insufficient named sites to build network for: {condition!r}"
            )

        top_site_keys = set(top_sites.index)
        fa_top = fa_sub[fa_sub["site_key"].isin(top_site_keys)].copy()

        # Site metadata (deduplicated)
        site_meta = (
            fa_top[["site_key", "name", "city", "country"]]
            .drop_duplicates("site_key")
            .set_index("site_key")
        )

        # Build edge list: sites sharing ≥1 trial
        edge_weights: dict[tuple[str, str], int] = {}
        for _, trial_group in fa_top.groupby("nct_id"):
            keys = sorted(trial_group["site_key"].unique())
            if len(keys) < 2:
                continue
            for a, b in itertools.combinations(keys, 2):
                pair = (a, b)
                edge_weights[pair] = edge_weights.get(pair, 0) + 1

        if not edge_weights:
            return self._empty_figure(
                f"No co-participating site pairs found for: {condition!r}<br>"
                f"(Try a more common condition with multi-site trials)"
            )

        # Country → colour mapping
        countries  = site_meta["country"].fillna("Unknown").unique().tolist()
        country_color = {
            c: COUNTRY_COLORS[i % len(COUNTRY_COLORS)]
            for i, c in enumerate(sorted(countries))
        }

        # Circular layout: group sites by country for visual clustering
        site_keys_ordered = (
            site_meta.sort_values("country").index.tolist()
        )
        n_nodes = len(site_keys_ordered)
        angles  = np.linspace(0, 2 * np.pi, n_nodes, endpoint=False)
        pos     = {
            k: (float(np.cos(a)), float(np.sin(a)))
            for k, a in zip(site_keys_ordered, angles)
        }

        max_weight  = max(edge_weights.values())
        max_trials  = int(top_sites.max())

        # ---- Edge traces ---------------------------------------------------
        edge_traces = []
        for (a, b), w in edge_weights.items():
            x0, y0 = pos[a]
            x1, y1 = pos[b]
            width   = 0.5 + 3.5 * (w / max_weight)
            alpha   = max(0.15, w / max_weight)
            color   = f"rgba(150,150,150,{alpha:.2f})"
            edge_traces.append(go.Scatter(
                x=[x0, x1, None],
                y=[y0, y1, None],
                mode="lines",
                line=dict(width=width, color=color),
                hoverinfo="none",
                showlegend=False,
            ))

        # ---- Node traces (one per country for legend) ----------------------
        node_traces = []
        seen_countries: set[str] = set()
        for site_key in site_keys_ordered:
            meta    = site_meta.loc[site_key]
            country = str(meta["country"]) if pd.notna(meta["country"]) else "Unknown"
            clr     = country_color[country]
            x, y    = pos[site_key]
            n_t     = int(top_sites.get(site_key, 1))
            size    = 8 + 22 * (n_t / max_trials)
            label   = f"{meta['name']}, {meta['city']}"

            node_traces.append(go.Scatter(
                x=[x], y=[y],
                mode="markers+text",
                text=[label[:30]],
                textposition="top center",
                textfont=dict(size=7),
                marker=dict(
                    size=size,
                    color=clr,
                    line=dict(width=1, color="white"),
                ),
                name=country if country not in seen_countries else "",
                showlegend=country not in seen_countries,
                hovertemplate=(
                    f"<b>{meta['name']}</b><br>"
                    f"{meta['city']}, {country}<br>"
                    f"Trials for condition: {n_t}"
                    "<extra></extra>"
                ),
                legendgroup=country,
            ))
            seen_countries.add(country)

        fig = go.Figure(data=edge_traces + node_traces)
        fig.update_layout(
            title=(
                f"Site Co-participation Network — {condition}<br>"
                f"<sup>Top {len(site_keys_ordered)} named sites | "
                f"{len(edge_weights)} co-participation links | "
                f"Node size ∝ trial count | Edge width ∝ shared trials</sup>"
            ),
            xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            legend=dict(title="Country", itemsizing="constant"),
            template=TEMPLATE,
            height=580,
            margin=dict(l=20, r=20, t=80, b=20),
        )
        return fig

    # ------------------------------------------------------------------
    # 4. recommend_sites
    # ------------------------------------------------------------------

    def recommend_sites(
        self,
        condition: str,
        target_enrollment: int,
        n_countries: int,
    ) -> pd.DataFrame:
        """
        Recommend an optimal site-country allocation for a planned trial.

        Countries are ranked by a composite performance score derived from
        historical AACT data for the given condition:

        .. code-block:: text

            score = completion_rate × log1p(n_completed) × speed_bonus

        where ``speed_bonus = 365 / max(avg_duration_days, 365)``
        (faster countries get a higher weight).

        Enrollment allocation is proportional to each country's score weight.
        Within each country, the recommended site count is estimated as
        ``ceil(allocated_enrollment / avg_enrollment_per_site)``.

        Columns in the returned DataFrame:

        ====================== ==============================================
        Column                 Description
        ====================== ==============================================
        rank                   Allocation rank (1 = highest priority)
        country                Recommended country
        performance_score      Composite score (higher = better history)
        completion_rate        Historical trial completion rate
        n_completed            Completed trial count (confidence indicator)
        avg_duration_days      Mean completion duration (days)
        allocated_enrollment   Recommended enrollment target for this country
        recommended_sites      Estimated number of sites needed
        rationale              Plain-English explanation string
        ====================== ==============================================

        Args:
            condition: Condition name (partial, case-insensitive).
            target_enrollment: Total planned enrollment across all countries.
            n_countries: Number of countries to include in the recommendation.

        Returns:
            DataFrame with one row per recommended country, sorted by rank.
            Returns an empty DataFrame with a log warning when data is
            insufficient.
        """
        perf = self.get_country_performance(condition)
        if perf.empty:
            log.warning("recommend_sites: no performance data for %r.", condition)
            return pd.DataFrame()

        # Require some track record
        perf = perf[perf["n_completed"] >= 3].copy()
        if perf.empty:
            log.warning(
                "recommend_sites: no country has ≥3 completed trials for %r.", condition
            )
            return pd.DataFrame()

        # Speed bonus: faster completion → higher score
        perf["speed_bonus"] = 365.0 / perf["avg_duration_days"].clip(lower=365.0)
        perf["speed_bonus"] = perf["speed_bonus"].fillna(1.0)
        perf["adj_score"]   = (
            perf["performance_score"] * perf["speed_bonus"]
        ).fillna(0.0)

        top = (
            perf.nlargest(n_countries, "adj_score")
            .reset_index(drop=True)
        )

        total_score = top["adj_score"].sum()
        if total_score == 0:
            top["weight"] = 1.0 / len(top)
        else:
            top["weight"] = top["adj_score"] / total_score

        top["allocated_enrollment"] = (
            (top["weight"] * target_enrollment).round().astype(int)
        )

        # Recommended site count: enrollment / avg sites-per-trial in country
        # Use avg_enrollment as a proxy for per-site capacity
        avg_e = top["avg_enrollment"].clip(lower=10.0)
        top["recommended_sites"] = (
            np.ceil(top["allocated_enrollment"] / avg_e).astype(int).clip(lower=1)
        )

        # Rationale string
        def _rationale(row: pd.Series) -> str:
            rate_pct = f"{row['completion_rate']:.0%}" if pd.notna(row["completion_rate"]) else "N/A"
            dur_yr   = f"{row['avg_duration_days'] / 365.25:.1f}" if pd.notna(row["avg_duration_days"]) else "N/A"
            return (
                f"{int(row['n_completed'])} completed trials; "
                f"completion rate {rate_pct}; "
                f"avg duration {dur_yr} yrs"
            )

        top["rationale"] = top.apply(_rationale, axis=1)
        top.index = top.index + 1
        top.index.name = "rank"

        result = top[[
            "country", "performance_score", "completion_rate",
            "n_completed", "avg_duration_days",
            "allocated_enrollment", "recommended_sites", "rationale",
        ]].reset_index()

        log.info(
            "recommend_sites: condition=%r target=%d n_countries=%d → %d rows",
            condition, target_enrollment, n_countries, len(result),
        )
        return result

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _get_condition_trials(self, condition: str) -> pd.DataFrame:
        """
        Return the studies DataFrame filtered to trials matching the condition.

        Args:
            condition: Case-insensitive substring condition name.

        Returns:
            Studies DataFrame. Empty DataFrame when no match found.
        """
        cond_df = self._conditions
        mask    = cond_df["downcase_name"].str.contains(
            condition.lower(), regex=False, na=False
        )
        nct_ids = cond_df.loc[mask, "nct_id"].unique()

        if len(nct_ids) == 0:
            return pd.DataFrame()

        result = self._studies[self._studies["nct_id"].isin(nct_ids)].copy()
        log.info("_get_condition_trials: %r → %d studies", condition, len(result))
        return result

    @staticmethod
    def _empty_figure(message: str) -> go.Figure:
        """Return a blank Plotly figure with a centred annotation."""
        fig = go.Figure()
        fig.add_annotation(
            text=message, xref="paper", yref="paper",
            x=0.5, y=0.5, showarrow=False,
            font=dict(size=13, color="grey"),
        )
        fig.update_layout(template=TEMPLATE, height=420)
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
    def _facilities(self) -> pd.DataFrame:
        """Facilities table (one row per site)."""
        if "_facilities_cache" not in self.__dict__:
            log.info("Loading facilities.parquet …")
            df = pd.read_parquet(self.data_dir / "facilities.parquet")
            self.__dict__["_facilities_cache"] = df
            log.info("Loaded facilities: %d rows", len(df))
        return self.__dict__["_facilities_cache"]

    def __repr__(self) -> str:
        return f"InvestigatorAnalyzer(data_dir={str(self.data_dir)!r})"
