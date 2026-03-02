"""
data_pipeline.py — DecisionLENS data loading, cleaning, and feature engineering.

Works from parquet flat-file snapshots produced by setup_data.py.
Replicates the logic in sql/enrollment_extract.sql as a pandas pipeline,
producing a single modeling DataFrame for use by src/models.py and the
Streamlit dashboard.

Typical usage:
    from src.data_pipeline import TrialDataPipeline

    pipeline = TrialDataPipeline()
    tables   = pipeline.load_raw_data("data/processed")
    df       = pipeline.engineer_features(tables)
    pipeline.save_processed_data("data/processed")
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

TABLES = [
    "studies",
    "calculated_values",
    "eligibilities",
    "designs",
    "facilities",
    "countries",
    "sponsors",
    "conditions",
    "interventions",
    "outcome_counts",
]

EXCLUDED_STATUSES = {"WITHDRAWN", "UNKNOWN STATUS"}

PHASE_ORDINAL: dict[str, float] = {
    "Phase 1":          1.0,
    "Phase 1/Phase 2":  1.5,
    "Phase 2":          2.0,
    "Phase 2/Phase 3":  2.5,
    "Phase 3":          3.0,
    "Phase 4":          4.0,
}

SPONSOR_TYPE_MAP: dict[str, str] = {
    "Industry":  "industry",
    "NIH":       "government",
    "U.S. Fed":  "government",
}

# Minimum start year for the modeling dataset
MIN_START_YEAR = 2008

# Minimum target enrollment to include (avoids pilot/feasibility noise)
MIN_ENROLLMENT = 10

# Statuses that receive a binary training label (all others → NaN).
# COMPLETED → 1 (successfully enrolled); TERMINATED → 0 (failed).
LABELED_STATUSES: frozenset[str] = frozenset({"COMPLETED", "TERMINATED"})

# Keywords in why_stopped that indicate an enrollment-related failure.
# Used only to set the informational terminated_for_enrollment flag —
# the binary label is 0 for ALL terminated trials regardless of reason.
RECRUITMENT_FAIL_KEYWORDS: tuple[str, ...] = (
    "enrollment",
    "accrual",
    "recruit",
    "enrol",
    "participant",
    "patient",
    "subject",
)


# ---------------------------------------------------------------------------
# TrialDataPipeline
# ---------------------------------------------------------------------------


class TrialDataPipeline:
    """
    Pipeline for loading, cleaning, and feature-engineering AACT clinical
    trial data from parquet flat files.

    The pipeline is stateful: after calling engineer_features(), the
    processed DataFrame is stored in self.modeling_df and can be
    persisted with save_processed_data().

    Args:
        min_start_year: Earliest study start year to include in the dataset.
        min_enrollment: Minimum target enrollment threshold.
    """

    def __init__(
        self,
        min_start_year: int = MIN_START_YEAR,
        min_enrollment: int = MIN_ENROLLMENT,
    ) -> None:
        self.min_start_year = min_start_year
        self.min_enrollment = min_enrollment
        self.modeling_df: Optional[pd.DataFrame] = None
        self._tables: dict[str, pd.DataFrame] = {}

    # ------------------------------------------------------------------
    # 1. Load
    # ------------------------------------------------------------------

    def load_raw_data(self, data_dir: str | Path) -> dict[str, pd.DataFrame]:
        """
        Load AACT parquet tables from a directory.

        Each parquet file corresponds to one AACT table (e.g. studies.parquet).
        Missing files are skipped with a warning rather than raising an error,
        so a partial dataset (e.g. synthetic subset) still loads cleanly.

        Args:
            data_dir: Path to directory containing .parquet files.

        Returns:
            Dict mapping table name to DataFrame.
        """
        data_dir = Path(data_dir)
        tables: dict[str, pd.DataFrame] = {}

        for table in TABLES:
            path = data_dir / f"{table}.parquet"
            if not path.exists():
                log.warning("Parquet file not found, skipping: %s", path)
                continue
            try:
                df = pd.read_parquet(path)
                tables[table] = df
                log.info("Loaded %-20s  %6d rows", table, len(df))
            except Exception as exc:
                log.error("Failed to load %s: %s", table, exc)

        self._tables = tables
        return tables

    # ------------------------------------------------------------------
    # 2. Clean studies
    # ------------------------------------------------------------------

    def clean_studies(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Clean the raw studies DataFrame.

        Steps:
          - Filter to interventional studies only.
          - Remove withdrawn / unknown-status trials.
          - Parse start_date and completion_date to datetime.
          - Apply year and minimum enrollment filters.
          - Drop rows with null target enrollment.
          - Encode phase as an ordinal numeric value.

        Args:
            df: Raw studies DataFrame (from load_raw_data).

        Returns:
            Cleaned studies DataFrame.
        """
        original_n = len(df)
        log.info("Cleaning studies table (%d rows) …", original_n)

        # Interventional only (case-insensitive — AACT uses mixed casing)
        if "study_type" in df.columns:
            df = df[df["study_type"].str.strip().str.upper() == "INTERVENTIONAL"].copy()

        # Drop withdrawn / unknown (normalise to uppercase before comparing)
        if "overall_status" in df.columns:
            df = df[~df["overall_status"].str.strip().str.upper().isin(EXCLUDED_STATUSES)].copy()

        # Parse dates
        for col in ("start_date", "completion_date", "primary_completion_date"):
            if col in df.columns:
                df[col] = pd.to_datetime(df[col], errors="coerce")

        # Year filter
        if "start_date" in df.columns:
            df = df[df["start_date"].dt.year >= self.min_start_year].copy()

        # Enrollment filters
        if "enrollment" in df.columns:
            df = df[df["enrollment"].notna()].copy()
            df = df[df["enrollment"] >= self.min_enrollment].copy()
            df["enrollment"] = pd.to_numeric(df["enrollment"], errors="coerce")

        # Phase: uppercase-normalise, repair all AACT variants, then ordinal-encode
        if "phase" in df.columns:
            # Step 1 — uppercase so every variant has a known form
            df["phase"] = df["phase"].astype(str).str.upper().str.strip()

            # Step 2 — map every observed AACT raw string to canonical "Phase X"
            _phase_repair: dict[str, str] = {
                # No-space variants
                "PHASE1":          "Phase 1",
                "EARLY_PHASE1":    "Phase 1",
                "EARLY PHASE 1":   "Phase 1",
                "PHASE 1":         "Phase 1",
                # Phase 2
                "PHASE2":          "Phase 2",
                "PHASE 2":         "Phase 2",
                # Phase 3
                "PHASE3":          "Phase 3",
                "PHASE 3":         "Phase 3",
                # Phase 4
                "PHASE4":          "Phase 4",
                "PHASE 4":         "Phase 4",
                # Combined
                "PHASE1/PHASE2":     "Phase 1/Phase 2",
                "PHASE 1/PHASE 2":   "Phase 1/Phase 2",
                "PHASE 1/2":         "Phase 1/Phase 2",
                "PHASE2/PHASE3":     "Phase 2/Phase 3",
                "PHASE 2/PHASE 3":   "Phase 2/Phase 3",
                "PHASE 2/3":         "Phase 2/Phase 3",
                # Null-like strings
                "NAN":  "N/A",
                "NONE": "N/A",
                "N/A":  "N/A",
            }
            df["phase"] = df["phase"].replace(_phase_repair)

            # Step 3 — any value not in the canonical set becomes N/A
            _valid_phases: set[str] = set(PHASE_ORDINAL.keys())
            df.loc[~df["phase"].isin(_valid_phases), "phase"] = "N/A"

            # Step 4 — ordinal numeric encoding for model features
            df["phase_numeric"] = df["phase"].map(PHASE_ORDINAL).fillna(0.0)

        # Boolean conversions (AACT stores as 't'/'f' strings)
        for bool_col in ("has_dmc", "is_fda_regulated_drug", "is_fda_regulated_device"):
            if bool_col in df.columns:
                df[bool_col] = df[bool_col].map({"t": True, "f": False, True: True, False: False})

        log.info(
            "Studies after cleaning: %d (removed %d rows)",
            len(df), original_n - len(df),
        )
        return df.reset_index(drop=True)

    # ------------------------------------------------------------------
    # 3. Feature engineering
    # ------------------------------------------------------------------

    def engineer_features(self, tables: dict[str, pd.DataFrame]) -> pd.DataFrame:
        """
        Join AACT tables and engineer the full modeling feature set.

        This mirrors the logic in sql/enrollment_extract.sql but operates
        on the parquet flat files via pandas.

        Engineered features:
          - n_eligibility_criteria  : eligibility text line count (complexity proxy)
          - n_facilities            : unique site count
          - n_countries             : unique country count
          - is_multicountry         : bool flag
          - sponsor_type            : industry / government / academic
          - phase_numeric           : ordinal phase encoding
          - condition_prevalence_proxy : historical trial frequency for condition
          - sponsor_historical_performance : median enrollment ratio for sponsor
          - geographic_concentration : HHI index of country site distribution
          - enrollment_ratio        : actual / target (capped, regression target)
          - enrollment_duration_days: days from start to completion (survival target)
          - enrollment_met_target   : bool, ratio >= 0.90 (classification target)
          - competing_trials_count  : other trials for same condition overlapping in time

        Args:
            tables: Dict of DataFrames from load_raw_data.

        Returns:
            Modeling DataFrame with one row per study.
        """
        if "studies" not in tables:
            raise ValueError("'studies' table is required but was not loaded.")

        df = self.clean_studies(tables["studies"].copy())

        # Uppercase for consistent filtering
        df["study_type"]    = df["study_type"].astype(str).str.upper().str.strip()
        df["overall_status"] = df["overall_status"].astype(str).str.upper().str.strip()

        df = df[~df["overall_status"].isin(EXCLUDED_STATUSES)]
        df = df[df["study_type"] == "INTERVENTIONAL"]

        # ---- Status distribution diagnostics --------------------------------
        status_counts = df["overall_status"].value_counts()
        log.info(
            "Status distribution after filtering (%d total rows):\n%s",
            len(df), status_counts.to_string(),
        )

        # ---- enrollment feature + enrollment_type_is_actual -----------------
        # enrollment = registered target size (used as a FEATURE, not a target).
        # enrollment_type_is_actual = 1 when the enrollment figure was confirmed
        # at trial close ('Actual'); 0 when it is the prospective estimate.
        if "enrollment_type" in df.columns:
            df["enrollment_type_is_actual"] = (
                df["enrollment_type"].astype(str).str.upper().str.strip() == "ACTUAL"
            ).astype(int)
        else:
            df["enrollment_type_is_actual"] = 0

        # overall_status: convert to readable Title Case for downstream use.
        # Replace underscores first so "NOT_YET_RECRUITING" → "Not Yet Recruiting".
        df["overall_status"] = (
            df["overall_status"]
            .str.replace("_", " ", regex=False)
            .str.title()
        )

        # ---- Calculated values (actual duration, n_facilities) -----------
        if "calculated_values" in tables:
            cv = tables["calculated_values"][
                ["nct_id", "actual_duration", "number_of_facilities"]
            ].copy()
            cv.columns = ["nct_id", "cv_duration_days", "n_facilities_cv"]
            df = df.merge(cv, on="nct_id", how="left")

        # ---- Country counts + HHI ----------------------------------------
        if "countries" in tables:
            ctry = tables["countries"].copy()
            ctry = ctry[ctry.get("removed", pd.Series("f", index=ctry.index)) != "t"]
            country_counts = (
                ctry.groupby("nct_id")["name"]
                .nunique()
                .reset_index(name="n_countries")
            )
            df = df.merge(country_counts, on="nct_id", how="left")
            df["n_countries"] = df["n_countries"].fillna(1).astype(int)
            df["is_multicountry"] = df["n_countries"] > 1
        else:
            df["n_countries"] = 1
            df["is_multicountry"] = False

        # ---- Facility counts + HHI index ---------------------------------
        if "facilities" in tables:
            fac = tables["facilities"].copy()
            if "status" in fac.columns:
                fac = fac[fac["status"] != "Withdrawn"]

            # Total site count per study
            n_fac = (
                fac.groupby("nct_id")
                .size()
                .reset_index(name="n_facilities")
            )
            df = df.merge(n_fac, on="nct_id", how="left")

            # HHI: Herfindahl-Hirschman Index of country distribution
            if "country" in fac.columns:
                country_shares = (
                    fac.groupby(["nct_id", "country"])
                    .size()
                    .reset_index(name="site_count")
                )
                country_shares["share"] = country_shares.groupby("nct_id")[
                    "site_count"
                ].transform(lambda x: x / x.sum())
                hhi = (
                    country_shares.groupby("nct_id")["share"]
                    .apply(lambda s: (s**2).sum())
                    .reset_index(name="geographic_concentration")
                )
                df = df.merge(hhi, on="nct_id", how="left")
            else:
                df["geographic_concentration"] = 1.0
        else:
            df["n_facilities"] = df.get("n_facilities_cv", np.nan)
            df["geographic_concentration"] = 1.0

        # Resolve n_facilities (prefer facilities table, fall back to cv)
        if "n_facilities_cv" in df.columns:
            df["n_facilities"] = df["n_facilities"].fillna(df["n_facilities_cv"])
            df.drop(columns=["n_facilities_cv"], inplace=True)
        df["n_facilities"] = df["n_facilities"].fillna(1).astype(int)

        # ---- Primary condition + prevalence proxy ------------------------
        if "conditions" in tables:
            cond = tables["conditions"].copy()
            # One condition per study (first listed)
            primary_cond = (
                cond.groupby("nct_id")["name"]
                .first()
                .reset_index()
                .rename(columns={"name": "condition"})
            )
            # Prevalence proxy: how many trials share this condition
            cond_freq = (
                cond["name"]
                .str.lower()
                .value_counts()
                .reset_index()
            )
            cond_freq.columns = ["condition_lower", "condition_prevalence_proxy"]
            primary_cond["condition_lower"] = primary_cond["condition"].str.lower()
            primary_cond = primary_cond.merge(cond_freq, on="condition_lower", how="left")
            df = df.merge(
                primary_cond[["nct_id", "condition", "condition_prevalence_proxy"]],
                on="nct_id", how="left",
            )
        else:
            df["condition"] = "Unknown"
            df["condition_prevalence_proxy"] = np.nan

        # ---- Primary intervention type -----------------------------------
        if "interventions" in tables:
            iv = tables["interventions"].copy()
            primary_iv = (
                iv.groupby("nct_id")["intervention_type"]
                .first()
                .reset_index()
            )
            df = df.merge(primary_iv, on="nct_id", how="left")
        else:
            df["intervention_type"] = "Unknown"

        # ---- Lead sponsor + type + historical performance ---------------
        if "sponsors" in tables:
            sp = tables["sponsors"].copy()
            if "lead_or_collaborator" in sp.columns:
                sp = sp[sp["lead_or_collaborator"] == "lead"]

            sp["sponsor_type"] = (
                sp["agency_class"].astype(str).str.strip().str.title()
                .map(SPONSOR_TYPE_MAP)
                .fillna("academic")
            )

            # Historical performance: count of completed trials per sponsor
            # (count-based proxy — avoids look-ahead bias; actual enrollment_ratio
            # is not available at feature-engineering time for real AACT flat files)
            completed_ncts = set(
                df.loc[df["overall_status"].str.upper() == "COMPLETED", "nct_id"]
            )
            sponsor_hist = (
                sp[sp["nct_id"].isin(completed_ncts)]
                .groupby("name")
                .size()
                .reset_index(name="sponsor_completed_trials")
            )
            # Merge sponsor info onto main df
            sp_info = sp[["nct_id", "name", "sponsor_type", "agency_class"]].rename(
                columns={"name": "sponsor_name", "agency_class": "sponsor_agency_class"}
            )
            df = df.merge(sp_info, on="nct_id", how="left")
            df = df.merge(
                sponsor_hist.rename(columns={"name": "sponsor_name"}),
                on="sponsor_name", how="left",
            )
            # Normalise to [0, 1] — sponsor_historical_performance
            max_trials = df["sponsor_completed_trials"].max()
            df["sponsor_historical_performance"] = (
                df["sponsor_completed_trials"].fillna(0) /
                max(max_trials, 1)
            )
        else:
            df["sponsor_name"] = "Unknown"
            df["sponsor_type"] = "academic"
            df["sponsor_historical_performance"] = 0.5

        # ---- Eligibility criteria complexity ----------------------------
        if "eligibilities" in tables:
            elig = tables["eligibilities"].copy()

            # Real AACT flat files store a single free-text 'criteria' blob per
            # trial; synthetic data may not populate this column at all.
            # Strategy:
            #   1. If 'criteria' exists and has content → count non-blank lines.
            #   2. Otherwise fall back to row-count per nct_id (one row = one
            #      eligibility criterion in some AACT schema versions).
            criteria_has_content = (
                "criteria" in elig.columns
                and elig["criteria"].notna().sum() > 0
            )
            if criteria_has_content:
                log.info(
                    "eligibilities.criteria: %d non-null values — parsing line counts.",
                    elig["criteria"].notna().sum(),
                )
                elig["n_eligibility_criteria"] = (
                    elig["criteria"]
                    .fillna("")
                    .apply(lambda t: sum(1 for l in t.splitlines() if l.strip()))
                )
            else:
                log.info(
                    "eligibilities.criteria column absent or empty — "
                    "using row-count per nct_id as complexity proxy."
                )
                row_counts = (
                    elig.groupby("nct_id")
                    .size()
                    .reset_index(name="n_eligibility_criteria")
                )
                elig = elig.merge(row_counts, on="nct_id", how="left")

            elig_cols = ["nct_id", "n_eligibility_criteria"]
            for col in ("gender", "healthy_volunteers"):
                if col in elig.columns:
                    elig_cols.append(col)

            df = df.merge(elig[elig_cols].drop_duplicates("nct_id"), on="nct_id", how="left")
        else:
            df["n_eligibility_criteria"] = np.nan

        # ---- Design features -------------------------------------------
        if "designs" in tables:
            des = tables["designs"][
                [c for c in ["nct_id", "intervention_model", "primary_purpose", "masking", "allocation"]
                 if c in tables["designs"].columns]
            ].copy()
            df = df.merge(des, on="nct_id", how="left")

        # ---- Derived targets -------------------------------------------
        df = self._compute_targets(df)

        # ---- Competing trials count ------------------------------------
        if "conditions" in tables:
            df = self._compute_competing_trials(df, tables["conditions"])

        # ---- Duration --------------------------------------------------
        # Priority: calculated_values.actual_duration > date arithmetic.
        if "cv_duration_days" in df.columns:
            df["enrollment_duration_days"] = pd.to_numeric(
                df["cv_duration_days"], errors="coerce"
            ).clip(lower=0)
            if "start_date" in df.columns and "completion_date" in df.columns:
                date_based = (
                    (df["completion_date"] - df["start_date"]).dt.days
                ).clip(lower=0)
                df["enrollment_duration_days"] = df[
                    "enrollment_duration_days"
                ].fillna(date_based)
            df.drop(columns=["cv_duration_days"], inplace=True)
        elif "start_date" in df.columns and "completion_date" in df.columns:
            df["enrollment_duration_days"] = (
                (df["completion_date"] - df["start_date"]).dt.days
            ).clip(lower=0)

        # Drop only clear data-entry errors (< 7 days is implausible for any trial).
        if "enrollment_duration_days" in df.columns:
            short_mask = df["enrollment_duration_days"].notna() & (df["enrollment_duration_days"] < 7)
            n_dropped  = int(short_mask.sum())
            if n_dropped:
                log.info("Dropping %d trials with enrollment_duration_days < 7.", n_dropped)
            df = df[~short_mask]

        # ---- Final dtype alignment ------------------------------------
        df["geographic_concentration"] = df["geographic_concentration"].fillna(1.0)
        df["n_eligibility_criteria"]   = df["n_eligibility_criteria"].fillna(
            df["n_eligibility_criteria"].median()
        )

        log.info("Feature engineering complete: %d rows × %d columns", *df.shape)
        self.modeling_df = df
        return df

    # ------------------------------------------------------------------
    # 4. Competitor data helper
    # ------------------------------------------------------------------

    def get_competitor_data(
        self,
        condition: str,
        start_date: str | pd.Timestamp,
        end_date: str | pd.Timestamp,
    ) -> pd.DataFrame:
        """
        Return trials competing for the same patient population in a time window.

        A trial is considered a competitor if:
          - It targets the same condition (case-insensitive substring match).
          - Its recruiting window overlaps [start_date, end_date].

        Args:
            condition:  Condition name or substring (e.g. "type 2 diabetes").
            start_date: Window start date.
            end_date:   Window end date.

        Returns:
            DataFrame of competing trials with key fields.

        Raises:
            RuntimeError: If engineer_features() has not been called first.
        """
        if self.modeling_df is None:
            raise RuntimeError("Call engineer_features() before get_competitor_data().")

        start_dt = pd.to_datetime(start_date)
        end_dt   = pd.to_datetime(end_date)
        df = self.modeling_df

        # Condition match
        cond_mask = (
            df["condition"]
            .fillna("")
            .str.lower()
            .str.contains(condition.lower(), regex=False)
        )

        # Temporal overlap: trial_start <= window_end AND trial_end >= window_start
        comp_end = df["completion_date"].fillna(pd.Timestamp("2030-12-31"))
        time_mask = (df["start_date"] <= end_dt) & (comp_end >= start_dt)

        cols = [
            "nct_id", "condition", "overall_status", "phase",
            "enrollment", "start_date", "completion_date",
            "sponsor_name", "sponsor_type", "n_countries", "n_facilities",
        ]
        available = [c for c in cols if c in df.columns]
        result = df.loc[cond_mask & time_mask, available].copy()
        log.info(
            "get_competitor_data('%s', %s–%s): %d trials",
            condition, start_date, end_date, len(result),
        )
        return result.reset_index(drop=True)

    # ------------------------------------------------------------------
    # 5. Site performance helper
    # ------------------------------------------------------------------

    def get_site_performance(self, country: Optional[str] = None) -> pd.DataFrame:
        """
        Aggregate site-level enrollment performance metrics.

        Requires the 'facilities' table to have been loaded.

        Args:
            country: Optional ISO country name filter (e.g. "United States").
                     If None, returns all countries.

        Returns:
            DataFrame with one row per (facility, country, condition) with
            performance metrics (trial count, completion rate, avg enrollment).

        Raises:
            RuntimeError: If engineer_features() has not been called first.
        """
        if self.modeling_df is None:
            raise RuntimeError("Call engineer_features() before get_site_performance().")
        if "facilities" not in self._tables:
            raise RuntimeError("'facilities' table was not loaded.")

        fac = self._tables["facilities"].copy()
        if "status" in fac.columns:
            fac = fac[fac["status"] != "Withdrawn"]
        if country:
            fac = fac[fac["country"].str.lower() == country.lower()]

        merged = fac.merge(
            self.modeling_df[[
                "nct_id", "condition", "enrollment_ratio",
                "enrollment_met_target", "enrollment_duration_days", "enrollment",
            ]],
            on="nct_id", how="left",
        )

        site_cols = ["facility_name" if "facility_name" in merged.columns else "name", "city", "country"]
        site_cols = [c for c in site_cols if c in merged.columns]
        group_cols = site_cols + ["condition"]

        agg = merged.groupby(group_cols, dropna=False).agg(
            n_trials=("nct_id", "nunique"),
            n_met_target=("enrollment_met_target", "sum"),
            avg_enrollment_ratio=("enrollment_ratio", "mean"),
            median_enrollment_ratio=("enrollment_ratio", "median"),
            avg_duration_days=("enrollment_duration_days", "mean"),
            avg_target_enrollment=("enrollment", "mean"),
        ).reset_index()

        agg["completion_rate_pct"] = (
            agg["n_met_target"] / agg["n_trials"].clip(lower=1) * 100
        ).round(1)

        agg["performance_score"] = (
            0.4 * agg["completion_rate_pct"].clip(upper=100) +
            0.4 * (agg["avg_enrollment_ratio"].clip(upper=1) * 100) +
            0.2 * (agg["n_trials"].clip(upper=20) / 20 * 100)
        ).round(1)

        agg.sort_values("performance_score", ascending=False, inplace=True)
        return agg.reset_index(drop=True)

    # ------------------------------------------------------------------
    # 6. Persist / load
    # ------------------------------------------------------------------

    def save_processed_data(self, output_dir: str | Path) -> None:
        """
        Save the modeling DataFrame to parquet.

        Args:
            output_dir: Directory to write modeling_df.parquet.

        Raises:
            RuntimeError: If engineer_features() has not been called first.
        """
        if self.modeling_df is None:
            raise RuntimeError("No processed data to save. Run engineer_features() first.")
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        path = output_dir / "modeling_df.parquet"
        self.modeling_df.to_parquet(path, index=False)
        log.info("Saved modeling dataset: %s (%d rows)", path, len(self.modeling_df))

    def load_processed_data(self, input_dir: str | Path) -> pd.DataFrame:
        """
        Load a previously saved modeling DataFrame from parquet.

        Args:
            input_dir: Directory containing modeling_df.parquet.

        Returns:
            Modeling DataFrame.

        Raises:
            FileNotFoundError: If the file does not exist.
        """
        path = Path(input_dir) / "modeling_df.parquet"
        if not path.exists():
            raise FileNotFoundError(f"No saved modeling data found at: {path}")
        self.modeling_df = pd.read_parquet(path)
        log.info("Loaded modeling dataset: %s (%d rows)", path, len(self.modeling_df))
        return self.modeling_df

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _compute_targets(self, df: pd.DataFrame) -> pd.DataFrame:
        """Derive classification and regression targets from trial outcome data.

        Classification target — enrollment_met_target
        ---------------------------------------------
        Uses overall_status as a proxy for enrollment success (AACT flat files
        do not provide both target and actual enrollment in the same row, so a
        ratio-based label is not available without a supplemental data source).

        Label mapping:
          - COMPLETED  → 1.0  (trial successfully reached its enrollment goal)
          - TERMINATED → 0.0  (trial failed; did not complete enrollment)
          - All other statuses (Recruiting, Active, Not Yet Recruiting, …)
            → NaN  (outcome unknown; row kept for inference but excluded from
                    supervised training)

        An informational flag ``terminated_for_enrollment`` is also set to True
        when a terminated trial's why_stopped text contains a recruitment-related
        keyword (see RECRUITMENT_FAIL_KEYWORDS).  This flag does not change the
        binary label — all terminated trials receive 0.

        Regression target — enrollment_duration_days
        --------------------------------------------
        completion_date − start_date, clipped to ≥ 0.  Meaningful only for
        COMPLETED and TERMINATED trials that have both date fields populated.

        Args:
            df: DataFrame with overall_status, and optionally why_stopped,
                start_date, and completion_date columns.

        Returns:
            DataFrame with enrollment_met_target, terminated_for_enrollment,
            and enrollment_duration_days columns added.
        """
        # ── classification label from overall_status ──────────────────────────
        # overall_status has been title-cased by engineer_features() before
        # this call ("Completed", "Terminated", …).  Uppercasing here ensures
        # the match is robust to any future ordering change.
        status = df["overall_status"].astype(str).str.upper().str.strip()

        is_completed  = status == "COMPLETED"
        is_terminated = status == "TERMINATED"

        df["enrollment_met_target"] = np.where(
            is_completed,  1.0,
            np.where(is_terminated, 0.0, np.nan),
        )

        # ── terminated_for_enrollment informational flag ──────────────────────
        if "why_stopped" in df.columns:
            why     = df["why_stopped"].fillna("").astype(str).str.lower()
            pattern = "|".join(RECRUITMENT_FAIL_KEYWORDS)
            df["terminated_for_enrollment"] = (
                is_terminated & why.str.contains(pattern, regex=True)
            )
        else:
            df["terminated_for_enrollment"] = False

        # ── label diagnostics ─────────────────────────────────────────────────
        n_total      = len(df)
        n_completed  = int(is_completed.sum())
        n_terminated = int(is_terminated.sum())
        n_labeled    = n_completed + n_terminated
        n_unlabeled  = n_total - n_labeled
        pct_met      = n_completed / n_labeled * 100 if n_labeled > 0 else 0.0
        log.info(
            "Target labels: %d / %d rows labeled "
            "(%d completed=1 [%.1f%%], %d terminated=0); "
            "%d rows unlabeled (NaN — excluded from training).",
            n_labeled, n_total,
            n_completed, pct_met, n_terminated,
            n_unlabeled,
        )

        return df

    def _compute_competing_trials(
        self,
        df: pd.DataFrame,
        conditions_table: pd.DataFrame,
    ) -> pd.DataFrame:
        """
        Compute per-study competing_trials_count.

        For each study, counts other studies targeting the same condition
        whose recruiting windows overlap.  Uses a vectorised merge approach
        rather than a per-row loop for performance.

        Args:
            df:               Main modeling DataFrame.
            conditions_table: Raw conditions DataFrame.

        Returns:
            df with 'competing_trials_count' column added.
        """
        if "condition" not in df.columns:
            df["competing_trials_count"] = 0
            return df

        log.info("Computing competing trial counts (this may take a moment) …")

        # Use lower-cased condition for matching
        df["_cond_lower"] = df["condition"].str.lower().fillna("")

        end_col = "completion_date"
        df["_end"] = df[end_col].fillna(pd.Timestamp("2035-12-31"))

        # Build a compact lookup: nct_id, condition, start, end
        lookup = df[["nct_id", "_cond_lower", "start_date", "_end"]].dropna(
            subset=["start_date"]
        )

        # Self-join on condition, then filter for date overlap
        joined = lookup.merge(
            lookup.rename(columns={
                "nct_id":      "nct_id_b",
                "start_date":  "start_b",
                "_end":        "end_b",
            }),
            on="_cond_lower",
            how="left",
        )
        # Exclude self-pairs and non-overlapping windows
        overlapping = joined[
            (joined["nct_id"] != joined["nct_id_b"]) &
            (joined["start_date"] <= joined["end_b"]) &
            (joined["_end"]        >= joined["start_b"])
        ]
        counts = (
            overlapping.groupby("nct_id")
            .size()
            .reset_index(name="competing_trials_count")
        )
        df = df.merge(counts, on="nct_id", how="left")
        df["competing_trials_count"] = df["competing_trials_count"].fillna(0).astype(int)
        df.drop(columns=["_cond_lower", "_end"], inplace=True)
        log.info("Competing trial counts computed.")
        return df
