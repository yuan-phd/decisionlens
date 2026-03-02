"""
conftest.py — Shared pytest fixtures for DecisionLENS test suite.

Provides two main fixtures:
  - ``minimal_studies_df``  : Raw studies DataFrame for TrialDataPipeline tests.
  - ``modeling_df``         : Engineered modeling DataFrame for EnrollmentForecaster tests.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.models import (
    TARGET_CLASS,
    TARGET_REG,
    EVENT_COL,
    NUMERIC_FEATURES,
    CATEGORICAL_FEATURES,
    BOOLEAN_FEATURES,
)

# ---------------------------------------------------------------------------
# RNG
# ---------------------------------------------------------------------------

_RNG = np.random.default_rng(42)


# ---------------------------------------------------------------------------
# Studies fixture (for TrialDataPipeline tests)
# ---------------------------------------------------------------------------

@pytest.fixture
def minimal_studies_df() -> pd.DataFrame:
    """
    Minimal AACT-style studies DataFrame that survives clean_studies().

    Contains a mix of statuses and phases so multiple code-paths are exercised.
    """
    n = 60
    rng = np.random.default_rng(1)
    return pd.DataFrame({
        "nct_id":          [f"NCT{i:08d}" for i in range(n)],
        "study_type":      ["Interventional"] * 50 + ["Observational"] * 10,
        "overall_status":  (
            ["COMPLETED"] * 30
            + ["TERMINATED"] * 10
            + ["RECRUITING"] * 10
            + ["WITHDRAWN"] * 5
            + ["UNKNOWN STATUS"] * 5
        ),
        "phase": (
            ["Phase 2"] * 20
            + ["Phase 3"] * 20
            + ["Phase 1"] * 10
            + ["PHASE 4"] * 5           # raw AACT variant → should normalise
            + ["N/A"] * 5
        ),
        "enrollment":      rng.integers(50, 500, size=n).tolist(),
        "start_date":      ["2015-01-01"] * n,
        "completion_date": ["2020-06-01"] * 30 + [None] * 30,
        "enrollment_type": ["ESTIMATED"] * n,
        "has_dmc":         (["t"] * 30 + ["f"] * 30),
    })


# ---------------------------------------------------------------------------
# Tables fixture (for engineer_features tests)
# ---------------------------------------------------------------------------

@pytest.fixture
def minimal_tables(minimal_studies_df: pd.DataFrame) -> dict[str, pd.DataFrame]:
    """
    Minimal set of AACT table DataFrames for TrialDataPipeline.engineer_features().

    Only includes the tables that are explicitly branched on in engineer_features().
    The studies table is a clean version of minimal_studies_df.
    """
    # Use only the Interventional non-withdrawn rows as the studies table
    studies = minimal_studies_df[
        minimal_studies_df["study_type"].str.upper() == "INTERVENTIONAL"
    ].copy()

    nct_ids = studies["nct_id"].tolist()
    n = len(nct_ids)

    # countries: two countries per study for ~half the studies
    ctry_rows = []
    for i, nct in enumerate(nct_ids):
        ctry_rows.append({"nct_id": nct, "name": "United States", "removed": "f"})
        if i % 2 == 0:
            ctry_rows.append({"nct_id": nct, "name": "Germany", "removed": "f"})
    countries = pd.DataFrame(ctry_rows)

    # designs
    designs = pd.DataFrame({
        "nct_id":             nct_ids,
        "intervention_model": ["Parallel Assignment"] * n,
        "masking":            ["Double"] * n,
    })

    # eligibilities (plain text; line count is the proxy)
    eligibilities = pd.DataFrame({
        "nct_id":   nct_ids,
        "criteria": [
            "Inclusion Criteria:\n  1. criterion\n  2. criterion\n"
            "Exclusion Criteria:\n  1. criterion"
        ] * n,
    })

    # sponsors
    sponsors = pd.DataFrame({
        "nct_id":               nct_ids,
        "lead_or_collaborator": ["lead"] * n,
        "agency_class":         ["INDUSTRY"] * (n // 2) + ["NIH"] * (n - n // 2),
        "name":                 ["BIG PHARMA CO"] * n,
    })

    return {
        "studies":      studies,
        "countries":    countries,
        "designs":      designs,
        "eligibilities": eligibilities,
        "sponsors":     sponsors,
    }


# ---------------------------------------------------------------------------
# Modeling DataFrame fixture (for EnrollmentForecaster tests)
# ---------------------------------------------------------------------------

def _make_modeling_df(n: int = 300) -> pd.DataFrame:
    """
    Build a synthetic modeling DataFrame that matches what
    TrialDataPipeline.engineer_features() produces.

    Guarantees:
      - ≥ 100 labeled rows (for classifier minimum)
      - ≥ 50 rows with valid positive duration (for regressor)
      - Both classes represented in enrollment_met_target
      - All required feature columns present
    """
    rng = np.random.default_rng(0)

    n_unlabeled  = 50                # active / recruiting trials (NaN)
    n_terminated = 50                # class 0
    n_completed  = n - n_unlabeled - n_terminated  # class 1  (≥ 200)

    rows: dict = {
        "nct_id": [f"NCT{i:08d}" for i in range(n)],
        # Numeric features
        "phase_numeric":                    rng.choice([1.0, 1.5, 2.0, 2.5, 3.0, 4.0], size=n),
        "n_facilities":                     rng.integers(1, 120, size=n),
        "n_countries":                      rng.integers(1, 25, size=n),
        "n_eligibility_criteria":           rng.integers(5, 55, size=n),
        "geographic_concentration":         rng.uniform(0.1, 1.0, size=n),
        "condition_prevalence_proxy":       rng.uniform(5.0, 250.0, size=n),
        "sponsor_historical_performance":   rng.uniform(0.3, 1.0, size=n),
        "competing_trials_count":           rng.integers(0, 60, size=n),
        "enrollment":                       rng.integers(50, 2500, size=n),
        "enrollment_type_is_actual":        rng.integers(0, 2, size=n),
        # Categorical features
        "sponsor_type": rng.choice(
            ["industry", "government", "academic", "other"], size=n
        ),
        "intervention_model": rng.choice(
            ["Parallel Assignment", "Single Group Assignment", "Crossover Assignment"],
            size=n,
        ),
        "masking": rng.choice(
            ["None (Open Label)", "Single", "Double", "Triple"], size=n
        ),
        # Boolean feature
        "is_multicountry": rng.integers(0, 2, size=n).astype(bool),
    }

    df = pd.DataFrame(rows)

    # Build classification target: NaN for unlabeled, 1 for completed, 0 for terminated
    labels = np.concatenate([
        np.full(n_unlabeled, np.nan),
        np.zeros(n_terminated),
        np.ones(n_completed),
    ])
    rng.shuffle(labels)
    df[TARGET_CLASS] = labels

    # Build regression target: positive days for labeled rows, NaN for unlabeled
    durations = np.where(
        df[TARGET_CLASS].notna(),
        rng.integers(30, 1500, size=n).astype(float),
        np.nan,
    )
    df[TARGET_REG] = durations

    # Survival event column (1 = completed, 0 = censored/terminated)
    df[EVENT_COL] = df[TARGET_CLASS].fillna(0).astype(int)

    return df


@pytest.fixture
def modeling_df() -> pd.DataFrame:
    """Pre-built modeling DataFrame with 300 rows for EnrollmentForecaster tests."""
    return _make_modeling_df(300)


@pytest.fixture
def fast_forecaster_params() -> tuple[dict, dict]:
    """
    Returns (clf_params, reg_params) tuned for speed (n_estimators=5)
    rather than accuracy, so model tests finish in < 5 s.
    """
    clf = {
        "n_estimators":     5,
        "max_depth":        3,
        "random_state":     42,
        "n_jobs":           1,
        "objective":        "binary:logistic",
        "scale_pos_weight": 5.0,
    }
    reg = {
        "n_estimators":     5,
        "max_depth":        3,
        "random_state":     42,
        "n_jobs":           1,
        "objective":        "reg:squarederror",
    }
    return clf, reg
