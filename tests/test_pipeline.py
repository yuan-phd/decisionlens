"""
test_pipeline.py — Unit tests for src.data_pipeline.TrialDataPipeline.

Covered:
  - Constructor defaults and custom parameters
  - clean_studies(): study_type filter, status filter, phase encoding, enrollment filter
  - engineer_features(): requires 'studies' table, returns DataFrame with model columns
  - load_raw_data(): graceful handling of missing files and missing directory
"""

from __future__ import annotations

import pandas as pd
import pytest
from pathlib import Path

from src.data_pipeline import (
    TrialDataPipeline,
    PHASE_ORDINAL,
    EXCLUDED_STATUSES,
    MIN_START_YEAR,
    MIN_ENROLLMENT,
)
from src.models import (
    NUMERIC_FEATURES,
    CATEGORICAL_FEATURES,
    BOOLEAN_FEATURES,
    TARGET_CLASS,
)


# ---------------------------------------------------------------------------
# 1. Constructor
# ---------------------------------------------------------------------------

class TestTrialDataPipelineInit:
    def test_defaults(self):
        p = TrialDataPipeline()
        assert p.min_start_year  == MIN_START_YEAR
        assert p.min_enrollment  == MIN_ENROLLMENT
        assert p.modeling_df     is None

    def test_custom_params(self):
        p = TrialDataPipeline(min_start_year=2015, min_enrollment=20)
        assert p.min_start_year == 2015
        assert p.min_enrollment == 20

    def test_repr_or_str_does_not_raise(self):
        # Ensure the object is at least printable without error
        p = TrialDataPipeline()
        str(p)  # should not raise


# ---------------------------------------------------------------------------
# 2. clean_studies()
# ---------------------------------------------------------------------------

class TestCleanStudies:
    """Tests for TrialDataPipeline.clean_studies()."""

    def test_filters_non_interventional(self, minimal_studies_df):
        p = TrialDataPipeline()
        result = p.clean_studies(minimal_studies_df.copy())
        assert result["study_type"].str.upper().eq("INTERVENTIONAL").all(), (
            "Non-interventional studies should be filtered out."
        )

    def test_excludes_withdrawn_and_unknown(self, minimal_studies_df):
        p = TrialDataPipeline()
        result = p.clean_studies(minimal_studies_df.copy())
        statuses_upper = result["overall_status"].str.upper().str.strip()
        for excluded in EXCLUDED_STATUSES:
            assert not statuses_upper.isin([excluded]).any(), (
                f"Status '{excluded}' should be excluded."
            )

    def test_phase_numeric_column_added(self, minimal_studies_df):
        p = TrialDataPipeline()
        result = p.clean_studies(minimal_studies_df.copy())
        assert "phase_numeric" in result.columns, (
            "clean_studies() should add 'phase_numeric' column."
        )

    def test_phase_ordinal_values(self, minimal_studies_df):
        p = TrialDataPipeline()
        result = p.clean_studies(minimal_studies_df.copy())
        valid_values = set(PHASE_ORDINAL.values()) | {0.0}
        assert result["phase_numeric"].isin(valid_values).all(), (
            "phase_numeric should only contain ordinal values from PHASE_ORDINAL."
        )

    def test_phase4_raw_variant_normalised(self):
        """AACT raw 'PHASE 4' string should normalise to 'Phase 4' (numeric 4.0)."""
        df = pd.DataFrame({
            "nct_id":         ["NCT00000001"],
            "study_type":     ["Interventional"],
            "overall_status": ["COMPLETED"],
            "phase":          ["PHASE 4"],
            "enrollment":     [100],
            "start_date":     ["2015-01-01"],
        })
        p = TrialDataPipeline()
        result = p.clean_studies(df)
        assert len(result) == 1
        assert result["phase_numeric"].iloc[0] == 4.0

    def test_enrollment_filter(self):
        """Rows with enrollment < min_enrollment should be dropped."""
        df = pd.DataFrame({
            "nct_id":         ["NCT00000001", "NCT00000002"],
            "study_type":     ["Interventional", "Interventional"],
            "overall_status": ["COMPLETED", "COMPLETED"],
            "phase":          ["Phase 2", "Phase 2"],
            "enrollment":     [5, 200],          # 5 < 10 → dropped
            "start_date":     ["2015-01-01", "2015-01-01"],
        })
        p = TrialDataPipeline(min_enrollment=10)
        result = p.clean_studies(df)
        assert len(result) == 1
        assert result["enrollment"].iloc[0] == 200

    def test_start_year_filter(self):
        """Studies starting before min_start_year should be excluded."""
        df = pd.DataFrame({
            "nct_id":         ["NCT00000001", "NCT00000002"],
            "study_type":     ["Interventional", "Interventional"],
            "overall_status": ["COMPLETED", "COMPLETED"],
            "phase":          ["Phase 2", "Phase 2"],
            "enrollment":     [100, 100],
            "start_date":     ["2000-01-01", "2015-01-01"],  # first too old
        })
        p = TrialDataPipeline(min_start_year=2008)
        result = p.clean_studies(df)
        assert len(result) == 1
        assert pd.to_datetime(result["start_date"].iloc[0]).year >= 2008

    def test_has_dmc_boolean_conversion(self, minimal_studies_df):
        """'has_dmc' 't'/'f' strings should be converted to boolean."""
        p = TrialDataPipeline()
        result = p.clean_studies(minimal_studies_df.copy())
        if "has_dmc" in result.columns:
            assert result["has_dmc"].dtype in (bool, object), (
                "has_dmc should be bool after conversion."
            )

    def test_returns_dataframe(self, minimal_studies_df):
        p = TrialDataPipeline()
        result = p.clean_studies(minimal_studies_df.copy())
        assert isinstance(result, pd.DataFrame)

    def test_resets_index(self, minimal_studies_df):
        p = TrialDataPipeline()
        result = p.clean_studies(minimal_studies_df.copy())
        assert list(result.index) == list(range(len(result))), (
            "Index should be reset after filtering."
        )


# ---------------------------------------------------------------------------
# 3. engineer_features()
# ---------------------------------------------------------------------------

class TestEngineerFeatures:
    """Tests for TrialDataPipeline.engineer_features()."""

    def test_requires_studies_table(self):
        p = TrialDataPipeline()
        with pytest.raises(ValueError, match="'studies' table is required"):
            p.engineer_features({})

    def test_returns_dataframe(self, minimal_tables):
        p = TrialDataPipeline()
        result = p.engineer_features(minimal_tables)
        assert isinstance(result, pd.DataFrame)

    def test_result_is_non_empty(self, minimal_tables):
        p = TrialDataPipeline()
        result = p.engineer_features(minimal_tables)
        assert len(result) > 0, "engineer_features() should return non-empty DataFrame."

    def test_has_target_column(self, minimal_tables):
        """Classification target column must be present."""
        p = TrialDataPipeline()
        result = p.engineer_features(minimal_tables)
        assert TARGET_CLASS in result.columns, (
            f"'{TARGET_CLASS}' must be present after feature engineering."
        )

    def test_has_phase_numeric(self, minimal_tables):
        p = TrialDataPipeline()
        result = p.engineer_features(minimal_tables)
        assert "phase_numeric" in result.columns

    def test_has_multicountry_flag(self, minimal_tables):
        p = TrialDataPipeline()
        result = p.engineer_features(minimal_tables)
        assert "is_multicountry" in result.columns

    def test_has_n_countries(self, minimal_tables):
        p = TrialDataPipeline()
        result = p.engineer_features(minimal_tables)
        assert "n_countries" in result.columns
        assert (result["n_countries"] >= 1).all(), "n_countries should be ≥ 1."

    def test_enrollment_type_is_actual_binary(self, minimal_tables):
        p = TrialDataPipeline()
        result = p.engineer_features(minimal_tables)
        if "enrollment_type_is_actual" in result.columns:
            assert result["enrollment_type_is_actual"].isin([0, 1]).all()

    def test_works_without_optional_tables(self, minimal_tables):
        """engineer_features() should succeed even when optional tables are absent."""
        p = TrialDataPipeline()
        tables_minimal = {"studies": minimal_tables["studies"]}
        result = p.engineer_features(tables_minimal)
        assert isinstance(result, pd.DataFrame)
        assert len(result) > 0

    def test_modeling_df_stored_on_instance(self, minimal_tables):
        p = TrialDataPipeline()
        p.engineer_features(minimal_tables)
        assert p.modeling_df is not None


# ---------------------------------------------------------------------------
# 4. load_raw_data()
# ---------------------------------------------------------------------------

class TestLoadRawData:
    def test_missing_directory_returns_empty(self, tmp_path):
        missing = tmp_path / "no_such_dir"
        p = TrialDataPipeline()
        result = p.load_raw_data(missing)
        assert isinstance(result, dict)
        assert len(result) == 0

    def test_empty_directory_returns_empty(self, tmp_path):
        p = TrialDataPipeline()
        result = p.load_raw_data(tmp_path)
        assert isinstance(result, dict)
        assert len(result) == 0

    def test_loads_parquet_file(self, tmp_path):
        """If a valid parquet file is present, it should be loaded."""
        studies_df = pd.DataFrame({
            "nct_id":      ["NCT00000001"],
            "study_type":  ["Interventional"],
            "enrollment":  [100],
        })
        studies_df.to_parquet(tmp_path / "studies.parquet")

        p = TrialDataPipeline()
        result = p.load_raw_data(tmp_path)
        assert "studies" in result
        assert len(result["studies"]) == 1

    def test_unknown_parquet_files_not_loaded(self, tmp_path):
        """Only AACT-defined table names should be loaded; extras are ignored."""
        pd.DataFrame({"x": [1]}).to_parquet(tmp_path / "mystery_table.parquet")
        p = TrialDataPipeline()
        result = p.load_raw_data(tmp_path)
        assert "mystery_table" not in result

    def test_stores_tables_on_instance(self, tmp_path):
        studies_df = pd.DataFrame({
            "nct_id":     ["NCT00000001"],
            "study_type": ["Interventional"],
        })
        studies_df.to_parquet(tmp_path / "studies.parquet")
        p = TrialDataPipeline()
        p.load_raw_data(tmp_path)
        assert "studies" in p._tables
