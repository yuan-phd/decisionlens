"""
test_models.py — Unit tests for src.models.EnrollmentForecaster.

Covered:
  - Constructor defaults and repr
  - Pre-fit guard (RuntimeError on predict before fit)
  - fit(): completes without error on synthetic labeled data
  - predict(): correct shape, column names, probability range, binary label
  - predict_survival(): returns S(t) DataFrame when lifelines is available
  - feature_importances: non-empty DataFrame after fit
  - save() / load() round-trip
  - Missing feature columns handled gracefully (filled with defaults)

Notes:
  - Uses fast XGBoost params (n_estimators=5) to keep test runtime < 10 s.
  - Tests are skipped when xgboost is not installed (CI-friendly).
"""

from __future__ import annotations

import importlib
import pytest
import numpy as np
import pandas as pd

from src.models import (
    EnrollmentForecaster,
    CLF_THRESHOLD,
    TARGET_CLASS,
    TARGET_REG,
    NUMERIC_FEATURES,
    CATEGORICAL_FEATURES,
    BOOLEAN_FEATURES,
)

# Skip entire module when xgboost is absent (CI / lightweight env)
xgboost_available = importlib.util.find_spec("xgboost") is not None
pytestmark = pytest.mark.skipif(
    not xgboost_available,
    reason="xgboost not installed",
)


# ---------------------------------------------------------------------------
# Helper — build a fitted forecaster (shared across several tests)
# ---------------------------------------------------------------------------

def _fitted_forecaster(modeling_df, fast_clf, fast_reg) -> EnrollmentForecaster:
    fc = EnrollmentForecaster(clf_params=fast_clf, reg_params=fast_reg)
    fc.fit(modeling_df)
    return fc


# ---------------------------------------------------------------------------
# 1. Constructor & repr
# ---------------------------------------------------------------------------

class TestEnrollmentForecasterInit:
    def test_default_construction(self):
        fc = EnrollmentForecaster()
        assert fc._is_fitted is False
        assert fc._clf_pipeline is None
        assert fc._reg_pipeline is None

    def test_repr_unfitted(self):
        fc = EnrollmentForecaster()
        r = repr(fc)
        assert "not fitted" in r

    def test_repr_fitted(self, modeling_df, fast_forecaster_params):
        clf_p, reg_p = fast_forecaster_params
        fc = EnrollmentForecaster(clf_params=clf_p, reg_params=reg_p)
        fc.fit(modeling_df)
        assert "fitted" in repr(fc)

    def test_custom_clf_params_stored(self):
        custom = {"n_estimators": 10, "random_state": 99}
        fc = EnrollmentForecaster(clf_params=custom)
        assert fc.clf_params["n_estimators"] == 10

    def test_custom_reg_params_stored(self):
        custom = {"n_estimators": 7, "random_state": 0}
        fc = EnrollmentForecaster(reg_params=custom)
        assert fc.reg_params["n_estimators"] == 7


# ---------------------------------------------------------------------------
# 2. Pre-fit guards
# ---------------------------------------------------------------------------

class TestPreFitGuards:
    def test_predict_raises_before_fit(self, modeling_df):
        fc = EnrollmentForecaster()
        with pytest.raises(RuntimeError):
            fc.predict(modeling_df.head(5))

    def test_predict_survival_raises_before_fit(self, modeling_df):
        fc = EnrollmentForecaster()
        with pytest.raises(RuntimeError):
            fc.predict_survival(modeling_df.head(5))

    def test_evaluate_raises_before_fit(self, modeling_df):
        fc = EnrollmentForecaster()
        with pytest.raises(RuntimeError):
            fc.evaluate(modeling_df)

    def test_feature_importances_raises_before_fit(self):
        fc = EnrollmentForecaster()
        with pytest.raises(RuntimeError):
            _ = fc.feature_importances

    def test_save_raises_before_fit(self, tmp_path):
        fc = EnrollmentForecaster()
        with pytest.raises(RuntimeError):
            fc.save(tmp_path / "model.joblib")

    def test_validate_columns_raises_on_missing_target(self, modeling_df):
        """fit() should raise ValueError if the target column is absent."""
        bad_df = modeling_df.drop(columns=[TARGET_CLASS])
        fc = EnrollmentForecaster()
        with pytest.raises(ValueError, match=TARGET_CLASS):
            fc.fit(bad_df)

    def test_fit_raises_on_too_few_labeled_rows(self):
        """fit() needs ≥ 100 labeled rows."""
        small_df = pd.DataFrame({
            "phase_numeric": [2.0] * 50,
            "n_facilities":  [10] * 50,
            "n_countries":   [2] * 50,
            "n_eligibility_criteria": [15] * 50,
            "geographic_concentration": [0.5] * 50,
            "condition_prevalence_proxy": [50.0] * 50,
            "sponsor_historical_performance": [0.7] * 50,
            "competing_trials_count": [10] * 50,
            "enrollment": [300] * 50,
            "enrollment_type_is_actual": [0] * 50,
            "sponsor_type": ["industry"] * 50,
            "intervention_model": ["Parallel Assignment"] * 50,
            "masking": ["Double"] * 50,
            "is_multicountry": [True] * 50,
            TARGET_CLASS: [1.0] * 40 + [0.0] * 10,
            TARGET_REG:   [365.0] * 50,
        })
        fc = EnrollmentForecaster()
        with pytest.raises(ValueError, match="too small"):
            fc.fit(small_df)


# ---------------------------------------------------------------------------
# 3. fit()
# ---------------------------------------------------------------------------

class TestFit:
    def test_fit_sets_is_fitted(self, modeling_df, fast_forecaster_params):
        clf_p, reg_p = fast_forecaster_params
        fc = EnrollmentForecaster(clf_params=clf_p, reg_params=reg_p)
        fc.fit(modeling_df)
        assert fc._is_fitted is True

    def test_fit_populates_pipelines(self, modeling_df, fast_forecaster_params):
        clf_p, reg_p = fast_forecaster_params
        fc = EnrollmentForecaster(clf_params=clf_p, reg_params=reg_p)
        fc.fit(modeling_df)
        assert fc._clf_pipeline is not None
        assert fc._reg_pipeline is not None

    def test_fit_stores_feature_names(self, modeling_df, fast_forecaster_params):
        clf_p, reg_p = fast_forecaster_params
        fc = EnrollmentForecaster(clf_params=clf_p, reg_params=reg_p)
        fc.fit(modeling_df)
        assert isinstance(fc._feature_names, list)
        assert len(fc._feature_names) > 0

    def test_fit_returns_self(self, modeling_df, fast_forecaster_params):
        clf_p, reg_p = fast_forecaster_params
        fc = EnrollmentForecaster(clf_params=clf_p, reg_params=reg_p)
        result = fc.fit(modeling_df)
        assert result is fc


# ---------------------------------------------------------------------------
# 4. predict()
# ---------------------------------------------------------------------------

class TestPredict:
    @pytest.fixture(autouse=True)
    def _setup(self, modeling_df, fast_forecaster_params):
        clf_p, reg_p = fast_forecaster_params
        self.fc = EnrollmentForecaster(clf_params=clf_p, reg_params=reg_p)
        self.fc.fit(modeling_df)
        self.modeling_df = modeling_df

    def test_predict_shape(self):
        n = 10
        result = self.fc.predict(self.modeling_df.head(n))
        assert len(result) == n

    def test_predict_required_columns(self):
        result = self.fc.predict(self.modeling_df.head(5))
        for col in ("p_completed", "pred_label", "pred_duration_days"):
            assert col in result.columns, f"Expected column '{col}' in predict output."

    def test_predict_nct_id_passthrough(self):
        """nct_id should be included in output if present in input."""
        result = self.fc.predict(self.modeling_df.head(5))
        assert "nct_id" in result.columns

    def test_predict_probability_in_unit_interval(self):
        result = self.fc.predict(self.modeling_df.head(50))
        assert (result["p_completed"] >= 0.0).all()
        assert (result["p_completed"] <= 1.0).all()

    def test_predict_label_binary(self):
        result = self.fc.predict(self.modeling_df.head(50))
        assert result["pred_label"].isin([0, 1]).all()

    def test_predict_label_threshold_consistent(self):
        """pred_label should be 1 iff p_completed >= CLF_THRESHOLD."""
        result = self.fc.predict(self.modeling_df.head(50))
        expected_label = (result["p_completed"] >= CLF_THRESHOLD).astype(int)
        pd.testing.assert_series_equal(
            result["pred_label"].reset_index(drop=True),
            expected_label.reset_index(drop=True),
            check_names=False,
        )

    def test_predict_duration_non_negative(self):
        result = self.fc.predict(self.modeling_df.head(50))
        assert (result["pred_duration_days"] >= 0).all()

    def test_predict_reset_index(self):
        """Index should always start at 0."""
        result = self.fc.predict(self.modeling_df.iloc[50:60])
        assert list(result.index) == list(range(10))

    def test_predict_missing_features_handled(self, modeling_df):
        """predict() should not crash when some feature columns are absent."""
        stripped = modeling_df.head(5).drop(columns=["n_facilities", "masking"],
                                             errors="ignore")
        result = self.fc.predict(stripped)
        assert len(result) == 5

    def test_predict_single_row(self):
        """Single-row prediction should return a 1-row DataFrame."""
        result = self.fc.predict(self.modeling_df.head(1))
        assert len(result) == 1


# ---------------------------------------------------------------------------
# 5. predict_survival()
# ---------------------------------------------------------------------------

class TestPredictSurvival:
    @pytest.fixture(autouse=True)
    def _setup(self, modeling_df, fast_forecaster_params):
        clf_p, reg_p = fast_forecaster_params
        self.fc = EnrollmentForecaster(clf_params=clf_p, reg_params=reg_p)
        self.fc.fit(modeling_df)
        self.modeling_df = modeling_df

    def test_predict_survival_returns_dataframe(self):
        lifelines_available = importlib.util.find_spec("lifelines") is not None
        if not lifelines_available:
            pytest.skip("lifelines not installed")
        result = self.fc.predict_survival(self.modeling_df.head(5))
        assert isinstance(result, pd.DataFrame)

    def test_predict_survival_shape(self):
        lifelines_available = importlib.util.find_spec("lifelines") is not None
        if not lifelines_available:
            pytest.skip("lifelines not installed")
        time_pts = [90, 180, 365]
        result = self.fc.predict_survival(self.modeling_df.head(5), time_points=time_pts)
        assert result.shape[0] == 5
        assert result.shape[1] == len(time_pts)

    def test_predict_survival_values_in_unit_interval(self):
        lifelines_available = importlib.util.find_spec("lifelines") is not None
        if not lifelines_available:
            pytest.skip("lifelines not installed")
        result = self.fc.predict_survival(self.modeling_df.head(10),
                                          time_points=[90, 365])
        assert ((result >= 0) & (result <= 1)).all().all()


# ---------------------------------------------------------------------------
# 6. feature_importances property
# ---------------------------------------------------------------------------

class TestFeatureImportances:
    def test_feature_importances_returns_dataframe(self, modeling_df,
                                                   fast_forecaster_params):
        clf_p, reg_p = fast_forecaster_params
        fc = EnrollmentForecaster(clf_params=clf_p, reg_params=reg_p)
        fc.fit(modeling_df)
        fi = fc.feature_importances
        assert isinstance(fi, pd.DataFrame)
        assert len(fi) > 0

    def test_feature_importances_columns(self, modeling_df, fast_forecaster_params):
        clf_p, reg_p = fast_forecaster_params
        fc = EnrollmentForecaster(clf_params=clf_p, reg_params=reg_p)
        fc.fit(modeling_df)
        fi = fc.feature_importances
        for col in ("feature", "clf_importance", "reg_importance", "mean_importance"):
            assert col in fi.columns

    def test_feature_importances_non_negative(self, modeling_df, fast_forecaster_params):
        clf_p, reg_p = fast_forecaster_params
        fc = EnrollmentForecaster(clf_params=clf_p, reg_params=reg_p)
        fc.fit(modeling_df)
        fi = fc.feature_importances
        assert (fi["mean_importance"] >= 0).all()


# ---------------------------------------------------------------------------
# 7. save() / load() round-trip
# ---------------------------------------------------------------------------

class TestSaveLoad:
    def test_save_creates_file(self, modeling_df, fast_forecaster_params, tmp_path):
        clf_p, reg_p = fast_forecaster_params
        fc = EnrollmentForecaster(clf_params=clf_p, reg_params=reg_p)
        fc.fit(modeling_df)
        path = tmp_path / "forecaster.joblib"
        fc.save(path)
        assert path.exists()

    def test_save_creates_parent_dirs(self, modeling_df, fast_forecaster_params,
                                      tmp_path):
        clf_p, reg_p = fast_forecaster_params
        fc = EnrollmentForecaster(clf_params=clf_p, reg_params=reg_p)
        fc.fit(modeling_df)
        nested = tmp_path / "models" / "v1" / "forecaster.joblib"
        fc.save(nested)
        assert nested.exists()

    def test_load_nonexistent_raises(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            EnrollmentForecaster.load(tmp_path / "missing.joblib")

    def test_save_load_predicts_identically(self, modeling_df, fast_forecaster_params,
                                            tmp_path):
        """Loaded model should produce bit-identical predictions."""
        clf_p, reg_p = fast_forecaster_params
        fc = EnrollmentForecaster(clf_params=clf_p, reg_params=reg_p)
        fc.fit(modeling_df)

        path = tmp_path / "forecaster.joblib"
        fc.save(path)
        fc2 = EnrollmentForecaster.load(path)

        sample = modeling_df.head(20)
        preds_orig   = fc.predict(sample)
        preds_loaded = fc2.predict(sample)

        pd.testing.assert_frame_equal(preds_orig, preds_loaded)

    def test_loaded_model_is_fitted(self, modeling_df, fast_forecaster_params,
                                    tmp_path):
        clf_p, reg_p = fast_forecaster_params
        fc = EnrollmentForecaster(clf_params=clf_p, reg_params=reg_p)
        fc.fit(modeling_df)
        path = tmp_path / "forecaster.joblib"
        fc.save(path)
        fc2 = EnrollmentForecaster.load(path)
        assert fc2._is_fitted is True
