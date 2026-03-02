DECISIONLENS — CORRECT BUILD ORDER

PHASE 1: FOUNDATION (done ✅)
Step 1 ✅  Project structure + requirements.txt + .env + .gitignore
Step 2 ✅  setup_data.py (synthetic data fallback)
Step 3 ✅  sql/ scripts (enrollment_extract, competitor_landscape, investigator_sites)
Step 4 ✅  src/data_pipeline.py

PHASE 2: UNDERSTAND THE DATA (do this next)
Step 5    notebooks/01_eda_trial_landscape.ipynb
          → Load synthetic data, 15-20 Plotly charts,
            distributions of phase/condition/year/sponsor,
            enrollment success rates, geography
Step 6    notebooks/02_feature_engineering.ipynb
          → Walk through each engineered feature,
            correlation matrix, outlier analysis,
            document why each feature matters clinically

PHASE 3: MODELLING
Step 7    src/models.py
          → EnrollmentForecaster class,
            XGBoost classifier + regressor, survival analysis
Step 8    notebooks/03_enrollment_model.ipynb
          → Train models, hyperparameter tuning,
            SHAP analysis, XGBoost vs Cox PH comparison

PHASE 4: INTELLIGENCE MODULES
Step 9    src/competitive_intel.py
Step 10   src/investigator_insights.py
Step 11   notebooks/04_competitive_intelligence.ipynb
          → Case studies: lung cancer, diabetes, Alzheimer's
Step 12   notebooks/05_model_evaluation.ipynb
          → ROC curves, calibration, error analysis

PHASE 5: GENERATIVE AI
Step 13   src/genai_utils.py
          → Claude API integration, prompt engineering,
            eligibility criteria analyzer

PHASE 6: DASHBOARD
Step 14   app/components/charts.py + sidebar.py
Step 15   app/streamlit_app.py (main entry point)
Step 16   app/pages/1_Enrollment_Forecast.py
Step 17   app/pages/2_Competitive_Intelligence.py
Step 18   app/pages/3_Investigator_Insights.py
Step 19   app/pages/4_Eligibility_Analyzer.py

PHASE 7: QUALITY & DOCUMENTATION
Step 20   tests/ (test_pipeline.py, test_models.py, test_genai.py)
Step 21   README.md (with DecisionLENS name, not TrialPulse)
Step 22   docs/methodology.md
Step 23   tableau/README.md