# DecisionLENS — Claude Code / VS Code AI Generation Prompt

> **HOW TO USE:** Copy everything below the line into Claude Code (or Cursor / Copilot chat in VS Code). Run it as a single prompt. The AI will scaffold and build the entire project. For Claude Code, paste the whole thing as your initial instruction. You may need to run it in phases (Module by Module) if the context window fills up — each module is self-contained for that reason.

---

## MASTER PROMPT — START HERE

You are building a complete, production-quality data science portfolio project called **DecisionLENS** — an AI-augmented clinical trial enrollment forecasting and decision support tool. This project demonstrates skills for a Data Scientist role in Clinical Operations at a major pharmaceutical company.

### CRITICAL CONTEXT

- The developer has a PhD in biophysics/bioinformatics and 4 years of data science experience outside pharma.
- This project must demonstrate: clinical trial domain knowledge, predictive modeling, interactive dashboards, generative AI integration, and SQL proficiency.
- All data comes from the **AACT database** (Aggregate Analysis of ClinicalTrials.gov) — a free PostgreSQL database from CTTI. For this project, we will download and use CSV/parquet flat-file snapshots instead of connecting to the live database. Download instructions: https://aact.ctti-clinicaltrials.org/pipe_files
- The project should be fully runnable locally with open-source tools.

---

## PROJECT STRUCTURE

Create the following directory structure and populate every file with complete, working code:

```
trialpulse/
├── README.md
├── requirements.txt
├── .env.example
├── setup_data.py                      # Script to download/prepare AACT data
├── sql/
│   ├── enrollment_extract.sql         # Core enrollment dataset extraction
│   ├── competitor_landscape.sql       # Competing trials query
│   └── investigator_sites.sql         # Site/investigator performance query
├── notebooks/
│   ├── 01_eda_trial_landscape.ipynb
│   ├── 02_feature_engineering.ipynb
│   ├── 03_enrollment_model.ipynb
│   ├── 04_competitive_intelligence.ipynb
│   └── 05_model_evaluation.ipynb
├── src/
│   ├── __init__.py
│   ├── data_pipeline.py               # Data loading, cleaning, feature engineering
│   ├── models.py                      # Enrollment prediction models
│   ├── competitive_intel.py           # Competitive landscape analysis
│   ├── investigator_insights.py       # Site/investigator analysis
│   └── genai_utils.py                 # LLM API integration for eligibility analysis
├── app/
│   ├── streamlit_app.py               # Main Streamlit entry point
│   ├── pages/
│   │   ├── 1_Enrollment_Forecast.py
│   │   ├── 2_Competitive_Intelligence.py
│   │   ├── 3_Investigator_Insights.py
│   │   └── 4_Eligibility_Analyzer.py
│   └── components/
│       ├── charts.py                  # Reusable Plotly chart functions
│       └── sidebar.py                 # Shared sidebar filters
├── tableau/
│   └── README.md                      # Instructions for Tableau Public companion dashboard
├── tests/
│   ├── test_pipeline.py
│   ├── test_models.py
│   └── test_genai.py
├── docs/
│   └── methodology.md                 # Analysis methodology documentation
└── .gitignore
```

---

## REQUIREMENTS.TXT

```
pandas>=2.0
numpy>=1.24
scikit-learn>=1.3
xgboost>=2.0
lifelines>=0.28
plotly>=5.18
streamlit>=1.30
anthropic>=0.40
python-dotenv>=1.0
requests>=2.31
pyarrow>=14.0
scipy>=1.11
shap>=0.44
pytest>=7.4
```

---

## MODULE 1: DATA PIPELINE & SQL (src/data_pipeline.py + sql/)

### SQL Scripts

Write 3 SQL scripts that would run against the AACT PostgreSQL schema. These are reference scripts showing SQL proficiency — the actual Python pipeline will work from downloaded flat files.

**sql/enrollment_extract.sql:**
- Query the `studies`, `calculated_values`, `eligibilities`, `designs`, `facilities`, `countries`, `sponsors`, `outcomes` tables.
- Extract: nct_id, study_type, phase, overall_status, enrollment (target), actual_enrollment, start_date, completion_date, number_of_facilities, number_of_countries, sponsor type (industry vs other), number of eligibility criteria, condition, intervention type, has_dmc, is_fda_regulated.
- Use CTEs and window functions (e.g., rank sponsors by historical enrollment performance).
- Filter to interventional studies from 2008 onward.

**sql/competitor_landscape.sql:**
- Given a condition and therapeutic area, find all recruiting/active trials.
- Calculate: number of competing trials, total competing enrollment targets, geographic overlap, temporal overlap.
- Use window functions and date range comparisons.

**sql/investigator_sites.sql:**
- Extract facility-level enrollment performance by country and investigator.
- Rank sites by historical trial completion rates.
- Use CTEs to aggregate across multiple trials per site.

### Python Data Pipeline (src/data_pipeline.py)

Build a comprehensive data pipeline class `TrialDataPipeline`:

```python
class TrialDataPipeline:
    """
    Pipeline for loading, cleaning, and preparing AACT clinical trial data.
    Works from downloaded CSV/parquet flat files.
    """
```

**Methods to implement:**

1. `load_raw_data(data_dir: str) -> dict[str, pd.DataFrame]` — Load all relevant AACT tables (studies, calculated_values, eligibilities, designs, facilities, countries, sponsors, conditions, interventions, outcome_counts). Handle missing files gracefully.

2. `clean_studies(df: pd.DataFrame) -> pd.DataFrame` — Filter to interventional studies, parse dates, remove withdrawn/unknown status, handle missing enrollment values, convert phases to ordinal encoding.

3. `engineer_features(tables: dict) -> pd.DataFrame` — Create the modeling dataset by joining tables and engineering these features:
   - `n_eligibility_criteria`: count of eligibility criteria lines (proxy for study complexity)
   - `n_facilities`: number of unique sites
   - `n_countries`: number of unique countries
   - `is_multicountry`: boolean flag
   - `sponsor_type`: industry vs. academic vs. government
   - `phase_numeric`: ordinal encoding of phase (1, 1.5, 2, 2.5, 3, 4)
   - `condition_prevalence_proxy`: frequency of condition in historical trials
   - `sponsor_historical_performance`: median enrollment ratio for this sponsor historically
   - `has_dmc`: whether data monitoring committee exists
   - `intervention_type`: drug, biological, device, procedure, etc.
   - `enrollment_ratio`: actual_enrollment / target_enrollment (target variable)
   - `enrollment_duration_days`: completion_date - start_date (target variable for survival model)
   - `enrollment_met_target`: binary flag (enrollment_ratio >= 0.9)
   - `competing_trials_count`: number of other active trials for same condition at same time
   - `geographic_concentration`: HHI index of site distribution across countries

4. `get_competitor_data(condition: str, start_date, end_date) -> pd.DataFrame` — Filter trials competing for same patient population in overlapping time window.

5. `get_site_performance(country: str = None) -> pd.DataFrame` — Aggregate site-level enrollment performance metrics.

6. `save_processed_data(output_dir: str)` and `load_processed_data(input_dir: str)` — Persist cleaned data.

**IMPORTANT:** Include thorough docstrings, type hints, and logging throughout. Code quality matters — this is a portfolio piece.

---

## MODULE 2: PREDICTIVE MODELS (src/models.py)

Build an `EnrollmentForecaster` class with two model approaches:

### Model A: Gradient Boosted Classifier + Regressor (XGBoost)

1. **Binary classifier**: Predicts `enrollment_met_target` (will trial meet ≥90% of target enrollment?). Use XGBoost with stratified k-fold cross-validation. Tune hyperparameters with Optuna or GridSearch.

2. **Regressor**: Predicts `enrollment_duration_days`. Use XGBoost regressor.

3. **Feature importance**: Use SHAP values to explain predictions. Build a method `explain_prediction(trial_features: pd.Series) -> dict` that returns SHAP waterfall data.

### Model B: Survival Analysis (Lifelines)

1. Use `lifelines.CoxPHFitter` to model time-to-enrollment-completion.
2. Treat trials still recruiting as right-censored observations.
3. Generate survival curves showing probability of completing enrollment over time.
4. Compare Kaplan-Meier curves across phases, therapeutic areas, and sponsor types.
5. Method: `predict_enrollment_curve(trial_features) -> pd.DataFrame` returning time points and cumulative enrollment probability.

### Evaluation Methods

- `evaluate_models() -> dict` — Return accuracy, AUC-ROC, precision, recall for classifier; MAE, RMSE, R² for regressor; concordance index for survival model.
- `cross_validate(n_splits=5) -> dict` — Stratified cross-validation results with confidence intervals.
- `plot_evaluation_results() -> plotly.Figure` — Visual model comparison.

### Key Implementation Notes
- Handle class imbalance with SMOTE or class weights.
- All methods should return interpretable outputs, not just raw numbers.
- Include a `predict(trial_params: dict) -> dict` method that takes user-friendly inputs (phase, condition, n_sites, n_countries, etc.) and returns a structured prediction result with confidence intervals.

---

## MODULE 3: COMPETITIVE INTELLIGENCE (src/competitive_intel.py)

Build a `CompetitiveAnalyzer` class:

1. `get_landscape(condition: str, therapeutic_area: str = None) -> dict` — Return:
   - Total competing trials (recruiting, active, not yet recruiting)
   - Enrollment competition intensity (sum of all target enrollments)
   - Geographic distribution (which countries have most competing sites)
   - Temporal density (how many trials started in last 6/12/24 months)
   - Top sponsors in this space

2. `plot_competition_map(condition: str) -> plotly.Figure` — Choropleth world map showing number of competing trial sites per country. Use `plotly.express.choropleth`.

3. `plot_competition_timeline(condition: str) -> plotly.Figure` — Gantt-style chart showing competing trials over time.

4. `calculate_recruitment_saturation(condition: str, country: str) -> float` — Estimate how saturated a country is for patient recruitment in this condition based on historical and current trial volume.

---

## MODULE 4: INVESTIGATOR & SITE INSIGHTS (src/investigator_insights.py)

Build an `InvestigatorAnalyzer` class:

1. `get_top_sites(condition: str, n: int = 20) -> pd.DataFrame` — Rank facilities by historical trial volume and completion rates for a given condition.

2. `get_country_performance(condition: str = None) -> pd.DataFrame` — Country-level metrics: avg trials per year, avg enrollment performance, avg time to completion.

3. `plot_site_network(condition: str) -> plotly.Figure` — Network graph (using plotly) showing relationships between sites that frequently participate in the same trials.

4. `recommend_sites(condition: str, target_enrollment: int, n_countries: int) -> pd.DataFrame` — Given study parameters, recommend optimal site distribution across countries based on historical performance. Output should include recommended country allocation and rationale.

---

## MODULE 5: GENERATIVE AI MODULE (src/genai_utils.py)

Build an `EligibilityAnalyzer` class using the Anthropic API (Claude):

```python
import anthropic
from dotenv import load_dotenv

class EligibilityAnalyzer:
    """
    Uses Claude API to analyze clinical trial eligibility criteria
    for enrollment risk factors and generate stakeholder briefings.
    """
```

### Methods:

1. `analyze_criteria(criteria_text: str) -> dict` — Send eligibility criteria to Claude API with a structured prompt. Return:
   - `risk_factors`: list of criteria likely to restrict enrollment
   - `severity_scores`: estimated impact of each restrictive criterion (high/medium/low)
   - `simplification_suggestions`: plain-English suggestions for broadening criteria
   - `estimated_population_impact`: qualitative estimate of how each criterion narrows eligible pool

   **Prompt engineering approach:** Use a system prompt that instructs Claude to act as a clinical trial enrollment optimization expert. Request structured JSON output. Include few-shot examples in the prompt of criteria analysis.

2. `generate_executive_briefing(prediction_results: dict, competition_data: dict, site_data: dict) -> str` — Take all model outputs and generate a 1-page executive summary for a Clinical Operations stakeholder. The briefing should cover: enrollment forecast, key risks, competitive landscape summary, and recommended actions.

   **Prompt engineering approach:** Use a system prompt with the persona of a senior clinical operations analyst writing for a VP. Specify format: 4 sections (Forecast Summary, Risk Assessment, Competitive Context, Recommendations), each 2-3 sentences max.

3. `compare_criteria(criteria_a: str, criteria_b: str) -> dict` — Compare two sets of eligibility criteria and highlight key differences that could affect enrollment.

### Implementation Notes:
- Use `anthropic` Python SDK.
- Store API key in `.env` file (provide `.env.example` with placeholder).
- Include retry logic and error handling for API calls.
- Cache API responses to avoid redundant calls during development.
- **Fallback mode**: If no API key is set, return mock/placeholder responses so the app still runs for demo purposes. This is critical — reviewers may not have an API key.

---

## MODULE 6: STREAMLIT DASHBOARD (app/)

### Main App (app/streamlit_app.py)

```python
"""
DecisionLENS — Clinical Trial Enrollment Forecasting & Decision Support
"""
```

- Set page config: wide layout, "DecisionLENS" title, clinical-themed favicon.
- Landing page with: project overview, key metrics cards (total trials analyzed, model accuracy, avg prediction confidence), and navigation guide.
- Use `st.sidebar` for global filters (date range, phase, therapeutic area).

### Page 1: Enrollment Forecast (app/pages/1_Enrollment_Forecast.py)

This is the core Study Decision Support page.

**Layout:**
- **Left panel**: Input form where user configures a hypothetical trial:
  - Phase (dropdown: I, I/II, II, II/III, III, IV)
  - Condition/therapeutic area (searchable dropdown from data)
  - Target enrollment (number input)
  - Number of sites (slider: 1-500)
  - Number of countries (slider: 1-50)
  - Sponsor type (radio: Industry, Academic, Government)
  - Has DMC (checkbox)
  - Intervention type (dropdown: Drug, Biological, Device, Procedure)
  - Number of eligibility criteria (slider: 5-50)

- **Right panel**: Results display:
  - **Risk gauge**: Plotly gauge chart showing probability of meeting enrollment target (green/yellow/red zones).
  - **Enrollment timeline**: Plotly line chart showing predicted enrollment curve with 80% confidence band, compared against the target line. Use the survival model output.
  - **SHAP explanation**: Horizontal bar chart showing top 10 features driving this specific prediction (positive = helps enrollment, negative = hurts).
  - **Benchmark comparison**: Box plot showing how this trial's predicted duration compares to historical trials in same phase and condition.

- **Bottom section**: Historical context — table of 10 most similar historical trials (by features) with their actual enrollment outcomes.

### Page 2: Competitive Intelligence (app/pages/2_Competitive_Intelligence.py)

**Layout:**
- **Top**: Condition selector + date range filter.
- **Section 1**: KPI cards — total competing trials, total competing enrollment, top 3 countries by competition, competition trend (increasing/decreasing).
- **Section 2**: World choropleth map (Plotly) showing site density per country for this condition. Interactive hover with details.
- **Section 3**: Competition timeline — Gantt chart of all competing trials colored by phase and sponsor.
- **Section 4**: Sponsor landscape — treemap showing market share of different sponsors in this condition space.

### Page 3: Investigator & Site Insights (app/pages/3_Investigator_Insights.py)

**Layout:**
- **Top**: Condition filter + country filter.
- **Section 1**: Top-performing sites table with sortable columns (facility name, city, country, n_trials, avg completion rate, avg enrollment ratio).
- **Section 2**: Country performance heatmap — showing countries ranked by enrollment efficiency for this condition.
- **Section 3**: Site recommendation engine — user inputs target enrollment and desired number of countries, gets a recommended allocation.

### Page 4: Eligibility Analyzer (app/pages/4_Eligibility_Analyzer.py)

**Layout:**
- **Input area**: Large text box where user pastes eligibility criteria (inclusion + exclusion).
- **Analyze button**: Triggers Claude API call.
- **Results (after analysis)**:
  - Risk factor table with severity flags (color-coded).
  - Simplification suggestions as expandable cards.
  - Population impact assessment visualization.
  - "Generate Executive Briefing" button that combines this analysis with enrollment prediction into a downloadable summary.
- **Example button**: Pre-loads a sample set of eligibility criteria so users can test without typing.
- **Fallback**: If no API key configured, show a notice and display mock analysis results.

### Shared Components (app/components/)

**charts.py:** Reusable Plotly chart functions with consistent styling:
- Use a clinical/pharma-appropriate color palette (blues, teals, clean whites — not generic rainbow).
- All charts should have: clear titles, axis labels, hover tooltips, and consistent font.
- Include functions: `risk_gauge()`, `enrollment_curve()`, `shap_waterfall()`, `competition_map()`, `competition_timeline()`, `site_heatmap()`.

**sidebar.py:** Shared sidebar filter components that persist across pages.

### Streamlit Styling Notes:
- Professional, clean design. No default Streamlit rainbow styling.
- Add a custom CSS block in the main app to refine typography and spacing.
- Use `st.metric()` cards for KPIs.
- Use `st.columns()` for layout.
- Use `st.expander()` for detailed methodology notes on each page.

---

## MODULE 7: NOTEBOOKS (notebooks/)

### 01_eda_trial_landscape.ipynb
- Load AACT data.
- Explore distributions: trials by phase, condition, year, sponsor type, geography.
- Visualize enrollment success rates over time.
- Identify top conditions by trial volume.
- ~15-20 charts with Plotly, all with clear annotations and markdown explanations.

### 02_feature_engineering.ipynb
- Walk through each engineered feature with rationale.
- Show correlation matrix of features.
- Analyze feature distributions and handle outliers.
- Document decisions (why each feature matters for enrollment prediction).

### 03_enrollment_model.ipynb
- Train both XGBoost and survival models.
- Hyperparameter tuning with cross-validation.
- SHAP analysis with summary and dependence plots.
- Compare models: XGBoost vs Cox PH — which works better and why.
- Clear markdown cells explaining every step.

### 04_competitive_intelligence.ipynb
- Demonstrate the competitive analysis pipeline.
- Case study: pick 2-3 real conditions (e.g., "Non-Small Cell Lung Cancer", "Type 2 Diabetes", "Alzheimer's Disease") and show full competitive landscape analysis.
- Geographic visualizations.

### 05_model_evaluation.ipynb
- Comprehensive model evaluation: ROC curves, calibration plots, precision-recall curves.
- Cross-validation results with confidence intervals.
- Error analysis: where does the model fail and why?
- Final model selection rationale.

**Notebook standards:** Every notebook must have:
- Markdown header with purpose and key findings.
- Numbered sections.
- Clean, commented code cells.
- Visualization after every analysis step.
- Summary/takeaway at the end.

---

## MODULE 8: DOCUMENTATION

### README.md

Write a compelling, professional README with:

1. **Hero section**: Project name, one-line description, badges (Python, Streamlit, XGBoost).
2. **Problem Statement**: Frame in clinical operations language — "Clinical Operations teams need data-driven decision support to predict enrollment performance, assess competitive landscapes, and optimize site selection."
3. **Solution Overview**: What DecisionLENS does, with a screenshot placeholder.
4. **Key Features**: Bullet list of all 4 dashboard modules.
5. **Technical Architecture**: Mermaid diagram showing data flow (AACT → Pipeline → Models → Dashboard).
6. **Getting Started**: Step-by-step setup instructions (clone, install, download data, run).
7. **Data Source**: AACT database description with link and citation.
8. **Methodology**: Brief overview of modeling approach with link to docs/methodology.md.
9. **Project Structure**: Directory tree with descriptions.
10. **Results**: Key findings — model accuracy, top predictive features, example insights.
11. **Future Enhancements**: What would come next (real-time data feeds, integration with CTMS, etc.).
12. **About the Author**: Brief bio placeholder.

### docs/methodology.md

Detailed methodology document covering:
- Data selection and preprocessing rationale.
- Feature engineering decisions with clinical context (why each feature matters for enrollment).
- Model selection rationale (why XGBoost + survival analysis).
- Evaluation framework.
- Limitations and assumptions.
- References to clinical operations literature.

---

## SETUP SCRIPT (setup_data.py)

Create a setup script that:
1. Creates necessary directories (data/raw, data/processed, models/).
2. Downloads AACT pipe-delimited files (or instructs user how to).
3. Loads relevant tables into pandas and saves as parquet.
4. Runs the data pipeline to create the modeling dataset.
5. Prints summary statistics of the prepared data.

Include a fallback: if AACT download fails, generate a **realistic synthetic dataset** (10,000 trials) with proper distributions so the project still runs end-to-end for demo purposes. The synthetic data should mimic real AACT schema and value distributions.

---

## GLOBAL CODE QUALITY STANDARDS

Apply these standards to ALL files:

1. **Type hints** on all function signatures.
2. **Docstrings** (Google style) on all classes and public methods.
3. **Logging** using Python `logging` module (not print statements).
4. **Error handling**: try/except with meaningful error messages.
5. **Constants**: No magic numbers — use named constants or config.
6. **DRY**: Extract repeated logic into utility functions.
7. **Testing**: Basic pytest tests for pipeline and model methods.
8. **.gitignore**: Include standard Python gitignore + data/, models/, .env, __pycache__, .ipynb_checkpoints.

---

## EXECUTION ORDER

Build the project in this sequence:
1. Project structure + requirements.txt + .gitignore + .env.example
2. setup_data.py (with synthetic data fallback)
3. sql/ scripts (all 3)
4. src/data_pipeline.py
5. src/models.py
6. src/competitive_intel.py
7. src/investigator_insights.py
8. src/genai_utils.py
9. app/components/ (charts.py, sidebar.py)
10. app/streamlit_app.py + all 4 pages
11. notebooks/ (all 5, in order)
12. tests/
13. README.md + docs/methodology.md
14. tableau/README.md

After each module, verify it works by running relevant tests or a quick smoke test. Fix any import errors or data path issues before moving to the next module.

---

## END OF PROMPT
