# DecisionLENS

**AI-augmented clinical trial enrollment forecasting and decision support.**

DecisionLENS is an end-to-end data science portfolio project that applies predictive modeling, survival analysis, and large-language-model reasoning to the problem of clinical trial enrollment planning — one of the costliest and most schedule-critical challenges in drug development.

> Built with: Python · XGBoost · Lifelines · Groq (Llama 3.3-70B) · Streamlit · Plotly · AACT / ClinicalTrials.gov data

---

## What it does

| Module | Question answered |
|--------|-------------------|
| **Enrollment Forecast** | What is the probability this trial will complete enrollment? When? |
| **Competitive Intelligence** | How saturated is the patient pool for this condition? Who are the competing sponsors? |
| **Investigator Insights** | Which sites and countries have the strongest historical completion records? |
| **Eligibility Analyzer** | Which inclusion/exclusion criteria most restrict the eligible population, and how can they be relaxed? |

---

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                      Data Layer                                  │
│  AACT (ClinicalTrials.gov) flat-file parquets                   │
│  setup_data.py ──► data/processed/*.parquet                     │
│  (falls back to 10,000-study synthetic dataset if unavailable)  │
└────────────┬────────────────────────────────────────────────────┘
             │
┌────────────▼────────────────────────────────────────────────────┐
│                   src/ — Core Modules                            │
│                                                                  │
│  data_pipeline.py      TrialDataPipeline                        │
│    load_raw_data()  ──►  clean_studies()  ──►  engineer_features│
│                                                                  │
│  models.py             EnrollmentForecaster                      │
│    XGBClassifier (P_completed)                                   │
│    XGBRegressor  (duration_days)                                 │
│    CoxPHFitter   (survival curve S(t))                           │
│    TreeExplainer (SHAP waterfall)                                │
│                                                                  │
│  competitive_intel.py  CompetitiveAnalyzer                       │
│    get_landscape()  plot_competition_map()  plot_timeline()      │
│                                                                  │
│  investigator_insights.py  InvestigatorAnalyzer                  │
│    get_top_sites()  get_country_performance()  recommend_sites() │
│                                                                  │
│  genai_utils.py        EligibilityAnalyzer (Groq / Llama 3.3)   │
│    analyze_criteria()  compare_criteria()  exec_briefing()       │
└────────────┬────────────────────────────────────────────────────┘
             │
┌────────────▼────────────────────────────────────────────────────┐
│                  app/ — Streamlit Dashboard                      │
│                                                                  │
│  streamlit_app.py         Home / KPI overview                   │
│  pages/1_Enrollment_Forecast.py    Risk gauge, survival curve   │
│  pages/2_Competitive_Intelligence.py  Choropleth, Gantt, treemap│
│  pages/3_Investigator_Insights.py  Site ranking, country heatmap│
│  pages/4_Eligibility_Analyzer.py   LLM risk analysis, briefing  │
└─────────────────────────────────────────────────────────────────┘
```

---

## Project Structure

```
decisionlens/
├── app/
│   ├── streamlit_app.py              # Home page
│   ├── components/
│   │   ├── charts.py                 # Reusable Plotly chart functions
│   │   ├── sidebar.py                # Global filter sidebar
│   │   └── _theme.py                 # Shared CSS theme
│   └── pages/
│       ├── 1_Enrollment_Forecast.py
│       ├── 2_Competitive_Intelligence.py
│       ├── 3_Investigator_Insights.py
│       └── 4_Eligibility_Analyzer.py
├── src/
│   ├── data_pipeline.py              # TrialDataPipeline
│   ├── models.py                     # EnrollmentForecaster
│   ├── competitive_intel.py          # CompetitiveAnalyzer
│   ├── investigator_insights.py      # InvestigatorAnalyzer
│   └── genai_utils.py                # EligibilityAnalyzer (Groq API)
├── notebooks/
│   ├── 01_eda_trial_landscape.ipynb  # Exploratory data analysis
│   ├── 02_feature_engineering.ipynb  # Feature construction & validation
│   ├── 03_enrollment_model.ipynb     # Model training & tuning
│   ├── 04_competitive_intelligence.ipynb
│   └── 05_model_evaluation.ipynb     # Threshold tuning, SHAP analysis
├── sql/
│   ├── enrollment_extract.sql        # Main modeling query
│   ├── competitor_landscape.sql      # Competing-trial query
│   └── investigator_sites.sql        # Site-performance query
├── tests/
│   ├── conftest.py                   # Shared fixtures
│   ├── test_pipeline.py
│   ├── test_models.py
│   └── test_genai.py
├── docs/
│   └── methodology.md               # Full technical methodology
├── models/
│   └── forecaster.joblib            # Saved EnrollmentForecaster
├── setup_data.py                    # One-time data download & preprocessing
├── requirements.txt
└── .env.example
```

---

## Quickstart

### 1 — Clone and install

```bash
git clone https://github.com/yourhandle/decisionlens.git
cd decisionlens

python -m venv venv
source venv/bin/activate          # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

### 2 — Configure environment

```bash
cp .env.example .env
```

Edit `.env` and set:

```
# Required for the Eligibility Analyzer (Page 4).
# Free key: https://console.groq.com/keys
# If not set, the app runs in demo mode with realistic mock responses.
GROQ_API_KEY=your_groq_key_here

# Path to your AACT data directory (created by setup_data.py).
# Defaults to ./data if not set.
AACT_DATA_DIR=./data
```

### 3 — Set up data

`setup_data.py` downloads the latest AACT flat-file snapshot (~2 GB) and converts it to parquets. If the download fails, it generates a 10,000-study synthetic dataset automatically.

```bash
python setup_data.py
```

Expected output:
```
Trying 2026-03-01 … OK
Downloaded: data/raw/20260301_clinical_trials.zip
Extracting … done
Converting to parquet … 10 tables
Preprocessing … done
Saved to data/processed/ (10 tables)
```

### 4 — Train the model

Run notebook **03** to train and save the `EnrollmentForecaster`:

```bash
jupyter notebook notebooks/03_enrollment_model.ipynb
# Or: jupyter nbconvert --to notebook --execute notebooks/03_enrollment_model.ipynb
```

This writes `models/forecaster.joblib`. A pre-trained model is included for convenience.

### 5 — Launch the dashboard

```bash
streamlit run app/streamlit_app.py
```

Opens at `http://localhost:8501`. All four pages are available from the sidebar.

---

## Model Performance

Performance was evaluated on a 10,000-study synthetic AACT snapshot using 5-fold stratified cross-validation (March 2026).

### XGBoost Classifier — P(Enrollment Completes)

| Metric | Default threshold (0.50) | Tuned threshold (0.93) |
|--------|--------------------------|------------------------|
| ROC-AUC | 0.787 ± 0.004 | 0.787 ± 0.004 |
| F1-macro | 0.564 | **0.660** |
| F1 (Completed) | 0.956 | 0.869 |
| F1 (Terminated) | 0.170 | **0.368** |

The threshold was raised from 0.50 to **0.93** to improve sensitivity to the Terminated (minority) class while preserving ROC-AUC. At the 0.50 threshold, the model nearly always predicted "Completed" — clinically useless. At 0.93, it surfaces roughly twice as many true termination risks.

**Why F1-Terminated is still modest:** AACT captures structural attributes (phase, sponsor, facility count) but lacks the operational signals that most reliably predict termination — site activation delays, protocol amendments, funding discontinuation, and regulatory holds. ROC-AUC = 0.787 confirms that a real signal exists; F1-Terminated will improve as site-level features are added.

### XGBoost Regressor — Enrollment Duration

| Metric | Value |
|--------|-------|
| R² | ~0.04 |
| Interpretation | Duration is driven by operational factors absent from AACT structural data. |
| Use | Rank-ordering and rough planning; prefer Cox PH intervals for uncertainty. |

### Cox Proportional Hazards — Survival Curve S(t)

Fits a parametric time-to-completion model on labeled trials with right-censoring for active/recruiting studies. Used to generate the enrollment-completion probability curve `P(completed by day t) = 1 − S(t)` shown on Page 1.

---

## Key Features

### Enrollment Forecast (Page 1)

- Configure a hypothetical trial (phase, sites, countries, sponsor type, masking)
- Outputs: completion probability gauge, predicted duration in months, risk label
- Survival curve: P(enrollment completed by day t) with predicted median marker
- SHAP waterfall: top 12 risk drivers with signed feature contributions
- Benchmark: box plot of historical duration for same phase + 10 nearest-neighbor trials

### Competitive Intelligence (Page 2)

- KPI cards: total/active/competing/terminated trials, competition intensity index
- Choropleth world map: trial site density by country
- Gantt timeline: concurrent trial start/end dates coloured by phase
- Sponsor treemap: market share by active-trial count

### Investigator Insights (Page 3)

- Sortable site ranking table: trials, completion rate, avg enrollment, city/country
- Country heatmap: z-score normalised performance across 5 metrics
- Site co-participation network: nodes = sites, edges = shared trials
- Site allocation recommender: distributes a patient target across countries by performance score

### Eligibility Analyzer (Page 4) — AI-Powered

- Pastes eligibility criteria text (or loads an NSCLC example)
- LLM (Llama 3.3-70B via Groq) identifies risk factors, assigns severity (high/medium/low), and suggests relaxations with estimated population impact
- Streaming executive briefing: 1-page VP-level summary (~300 words)
- Side-by-side protocol comparison: which criteria differ, which protocol is more enrollment-friendly
- **Demo mode**: runs with realistic mock responses when `GROQ_API_KEY` is not set

---

## Notebooks

| Notebook | Purpose |
|----------|---------|
| [01_eda_trial_landscape.ipynb](notebooks/01_eda_trial_landscape.ipynb) | Distribution of trial statuses, phases, enrollment figures; temporal trends |
| [02_feature_engineering.ipynb](notebooks/02_feature_engineering.ipynb) | Feature construction, correlation analysis, missing-value audit |
| [03_enrollment_model.ipynb](notebooks/03_enrollment_model.ipynb) | XGBoost training, cross-validation, SHAP importances, model serialisation |
| [04_competitive_intelligence.ipynb](notebooks/04_competitive_intelligence.ipynb) | Competitive landscape deep-dive, sponsor analysis, site co-participation network |
| [05_model_evaluation.ipynb](notebooks/05_model_evaluation.ipynb) | Threshold sweep (F1 vs threshold curve), ROC/PR curves, calibration, confusion matrices |

---

## Tests

```bash
pytest tests/ -v
# 89 passed, 7 skipped (live-mode Groq tests require GROQ_API_KEY) in ~5 s
```

| Test file | What's tested |
|-----------|---------------|
| `test_pipeline.py` | `TrialDataPipeline` — init, clean_studies, engineer_features, load_raw_data |
| `test_models.py` | `EnrollmentForecaster` — fit, predict, survival, save/load, pre-fit guards |
| `test_genai.py` | `EligibilityAnalyzer` — demo mode, all three methods, caching, live mocks |

---

## SQL Scripts

The `sql/` directory contains equivalent PostgreSQL queries for teams with live AACT access:

| Script | Purpose |
|--------|---------|
| `enrollment_extract.sql` | Full modeling dataset — joins 8 AACT tables, engineers all features |
| `competitor_landscape.sql` | Active competing trials, sponsor market share, site distribution |
| `investigator_sites.sql` | Site completion rates, enrollment performance by country |

---

## Stack

| Layer | Technology |
|-------|-----------|
| Data | [AACT / ClinicalTrials.gov](https://aact.ctti-clinicaltrials.org/) (public) |
| Data wrangling | pandas, pyarrow |
| Modeling | scikit-learn, XGBoost, lifelines (Cox PH) |
| Explainability | SHAP (TreeExplainer) |
| Generative AI | [Groq API](https://console.groq.com/) — `llama-3.3-70b-versatile` |
| Visualization | Plotly Express / Graph Objects |
| Dashboard | Streamlit |
| Testing | pytest |

---

## Configuration Reference

All settings are loaded from `.env` at startup (see `.env.example`):

| Variable | Default | Description |
|----------|---------|-------------|
| `GROQ_API_KEY` | _(none)_ | Groq API key for the Eligibility Analyzer. Get free at [console.groq.com/keys](https://console.groq.com/keys). App runs in demo mode without it. |
| `AACT_DATA_DIR` | `./data` | Root directory for AACT flat files. `setup_data.py` creates `raw/` and `processed/` subdirectories inside it. |
| `AACT_USERNAME` | _(none)_ | AACT PostgreSQL credentials — needed only for live database queries, not for flat-file mode. |
| `AACT_PASSWORD` | _(none)_ | See above. |

---

## Methodology

See [docs/methodology.md](docs/methodology.md) for a detailed walkthrough of:
- AACT data schema and preprocessing decisions
- Feature engineering rationale (all 14 features)
- Classifier threshold selection and imbalanced-class handling
- Survival model right-censoring and interpretation
- LLM prompt design and structured output schema

---

## Data Notes

- **Source**: [AACT flat-file downloads](https://aact.ctti-clinicaltrials.org/pipe_files) — pipe-delimited (`|`), UTF-8, ~2.2 GB compressed
- **Scope**: Interventional trials started ≥ 2008 with target enrollment ≥ 10
- **Labels**: COMPLETED → 1, TERMINATED → 0; RECRUITING/ACTIVE/etc. → unlabeled
- **Synthetic fallback**: `setup_data.py` generates 10,000 seeded synthetic studies when the download is unavailable; all structural patterns are preserved

---

## License

MIT — see `LICENSE` for details.
