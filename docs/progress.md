# DecisionLENS — Build Progress

**Last updated:** 2026-02-28
**Build plan:** `decisionlens_steps.md`
**Full spec:** `decisionlens_prompt.md`

---

## Current Status

```
PHASE 1: FOUNDATION       ✅ Complete
PHASE 2: UNDERSTAND DATA  🔄 In progress (Step 5 done, Step 6 next)
PHASE 3: MODELLING        ⏳ Pending
PHASE 4: INTELLIGENCE     ⏳ Pending
PHASE 5: GENERATIVE AI    ⏳ Pending
PHASE 6: DASHBOARD        ⏳ Pending
PHASE 7: QUALITY & DOCS   ⏳ Pending
```

---

## PHASE 1: FOUNDATION ✅

### Step 1 — Project structure + config files

| File | Purpose |
|------|---------|
| `README.md` | Project overview, one-paragraph description |
| `requirements.txt` | All Python dependencies (pandas, xgboost, lifelines, streamlit, anthropic, shap, …) |
| `.env` | Local secrets — ANTHROPIC_API_KEY, AACT credentials, AACT_DATA_DIR=./data |
| `.env.example` | Placeholder version safe to commit |
| `.gitignore` | Excludes .env, data/, models/, __pycache__, .ipynb_checkpoints |
| `src/__init__.py` | Makes src/ a Python package |

**Directories created:** `sql/`, `notebooks/`, `app/`, `src/`, `docs/`

---

### Step 2 — `setup_data.py`

Handles all data acquisition and preparation in one script.

**How it works:**
1. Looks for any `*.zip` in `data/raw/` (most-recently-modified wins if multiple)
2. Prints the first 5 filenames from the zip for verification before extracting
3. Extracts 10 required AACT tables as pipe-delimited `.txt` files into `data/raw/`
4. Reads each with `sep='|'` and saves as parquet to `data/processed/`
5. Falls back to generating a realistic **10,000-study synthetic dataset** if no zip found

**CLI:**
```bash
python3 setup_data.py                    # auto-detect zip, fallback to synthetic
python3 setup_data.py --force-synthetic  # synthetic only (no zip needed)
python3 setup_data.py --data-dir ./data  # explicit root path
```

**Data directory layout (correct):**
```
data/
├── raw/       ← place AACT zip here; .txt files extracted here
└── processed/ ← parquet files written here (one per table)
```

**Env var:** `AACT_DATA_DIR=./data` (root dir, NOT data/raw)

**Parquet files currently in `data/processed/` — real AACT data:**

| Table | Size |
|-------|------|
| studies.parquet | 115 MB |
| eligibilities.parquet | 424 MB |
| interventions.parquet | 104 MB |
| facilities.parquet | 100 MB |
| conditions.parquet | 31 MB |
| sponsors.parquet | 22 MB |
| designs.parquet | 22 MB |
| outcome_counts.parquet | 21 MB |
| calculated_values.parquet | 10 MB |
| countries.parquet | 8 MB |

---

### Step 3 — SQL scripts (`sql/`)

Three reference scripts for the AACT PostgreSQL schema (`ctgov` schema).
These demonstrate SQL proficiency — the Python pipeline works from parquet flat files.

**`sql/enrollment_extract.sql`** (303 lines)
- 11 CTEs covering studies, calculated_values, eligibilities, designs, facilities, countries, sponsors, conditions, interventions
- Window functions: sponsor rank by historical volume, phase ordinal encoding
- Derived targets: `enrollment_met_target` (binary), `enrollment_ratio` (continuous)
- Filter: interventional studies, start_date >= 2008

**`sql/competitor_landscape.sql`** (258 lines)
- Parameterized via psql `\set` variables (condition, date window)
- Self-join on date overlap to count concurrent competitors
- 4 result sets: per-trial detail, sponsor market share, temporal density, phase distribution
- Window functions: `RANK() OVER`, running cumulative total

**`sql/investigator_sites.sql`** (296 lines)
- Facility-level performance aggregates: completion rate, enrollment ratio, composite score
- Country-level aggregates with enrollment efficiency metric
- Site co-participation network edges (sites sharing ≥2 trials)
- 3 result sets: site performance, country performance, network edges

---

### Step 4 — `src/data_pipeline.py` (721 lines)

**Class: `TrialDataPipeline`**

Replicates the SQL logic on parquet flat files via pandas.

| Method | Description |
|--------|-------------|
| `load_raw_data(data_dir)` | Loads all 10 parquet tables; missing files skipped gracefully |
| `clean_studies(df)` | Interventional filter, date parsing, phase ordinal encoding, `t`/`f` → bool |
| `engineer_features(tables)` | Full join + all feature engineering (see below) |
| `get_competitor_data(condition, start, end)` | Condition substring match + date window overlap filter |
| `get_site_performance(country=None)` | Facility-level aggregation with composite performance score |
| `save_processed_data(output_dir)` | Saves `modeling_df.parquet` |
| `load_processed_data(input_dir)` | Loads saved modeling DataFrame |

**Engineered features:**

| Feature | Method |
|---------|--------|
| `phase_numeric` | Ordinal encoding: 1.0, 1.5, 2.0, 2.5, 3.0, 4.0 |
| `n_facilities` | Facility table count; falls back to calculated_values |
| `n_countries` | Unique countries per study |
| `is_multicountry` | n_countries > 1 |
| `geographic_concentration` | HHI index of country site-share distribution |
| `sponsor_type` | industry / government / academic |
| `sponsor_historical_performance` | Normalised completed-trial count per sponsor |
| `condition_prevalence_proxy` | Frequency of condition across all trials |
| `intervention_type` | First-listed intervention type |
| `n_eligibility_criteria` | Newline-split count of free-text criteria |
| `enrollment_ratio` | actual / target (capped at 3.0) |
| `enrollment_duration_days` | completion_date − start_date |
| `enrollment_met_target` | enrollment_ratio ≥ 0.90 (classification label) |
| `competing_trials_count` | Vectorised self-join: same condition + overlapping dates |

---

## PHASE 2: UNDERSTAND THE DATA 🔄

### Step 5 ✅ — `notebooks/01_eda_trial_landscape.ipynb`

20 Plotly charts across 10 sections. Uses real AACT data via `TrialDataPipeline`.
Auto-runs `setup_data.py --force-synthetic` if processed data not yet present.

| # | Chart type | What it shows |
|---|-----------|---------------|
| 1 | Bar | Trial starts per year 2008–2024 |
| 2 | Donut | Trial status distribution |
| 3 | Bar | Phase distribution |
| 4 | Box | Target enrollment by phase |
| 5 | Heatmap | Phase × intervention model |
| 6 | H-bar | Top 20 conditions by trial count |
| 7 | H-bar | Top 15 conditions by total enrollment target |
| 8 | Donut | Sponsor type split |
| 9 | Bar | Enrollment success rate by sponsor type |
| 10 | Histogram | Enrollment ratio distribution (90% threshold marked) |
| 11 | Bar + CI | Success rate by phase with 95% confidence intervals |
| 12 | Dual-axis | Success rate over time + trial volume |
| 13 | H-bar | Top 20 countries by site count |
| 14 | Choropleth | World map of site density |
| 15 | Bar | Multi-country vs single-country success rate |
| 16 | Violin | Duration distribution by phase |
| 17 | Scatter | Target enrollment vs duration |
| 18 | H-bar | Intervention type distribution |
| 19 | H-bar | Success rate by intervention type |
| 20 | Bubble | Site count vs enrollment ratio |

**Colour palette:** Clinical blues (`#1a6fa8`, `#8ecae6`, `#2a9d8f`) — no rainbow defaults.

---

### Step 6 ⏳ — `notebooks/02_feature_engineering.ipynb` — NEXT

---

## What's Left (in order)

```
Step 6   notebooks/02_feature_engineering.ipynb
Step 7   src/models.py              (EnrollmentForecaster — XGBoost + Cox PH)
Step 8   notebooks/03_enrollment_model.ipynb
Step 9   src/competitive_intel.py   (CompetitiveAnalyzer)
Step 10  src/investigator_insights.py (InvestigatorAnalyzer)
Step 11  notebooks/04_competitive_intelligence.ipynb
Step 12  notebooks/05_model_evaluation.ipynb
Step 13  src/genai_utils.py          (EligibilityAnalyzer — Claude API)
Step 14  app/components/charts.py + sidebar.py
Step 15  app/streamlit_app.py
Step 16  app/pages/1_Enrollment_Forecast.py
Step 17  app/pages/2_Competitive_Intelligence.py
Step 18  app/pages/3_Investigator_Insights.py
Step 19  app/pages/4_Eligibility_Analyzer.py
Step 20  tests/
Step 21  README.md (full version with badges, architecture, results)
Step 22  docs/methodology.md
Step 23  tableau/README.md
```

---

## Key Technical Decisions Made

| Decision | Rationale |
|----------|-----------|
| Parquet over CSV for processed data | ~10× faster reads; preserves dtypes |
| Pandas pipeline mirrors SQL scripts | Portfolio shows both SQL and Python skills |
| HHI for geographic concentration | Standard economics metric; interpretable for pharma audience |
| Vectorised self-join for competing_trials_count | Avoids slow per-row loop on 500k+ trial dataset |
| Sponsor historical performance normalised to [0,1] | Avoids leakage from raw trial counts; scale-invariant |
| `AACT_DATA_DIR=./data` (root, not raw subdir) | setup_data.py appends raw/ and processed/ itself |
| Any *.zip accepted in data/raw/ | AACT download gives random filenames |

---

## Environment Notes

- Python 3.11 (pyenv), virtualenv at `venv/`
- Real AACT data loaded: Feb 2026 snapshot (~857 MB total parquet)
- `ANTHROPIC_API_KEY` not yet set — genai_utils.py will use fallback/mock mode until set
- Run from repo root: `python3 setup_data.py`, `streamlit run app/streamlit_app.py`
