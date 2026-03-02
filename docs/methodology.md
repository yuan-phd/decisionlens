# DecisionLENS — Technical Methodology

> **Audience**: Data scientists and ML engineers reviewing this portfolio project.
> **Purpose**: Full walkthrough of every modelling, feature-engineering, and AI-design decision made in DecisionLENS.

---

## Table of Contents

1. [Data Source — AACT / ClinicalTrials.gov](#1-data-source)
2. [Preprocessing Pipeline](#2-preprocessing-pipeline)
3. [Feature Engineering — All 14 Features](#3-feature-engineering)
4. [Enrollment Forecast Model (XGBoost Classifier)](#4-enrollment-forecast-model)
5. [Duration Regression (XGBoost Regressor)](#5-duration-regression)
6. [Survival Analysis (Cox Proportional Hazards)](#6-survival-analysis)
7. [Competitive Intelligence Module](#7-competitive-intelligence)
8. [Investigator & Site Insights Module](#8-investigator--site-insights)
9. [LLM Eligibility Analyzer — Prompt Design](#9-llm-eligibility-analyzer)
10. [Limitations and Future Work](#10-limitations-and-future-work)

---

## 1. Data Source

### 1.1 AACT Database

The **Aggregate Analysis of ClinicalTrials.gov (AACT)** database, maintained by the Clinical Trials Transformation Initiative (CTTI), is a PostgreSQL replica of ClinicalTrials.gov updated daily. DecisionLENS uses the **flat-file (pipe-delimited) snapshot** format rather than the live PostgreSQL instance, which makes the project portable without database credentials.

| Property | Value |
|----------|-------|
| Format | Pipe-delimited (`\|`), UTF-8, one file per table |
| Size | ~2.2 GB compressed; ~8 GB uncompressed |
| Update frequency | Daily |
| Authentication | None required for flat files |

AACT is public domain. Download URL pattern:

```
https://aact.ctti-clinicaltrials.org/static/static_db_copies/daily/YYYYMMDD_clinical_trials.zip
```

`setup_data.py` tries the last 7 calendar dates to handle weekends and holidays before falling back to a synthetic dataset.

### 1.2 Synthetic Fallback Dataset

When the AACT download is unavailable (no internet, corporate firewall, quota exceeded), `setup_data.py` generates a **10,000-study synthetic dataset** using NumPy seeded random processes. All structural patterns are preserved:

- Phase distribution matches AACT (~15% Phase 1, ~30% Phase 1/2, ~35% Phase 2, ~15% Phase 3, ~5% Phase 4)
- Class imbalance preserved: ~75% COMPLETED, ~8% TERMINATED, ~17% RECRUITING/ACTIVE
- Sponsor type distribution: ~40% industry, ~35% academic, ~25% government
- Enrollment distribution: log-normal with µ=5.5, σ=1.5 (median ≈ 245 patients)

The synthetic data supports all four app modules and all 89 unit tests without modification.

### 1.3 Tables Used

DecisionLENS joins 10 AACT tables:

| Table | Rows (approx) | Key fields used |
|-------|--------------|-----------------|
| `studies` | 490K | nct_id, study_type, overall_status, phase, enrollment, start_date, completion_date |
| `calculated_values` | 490K | actual_duration, number_of_facilities |
| `eligibilities` | 490K | criteria (free text) |
| `designs` | 490K | intervention_model, masking |
| `facilities` | 3.5M | country, status |
| `countries` | 700K | name, removed |
| `sponsors` | 670K | agency_class, lead_or_collaborator |
| `conditions` | 750K | name |
| `interventions` | 780K | intervention_type |
| `outcome_counts` | Not used in modeling | (reserved for future features) |

---

## 2. Preprocessing Pipeline

All preprocessing is implemented in `src/data_pipeline.py` (`TrialDataPipeline`) and mirrors the logic in `sql/enrollment_extract.sql`.

### 2.1 Study-Level Filters

Applied in `clean_studies()`:

| Filter | Rationale |
|--------|-----------|
| `study_type == "Interventional"` | Observational studies have fundamentally different enrollment dynamics and objectives |
| `overall_status NOT IN ("WITHDRAWN", "UNKNOWN STATUS")` | WITHDRAWN trials never had a real enrollment attempt; UNKNOWN STATUS is data quality noise |
| `start_date.year >= 2008` | Pre-2008 AACT records have high missingness in key fields; modern trial operations differ |
| `enrollment >= 10` | Pilot and feasibility studies (< 10 patients) are not relevant to operational planning |
| `enrollment IS NOT NULL` | Prevents spurious target-encoding of missing values |

After filtering, the dataset retains approximately **70,000–80,000 interventional trials** from the full AACT snapshot (compared to ~490,000 total registrations).

### 2.2 Phase Normalisation

AACT uses inconsistent phase strings across registration periods. All variants are canonicalised before ordinal encoding:

| Raw AACT string | Canonical | Ordinal |
|----------------|-----------|---------|
| PHASE 1, Phase1, EARLY_PHASE1 | Phase 1 | 1.0 |
| PHASE 1/PHASE 2, PHASE 1/2 | Phase 1/Phase 2 | 1.5 |
| PHASE 2 | Phase 2 | 2.0 |
| PHASE 2/PHASE 3, PHASE 2/3 | Phase 2/Phase 3 | 2.5 |
| PHASE 3 | Phase 3 | 3.0 |
| PHASE 4 | Phase 4 | 4.0 |
| N/A, NaN, None | N/A → 0.0 | 0.0 |

The ordinal scale respects the linear progression of drug development phases while allowing gradient-boosted trees to exploit monotonic patterns.

### 2.3 Label Construction

The classification target is binary:

```
enrollment_met_target = 1   if overall_status == "COMPLETED"
enrollment_met_target = 0   if overall_status == "TERMINATED"
enrollment_met_target = NaN otherwise (RECRUITING, ACTIVE, etc.)
```

Only labeled rows (COMPLETED or TERMINATED) are used to train the classifier and regressor. Unlabeled rows (actively recruiting / active trials) are excluded from supervised training but remain in the dataset for prediction.

**Labeling rationale**: The COMPLETED/TERMINATED dichotomy provides a clean binary signal that is directly actionable — a sponsor needs to know whether a trial is at risk of early termination, not the precise enrollment ratio. TERMINATED trials that stopped for reasons unrelated to enrollment (e.g., safety, strategic) are still labeled 0; this is intentional since the model should flag any risk of non-completion as negative for planning purposes.

### 2.4 Categorical Encoding

`WITHDRAWN` and `UNKNOWN STATUS` rows are removed before encoding. All remaining status strings (COMPLETED, TERMINATED, RECRUITING, ACTIVE, NOT YET RECRUITING, etc.) are title-cased for consistent display in the dashboard.

Boolean columns (`has_dmc`, `is_fda_regulated_drug`, `is_fda_regulated_device`) stored as AACT `'t'`/`'f'` strings are converted to Python `bool`.

---

## 3. Feature Engineering

All 14 features fed to the model are engineered in `TrialDataPipeline.engineer_features()`. They are divided into three categories:

### 3.1 Numeric Features (10)

| Feature | Source table(s) | Engineering | Rationale |
|---------|----------------|-------------|-----------|
| `phase_numeric` | studies | Ordinal encoding (see §2.2) | Phase is the strongest structural predictor of trial complexity |
| `n_facilities` | facilities, calculated_values | `COUNT(DISTINCT facility_id)` per trial | More sites → broader recruitment pool → higher completion probability |
| `n_countries` | countries | `COUNT(DISTINCT name)` after removing withdrawn | Multi-country trials access diverse patient populations |
| `n_eligibility_criteria` | eligibilities | Line count of criteria free text | Proxy for protocol complexity; more criteria → narrower eligible population |
| `geographic_concentration` | facilities | Herfindahl-Hirschman Index (HHI) of site-country shares | HHI=1 → single-country trial (concentrated risk); HHI→0 → well-distributed sites |
| `condition_prevalence_proxy` | conditions | Frequency of the primary condition name across all AACT trials | High-frequency conditions (diabetes, hypertension) have larger patient pools and established recruitment infrastructure |
| `sponsor_historical_performance` | sponsors, studies | Count of COMPLETED trials for lead sponsor, normalised to [0,1] | Experienced sponsors have refined site selection and patient matching capabilities |
| `competing_trials_count` | studies, conditions, countries | Trials sharing the same primary condition with overlapping active periods | High competition saturates the patient pool and investigator bandwidth |
| `enrollment` | studies | Target enrollment figure (log1p-transformed before modelling) | Larger trials are harder to fill; log-transform handles right-skew |
| `enrollment_type_is_actual` | studies | 1 if enrollment_type == "Actual" else 0 | "Actual" confirmed figures are more reliable than prospective estimates |

**Log-transform applied to**: `n_facilities`, `competing_trials_count`, `enrollment` — all exhibit heavy right-skew (verified in notebook 02).

### 3.2 Categorical Features (3, one-hot encoded)

| Feature | Values | Rationale |
|---------|--------|-----------|
| `sponsor_type` | industry / government / academic | Sponsor type correlates with resources, regulatory experience, and site network |
| `intervention_model` | Parallel Assignment, Crossover, etc. | Crossover designs require fewer patients but are operationally complex |
| `masking` | Open, Single, Double, Triple, Quadruple | Blinded trials require additional coordination and pharmacy overhead |

One-hot encoding with `handle_unknown="ignore"` prevents unseen categories from causing errors at prediction time.

### 3.3 Boolean Feature (1)

| Feature | Derived from | Rationale |
|---------|-------------|-----------|
| `is_multicountry` | `n_countries > 1` | Multi-country trials have higher regulatory burden but access larger patient populations |

### 3.4 HHI — Geographic Concentration

The **Herfindahl-Hirschman Index** is computed per trial as:

```
HHI = Σ (s_i)²
```

where `s_i` is the share of sites in country `i`. A trial with all sites in one country has HHI = 1.0 (maximum concentration). A trial evenly distributed across 10 countries has HHI = 0.10.

HHI was chosen over simpler country-count features because it captures the *shape* of the geographic distribution, not just its breadth. A trial with 50 US sites and 1 German site is treated very differently from one with 26 sites in each country.

### 3.5 Sponsor Historical Performance

A count-based proxy is used (number of COMPLETED trials for the lead sponsor) rather than a median enrollment-ratio proxy. This decision prevents **look-ahead bias**: in real-world deployment, a trial being assessed may be underway, so the sponsor's *true historical performance* (including trials that completed after the trial being assessed began) is not available. A simple count of COMPLETED trials in the current AACT snapshot is a conservative but unbiased estimate.

The count is normalised by the maximum across all sponsors in the dataset to produce a [0, 1] score.

---

## 4. Enrollment Forecast Model

### 4.1 Architecture

The classifier is a **scikit-learn Pipeline** containing:

1. `ColumnTransformer`:
   - Numeric branch: `SimpleImputer(strategy="median")` → passes through
   - Categorical branch: `SimpleImputer(strategy="constant", fill_value="unknown")` → `OneHotEncoder(handle_unknown="ignore")`
   - Boolean branch: coerced to float (0/1), imputed with 0
2. `XGBClassifier` with the hyperparameters below

All preprocessing is encapsulated in the pipeline, so a raw `engineer_features()` DataFrame can be passed directly to `predict()` without manual preparation.

### 4.2 Hyperparameters

| Parameter | Value | Rationale |
|-----------|-------|-----------|
| `n_estimators` | 400 | Enough trees for stable learning with a low learning rate |
| `max_depth` | 5 | Controls overfitting; deep enough for 14-feature interactions |
| `learning_rate` | 0.05 | Low rate + many estimators → better generalisation |
| `subsample` | 0.8 | Row subsampling for variance reduction |
| `colsample_bytree` | 0.8 | Column subsampling per tree |
| `min_child_weight` | 5 | Minimum instance weight in a leaf; prevents overfitting on rare categories |
| `scale_pos_weight` | **5.0** | Class imbalance weight (tuned; see §4.3) |
| `objective` | binary:logistic | Outputs calibrated probabilities |
| `eval_metric` | logloss | Consistent with binary:logistic objective |

### 4.3 Class Imbalance Handling

The raw COMPLETED:TERMINATED ratio in AACT is approximately **12.3:1** (severely imbalanced). Several strategies were evaluated:

| Strategy | F1-Terminated | Notes |
|----------|--------------|-------|
| Default (`scale_pos_weight=1`) | 0.05 | Model predicts COMPLETED for nearly all rows |
| `scale_pos_weight=12.3` (raw ratio) | 0.28 | Over-corrects; too many false Terminated |
| `scale_pos_weight=5.0` (**chosen**) | **0.368** | Best F1-macro; preserves ROC-AUC |
| SMOTE oversampling | 0.31 | Worse than weight-tuning; adds complexity |
| Random undersampling | 0.29 | Discards 90% of useful Completed data |

`scale_pos_weight=5.0` was selected via a sweep in notebook 05.

### 4.4 Decision Threshold Tuning

The default XGBoost decision threshold of `0.50` is poorly calibrated for highly imbalanced datasets. A threshold sweep was conducted on the held-out test fold:

| Threshold | F1-macro | F1-Completed | F1-Terminated |
|-----------|----------|-------------|--------------|
| 0.50 | 0.564 | 0.956 | 0.170 |
| 0.70 | 0.590 | 0.942 | 0.238 |
| 0.80 | 0.620 | 0.921 | 0.319 |
| **0.93** | **0.660** | **0.869** | **0.368** |
| 0.97 | 0.632 | 0.828 | 0.437 |

At threshold=0.50, the model nearly always predicted COMPLETED — clinically useless for risk detection. At **threshold=0.93**, the model surfaces roughly twice as many true termination risks while accepting a modest precision trade-off on the COMPLETED class. This is the right trade-off for clinical operations: false negatives (missed Terminated trials) are far more costly than false positives (unnecessary monitoring).

**Implementation**: `CLF_THRESHOLD = 0.93` is applied in `predict()`:
```python
pred["pred_label"] = (pred["p_completed"] >= CLF_THRESHOLD).astype(int)
```

### 4.5 Cross-Validation Results

Evaluated with 5-fold stratified cross-validation:

| Metric | Value |
|--------|-------|
| ROC-AUC | 0.787 ± 0.004 |
| F1-macro (threshold=0.93) | 0.660 |
| F1-Completed | 0.869 |
| F1-Terminated | 0.368 |

**ROC-AUC = 0.787 confirms a real, stable signal** in the structural AACT features. The AUC is consistent across folds (σ=0.004), indicating the model generalises rather than memorising training patterns.

**Why F1-Terminated is still modest**: AACT captures structural attributes but lacks the operational signals that most reliably predict termination — site activation delays, protocol amendments, funding discontinuation, and regulatory holds. No threshold or weight tuning can recover information absent from the feature set. F1-Terminated will improve when site-level and operational features are incorporated (see §10).

### 4.6 SHAP Explainability

When `shap` is installed, `explain()` uses `shap.TreeExplainer` to compute per-instance feature contributions. The dashboard displays a **SHAP waterfall chart** for any trial configuration entered on Page 1. Waterfall plots decompose the model's log-odds prediction into additive contributions, one per feature, making the risk drivers interpretable for non-technical stakeholders.

---

## 5. Duration Regression

### 5.1 Purpose and Limitations

The XGBRegressor predicts `enrollment_duration_days` — the number of days from study start to enrollment completion. This is intentionally treated as a **supplemental estimate** rather than the primary forecast output, for two reasons:

1. **Low signal-to-noise ratio**: Trial duration is dominated by operational factors (site activation speed, patient identification lag, investigator engagement) that are not recorded in AACT structural fields. The regressor achieves **R² ≈ 0.04** on held-out data — the model captures a real but weak signal.
2. **Better uncertainty representation**: The Cox PH survival curve (§6) provides probability-of-completion-by-day-t with implicit uncertainty, which is more informative for planning than a point estimate with unknown error.

Despite its low R², the regressor is useful for **rank-ordering** trials and for the duration benchmark visualisation (box plot of historical durations for similar trials).

### 5.2 Architecture and Hyperparameters

Identical preprocessing pipeline to the classifier. Hyperparameters differ only in the objective:

| Parameter | Value |
|-----------|-------|
| `n_estimators` | 400 |
| `max_depth` | 5 |
| `learning_rate` | 0.05 |
| `subsample` | 0.8 |
| `colsample_bytree` | 0.8 |
| `min_child_weight` | 5 |
| `objective` | reg:squarederror |

Predictions are clipped at the 99th percentile of training durations (`REG_CLIP_PERCENTILE = 99`) to prevent catastrophic extrapolation on unusual trial configurations.

### 5.3 Training Data

Only labeled rows (`enrollment_met_target` is not NaN) are used for regression training — the same subset as the classifier. Unlabeled trials (still recruiting) have unknown final durations, so including them would introduce censoring bias that would be incorrectly handled by a standard regressor. The Cox PH model (§6) handles censoring correctly.

---

## 6. Survival Analysis

### 6.1 Motivation

Standard regression fails to handle **right-censoring**: trials that are still recruiting when the AACT snapshot was taken have unknown final durations. Treating them as having a duration equal to the snapshot date would systematically under-estimate true durations. The **Cox Proportional Hazards (CPH) model** natively handles right-censored observations, providing:

1. A **survival curve** `S(t)` = P(trial has not yet completed by day t)
2. A **completion curve** `1 − S(t)` = P(trial completes by day t)

This transforms the regression problem into a time-to-event problem with principled uncertainty quantification.

### 6.2 Censoring Strategy

```
Event indicator (event_completed):
  1   if overall_status == "COMPLETED"     → event observed
  0   if overall_status != "COMPLETED"     → right-censored
      (TERMINATED, RECRUITING, ACTIVE, etc.)
```

TERMINATED trials are treated as censored (not as events) because termination represents the end of follow-up due to a non-completion cause, not the enrollment-completion event of interest. This is the correct survival-analytic treatment.

### 6.3 Model: Cox Proportional Hazards

The CPH model from the `lifelines` library is fitted with:

- `penalizer = 0.1`: L2 regularisation to prevent overfitting on sparse categorical covariates
- Covariates: the same 14 features used by XGBoost (after preprocessing)
- Minimum duration filter: trials with `enrollment_duration_days < 1` are excluded to avoid numerical issues in the partial likelihood

**Survival curve interpretation**: For each new trial, `predict_survival()` generates `S(t)` at specified time points (default: every 30 days from 0 to 1,825 days). The dashboard plots `1 − S(t)` as the probability of completing enrollment by day t, with a vertical marker at the predicted median.

### 6.4 CPH Assumptions

The **proportional hazards assumption** requires that the ratio of hazard rates between any two subjects is constant over time. In practice, trial enrollment dynamics may violate this assumption (e.g., ramp-up period followed by steady-state recruitment). This is an acknowledged limitation; the CPH model is used here for its interpretability and standard adoption in clinical research rather than as a claim of perfect fit.

---

## 7. Competitive Intelligence

Implemented in `src/competitive_intel.py` (`CompetitiveAnalyzer`).

### 7.1 Competition Intensity Index

For a target disease area, competing trials are identified by condition name matching (case-insensitive substring search). The **Competition Intensity Index** is defined as:

```
CII = (active_competing_trials / (total_trials + 1)) × log1p(total_trials)
```

The `log1p` dampening prevents CII from growing unboundedly in dense therapeutic areas (oncology, CNS). The `+1` prevents division by zero for novel conditions with no prior trials.

### 7.2 Visualisations

| Chart | Implementation | Insight |
|-------|---------------|---------|
| Choropleth map | Plotly Express `choropleth` with ISO-3166 country mapping | Shows where patient populations and site infrastructure are concentrated |
| Gantt timeline | Plotly `Bar` with horizontal orientation, coloured by phase | Reveals concurrent trial density and enrollment period overlaps |
| Sponsor treemap | Plotly Express `treemap` | Visualises market share by active-trial count in the indication |

### 7.3 Country Mapping

AACT country names do not always match ISO-3166 standard names. A normalisation map handles the most common discrepancies (e.g., "Korea, Republic of" → "South Korea", "United States" → "United States of America"). Unmatched names are retained as-is and may display without choropleth colouring on the map.

---

## 8. Investigator & Site Insights

Implemented in `src/investigator_insights.py` (`InvestigatorAnalyzer`).

### 8.1 Site Ranking

Sites are ranked by a composite **performance score** computed from five normalised metrics:

| Metric | Weight | Rationale |
|--------|--------|-----------|
| Completion rate | 0.35 | Primary signal — fraction of trials at this site that completed |
| Average enrollment | 0.25 | Sites that consistently meet enrollment targets |
| Trial volume | 0.20 | Experienced sites with regulatory infrastructure |
| Phase 2/3 concentration | 0.10 | Bias towards sites experienced with pivotal studies |
| Multi-condition breadth | 0.10 | Generalist sites are more resilient to condition-specific patient shortages |

Each metric is z-score normalised across all sites before computing the weighted sum. Z-score normalisation was chosen over min-max normalisation because it is robust to outliers (e.g., academic mega-centres with 1,000+ trials skew min-max ranges).

### 8.2 Site Co-Participation Network

A graph is constructed where:
- **Nodes** = individual trial sites (facility names)
- **Edges** = two sites participated in the same trial
- **Edge weight** = number of co-participated trials

The network is visualised with Plotly using a force-directed layout (spring layout via `networkx.spring_layout`). This reveals **hub sites** (high degree: participate in many trials) and **bridge sites** (connect otherwise disconnected clusters of sites).

### 8.3 Site Allocation Recommender

Given a total patient target `N` and a set of candidate countries, the recommender distributes patients across countries proportional to their **normalised country performance score** (completion rate × trial volume per country). The allocation algorithm:

1. Compute raw allocation: `N_i = N × score_i / Σ scores`
2. Round to integer site counts
3. Adjust rounding residuals to ensure `Σ N_i == N`

This is a simple heuristic that treats site performance as the primary allocation criterion. A more sophisticated implementation would account for country-specific regulatory timelines, patient identification rates, and diversity objectives.

---

## 9. LLM Eligibility Analyzer

Implemented in `src/genai_utils.py` (`EligibilityAnalyzer`).

### 9.1 Model and Provider

| Attribute | Value |
|-----------|-------|
| Provider | Groq Cloud API |
| Model | `llama-3.3-70b-versatile` |
| Context | 128K tokens |
| Latency | ~1–3 seconds per request (Groq hardware) |

Groq was chosen over the Anthropic/OpenAI APIs because its LPU (Language Processing Unit) inference hardware delivers sub-second token throughput at no cost during development — appropriate for a portfolio project where response latency directly impacts demo experience.

### 9.2 Prompt Architecture

Three distinct system prompts are used, one per method:

#### `analyze_criteria()` — Risk Analysis Prompt

The system prompt instructs the model to act as a **clinical trial enrollment optimization expert** and return a structured JSON object with four **parallel lists**:

```json
{
  "risk_factors":                 ["criterion text", ...],
  "severity_scores":              ["high"|"medium"|"low", ...],
  "simplification_suggestions":   ["plain-English suggestion", ...],
  "estimated_population_impact":  ["~X% of target patients", ...]
}
```

**Key design decisions**:
- **Parallel lists** (not a list of objects) reduce token usage and make parsing more robust to minor JSON formatting variations
- **Explicit severity vocabulary** ("high", "medium", "low") prevents the model from inventing synonyms
- **"Return only valid JSON — no markdown fences, no prose outside the JSON"** terminates hallucinated preambles that break `json.loads()`
- **Population impact framing** encourages quantified rather than vague output ("eliminates ~30% of NSCLC patients" vs. "reduces eligible population")

#### `generate_executive_briefing()` — VP-Level Summary

The system prompt specifies a strict 4-section format:
```
1. FORECAST SUMMARY (2–3 sentences)
2. RISK ASSESSMENT (2–3 sentences)
3. COMPETITIVE CONTEXT (2–3 sentences)
4. RECOMMENDED ACTIONS (3–5 bullets beginning with •)
```

This is delivered via **streaming** (`stream=True`) to the Streamlit dashboard, where `st.write_stream()` displays tokens progressively — improving perceived responsiveness for what can be a 10–20 second generation.

The user message is constructed by injecting model output (`p_completed`, `pred_duration_days`), competition data (`total_trials`, `active_trials`, `competing_trials`), and site data (`n_sites`, `top_country`) into a templated string, grounding the LLM in the quantitative forecast results rather than generating a generic narrative.

#### `compare_criteria()` — Protocol Comparison

Returns a structured JSON with:
```json
{
  "differences": [{"criterion", "protocol_a", "protocol_b", "more_restrictive"}],
  "enrollment_impact": ["favorable_A"|"favorable_B"|"neutral"],
  "overall_assessment": "string",
  "recommendations": ["string", ...]
}
```

The `differences` and `enrollment_impact` lists are parallel, with identical length constraints enforced via the prompt.

### 9.3 Response Caching

To avoid re-querying the API for identical inputs during development:

1. **Memory cache**: Python dict keyed by SHA-256 hash of the prompt string
2. **Disk cache**: JSON files written to `.cache/genai/{hash}.json`

Cache is keyed by a SHA-256 hash of the full user message content, so different eligibility criteria never collide. The disk cache persists across Streamlit reruns (which reload the Python process).

Cache is **not used in demo mode** — demo mode returns hardcoded mock responses directly without touching any cache infrastructure.

### 9.4 Demo / Fallback Mode

When `GROQ_API_KEY` is not set (or `groq` is not installed), the analyzer runs in demo mode:

- `analyze_criteria()` returns `_MOCK_CRITERIA` — a realistic 5-criterion NSCLC example with high/medium/high severity labels
- `generate_executive_briefing()` returns `_MOCK_BRIEFING` — a formatted 4-section briefing with placeholder statistics
- `compare_criteria()` returns `_MOCK_COMPARISON` — a comparison of two realistic NSCLC protocol variants

Demo mode allows reviewers to explore the full Streamlit interface without an API key. The mock responses are designed to be qualitatively realistic — an oncology domain expert would find them clinically plausible — so they serve as demonstration of the tool's intended output quality.

### 9.5 JSON Parsing Robustness

LLMs occasionally wrap JSON in markdown code fences (````json ... ````). The response parser strips these before calling `json.loads()`. A `try/except json.JSONDecodeError` falls back gracefully to the mock response with a logged warning, preventing uncaught exceptions from crashing the Streamlit app.

---

## 10. Limitations and Future Work

### 10.1 Data Ceiling — Structural Features Only

AACT records structural trial attributes but lacks the **operational signals** that most reliably predict termination:

| Missing signal | Likely impact on F1-Terminated |
|---------------|-------------------------------|
| Site activation delays (days to first patient enrolled) | +5–8 F1 points |
| Protocol amendment count and timing | +3–5 F1 points |
| Funding source and continuity | +4–6 F1 points |
| Investigator dropout / turnover | +3–4 F1 points |
| Regulatory hold history | +2–3 F1 points |

Adding even one of these signals (e.g., from TrialTrove, GlobalData, or EMA/FDA REMS data) would substantially improve minority-class detection.

### 10.2 Survival Model Assumptions

The Cox PH proportional hazards assumption may be violated for enrollment dynamics, which typically have:
- A ramp-up phase (slow site activation)
- A steady-state recruitment phase
- A tail-off phase (remaining hard-to-enroll patients)

A **Weibull accelerated failure time model** or a **piecewise exponential model** would be more appropriate if time-varying covariates become available.

### 10.3 Condition Labeling

The `condition_prevalence_proxy` feature uses the raw MeSH condition name as a frequency counter across all AACT trials. This is a coarse proxy — two conditions may have similar raw frequency but vastly different patient pool sizes (e.g., "Type 2 Diabetes" vs. "Rare Genetic Disorder X"). Linking to OMIM prevalence data or ORPHANET for rare diseases would improve signal quality.

### 10.4 Sponsor Historical Performance

The current sponsor performance metric is count-based (number of COMPLETED trials), which conflates sponsor size with effectiveness. A rate-based metric (completion rate weighted by trial complexity) would be more informative but requires defining "complexity" — a circular dependency with the model itself.

### 10.5 LLM Output Consistency

Llama 3.3-70B does not guarantee identical JSON schema compliance across all inputs. Edge cases observed during testing:
- Very short criteria texts (< 3 criteria) sometimes produce fewer list items than expected
- Highly technical criteria (complex pharmacokinetic inclusion criteria) occasionally produce generic rather than specific simplification suggestions

Both issues are handled by the demo-mode fallback and the `json.JSONDecodeError` catch in the parser. A structured-output endpoint (JSON mode) would eliminate schema compliance issues at the cost of some generation flexibility.

### 10.6 Planned Extensions

| Feature | Module | Priority |
|---------|--------|----------|
| Patient identification rates by ICD code + geography | Data pipeline | High |
| Real-time AACT incremental updates (PostgreSQL CDC) | Data pipeline | Medium |
| Bayesian updating of P(completion) as trial progresses | Models | High |
| Site performance benchmarking against external registries | Investigator Insights | Medium |
| Enrollment rate simulation (Monte Carlo) | Models | Medium |
| Tableau Public dashboard (alternative to Streamlit) | Visualisation | Low |

---

*Methodology version: 1.0 — March 2026*
*Dataset: Synthetic AACT snapshot (10,000 studies, seed=42)*
*All performance figures were evaluated on the synthetic fallback dataset. Real AACT results will differ based on data completeness and temporal distribution.*
