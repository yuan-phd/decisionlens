# CLAUDE.md — DecisionLENS v2
## Agentic Data Quality Agent for Clinical Trials

---

## Project Context

This is **DecisionLENS v2** — an agentic data quality system built as a demo for **Rivia** (rivia.com), a $15M Series A clinical trial intelligence platform. Rivia's CEO described their key product bet as *"an agentic data quality product that continuously surfaces, prioritises, and manages the thousands of data issues that arise during a trial."* This project mirrors that product.

**Critical framing — say this everywhere:**
> "This demo uses AACT registry data as a proxy for EDC/CTMS data. The same agent architecture applies to live trial execution data — check logic changes, orchestration and audit trail remain identical."

This framing must appear as a persistent banner on every Streamlit page.

---

## Repository Structure

```
/Users/ye/work/portfolio/decisionlens/
├── agents/
│   ├── orchestrator.py      # LangGraph agent — 5 nodes + loop
│   └── state.py             # DataQualityState TypedDict
├── checks/
│   ├── models.py            # Issue dataclass — shared return type
│   ├── temporal.py          # Check A: date consistency
│   ├── enrollment.py        # Check B: enrollment anomalies
│   ├── status.py            # Check C: status inconsistencies
│   ├── endpoints.py         # Check D: endpoint & design gaps
│   └── crossfield.py        # Check E: cross-field validation
├── models/
│   ├── xgb_classifier.pkl   # v1 XGBoost model (AUC=0.787, 573k trials)
│   └── risk_scorer.py       # Wrapper: loads v1 model, returns risk score
├── llmops/
│   └── tracker.py           # LLMTracker: logs every LLM call
├── evaluation/
│   ├── gold_set.py          # 20 manually labelled issues
│   └── metrics.py           # Precision/recall/F1/Cohen's kappa
├── output/
│   └── report.py            # Structured JSON output + provenance
├── src/                     # v1 modules — DO NOT MODIFY
│   ├── data_pipeline.py
│   ├── models.py
│   ├── competitive_intel.py
│   ├── investigator_insights.py
│   └── genai_utils.py
├── data/
│   └── processed/           # AACT parquet files (10 tables, 3.4M records)
├── app_v2.py                # Streamlit dashboard — 4 pages
└── README_v2.md
```

---

## Technology Stack

- **Python 3.14** (M1 Max Mac)
- **LangGraph** — agent orchestration (if incompatible, use raw ReAct loop)
- **XGBoost** — v1 completion risk model already trained
- **OpenAI GPT-4o-mini** — LLM prioritisation (Groq as fallback)
- **Pandas + PyArrow** — AACT data access via parquet files
- **Streamlit + Plotly** — dashboard
- **python-dotenv** — key management

Always run `from dotenv import load_dotenv; load_dotenv()` at the top of every file that needs API keys.

---

## Data Sources

AACT parquet files at `data/processed/`:
- `studies.parquet` — primary trial table (nct_id, overall_status, phase, enrollment, start_date, completion_date, last_update_posted)
- `eligibilities.parquet` — eligibility criteria text
- `facilities.parquet` — investigator sites
- `conditions.parquet` — trial conditions/indications
- `sponsors.parquet` — sponsor information
- `designs.parquet` — trial design (randomization, masking, arms)
- `outcome_counts.parquet` — reported results
- `calculated_values.parquet` — derived AACT fields
- `interventions.parquet` — intervention details
- `countries.parquet` — geographic distribution

Always load via pandas: `pd.read_parquet('data/processed/studies.parquet')`

---

## v1 Risk Model

The v1 XGBoost classifier is already trained and saved at `models/xgb_classifier.pkl`.

- Trained on 573,000 real AACT trials
- AUC = 0.787, 5-fold CV stable ±0.004
- Predicts completion probability (higher = more likely to complete)
- **Completion risk score = 1 - completion_probability**
- Features: enrollment, n_facilities, phase_numeric, sponsor_historical_performance, competing_trials_count, intervention_model, masking, is_multicountry, geographic_concentration, condition_prevalence_proxy

The feature pipeline is in `src/data_pipeline.py` — use `TrialDataPipeline.engineer_features()` to extract features for any trial.

**Escalation logic:**
- HIGH issue + risk_score > 0.6 → upgrade to CRITICAL
- MEDIUM issue + risk_score > 0.7 → upgrade to HIGH

---

## Core Data Models

### Issue (checks/models.py)
```python
@dataclass
class Issue:
    trial_id: str
    check_name: str
    check_category: str      # temporal/enrollment/status/endpoint/crossfield
    severity_rule: str       # HIGH/MEDIUM/LOW (rule-based, before LLM)
    finding: str             # factual, no interpretation
    data_points: dict        # actual values that triggered the flag
    source_tables: list[str]
    source_columns: list[str]
    timestamp: str           # ISO format
```

### PrioritisedIssue (output/report.py)
```python
@dataclass
class PrioritisedIssue:
    # From rule engine
    trial_id: str
    check_name: str
    check_category: str
    finding: str
    data_points: dict

    # From v1 risk model
    completion_risk_score: float
    risk_model_version: str      # "xgboost_v1_auc0787_573k_trials"

    # From LLM prioritisation
    severity: str                # CRITICAL/HIGH/MEDIUM/LOW
    explanation: str             # 2-3 sentences, no jargon
    potential_impact: str
    suggested_action: str
    confidence: float            # 0.0-1.0
    related_trials: list[str]

    # Provenance — always present, never optional
    provenance: dict             # see provenance spec below
```

### Provenance spec (every issue must have this):
```python
provenance = {
    "source_tables": list[str],
    "source_columns": list[str],
    "check_rule": str,
    "data_accessed_at": str,        # ISO timestamp
    "context_enriched": bool,
    "enrichment_source": str,
    "agent_iteration": int,
    "llm_call_ids": list[str],      # links to LLMOps log
    "risk_model_used": bool,
    "risk_model_version": str
}
```

---

## Agent Architecture (agents/)

### State (agents/state.py)
```python
class DataQualityState(TypedDict):
    scope: str
    scope_filters: dict
    plan: list[str]
    checks_completed: list[str]
    issues_found: list[Issue]
    needs_deeper_investigation: list[str]
    context_results: list[dict]
    iteration: int                   # max 3
    risk_scores: dict[str, float]
    llm_calls: list[dict]
    total_tokens: int
    total_latency_ms: float
    estimated_cost: float
    prioritised_issues: list[dict]
    pattern_library_updates: list[dict]
    final_report: dict
```

### Agent Flow (agents/orchestrator.py)
5 nodes with conditional edges:

1. **PLANNING NODE** — LLM decides which checks to run and in what order based on scope. Not hardcoded. Output: ordered list of check names.

2. **SCAN NODE** — Executes each planned check function as a tool call against AACT. Attaches v1 risk score to each issue. Output: list of Issue objects with risk scores.

3. **REVIEW NODE** — LLM reviews all issues. Decides if any need deeper investigation. Output: list of trial IDs needing context OR signal to proceed to prioritisation.

4. **CONTEXT ENRICHMENT NODE** — For flagged trials: fetch eligibility text, query related trials (same sponsor/condition), check cross-trial patterns. Loops back to REVIEW (max 3 iterations total).

5. **PRIORITISATION NODE** — LLM generates final PrioritisedIssue for each finding using all accumulated context + risk scores. Updates pattern library.

**This is genuinely agentic because:**
- Planning node decides checks (not hardcoded)
- Review node reasons about what needs investigation
- Context node fetches different data per issue type
- Loop allows multi-step reasoning (max 3x)
- Prioritisation uses full accumulated context

---

## LLMOps (llmops/tracker.py)

Every LLM call in the agent MUST go through LLMTracker. No exceptions.

```python
class LLMTracker:
    def track_call(self, model, purpose, prompt_tokens,
                   completion_tokens, latency_ms) -> dict:
        # purpose values: "planning" / "review" / "context" / "prioritisation"
        # Returns call record with uuid, timestamp, cost estimate
    
    def get_summary(self) -> dict:
        # Returns: total_calls, total_tokens, total_latency_ms,
        #          total_estimated_cost_usd, calls list
```

Cost estimation:
- GPT-4o-mini: $0.15/1M input tokens, $0.60/1M output tokens
- Display total scan cost in dashboard

---

## LLM Prioritisation Prompt

Use this exact prompt structure in the prioritisation node:

```
You are a clinical trial data quality analyst reviewing automated 
findings from the AACT clinical trials registry.

Trial: {trial_id}
Trial metadata: {trial_metadata}
Findings from automated checks: {findings}
Completion risk score from predictive model: {risk_score}/1.0
  (trained on 573,000 real trials, AUC=0.787 — higher = higher risk)
Additional context: {context}
Related trial patterns: {related_patterns}

For each finding provide:
1. severity: CRITICAL/HIGH/MEDIUM/LOW
   CRITICAL = HIGH finding + completion risk >0.6
   HIGH = data integrity, unreported safety events, regulatory risk
   MEDIUM = timeline delays, process gaps
   LOW = minor inconsistency

2. explanation: 2-3 sentences for a clinical ops manager. No jargon.

3. potential_impact: What goes wrong if ignored?

4. suggested_action: Specific next step.

5. confidence: 0.0-1.0

Respond in JSON array. No markdown, no preamble.
```

---

## Streamlit Dashboard (app_v2.py)

4 pages. AACT framing banner on EVERY page — no exceptions:
```
⚠️ Data source: AACT registry (proxy for EDC/CTMS). 
Same agent architecture applies to live trial execution data.
```

### Page 1: Scan Control & Overview
- Scope selectors: therapeutic area, phase, date range
- "Run Scan" button → show agent progress
- Summary cards: trials scanned, CRITICAL/HIGH/MEDIUM/LOW counts
- LLMOps card: calls, tokens, cost, latency

### Page 2: Prioritised Issues List
- Table sorted by severity → confidence
- Columns: Rank, Trial ID, Severity (color badge), Check, Explanation preview, Risk Score, Confidence
- CRITICAL rows: red banner
- Expandable rows: full explanation, data points, action, provenance
- Filter by severity, check category
- Export JSON button

### Page 3: Trial Deep Dive (NEVER CUT)
- Select trial → all issues for that trial
- Trial metadata panel
- **v1 risk model: completion risk score + key risk factors** ← most differentiated
- Eligibility criteria text
- Related trials panel (cut this if time tight, keep everything else)

### Page 4: Audit Log & Evaluation
- Audit tab: every agent action, reasoning, LLM calls, exportable JSON
- Evaluation tab: gold set metrics summary card (precision/recall/F1/kappa)
- Pattern library (if implemented)

---

## Evaluation (evaluation/)

### Gold Set (evaluation/gold_set.py)
- 20 issues labelled from real agent findings on Saturday evening
- Distribution: ~5 HIGH, ~8 MEDIUM, ~7 LOW
- Include 2-3 edge cases
- Format:
```python
{
    "trial_id": str,
    "check_name": str,
    "finding": str,
    "gold_severity": str,      # HIGH/MEDIUM/LOW
    "gold_rationale": str      # why you labelled it this way
}
```

### Metrics (evaluation/metrics.py)
- Precision, Recall, F1 per severity level
- Overall accuracy
- Cohen's kappa
- Display as summary card in dashboard

---

## Output Format (output/report.py)

Full scan report JSON structure:
```json
{
  "scan_metadata": {
    "scan_id": "uuid",
    "timestamp": "ISO",
    "scope": "...",
    "data_source_note": "AACT registry (proxy for EDC/CTMS)",
    "trials_scanned": 0,
    "checks_executed": [],
    "agent_iterations": 0,
    "issues": {"critical": 0, "high": 0, "medium": 0, "low": 0}
  },
  "llmops_summary": {
    "total_llm_calls": 0,
    "total_tokens": 0,
    "total_latency_ms": 0,
    "estimated_cost_usd": 0.0,
    "calls": []
  },
  "issues": [],
  "pattern_library_updates": [],
  "audit_log": []
}
```

---

## Pattern Library (output/patterns.json)

Persists across scans. After each scan, append new patterns:
```json
{
  "pattern_id": "P001",
  "discovered": "ISO date",
  "scope": "Phase III Oncology",
  "observation": "...",
  "evidence": {"n_trials": 0, "n_affected": 0},
  "implication": "...",
  "times_confirmed": 1
}
```

On next scan: check if known patterns recur, increment times_confirmed.

---

## Never Cut List

These ship no matter what. Do not remove or simplify:

1. Agent loop (all 5 nodes + review/context loop max 3x)
2. LLMOps logging on every LLM call
3. Provenance on every PrioritisedIssue
4. v1 risk score integration + CRITICAL escalation logic
5. Page 3 trial deep dive
6. AACT vs EDC framing on every Streamlit page
7. Structured JSON export with audit log

## Cut Order If Time Tight

1. Pattern library — cut first
2. Evaluation tab detail — keep summary card only
3. Enrollment velocity check — bonus only
4. Page 1 real-time streaming — static summary acceptable
5. Page 3 related trials panel — keep risk score + eligibility

---

## Code Quality Requirements

- Type hints on all functions
- Docstrings on all classes and public methods
- Logging via Python `logging` module (not print statements) — use `log = logging.getLogger(__name__)`
- Error handling: every AACT query in try/except, graceful degradation if table missing
- No hardcoded paths — use `pathlib.Path` relative to project root
- All API keys via `os.environ` with `load_dotenv()` — never hardcoded
- Every LLM call wrapped in LLMTracker — no raw API calls outside tracker

---

## What NOT to Build

- ❌ Real-time continuous monitoring — on-demand scan only
- ❌ External data sources beyond AACT
- ❌ User authentication or multi-tenant
- ❌ NL→SQL interface (that's Rivia Spark — they already have it)
- ❌ Over-engineered agent — 3 iterations max
- ❌ Docker or deployment config — local demo only

---

## Demo Script (memorise this)

```
[FRAMING — always first]
"This demo uses AACT registry data as a proxy for EDC/CTMS. 
The architecture is portable — check logic changes, 
orchestration and audit trail remain identical."

[SCAN]
"Scanning Phase III oncology trials, completed 2023-2025.
Agent planned 4 check categories, scanned 847 trials,
2 iterations, 23 issues found."

[CRITICAL ISSUE]
"Top finding: Trial enrolled 340 patients, reported results 
for only 12. Agent flagged HIGH severity. The completion risk 
model — trained on 573,000 real trials — scores this at 0.72. 
Combined signal: CRITICAL."

[PROVENANCE]
"Every finding has full provenance — source tables, columns,
check rule, timestamps, LLM call IDs. Full audit trail."

[LLMOps]
"7 LLM calls, 47k tokens, $0.08 estimated cost, 12.3s latency.
Track this in production to manage cost and optimise prompts."

[EVALUATION]
"20-sample gold set from real findings. Severity precision 0.83,
recall 0.79. This is how I iterate on prompt quality."
```

---

## Cover Note to Rivia

> "I built a proactive data quality agent that mirrors what your CEO 
> described as Rivia's key product bet. It runs on AACT registry data 
> as a proxy for EDC/CTMS — the architecture is designed to be portable. 
> The system combines deterministic checks, an agentic loop with genuine 
> reasoning, LLM prioritisation with full provenance, and a pre-trained 
> completion risk model trained on 573,000 real trials. Happy to walk 
> through the architecture on a call."

---

## Weekend Build Order

| Phase | Files | Priority |
|---|---|---|
| 1 | checks/models.py, checks/temporal.py, checks/enrollment.py, checks/status.py, checks/endpoints.py, checks/crossfield.py | MUST |
| 2 | models/risk_scorer.py | MUST |
| 3 | agents/state.py, agents/orchestrator.py | MUST |
| 4 | llmops/tracker.py + LLM prioritisation in orchestrator | MUST |
| 5 | output/report.py + provenance | MUST |
| 6 | output/patterns.json + pattern library logic | CUTTABLE |
| 7 | evaluation/gold_set.py, evaluation/metrics.py | SHOULD |
| 8 | app_v2.py (4 pages) | MUST |
| BONUS | Enrollment velocity check in checks/crossfield.py | BONUS |
