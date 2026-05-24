# DecisionLENS v2 — Agentic Data Quality for Clinical Trials

**LangGraph agent that proactively detects, prioritises, and clusters data quality issues across clinical trial registries — with multi-source cross-validation, LLMOps tracing, and structured provenance for regulated environments.**

> Built with: Python · LangGraph · GPT-4o-mini · XGBoost · 3 MCP servers (AACT, PubMed, OpenFDA) · Streamlit · 100K real AACT trials

**Live demo:** [decisionlens-demo.streamlit.app](https://decisionlens-demo.streamlit.app/) (pre-computed interactive report viewer)

**v1 analytics platform:** [decisionlens-yuan.streamlit.app](https://decisionlens-yuan.streamlit.app/) (enrollment forecasting, competitive intelligence, survival analysis)

---

## What it does

DecisionLENS v2 is an agentic data quality system for clinical trial data. Given a scope (e.g. "Recruiting Phase III trials"), the agent:

1. **Plans** which checks to run across 7 categories
2. **Scans** trial records from AACT (100K real ClinicalTrials.gov trials)
3. **Cross-validates** findings against PubMed publications and OpenFDA adverse event reports
4. **Prioritises** findings using a rule engine + XGBoost risk model (AUC=0.799) + LLM severity assessment
5. **Clusters** related issues into root cause hypotheses
6. **Logs** every LLM call with token counts, cost, and latency for LLMOps monitoring

The output is a structured scan report with full provenance — every finding traces back to which data sources were queried, what the rule engine found, what the risk model predicted, and why the LLM assigned that severity.

---

## Architecture

```
![Agentic architecture](/figures/architecture.png)
```

---

## Check Categories

The agent runs checks across 7 categories (A–G):

| Category | Check | Example finding |
|----------|-------|----------------|
| A. Temporal | Recruiting past completion date | "Status is Recruiting but completion date was 287 days ago" |
| B. Enrollment | Phase 3 underpowered | "Planned enrollment is 38 (< 50 threshold)" |
| C. Status | Completed with future end date | "Completion date is 2027-06-01 but status is Completed" |
| D. Endpoints | Missing primary outcome | "No primary outcome registered for Phase 3 trial" |
| E. Cross-field | Missing design fields | "Late-phase trial missing allocation field" |
| F. Publication | No publications for completed trial | Cross-validated against PubMed API |
| G. Safety | Incomplete AE profile vs FDA | Cross-validated against OpenFDA adverse events |

Categories A–E use AACT data only. Categories F and G cross-validate against external sources (PubMed, OpenFDA) — these are the "multi-source" findings that demonstrate the agent's ability to synthesise across data silos.

---

## Severity & Risk Model

Findings are assigned severity through a three-stage process:

1. **Rule engine** assigns initial severity (HIGH/MEDIUM/LOW) based on check logic
2. **XGBoost risk model** (v2, AUC=0.799, trained on 22K real AACT trials) predicts trial completion risk
3. **Escalation**: HIGH findings on trials with reliable risk score ≥ 0.6 are escalated to CRITICAL

The dashboard translates these into human-readable conclusions:

- **Layer 1** (visible immediately): severity badge + finding + suggested action
- **Layer 2** (click to expand): "Why this severity?" explanation, sources checked, assessment confidence

---

## Evaluation

Severity classification evaluated on an 18-sample gold set (single annotator):

| Metric | Value |
|--------|-------|
| Accuracy | 0.722 (13/18) |
| Cohen's κ | 0.596 (moderate agreement) |
| CRITICAL precision | 1.000 (zero false escalations) |
| Match rate | 100% (18/18 gold issues found) |

The system over-classifies moderate recruiting delays as HIGH (should be MEDIUM). It never false-escalates to CRITICAL — 100% precision on the highest-priority class.

> This gold set evaluates severity classification accuracy only. Detection recall evaluation would require an independently annotated trial dataset — out of scope for this demo but a natural next step in production validation. Gold set labelled by single annotator — in production, 2–3 clinical data managers with adjudication would be used.

---

## LLMOps

Every scan logs:

| Field | Example |
|-------|---------|
| LLM calls | 6 |
| Total tokens | 5,484 |
| Cost | $0.0017 |
| Latency | 43.8 s |
| Model | gpt-4o-mini |
| Purposes | planning, review, prioritisation, clustering |

Per-call tracing is visible in the Audit Log (Page 4) with call_id, purpose, tokens, cost, latency, and status for each invocation.

---

## Project Structure

```
decisionlens/
├── agents/
│   ├── orchestrator.py           # LangGraph 5-node agent
│   └── test_agent.py             # E2E agent tests
├── checks/
│   ├── models.py                 # Issue dataclass, filter_trial_scope
│   ├── temporal.py               # Check A: temporal consistency
│   ├── enrollment.py             # Check B: enrollment adequacy
│   ├── status.py                 # Check C: status consistency
│   ├── endpoints.py              # Check D: endpoint completeness
│   ├── crossfield.py             # Check E: cross-field validation
│   ├── publication.py            # Check F: publication cross-validation
│   └── safety.py                 # Check G: FDA safety cross-validation
├── mcp_servers/
│   ├── aact_server.py            # AACT MCP server (local parquet)
│   ├── pubmed_server.py          # PubMed MCP server (API + cache)
│   ├── openfda_server.py         # OpenFDA MCP server (API + cache)
│   └── cache/                    # Persistent API response cache
├── models/
│   ├── risk_scorer.py            # XGBoost risk scoring + escalation
│   ├── xgb_classifier.pkl        # Trained classifier (AUC=0.799)
│   ├── cox_ph.pkl                # Cox PH survival model
│   └── test_risk_scorer.py       # Risk scorer tests
├── llmops/
│   ├── tracker.py                # Per-call LLM usage tracking
│   └── test_tracker.py           # Tracker unit tests
├── evaluation/
│   ├── gold_set.json             # 18-sample labelled gold set
│   ├── metrics.py                # Evaluation pipeline
│   └── metrics_results.json      # Latest evaluation results
├── output/
│   ├── report.py                 # ScanReport with to_dict/from_dict
│   ├── reports/                  # Saved scan reports (JSON)
│   ├── demo/                     # Pre-computed demo scope reports
│   └── test_report.py            # Report roundtrip tests
├── tests/
│   ├── test_checks_edge.py       # Edge case tests (20 assertions)
│   └── test_agent_resilience.py  # LLM failure graceful degradation
├── src/
│   ├── data_pipeline.py          # TrialDataPipeline
│   └── models.py                 # EnrollmentForecaster (v1 model training)
├── notebooks/
│   ├── 01_eda_trial_landscape.ipynb
│   ├── 02_feature_engineering.ipynb
│   └── 03_enrollment_model.ipynb
├── data/
│   ├── processed/                # 100K real AACT trial parquets
│   └── deployment/               # Deployment sample
├── app_v2.py                     # Live agent dashboard
├── app_demo.py                   # Pre-computed demo viewer
├── app_lib.py                    # Shared dashboard components
├── pre_warm_demo.py              # Generate demo scope reports
├── setup_data.py                 # AACT data setup
└── requirements.txt
```

---

## Quickstart

### Live mode (full agent)

```bash
git clone https://github.com/yuan-phd/decisionlens.git
cd decisionlens

python -m venv venv
source venv/bin/activate
pip install -r requirements.txt

# Set OpenAI API key for the LLM agent
export OPENAI_API_KEY=your_key_here

# Set up data (downloads AACT flat files → parquet)
python setup_data.py

# Train the risk model (or use the included pre-trained model)
jupyter notebook notebooks/03_enrollment_model.ipynb

# Launch live dashboard
streamlit run app_v2.py
```

Select Phase 3 + Recruiting + limit 50 → click Run Scan → ~60 seconds → results.

### Demo mode (no API key needed)

```bash
# Launch pre-computed report viewer
streamlit run app_demo.py
```

Select a demo scope from the sidebar → browse pre-computed results across all 4 pages.

---

## Tests

```bash
# Run all test suites
python -m pytest tests/test_checks_edge.py -v        # 20 edge case assertions
python models/test_risk_scorer.py                      # Risk scorer + escalation
python -m pytest tests/test_agent_resilience.py -v     # LLM failure degradation
python agents/test_agent.py                            # E2E agent scan
python llmops/test_tracker.py                          # LLMOps tracker
python output/test_report.py                           # Report roundtrip

# All 6 test files pass on real AACT data
```

---

## Data

- **Source**: [AACT / ClinicalTrials.gov](https://aact.ctti-clinicaltrials.org/) flat-file snapshot
- **Scale**: 100K real clinical trials (deployment sample from 573K full database)
- **Tables**: studies, calculated_values, eligibilities, designs, facilities, countries, sponsors, conditions, interventions, outcome_counts, outcomes
- **Labels**: COMPLETED → 1, TERMINATED → 0; RECRUITING/ACTIVE/etc. → unlabeled (used for Cox PH censoring)

---

## v1 → v2 Evolution

| | v1 | v2 |
|---|---|---|
| **Purpose** | Enrollment forecasting & analytics | Proactive data quality monitoring |
| **Architecture** | Monolithic (src/ modules) | Agentic (LangGraph + MCP servers) |
| **Data sources** | AACT only | AACT + PubMed + OpenFDA |
| **Model** | XGBoost (AUC=0.787, synthetic) | XGBoost (AUC=0.799, real AACT) |
| **LLM usage** | Eligibility analysis (Llama 3.3) | Planning, prioritisation, clustering (GPT-4o-mini) |
| **Output** | Dashboard with charts | Structured scan report with provenance |
| **Evaluation** | AUC/F1 only | Gold set + Cohen's κ + per-severity P/R/F1 |
| **Audit** | None | Per-call LLMOps tracing |

v1 remains live at [decisionlens-yuan.streamlit.app](https://decisionlens-yuan.streamlit.app/) — enrollment forecasting, competitive intelligence, investigator insights, and LLM eligibility analysis.

---

## Design Decisions

**Why MCP servers instead of direct API calls?** MCP provides a standardised tool interface that the LangGraph agent can discover and invoke dynamically. Each server handles its own caching, rate limiting, and error handling — the agent doesn't need to know the implementation details of PubMed vs OpenFDA.

**Why rule engine + risk model + LLM (three-stage severity)?** Rules catch deterministic patterns (missing fields, date inconsistencies). The risk model adds probabilistic context from 22K historical trials. The LLM synthesises both signals with domain reasoning. No single approach covers all cases — the combination is more robust than any one alone.

**Why root cause clustering?** Individual findings are actionable but don't reveal systemic patterns. Clustering groups related issues (e.g. "5 trials from the same sponsor all missing primary outcomes") into hypotheses that suggest process-level fixes rather than per-trial patches. This is LLM-assisted hypothesis generation, not statistical causal inference.

**Why two-layer display?** Clinical operations managers need conclusions ("this trial has a critical issue, take this action"). Engineers need provenance ("which rule fired, what was the risk score, which sources were checked"). The two-layer design serves both audiences from the same interface.

---

## License

MIT — see `LICENSE` for details.