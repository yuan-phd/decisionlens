# CLAUDE.md — DecisionLENS v2
## Agentic Data Quality Agent for Clinical Trials

---

## Project Context

This is **DecisionLENS v2** — an agentic data quality system built as a demo for **Rivia** (rivia.com), a $15M Series A clinical trial intelligence platform. Rivia's CEO described their key product bet as *"an agentic data quality product that continuously surfaces, prioritises, and manages the thousands of data issues that arise during a trial."* This project mirrors that product.

**Rivia's core mission:** Unify fragmented clinical data sources into actionable intelligence. This demo must demonstrate multi-source unification — not just single-database checks.

**Critical framing — say this everywhere:**
> "This demo uses AACT registry data, PubMed literature, and OpenFDA safety data as proxies for EDC/CTMS data. The same agent architecture applies to live trial execution data — check logic changes, orchestration and audit trail remain identical."

This framing must appear as a persistent banner on every Streamlit page.

---

## Repository Structure

```
/Users/ye/work/portfolio/decisionlens/
├── mcp_servers/
│   ├── aact_server.py          # MCP 1: AACT database tools
│   ├── pubmed_server.py        # MCP 2: PubMed E-utilities tools
│   ├── openfda_server.py       # MCP 3: OpenFDA API tools
│   └── cache/                  # API response cache (JSON files)
├── agents/
│   ├── orchestrator.py         # LangGraph agent — 5 nodes + loop
│   └── state.py                # DataQualityState TypedDict
├── checks/
│   ├── models.py               # Issue dataclass — shared return type
│   ├── temporal.py             # Check A: date consistency
│   ├── enrollment.py           # Check B: enrollment anomalies
│   ├── status.py               # Check C: status inconsistencies
│   ├── endpoints.py            # Check D: endpoint & design gaps
│   ├── crossfield.py           # Check E: cross-field validation
│   ├── publication.py          # Check F: publication vs registry (cross-source)
│   └── safety.py               # Check G: trial safety vs FDA (cross-source)
├── models/
│   ├── xgb_classifier.pkl      # v1 XGBoost model (AUC=0.787, 573k trials)
│   └── risk_scorer.py          # Wrapper: loads v1 model, returns risk score
├── llmops/
│   └── tracker.py              # LLMTracker: logs every LLM call
├── evaluation/
│   ├── gold_set.py             # 20 manually labelled issues
│   └── metrics.py              # Precision/recall/F1/Cohen's kappa
├── output/
│   ├── report.py               # Structured JSON output + provenance
│   └── patterns.json           # Pattern library (persists across scans)
├── src/                        # v1 modules — DO NOT MODIFY
│   ├── data_pipeline.py
│   ├── models.py
│   ├── competitive_intel.py
│   ├── investigator_insights.py
│   └── genai_utils.py
├── data/
│   └── processed/              # AACT parquet files (10 tables, 3.4M records)
├── app_v2.py                   # Streamlit dashboard — 4 pages
└── README_v2.md
```

---

## Technology Stack

- **Python 3.14** (M1 Max Mac)
- **LangGraph** — agent orchestration (if incompatible, use raw ReAct loop)
- **MCP (Model Context Protocol)** — 3 servers exposing tools to the agent
- **XGBoost** — v1 completion risk model already trained
- **OpenAI GPT-4o-mini** — LLM prioritisation (Groq as fallback)
- **Pandas + PyArrow** — AACT data access via parquet files
- **Streamlit + Plotly** — dashboard
- **python-dotenv** — key management

Always run `from dotenv import load_dotenv; load_dotenv()` at the top of every file that needs API keys.

---

## MCP Architecture

### Why MCP (not hardcoded tools)

Rivia's core value proposition is unifying fragmented clinical data sources. A demo with one database shows "data quality checks." A demo with 3 MCP servers cross-validating across sources shows "unifying fragmented clinical data into actionable intelligence" — their exact mission.

The 3 MCP servers expose all tools to the LangGraph agent via the MCP protocol. The agent decides which tools to call and in what order — it is not hardcoded.

---

### MCP Server Pattern (all 3 servers follow this)

```python
from mcp.server import Server
from mcp.types import Tool, TextContent
import json

server = Server("server-name")

@server.list_tools()
async def list_tools():
    return [
        Tool(
            name="tool_name",
            description="What this tool does",
            inputSchema={
                "type": "object",
                "properties": {
                    "param": {"type": "string", "description": "..."}
                },
                "required": ["param"]
            }
        )
    ]

@server.call_tool()
async def call_tool(name: str, arguments: dict):
    if name == "tool_name":
        result = your_function(**arguments)
        return [TextContent(type="text", text=json.dumps(result))]
```

---

### MCP 1: AACT Clinical Trials Registry (mcp_servers/aact_server.py)

**Data source:** AACT parquet files at data/processed/

```
Tools:

  search_trials(therapeutic_area, phase, status, date_range, limit)
    → Returns trial metadata (NCT ID, sponsor, enrollment, dates, status)

  get_trial_details(nct_id)
    → Full trial record: design, arms, conditions, interventions, outcomes

  get_eligibility_criteria(nct_id)
    → Raw eligibility text for LLM extraction

  get_trial_results(nct_id)
    → Reported results, adverse events, outcome measures

  get_related_trials(nct_id)
    → Trials with same sponsor, same condition, or same intervention

  run_quality_check(check_name, therapeutic_area, phase, limit)
    → Executes one of the 5 deterministic check functions (A-E)
    → Returns list of Issue objects as JSON
```

---

### MCP 2: PubMed Academic Literature (mcp_servers/pubmed_server.py)

**Data source:** PubMed E-utilities API (free, no key needed, 3 req/sec)
- Base URL: `https://eutils.ncbi.nlm.nih.gov/entrez/eutils/`
- Add 0.35s sleep between calls to respect rate limit
- Cache all responses in mcp_servers/cache/

```
Tools:

  search_publications(query, max_results=10)
    → Search PubMed by keywords, trial ID, drug name, or PI name
    → Returns: PMID, title, authors, journal, date, abstract snippet

  get_publication_details(pmid)
    → Full abstract, MeSH terms, publication type

  search_trial_publications(nct_id)
    → Search PubMed for papers mentioning this NCT ID
    → Many papers cite their NCT ID in abstract or as secondary ID

  search_author_publications(author_name, years=3)
    → Recent publications by a specific author
    → Used to check if trial PI published results elsewhere
```

---

### MCP 3: OpenFDA Drug Adverse Events (mcp_servers/openfda_server.py)

**Data source:** OpenFDA API (free, no key needed, <240 req/min)
- Base URL: `https://api.fda.gov/drug/event.json`
- Cache all responses in mcp_servers/cache/

```
Tools:

  search_adverse_events(drug_name, date_range=None, limit=10)
    → FDA adverse event reports for a specific drug
    → Returns: report count, serious event count, top reactions

  get_drug_safety_summary(drug_name)
    → Aggregated safety profile: total reports, serious %,
      death %, top 10 reactions, trend over time

  compare_trial_vs_fda_events(nct_id, drug_name)
    → Compares adverse events in AACT trial results vs
      FDA post-market reports for same drug
```

---

### API Caching (all external APIs)

```python
import hashlib, json
from pathlib import Path

CACHE_DIR = Path("mcp_servers/cache")
CACHE_DIR.mkdir(exist_ok=True)

def cached_api_call(func, cache_key_args, **kwargs):
    key = hashlib.md5(json.dumps(cache_key_args, sort_keys=True).encode()).hexdigest()
    cache_path = CACHE_DIR / f"{func.__name__}_{key}.json"
    if cache_path.exists():
        return json.loads(cache_path.read_text())
    result = func(**kwargs)
    cache_path.write_text(json.dumps(result))
    return result
```

### API Error Handling

External APIs can fail. Each MCP server must return structured errors:

```python
try:
    fda_data = get_drug_safety_summary(drug_name=drug)
except Exception as e:
    return {
        "source": "openfda",
        "status": "unavailable",
        "reason": str(e),
        "impact": "FDA safety cross-validation skipped for this trial"
    }
```

Agent treats API failure as "data unavailable" — not "no issues found." Audit log records which sources were reachable.

---

## Data Sources

### AACT Parquet Files (data/processed/)
- `studies.parquet` — primary trial table
- `eligibilities.parquet` — eligibility criteria text
- `facilities.parquet` — investigator sites
- `conditions.parquet` — trial conditions/indications
- `sponsors.parquet` — sponsor information
- `designs.parquet` — trial design
- `outcome_counts.parquet` — reported results
- `calculated_values.parquet` — derived fields
- `interventions.parquet` — intervention details
- `countries.parquet` — geographic distribution

Always load via pandas: `pd.read_parquet('data/processed/studies.parquet')`

### External APIs (free, no keys needed)
- **PubMed:** `https://eutils.ncbi.nlm.nih.gov/entrez/eutils/`
- **OpenFDA:** `https://api.fda.gov/drug/event.json`

---

## v1 Risk Model

The v1 XGBoost classifier is already trained and saved at `models/xgb_classifier.pkl`.

- Trained on 573,000 real AACT trials
- AUC = 0.787, 5-fold CV stable ±0.004
- **Completion risk score = 1 - completion_probability**
- Features: enrollment, n_facilities, phase_numeric, sponsor_historical_performance, competing_trials_count, intervention_model, masking, is_multicountry, geographic_concentration, condition_prevalence_proxy

Feature pipeline: `src/data_pipeline.py` → `TrialDataPipeline.engineer_features()`

**Escalation logic:** (thresholds inclusive)
- HIGH issue + risk_score >= 0.6 → upgrade to CRITICAL
- MEDIUM issue + risk_score >= 0.7 → upgrade to HIGH

---

## Core Data Models

### Issue (checks/models.py)
```python
@dataclass
class Issue:
    trial_id: str
    check_name: str
    check_category: str      # temporal/enrollment/status/endpoint/crossfield/publication/safety
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

    # Multi-source provenance — always present, never optional
    provenance: dict             # see provenance spec below
```

### Multi-Source Provenance Spec
```python
provenance = {
    "sources_queried": ["aact", "pubmed", "openfda"],
    "sources_available": ["aact", "pubmed"],  # partial if API down
    "aact": {
        "tables": list[str],
        "columns": list[str],
        "check_rule": str
    },
    "pubmed": {
        "query": str,
        "results_found": int,
        "pmids": list[str]
    },
    "openfda": {
        "drug_queried": str,
        "total_adverse_events": int,
        "serious_events": int
    },
    "data_accessed_at": str,
    "context_enriched": bool,
    "agent_iteration": int,
    "llm_call_ids": list[str],
    "risk_model_used": bool,
    "risk_model_version": str
}
```

---

## Check Functions

### Checks A-E: Single-Source (AACT only)

#### Check A: Temporal Consistency (checks/temporal.py)
Flag: completion_date < start_date, status "Recruiting" but completion_date in past, results_first_submitted >2 years after completion, last_update >2 years ago for non-terminated trials.

#### Check B: Enrollment Anomalies (checks/enrollment.py)
Flag: enrollment = 0 but status "Completed", actual < 20% of anticipated, actual > 500% of anticipated, completed >1yr ago but no results, results reported for <50% of enrolled patients.

#### Check C: Status Inconsistencies (checks/status.py)
Flag: "Completed" but no results, "Active not recruiting" >3 years, "Recruiting" but last_update >2 years ago, withdrawn with no reason, status changed but no date update.

#### Check D: Endpoint & Design Gaps (checks/endpoints.py)
Flag: Phase III but no primary outcome, outcome has no timeframe, interventional study with no arm, randomized but only 1 arm, missing masking for RCTs.

#### Check E: Cross-Field Validation (checks/crossfield.py)
Flag: Phase I with enrollment >500, pediatric study with no pediatric conditions, results but no adverse events, multi-site trial with 1 facility, completion date passed but no results or termination reason.

All check functions signature:
```python
def check_xxx(trial_id=None, therapeutic_area=None, limit=100) -> list[Issue]:
```

---

### Checks F-G: Cross-Source (MCP 2 + MCP 3)

#### Check F: Publication vs Registry (checks/publication.py)
Cross-validate AACT against PubMed:
- HIGH: PubMed publication reports different enrollment/outcome than AACT
- MEDIUM: Completed >2yr ago, no AACT results, but PubMed has publications from trial authors
- MEDIUM: Withdrawn/terminated but PI published results from same population
- LOW: Completed, has AACT results, but zero publications found

#### Check G: Trial Safety vs FDA (checks/safety.py)
Cross-validate AACT against OpenFDA:
- HIGH: Trial reports zero AEs but OpenFDA has significant reports for same drug
- HIGH: FDA recall or safety alert during trial period, trial status unchanged
- MEDIUM: Trial AE profile incomplete vs FDA post-market data
- LOW (positive): Clean trial safety profile confirmed by FDA data

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

### Agent Flow — 5 Nodes (agents/orchestrator.py)

1. **PLANNING NODE** — LLM decides which checks to run across all 3 MCP servers based on scope. Not hardcoded.

2. **SCAN NODE** — Executes planned checks via MCP tool calls. AACT checks A-E + cross-source checks F-G if relevant. Attaches v1 risk score.

3. **REVIEW NODE** — LLM reviews all findings. Decides if any need deeper multi-source investigation.

4. **CONTEXT ENRICHMENT NODE** — For flagged trials: AACT (eligibility, related trials) + PubMed (publications) + OpenFDA (adverse events). Loops back to REVIEW max 3x.

5. **PRIORITISATION NODE** — LLM generates final assessment using full multi-source context + risk scores. Updates pattern library.

### Example multi-source reasoning:
```
AACT: NCT05123456 completed 2023, 340 enrolled, 12 results reported
PubMed: 2 publications from trial PI citing "340-patient Phase III trial"
OpenFDA: 23 adverse event reports for trial drug, 4 serious

Agent: "Results exist (published) but not in registry.
        Active FDA safety signals confirm urgency."
Severity: CRITICAL — impossible finding with single source
```

---

## LLMOps (llmops/tracker.py)

Every LLM call MUST go through LLMTracker. No exceptions.

```python
class LLMTracker:
    def track_call(self, model, purpose, prompt_tokens,
                   completion_tokens, latency_ms) -> dict:
        # purpose: "planning"/"review"/"context"/"prioritisation"

    def get_summary(self) -> dict:
        # total_calls, total_tokens, total_latency_ms,
        # total_estimated_cost_usd, calls list
```

Cost: GPT-4o-mini = $0.15/1M input, $0.60/1M output tokens.

---

## LLM Prioritisation Prompt

```
You are a clinical trial data quality analyst reviewing automated
findings from AACT registry, PubMed literature, and FDA adverse
event database.

Trial: {trial_id}
Trial metadata: {trial_metadata}
Findings: {findings}
Completion risk score: {risk_score}/1.0 (AUC=0.787, 573k trials)
PubMed context: {pubmed_context}
OpenFDA context: {openfda_context}
Related patterns: {related_patterns}

For each finding:
1. severity: CRITICAL/HIGH/MEDIUM/LOW
   CRITICAL = HIGH + risk>=0.6 OR cross-source discrepancy 2+ sources
2. explanation: 2-3 sentences, no jargon, cite sources used
3. potential_impact
4. suggested_action
5. confidence: 0.0-1.0 (higher when multiple sources agree)

JSON array. No markdown, no preamble.
```

---

## Streamlit Dashboard (app_v2.py)

Persistent banner ALL pages:
```
⚠️ Data sources: AACT registry + PubMed + OpenFDA (proxies for EDC/CTMS).
Same agent architecture applies to live trial execution data.
```

### Page 1: Scan Control & Overview
- Scope selectors + data source availability indicators (AACT ✅ PubMed ✅ OpenFDA ✅)
- Run Scan button + progress
- Summary cards: trials scanned, CRITICAL/HIGH/MEDIUM/LOW
- LLMOps card: calls, tokens, cost, latency

### Page 2: Prioritised Issues List
- Table: Rank, Trial ID, Severity, Check, Sources Used, Explanation, Risk Score, Confidence
- CRITICAL rows: red banner
- Cross-source badge: "AACT + PubMed" or "AACT + FDA"
- Expandable: full explanation, data points, action, multi-source provenance
- Export JSON button

### Page 3: Trial Deep Dive (NEVER CUT)
- Trial metadata (AACT)
- **v1 risk model: score + key risk factors** ← most differentiated
- PubMed panel: publications found
- OpenFDA panel: adverse event summary
- Eligibility criteria text

### Page 4: Audit Log & Evaluation
- Audit tab: every agent action, MCP calls, LLM calls, exportable JSON
- Data sources tab: MCP server status, response times, cache hits
- Evaluation tab: gold set metrics summary card
- Pattern library (if implemented)

---

## Evaluation

### Gold Set — 20 issues from real agent findings (Saturday evening)
Include cross-source findings. Distribution: ~5 HIGH, ~8 MEDIUM, ~7 LOW, 2-3 edge cases.

### Metrics
Precision, Recall, F1 per severity + overall accuracy + Cohen's kappa.

---

## Never Cut List

1. Agent loop (all 5 nodes + max 3 iterations)
2. All 3 MCP servers (AACT + PubMed + OpenFDA)
3. LLMOps logging on every LLM call
4. Multi-source provenance on every issue
5. v1 risk score + CRITICAL escalation
6. Page 3 trial deep dive
7. AACT vs EDC framing on every page
8. Structured JSON export with audit log

## Cut Order If Time Tight

1. Pattern library
2. Evaluation tab detail (keep summary card)
3. Enrollment velocity check (bonus only)
4. Check G safety cross-validation (keep Check F)
5. Page 1 real-time streaming (static summary ok)
6. Page 3 related trials panel

---

## Code Quality

- Type hints + docstrings on all public functions/classes
- `logging` module throughout — `log = logging.getLogger(__name__)`
- All AACT queries in try/except
- All external API calls cached + error handled
- `pathlib.Path` for all file paths
- `load_dotenv()` at top of every file needing keys
- Every LLM call through LLMTracker

## What NOT to Build

- ❌ Real-time continuous monitoring
- ❌ User auth or multi-tenant
- ❌ NL→SQL interface (that's Rivia Spark)
- ❌ Over-engineered agent (3 iterations max)
- ❌ Docker or deployment config

---

## Weekend Build Order

| Phase | Files | Priority |
|---|---|---|
| 1 | checks/models.py + checks A-E | MUST |
| 2 | models/risk_scorer.py | MUST |
| 3 | mcp_servers/aact_server.py | MUST |
| 4 | mcp_servers/pubmed_server.py | MUST |
| 5 | mcp_servers/openfda_server.py | MUST |
| 6 | checks/publication.py + checks/safety.py | MUST |
| 7 | agents/state.py + agents/orchestrator.py | MUST |
| 8 | llmops/tracker.py + LLM prioritisation | MUST |
| 9 | output/report.py + multi-source provenance | MUST |
| 10 | output/patterns.json (cuttable) | CUTTABLE |
| 11 | evaluation/gold_set.py + metrics.py | SHOULD |
| 12 | app_v2.py (4 pages) | MUST |
| BONUS | Enrollment velocity check | BONUS |

---

## Demo Script

```
[FRAMING — always first]
"AACT registry, PubMed, and OpenFDA as proxies for EDC/CTMS.
Portable architecture — check logic changes, orchestration stays."

[CRITICAL CROSS-SOURCE FINDING]
"This finding was impossible with a single source.
AACT: 340 enrolled, 12 results reported.
PubMed: results exist — published by trial PI, never submitted.
OpenFDA: 4 serious adverse events for the trial drug.
Three sources. One CRITICAL finding. This is what unifying
fragmented clinical data looks like in practice."

[PROVENANCE]
"Every finding shows which sources were consulted, what each
returned, which check rule fired, which LLM calls were made."

[LLMOps]
"7 LLM calls, 47k tokens, $0.08, 12.3s."

[EVALUATION]
"20-sample gold set, precision 0.83, recall 0.79."
```

---

## Cover Note to Rivia

> "I built a proactive data quality agent that mirrors what your CEO
> described as Rivia's key product bet. It unifies AACT registry,
> PubMed, and OpenFDA into coherent findings — surfacing issues
> no single-source system can detect. It includes a pre-trained
> completion risk model trained on 573,000 real trials, full
> LLMOps instrumentation, and auditable multi-source provenance
> on every finding. Happy to walk through the architecture on a call."
