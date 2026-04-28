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
│   ├── aact_server.py          # MCP 1: AACT parquet tools (5 tools, see spec below)
│   ├── pubmed_server.py        # MCP 2: PubMed E-utilities (4 tools)
│   ├── openfda_server.py       # MCP 3: OpenFDA API (3 tools)
│   └── cache/                  # API response cache (gitignored)
├── agents/
│   ├── orchestrator.py         # LangGraph agent — 6 nodes + review loop
│   ├── state.py                # DataQualityState TypedDict + caps
│   └── test_agent.py
├── checks/
│   ├── models.py               # Issue dataclass + filter_trial_scope
│   ├── temporal.py             # Check A: date consistency
│   ├── enrollment.py           # Check B: enrollment anomalies
│   ├── status.py               # Check C: status inconsistencies
│   ├── endpoints.py            # Check D: endpoint & design gaps
│   ├── crossfield.py           # Check E: cross-field validation
│   ├── publication.py          # Check F: publication vs registry (cross-source)
│   ├── safety.py               # Check G: trial safety vs FDA (cross-source)
│   ├── demo_overrides.py       # Determinism shims for demo mode
│   ├── LIMITATIONS.md          # AACT parquet schema gaps documented
│   ├── test_checks.py
│   └── test_cross_source.py
├── models/
│   ├── xgb_classifier.pkl      # v2 XGBoost (AUC=0.799, 22K real AACT) — gitignored
│   ├── xgb_regressor.pkl       # v2 enrollment regressor — gitignored
│   ├── cox_ph.pkl              # Cox PH survival (46,897 obs) — gitignored
│   ├── forecaster.joblib       # v1 EnrollmentForecaster — gitignored
│   ├── risk_scorer.py          # Returns {risk_score, risk_reliable, risk_model_version}
│   └── test_risk_scorer.py
├── llmops/
│   ├── tracker.py              # LLMTracker — logs every LLM call
│   └── test_tracker.py
├── evaluation/
│   ├── gold_set.py             # Loader for gold_set.json
│   ├── gold_set.json           # 18 manually labelled issues (single annotator)
│   ├── metrics.py              # Precision/recall/F1/Cohen's kappa
│   └── metrics_results.json    # Last evaluation run output
├── output/
│   ├── report.py               # ScanReport / PrioritisedIssue dataclasses
│   ├── demo/                   # Pre-computed scans for app_demo.py (3 scopes)
│   ├── reports/                # Live-scan outputs (gitignored)
│   └── test_report.py
├── tests/
│   ├── test_agent_resilience.py
│   └── test_checks_edge.py
├── src/                        # v1 modules — DO NOT MODIFY
│   ├── data_pipeline.py
│   ├── models.py
│   ├── competitive_intel.py
│   ├── investigator_insights.py
│   └── genai_utils.py
├── data/
│   └── processed/              # AACT parquet (11 tables, 100K trials, 265 MB) — gitignored
├── app_v2.py                   # Streamlit live mode — runs the agent (needs OpenAI key)
├── app_demo.py                 # Streamlit demo mode — loads pre-computed reports, zero deps
├── app_lib.py                  # Shared UI helpers (severity badges, scan-state guards)
├── pre_warm_demo.py            # Regenerates output/demo/*.json
└── README.md
```

---

## Technology Stack

- **Python 3.14** (M1 Max Mac)
- **LangGraph** — 6-node StateGraph orchestration
- **MCP (Model Context Protocol)** — 3 servers exposing tools to the agent
- **XGBoost** — v2 completion risk model (AUC=0.799 on 22K real AACT trials)
- **OpenAI GPT-4o-mini** — all LLM calls (planning, review, prioritisation, clustering)
- **Pandas + PyArrow** — AACT data access via parquet files
- **Streamlit + Plotly** — dashboard (live + demo entry points)
- **python-dotenv** — key management

> v1 modules in `src/` still use Groq for LLM calls (kept for backwards-compat — `groq>=0.4` in `requirements.txt`). All v2 code paths use OpenAI exclusively.

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

**Data source:** AACT parquet files at data/processed/ (11 tables, 100K trials)

```
Tools (5):

  run_quality_check(check_name, therapeutic_area, phase, limit)
    → Executes one of the deterministic check functions (A-E)
    → Returns list of Issue.to_dict() payloads as JSON

  get_risk_score(nct_id)
    → Wraps risk_scorer.score_trial — returns
      {risk_score, risk_reliable, risk_model_version}.
      Surfaced as a tool so the agent can pull risk context
      mid-flow (used by node_review and node_context).

  get_trial_details(nct_id)
    → Full trial record: design, arms, conditions, interventions

  get_eligibility_criteria(nct_id)
    → Raw eligibility text for LLM extraction

  get_related_trials(nct_id)
    → Trials with same sponsor, same condition, or same intervention
```

> Plan-vs-built note: the original CLAUDE.md spec'd `search_trials` and
> `get_trial_results` instead of `get_risk_score`. The agent never needed
> `search_trials` (scopes are passed to `run_quality_check` directly via
> filters), and `get_trial_results` collapsed into `get_trial_details`.
> `get_risk_score` was added to expose the v2 model as a first-class tool.

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

### AACT Parquet Files (data/processed/, gitignored, ~265 MB)

11 tables, **100,000 trials in `studies.parquet`** (down-sampled snapshot
of the full AACT registry — not 3.4M rows). Schema gaps are documented
in `checks/LIMITATIONS.md`.

- `studies.parquet` — primary trial table (100K rows, 72 cols)
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

## v2 Risk Model

XGBoost classifier saved at `models/xgb_classifier.pkl`, loaded by
`models/risk_scorer.py`. Version string: `xgboost_v2_auc0799_22k_trials_realAACT`.

- **v2 retrain** on 22K real AACT trials drawn from the local parquet snapshot
- AUC = 0.799 (held-out test set)
- **Completion risk score = 1 - completion_probability**
- Same feature pipeline as v1: `src/data_pipeline.py` → `TrialDataPipeline.engineer_features()`

### Reliability gating (NEW vs original spec)

`risk_scorer.score_trial(nct_id)` returns:

```python
{
    "risk_score": float,         # 0.0–1.0
    "risk_reliable": bool,       # False when feature pipeline failed
    "risk_model_version": str,   # "xgboost_v2_auc0799_22k_trials_realAACT"
}
```

When the feature pipeline can't build a vector for a trial (missing
joins, malformed enrollment, etc.) the scorer returns a **neutral
fallback**: `risk_score=0.5`, `risk_reliable=False`. Downstream
escalation logic must check `risk_reliable` before acting.

### Escalation logic (thresholds inclusive)

```python
HIGH_TO_CRITICAL_THRESHOLD = 0.6
MEDIUM_TO_HIGH_THRESHOLD   = 0.7
```

Applied by `escalate_severity(severity_rule, risk)`:

- **Only when `risk["risk_reliable"]` is True** —
  neutral-fallback scores must not escalate.
- HIGH    + risk_score >= 0.6 → CRITICAL
- MEDIUM  + risk_score >= 0.7 → HIGH
- LOW never escalates.

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

    # From v2 risk model
    completion_risk_score: float
    risk_reliable: bool          # False → neutral fallback, do not escalate
    risk_model_version: str      # "xgboost_v2_auc0799_22k_trials_realAACT"

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

### ScanReport (output/report.py)
The full scan output written to `output/reports/<scan_id>.json` and
`output/demo/*.json`:

```python
@dataclass
class ScanReport:
    scan_id: str
    timestamp: str
    scope: str
    scope_filters: dict
    data_source_note: str        # AACT-as-EDC framing
    trials_scanned: int
    checks_executed: list[str]
    agent_iterations: int
    issues: dict                 # {critical, high, medium, low} counts
    llmops_summary: dict         # from LLMTracker.get_summary()
    prioritised_issues: list[PrioritisedIssue]
    root_cause_clusters: list[dict]   # produced by node_cluster (see below)
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
MAX_ISSUES_PER_CHECK: int = 200
MAX_TOTAL_ISSUES: int = 500

class DataQualityState(TypedDict):
    # Scope
    scope: str
    scope_filters: dict

    # Planning
    plan: list[str]
    checks_completed: list[str]

    # Scan output — Issue.to_dict() payloads, not dataclasses,
    # so the whole state stays JSON-serialisable
    issues_found: list[dict]
    risk_scores: dict[str, dict]   # nct_id -> risk payload from risk_scorer

    # Review + context loop
    needs_deeper_investigation: list[str]
    context_results: list[dict]
    iteration: int                 # capped at 3

    # LLMOps
    llm_calls: list[dict]
    total_tokens: int
    total_latency_ms: float
    estimated_cost: float

    # Final output
    prioritised_issues: list[dict]
    pattern_library_updates: list[dict]   # not currently populated (cut)
    final_report: dict             # holds root_cause_clusters, llmops_summary, etc.
```

`make_initial_state(scope, scope_filters)` returns a fresh state with
empty collections and zeroed counters.

### Agent Flow — 6 Nodes (agents/orchestrator.py)

Graph wiring:

```
START → plan → scan → review ─┬─→ context → review (loop, max 3)
                              └─→ prioritise → cluster → END
```

1. **PLANNING NODE** (`node_plan`) — LLM decides which checks to run across all 3 MCP servers based on scope. Not hardcoded. `purpose="planning"`.

2. **SCAN NODE** (`node_scan`) — Executes planned checks via MCP tool calls. AACT checks A-E + cross-source checks F-G if relevant. Attaches v2 risk score (with `risk_reliable` flag). Caps enforced via `MAX_ISSUES_PER_CHECK=200` and `MAX_TOTAL_ISSUES=500` in `agents/state.py`.

3. **REVIEW NODE** (`node_review`) — LLM reviews all findings. Decides if any need deeper multi-source investigation. `purpose="review"`.

4. **CONTEXT ENRICHMENT NODE** (`node_context`) — For flagged trials: AACT (eligibility, related trials) + PubMed (publications) + OpenFDA (adverse events). Loops back to REVIEW max 3x. `purpose="context"` (per-fetch).

5. **PRIORITISATION NODE** (`node_prioritise`) — LLM generates final per-issue assessment (severity, explanation, action) using full multi-source context + risk scores. **Async-batched** via `_call_llm_batch_async` to keep wall-clock latency low for large issue sets. `purpose="prioritisation"`.

6. **CLUSTER NODE** (`node_cluster`) — One LLM call groups prioritised issues into root-cause clusters. Output lands in `state["final_report"]["root_cause_clusters"]` (list of `{cluster_id, root_cause, pattern, affected_trials, issue_indices, cluster_severity, recommended_action}`). Falls back to empty list on LLM failure so the scan still produces a saveable report. `purpose="clustering"`.

> Note: `state` does not have a top-level `clusters` field; clusters live inside `final_report` so they ride along with the report serialisation.

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
        # purpose: "planning" | "review" | "context"
        #        | "prioritisation" | "clustering"

    def get_summary(self) -> dict:
        # total_calls, total_tokens, total_latency_ms,
        # total_estimated_cost_usd, calls list
```

Per-model cost table (USD per 1M tokens, input / output):

| Model | Input | Output |
|---|---|---|
| `gpt-4o-mini` (default) | $0.15 | $0.60 |
| `gpt-4o` | $2.50 | $10.00 |
| `gpt-4.1-mini` | $0.40 | $1.60 |

Unknown models log once and cost as $0 so tracking is never blocked.

---

## LLM Prompts

Prompts live as module-level constants in `agents/orchestrator.py`:

| Constant | Used by | Purpose |
|---|---|---|
| `PLANNING_SYSTEM` / `PLANNING_USER_TEMPLATE` | `node_plan` | Pick checks for the scope |
| `REVIEW_SYSTEM` / `REVIEW_USER_TEMPLATE` | `node_review` | Decide if findings need deeper investigation |
| `PRIORITISATION_SYSTEM` / `BATCH_PRIORITISATION_USER_TEMPLATE` | `node_prioritise` | Per-issue severity, explanation, action — async-batched |
| `CLUSTER_SYSTEM` / `CLUSTER_USER_TEMPLATE` | `node_cluster` | Group prioritised issues into 3-7 root-cause clusters |

Prioritisation prompt sketch (real version cites the v2 risk model
version string and respects `risk_reliable` rather than hard-coding
an AUC):

```
Trial: {trial_id}
Trial metadata: {trial_metadata}
Findings: {findings}
Completion risk: {risk_score}/1.0  (reliable={risk_reliable},
                                    model={risk_model_version})
PubMed context: {pubmed_context}
OpenFDA context: {openfda_context}

For each finding emit:
1. severity: CRITICAL/HIGH/MEDIUM/LOW
   (CRITICAL = HIGH + risk_reliable AND risk>=0.6
              OR cross-source discrepancy across 2+ sources)
2. explanation: 2-3 sentences, no jargon, cite sources used
3. potential_impact
4. suggested_action
5. confidence: 0.0-1.0 (higher when multiple sources agree)

JSON array. No markdown, no preamble.
```

Cluster prompt produces 3-7 clusters, each:
`{cluster_id, root_cause, pattern, affected_trials, issue_indices,
cluster_severity, recommended_action}`. Issues without a meaningful
shared cause go into a single `RC_UNCLUSTERED` group.

---

## Streamlit Dashboard

Two entry points share a single set of helpers:

| File | Mode | Runtime requirements |
|---|---|---|
| [app_v2.py](app_v2.py) | **Live** — runs the LangGraph agent | OpenAI key, AACT parquets, model binaries |
| [app_demo.py](app_demo.py) | **Demo** — reads pre-computed reports | None — boots clean on Streamlit Cloud with no secrets, no parquet, no `.pkl` |
| [app_lib.py](app_lib.py) | Shared UI helpers (severity badges, scan-state guards) — imported by both |

`app_demo.py` is what's deployed publicly. It loads the three
pre-computed `output/demo/*.json` reports built by `pre_warm_demo.py`
(recruiting Phase 3, oncology Phase 3, all Phase 3) and renders them
through the same page functions as live mode. Streamlit Cloud safety
is verified: no module-level imports from `agents/`, `checks/`,
`mcp_servers/`, or `models/`.

Persistent banner ALL pages (rendered by `_render_demo_banner` in demo
mode and `render_banner` in live mode):

```
⚠️ Data sources: AACT registry + PubMed + OpenFDA (proxies for EDC/CTMS).
Same agent architecture applies to live trial execution data.
```

### Page 1: Scan Control & Overview
- Scope selectors + data source availability indicators (AACT ✅ PubMed ✅ OpenFDA ✅)
- Run Scan button + progress (live mode); pre-computed scope picker (demo mode)
- Summary cards: trials scanned, CRITICAL/HIGH/MEDIUM/LOW
- LLMOps card: calls, tokens, cost, latency
- Cluster preview: top root-cause groups from `final_report["root_cause_clusters"]`

### Page 2: Prioritised Issues List
- Table: Rank, Trial ID, Severity, Check, Sources Used, Explanation, Risk Score, Confidence
- CRITICAL rows: red banner
- Cross-source badge: "AACT + PubMed" or "AACT + FDA"
- **Two-layer display** — Layer 1: severity badge + finding + suggested
  action (visible immediately). Layer 2 (click to expand): "Why this
  severity?", sources checked, assessment confidence, full provenance.
- Filter sidebar (severity, check category, source)
- Export JSON button

### Page 3: Trial Deep Dive (NEVER CUT)
- Trial metadata (AACT)
- **v2 risk model: score + key risk factors + `risk_reliable` flag** ← most differentiated
- PubMed panel: publications found
- OpenFDA panel: adverse event summary
- Eligibility criteria text

### Page 4: Audit Log & Evaluation
- Audit tab: every agent action, MCP calls, LLM calls per `call_id` with tokens / cost / latency, exportable JSON
- Data sources tab: MCP server status, response times, cache hits
- Evaluation tab: gold-set metrics summary card (loads `evaluation/metrics_results.json`)
- Pattern library section: gated on `output/patterns.json` existing — silently absent in current build (pattern library was cut)

---

## Evaluation

### Gold Set — 18 issues, single annotator
Stored as `evaluation/gold_set.json`, loaded by `evaluation/gold_set.py`.
Built from real agent findings (recruiting Phase 3 scope) and includes
cross-source findings (Check F / Check G).

### Metrics
Precision, recall, F1 per severity + overall accuracy + Cohen's κ.
Run by `evaluation/metrics.py`; last results pinned at
`evaluation/metrics_results.json`:

| Metric | Value |
|---|---|
| Match rate (detection) | 1.000 (18/18) |
| Accuracy (severity) | 0.722 |
| Cohen's κ | 0.596 (moderate agreement) |
| **CRITICAL precision** | **1.000** (4 TP, 0 FP) — no false escalations |
| HIGH precision / recall | 0.444 / 1.000 |
| MEDIUM precision / recall | 1.000 / 0.556 |

Known weakness: system over-classifies moderate recruiting delays as
HIGH (should be MEDIUM). Production validation would need 2-3 annotators
with adjudication and an independently-labelled detection-recall set.

---

## Never Cut List (kept in shipped build)

1. Agent loop — all 6 nodes + max 3 review iterations
2. All 3 MCP servers (AACT + PubMed + OpenFDA)
3. LLMOps logging on every LLM call (5 purposes incl. `clustering`)
4. Multi-source provenance on every issue
5. v2 risk score + reliability gating + CRITICAL escalation
6. Page 3 trial deep dive
7. AACT vs EDC framing on every page
8. Structured JSON export with audit log
9. Demo mode (`app_demo.py` + pre-computed `output/demo/*.json`)

## Cuts Actually Taken

1. ✂️ **Pattern library** (`output/patterns.json`) — never built. `app_demo.py` reads it defensively (silent absence).
2. ✂️ **Page 3 related-trials panel** — collapsed into Page 3 Deep Dive proper.
3. ✂️ **AACT MCP `search_trials` and `get_trial_results`** — agent never needed them; scopes flow directly through `run_quality_check`, results were folded into `get_trial_details`.
4. ✂️ **Enrollment velocity check** (BONUS) — not built.

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

## Build Order — what shipped

| Phase | Files | Status |
|---|---|---|
| 1 | checks/models.py + checks A-E | ✅ shipped |
| 2 | models/risk_scorer.py (v2 retrain, AUC=0.799) | ✅ shipped |
| 3 | mcp_servers/aact_server.py (5 tools — see MCP 1 spec) | ✅ shipped |
| 4 | mcp_servers/pubmed_server.py | ✅ shipped |
| 5 | mcp_servers/openfda_server.py | ✅ shipped |
| 6 | checks/publication.py + checks/safety.py | ✅ shipped |
| 7 | agents/state.py + agents/orchestrator.py | ✅ shipped (6 nodes incl. cluster) |
| 8 | llmops/tracker.py + LLM prioritisation | ✅ shipped |
| 9 | output/report.py + multi-source provenance | ✅ shipped |
| 10 | output/patterns.json | ✂️ cut |
| 11 | evaluation/gold_set.json + metrics.py | ✅ shipped (18-sample gold set) |
| 12 | app_v2.py (4 pages) | ✅ shipped |
| 13 | app_demo.py + app_lib.py + pre_warm_demo.py | ✅ added (demo deploy path) |
| 14 | Test suites (agents, checks, llmops, models, output, tests/) | ✅ shipped |
| BONUS | Enrollment velocity check | ✂️ not built |

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

[LLMOps — read live from LLMTracker.get_summary() at demo time]
"6 LLM calls, ~5.5k tokens, ~$0.0017, 43.8s." (illustrative —
exact numbers vary per scope; pull from final_report["llmops_summary"])

[EVALUATION]
"18-sample gold set: 100% match rate (every gold issue surfaced),
72% accuracy on severity, Cohen's κ 0.596, and CRITICAL precision
1.000 — zero false escalations on the highest-priority class."

[ROOT-CAUSE CLUSTERS]
"Beyond per-issue findings, the agent groups them into 3-7 root-cause
clusters — so reviewers see 'these 14 trials share the same eligibility
cliff' instead of 14 disconnected tickets."
```

---

## Cover Note to Rivia

> "I built a proactive data quality agent that mirrors what your CEO
> described as Rivia's key product bet. It unifies AACT registry,
> PubMed, and OpenFDA into coherent findings — surfacing issues
> no single-source system can detect. It includes a v2 XGBoost
> completion-risk model (AUC=0.799 on 22K real AACT trials) with
> reliability-gated escalation, a 6-node LangGraph agent that
> clusters findings into root causes, full LLMOps instrumentation,
> and auditable multi-source provenance on every finding. Live demo
> ships pre-computed reports so the architecture is visible without
> exposing API keys. Happy to walk through it on a call."
