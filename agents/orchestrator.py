"""5-node LangGraph orchestrator for the DecisionLENS v2 agent.

Flow
----
    PLAN → SCAN → REVIEW ─cond─→ PRIORITISE → END
                     ↑              │
                     └── CONTEXT ←──┘  (when investigate list non-empty
                                         AND iteration < MAX_ITERATIONS)

Each node receives the full ``DataQualityState`` and returns a partial
update dict. LangGraph merges the updates.

Design notes
------------
* We import MCP ``tool_*`` functions directly (no stdio round-trip) —
  the protocol envelope is preserved but calls are in-process for
  latency and easier error inspection.
* Every LLM call goes through ``_call_llm`` which records one entry in
  ``state["llm_calls"]`` and updates the running totals. Phase 8 will
  lift this helper into ``llmops/tracker.py`` without changing the
  call sites.
* LLM failures degrade gracefully — planning falls back to "run all
  applicable checks", review falls back to "no deeper investigation",
  prioritisation falls back to deterministic severity using the rule
  engine's value plus ``escalate_severity``.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import time
from typing import Any, Callable

from dotenv import load_dotenv
from langgraph.graph import END, START, StateGraph
from openai import AsyncOpenAI, OpenAI

from agents.state import (
    MAX_ISSUES_PER_CHECK,
    MAX_TOTAL_ISSUES,
    DataQualityState,
)
from checks.demo_overrides import DEMO_DRUG_MAP
from checks.publication import check_publication_consistency
from checks.safety import check_safety_cross_validation, _drug_names_for_trial
from llmops.tracker import LLMTracker
from mcp_servers.aact_server import (
    tool_get_eligibility_criteria,
    tool_get_related_trials,
    tool_get_trial_details,
    tool_run_quality_check,
)
from mcp_servers.openfda_server import tool_get_drug_safety_summary
from mcp_servers.pubmed_server import tool_search_trial_publications
from models.risk_scorer import (
    RISK_MODEL_VERSION,
    escalate_severity,
    get_batch_risk_scores,
)

load_dotenv()

log = logging.getLogger(__name__)

LLM_MODEL = "gpt-4o-mini"
# Cost table lives in llmops.tracker.MODEL_COSTS.

MAX_ITERATIONS = 3
MAX_INVESTIGATE_TRIALS = 5
# Keep at most this many issues per rule (check_name) across the whole
# scan, sorted by risk score descending. Stricter than the earlier
# PER_CHECK_TYPE_SAMPLE_* logic and applied globally, not per-check.
MAX_PER_CHECK_NAME = 5
# Prioritisation batches this many trials per LLM call. Larger = fewer
# calls + less latency + cheaper; smaller = easier for the model to
# follow the JSON contract. 8 is the Phase 8 tuning target.
BATCH_SIZE = 8

# Identifiers we accept in planning output → canonical check names.
CHECK_ALIASES: dict[str, str] = {
    "a": "temporal", "temporal": "temporal",
    "b": "enrollment", "enrollment": "enrollment",
    "c": "status", "status": "status",
    "d": "endpoint", "endpoints": "endpoint", "endpoint": "endpoint",
    "e": "crossfield", "crossfield": "crossfield",
    "f": "publication", "publication": "publication",
    "g": "safety", "safety": "safety",
}

DEFAULT_PLAN: list[str] = [
    "temporal", "enrollment", "status", "endpoint", "crossfield",
    "publication", "safety",
]


# ---------------------------------------------------------------------------
# LLM + tracking
# ---------------------------------------------------------------------------

_client: OpenAI | None = None
_async_client: AsyncOpenAI | None = None

# Scan-scoped LLMTracker. ``run_scan`` replaces this with a fresh
# instance at the start of every scan; node helpers below read it via
# ``_get_tracker()``.
_tracker: LLMTracker | None = None


def _get_client() -> OpenAI:
    global _client
    if _client is None:
        if not os.getenv("OPENAI_API_KEY"):
            raise RuntimeError("OPENAI_API_KEY is not set")
        _client = OpenAI()
    return _client


def _get_async_client() -> AsyncOpenAI:
    global _async_client
    if _async_client is None:
        if not os.getenv("OPENAI_API_KEY"):
            raise RuntimeError("OPENAI_API_KEY is not set")
        _async_client = AsyncOpenAI()
    return _async_client


def _get_tracker() -> LLMTracker:
    """Lazily create a tracker if one isn't already set by run_scan()."""
    global _tracker
    if _tracker is None:
        _tracker = LLMTracker()
    return _tracker


def _record_call(
    state: DataQualityState, record: dict,
) -> None:
    """Mirror a tracker record into the live state for in-flight nodes."""
    state["llm_calls"].append(record)
    state["total_tokens"] += (
        record["prompt_tokens"] + record["completion_tokens"]
    )
    state["total_latency_ms"] += record["latency_ms"]
    state["estimated_cost"] += record["estimated_cost_usd"]


def _call_llm(
    state: DataQualityState,
    *,
    purpose: str,
    system: str,
    user: str,
    force_json: bool = True,
) -> dict | None:
    """Call OpenAI, record via LLMTracker, return parsed JSON (or None)."""
    tracker = _get_tracker()
    started = time.perf_counter()
    try:
        client = _get_client()
        kwargs: dict[str, Any] = {
            "model": LLM_MODEL,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            "temperature": 0.2,
        }
        if force_json:
            kwargs["response_format"] = {"type": "json_object"}
        resp = client.chat.completions.create(**kwargs)
        latency_ms = (time.perf_counter() - started) * 1000.0

        usage = getattr(resp, "usage", None)
        pt = getattr(usage, "prompt_tokens", 0) if usage else 0
        ct = getattr(usage, "completion_tokens", 0) if usage else 0

        record = tracker.track_call(
            model=LLM_MODEL, purpose=purpose,
            prompt_tokens=pt, completion_tokens=ct,
            latency_ms=latency_ms,
        )
        _record_call(state, record)

        content = resp.choices[0].message.content or "{}"
        try:
            return json.loads(content)
        except json.JSONDecodeError as exc:
            log.warning("LLM %s returned non-JSON content: %s", purpose, exc)
            return None
    except Exception as exc:  # noqa: BLE001
        latency_ms = (time.perf_counter() - started) * 1000.0
        log.exception("LLM call (%s) failed: %s", purpose, exc)
        record = tracker.track_call(
            model=LLM_MODEL, purpose=purpose,
            prompt_tokens=0, completion_tokens=0,
            latency_ms=latency_ms, status="error", error=str(exc),
        )
        _record_call(state, record)
        return None


async def _call_llm_batch_async(
    state: DataQualityState, user_prompt: str,
) -> dict | None:
    """Async twin of ``_call_llm`` for prioritisation batches.

    Note: when batches run concurrently ``state["total_latency_ms"]``
    remains the *sum* of per-call latencies (i.e., total compute time)
    — not wall time. Wall time is tracked by the caller via
    ``time.perf_counter()`` around the ``asyncio.run`` invocation.
    """
    tracker = _get_tracker()
    started = time.perf_counter()
    try:
        client = _get_async_client()
        resp = await client.chat.completions.create(
            model=LLM_MODEL,
            messages=[
                {"role": "system", "content": PRIORITISATION_SYSTEM},
                {"role": "user", "content": user_prompt},
            ],
            temperature=0.2,
            response_format={"type": "json_object"},
        )
        latency_ms = (time.perf_counter() - started) * 1000.0

        usage = getattr(resp, "usage", None)
        pt = getattr(usage, "prompt_tokens", 0) if usage else 0
        ct = getattr(usage, "completion_tokens", 0) if usage else 0

        record = tracker.track_call(
            model=LLM_MODEL, purpose="prioritisation",
            prompt_tokens=pt, completion_tokens=ct,
            latency_ms=latency_ms,
        )
        _record_call(state, record)

        content = resp.choices[0].message.content or "{}"
        try:
            return json.loads(content)
        except json.JSONDecodeError as exc:
            log.warning("async batch returned non-JSON content: %s", exc)
            return None
    except Exception as exc:  # noqa: BLE001
        latency_ms = (time.perf_counter() - started) * 1000.0
        log.exception("async batch LLM call failed: %s", exc)
        record = tracker.track_call(
            model=LLM_MODEL, purpose="prioritisation",
            prompt_tokens=0, completion_tokens=0,
            latency_ms=latency_ms, status="error", error=str(exc),
        )
        _record_call(state, record)
        return None


# ---------------------------------------------------------------------------
# Check dispatch
# ---------------------------------------------------------------------------


def _run_single_source_check(check_name: str, filters: dict) -> list[dict]:
    """Execute one of the A–E checks via the AACT MCP tool."""
    resp = tool_run_quality_check(
        check_name=check_name,
        therapeutic_area=filters.get("therapeutic_area"),
        phase=filters.get("phase"),
        overall_status=filters.get("overall_status"),
        limit=int(filters.get("limit", 100)),
    )
    if resp.get("status") != "ok":
        log.warning("%s: %s", check_name, resp.get("reason"))
        return []
    return resp.get("issues") or []


def _run_cross_source_check(check_name: str, filters: dict) -> list[dict]:
    """Execute one of the F/G checks and convert Issue objects to dicts."""
    fn: Callable[..., list] = (
        check_publication_consistency
        if check_name == "publication"
        else check_safety_cross_validation
    )
    issues = fn(
        therapeutic_area=filters.get("therapeutic_area"),
        phase=filters.get("phase"),
        overall_status=filters.get("overall_status"),
        limit=int(filters.get("limit", 100)),
    )
    return [i.to_dict() for i in issues]


def _dispatch_check(check_name: str, filters: dict) -> list[dict]:
    if check_name in {"publication", "safety"}:
        return _run_cross_source_check(check_name, filters)
    return _run_single_source_check(check_name, filters)


# ---------------------------------------------------------------------------
# Risk attachment + sampling
# ---------------------------------------------------------------------------


def _attach_risk(
    issues: list[dict], state: DataQualityState,
) -> None:
    """Batch-score the trial_ids in ``issues`` and mutate each issue in place.

    Adds ``risk_score``, ``risk_reliable``, ``risk_model_version``,
    ``severity`` (post-escalation), and stashes the payload in
    ``state["risk_scores"]`` for reuse in prioritisation.
    """
    trial_ids = list({i["trial_id"] for i in issues})
    if not trial_ids:
        return
    scores = get_batch_risk_scores(trial_ids)
    state["risk_scores"].update(scores)
    for issue in issues:
        payload = scores.get(issue["trial_id"])
        if not payload:
            continue
        issue["risk_score"] = payload["risk_score"]
        issue["risk_reliable"] = payload["risk_reliable"]
        issue["risk_model_version"] = payload["risk_model_version"]
        issue["severity"] = escalate_severity(
            issue.get("severity_rule", "LOW"), payload,
        )


def _apply_per_check_cap(
    issues: list[dict], state: DataQualityState,
) -> tuple[list[dict], int, int]:
    """Keep the top ``MAX_PER_CHECK_NAME`` issues per ``check_name``.

    Sort is by ``risk_score`` descending (falls back to 0 when a score
    could not be attached). Every truncated check emits one audit
    entry into ``state["pattern_library_updates"]``.

    Returns ``(kept, n_checks_capped, n_issues_dropped)``.
    """
    buckets: dict[str, list[dict]] = {}
    for issue in issues:
        buckets.setdefault(issue.get("check_name", "?"), []).append(issue)

    kept: list[dict] = []
    n_checks_capped = 0
    n_dropped = 0
    for check_name, group in buckets.items():
        if len(group) > MAX_PER_CHECK_NAME:
            group_sorted = sorted(
                group, key=lambda i: i.get("risk_score", 0.0), reverse=True,
            )
            kept.extend(group_sorted[:MAX_PER_CHECK_NAME])
            dropped = len(group) - MAX_PER_CHECK_NAME
            n_dropped += dropped
            n_checks_capped += 1
            note = (
                f"{check_name}: {len(group)} issues found, "
                f"kept top {MAX_PER_CHECK_NAME} by risk score"
            )
            state["pattern_library_updates"].append({
                "source": "scan_cap",
                "check_name": check_name,
                "found": len(group),
                "kept": MAX_PER_CHECK_NAME,
                "note": note,
            })
            log.info("scan_cap: %s", note)
        else:
            kept.extend(group)
    return kept, n_checks_capped, n_dropped


# ---------------------------------------------------------------------------
# Node 1 — PLANNING
# ---------------------------------------------------------------------------


PLANNING_SYSTEM = (
    "You are planning a clinical trial data quality scan. "
    "Return JSON only — no markdown, no preamble."
)

PLANNING_USER_TEMPLATE = """\
Scope: {scope}
Scope filters: {filters}
Available checks: temporal(A), enrollment(B), status(C), endpoints(D), \
crossfield(E), publication(F, requires PubMed), safety(G, requires FDA)

Select which checks to run for this scope and explain why.
Return JSON: {{"checks": ["temporal", ...], "rationale": "one paragraph"}}
"""


def node_plan(state: DataQualityState) -> dict:
    """Ask the LLM which checks to run for the scope."""
    log.info("node_plan: scope=%r", state["scope"])
    user = PLANNING_USER_TEMPLATE.format(
        scope=state["scope"], filters=json.dumps(state["scope_filters"]),
    )
    try:
        parsed = _call_llm(
            state, purpose="planning", system=PLANNING_SYSTEM, user=user,
        )
    except Exception as exc:  # noqa: BLE001 — node-level graceful degradation
        log.exception("node_plan: LLM raised unexpectedly: %s", exc)
        parsed = None

    plan: list[str] = []
    rationale: str | None = None
    if parsed and isinstance(parsed.get("checks"), list):
        for raw in parsed["checks"]:
            key = str(raw).strip().lower()
            canonical = CHECK_ALIASES.get(key)
            if canonical and canonical not in plan:
                plan.append(canonical)
        rationale = parsed.get("rationale")

    if not plan:
        log.warning("planning returned no usable checks; falling back to all 7")
        plan = list(DEFAULT_PLAN)

    state.setdefault("pattern_library_updates", [])
    if rationale:
        # Stash the rationale in-state for the final report. Pattern
        # library persistence is Phase 10; we log an ephemeral note.
        state["pattern_library_updates"].append({
            "source": "planning", "rationale": rationale,
        })
    return {"plan": plan}


# ---------------------------------------------------------------------------
# Node 2 — SCAN
# ---------------------------------------------------------------------------


def node_scan(state: DataQualityState) -> dict:
    """Run each planned check, batch-score all findings, apply caps.

    Flow:
      1. Run every planned check, collecting raw issues (no risk yet).
      2. Batch-risk-score all trial_ids in a single call and attach
         ``risk_score`` / ``risk_reliable`` / escalated ``severity``
         to every issue.
      3. Cap per ``check_name`` to top ``MAX_PER_CHECK_NAME`` by risk.
      4. Apply the global ``MAX_TOTAL_ISSUES`` safety cap.
    """
    log.info("node_scan: plan=%s", state["plan"])
    filters = dict(state["scope_filters"])
    raw: list[dict] = []
    completed: list[str] = []

    # --- Step 1: collect from every planned check --------------------
    for check_name in state["plan"]:
        per_check_filters = dict(filters)
        per_check_filters["limit"] = min(
            int(per_check_filters.get("limit", 100)), MAX_ISSUES_PER_CHECK,
        )
        issues = _dispatch_check(check_name, per_check_filters)
        completed.append(check_name)
        log.info("scan: %s -> %d raw issues", check_name, len(issues))
        if issues:
            raw.extend(issues)
    before_cap = len(raw)

    # --- Step 2: single batch risk-scoring ---------------------------
    if raw:
        _attach_risk(raw, state)

    # --- Step 3: per-check-name cap ----------------------------------
    kept, n_checks_capped, n_dropped = _apply_per_check_cap(raw, state)

    # --- Step 4: global safety cap -----------------------------------
    globally_dropped = 0
    if len(kept) > MAX_TOTAL_ISSUES:
        kept = sorted(
            kept, key=lambda i: i.get("risk_score", 0.0), reverse=True,
        )[:MAX_TOTAL_ISSUES]
        globally_dropped = before_cap - len(kept) - n_dropped

    log.info(
        "scan: %d raw → %d kept across %d check(s); "
        "per-check cap applied: %d check(s) truncated, %d issues dropped",
        before_cap, len(kept), len(completed), n_checks_capped, n_dropped,
    )
    state["pattern_library_updates"].append({
        "source": "scan_summary",
        "before_cap": before_cap,
        "after_cap": len(kept),
        "checks_capped": n_checks_capped,
        "issues_dropped": n_dropped + globally_dropped,
    })
    return {"issues_found": kept, "checks_completed": completed}


# ---------------------------------------------------------------------------
# Node 3 — REVIEW
# ---------------------------------------------------------------------------


REVIEW_SYSTEM = (
    "You are triaging automated clinical trial data quality findings. "
    "Return JSON only."
)

REVIEW_USER_TEMPLATE = """\
Review these {n} data quality issues found across {n_trials} trials.
Which trials warrant deeper cross-source investigation? Consider:
severity, risk score, issue combinations on the same trial.

Issues summary (trial_id → list of (check_name, severity, risk)):
{summary}

Return JSON: {{"investigate": ["NCTxxxx", ...], "rationale": "one paragraph"}}
Limit to top {max_trials} trials maximum.
"""


def _issue_summary_for_review(issues: list[dict], max_lines: int = 60) -> str:
    """Compact per-trial roll-up for the review prompt."""
    by_trial: dict[str, list[str]] = {}
    risk_by_trial: dict[str, float] = {}
    reliable_by_trial: dict[str, bool] = {}
    for i in issues:
        trial = i.get("trial_id", "?")
        by_trial.setdefault(trial, []).append(
            f"{i.get('check_name', '?')}/"
            f"{i.get('severity') or i.get('severity_rule')}"
        )
        risk_by_trial[trial] = i.get("risk_score", 0.5)
        reliable_by_trial[trial] = i.get("risk_reliable", False)

    lines: list[str] = []
    ranked = sorted(
        by_trial.items(),
        key=lambda kv: (
            -sum(1 for x in kv[1] if "CRITICAL" in x or "HIGH" in x),
            -risk_by_trial.get(kv[0], 0.0),
        ),
    )
    for trial, items in ranked[:max_lines]:
        rel = "T" if reliable_by_trial.get(trial) else "F"
        risk = risk_by_trial.get(trial, 0.5)
        lines.append(f"  {trial} risk={risk:.2f}({rel}) {items}")
    return "\n".join(lines) if lines else "(none)"


def node_review(state: DataQualityState) -> dict:
    """Ask the LLM which trials deserve deeper multi-source context."""
    state["iteration"] += 1
    issues = state["issues_found"]
    n_trials = len({i.get("trial_id") for i in issues})
    log.info(
        "node_review iter=%d: %d issues across %d trials",
        state["iteration"], len(issues), n_trials,
    )

    if not issues:
        return {"needs_deeper_investigation": [], "iteration": state["iteration"]}

    user = REVIEW_USER_TEMPLATE.format(
        n=len(issues), n_trials=n_trials,
        summary=_issue_summary_for_review(issues),
        max_trials=MAX_INVESTIGATE_TRIALS,
    )
    try:
        parsed = _call_llm(
            state, purpose="review", system=REVIEW_SYSTEM, user=user,
        )
    except Exception as exc:  # noqa: BLE001 — node-level graceful degradation
        log.exception("node_review: LLM raised unexpectedly: %s", exc)
        parsed = None

    trial_ids_in_issues = {i["trial_id"] for i in issues}
    already_enriched = {
        ctx.get("trial_id") for ctx in state.get("context_results", [])
    }
    investigate: list[str] = []
    if parsed and isinstance(parsed.get("investigate"), list):
        for candidate in parsed["investigate"]:
            tid = str(candidate).strip()
            if (
                tid in trial_ids_in_issues
                and tid not in already_enriched
                and tid not in investigate
            ):
                investigate.append(tid)
            if len(investigate) >= MAX_INVESTIGATE_TRIALS:
                break

    return {
        "needs_deeper_investigation": investigate,
        "iteration": state["iteration"],
    }


# ---------------------------------------------------------------------------
# Node 4 — CONTEXT ENRICHMENT
# ---------------------------------------------------------------------------


def _drug_name_for_trial(nct_id: str) -> str | None:
    """Resolve a drug name via DEMO_DRUG_MAP first, then interventions."""
    names = _drug_names_for_trial(nct_id)
    return names[0] if names else None


def node_context(state: DataQualityState) -> dict:
    """Enrich each flagged trial with AACT + PubMed + OpenFDA context."""
    investigate = state.get("needs_deeper_investigation", [])
    log.info("node_context: enriching %d trial(s)", len(investigate))
    new_results: list[dict] = []

    for nct_id in investigate:
        entry: dict[str, Any] = {
            "trial_id": nct_id,
            "aact": {},
            "pubmed": {},
            "openfda": {},
        }
        try:
            entry["aact"]["trial_details"] = tool_get_trial_details(nct_id)
            entry["aact"]["eligibility"] = tool_get_eligibility_criteria(nct_id)
            entry["aact"]["related"] = tool_get_related_trials(
                nct_id, limit=5,
            )
        except Exception as exc:  # noqa: BLE001
            log.warning("aact enrichment failed for %s: %s", nct_id, exc)

        try:
            entry["pubmed"]["publications"] = tool_search_trial_publications(
                nct_id, max_results=5,
            )
        except Exception as exc:  # noqa: BLE001
            log.warning("pubmed enrichment failed for %s: %s", nct_id, exc)

        drug = _drug_name_for_trial(nct_id)
        if drug:
            try:
                entry["openfda"]["drug_name"] = drug
                entry["openfda"]["safety_summary"] = tool_get_drug_safety_summary(
                    drug,
                )
            except Exception as exc:  # noqa: BLE001
                log.warning(
                    "openfda enrichment failed for %s/%s: %s", nct_id, drug, exc,
                )
        else:
            entry["openfda"]["note"] = "no drug intervention mapped"

        new_results.append(entry)

    all_results = list(state.get("context_results", [])) + new_results
    return {"context_results": all_results, "needs_deeper_investigation": []}


def route_from_review(state: DataQualityState) -> str:
    """Conditional edge: loop back to context if work remains and we have budget."""
    if state["iteration"] >= MAX_ITERATIONS:
        return "prioritise"
    if not state.get("needs_deeper_investigation"):
        return "prioritise"
    return "context"


# ---------------------------------------------------------------------------
# Node 5 — PRIORITISATION
# ---------------------------------------------------------------------------


PRIORITISATION_SYSTEM = (
    "You are a clinical trial data quality analyst reviewing automated "
    "findings from AACT registry, PubMed literature, and FDA adverse "
    "event database. Return JSON only — no markdown, no preamble."
)

# Batched prompt — one LLM call covers up to BATCH_SIZE trials. The
# model must echo ``trial_id`` + ``check_name`` on every response entry
# so we can route each LLM finding back to its source issue.
BATCH_PRIORITISATION_USER_TEMPLATE = """\
You are a clinical trial data quality analyst.
Sources: AACT + PubMed + OpenFDA.

Analyse these {n} trials and return one JSON object per finding.

Trials and findings:
{batch_findings}

For each finding provide:
trial_id, check_name, severity (CRITICAL/HIGH/MEDIUM/LOW),
explanation (2-3 sentences, no jargon),
potential_impact, suggested_action,
confidence (0.0-1.0)

confidence = how confident YOU are in your severity
assessment given the evidence. Higher when multiple
sources agree, when the rule fired clearly, or when
the data_points strongly support the severity. Lower
when sources disagree, evidence is thin, or the call
is borderline. Default to 0.7 for a clear single-source
finding, 0.85+ when 2+ sources agree, 0.5 or below
when evidence is ambiguous. Do NOT echo the risk score
shown above — confidence is about your judgement, not
the v1 risk model.

CRITICAL = HIGH + risk>=0.6 + reliable=True
OR cross-source discrepancy 2+ sources

Return a JSON object: {{"findings": [ ... one entry per finding ... ]}}
No markdown, no preamble.
"""


def _render_trial_block(
    trial_id: str, findings: list[dict], risk: dict, ctx: dict,
) -> str:
    """Render one trial's findings + context as a block for the batch prompt."""
    lines = [
        f"Trial {trial_id}  risk={risk['risk_score']:.2f} "
        f"reliable={risk['risk_reliable']}",
        f"  pubmed : {_compact_pubmed(ctx)}",
        f"  openfda: {_compact_openfda(ctx)}",
        "  findings:",
    ]
    for f in findings:
        lines.append(
            f"    - check_name={f.get('check_name')!r} "
            f"severity_rule={f.get('severity_rule')} "
            f"severity_post_risk={f.get('severity')} "
            f"finding={f.get('finding')!r} "
            f"data_points={json.dumps(f.get('data_points') or {}, default=str)}"
        )
    return "\n".join(lines)


def _context_for(state: DataQualityState, nct_id: str) -> dict:
    """Find the CONTEXT node's entry for a trial (or empty dict)."""
    for ctx in state.get("context_results", []):
        if ctx.get("trial_id") == nct_id:
            return ctx
    return {}


def _compact_pubmed(ctx: dict) -> str:
    pubmed = (ctx.get("pubmed") or {}).get("publications") or {}
    if pubmed.get("status") != "ok":
        return "unavailable"
    total = pubmed.get("total_found", 0)
    sample = (pubmed.get("publications") or [])[:3]
    titles = [p.get("title", "?") for p in sample if p.get("title")]
    return f"total_found={total}; sample_titles={titles}"


def _compact_openfda(ctx: dict) -> str:
    fda = ctx.get("openfda") or {}
    if not fda:
        return "not queried"
    summary = fda.get("safety_summary")
    if not summary or summary.get("status") != "ok":
        return fda.get("note") or "unavailable"
    return (
        f"drug={fda.get('drug_name')}; total_reports={summary.get('total_reports')}"
        f"; serious_pct={summary.get('serious_pct')}"
        f"; death_pct={summary.get('death_pct')}"
    )


def _sources_used(findings: list[dict], ctx: dict) -> list[str]:
    """Derive provenance.sources_queried from an issue's category + context."""
    cats = {f.get("check_category") for f in findings}
    sources = {"aact"}
    if "publication" in cats or (ctx.get("pubmed") or {}).get("publications"):
        sources.add("pubmed")
    if "safety" in cats or (ctx.get("openfda") or {}).get("safety_summary"):
        sources.add("openfda")
    return sorted(sources)


def _fallback_prioritisation(issue: dict) -> dict:
    """Deterministic fallback when the LLM call fails or the shape is wrong."""
    return {
        "severity": issue.get("severity") or issue.get("severity_rule", "LOW"),
        "explanation": issue.get("finding", "No LLM explanation available."),
        "potential_impact": "LLM prioritisation unavailable.",
        "suggested_action": "Review the raw finding manually.",
        "confidence": 0.3,
    }


def _deterministic_low_summary(issue: dict) -> dict:
    """Skip-LLM template for rule-engine LOW issues (saves tokens + latency).

    Applied in ``node_prioritise`` before batches are assembled. LOW
    never escalates under the risk model, so this path preserves the
    final severity accurately.
    """
    return {
        "severity": "LOW",
        "explanation": f"Minor data quality note: {issue.get('finding', '')}",
        "potential_impact": "",
        "suggested_action": "Monitor. No immediate action required.",
        "confidence": 0.70,
    }


def _neutral_risk() -> dict:
    return {
        "risk_score": 0.5, "risk_reliable": False,
        "risk_model_version": RISK_MODEL_VERSION, "note": "",
    }


def _pubmed_provenance(finding: dict, ctx: dict) -> dict:
    """Normalise PubMed provenance to the spec shape (query/results/pmids)."""
    raw = (ctx.get("pubmed") or {}).get("publications") or {}
    if isinstance(raw, dict) and raw.get("status") == "ok":
        return {
            "query": raw.get("query") or "",
            "results_found": int(raw.get("total_found", 0) or 0),
            "pmids": [
                p.get("pmid") for p in (raw.get("publications") or [])
                if p.get("pmid")
            ],
        }
    # Fallback: the publication check itself queried PubMed — pull from
    # the finding's data_points so the evidence trail is preserved.
    if finding.get("check_category") == "publication":
        dp = finding.get("data_points") or {}
        return {
            "query": f"{finding.get('trial_id', '')}[All Fields]",
            "results_found": int(dp.get("pubmed_count", 0) or 0),
            "pmids": list(dp.get("pmids_found") or []),
        }
    return {"query": "", "results_found": 0, "pmids": []}


def _openfda_provenance(finding: dict, ctx: dict) -> dict:
    """Normalise OpenFDA provenance to the spec shape (drug/totals/serious)."""
    fda = ctx.get("openfda") or {}
    drug = fda.get("drug_name") or ""
    summary = fda.get("safety_summary") or {}
    if isinstance(summary, dict) and summary.get("status") == "ok":
        return {
            "drug_queried": drug,
            "total_adverse_events": int(summary.get("total_reports", 0) or 0),
            "serious_events": int(summary.get("serious_reports", 0) or 0),
        }
    # Fallback: the safety check itself queried OpenFDA — pull from
    # data_points so provenance survives when context wasn't enriched.
    if finding.get("check_category") == "safety":
        dp = finding.get("data_points") or {}
        return {
            "drug_queried": dp.get("intervention_name") or drug,
            "total_adverse_events": int(dp.get("fda_ae_count", 0) or 0),
            "serious_events": 0,
        }
    return {"drug_queried": drug, "total_adverse_events": 0, "serious_events": 0}


def _build_prioritised_entry(
    finding: dict, trial_id: str, risk: dict, ctx: dict,
    summary: dict, sources: list[str], related: list[str],
    state: DataQualityState,
) -> dict:
    """Assemble one PrioritisedIssue from a raw finding + LLM summary."""
    provenance = {
        "sources_queried": sources,
        "sources_available": sources,
        "aact": {
            "tables": finding.get("source_tables", []),
            "columns": finding.get("source_columns", []),
            "check_rule": finding.get("check_name"),
        },
        "pubmed": _pubmed_provenance(finding, ctx),
        "openfda": _openfda_provenance(finding, ctx),
        "data_accessed_at": finding.get("timestamp"),
        "context_enriched": bool(ctx),
        "agent_iteration": state["iteration"],
        "llm_call_ids": [c["call_id"] for c in state["llm_calls"][-3:]],
        "risk_model_used": True,
        "risk_model_version": risk["risk_model_version"],
        "risk_reliable": risk["risk_reliable"],
    }
    return {
        "trial_id": trial_id,
        "check_name": finding.get("check_name"),
        "check_category": finding.get("check_category"),
        "finding": finding.get("finding"),
        "data_points": finding.get("data_points", {}),
        "completion_risk_score": risk["risk_score"],
        "risk_reliable": risk["risk_reliable"],
        "risk_model_version": risk["risk_model_version"],
        "severity": summary.get(
            "severity",
            finding.get("severity") or finding.get("severity_rule"),
        ),
        "explanation": summary.get("explanation", ""),
        "potential_impact": summary.get("potential_impact", ""),
        "suggested_action": summary.get("suggested_action", ""),
        "confidence": float(summary.get("confidence", 0.5) or 0.5),
        "related_trials": related,
        "provenance": provenance,
    }


def node_prioritise(state: DataQualityState) -> dict:
    """Generate a PrioritisedIssue for every scanned finding, batched.

    LOW issues bypass the LLM and receive a deterministic template.
    HIGH / MEDIUM / CRITICAL (by rule-engine ``severity_rule``) go
    through batched LLM prioritisation.
    """
    issues = state["issues_found"]
    n_trials = len({i['trial_id'] for i in issues}) if issues else 0
    log.info(
        "node_prioritise: %d finding(s) across %d trial(s) — BATCH_SIZE=%d",
        len(issues), n_trials, BATCH_SIZE,
    )
    prioritised: list[dict] = []
    if not issues:
        return {
            "prioritised_issues": [],
            "final_report": _build_final_report(state, []),
        }

    # Split by rule-engine severity. LOW never escalates so skipping
    # the LLM for those findings is correctness-preserving.
    low_issues = [i for i in issues if i.get("severity_rule") == "LOW"]
    llm_issues = [i for i in issues if i.get("severity_rule") != "LOW"]

    if low_issues:
        audit_msg = (
            f"Skipped LLM for {len(low_issues)} LOW issues "
            "(deterministic template)"
        )
        log.info(audit_msg)
        state["pattern_library_updates"].append({
            "source": "prioritisation_skip",
            "note": audit_msg,
            "count": len(low_issues),
        })

    # --- LOW path: deterministic template, no LLM ---------------------
    for finding in low_issues:
        tid = finding["trial_id"]
        ctx = _context_for(state, tid)
        risk = state["risk_scores"].get(tid) or _neutral_risk()
        sources = _sources_used([finding], ctx)
        related = [
            r.get("trial_id") for r in
            ((ctx.get("aact") or {}).get("related") or {})
            .get("related_by_condition", [])
        ][:5]
        prioritised.append(_build_prioritised_entry(
            finding, tid, risk, ctx,
            _deterministic_low_summary(finding), sources, related, state,
        ))

    if not llm_issues:
        log.info(
            "node_prioritise: 0 LLM batch call(s); %d LOW via template",
            len(low_issues),
        )
        return {
            "prioritised_issues": prioritised,
            "final_report": _build_final_report(state, prioritised),
        }

    # --- HIGH/MEDIUM/CRITICAL path: batched LLM in parallel ----------
    # Group remaining findings by trial; stable ordering for deterministic batches.
    by_trial: dict[str, list[dict]] = {}
    for i in llm_issues:
        by_trial.setdefault(i["trial_id"], []).append(i)
    trial_ids = list(by_trial.keys())

    # 1) Build all prompts up front — each batch is a (trial_ids, prompt) pair.
    batches: list[tuple[list[str], str]] = []
    for start in range(0, len(trial_ids), BATCH_SIZE):
        batch_trial_ids = trial_ids[start:start + BATCH_SIZE]
        blocks: list[str] = []
        for tid in batch_trial_ids:
            ctx = _context_for(state, tid)
            risk = state["risk_scores"].get(tid) or _neutral_risk()
            blocks.append(_render_trial_block(tid, by_trial[tid], risk, ctx))
        user = BATCH_PRIORITISATION_USER_TEMPLATE.format(
            n=len(batch_trial_ids),
            batch_findings="\n\n".join(blocks),
        )
        batches.append((batch_trial_ids, user))

    # 2) Fire all batches concurrently and await them together.
    wall_started = time.perf_counter()

    async def _run_all() -> list[dict | None]:
        # return_exceptions=True prevents one raising batch from
        # cancelling its siblings. Exceptions are normalised to None so
        # downstream mapping falls back cleanly per-trial, and an error
        # record is written to the tracker so provenance reflects the
        # failure even when the batch function was replaced outright
        # (e.g. by tests) and never ran its own error handler.
        raw = await asyncio.gather(*(
            _call_llm_batch_async(state, prompt) for _, prompt in batches
        ), return_exceptions=True)
        out: list[dict | None] = []
        for r in raw:
            if isinstance(r, BaseException):
                log.warning("node_prioritise: batch raised %r", r)
                tracker = _get_tracker()
                record = tracker.track_call(
                    model=LLM_MODEL, purpose="prioritisation",
                    prompt_tokens=0, completion_tokens=0,
                    latency_ms=0.0, status="error", error=str(r),
                )
                _record_call(state, record)
                out.append(None)
            else:
                out.append(r)
        return out

    parsed_results: list[dict | None] = asyncio.run(_run_all()) if batches else []
    wall_ms = (time.perf_counter() - wall_started) * 1000.0

    # 3) Map each batch response back to its source findings.
    for (batch_trial_ids, _prompt), parsed in zip(batches, parsed_results):
        by_key: dict[tuple[str, str], dict] = {}
        llm_findings = (parsed or {}).get("findings")
        if isinstance(llm_findings, list):
            for lf in llm_findings:
                if isinstance(lf, dict):
                    key = (
                        str(lf.get("trial_id", "")).strip(),
                        str(lf.get("check_name", "")).strip(),
                    )
                    if key[0] and key[1]:
                        by_key[key] = lf

        for tid in batch_trial_ids:
            ctx = _context_for(state, tid)
            risk = state["risk_scores"].get(tid) or _neutral_risk()
            trial_findings = by_trial[tid]
            sources = _sources_used(trial_findings, ctx)
            related = [
                r.get("trial_id") for r in
                ((ctx.get("aact") or {}).get("related") or {})
                .get("related_by_condition", [])
            ][:5]

            for finding in trial_findings:
                key = (tid, finding.get("check_name", ""))
                llm = by_key.get(key)
                summary = llm if llm else _fallback_prioritisation(finding)
                prioritised.append(_build_prioritised_entry(
                    finding, tid, risk, ctx, summary, sources, related, state,
                ))

    state["pattern_library_updates"].append({
        "source": "prioritisation_wall",
        "batches": len(batches),
        "wall_ms": round(wall_ms, 1),
        "note": (
            f"Prioritisation ran {len(batches)} batch(es) concurrently "
            f"in {wall_ms:.0f} ms wall-clock."
        ),
    })
    log.info(
        "node_prioritise: %d async batch call(s) for %d non-LOW trial(s) "
        "in %.0f ms wall; %d LOW via template",
        len(batches), len(trial_ids), wall_ms, len(low_issues),
    )
    report = _build_final_report(state, prioritised)
    return {"prioritised_issues": prioritised, "final_report": report}


# ---------------------------------------------------------------------------
# Node 6 — ROOT-CAUSE CLUSTERING
# ---------------------------------------------------------------------------


CLUSTER_SYSTEM = (
    "You are a clinical trial data quality analyst reviewing the output "
    "of an automated scan."
)

CLUSTER_USER_TEMPLATE = """
{n_issues} data quality findings from {n_trials} trials.
Scope: {scope}

Group into root cause clusters — issues sharing an
underlying operational or systemic cause. Do NOT
group by check type alone. Look for:
- Same lead sponsor with similar issues across trials
  → sponsor-level process failure
- Multiple check types firing on same trial
  → single trial with compounding problems
- Temporal clustering: issues from same period
  → regulatory or reporting change
- Cross-source discrepancy across multiple trials
  → systematic reporting gap
- Same issue type concentrated in one disease area
  → domain-specific data challenge

FINDINGS:
{issues_json}

For each cluster return:
- cluster_id: "RC001", "RC002", etc.
- root_cause: one sentence diagnosing underlying cause
- pattern: what issues have in common (name sponsors,
  check types, therapeutic areas — be specific)
- affected_trials: list of NCT IDs
- issue_indices: list of rank numbers from input
- cluster_severity: highest severity among grouped issues
- recommended_action: one systemic fix (not per-issue)

Issues without meaningful cluster → single group:
cluster_id: "RC_UNCLUSTERED"
root_cause: "Individual findings without shared pattern"
no recommended_action

Aim for 3-7 clusters. Do not force clustering.
Return JSON array. No markdown, no preamble.
"""


def _build_cluster_input(prioritised_issues: list[dict]) -> list[dict]:
    """Slim per-issue payload for the clustering prompt (no full provenance)."""
    return [
        {
            "rank": i + 1,
            "trial_id": issue["trial_id"],
            "severity": issue["severity"],
            "check_name": issue["check_name"],
            "check_category": issue["check_category"],
            "finding": issue["finding"],
            "completion_risk": issue.get(
                "completion_risk_score", "N/A"),
            "sources_queried": issue.get(
                "provenance", {}).get(
                "sources_queried", []),
            "sponsor": issue.get(
                "data_points", {}).get(
                "lead_sponsor", "unknown"),
            "condition": issue.get(
                "data_points", {}).get(
                "condition", "unknown"),
        }
        for i, issue in enumerate(prioritised_issues)
    ]


def _extract_clusters_from_llm(parsed: Any) -> list[dict]:
    """Pull a cluster list out of whatever shape the LLM returned.

    The prompt asks for a JSON array but the API runs in ``json_object``
    mode so the array is almost always wrapped in a single-key object.
    Accept either shape, plus common wrapper keys as a safety net.
    """
    if isinstance(parsed, list):
        return parsed
    if isinstance(parsed, dict):
        for key in ("clusters", "root_cause_clusters", "result", "items"):
            val = parsed.get(key)
            if isinstance(val, list):
                return val
        # Fallback: first list-valued field in the response
        for val in parsed.values():
            if isinstance(val, list):
                return val
    return []


def node_cluster(state: DataQualityState) -> dict:
    """Run root-cause clustering over the prioritised issues (Node 6).

    Produces ``state["final_report"]["root_cause_clusters"]``. One LLM
    call, tracked via LLMTracker with ``purpose="clustering"``. Falls
    back to an empty cluster list on LLM failure so the scan still
    produces a saveable report.
    """
    prioritised = state.get("prioritised_issues") or []
    fr = dict(state.get("final_report") or {})

    if not prioritised:
        log.info("node_cluster: no prioritised issues; skipping clustering.")
        fr["root_cause_clusters"] = []
        return {"final_report": fr}

    issues_payload = _build_cluster_input(prioritised)
    user = CLUSTER_USER_TEMPLATE.format(
        n_issues=len(prioritised),
        n_trials=len({i["trial_id"] for i in prioritised}),
        scope=state.get("scope", ""),
        issues_json=json.dumps(issues_payload, default=str),
    )

    log.info(
        "node_cluster: clustering %d finding(s) across %d trial(s)",
        len(prioritised),
        len({i["trial_id"] for i in prioritised}),
    )
    try:
        parsed = _call_llm(
            state, purpose="clustering",
            system=CLUSTER_SYSTEM, user=user,
        )
    except Exception as exc:  # noqa: BLE001 — node-level graceful degradation
        log.exception("node_cluster: LLM raised unexpectedly: %s", exc)
        parsed = None
    clusters = _extract_clusters_from_llm(parsed)

    # Sanity pass: ensure every cluster dict has the schema fields; drop
    # obvious malformed entries rather than propagating garbage.
    cleaned: list[dict] = []
    for c in clusters:
        if not isinstance(c, dict):
            continue
        cleaned.append({
            "cluster_id": str(c.get("cluster_id") or "RC_UNCLUSTERED"),
            "root_cause": str(c.get("root_cause") or ""),
            "pattern": str(c.get("pattern") or ""),
            "affected_trials": [
                str(t) for t in (c.get("affected_trials") or []) if t
            ],
            "issue_indices": [
                int(idx) for idx in (c.get("issue_indices") or [])
                if isinstance(idx, (int, float))
            ],
            "cluster_severity": str(c.get("cluster_severity") or "LOW"),
            "recommended_action": str(c.get("recommended_action") or ""),
        })

    fr["root_cause_clusters"] = cleaned
    log.info(
        "node_cluster: produced %d cluster(s) (%d non-unclustered)",
        len(cleaned),
        sum(1 for c in cleaned if c["cluster_id"] != "RC_UNCLUSTERED"),
    )
    return {"final_report": fr}


def _build_final_report(
    state: DataQualityState, prioritised: list[dict],
) -> dict:
    """Compact roll-up used by the Streamlit dashboard and export.

    LLMOps data lives in a single field — ``llmops_summary`` — sourced
    from the module-level ``LLMTracker`` (LangGraph doesn't reduce
    integer ``+=`` mutations across nodes, so state fields are stale).
    """
    severity_counts: dict[str, int] = {}
    for p in prioritised:
        s = p.get("severity", "LOW")
        severity_counts[s] = severity_counts.get(s, 0) + 1
    return {
        "scope": state["scope"],
        "scope_filters": state["scope_filters"],
        "checks_completed": state["checks_completed"],
        "issues_found": len(state["issues_found"]),
        "trials_with_issues": len({
            i["trial_id"] for i in state["issues_found"]
        }),
        "trials_investigated": [
            c.get("trial_id") for c in state.get("context_results", [])
        ],
        "iterations_used": state["iteration"],
        "severity_counts": severity_counts,
        "llmops_summary": _get_tracker().get_summary(),
        "risk_model_version": RISK_MODEL_VERSION,
    }


# ---------------------------------------------------------------------------
# Graph assembly
# ---------------------------------------------------------------------------


def build_graph():  # noqa: ANN201 — langgraph CompiledGraph typing is loose
    """Construct and compile the 6-node StateGraph."""
    graph: StateGraph = StateGraph(DataQualityState)
    graph.add_node("plan", node_plan)
    graph.add_node("scan", node_scan)
    graph.add_node("review", node_review)
    graph.add_node("context", node_context)
    graph.add_node("prioritise", node_prioritise)
    graph.add_node("cluster", node_cluster)

    graph.add_edge(START, "plan")
    graph.add_edge("plan", "scan")
    graph.add_edge("scan", "review")
    graph.add_conditional_edges(
        "review",
        route_from_review,
        {"context": "context", "prioritise": "prioritise"},
    )
    graph.add_edge("context", "review")
    graph.add_edge("prioritise", "cluster")
    graph.add_edge("cluster", END)

    return graph.compile()


def run_scan(
    scope: str, scope_filters: dict,
    save_report: bool = True,
    reports_dir: str | None = None,
) -> tuple[DataQualityState, "ScanReport"]:
    """Run one scan end-to-end and return ``(state, report)``.

    Steps:
      1. Instantiate a fresh ``LLMTracker`` so every scan starts clean.
      2. Invoke the compiled StateGraph on a seeded initial state.
      3. Stamp the tracker summary onto ``final_report["llmops_summary"]``.
      4. Build a ``ScanReport`` from the final state and persist to
         ``output/reports/<scan_id>.json`` (unless ``save_report=False``).
    """
    from agents.state import make_initial_state  # local imports avoid cycles
    from output.report import ScanReport
    global _tracker
    _tracker = LLMTracker()
    app = build_graph()
    initial = make_initial_state(scope, scope_filters)
    state: DataQualityState = app.invoke(initial)
    if "final_report" in state and isinstance(state["final_report"], dict):
        state["final_report"]["llmops_summary"] = _tracker.get_summary()

    report = ScanReport.from_agent_state(state)
    if save_report:
        kwargs = {"path": reports_dir} if reports_dir else {}
        report.save(**kwargs)
    return state, report
