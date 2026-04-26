"""DataQualityState — the TypedDict shared by every orchestrator node.

All five nodes in ``agents/orchestrator.py`` read from and return a
partial update of this state. LangGraph merges node updates into the
running state automatically. Storing issues/risk as plain ``dict`` (not
dataclasses) keeps the state JSON-serialisable for logging and for
future persistence.
"""

from __future__ import annotations

from typing import TypedDict

# Hard caps applied by the scan node so one agent run can never flood
# the pipeline. The MCP layer also enforces its own ``MAX_SCAN_LIMIT``
# per tool call — these are the downstream equivalents.
MAX_ISSUES_PER_CHECK: int = 200
MAX_TOTAL_ISSUES: int = 500


class DataQualityState(TypedDict):
    """Shape of the state object passed between orchestrator nodes."""

    # Scope — set by the caller in make_initial_state()
    scope: str
    scope_filters: dict           # e.g. {"therapeutic_area": "oncology",
                                  #       "phase": "Phase 3", "limit": 50}

    # Planning
    plan: list[str]               # check names in execution order
    checks_completed: list[str]   # names of checks actually run

    # Scan output — issues are Issue.to_dict() payloads
    issues_found: list[dict]
    risk_scores: dict[str, dict]  # nct_id -> risk payload from risk_scorer

    # Review + context loop
    needs_deeper_investigation: list[str]  # nct_ids
    context_results: list[dict]
    iteration: int                # review-loop count, capped at 3

    # LLMOps
    llm_calls: list[dict]
    total_tokens: int
    total_latency_ms: float
    estimated_cost: float

    # Final output
    prioritised_issues: list[dict]
    pattern_library_updates: list[dict]
    final_report: dict


def make_initial_state(scope: str, scope_filters: dict) -> DataQualityState:
    """Seed a fresh state dict with empty collections and zeroed counters."""
    return DataQualityState(
        scope=scope,
        scope_filters=dict(scope_filters),
        plan=[],
        checks_completed=[],
        issues_found=[],
        risk_scores={},
        needs_deeper_investigation=[],
        context_results=[],
        iteration=0,
        llm_calls=[],
        total_tokens=0,
        total_latency_ms=0.0,
        estimated_cost=0.0,
        prioritised_issues=[],
        pattern_library_updates=[],
        final_report={},
    )
