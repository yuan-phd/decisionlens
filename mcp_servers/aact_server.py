"""MCP server — AACT clinical trials registry tools.

Exposes the Phase 1 rule engine, the v1 risk model, and a handful of
direct AACT lookups as MCP tools so the LangGraph agent can call them
through the standard protocol rather than importing Python functions
directly.

Run as a stdio server:
    venv/bin/python -m mcp_servers.aact_server

Tools (see ``list_tools`` for full inputSchemas):
    run_quality_check       → one of the 5 check_* functions (A–E)
    get_risk_score          → v1 XGBoost completion-risk score
    get_trial_details       → full studies.parquet row for one NCT ID
    get_eligibility_criteria→ eligibilities.parquet row for one NCT ID
    get_related_trials      → trials sharing sponsor or condition

Design rules (from CLAUDE.md):
    - Every tool wraps its work in try/except and returns a structured
      ``{"status": "error", ...}`` payload rather than crashing.
    - Cache directory present at mcp_servers/cache/ for consistency with
      the PubMed / OpenFDA servers (AACT itself is local-only and does
      not need a response cache).
"""

from __future__ import annotations

import asyncio
import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Optional

import pandas as pd
from dotenv import load_dotenv
from mcp.server import Server
from mcp.server.stdio import stdio_server
from mcp.types import TextContent, Tool

from checks.crossfield import check_crossfield_validation
from checks.endpoints import check_endpoints_gaps
from checks.enrollment import check_enrollment_anomalies
from checks.models import load_table
from checks.status import check_status_inconsistencies
from checks.temporal import check_temporal_consistency
from models.risk_scorer import (
    RISK_MODEL_VERSION,
    get_completion_risk_score,
)

load_dotenv()

log = logging.getLogger(__name__)

PROJECT_ROOT: Path = Path(__file__).resolve().parent.parent
CACHE_DIR: Path = PROJECT_ROOT / "mcp_servers" / "cache"
CACHE_DIR.mkdir(parents=True, exist_ok=True)

SERVER_NAME = "aact"

# Guardrails applied to any caller-supplied ``limit`` parameter so an
# over-eager LLM (or a bug) cannot launch an unbounded scan.
MAX_SCAN_LIMIT: int = 200

# When the caller requests a phase filter we scan this many times the
# user-facing limit before post-filtering, so enough trials survive the
# filter to hit ``limit``. Applied internally; the user-facing limit is
# still capped at MAX_SCAN_LIMIT.
PHASE_PREFETCH_MULTIPLIER: int = 5


def _cap_limit(limit: int) -> tuple[int, bool]:
    """Cap ``limit`` at ``MAX_SCAN_LIMIT``. Returns (capped_value, was_capped)."""
    requested = int(limit)
    capped = min(requested, MAX_SCAN_LIMIT)
    return capped, capped < requested


# Accept both letter (A–E) and category-name aliases for check_name.
CHECK_REGISTRY: dict[str, Callable[..., list]] = {
    "a":          check_temporal_consistency,
    "temporal":   check_temporal_consistency,
    "b":          check_enrollment_anomalies,
    "enrollment": check_enrollment_anomalies,
    "c":          check_status_inconsistencies,
    "status":     check_status_inconsistencies,
    "d":          check_endpoints_gaps,
    "endpoint":   check_endpoints_gaps,
    "endpoints":  check_endpoints_gaps,
    "e":          check_crossfield_validation,
    "crossfield": check_crossfield_validation,
}


def _error(tool: str, reason: str, **extra: Any) -> dict:
    """Build the standard structured-error payload for a tool failure."""
    payload = {
        "status": "error",
        "tool": tool,
        "reason": reason,
        "source": SERVER_NAME,
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }
    payload.update(extra)
    return payload


def _ok(tool: str, **payload: Any) -> dict:
    """Build the standard success payload."""
    return {
        "status": "ok",
        "tool": tool,
        "source": SERVER_NAME,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        **payload,
    }


# ---------------------------------------------------------------------------
# Tool implementations
# ---------------------------------------------------------------------------


def tool_run_quality_check(
    check_name: str,
    therapeutic_area: Optional[str] = None,
    phase: Optional[str] = None,
    overall_status: Optional[str] = None,
    limit: int = 100,
) -> dict:
    """Run one of the 5 deterministic check functions (A–E).

    ``phase`` and ``overall_status`` are pushed *into*
    ``filter_trial_scope`` (not post-filtered) so the user-facing
    ``limit`` reliably returns the requested number of matching trials
    regardless of how the underlying parquet is ordered.
    """
    tool = "run_quality_check"
    try:
        key = (check_name or "").strip().lower()
        fn = CHECK_REGISTRY.get(key)
        if fn is None:
            return _error(
                tool,
                f"unknown check_name {check_name!r}",
                valid_names=sorted(set(CHECK_REGISTRY)),
            )

        capped_limit, limit_capped = _cap_limit(limit)
        issues = fn(
            therapeutic_area=therapeutic_area,
            phase=phase,
            overall_status=overall_status,
            limit=capped_limit,
        )

        payload: dict[str, Any] = dict(
            check_name=fn.__name__,
            check_alias=key,
            therapeutic_area=therapeutic_area,
            phase=phase,
            overall_status=overall_status,
            phase_filter_applied=phase is not None,
            status_filter_applied=overall_status is not None,
            limit=capped_limit,
            limit_capped=limit_capped,
            max_scan_limit=MAX_SCAN_LIMIT,
            issues_found=len(issues),
            issues=[i.to_dict() for i in issues],
        )
        return _ok(tool, **payload)
    except Exception as exc:  # noqa: BLE001 — graceful degradation
        log.exception("%s failed: %s", tool, exc)
        return _error(tool, str(exc))


def tool_get_risk_score(trial_id: str) -> dict:
    """Return the v1 completion-risk score payload for one trial."""
    tool = "get_risk_score"
    try:
        risk = get_completion_risk_score(trial_id)
        return _ok(
            tool,
            trial_id=trial_id,
            completion_risk_score=risk["risk_score"],
            risk_reliable=risk["risk_reliable"],
            risk_model_version=risk["risk_model_version"],
            note=risk["note"],
        )
    except Exception as exc:  # noqa: BLE001
        log.exception("%s failed for %s: %s", tool, trial_id, exc)
        return _error(tool, str(exc), trial_id=trial_id)


def _row_to_json_safe(row: pd.Series) -> dict:
    """Convert a pandas row to a JSON-serialisable dict (dates → ISO,
    NaN → None)."""
    out: dict[str, Any] = {}
    for col, val in row.items():
        if pd.isna(val):
            out[col] = None
        elif isinstance(val, (pd.Timestamp, datetime)):
            out[col] = val.isoformat()
        elif hasattr(val, "item"):  # numpy scalars
            out[col] = val.item()
        else:
            out[col] = val
    return out


def tool_get_trial_details(nct_id: str) -> dict:
    """Return the full studies.parquet row for ``nct_id`` as a dict."""
    tool = "get_trial_details"
    try:
        studies = load_table("studies")
        if studies is None:
            return _error(tool, "studies table unavailable", trial_id=nct_id)
        match = studies[studies["nct_id"] == nct_id]
        if match.empty:
            return _error(tool, "trial not found", trial_id=nct_id)
        return _ok(
            tool,
            trial_id=nct_id,
            details=_row_to_json_safe(match.iloc[0]),
        )
    except Exception as exc:  # noqa: BLE001
        log.exception("%s failed for %s: %s", tool, nct_id, exc)
        return _error(tool, str(exc), trial_id=nct_id)


def tool_get_eligibility_criteria(nct_id: str) -> dict:
    """Return the eligibilities.parquet row for ``nct_id`` as a dict.

    Includes the full criteria text — callers are responsible for
    passing it to the LLM / truncating as needed.
    """
    tool = "get_eligibility_criteria"
    try:
        elig = load_table("eligibilities")
        if elig is None:
            return _error(
                tool, "eligibilities table unavailable", trial_id=nct_id,
            )
        match = elig[elig["nct_id"] == nct_id]
        if match.empty:
            return _error(tool, "trial not found", trial_id=nct_id)
        row = _row_to_json_safe(match.iloc[0])
        return _ok(
            tool,
            trial_id=nct_id,
            gender=row.get("gender"),
            minimum_age=row.get("minimum_age"),
            maximum_age=row.get("maximum_age"),
            healthy_volunteers=row.get("healthy_volunteers"),
            criteria=row.get("criteria"),
            criteria_length=(
                len(row["criteria"]) if isinstance(row.get("criteria"), str) else 0
            ),
        )
    except Exception as exc:  # noqa: BLE001
        log.exception("%s failed for %s: %s", tool, nct_id, exc)
        return _error(tool, str(exc), trial_id=nct_id)


def tool_get_related_trials(nct_id: str, limit: int = 10) -> dict:
    """Return trials sharing a condition or a lead sponsor with ``nct_id``.

    The returned lists are deduplicated and exclude the query trial
    itself. Results are bounded by ``limit`` per relationship kind.
    """
    tool = "get_related_trials"
    try:
        capped_limit, limit_capped = _cap_limit(limit)
        conditions = load_table("conditions")
        sponsors = load_table("sponsors")

        by_condition: list[dict] = []
        shared_conditions: list[str] = []
        if conditions is not None and "nct_id" in conditions.columns:
            target_conds = set(
                conditions.loc[
                    conditions["nct_id"] == nct_id, "downcase_name"
                ].dropna().tolist()
            )
            shared_conditions = sorted(target_conds)
            if target_conds:
                cohort = conditions[
                    (conditions["downcase_name"].isin(target_conds))
                    & (conditions["nct_id"] != nct_id)
                ]
                by_condition = [
                    {"trial_id": row["nct_id"],
                     "shared_condition": row["downcase_name"]}
                    for _, row in cohort.head(capped_limit).iterrows()
                ]

        by_sponsor: list[dict] = []
        shared_sponsor: Optional[str] = None
        if sponsors is not None and "nct_id" in sponsors.columns:
            target_lead = sponsors[
                (sponsors["nct_id"] == nct_id)
                & (sponsors.get("lead_or_collaborator") == "lead")
            ]
            if not target_lead.empty:
                sponsor_name = target_lead.iloc[0]["name"]
                shared_sponsor = sponsor_name
                cohort = sponsors[
                    (sponsors["name"] == sponsor_name)
                    & (sponsors.get("lead_or_collaborator") == "lead")
                    & (sponsors["nct_id"] != nct_id)
                ]
                by_sponsor = [
                    {"trial_id": row["nct_id"], "shared_sponsor": sponsor_name}
                    for _, row in cohort.head(capped_limit).iterrows()
                ]

        return _ok(
            tool,
            trial_id=nct_id,
            shared_conditions=shared_conditions,
            shared_sponsor=shared_sponsor,
            related_by_condition=by_condition,
            related_by_sponsor=by_sponsor,
            limit=capped_limit,
            limit_capped=limit_capped,
            max_scan_limit=MAX_SCAN_LIMIT,
        )
    except Exception as exc:  # noqa: BLE001
        log.exception("%s failed for %s: %s", tool, nct_id, exc)
        return _error(tool, str(exc), trial_id=nct_id)


# ---------------------------------------------------------------------------
# MCP server wiring
# ---------------------------------------------------------------------------

server: Server = Server(SERVER_NAME)

TOOL_DISPATCH: dict[str, Callable[..., dict]] = {
    "run_quality_check":       tool_run_quality_check,
    "get_risk_score":          tool_get_risk_score,
    "get_trial_details":       tool_get_trial_details,
    "get_eligibility_criteria":tool_get_eligibility_criteria,
    "get_related_trials":      tool_get_related_trials,
}


@server.list_tools()
async def list_tools() -> list[Tool]:
    """Declare every AACT tool exposed to the agent."""
    return [
        Tool(
            name="run_quality_check",
            description=(
                "Run one of the 5 AACT data-quality checks (A–E). "
                "Returns a list of Issue objects as JSON. "
                "check_name accepts letters (A/B/C/D/E) or names "
                "(temporal/enrollment/status/endpoints/crossfield). "
                "limit is capped at MAX_SCAN_LIMIT=200. When phase is "
                "specified the scan pre-fetches limit*5 trials and "
                "post-filters, so raising limit only helps up to 200."
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "check_name": {
                        "type": "string",
                        "description": (
                            "A/B/C/D/E or "
                            "temporal/enrollment/status/endpoints/crossfield"
                        ),
                    },
                    "therapeutic_area": {
                        "type": "string",
                        "description": (
                            "Optional. oncology / cardiology / neurology / "
                            "immunology / respiratory / infectious_disease / "
                            "endocrinology / psychiatry"
                        ),
                    },
                    "phase": {
                        "type": "string",
                        "description": (
                            "Optional. Exact AACT phase string, e.g. "
                            "'Phase 1', 'Phase 2', 'Phase 3', 'Phase 4', 'N/A'."
                        ),
                    },
                    "limit": {
                        "type": "integer",
                        "description": "Max trials to scan (default 100).",
                        "default": 100,
                    },
                },
                "required": ["check_name"],
            },
        ),
        Tool(
            name="get_risk_score",
            description=(
                "Return the v1 XGBoost completion-risk score for a trial "
                "(1 - P(completed)). 0.5 neutral fallback if the trial "
                "cannot be scored."
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "trial_id": {
                        "type": "string",
                        "description": "NCT ID, e.g. NCT18032585.",
                    },
                },
                "required": ["trial_id"],
            },
        ),
        Tool(
            name="get_trial_details",
            description=(
                "Return the full studies.parquet row for an NCT ID "
                "(nct_id, overall_status, phase, enrollment, dates, "
                "sponsor flags, etc.) as JSON."
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "nct_id": {"type": "string", "description": "NCT ID."},
                },
                "required": ["nct_id"],
            },
        ),
        Tool(
            name="get_eligibility_criteria",
            description=(
                "Return the eligibility criteria row for an NCT ID: gender, "
                "age bounds, healthy volunteers flag, full criteria text."
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "nct_id": {"type": "string", "description": "NCT ID."},
                },
                "required": ["nct_id"],
            },
        ),
        Tool(
            name="get_related_trials",
            description=(
                "Return trials that share a condition or lead sponsor with "
                "the target trial. Useful for context enrichment."
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "nct_id": {"type": "string", "description": "NCT ID."},
                    "limit": {
                        "type": "integer",
                        "description": "Max related trials per relationship.",
                        "default": 10,
                    },
                },
                "required": ["nct_id"],
            },
        ),
    ]


@server.call_tool()
async def call_tool(name: str, arguments: dict) -> list[TextContent]:
    """Dispatch a tool call to its implementation; always return TextContent."""
    fn = TOOL_DISPATCH.get(name)
    if fn is None:
        payload = _error(name, f"unknown tool {name!r}")
    else:
        try:
            payload = fn(**(arguments or {}))
        except TypeError as exc:
            # Bad argument names / types from the caller.
            log.warning("%s called with bad arguments %s: %s",
                        name, arguments, exc)
            payload = _error(name, f"invalid arguments: {exc}")
        except Exception as exc:  # noqa: BLE001 — never crash the server
            log.exception("%s crashed unexpectedly: %s", name, exc)
            payload = _error(name, str(exc))
    return [TextContent(type="text", text=json.dumps(payload, default=str))]


async def _amain() -> None:
    """Serve MCP over stdio until the client disconnects."""
    log.info("Starting AACT MCP server (stdio). Cache at %s", CACHE_DIR)
    async with stdio_server() as (read, write):
        await server.run(read, write, server.create_initialization_options())


def main() -> None:
    """Entry point: ``python -m mcp_servers.aact_server``."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
    )
    asyncio.run(_amain())


if __name__ == "__main__":
    main()
