"""MCP server — OpenFDA adverse-event tools.

Data source: OpenFDA drug/event endpoint
(https://api.fda.gov/drug/event.json). Free, no API key, <240 req/min.

Tools:
    search_adverse_events       → sample reports + totals + top reactions
    get_drug_safety_summary     → aggregated safety profile + year trend
    compare_trial_vs_fda_events → AACT trial SAE count vs FDA post-market
                                  (check G — cross-source safety)

Response envelope matches the other MCP servers: ``status``, ``tool``,
``source: "openfda"``, ``timestamp``. Every HTTP response is cached to
``mcp_servers/cache/`` keyed by an md5 of the query params. Tools never
crash — any failure becomes a structured error payload.

A note on OpenFDA's quirk: the API returns HTTP 404 when a query
matches zero records. We translate that to an empty-but-successful
result rather than an error so downstream callers can distinguish
"clean drug" from "API down".

Run as a stdio server:
    venv/bin/python -m mcp_servers.openfda_server
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import logging
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Optional

import pandas as pd
import requests
from dotenv import load_dotenv
from mcp.server import Server
from mcp.server.stdio import stdio_server
from mcp.types import TextContent, Tool

from checks.models import load_table

load_dotenv()

log = logging.getLogger(__name__)

PROJECT_ROOT: Path = Path(__file__).resolve().parent.parent
CACHE_DIR: Path = PROJECT_ROOT / "mcp_servers" / "cache"
CACHE_DIR.mkdir(parents=True, exist_ok=True)

SERVER_NAME = "openfda"
BASE_URL = "https://api.fda.gov/drug/event.json"
HTTP_TIMEOUT_SEC = 20
MAX_RESULTS_CEILING = 100
TOP_REACTIONS_DEFAULT = 10

# Threshold below which an absent/low trial-reported AE count combined
# with a non-trivial FDA post-market count triggers a discrepancy flag.
FDA_AE_DISCREPANCY_THRESHOLD = 10


# ---------------------------------------------------------------------------
# Response envelope helpers
# ---------------------------------------------------------------------------


def _error(tool: str, reason: str, **extra: Any) -> dict:
    """Structured error payload."""
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
    """Structured success payload."""
    return {
        "status": "ok",
        "tool": tool,
        "source": SERVER_NAME,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        **payload,
    }


# ---------------------------------------------------------------------------
# HTTP + cache layer
# ---------------------------------------------------------------------------


def _hash_key(data: Any) -> str:
    """Stable hash over a JSON-serialisable object."""
    blob = json.dumps(data, sort_keys=True, default=str).encode("utf-8")
    return hashlib.md5(blob).hexdigest()


def _cache_path(params: dict) -> Path:
    return CACHE_DIR / f"openfda_{_hash_key(params)}.json"


def _openfda_get(params: dict) -> dict:
    """GET from OpenFDA with caching.

    Translates OpenFDA's 404-for-zero-matches convention into an
    empty-but-successful response so callers can tell "no AEs on record"
    apart from "API unreachable".
    """
    path = _cache_path(params)
    if path.exists():
        try:
            return json.loads(path.read_text())
        except Exception as exc:  # noqa: BLE001 — fall through to live call
            log.warning("cache read failed (%s), refetching: %s", path.name, exc)

    log.info("openfda live call: params=%s", params)
    r = requests.get(BASE_URL, params=params, timeout=HTTP_TIMEOUT_SEC)
    if r.status_code == 404:
        data: dict = {
            "meta": {"results": {"total": 0}},
            "results": [],
            "_empty": True,
        }
    else:
        r.raise_for_status()
        data = r.json()

    try:
        path.write_text(json.dumps(data))
    except Exception as exc:  # noqa: BLE001 — cache write is non-fatal
        log.warning("cache write failed (%s): %s", path.name, exc)

    return data


def _drug_search_clause(drug_name: str) -> str:
    """Build a drug-search clause that matches generic, brand, and label strings.

    OpenFDA exposes several parallel drug-name fields; we OR across them
    so that ``"pembrolizumab"`` (generic) and ``"Keytruda"`` (brand)
    both match.
    """
    name = drug_name.strip()
    return (
        f'(patient.drug.openfda.generic_name:"{name}"'
        f' OR patient.drug.openfda.brand_name:"{name}"'
        f' OR patient.drug.medicinalproduct:"{name}")'
    )


def _cap_limit(n: int) -> int:
    """Clamp a user-supplied limit to ``[1, MAX_RESULTS_CEILING]``."""
    return max(1, min(int(n), MAX_RESULTS_CEILING))


# ---------------------------------------------------------------------------
# Report parsing
# ---------------------------------------------------------------------------


def _parse_report(report: dict) -> dict:
    """Extract the fields we care about from one raw OpenFDA report."""
    reactions = []
    for rxn in (report.get("patient") or {}).get("reaction", []) or []:
        term = rxn.get("reactionmeddrapt")
        if term:
            reactions.append(term)
    return {
        "report_id": report.get("safetyreportid"),
        "date": report.get("receivedate") or report.get("receiptdate"),
        "serious": report.get("serious") == "1",
        "serious_death": report.get("seriousnessdeath") == "1",
        "reactions": reactions,
    }


def _count_query(drug_name: str, count_field: str, limit: int = 10) -> list[dict]:
    """Run a faceted-count OpenFDA query and return ``results`` list."""
    params = {
        "search": _drug_search_clause(drug_name),
        "count": count_field,
        "limit": limit,
    }
    data = _openfda_get(params)
    return data.get("results") or []


def _total_for(drug_name: str, extra: str = "") -> int:
    """Return ``meta.results.total`` for a drug query + optional extra clause."""
    search = _drug_search_clause(drug_name)
    if extra:
        search = f"{search} AND {extra}"
    params = {"search": search, "limit": 1}
    data = _openfda_get(params)
    return int(((data.get("meta") or {}).get("results") or {}).get("total", 0))


TREND_DAILY_LIMIT = 500  # OpenFDA rejects larger count-query limits with 403


def _trend_by_year(drug_name: str) -> dict[str, int]:
    """Aggregate daily receive-date counts into a ``{YYYY: count}`` dict.

    Returns an empty dict on failure (the trend is optional context; a
    403 / timeout on this call must not fail the whole summary).
    """
    params = {
        "search": _drug_search_clause(drug_name),
        "count": "receivedate",
        "limit": TREND_DAILY_LIMIT,
    }
    try:
        data = _openfda_get(params)
    except Exception as exc:  # noqa: BLE001 — trend is best-effort
        log.warning(
            "receivedate trend unavailable for %r (%s); returning empty trend.",
            drug_name, exc,
        )
        return {}
    year_counts: Counter[str] = Counter()
    for row in data.get("results") or []:
        date_str = row.get("time") or ""
        if len(date_str) >= 4 and date_str[:4].isdigit():
            year_counts[date_str[:4]] += int(row.get("count", 0))
    return dict(sorted(year_counts.items()))


# ---------------------------------------------------------------------------
# Tool implementations
# ---------------------------------------------------------------------------


def tool_search_adverse_events(drug_name: str, limit: int = 10) -> dict:
    """Return a sample of FDA adverse-event reports for a drug plus totals."""
    tool = "search_adverse_events"
    try:
        if not drug_name or not drug_name.strip():
            return _error(tool, "drug_name is empty")
        capped = _cap_limit(limit)

        # 1) sample reports + total count in one call
        sample_params = {
            "search": _drug_search_clause(drug_name),
            "limit": capped,
        }
        sample_data = _openfda_get(sample_params)
        raw_reports = sample_data.get("results") or []
        total_count = int(
            ((sample_data.get("meta") or {}).get("results") or {}).get("total", 0)
        )
        sample_reports = [_parse_report(r) for r in raw_reports]

        # 2) serious count — separate query
        serious_count = (
            _total_for(drug_name, extra="serious:1") if total_count else 0
        )

        # 3) top reactions by faceted count
        top_rxn_rows = (
            _count_query(
                drug_name, "patient.reaction.reactionmeddrapt.exact",
                limit=TOP_REACTIONS_DEFAULT,
            ) if total_count else []
        )
        top_reactions = [
            {"reaction": r.get("term"), "count": int(r.get("count", 0))}
            for r in top_rxn_rows if r.get("term")
        ]

        return _ok(
            tool,
            drug_name=drug_name,
            total_count=total_count,
            serious_count=serious_count,
            top_reactions=top_reactions,
            sample_reports=sample_reports,
            max_results=capped,
        )
    except Exception as exc:  # noqa: BLE001
        log.exception("%s failed for %r: %s", tool, drug_name, exc)
        return _error(tool, str(exc), drug_name=drug_name)


def tool_get_drug_safety_summary(drug_name: str) -> dict:
    """Return an aggregated safety profile for a drug."""
    tool = "get_drug_safety_summary"
    try:
        if not drug_name or not drug_name.strip():
            return _error(tool, "drug_name is empty")

        total = _total_for(drug_name)
        serious = _total_for(drug_name, extra="serious:1") if total else 0
        deaths = (
            _total_for(drug_name, extra="seriousnessdeath:1") if total else 0
        )

        top_rxn_rows = (
            _count_query(
                drug_name, "patient.reaction.reactionmeddrapt.exact", limit=10,
            ) if total else []
        )
        top_10 = [
            {"reaction": r.get("term"), "count": int(r.get("count", 0))}
            for r in top_rxn_rows if r.get("term")
        ]

        trend = _trend_by_year(drug_name) if total else {}

        def _pct(n: int, d: int) -> float:
            return round(100.0 * n / d, 2) if d else 0.0

        return _ok(
            tool,
            drug_name=drug_name,
            total_reports=total,
            serious_reports=serious,
            serious_pct=_pct(serious, total),
            death_reports=deaths,
            death_pct=_pct(deaths, total),
            top_10_reactions=top_10,
            reporting_trend=trend,
            trend_available=bool(trend),
            trend_sample_cap=TREND_DAILY_LIMIT,
        )
    except Exception as exc:  # noqa: BLE001
        log.exception("%s failed for %r: %s", tool, drug_name, exc)
        return _error(tool, str(exc), drug_name=drug_name)


def _trial_sae_count(nct_id: str) -> tuple[Optional[int], str]:
    """Return (sae_subject_count, source_note) from AACT calculated_values.

    Returns ``(None, note)`` if the trial row isn't present so the caller
    can distinguish "unknown" from "zero". Note explains the proxy used
    (CLAUDE.md / LIMITATIONS.md: direct AE-reported flag is not available).
    """
    calc = load_table("calculated_values")
    if calc is None or "nct_id" not in calc.columns:
        return None, "calculated_values table unavailable"
    match = calc[calc["nct_id"] == nct_id]
    if match.empty:
        return None, "trial not found in calculated_values"
    row = match.iloc[0]
    val = row.get("number_of_sae_subjects")
    if pd.isna(val):
        return None, "number_of_sae_subjects is null for this trial"
    return int(val), "proxy: calculated_values.number_of_sae_subjects"


def tool_compare_trial_vs_fda_events(nct_id: str, drug_name: str) -> dict:
    """Compare AACT trial SAE count against FDA post-market report volume.

    Discrepancy flags (``discrepancy_flag=True``) when the trial reports
    zero SAE subjects *and* FDA has at least
    ``FDA_AE_DISCREPANCY_THRESHOLD`` events on record for the same drug.
    This is the rule-level gate for check G; final severity is decided
    by the agent's LLM prioritisation using the ``discrepancy_note``.
    """
    tool = "compare_trial_vs_fda_events"
    try:
        if not nct_id or not nct_id.strip():
            return _error(tool, "nct_id is empty")
        if not drug_name or not drug_name.strip():
            return _error(tool, "drug_name is empty", nct_id=nct_id)

        trial_ae, trial_note = _trial_sae_count(nct_id)
        fda_ae = _total_for(drug_name)

        if trial_ae is None:
            flag = False
            note = (
                f"Trial AE data unavailable ({trial_note}); FDA has "
                f"{fda_ae} report(s) for {drug_name}. Cannot compare."
            )
        elif trial_ae == 0 and fda_ae >= FDA_AE_DISCREPANCY_THRESHOLD:
            flag = True
            note = (
                f"Trial reports 0 SAE subjects but FDA has {fda_ae} "
                f"post-market event report(s) for {drug_name} "
                f"(threshold {FDA_AE_DISCREPANCY_THRESHOLD})."
            )
        elif trial_ae == 0 and fda_ae == 0:
            flag = False
            note = (
                "Both trial and FDA report zero events — clean profile "
                "or drug not yet marketed."
            )
        else:
            flag = False
            note = (
                f"Trial: {trial_ae} SAE subject(s); FDA: {fda_ae} event(s). "
                "Volumes are not directly comparable (trial subjects vs "
                "post-market reports); no rule-level discrepancy."
            )

        return _ok(
            tool,
            nct_id=nct_id,
            drug_name=drug_name,
            trial_ae_count=trial_ae,
            trial_ae_source=trial_note,
            fda_ae_count=fda_ae,
            discrepancy_flag=flag,
            discrepancy_note=note,
            threshold=FDA_AE_DISCREPANCY_THRESHOLD,
        )
    except Exception as exc:  # noqa: BLE001
        log.exception(
            "%s failed for %s / %r: %s", tool, nct_id, drug_name, exc,
        )
        return _error(tool, str(exc), nct_id=nct_id, drug_name=drug_name)


# ---------------------------------------------------------------------------
# MCP server wiring
# ---------------------------------------------------------------------------

server: Server = Server(SERVER_NAME)

TOOL_DISPATCH: dict[str, Callable[..., dict]] = {
    "search_adverse_events":       tool_search_adverse_events,
    "get_drug_safety_summary":     tool_get_drug_safety_summary,
    "compare_trial_vs_fda_events": tool_compare_trial_vs_fda_events,
}


@server.list_tools()
async def list_tools() -> list[Tool]:
    """Declare OpenFDA tools exposed to the agent."""
    return [
        Tool(
            name="search_adverse_events",
            description=(
                "Return total count, serious count, top reactions, and a "
                "small sample of raw FDA adverse-event reports for a drug."
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "drug_name": {
                        "type": "string",
                        "description": "Generic or brand drug name.",
                    },
                    "limit": {
                        "type": "integer",
                        "description": "Sample report count (default 10, cap 100).",
                        "default": 10,
                    },
                },
                "required": ["drug_name"],
            },
        ),
        Tool(
            name="get_drug_safety_summary",
            description=(
                "Aggregated safety profile for a drug: total reports, serious/"
                "death percentages, top 10 reactions, and yearly reporting trend."
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "drug_name": {
                        "type": "string",
                        "description": "Generic or brand drug name.",
                    },
                },
                "required": ["drug_name"],
            },
        ),
        Tool(
            name="compare_trial_vs_fda_events",
            description=(
                "Compare AACT trial SAE-subject count against FDA post-market "
                "report volume for the same drug. Flags a discrepancy when "
                "the trial reports 0 SAE subjects and FDA has >= "
                f"{FDA_AE_DISCREPANCY_THRESHOLD} events. Implements the "
                "rule-level gate for check G."
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "nct_id": {"type": "string", "description": "NCT ID."},
                    "drug_name": {
                        "type": "string",
                        "description": "Generic or brand drug name.",
                    },
                },
                "required": ["nct_id", "drug_name"],
            },
        ),
    ]


@server.call_tool()
async def call_tool(name: str, arguments: dict) -> list[TextContent]:
    """Dispatch tool invocation; always return a single TextContent."""
    fn = TOOL_DISPATCH.get(name)
    if fn is None:
        payload = _error(name, f"unknown tool {name!r}")
    else:
        try:
            payload = fn(**(arguments or {}))
        except TypeError as exc:
            log.warning("%s called with bad arguments %s: %s",
                        name, arguments, exc)
            payload = _error(name, f"invalid arguments: {exc}")
        except Exception as exc:  # noqa: BLE001 — never crash the server
            log.exception("%s crashed unexpectedly: %s", name, exc)
            payload = _error(name, str(exc))
    return [TextContent(type="text", text=json.dumps(payload, default=str))]


async def _amain() -> None:
    """Serve MCP over stdio until the client disconnects."""
    log.info("Starting OpenFDA MCP server (stdio). Cache at %s", CACHE_DIR)
    async with stdio_server() as (read, write):
        await server.run(read, write, server.create_initialization_options())


def main() -> None:
    """Entry point: ``python -m mcp_servers.openfda_server``."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
    )
    asyncio.run(_amain())


if __name__ == "__main__":
    main()
