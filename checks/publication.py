"""Check F — publication vs registry cross-source validation.

Cross-validates AACT registry data (studies + outcome_counts) against
PubMed literature via the PubMed MCP tool. For each scoped trial we
ask PubMed for publications citing the NCT ID and flag inconsistencies
between what was published and what the registry shows.

Rules emitted (severity_rule before LLM prioritisation):
- HIGH  ``selective_reporting_suspected``   — completed >2y ago,
  zero AACT outcome rows, but PubMed has publications citing NCT ID
- MEDIUM ``publication_enrollment_mismatch`` — completed with AACT
  results, PubMed abstracts cite an enrollment number that doesn't
  match the AACT enrollment value within 20 % tolerance (best-effort
  regex over abstract snippets — false positives are possible)
- MEDIUM ``stopped_trial_has_publications`` — Withdrawn/Terminated
  trial but PubMed has publications citing the NCT ID. Spec asks for
  PI-match; AACT parquet has no investigator column, so we simplify
  to "any publication citing this NCT."
- LOW  ``completed_no_publications``       — completed with AACT
  results but zero PubMed papers found (informational)
"""

from __future__ import annotations

import logging
import re
from typing import Optional

import pandas as pd

from checks.models import Issue, filter_trial_scope, load_table
from mcp_servers.pubmed_server import tool_search_trial_publications

log = logging.getLogger(__name__)

CATEGORY = "publication"
SOURCE_TABLES = ["studies", "outcome_counts", "pubmed"]
SOURCE_COLUMNS = [
    "nct_id", "completion_date", "overall_status", "pmids_found",
]

COMPLETION_STALENESS_YEARS = 2
ENROLLMENT_TOLERANCE = 0.20  # 20 % tolerance on AACT vs PubMed enrollment
PUBMED_MAX_RESULTS = 10

# Best-effort regex to harvest enrollment numbers from abstract text:
# "616 patients", "340 subjects", "the 1,200-participant trial" (commas stripped).
ENROLLMENT_PATTERN = re.compile(
    r"(\d[\d,]{1,5})[\s\-](?:patients?|subjects?|participants?)",
    re.IGNORECASE,
)


def _pubmed_enrollments(publications: list[dict]) -> list[int]:
    """Extract likely enrollment numbers from publication abstract snippets."""
    nums: set[int] = set()
    for pub in publications:
        snippet = pub.get("abstract_snippet") or ""
        for m in ENROLLMENT_PATTERN.finditer(snippet):
            try:
                nums.add(int(m.group(1).replace(",", "")))
            except ValueError:
                continue
    return sorted(nums)


def check_publication_consistency(
    nct_id: Optional[str] = None,
    therapeutic_area: Optional[str] = None,
    phase: Optional[str] = None,
    overall_status: Optional[str] = None,
    limit: int = 100,
) -> list[Issue]:
    """Run all publication-consistency rules over the scoped trial set.

    Each scoped trial triggers one PubMed MCP call
    (``tool_search_trial_publications``). PubMed responses are cached on
    disk so re-runs on the same scope are free.
    """
    issues: list[Issue] = []
    try:
        studies = load_table("studies")
        if studies is None:
            return issues
        if "completion_date" in studies.columns:
            studies["completion_date"] = pd.to_datetime(
                studies["completion_date"], errors="coerce",
            )

        scoped = filter_trial_scope(
            studies, trial_id=nct_id,
            therapeutic_area=therapeutic_area, phase=phase,
            overall_status=overall_status, limit=limit,
        )
        if scoped.empty:
            log.info(
                "publication check: trial(s) not in AACT scope — "
                "nothing to evaluate (nct_id=%s, therapeutic_area=%s, limit=%d).",
                nct_id, therapeutic_area, limit,
            )
            return issues

        outcome_counts = load_table("outcome_counts")
        trials_with_outcomes = (
            set(outcome_counts["nct_id"].unique())
            if outcome_counts is not None
            and "nct_id" in outcome_counts.columns
            else set()
        )

        now = pd.Timestamp.now().normalize()
        staleness_cutoff = now - pd.DateOffset(years=COMPLETION_STALENESS_YEARS)

        for _, row in scoped.iterrows():
            nct = row.get("nct_id")
            status = row.get("overall_status")
            status_str = status.strip() if isinstance(status, str) else ""
            end = row.get("completion_date")
            planned = row.get("enrollment")
            actual = row.get("actual_enrollment")
            aact_enrollment = actual if pd.notna(actual) else planned
            has_results = nct in trials_with_outcomes

            pubmed = tool_search_trial_publications(
                nct, max_results=PUBMED_MAX_RESULTS,
            )
            if pubmed.get("status") != "ok":
                log.warning(
                    "pubmed lookup failed for %s: %s",
                    nct, pubmed.get("reason"),
                )
                continue

            publications = pubmed.get("publications") or []
            pmids = [p.get("pmid") for p in publications if p.get("pmid")]
            pub_count = len(pmids)

            # Rule HIGH — selective reporting suspected
            if (
                status_str == "Completed"
                and pd.notna(end) and end < staleness_cutoff
                and not has_results
                and pub_count > 0
            ):
                issues.append(Issue(
                    trial_id=nct,
                    check_name="selective_reporting_suspected",
                    check_category=CATEGORY,
                    severity_rule="HIGH",
                    finding=(
                        f"Trial completed {end.date().isoformat()} "
                        f"(>{COMPLETION_STALENESS_YEARS}y ago), no outcome "
                        f"rows in AACT, but PubMed has {pub_count} "
                        "publication(s) citing this NCT ID."
                    ),
                    data_points={
                        "completion_date": end.date().isoformat(),
                        "aact_outcome_rows": 0,
                        "pubmed_count": pub_count,
                        "pmids_found": pmids,
                    },
                    source_tables=SOURCE_TABLES,
                    source_columns=SOURCE_COLUMNS,
                ))
                continue  # Don't double-flag with the LOW/MEDIUM rules

            # Rule MEDIUM — enrollment mismatch
            if (
                status_str == "Completed" and has_results and pub_count > 0
                and pd.notna(aact_enrollment) and aact_enrollment > 0
            ):
                pub_ns = _pubmed_enrollments(publications)
                if pub_ns and not any(
                    abs(n - aact_enrollment) / aact_enrollment
                    <= ENROLLMENT_TOLERANCE
                    for n in pub_ns
                ):
                    issues.append(Issue(
                        trial_id=nct,
                        check_name="publication_enrollment_mismatch",
                        check_category=CATEGORY,
                        severity_rule="MEDIUM",
                        finding=(
                            f"AACT enrollment is {int(aact_enrollment)} but "
                            f"PubMed abstracts mention {pub_ns} — none "
                            f"within {int(ENROLLMENT_TOLERANCE * 100)}% "
                            "tolerance."
                        ),
                        data_points={
                            "aact_enrollment": float(aact_enrollment),
                            "pubmed_enrollment_numbers": pub_ns,
                            "tolerance_pct": ENROLLMENT_TOLERANCE * 100,
                            "pmids_found": pmids,
                        },
                        source_tables=SOURCE_TABLES,
                        source_columns=SOURCE_COLUMNS,
                    ))

            # Rule MEDIUM — stopped trial still has publications
            if status_str in ("Withdrawn", "Terminated") and pub_count > 0:
                issues.append(Issue(
                    trial_id=nct,
                    check_name="stopped_trial_has_publications",
                    check_category=CATEGORY,
                    severity_rule="MEDIUM",
                    finding=(
                        f"Trial is {status_str} but PubMed has {pub_count} "
                        "publication(s) citing this NCT ID."
                    ),
                    data_points={
                        "overall_status": status_str,
                        "pubmed_count": pub_count,
                        "pmids_found": pmids,
                    },
                    source_tables=SOURCE_TABLES,
                    source_columns=SOURCE_COLUMNS,
                ))

            # Rule LOW — completed with results but no publications
            if status_str == "Completed" and has_results and pub_count == 0:
                issues.append(Issue(
                    trial_id=nct,
                    check_name="completed_no_publications",
                    check_category=CATEGORY,
                    severity_rule="LOW",
                    finding=(
                        "Completed trial with AACT outcome rows posted but "
                        "no PubMed publications found."
                    ),
                    data_points={
                        "overall_status": status_str,
                        "aact_outcome_rows": "present",
                        "pubmed_count": 0,
                    },
                    source_tables=SOURCE_TABLES,
                    source_columns=SOURCE_COLUMNS,
                ))

    except Exception as exc:  # noqa: BLE001 — graceful degradation
        log.exception("check_publication_consistency failed: %s", exc)

    log.info("publication check produced %d issues", len(issues))
    return issues
