"""Check G — trial safety vs FDA post-market cross-validation.

For each scoped trial we:
  1. Resolve the trial's first Drug-type intervention name via
     ``interventions.parquet``.
  2. Call ``tool_compare_trial_vs_fda_events(nct_id, drug_name)`` to
     compare AACT's ``number_of_sae_subjects`` against OpenFDA's
     post-market event volume for the same drug.
  3. Optionally call ``tool_get_drug_safety_summary`` for context that
     accompanies HIGH findings (top reactions, serious %).

Rules emitted:
- HIGH  ``zero_aes_with_fda_signals``      — trial reports 0 SAE
  subjects but the MCP comparison flags a discrepancy with OpenFDA.
- MEDIUM ``incomplete_ae_profile_vs_fda``  — trial reports some SAE
  subjects but FDA post-market volume is much larger (ratio
  threshold — see ``MEDIUM_RATIO``). The MCP tool's discrepancy_flag
  itself fires only on the zero-AE case, so this MEDIUM rule is
  derived client-side from raw counts.
- LOW   ``trial_safety_aligns_with_fda``   — completed trial with
  no discrepancy flagged; recorded as a positive signal.
"""

from __future__ import annotations

import logging
import re
from typing import Optional

import pandas as pd

from checks.models import Issue, filter_trial_scope, load_table
from mcp_servers.openfda_server import tool_compare_trial_vs_fda_events

log = logging.getLogger(__name__)

CATEGORY = "safety"
SOURCE_TABLES = [
    "studies", "calculated_values", "interventions", "openfda",
]
SOURCE_COLUMNS = [
    "nct_id", "number_of_sae_subjects", "intervention_name", "fda_ae_count",
]

# Thresholds for the MEDIUM client-side rule. Trial subjects and FDA
# post-market events are different units, so the ratio has to be large
# enough to carry signal through that noise.
MEDIUM_RATIO: int = 50          # fda_ae >= 50 * trial_ae
MEDIUM_MIN_FDA_EVENTS: int = 50  # absolute floor to avoid tiny-denominator noise

# Module-level cache so a limit=N scan doesn't reload interventions.parquet
# once per trial. Reset by tests via ``_interventions_cache = None``.
_interventions_cache: Optional[pd.DataFrame] = None


def _sanitise_drug_name(name: str) -> str:
    """Strip unicode symbols and dosage info that break FDA API queries.

    Real AACT intervention names often arrive with trademark symbols
    (``Ozempic®``) or embedded dosing/formulation prefixes
    (``Drug: Pegylated rhG-CSF: 100µg/kg``) — the OpenFDA query syntax
    rejects these as 400 Bad Request. We trim to the bare drug name
    before sending it on the wire.
    """
    name = re.sub(r"[®™µ]", "", name)
    name = re.sub(
        r"\d+\s*(mg|mcg|µg|ml|kg)(/kg)?", "", name, flags=re.IGNORECASE,
    )
    name = name.split(":")[-1].strip()
    name = name.split("(")[0].strip()
    return name.strip()


def _drug_names_for_trial(nct_id: str) -> list[str]:
    """Return Drug-type intervention names for a trial (may be empty).

    Real AACT data already supplies real drug names in
    ``interventions.parquet`` — every Drug-type intervention is eligible
    for OpenFDA cross-validation, no overlay required. Names are passed
    through ``_sanitise_drug_name`` so OpenFDA's strict query parser
    doesn't choke on unicode or embedded dosing.
    """
    global _interventions_cache
    if _interventions_cache is None:
        _interventions_cache = load_table("interventions")

    interventions = _interventions_cache
    if interventions is None or "nct_id" not in interventions.columns:
        return []
    # Case-insensitive on intervention_type because AACT raw uses ``DRUG``
    # while earlier synthetic exports used ``Drug``.
    itype = interventions.get("intervention_type")
    if itype is None:
        return []
    is_drug = itype.astype(str).str.strip().str.upper() == "DRUG"
    rows = interventions[(interventions["nct_id"] == nct_id) & is_drug]
    cleaned = (
        _sanitise_drug_name(str(n))
        for n in rows["name"].dropna().tolist()
    )
    return [n for n in cleaned if n]


def check_safety_cross_validation(
    nct_id: Optional[str] = None,
    therapeutic_area: Optional[str] = None,
    phase: Optional[str] = None,
    overall_status: Optional[str] = None,
    limit: int = 100,
) -> list[Issue]:
    """Run all safety cross-validation rules over the scoped trial set.

    Each scoped trial triggers one OpenFDA MCP call
    (``tool_compare_trial_vs_fda_events``). Trials without a Drug-type
    intervention are skipped (nothing to compare against FDA).
    """
    issues: list[Issue] = []
    try:
        studies = load_table("studies")
        if studies is None:
            return issues

        scoped = filter_trial_scope(
            studies, trial_id=nct_id,
            therapeutic_area=therapeutic_area, phase=phase,
            overall_status=overall_status, limit=limit,
        )

        if scoped.empty:
            log.info(
                "safety check: trial(s) not in AACT scope — nothing to "
                "evaluate (nct_id=%s, therapeutic_area=%s, limit=%d).",
                nct_id, therapeutic_area, limit,
            )
            return issues

        for _, row in scoped.iterrows():
            nct = row.get("nct_id")
            status = row.get("overall_status")
            status_str = status.strip() if isinstance(status, str) else ""

            drug_names = _drug_names_for_trial(nct)
            if not drug_names:
                log.debug("skipping %s: no Drug-type intervention", nct)
                continue
            drug = drug_names[0]

            comparison = tool_compare_trial_vs_fda_events(nct, drug)
            if comparison.get("status") != "ok":
                log.warning(
                    "openfda comparison failed for %s/%r: %s",
                    nct, drug, comparison.get("reason"),
                )
                continue

            trial_ae = comparison.get("trial_ae_count")
            fda_ae = int(comparison.get("fda_ae_count", 0) or 0)
            flag = bool(comparison.get("discrepancy_flag"))
            mcp_note = comparison.get("discrepancy_note", "")

            # Rule HIGH — zero trial AEs, FDA has signals
            if flag and trial_ae == 0:
                issues.append(Issue(
                    trial_id=nct,
                    check_name="zero_aes_with_fda_signals",
                    check_category=CATEGORY,
                    severity_rule="HIGH",
                    finding=mcp_note or (
                        f"Trial reports 0 SAE subjects but FDA has "
                        f"{fda_ae} event(s) for {drug}."
                    ),
                    data_points={
                        "trial_ae_count": trial_ae,
                        "fda_ae_count": fda_ae,
                        "intervention_name": drug,
                        "discrepancy_flag": True,
                    },
                    source_tables=SOURCE_TABLES,
                    source_columns=SOURCE_COLUMNS,
                ))
                continue  # Don't double-flag this trial

            # Rule MEDIUM — trial has some AEs, FDA volume is much larger
            if (
                trial_ae is not None and trial_ae > 0
                and fda_ae >= MEDIUM_MIN_FDA_EVENTS
                and fda_ae >= MEDIUM_RATIO * trial_ae
            ):
                ratio = round(fda_ae / trial_ae, 1)
                issues.append(Issue(
                    trial_id=nct,
                    check_name="incomplete_ae_profile_vs_fda",
                    check_category=CATEGORY,
                    severity_rule="MEDIUM",
                    finding=(
                        f"Trial reports {trial_ae} SAE subject(s); FDA has "
                        f"{fda_ae} post-market event(s) for {drug} "
                        f"(ratio {ratio}× >= threshold {MEDIUM_RATIO}×)."
                    ),
                    data_points={
                        "trial_ae_count": trial_ae,
                        "fda_ae_count": fda_ae,
                        "ratio": ratio,
                        "ratio_threshold": MEDIUM_RATIO,
                        "intervention_name": drug,
                    },
                    source_tables=SOURCE_TABLES,
                    source_columns=SOURCE_COLUMNS,
                ))
                continue

            # Rule LOW (positive signal) — completed, no discrepancy
            if (
                status_str == "Completed" and not flag
                and trial_ae is not None
            ):
                issues.append(Issue(
                    trial_id=nct,
                    check_name="trial_safety_aligns_with_fda",
                    check_category=CATEGORY,
                    severity_rule="LOW",
                    finding=(
                        f"Completed trial reports {trial_ae} SAE subject(s); "
                        f"FDA has {fda_ae} post-market event(s) for {drug}. "
                        "No rule-level discrepancy."
                    ),
                    data_points={
                        "trial_ae_count": trial_ae,
                        "fda_ae_count": fda_ae,
                        "intervention_name": drug,
                        "discrepancy_flag": False,
                    },
                    source_tables=SOURCE_TABLES,
                    source_columns=SOURCE_COLUMNS,
                ))

    except Exception as exc:  # noqa: BLE001 — graceful degradation
        log.exception("check_safety_cross_validation failed: %s", exc)

    log.info("safety check produced %d issues", len(issues))
    return issues
