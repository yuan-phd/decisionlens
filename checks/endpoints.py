"""Check D — endpoint and design gaps.

Flags trials missing structural design fields (masking, allocation,
intervention model) or required endpoints (no Primary outcome). Late
phase trials carry stricter expectations than early phase.
"""

from __future__ import annotations

import logging
from typing import Optional

import pandas as pd

from checks.models import Issue, filter_trial_scope, load_table

log = logging.getLogger(__name__)

CATEGORY = "endpoint"
LATE_PHASES = {"Phase 3", "Phase 2/Phase 3", "Phase 4"}


def check_endpoints_gaps(
    trial_id: Optional[str] = None,
    therapeutic_area: Optional[str] = None,
    phase: Optional[str] = None,
    overall_status: Optional[str] = None,
    limit: int = 100,
) -> list[Issue]:
    """Run all endpoint and design-gap rules over the scoped trial set.

    Rules:
      1. No Primary outcome row in ``outcome_counts`` — HIGH.
      2. Late-phase (Phase 3 / 2-3 / 4) trial with missing
         masking, allocation, or intervention_model — HIGH.
      3. Late-phase trial with allocation = 'Non-Randomized' — MEDIUM.
      4. Late-phase Treatment trial with masking = 'None (Open Label)' — MEDIUM.
      5. Completed trial with no Secondary outcome rows — LOW.
    """
    issues: list[Issue] = []
    try:
        studies = load_table("studies")
        if studies is None:
            return issues

        scoped = filter_trial_scope(
            studies, trial_id=trial_id,
            therapeutic_area=therapeutic_area, phase=phase,
            overall_status=overall_status, limit=limit,
        )
        if scoped.empty:
            return issues

        designs = load_table("designs")
        designs_indexed = (
            designs.set_index("nct_id") if designs is not None
            and "nct_id" in designs.columns else None
        )

        outcome_counts = load_table("outcome_counts")
        if outcome_counts is not None and "nct_id" in outcome_counts.columns:
            primary_trials = set(
                outcome_counts.loc[
                    outcome_counts["outcome_type"] == "Primary", "nct_id"
                ].unique()
            )
            secondary_trials = set(
                outcome_counts.loc[
                    outcome_counts["outcome_type"] == "Secondary", "nct_id"
                ].unique()
            )
        else:
            primary_trials, secondary_trials = set(), set()

        for _, row in scoped.iterrows():
            nct = row.get("nct_id")
            phase = row.get("phase")
            phase_str = phase.strip() if isinstance(phase, str) else ""
            status = row.get("overall_status")
            status_str = status.strip() if isinstance(status, str) else ""

            design_row = (
                designs_indexed.loc[nct]
                if designs_indexed is not None and nct in designs_indexed.index
                else None
            )
            # ``loc`` may return a DataFrame if duplicates exist; pick the first.
            if isinstance(design_row, pd.DataFrame):
                design_row = design_row.iloc[0]

            # Rule 1 — no primary outcome registered
            # Observational / registry studies (phase N/A or null) are not
            # required to declare a primary outcome — downgrade to LOW.
            if nct not in primary_trials:
                is_non_phased = phase_str in ("", "N/A")
                severity = "LOW" if is_non_phased else "HIGH"
                issues.append(Issue(
                    trial_id=nct,
                    check_name="missing_primary_outcome",
                    check_category=CATEGORY,
                    severity_rule=severity,
                    finding=(
                        "No Primary outcome row registered in outcome_counts."
                    ),
                    data_points={
                        "primary_outcome_rows": 0,
                        "phase": phase_str or None,
                    },
                    source_tables=["studies", "outcome_counts"],
                    source_columns=["nct_id", "phase", "outcome_type"],
                ))

            # Rule 2 — late-phase missing structural design fields
            if phase_str in LATE_PHASES and design_row is not None:
                missing = []
                for field in ("masking", "allocation", "intervention_model"):
                    val = design_row.get(field) if hasattr(design_row, "get") else None
                    if pd.isna(val) or not str(val).strip():
                        missing.append(field)
                if missing:
                    issues.append(Issue(
                        trial_id=nct,
                        check_name="latephase_missing_design_field",
                        check_category=CATEGORY,
                        severity_rule="HIGH",
                        finding=(
                            f"Late-phase trial ({phase_str}) missing design "
                            f"field(s): {', '.join(missing)}."
                        ),
                        data_points={
                            "phase": phase_str,
                            "missing_fields": missing,
                        },
                        source_tables=["studies", "designs"],
                        source_columns=[
                            "nct_id", "phase",
                            "masking", "allocation", "intervention_model",
                        ],
                    ))

            # Rule 3 — late-phase non-randomized
            if (
                phase_str in LATE_PHASES and design_row is not None
                and str(design_row.get("allocation", "")).strip()
                == "Non-Randomized"
            ):
                issues.append(Issue(
                    trial_id=nct,
                    check_name="latephase_non_randomized",
                    check_category=CATEGORY,
                    severity_rule="MEDIUM",
                    finding=(
                        f"Late-phase trial ({phase_str}) is Non-Randomized."
                    ),
                    data_points={
                        "phase": phase_str,
                        "allocation": "Non-Randomized",
                    },
                    source_tables=["studies", "designs"],
                    source_columns=["nct_id", "phase", "allocation"],
                ))

            # Rule 4 — late-phase Treatment open-label
            if (
                phase_str in LATE_PHASES and design_row is not None
                and str(design_row.get("masking", "")).strip()
                == "None (Open Label)"
                and str(design_row.get("primary_purpose", "")).strip()
                == "Treatment"
            ):
                issues.append(Issue(
                    trial_id=nct,
                    check_name="latephase_treatment_open_label",
                    check_category=CATEGORY,
                    severity_rule="MEDIUM",
                    finding=(
                        f"Late-phase Treatment trial ({phase_str}) is Open Label "
                        "(no masking)."
                    ),
                    data_points={
                        "phase": phase_str,
                        "masking": "None (Open Label)",
                        "primary_purpose": "Treatment",
                    },
                    source_tables=["studies", "designs"],
                    source_columns=[
                        "nct_id", "phase", "masking", "primary_purpose",
                    ],
                ))

            # Rule 5 — Completed trial with zero secondary outcomes
            if (
                status_str == "Completed"
                and nct not in secondary_trials
            ):
                issues.append(Issue(
                    trial_id=nct,
                    check_name="completed_no_secondary_outcomes",
                    check_category=CATEGORY,
                    severity_rule="LOW",
                    finding=(
                        "Completed trial has no Secondary outcome rows."
                    ),
                    data_points={
                        "overall_status": status_str,
                        "secondary_outcome_rows": 0,
                    },
                    source_tables=["studies", "outcome_counts"],
                    source_columns=[
                        "nct_id", "overall_status", "outcome_type",
                    ],
                ))

    except Exception as exc:  # noqa: BLE001 — graceful degradation
        log.exception("check_endpoints_gaps failed: %s", exc)

    log.info("endpoints check produced %d issues", len(issues))
    return issues
