"""Check B — enrollment anomalies.

Flags trials where target/actual enrollment values are missing,
non-positive, implausibly large, or where the realised enrollment falls
far short of the target for a completed trial.
"""

from __future__ import annotations

import logging
from typing import Optional

import pandas as pd

from checks.models import Issue, filter_trial_scope, load_table

log = logging.getLogger(__name__)

CATEGORY = "enrollment"
SOURCE_TABLES = ["studies"]
SOURCE_COLUMNS = [
    "nct_id", "overall_status", "phase",
    "enrollment", "actual_enrollment",
]

# Plausibility thresholds.
SHORTFALL_FRACTION = 0.30        # actual < 30% of planned -> HIGH
PHASE3_MIN_ENROLLMENT = 50       # Phase III with < 50 patients is suspicious
ENROLLMENT_MAX_PLAUSIBLE = 50_000


def check_enrollment_anomalies(
    trial_id: Optional[str] = None,
    therapeutic_area: Optional[str] = None,
    phase: Optional[str] = None,
    overall_status: Optional[str] = None,
    limit: int = 100,
) -> list[Issue]:
    """Run all enrollment anomaly rules over the scoped trial set.

    Rules:
      1. ``enrollment`` <= 0 — HIGH.
      2. Completed trial with ``actual_enrollment`` < 30% of target — HIGH.
      3. Phase 3 trial with target ``enrollment`` < 50 — MEDIUM.
      4. ``enrollment`` > 50,000 — MEDIUM (likely data entry error).
      5. Status = Completed but ``actual_enrollment`` is null — HIGH.
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

        for _, row in scoped.iterrows():
            nct = row.get("nct_id")
            planned = row.get("enrollment")
            actual = row.get("actual_enrollment")
            status = row.get("overall_status")
            phase = row.get("phase")

            # Rule 1 — non-positive planned enrollment
            if pd.notna(planned) and planned <= 0:
                issues.append(Issue(
                    trial_id=nct,
                    check_name="enrollment_non_positive",
                    check_category=CATEGORY,
                    severity_rule="HIGH",
                    finding=(
                        f"Planned enrollment is {int(planned)} (must be > 0)."
                    ),
                    data_points={"enrollment": float(planned)},
                    source_tables=SOURCE_TABLES,
                    source_columns=["nct_id", "enrollment"],
                ))

            # Rule 2 — completed trial with severe shortfall
            if (
                isinstance(status, str) and status.strip() == "Completed"
                and pd.notna(planned) and planned > 0 and pd.notna(actual)
                and actual < SHORTFALL_FRACTION * planned
            ):
                pct = round(100.0 * actual / planned, 1)
                issues.append(Issue(
                    trial_id=nct,
                    check_name="enrollment_shortfall",
                    check_category=CATEGORY,
                    severity_rule="HIGH",
                    finding=(
                        f"Completed trial enrolled {int(actual)} of "
                        f"{int(planned)} planned ({pct}%)."
                    ),
                    data_points={
                        "enrollment": float(planned),
                        "actual_enrollment": float(actual),
                        "shortfall_pct": pct,
                        "threshold_pct": SHORTFALL_FRACTION * 100,
                    },
                    source_tables=SOURCE_TABLES,
                    source_columns=[
                        "nct_id", "overall_status",
                        "enrollment", "actual_enrollment",
                    ],
                ))

            # Rule 3 — under-powered Phase 3
            if (
                isinstance(phase, str) and phase.strip() == "Phase 3"
                and pd.notna(planned) and 0 < planned < PHASE3_MIN_ENROLLMENT
            ):
                issues.append(Issue(
                    trial_id=nct,
                    check_name="phase3_underpowered",
                    check_category=CATEGORY,
                    severity_rule="MEDIUM",
                    finding=(
                        f"Phase 3 trial planned enrollment is {int(planned)} "
                        f"(< {PHASE3_MIN_ENROLLMENT} threshold)."
                    ),
                    data_points={
                        "phase": phase,
                        "enrollment": float(planned),
                        "threshold": PHASE3_MIN_ENROLLMENT,
                    },
                    source_tables=SOURCE_TABLES,
                    source_columns=["nct_id", "phase", "enrollment"],
                ))

            # Rule 4 — implausibly large planned enrollment
            if pd.notna(planned) and planned > ENROLLMENT_MAX_PLAUSIBLE:
                issues.append(Issue(
                    trial_id=nct,
                    check_name="enrollment_implausibly_large",
                    check_category=CATEGORY,
                    severity_rule="MEDIUM",
                    finding=(
                        f"Planned enrollment of {int(planned)} exceeds "
                        f"plausibility threshold ({ENROLLMENT_MAX_PLAUSIBLE:,})."
                    ),
                    data_points={
                        "enrollment": float(planned),
                        "threshold": ENROLLMENT_MAX_PLAUSIBLE,
                    },
                    source_tables=SOURCE_TABLES,
                    source_columns=["nct_id", "enrollment"],
                ))

            # Rule 5 — Completed trial missing actual_enrollment
            if (
                isinstance(status, str) and status.strip() == "Completed"
                and pd.isna(actual)
            ):
                issues.append(Issue(
                    trial_id=nct,
                    check_name="completed_missing_actual_enrollment",
                    check_category=CATEGORY,
                    severity_rule="HIGH",
                    finding=(
                        "Trial marked Completed but actual_enrollment is missing."
                    ),
                    data_points={
                        "overall_status": status,
                        "enrollment": None if pd.isna(planned) else float(planned),
                        "actual_enrollment": None,
                    },
                    source_tables=SOURCE_TABLES,
                    source_columns=[
                        "nct_id", "overall_status", "actual_enrollment",
                    ],
                ))

    except Exception as exc:  # noqa: BLE001 — graceful degradation
        log.exception("check_enrollment_anomalies failed: %s", exc)

    log.info("enrollment check produced %d issues", len(issues))
    return issues
