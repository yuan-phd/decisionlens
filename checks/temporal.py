"""Check A — temporal / date consistency.

Flags trials whose ``start_date`` / ``completion_date`` combinations are
internally inconsistent or outside plausible bounds. Returns ``Issue``
objects in the standard rule-engine format.
"""

from __future__ import annotations

import logging
from typing import Optional

import pandas as pd

from checks.models import Issue, filter_trial_scope, load_table

log = logging.getLogger(__name__)

CATEGORY = "temporal"
SOURCE_TABLES = ["studies"]
SOURCE_COLUMNS = [
    "nct_id", "overall_status", "phase",
    "start_date", "completion_date",
]

# Plausibility bounds. Trials posted to ClinicalTrials.gov go back to
# ~2000; anything before 1990 is almost certainly a data entry error.
EARLIEST_PLAUSIBLE_START = pd.Timestamp("1990-01-01")
FUTURE_HORIZON_YEARS = 20
MAX_PHASE1_DURATION_DAYS = 15 * 365  # 15 years for a Phase 1 is suspicious


def check_temporal_consistency(
    trial_id: Optional[str] = None,
    therapeutic_area: Optional[str] = None,
    phase: Optional[str] = None,
    overall_status: Optional[str] = None,
    limit: int = 100,
) -> list[Issue]:
    """Run all temporal consistency rules over the scoped trial set.

    Rules:
      1. ``completion_date`` strictly before ``start_date`` — HIGH.
      2. ``completion_date`` more than 20 years in the future — MEDIUM.
      3. ``start_date`` before 1990 — MEDIUM.
      4. Phase 1 trial with duration > 15 years — MEDIUM.
      5. Status = Completed but ``completion_date`` is in the future — HIGH.
    """
    issues: list[Issue] = []
    try:
        studies = load_table("studies")
        if studies is None:
            return issues

        for col in ("start_date", "completion_date"):
            if col in studies.columns:
                studies[col] = pd.to_datetime(studies[col], errors="coerce")

        scoped = filter_trial_scope(
            studies, trial_id=trial_id,
            therapeutic_area=therapeutic_area, phase=phase,
            overall_status=overall_status, limit=limit,
        )
        if scoped.empty:
            return issues

        now = pd.Timestamp.now().normalize()
        future_horizon = now + pd.DateOffset(years=FUTURE_HORIZON_YEARS)

        for _, row in scoped.iterrows():
            nct = row.get("nct_id")
            start = row.get("start_date")
            end = row.get("completion_date")
            status = row.get("overall_status")
            phase = row.get("phase")

            # Rule 1 — completion strictly before start
            if pd.notna(start) and pd.notna(end) and end < start:
                issues.append(Issue(
                    trial_id=nct,
                    check_name="completion_before_start",
                    check_category=CATEGORY,
                    severity_rule="HIGH",
                    finding=(
                        f"Completion date {end.date().isoformat()} precedes "
                        f"start date {start.date().isoformat()}."
                    ),
                    data_points={
                        "start_date": start.date().isoformat(),
                        "completion_date": end.date().isoformat(),
                    },
                    source_tables=SOURCE_TABLES,
                    source_columns=["nct_id", "start_date", "completion_date"],
                ))

            # Rule 2 — completion far in the future
            if pd.notna(end) and end > future_horizon:
                issues.append(Issue(
                    trial_id=nct,
                    check_name="completion_far_future",
                    check_category=CATEGORY,
                    severity_rule="MEDIUM",
                    finding=(
                        f"Completion date {end.date().isoformat()} is more than "
                        f"{FUTURE_HORIZON_YEARS} years in the future."
                    ),
                    data_points={
                        "completion_date": end.date().isoformat(),
                        "horizon_years": FUTURE_HORIZON_YEARS,
                    },
                    source_tables=SOURCE_TABLES,
                    source_columns=["nct_id", "completion_date"],
                ))

            # Rule 3 — implausibly old start
            if pd.notna(start) and start < EARLIEST_PLAUSIBLE_START:
                issues.append(Issue(
                    trial_id=nct,
                    check_name="start_before_1990",
                    check_category=CATEGORY,
                    severity_rule="MEDIUM",
                    finding=(
                        f"Start date {start.date().isoformat()} predates 1990 "
                        "(implausible for a registered trial)."
                    ),
                    data_points={"start_date": start.date().isoformat()},
                    source_tables=SOURCE_TABLES,
                    source_columns=["nct_id", "start_date"],
                ))

            # Rule 4 — Phase 1 trial running > 15 years
            if (
                pd.notna(start) and pd.notna(end)
                and isinstance(phase, str) and phase.strip() == "Phase 1"
                and (end - start).days > MAX_PHASE1_DURATION_DAYS
            ):
                duration_years = round((end - start).days / 365.25, 1)
                issues.append(Issue(
                    trial_id=nct,
                    check_name="phase1_duration_implausible",
                    check_category=CATEGORY,
                    severity_rule="MEDIUM",
                    finding=(
                        f"Phase 1 trial duration is {duration_years} years "
                        f"(> {MAX_PHASE1_DURATION_DAYS // 365} year threshold)."
                    ),
                    data_points={
                        "phase": phase,
                        "start_date": start.date().isoformat(),
                        "completion_date": end.date().isoformat(),
                        "duration_years": duration_years,
                    },
                    source_tables=SOURCE_TABLES,
                    source_columns=[
                        "nct_id", "phase", "start_date", "completion_date",
                    ],
                ))

            # Rule 5 — Completed status but completion date in the future
            if (
                isinstance(status, str) and status.strip() == "Completed"
                and pd.notna(end) and end > now
            ):
                issues.append(Issue(
                    trial_id=nct,
                    check_name="completed_with_future_end",
                    check_category=CATEGORY,
                    severity_rule="HIGH",
                    finding=(
                        f"Status is 'Completed' but completion date "
                        f"{end.date().isoformat()} is in the future."
                    ),
                    data_points={
                        "overall_status": status,
                        "completion_date": end.date().isoformat(),
                        "today": now.date().isoformat(),
                    },
                    source_tables=SOURCE_TABLES,
                    source_columns=[
                        "nct_id", "overall_status", "completion_date",
                    ],
                ))

    except Exception as exc:  # noqa: BLE001 — graceful degradation
        log.exception("check_temporal_consistency failed: %s", exc)

    log.info("temporal check produced %d issues over %s trial(s)",
             len(issues), 0 if studies is None else len(scoped))
    return issues
