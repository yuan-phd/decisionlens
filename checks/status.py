"""Check C — status inconsistencies.

Flags trials whose ``overall_status`` is internally inconsistent with
other recorded fields: missing stop reasons for terminated trials, stale
"Recruiting" labels past the completion date, and completed trials with
no posted outcome counts.
"""

from __future__ import annotations

import logging
from typing import Optional

import pandas as pd

from checks.models import Issue, filter_trial_scope, load_table

log = logging.getLogger(__name__)

CATEGORY = "status"

# Statuses that legally require a ``why_stopped`` explanation per
# ClinicalTrials.gov / FDAAA reporting rules.
STOPPED_STATUSES = {"Terminated", "Withdrawn", "Suspended"}
STALE_ACTIVE_YEARS = 5


def check_status_inconsistencies(
    trial_id: Optional[str] = None,
    therapeutic_area: Optional[str] = None,
    phase: Optional[str] = None,
    overall_status: Optional[str] = None,
    limit: int = 100,
) -> list[Issue]:
    """Run all status-inconsistency rules over the scoped trial set.

    Rules:
      1. Status in {Terminated, Withdrawn, Suspended} but ``why_stopped``
         is null/empty — HIGH (regulatory requirement).
      2. Status = Recruiting but ``completion_date`` is in the past — HIGH.
      3. Status = "Active, not recruiting" with completion_date >5y past — MEDIUM.
      4. Status = Completed but no rows in ``outcome_counts`` — HIGH.
      5. Status = Completed with at least one outcome row but zero count — LOW.
    """
    issues: list[Issue] = []
    try:
        studies = load_table("studies")
        if studies is None:
            return issues

        if "completion_date" in studies.columns:
            studies["completion_date"] = pd.to_datetime(
                studies["completion_date"], errors="coerce"
            )

        scoped = filter_trial_scope(
            studies, trial_id=trial_id,
            therapeutic_area=therapeutic_area, phase=phase,
            overall_status=overall_status, limit=limit,
        )
        if scoped.empty:
            return issues

        outcome_counts = load_table("outcome_counts")
        if outcome_counts is not None and "nct_id" in outcome_counts.columns:
            outcomes_by_trial = outcome_counts.groupby("nct_id")
            trials_with_outcomes = set(outcomes_by_trial.groups.keys())
        else:
            outcomes_by_trial = None
            trials_with_outcomes = set()

        now = pd.Timestamp.now().normalize()
        stale_threshold = now - pd.DateOffset(years=STALE_ACTIVE_YEARS)

        for _, row in scoped.iterrows():
            nct = row.get("nct_id")
            status = row.get("overall_status")
            end = row.get("completion_date")
            why_stopped = row.get("why_stopped")

            status_str = status.strip() if isinstance(status, str) else ""

            # Rule 1 — stopped without a reason
            if status_str in STOPPED_STATUSES and (
                pd.isna(why_stopped) or not str(why_stopped).strip()
            ):
                issues.append(Issue(
                    trial_id=nct,
                    check_name="stopped_missing_reason",
                    check_category=CATEGORY,
                    severity_rule="HIGH",
                    finding=(
                        f"Status is '{status_str}' but why_stopped is empty "
                        "(reporting requirement)."
                    ),
                    data_points={
                        "overall_status": status_str,
                        "why_stopped": None,
                    },
                    source_tables=["studies"],
                    source_columns=[
                        "nct_id", "overall_status", "why_stopped",
                    ],
                ))

            # Rule 2 — Recruiting status past planned completion
            if (
                status_str == "Recruiting"
                and pd.notna(end) and end < now
            ):
                days_past = (now - end).days
                issues.append(Issue(
                    trial_id=nct,
                    check_name="recruiting_past_completion",
                    check_category=CATEGORY,
                    severity_rule="HIGH",
                    finding=(
                        f"Status is 'Recruiting' but completion date "
                        f"{end.date().isoformat()} was {days_past} days ago."
                    ),
                    data_points={
                        "overall_status": status_str,
                        "completion_date": end.date().isoformat(),
                        "days_past_completion": int(days_past),
                    },
                    source_tables=["studies"],
                    source_columns=[
                        "nct_id", "overall_status", "completion_date",
                    ],
                ))

            # Rule 3 — Active not recruiting, long past completion
            if (
                status_str == "Active, not recruiting"
                and pd.notna(end) and end < stale_threshold
            ):
                years_past = round((now - end).days / 365.25, 1)
                issues.append(Issue(
                    trial_id=nct,
                    check_name="active_status_stale",
                    check_category=CATEGORY,
                    severity_rule="MEDIUM",
                    finding=(
                        f"Status is 'Active, not recruiting' but completion "
                        f"date was {years_past} years ago."
                    ),
                    data_points={
                        "overall_status": status_str,
                        "completion_date": end.date().isoformat(),
                        "years_past_completion": years_past,
                    },
                    source_tables=["studies"],
                    source_columns=[
                        "nct_id", "overall_status", "completion_date",
                    ],
                ))

            # Rule 4 — Completed but no posted outcomes
            if status_str == "Completed" and nct not in trials_with_outcomes:
                issues.append(Issue(
                    trial_id=nct,
                    check_name="completed_no_outcomes_posted",
                    check_category=CATEGORY,
                    severity_rule="HIGH",
                    finding=(
                        "Trial marked Completed but no outcome rows posted in "
                        "outcome_counts."
                    ),
                    data_points={
                        "overall_status": status_str,
                        "outcome_rows": 0,
                    },
                    source_tables=["studies", "outcome_counts"],
                    source_columns=[
                        "nct_id", "overall_status", "outcome_type", "count",
                    ],
                ))

            # Rule 5 — Completed with outcome rows but zero count on any
            elif (
                status_str == "Completed"
                and outcomes_by_trial is not None
                and nct in trials_with_outcomes
            ):
                trial_outcomes = outcomes_by_trial.get_group(nct)
                zero_count_rows = trial_outcomes[trial_outcomes["count"] <= 0]
                if not zero_count_rows.empty:
                    issues.append(Issue(
                        trial_id=nct,
                        check_name="completed_zero_outcome_count",
                        check_category=CATEGORY,
                        severity_rule="LOW",
                        finding=(
                            f"Completed trial has {len(zero_count_rows)} "
                            "outcome row(s) with count <= 0."
                        ),
                        data_points={
                            "overall_status": status_str,
                            "zero_count_rows": int(len(zero_count_rows)),
                            "outcome_types": zero_count_rows[
                                "outcome_type"
                            ].tolist(),
                        },
                        source_tables=["studies", "outcome_counts"],
                        source_columns=[
                            "nct_id", "overall_status",
                            "outcome_type", "count",
                        ],
                    ))

    except Exception as exc:  # noqa: BLE001 — graceful degradation
        log.exception("check_status_inconsistencies failed: %s", exc)

    log.info("status check produced %d issues", len(issues))
    return issues
