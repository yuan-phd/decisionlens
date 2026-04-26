"""Check E — cross-field validation.

Flags inconsistencies that span more than one AACT table: eligibility
age inversion, healthy-volunteer mismatches with serious indications,
mismatch between ``calculated_values.number_of_facilities`` and the
``facilities`` table, and Phase 3 trials with fewer planned patients
than facilities recruiting them.
"""

from __future__ import annotations

import logging
from typing import Optional

import pandas as pd

from checks.models import Issue, filter_trial_scope, load_table

log = logging.getLogger(__name__)

CATEGORY = "crossfield"

# Indications that are inconsistent with healthy-volunteer enrollment.
SERIOUS_CONDITION_KEYWORDS = (
    "cancer", "carcinoma", "tumor", "tumour", "leukemia", "lymphoma",
    "melanoma", "myeloma", "alzheimer", "parkinson", "hiv",
    "hepatitis", "stroke", "myocardial", "heart failure",
)

CRITERIA_MIN_LENGTH = 100  # < 100 chars suggests insufficient detail


def check_crossfield_validation(
    trial_id: Optional[str] = None,
    therapeutic_area: Optional[str] = None,
    phase: Optional[str] = None,
    overall_status: Optional[str] = None,
    limit: int = 100,
) -> list[Issue]:
    """Run all cross-field validation rules over the scoped trial set.

    Rules:
      1. Eligibility ``minimum_age_num`` > ``maximum_age_num`` — HIGH.
      2. ``healthy_volunteers = Yes`` but conditions list a serious
         disease (cancer, HIV, neurodegenerative, etc.) — MEDIUM.
      3. ``calculated_values.number_of_facilities`` > 0 but no rows in
         ``facilities`` for that trial — HIGH.
      4. Phase 3 trial with planned ``enrollment`` < facility count — MEDIUM.
      5. Eligibility ``criteria`` text shorter than 100 chars — LOW.
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
        scoped_ids = set(scoped["nct_id"].tolist())

        eligibilities = load_table("eligibilities")
        eligibilities_indexed = (
            eligibilities.set_index("nct_id")
            if eligibilities is not None
            and "nct_id" in eligibilities.columns else None
        )

        calc_values = load_table("calculated_values")
        calc_indexed = (
            calc_values.set_index("nct_id")
            if calc_values is not None
            and "nct_id" in calc_values.columns else None
        )

        facilities = load_table("facilities")
        facility_counts = (
            facilities.groupby("nct_id").size()
            if facilities is not None and "nct_id" in facilities.columns
            else None
        )

        conditions = load_table("conditions")
        if conditions is not None and "nct_id" in conditions.columns:
            cond_by_trial: dict[str, list[str]] = (
                conditions[conditions["nct_id"].isin(scoped_ids)]
                .groupby("nct_id")["downcase_name"]
                .apply(list)
                .to_dict()
            )
        else:
            cond_by_trial = {}

        for _, row in scoped.iterrows():
            nct = row.get("nct_id")
            phase = row.get("phase")
            phase_str = phase.strip() if isinstance(phase, str) else ""
            planned = row.get("enrollment")

            elig = (
                eligibilities_indexed.loc[nct]
                if eligibilities_indexed is not None
                and nct in eligibilities_indexed.index else None
            )
            if isinstance(elig, pd.DataFrame):
                elig = elig.iloc[0]

            calc = (
                calc_indexed.loc[nct]
                if calc_indexed is not None and nct in calc_indexed.index
                else None
            )
            if isinstance(calc, pd.DataFrame):
                calc = calc.iloc[0]

            facility_count = (
                int(facility_counts.get(nct, 0))
                if facility_counts is not None else 0
            )

            # Rule 1 — age inversion (using parsed numeric ages)
            if calc is not None:
                min_age = calc.get("minimum_age_num")
                max_age = calc.get("maximum_age_num")
                if (
                    pd.notna(min_age) and pd.notna(max_age)
                    and min_age > max_age
                ):
                    issues.append(Issue(
                        trial_id=nct,
                        check_name="age_range_inverted",
                        check_category=CATEGORY,
                        severity_rule="HIGH",
                        finding=(
                            f"Eligibility minimum age ({min_age}) exceeds "
                            f"maximum age ({max_age})."
                        ),
                        data_points={
                            "minimum_age_num": float(min_age),
                            "maximum_age_num": float(max_age),
                        },
                        source_tables=["calculated_values"],
                        source_columns=[
                            "nct_id", "minimum_age_num", "maximum_age_num",
                        ],
                    ))

            # Rule 2 — healthy volunteers + serious indication
            if elig is not None:
                hv = elig.get("healthy_volunteers")
                if isinstance(hv, str) and hv.strip().lower() == "yes":
                    trial_conditions = cond_by_trial.get(nct, [])
                    matched = [
                        c for c in trial_conditions
                        if any(kw in c for kw in SERIOUS_CONDITION_KEYWORDS)
                    ]
                    if matched:
                        issues.append(Issue(
                            trial_id=nct,
                            check_name="healthy_volunteers_serious_condition",
                            check_category=CATEGORY,
                            severity_rule="MEDIUM",
                            finding=(
                                f"Trial accepts healthy volunteers but "
                                f"condition(s) include: {', '.join(matched)}."
                            ),
                            data_points={
                                "healthy_volunteers": hv,
                                "conditions": matched,
                            },
                            source_tables=["eligibilities", "conditions"],
                            source_columns=[
                                "nct_id", "healthy_volunteers", "downcase_name",
                            ],
                        ))

            # Rule 3 — calculated_values claims facilities, but none in facilities
            if calc is not None:
                claimed = calc.get("number_of_facilities")
                if (
                    pd.notna(claimed) and claimed > 0
                    and facility_counts is not None
                    and facility_count == 0
                ):
                    issues.append(Issue(
                        trial_id=nct,
                        check_name="facility_count_mismatch",
                        check_category=CATEGORY,
                        severity_rule="HIGH",
                        finding=(
                            f"calculated_values reports "
                            f"{int(claimed)} facilities but facilities table "
                            "has 0 rows for this trial."
                        ),
                        data_points={
                            "number_of_facilities_calc": float(claimed),
                            "facility_rows": 0,
                        },
                        source_tables=["calculated_values", "facilities"],
                        source_columns=[
                            "nct_id", "number_of_facilities", "country",
                        ],
                    ))

            # Rule 4 — Phase 3 with fewer planned patients than facilities
            if (
                phase_str == "Phase 3"
                and pd.notna(planned) and planned > 0
                and facility_count > 0
                and planned < facility_count
            ):
                issues.append(Issue(
                    trial_id=nct,
                    check_name="phase3_enrollment_below_site_count",
                    check_category=CATEGORY,
                    severity_rule="MEDIUM",
                    finding=(
                        f"Phase 3 trial planned enrollment ({int(planned)}) "
                        f"is less than facility count ({facility_count})."
                    ),
                    data_points={
                        "phase": phase_str,
                        "enrollment": float(planned),
                        "facility_count": facility_count,
                    },
                    source_tables=["studies", "facilities"],
                    source_columns=["nct_id", "phase", "enrollment"],
                ))

            # Rule 5 — eligibility criteria text very short
            if elig is not None:
                criteria = elig.get("criteria")
                if isinstance(criteria, str) and len(criteria) < CRITERIA_MIN_LENGTH:
                    issues.append(Issue(
                        trial_id=nct,
                        check_name="eligibility_criteria_too_short",
                        check_category=CATEGORY,
                        severity_rule="LOW",
                        finding=(
                            f"Eligibility criteria text is only "
                            f"{len(criteria)} characters (< "
                            f"{CRITERIA_MIN_LENGTH} threshold)."
                        ),
                        data_points={
                            "criteria_length": len(criteria),
                            "threshold": CRITERIA_MIN_LENGTH,
                        },
                        source_tables=["eligibilities"],
                        source_columns=["nct_id", "criteria"],
                    ))

    except Exception as exc:  # noqa: BLE001 — graceful degradation
        log.exception("check_crossfield_validation failed: %s", exc)

    log.info("crossfield check produced %d issues", len(issues))
    return issues
