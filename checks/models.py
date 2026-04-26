"""Shared data models and helpers for the data quality rule engine.

Defines the ``Issue`` dataclass returned by every check function, plus a
small loader/filter helper so every check can resolve AACT parquet files
relative to the project root and apply the standard
``trial_id`` / ``therapeutic_area`` / ``limit`` parameters consistently.
"""

from __future__ import annotations

import logging
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import pandas as pd

log = logging.getLogger(__name__)

# Project root resolved from this file's location: checks/models.py -> repo root.
PROJECT_ROOT: Path = Path(__file__).resolve().parent.parent
DATA_DIR: Path = PROJECT_ROOT / "data" / "processed"


@dataclass
class Issue:
    """A single data quality finding produced by a deterministic check.

    Attributes mirror the ``Issue`` spec in CLAUDE.md. ``severity_rule`` is
    the rule-engine severity *before* the LLM prioritisation node may
    upgrade or downgrade it. ``finding`` should be factual (no
    interpretation); interpretation belongs in the prioritisation step.
    """

    trial_id: str
    check_name: str
    check_category: str        # temporal | enrollment | status | endpoint | crossfield
    severity_rule: str         # HIGH | MEDIUM | LOW
    finding: str               # factual statement, no interpretation
    data_points: dict          # raw values that triggered the flag
    source_tables: list[str]
    source_columns: list[str]
    timestamp: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )

    def to_dict(self) -> dict:
        """Return a JSON-serialisable dict view of this issue."""
        return asdict(self)


# Keyword-based therapeutic area filter. Conservative — case-insensitive
# substring match against ``conditions.downcase_name``.
THERAPEUTIC_AREA_KEYWORDS: dict[str, tuple[str, ...]] = {
    "oncology": (
        "cancer", "carcinoma", "tumor", "tumour", "leukemia", "lymphoma",
        "melanoma", "sarcoma", "myeloma", "neoplasm", "oncolog",
    ),
    "cardiology": (
        "cardio", "heart", "atrial", "ventricular", "hypertension",
        "myocardial", "coronary", "arrhythmia",
    ),
    "neurology": (
        "alzheimer", "parkinson", "epilepsy", "multiple sclerosis",
        "stroke", "dementia", "neurolog",
    ),
    "immunology": (
        "psoriasis", "rheumatoid", "crohn", "ulcerative colitis",
        "autoimmune", "lupus",
    ),
    "respiratory": (
        "asthma", "copd", "chronic obstructive pulmonary", "pulmonary",
    ),
    "infectious_disease": (
        "hiv", "hepatitis", "tuberculosis", "covid", "infection",
    ),
    "endocrinology": (
        "diabetes", "thyroid", "obesity",
    ),
    "psychiatry": (
        "depress", "anxiety", "schizophrenia", "bipolar",
    ),
}


def load_table(name: str) -> Optional[pd.DataFrame]:
    """Load an AACT parquet table by short name (e.g. ``"studies"``).

    Returns ``None`` and logs a warning if the file is missing or unreadable
    so callers can degrade gracefully (per CLAUDE.md error-handling rule).
    """
    path = DATA_DIR / f"{name}.parquet"
    try:
        if not path.exists():
            log.warning("AACT table not found: %s", path)
            return None
        return pd.read_parquet(path)
    except Exception as exc:  # noqa: BLE001 — graceful degradation
        log.exception("Failed to read parquet %s: %s", path, exc)
        return None


def filter_trial_scope(
    studies: pd.DataFrame,
    trial_id: Optional[str] = None,
    therapeutic_area: Optional[str] = None,
    phase: Optional[str] = None,
    overall_status: Optional[str] = None,
    limit: int = 100,
) -> pd.DataFrame:
    """Apply the standard scope filters used by every check function.

    Order: ``trial_id`` (exact match) → ``therapeutic_area`` (keyword match
    against ``conditions.downcase_name``) → ``phase`` (exact match against
    ``studies.phase``) → ``overall_status`` (case-insensitive match against
    ``studies.overall_status``) → ``limit`` (head N rows).

    Phase and status filters apply *before* ``head()`` — this matters when
    matches are not balanced in the natural row order (e.g., the first 5k
    oncology rows in real AACT contain zero Phase 3 trials, so
    post-filtering would always return empty). The status filter is
    case-insensitive because AACT raw uses uppercase
    (``COMPLETED`` / ``RECRUITING``) while ``_normalise_studies_df``
    title-cases values, so callers may pass either form.

    Unknown therapeutic areas or missing tables degrade to a no-op for that
    filter rather than raising.
    """
    df = studies
    if trial_id is not None:
        df = df[df["nct_id"] == trial_id]
        return df.head(limit) if limit else df

    if therapeutic_area is not None:
        keywords = THERAPEUTIC_AREA_KEYWORDS.get(therapeutic_area.lower())
        if keywords is None:
            log.warning(
                "Unknown therapeutic_area %r; no filter applied. Known: %s",
                therapeutic_area, sorted(THERAPEUTIC_AREA_KEYWORDS),
            )
        else:
            conditions = load_table("conditions")
            if conditions is None or "downcase_name" not in conditions.columns:
                log.warning(
                    "Cannot filter by therapeutic_area: conditions table unavailable."
                )
            else:
                pattern = "|".join(keywords)
                matched = conditions[
                    conditions["downcase_name"].str.contains(
                        pattern, case=False, na=False, regex=True
                    )
                ]["nct_id"].unique()
                df = df[df["nct_id"].isin(matched)]

    if phase is not None and "phase" in df.columns:
        df = df[df["phase"] == phase]

    if overall_status is not None and "overall_status" in df.columns:
        target = str(overall_status).strip().upper()
        df = df[df["overall_status"].astype(str).str.strip().str.upper() == target]

    if limit and len(df) > limit:
        df = df.head(limit)
    return df
