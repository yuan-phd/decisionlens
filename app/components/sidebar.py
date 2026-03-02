"""
sidebar.py — DecisionLENS global Streamlit sidebar filters.

``render_sidebar()`` builds the full left-rail filter panel and returns a
``SidebarState`` TypedDict so every page can read the user's selections
from a single, consistent dict.

Typical usage (in any page file)::

    from app.components.sidebar import render_sidebar

    state = render_sidebar()

    # state["condition_query"]  → str, free-text condition search
    # state["phases"]           → list[str], selected phase strings
    # state["sponsor_types"]    → list[str], selected sponsor categories
    # state["date_range"]       → tuple[date, date], (start, end) filter
    # state["min_enrollment"]   → int, minimum planned enrollment
    # state["max_enrollment"]   → int, maximum planned enrollment
    # state["data_dir"]         → str, processed data directory path
    # state["data_dir"]         → str, processed data directory path

This module has no side-effects at import time and does NOT import
Streamlit at module level so that unit tests and notebooks can import
``charts.py`` without triggering Streamlit's runtime.
"""

from __future__ import annotations

import datetime
from pathlib import Path
from typing import TypedDict

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

#: Default data directory (relative to project root).
DEFAULT_DATA_DIR: str = "data/processed"

#: All recognised AACT clinical phase strings (sorted naturally).
ALL_PHASES: list[str] = [
    "Phase 1",
    "Phase 1/Phase 2",
    "Phase 2",
    "Phase 2/Phase 3",
    "Phase 3",
    "Phase 4",
    "N/A",
]

#: Sponsor type categories derived from AACT ``sponsors.agency_class``.
ALL_SPONSOR_TYPES: list[str] = [
    "INDUSTRY",
    "NIH",
    "OTHER_GOV",
    "FED",
    "INDIV",
    "NETWORK",
    "OTHER",
    "UNKNOWN",
]

#: Date limits for the trial start-date filter.
DATE_MIN: datetime.date = datetime.date(1990, 1, 1)
DATE_MAX: datetime.date = datetime.date(2030, 12, 31)

#: Enrollment target slider limits.
ENROLLMENT_MIN: int = 0
ENROLLMENT_MAX: int = 50_000
ENROLLMENT_STEP: int = 50


# ---------------------------------------------------------------------------
# SidebarState type
# ---------------------------------------------------------------------------

class SidebarState(TypedDict):
    """Dict returned by ``render_sidebar()`` with all active filter values."""

    condition_query: str
    """Free-text condition/indication search string (may be empty)."""

    phases: list[str]
    """Selected phase strings from ALL_PHASES."""

    sponsor_types: list[str]
    """Selected sponsor type strings from ALL_SPONSOR_TYPES."""

    date_range: tuple[datetime.date, datetime.date]
    """(start, end) filter for trial start_date."""

    min_enrollment: int
    """Lower bound for planned enrollment (enrollment_type == ANTICIPATED/ACTUAL)."""

    max_enrollment: int
    """Upper bound for planned enrollment."""

    data_dir: str
    """Resolved path to the processed parquet data directory."""


# ---------------------------------------------------------------------------
# render_sidebar
# ---------------------------------------------------------------------------

def render_sidebar() -> SidebarState:
    """
    Render the DecisionLENS global sidebar and return the active filter state.

    This function must only be called from within a running Streamlit app
    (i.e. after ``import streamlit as st``). Importing this module is safe
    outside Streamlit; only calling ``render_sidebar()`` requires the runtime.

    Returns:
        SidebarState dict containing all user-selected filter values.

    Raises:
        ImportError: If ``streamlit`` is not installed.
    """
    try:
        import streamlit as st
    except ImportError as exc:  # pragma: no cover
        raise ImportError(
            "streamlit is required to call render_sidebar(). "
            "Install it with: pip install streamlit"
        ) from exc

    # ------------------------------------------------------------------
    # Logo / header
    # ------------------------------------------------------------------
    st.sidebar.markdown(
        """
        <div style="text-align:center; padding: 0.5rem 0 1rem 0;">
            <span style="font-size:2rem;">🔬</span><br>
            <span style="font-size:1.4rem; font-weight:700; color:#0077b6;">
                DecisionLENS
            </span><br>
            <span style="font-size:0.75rem; color:#888;">
                Clinical Trial Intelligence
            </span>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.sidebar.divider()

    # ------------------------------------------------------------------
    # 1. Condition / indication filter
    # ------------------------------------------------------------------
    st.sidebar.subheader("🔍 Condition / Indication")
    condition_query: str = st.sidebar.text_input(
        label="Search condition",
        value="",
        placeholder="e.g.  Breast Cancer",
        help=(
            "Case-insensitive substring search against the AACT conditions table. "
            "Leave blank to include all conditions."
        ),
    )

    # ------------------------------------------------------------------
    # 2. Trial phase filter
    # ------------------------------------------------------------------
    st.sidebar.subheader("📊 Trial Phase")
    phases: list[str] = st.sidebar.multiselect(
        label="Phases",
        options=ALL_PHASES,
        default=ALL_PHASES,
        help="Filter by clinical trial phase.  Select all for no filtering.",
    )
    if not phases:
        phases = ALL_PHASES[:]   # treat empty selection as "all"

    # ------------------------------------------------------------------
    # 3. Sponsor type filter
    # ------------------------------------------------------------------
    st.sidebar.subheader("🏢 Sponsor Type")
    _common_sponsors = ["INDUSTRY", "NIH", "OTHER_GOV", "OTHER"]
    sponsor_types: list[str] = st.sidebar.multiselect(
        label="Sponsor types",
        options=ALL_SPONSOR_TYPES,
        default=_common_sponsors,
        help=(
            "Filter by AACT agency_class.  "
            "INDUSTRY = pharma/biotech; NIH = US National Institutes of Health."
        ),
    )
    if not sponsor_types:
        sponsor_types = ALL_SPONSOR_TYPES[:]

    # ------------------------------------------------------------------
    # 4. Trial start-date range
    # ------------------------------------------------------------------
    st.sidebar.subheader("📅 Start Date Range")
    default_start = datetime.date(2010, 1, 1)
    default_end   = datetime.date.today()

    date_range: tuple[datetime.date, datetime.date] = st.sidebar.date_input(  # type: ignore[assignment]
        label="Start date window",
        value=(default_start, default_end),
        min_value=DATE_MIN,
        max_value=DATE_MAX,
        help="Include only trials whose start_date falls within this window.",
    )

    # Ensure we always have a 2-tuple (date_input may return a single date)
    if isinstance(date_range, datetime.date):
        date_range = (date_range, date_range)
    elif len(date_range) == 1:
        date_range = (date_range[0], date_range[0])

    # ------------------------------------------------------------------
    # 5. Enrollment size range
    # ------------------------------------------------------------------
    st.sidebar.subheader("👥 Enrollment Size")
    enroll_min, enroll_max = st.sidebar.slider(
        label="Planned enrollment (# patients)",
        min_value=ENROLLMENT_MIN,
        max_value=ENROLLMENT_MAX,
        value=(ENROLLMENT_MIN, 5_000),
        step=ENROLLMENT_STEP,
        help="Filter by the trial's planned/actual enrollment count.",
    )

    # ------------------------------------------------------------------
    # 6. Advanced — data directory
    # ------------------------------------------------------------------
    with st.sidebar.expander("⚙️ Advanced", expanded=False):
        data_dir_input: str = st.text_input(
            label="Data directory",
            value=DEFAULT_DATA_DIR,
            help=(
                "Path to the processed Parquet data directory created by "
                "setup_data.py.  Relative paths are resolved from the project root."
            ),
        )

        # Validate the path and warn if it doesn't exist
        data_dir_resolved = str(Path(data_dir_input).expanduser())
        if not Path(data_dir_resolved).exists():
            st.warning(
                f"Directory `{data_dir_input}` not found — "
                "run `python setup_data.py` to generate the data.",
                icon="⚠️",
            )

    st.sidebar.divider()
    st.sidebar.caption(
        "Data: AACT / ClinicalTrials.gov | "
        "Models: XGBoost + Cox PH | "
        "AI: Claude Opus 4.6"
    )

    # ------------------------------------------------------------------
    # Assemble state dict
    # ------------------------------------------------------------------
    return SidebarState(
        condition_query=condition_query.strip(),
        phases=phases,
        sponsor_types=sponsor_types,
        date_range=(date_range[0], date_range[1]),
        min_enrollment=int(enroll_min),
        max_enrollment=int(enroll_max),
        data_dir=data_dir_resolved,
    )
