"""Shared UI helpers for ``app_v2.py`` Streamlit pages.

Keeping these helpers in one module means each page in ``app_v2.py``
stays narrative — describing *what this page shows* — while the visual
primitives (severity badges, expanders, scan-state guards) live here.

Conventions:
* Functions that touch Streamlit import ``streamlit`` lazily so this
  module can also be imported by non-UI code (tests, evaluation
  scripts) without paying Streamlit's import cost.
* All session-state access goes through the ``has_scan`` /
  ``get_scan_state`` / ``get_scan_report`` trio so pages never have to
  guard for None themselves.
"""

from __future__ import annotations

from typing import Any


# ---------------------------------------------------------------------------
# 1. Severity helpers
# ---------------------------------------------------------------------------

def severity_color(severity: str) -> str:
    """Return a coloured-circle emoji for a severity label."""
    return {"CRITICAL": "🔴", "HIGH": "🟠",
            "MEDIUM": "🟡", "LOW": "🟢"}.get(severity, "⚪")


def severity_sort_key(severity: str) -> int:
    """Return a sort key so CRITICAL comes first, unknowns last."""
    return {"CRITICAL": 0, "HIGH": 1,
            "MEDIUM": 2, "LOW": 3}.get(severity, 4)


def severity_badge(severity: str) -> str:
    """Return ``"<emoji> SEVERITY"`` — used in markdown headings."""
    return f"{severity_color(severity)} {severity}"


# ---------------------------------------------------------------------------
# Issue-rendering helpers
# ---------------------------------------------------------------------------

# Human-readable display names for the rule-engine ``check_name`` slug —
# used in expander headers so the UI doesn't expose snake_case to the
# user. Anything not listed here falls back to title-cased slug.
CHECK_DISPLAY_NAMES: dict[str, str] = {
    "missing_primary_outcome":         "Missing Primary Outcome",
    "recruiting_past_completion":      "Recruiting Past Completion",
    "latephase_missing_design_field":  "Missing Design Field",
    "phase3_underpowered":             "Underpowered Phase 3 Trial",
    "completed_with_future_end":       "Future Completion Date",
    "completed_no_secondary_outcomes": "No Secondary Outcomes",
    "stopped_missing_reason":          "Stopped Without Reason",
    "active_stale":                    "Stale Active Trial",
    "active_status_stale":             "Stale Active Trial",
    "completed_no_outcomes_posted":    "No Outcomes Posted",
    "age_inverted":                    "Inverted Age Range",
    "zero_aes_with_fda_signals":       "Zero AEs with FDA Signals",
    "incomplete_ae_profile_vs_fda":    "Incomplete AE Profile",
    "trial_safety_aligns_with_fda":    "Safety Confirmed by FDA",
    "completed_no_publications":       "No Publications Found",
}


def friendly_check_name(check_name: str) -> str:
    """Return a human-readable label for a snake_case check name."""
    if not check_name:
        return ""
    if check_name in CHECK_DISPLAY_NAMES:
        return CHECK_DISPLAY_NAMES[check_name]
    return check_name.replace("_", " ").title()


def _risk_descriptor(risk_score: float, risk_reliable: bool) -> str:
    """Translate (risk_score, risk_reliable) into a prose phrase.

    The v1 model output is a probability in [0, 1] of *not* completing.
    These bands are coarse on purpose — the user reads narrative, not
    decimals — and align with the escalation thresholds (0.6 / 0.7).
    """
    if not risk_reliable:
        return "data unavailable for this trial"
    s = float(risk_score)
    if s >= 0.9:
        return f"score {s:.3f} — extreme failure risk"
    if s >= 0.6:
        return f"score {s:.3f} — high failure risk"
    if s >= 0.3:
        return f"score {s:.3f} — elevated risk"
    if s >= 0.1:
        return f"score {s:.3f} — low risk"
    return f"score {s:.3f} — very low risk"


def _confidence_label(confidence: float) -> str:
    """Translate the LLM's confidence float into a qualitative label."""
    c = float(confidence)
    if c >= 0.8:
        return "High — multiple sources confirm"
    if c >= 0.6:
        return "Moderate — single source assessment"
    return "Limited — insufficient data"


def _why_severity_paragraph(
    severity: str, finding: str, check_name: str,
    risk_score: float, risk_reliable: bool,
) -> str:
    """Build the 'Why this severity?' prose for layer 2.

    Combines what the rule engine matched, what the v1 risk model said,
    and the final post-escalation severity into one short paragraph.
    """
    rule_part = (
        f"Rule engine matched **{friendly_check_name(check_name)}** — "
        f"{finding}"
    )
    risk_part = (
        f"Completion risk model: "
        f"{_risk_descriptor(risk_score, risk_reliable)}"
        + (
            "."
            if risk_reliable
            else " — severity is based on the rule engine alone."
        )
    )
    final_part = f"Combined assessment: **{severity}**."
    return f"{rule_part} {risk_part} {final_part}"


# ---------------------------------------------------------------------------
# 2. Persistent banner
# ---------------------------------------------------------------------------

def render_banner() -> None:
    """Render the persistent data-source caption shown on every page."""
    import streamlit as st
    st.caption(
        "📊 Data: AACT + PubMed + OpenFDA "
        "(proxy for EDC/CTMS) · "
        "On-demand scan (production: event-triggered) · "
        "Trial-level data (production: patient-visit level)"
    )


# ---------------------------------------------------------------------------
# 3. Session-state helpers
# ---------------------------------------------------------------------------

def has_scan() -> bool:
    """Return True iff a scan has been run in this session."""
    import streamlit as st
    return (
        "scan_state" in st.session_state
        and "scan_report" in st.session_state
        and st.session_state["scan_state"] is not None
    )


def get_scan_state() -> dict:
    """Return the cached scan state dict (or ``{}`` if no scan yet)."""
    import streamlit as st
    return st.session_state.get("scan_state") or {}


def get_scan_report() -> Any:
    """Return the cached ScanReport (object or dict), or ``None``."""
    import streamlit as st
    return st.session_state.get("scan_report", None)


def get_prioritised_issues() -> list[dict]:
    """Return prioritised issues sorted CRITICAL → LOW."""
    state = get_scan_state()
    issues = state.get("prioritised_issues", []) or []
    return sorted(
        issues,
        key=lambda x: severity_sort_key(x.get("severity", "LOW")),
    )


def get_clusters() -> list[dict]:
    """Return root-cause clusters (excluding RC_UNCLUSTERED), sorted by severity.

    Defensive against both shapes the orchestrator can hand back —
    a ``ScanReport`` dataclass instance *or* a plain dict. Output is
    sorted CRITICAL → LOW so the highest-priority cluster surfaces
    first in the rendered list.
    """
    report = get_scan_report()
    if not report:
        return []
    if hasattr(report, "root_cause_clusters"):
        clusters = report.root_cause_clusters
    elif isinstance(report, dict):
        clusters = report.get("root_cause_clusters", [])
    else:
        return []
    meaningful = [c for c in (clusters or [])
                  if c.get("cluster_id") != "RC_UNCLUSTERED"]
    return sorted(
        meaningful,
        key=lambda c: severity_sort_key(c.get("cluster_severity", "LOW")),
    )


# ---------------------------------------------------------------------------
# 4. Issue renderer
# ---------------------------------------------------------------------------

def render_issue_expander(
    issue: dict,
    show_draft_query: bool = True,
    demo_mode: bool = False,
) -> None:
    """Render one prioritised issue as a Streamlit expander.

    ``show_draft_query`` controls whether the GPT-4o-mini "Draft Query
    Letter" button is shown — pages embedded in dense lists may want to
    suppress it to keep the rerun cost predictable.

    ``demo_mode=True`` (used by the read-only ``app_demo.py``) replaces
    the Draft Query button with a caption noting that draft generation
    requires the live app — no OpenAI calls are made on the demo host.
    """
    import streamlit as st
    from openai import OpenAI
    from dotenv import load_dotenv
    load_dotenv()

    trial_id = issue.get("trial_id", "")
    check_name = issue.get("check_name", "")
    severity = issue.get("severity", "LOW")
    finding = issue.get("finding", "")
    suggested_action = issue.get("suggested_action", "")
    confidence = float(issue.get("confidence", 0.0) or 0.0)
    risk_score = float(issue.get("completion_risk_score", 0.5) or 0.5)
    provenance = issue.get("provenance", {}) or {}

    # risk_reliable lives on provenance, not the top-level issue dict —
    # see orchestrator._build_prioritised_entry.
    risk_reliable = bool(provenance.get("risk_reliable", False))
    sources = provenance.get("sources_queried", []) or []

    label = (
        f"{severity_color(severity)} {severity} — "
        f"{trial_id} · {friendly_check_name(check_name)}"
    )

    with st.expander(label):
        # ── LAYER 1 — surface (visible immediately on expand) ──────────────
        st.info(f"**Finding:** {finding}")
        if suggested_action:
            st.warning(f"**Suggested action:** {suggested_action}")

        st.divider()

        # ── LAYER 2 — explanation + provenance ─────────────────────────────
        st.markdown("**Why this severity?**")
        st.write(_why_severity_paragraph(
            severity, finding, check_name, risk_score, risk_reliable,
        ))

        st.markdown("**Sources checked:**")
        all_sources = ["aact", "pubmed", "openfda"]
        badge_line = " · ".join(
            f"{src.upper()} {'✅' if src in sources else '◻️'}"
            for src in all_sources
        )
        st.markdown(badge_line)

        st.markdown("**Assessment confidence:**")
        st.write(f"{_confidence_label(confidence)}  *(score {confidence:.2f})*")

        # Engineering detail — kept available but folded away.
        data_points = issue.get("data_points", {}) or {}
        if data_points:
            with st.expander("Data points"):
                st.json(data_points)

        with st.expander("Provenance"):
            st.json(provenance)

        if show_draft_query and demo_mode:
            st.caption(
                "📝 Draft query generation available in live mode."
            )
        elif show_draft_query:
            btn_key = f"draft_{trial_id}_{check_name}"
            if st.button("📝 Draft Query Letter", key=btn_key):
                with st.spinner("Drafting query..."):
                    try:
                        client = OpenAI()
                        prompt = (
                            f"Draft a formal data query letter to the "
                            f"trial site:\n"
                            f"Trial: {trial_id}\n"
                            f"Issue: {finding}\n"
                            f"Severity: {severity}\n"
                            f"Action needed: {suggested_action}\n\n"
                            f"Format as professional letter from "
                            f"Clinical Data Manager. Include specific "
                            f"discrepancy, request clarification, "
                            f"10 business day deadline. Under 200 words."
                        )
                        response = client.chat.completions.create(
                            model="gpt-4o-mini",
                            messages=[
                                {"role": "user", "content": prompt}
                            ],
                            max_tokens=300,
                        )
                        st.text_area(
                            "Query Letter",
                            response.choices[0].message.content,
                            height=300,
                            key=f"letter_{btn_key}",
                        )
                    except Exception as e:  # noqa: BLE001
                        st.error(f"Failed to generate query: {e}")


# ---------------------------------------------------------------------------
# 5. Cluster renderer
# ---------------------------------------------------------------------------

def render_cluster_expander(cluster: dict) -> None:
    """Render one root-cause cluster as a Streamlit expander."""
    import streamlit as st
    cluster_id = cluster.get("cluster_id", "")
    root_cause = cluster.get("root_cause", "")
    severity = cluster.get("cluster_severity", "LOW")
    trials = cluster.get("affected_trials", []) or []
    action = cluster.get("recommended_action", "")
    count = cluster.get(
        "issue_count", len(cluster.get("issue_indices", []) or []),
    )

    label = (
        f"{severity_color(severity)} "
        f"{cluster_id} — {root_cause} "
        f"({count} issues, {len(trials)} trials)"
    )

    with st.expander(label):
        col1, col2 = st.columns(2)
        with col1:
            st.markdown(f"**Severity:** {severity_badge(severity)}")
            st.markdown(f"**Issues:** {count}")
        with col2:
            st.markdown(f"**Trials affected:** {len(trials)}")

        st.markdown("**Recommended Action:**")
        st.warning(action)
        st.markdown("**Affected Trials:**")
        st.code(", ".join(trials))


# ---------------------------------------------------------------------------
# 6. No-scan warning
# ---------------------------------------------------------------------------

def require_scan() -> bool:
    """Render an info card and return False when no scan has run yet."""
    import streamlit as st
    if not has_scan():
        st.info(
            "🔍 No scan results yet. "
            "Go to Page 1 and run a scan first."
        )
        return False
    return True
