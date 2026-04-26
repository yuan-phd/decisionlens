"""DecisionLENS v2 — Streamlit dashboard.

Four pages:
  1. Scan Control & Overview      — kick off a scan, show summary cards
  2. Prioritised Issues           — root-cause clusters + per-issue cards
  3. Trial Deep Dive              — per-trial multi-source breakdown
  4. Audit Log & Evaluation       — audit log + gold-set metrics

Run with:
    venv/bin/streamlit run app_v2.py

Most rendering primitives live in ``app_lib`` so each page reads as
narrative — see ``app_lib.py`` for severity badges, the persistent
banner, scan-state guards, and the issue/cluster expanders.
"""

from __future__ import annotations

import json
import logging
import sys
from collections import Counter
from pathlib import Path
from typing import Any

import streamlit as st
from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

load_dotenv()

st.set_page_config(
    page_title="DecisionLENS v2",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Project imports happen after st.set_page_config + load_dotenv so the
# heavy modules (orchestrator → openai/xgboost/pandas) only import once
# Streamlit has booted, and OPENAI_API_KEY is already in os.environ.
from agents.orchestrator import run_scan  # noqa: E402
from app_lib import (  # noqa: E402
    get_clusters,
    get_prioritised_issues,
    get_scan_report,
    get_scan_state,
    render_banner,
    render_cluster_expander,
    render_issue_expander,
    require_scan,
    severity_badge,
    severity_color,
    severity_sort_key,
)
from mcp_servers.aact_server import (  # noqa: E402
    tool_get_eligibility_criteria,
    tool_get_trial_details,
)
from mcp_servers.pubmed_server import tool_search_trial_publications  # noqa: E402
from models.risk_scorer import get_completion_risk_score  # noqa: E402

log = logging.getLogger(__name__)

PHASE_OPTIONS: list[str] = ["All", "Phase 1", "Phase 2", "Phase 3", "Phase 4"]
STATUS_OPTIONS: list[str] = [
    "All", "Recruiting", "Completed", "Active_Not_Recruiting", "Terminated",
]

METRICS_PATH: Path = PROJECT_ROOT / "evaluation" / "metrics_results.json"
PATTERNS_PATH: Path = PROJECT_ROOT / "output" / "patterns.json"


# ---------------------------------------------------------------------------
# PAGE 1 — Scan Control & Overview
# ---------------------------------------------------------------------------

def page_overview() -> None:
    """Scan Control & Overview — sidebar inputs, run button, summary cards."""
    st.title("DecisionLENS v2 — Agentic Data Quality")
    render_banner()

    with st.sidebar:
        st.header("Scan scope")
        therapeutic_area = st.text_input(
            "Therapeutic area", value="oncology",
        )
        phase = st.selectbox(
            "Phase", PHASE_OPTIONS, index=PHASE_OPTIONS.index("Phase 3"),
        )
        status_select = st.selectbox(
            "Trial Status", STATUS_OPTIONS, index=0,
        )
        limit = st.slider(
            "Trial limit", min_value=20, max_value=200, value=50, step=10,
        )

    st.subheader("Data sources")
    c1, c2, c3 = st.columns(3)
    with c1:
        st.success("AACT ✅")
    with c2:
        st.success("PubMed ✅")
    with c3:
        st.success("OpenFDA ✅")

    if st.button("▶  Run scan", type="primary"):
        scope = (
            f"{phase} {therapeutic_area} trials"
            if phase != "All"
            else f"{therapeutic_area} trials"
        )
        scope_filters: dict[str, Any] = {
            "therapeutic_area": therapeutic_area,
            "limit": limit,
        }
        if phase != "All":
            scope_filters["phase"] = phase
        if status_select != "All":
            scope_filters["overall_status"] = status_select

        with st.status("Agent running...", expanded=True) as status:
            try:
                status.write("Planning checks across AACT + PubMed + OpenFDA...")
                status.write("Running scan (~60s) — scanning trials, "
                             "computing risk scores, prioritising findings...")
                state, report = run_scan(scope, scope_filters)
                st.session_state["scan_state"] = state
                st.session_state["scan_report"] = report
                status.write(
                    f"Scan complete — "
                    f"{len(state.get('prioritised_issues') or [])} issue(s) "
                    f"found across "
                    f"{len({i['trial_id'] for i in state.get('prioritised_issues') or []})} "
                    f"trial(s)."
                )
                status.update(label="Scan complete", state="complete")
            except Exception as exc:  # noqa: BLE001
                status.update(label=f"Scan failed: {exc}", state="error")
                st.error(f"Scan failed: {exc}")
                log.exception("Scan failed")
                return

        st.success("Scan complete.")

    state = get_scan_state()
    report = get_scan_report()
    if not state:
        st.info("No scan results yet. Configure scope in the sidebar and click "
                "**Run scan**.")
        return

    fr = state.get("final_report") or {}
    sev = fr.get("severity_counts") or {}
    prioritised = state.get("prioritised_issues") or []
    trials = {p["trial_id"] for p in prioritised}
    meaningful_clusters = get_clusters()

    st.subheader("Scan summary")
    cols = st.columns(6)
    cols[0].metric("Trials Scanned", len(trials))
    cols[1].metric("CRITICAL", int(sev.get("CRITICAL", 0)))
    cols[2].metric("HIGH", int(sev.get("HIGH", 0)))
    cols[3].metric("MEDIUM", int(sev.get("MEDIUM", 0)))
    cols[4].metric("LOW", int(sev.get("LOW", 0)))
    cols[5].metric("Root Cause Clusters", len(meaningful_clusters))

    llmops = (fr.get("llmops_summary") or {})
    with st.expander("LLM Usage"):
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("LLM Calls", llmops.get("total_llm_calls", 0))
        c2.metric("Total Tokens", llmops.get("total_tokens", 0))
        c3.metric(
            "Cost ($)",
            f"{llmops.get('total_estimated_cost_usd', 0.0):.4f}",
        )
        latency_s = float(llmops.get("total_latency_ms", 0.0)) / 1000.0
        c4.metric("Latency (s)", f"{latency_s:.1f}")

    if report is not None and hasattr(report, "scan_id"):
        st.caption(
            f"scan_id `{report.scan_id}` · saved to "
            f"`output/reports/{report.scan_id}.json`"
        )


# ---------------------------------------------------------------------------
# PAGE 2 — Prioritised Issues
# ---------------------------------------------------------------------------

def page_issues() -> None:
    """Prioritised Issues — clusters on top, individual issues below."""
    st.title("Prioritised issues")
    render_banner()

    if not require_scan():
        return

    prioritised = get_prioritised_issues()
    clusters = get_clusters()

    # --- sidebar filters --------------------------------------------------
    with st.sidebar:
        st.header("Filter")
        all_severities = sorted(
            {p.get("severity", "LOW") for p in prioritised},
            key=severity_sort_key,
        )
        all_categories = sorted(
            {p.get("check_category", "?") for p in prioritised}
        )
        sev_pick = st.multiselect(
            "Severity", all_severities, default=all_severities,
        )
        cat_pick = st.multiselect(
            "Check category", all_categories, default=all_categories,
        )

    # --- clusters ---------------------------------------------------------
    st.subheader("Root cause clusters")
    if not clusters:
        st.caption("No meaningful clusters surfaced this scan.")
    else:
        for c in clusters:
            render_cluster_expander(c)

    # --- individual issues ------------------------------------------------
    st.subheader("Individual findings")
    filtered = [
        p for p in prioritised
        if p.get("severity") in sev_pick
        and p.get("check_category") in cat_pick
    ]
    # Already CRITICAL→LOW from get_prioritised_issues; resort with
    # confidence as the secondary key.
    filtered.sort(
        key=lambda p: (
            severity_sort_key(p.get("severity", "LOW")),
            -float(p.get("confidence", 0.0) or 0.0),
        ),
    )
    st.caption(
        f"Showing {len(filtered)} of {len(prioritised)} prioritised issue(s)."
    )

    for issue in filtered:
        render_issue_expander(issue)

    # --- export full report JSON -----------------------------------------
    report = get_scan_report()
    if report is not None and hasattr(report, "to_dict"):
        st.divider()
        report_json = json.dumps(report.to_dict(), indent=2, default=str)
        st.download_button(
            "⬇ Download Full Report (JSON)",
            data=report_json,
            file_name=f"scan_{report.scan_id}.json",
            mime="application/json",
        )


# ---------------------------------------------------------------------------
# PAGE 3 — Trial Deep Dive
# ---------------------------------------------------------------------------

def page_deep_dive() -> None:
    """Per-trial multi-source breakdown."""
    st.title("Trial deep dive")
    render_banner()

    if not require_scan():
        return

    prioritised = get_prioritised_issues()
    flagged_ids = sorted({p["trial_id"] for p in prioritised})
    if not flagged_ids:
        st.warning("No flagged trials in the most recent scan.")
        return

    nct_id = st.selectbox("Trial", flagged_ids)
    if not nct_id:
        return

    # --- Panel 1: AACT trial metadata ------------------------------------
    st.subheader("Trial metadata (AACT)")
    details = tool_get_trial_details(nct_id)
    if details.get("status") == "ok":
        d = details.get("details", {}) or {}
        c1, c2, c3 = st.columns(3)
        c1.write(f"**NCT ID:** {nct_id}")
        c1.write(f"**Status:** {d.get('overall_status', '—')}")
        c1.write(f"**Phase:** {d.get('phase', '—')}")
        c2.write(f"**Sponsor:** {d.get('source', '—')}")
        c2.write(f"**Enrollment:** {d.get('enrollment', '—')}")
        c2.write(f"**Type:** {d.get('study_type', '—')}")
        c3.write(f"**Start:** {d.get('start_date', '—')}")
        c3.write(f"**Completion:** {d.get('completion_date', '—')}")
        c3.write(
            f"**Last update:** {d.get('last_update_submitted_date', '—')}"
        )
        with st.expander("Full AACT row"):
            st.json(d)
    else:
        st.warning(
            f"AACT lookup failed: {details.get('reason', 'unknown error')}"
        )

    # --- Panel 2: Risk model ---------------------------------------------
    st.subheader("v1 completion risk model")
    risk = get_completion_risk_score(nct_id)
    st.metric("Completion Risk Score", f"{risk['risk_score']:.3f}")
    st.caption(
        f"Reliable: {risk['risk_reliable']} · "
        f"{risk['risk_model_version']}"
    )
    if not risk["risk_reliable"]:
        st.warning(risk.get("note", "Risk score unreliable."))

    # --- Panel 3: Cross-source context -----------------------------------
    st.subheader("Cross-source context")
    pub_col, fda_col = st.columns(2)

    with pub_col:
        st.markdown("**PubMed**")
        with st.spinner("Searching PubMed..."):
            pubs = tool_search_trial_publications(nct_id, max_results=5)
        if pubs.get("status") == "ok":
            n = pubs.get("total_found", 0)
            st.write(f"Publications found: {n}")
            for pub in pubs.get("publications", [])[:5]:
                title = pub.get("title", "(untitled)")
                pmid = pub.get("pmid", "?")
                st.write(f"- [{pmid}] {title}")
        else:
            st.caption(
                f"PubMed unavailable: {pubs.get('reason', 'unknown error')}"
            )

    with fda_col:
        st.markdown("**OpenFDA** (from issue provenance)")
        fda_data: dict[str, Any] = {}
        for p in prioritised:
            if p["trial_id"] != nct_id:
                continue
            fda = (p.get("provenance") or {}).get("openfda") or {}
            if fda.get("drug_queried"):
                fda_data = fda
                break
        if fda_data:
            st.write(f"Drug queried: `{fda_data.get('drug_queried', '—')}`")
            st.write(
                f"Total adverse events: "
                f"{fda_data.get('total_adverse_events', 0)}"
            )
            st.write(
                f"Serious events: {fda_data.get('serious_events', 0)}"
            )
        else:
            st.caption("No OpenFDA cross-source data attached to this trial.")

    # --- Panel 4: All issues for this trial ------------------------------
    st.subheader("All issues for this trial")
    trial_issues = [p for p in prioritised if p["trial_id"] == nct_id]
    trial_issues.sort(
        key=lambda p: severity_sort_key(p.get("severity", "LOW")),
    )
    for p in trial_issues:
        sev = p.get("severity", "LOW")
        with st.expander(f"{severity_color(sev)} {p.get('check_name', '?')}"):
            st.write(f"**Severity:** {severity_badge(sev)}")
            st.write(f"**Finding:** {p.get('finding', '—')}")
            st.write(f"**Explanation:** {p.get('explanation', '—')}")
            st.write(
                f"**Suggested action:** {p.get('suggested_action', '—')}"
            )
            st.json(p.get("data_points") or {})

    # --- Panel 5: Eligibility criteria -----------------------------------
    with st.expander("Eligibility criteria"):
        elig = tool_get_eligibility_criteria(nct_id)
        if elig.get("status") == "ok":
            st.write(f"**Gender:** {elig.get('gender', '—')}")
            st.write(
                f"**Age:** {elig.get('minimum_age', '—')} – "
                f"{elig.get('maximum_age', '—')}"
            )
            st.write(
                f"**Healthy volunteers:** "
                f"{elig.get('healthy_volunteers', '—')}"
            )
            criteria = elig.get("criteria") or "(no criteria text)"
            st.text_area(
                "Criteria text", criteria, height=300,
                key=f"elig_{nct_id}",
            )
        else:
            st.caption(
                f"Eligibility unavailable: "
                f"{elig.get('reason', 'unknown error')}"
            )


# ---------------------------------------------------------------------------
# PAGE 4 — Audit Log & Evaluation
# ---------------------------------------------------------------------------

def _audit_tab() -> None:
    """Audit log + LLMOps per-call breakdown + report download."""
    if not require_scan():
        return

    state = get_scan_state()
    report = get_scan_report()
    audit = state.get("llm_calls") or []
    if audit:
        st.markdown("**Audit log — every LLM call**")
        rows = [
            {
                "purpose": c.get("purpose"),
                "model": c.get("model"),
                "tokens": (
                    c.get("prompt_tokens", 0) + c.get("completion_tokens", 0)
                ),
                "cost_usd": c.get("estimated_cost_usd", 0.0),
                "latency_ms": c.get("latency_ms", 0.0),
                "status": c.get("status", "ok"),
                "call_id": (c.get("call_id") or "")[:8],
            }
            for c in audit
        ]
        st.dataframe(rows, width="stretch")
        st.caption(
            f"{len(audit)} call(s) · "
            f"by purpose: {dict(Counter(c.get('purpose') for c in audit))}"
        )
    else:
        st.caption("No LLM calls recorded for this scan.")

    if report is not None and hasattr(report, "to_dict"):
        report_dict = report.to_dict()
        st.download_button(
            "⬇ Download full report (JSON)",
            data=json.dumps(report_dict, indent=2, default=str),
            file_name=f"{report.scan_id}.json",
            mime="application/json",
        )


def _evaluation_tab() -> None:
    """Show stored gold-set metrics + EVALUATION_NOTE + pattern library."""
    if not METRICS_PATH.exists():
        st.info(
            "No metrics_results.json yet. Run "
            "`venv/bin/python -m evaluation.metrics` to generate it."
        )
    else:
        try:
            metrics = json.loads(METRICS_PATH.read_text())
        except Exception as exc:  # noqa: BLE001
            st.error(f"Failed to parse metrics file: {exc}")
            metrics = None

        if metrics:
            c1, c2, c3 = st.columns(3)
            c1.metric("Accuracy", f"{metrics.get('accuracy', 0.0):.3f}")
            c2.metric(
                "Cohen's κ", f"{metrics.get('cohens_kappa', 0.0):.3f}",
            )
            c3.metric("Match rate", f"{metrics.get('match_rate', 0.0):.1%}")

            per_sev = metrics.get("per_severity") or {}
            if per_sev:
                rows = [
                    {
                        "severity": sev,
                        "precision": v.get("precision"),
                        "recall": v.get("recall"),
                        "f1": v.get("f1"),
                        "support": v.get("support"),
                    }
                    for sev, v in per_sev.items()
                ]
                st.markdown("**Per-severity metrics**")
                st.dataframe(rows, width="stretch")

            note = metrics.get("evaluation_note", "")
            if note:
                st.caption(note)

    if PATTERNS_PATH.exists():
        try:
            patterns = json.loads(PATTERNS_PATH.read_text())
            with st.expander("Pattern library (output/patterns.json)"):
                st.json(patterns)
        except Exception as exc:  # noqa: BLE001
            st.caption(f"patterns.json present but unreadable: {exc}")


def page_audit_eval() -> None:
    """Page 4 — Audit Log + Evaluation in two tabs."""
    st.title("Audit log & evaluation")
    render_banner()

    tab1, tab2 = st.tabs(["Audit Log", "Evaluation"])
    with tab1:
        _audit_tab()
    with tab2:
        _evaluation_tab()


# ---------------------------------------------------------------------------
# Router
# ---------------------------------------------------------------------------

PAGES: dict[str, Any] = {
    "1 · Scan Control & Overview": page_overview,
    "2 · Prioritised Issues":      page_issues,
    "3 · Trial Deep Dive":         page_deep_dive,
    "4 · Audit Log & Evaluation":  page_audit_eval,
}


def main() -> None:
    """Streamlit entry point — render the selected page."""
    with st.sidebar:
        st.markdown("### DecisionLENS v2")
        page_choice = st.radio("Navigate", list(PAGES.keys()))
    PAGES[page_choice]()


if __name__ == "__main__":
    main()
