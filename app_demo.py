"""DecisionLENS v2 — read-only report viewer for Streamlit Cloud.

Parallel to ``app_v2.py`` but with the live agent stripped out: it
loads the most-recently-saved scan report from ``output/reports/`` and
renders the same 4-page UI without ever calling OpenAI, the agent, or
the OpenFDA / PubMed APIs from the page handlers.

Run with:
    venv/bin/streamlit run app_demo.py

The deployment target (Streamlit Cloud) gets the saved JSON report,
the v1 model artefacts (for Page 3 risk scoring), AACT parquet files
(for Page 3 trial details), and the cached MCP API responses — all
checked into the repo. No API keys are needed; if the cache or
parquets are missing for a given trial, the existing code paths
degrade gracefully ("trial details unavailable", etc.).
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
    page_title="DecisionLENS v2 — Demo",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Project imports happen after st.set_page_config so heavy modules
# (sklearn / xgboost / pandas) only import once Streamlit has booted.
from app_lib import (  # noqa: E402
    get_clusters,
    get_prioritised_issues,
    get_scan_report,
    get_scan_state,
    render_banner,
    render_cluster_expander,
    render_issue_expander,
    severity_badge,
    severity_color,
    severity_sort_key,
)
from output.report import ScanReport  # noqa: E402

log = logging.getLogger(__name__)

DEMO_DIR: Path = PROJECT_ROOT / "output" / "demo"
METRICS_PATH: Path = PROJECT_ROOT / "evaluation" / "metrics_results.json"
PATTERNS_PATH: Path = PROJECT_ROOT / "output" / "patterns.json"

DEMO_BANNER = (
    "🔍 **Interactive report viewer** — displaying pre-computed scan "
    "results. Live scanning available locally via "
    "[GitHub](https://github.com/yuan-phd/decisionlens)."
)

# Pre-computed reports produced by ``pre_warm_demo.py``. Each entry
# embeds a ``trial_cache`` so Page 3 can render trial details without
# touching the AACT parquets, the v1 model, PubMed, or OpenFDA at runtime.
DEMO_SCOPES: dict[str, str] = {
    "Recruiting Phase III Trials": "demo_recruiting_phase3.json",
    "Recruiting Oncology Trials":  "demo_oncology_phase3.json",
    "Recruiting All Phases":       "demo_all_phase3.json",
}


# ---------------------------------------------------------------------------
# Report loading
# ---------------------------------------------------------------------------

@st.cache_resource(show_spinner=False)
def _load_demo_file(filename: str) -> dict | None:
    """Read one ``output/demo/<filename>`` JSON; cached per filename."""
    path = DEMO_DIR / filename
    if not path.exists():
        log.warning("Demo report %s not found", path)
        return None
    try:
        return json.loads(path.read_text())
    except Exception as exc:  # noqa: BLE001
        log.exception("Failed to read %s: %s", path, exc)
        return None


def load_demo_report(
    scope_label: str,
) -> tuple[dict, ScanReport, dict, Path] | None:
    """Load the demo report for a sidebar-selected scope.

    Returns ``(state_dict, report, trial_cache, source_path)``.
    ``state_dict`` mirrors live-agent state so the existing app_lib
    helpers continue to work; ``trial_cache`` is the per-trial metadata
    blob the pre-warm script embedded into the saved JSON.
    Returns ``None`` when the file is missing or unreadable.
    """
    filename = DEMO_SCOPES.get(scope_label)
    if not filename:
        return None
    data = _load_demo_file(filename)
    if data is None:
        return None
    report = ScanReport.from_dict(data)
    state = _state_dict_from_report(data)
    trial_cache = data.get("trial_cache") or {}
    return state, report, trial_cache, DEMO_DIR / filename


def _state_dict_from_report(data: dict) -> dict:
    """Build a minimal ``state``-shaped dict from a saved report.

    The pages and ``app_lib`` helpers expect ``state.get("…")`` access
    on a few well-known keys (``prioritised_issues``, ``final_report``,
    ``llm_calls``). This adapter rebuilds those from the saved JSON so
    the demo viewer can reuse the same rendering paths as the live app.
    """
    severity_counts: dict[str, int] = {}
    for sev_key, n in (data.get("issues") or {}).items():
        severity_counts[str(sev_key).upper()] = int(n or 0)
    return {
        "scope": data.get("scope", ""),
        "scope_filters": data.get("scope_filters") or {},
        "prioritised_issues": data.get("prioritised_issues") or [],
        "final_report": {
            "scope": data.get("scope", ""),
            "scope_filters": data.get("scope_filters") or {},
            "severity_counts": severity_counts,
            "llmops_summary": data.get("llmops_summary") or {},
            "root_cause_clusters": data.get("root_cause_clusters") or [],
            "checks_completed": data.get("checks_executed") or [],
            "iterations_used": data.get("agent_iterations", 0),
            "trials_with_issues": data.get("trials_scanned", 0),
        },
        "llm_calls": data.get("audit_log") or [],
        "context_results": [],
        "issues_found": [],
        "pattern_library_updates": data.get("pattern_library_updates") or [],
    }


def _bind_session_state(scope_label: str) -> bool:
    """Populate session state for the given demo scope.

    Re-binds whenever the user switches scope so all four pages see a
    consistent view. Returns True when the corresponding pre-computed
    report file was found and parsed; False otherwise.
    """
    if st.session_state.get("demo_scope") == scope_label \
            and st.session_state.get("scan_state") is not None:
        return True
    loaded = load_demo_report(scope_label)
    if loaded is None:
        return False
    state, report, trial_cache, _ = loaded
    st.session_state["scan_state"] = state
    st.session_state["scan_report"] = report
    st.session_state["trial_cache"] = trial_cache
    st.session_state["demo_scope"] = scope_label
    return True


def _render_demo_banner() -> None:
    """Render the demo-mode banner that appears on every page."""
    st.info(DEMO_BANNER)


def _selected_scope() -> str:
    """Sidebar scope picker — returns the label currently selected."""
    return st.sidebar.selectbox(
        "Demo scope",
        list(DEMO_SCOPES.keys()),
        index=0,
        key="demo_scope_picker",
    )


def _render_scan_caption(report: ScanReport, source_path: Path) -> None:
    """One-line caption explaining what scan is being shown."""
    st.caption(
        f"Showing scan `{report.scan_id}` from "
        f"`output/reports/{source_path.name}` · "
        f"saved {report.timestamp[:19]} · "
        f"scope: {report.scope!r}"
    )


# ---------------------------------------------------------------------------
# PAGE 1 — Scan Overview (read-only metrics + LLMOps + cluster summary)
# ---------------------------------------------------------------------------

def page_overview() -> None:
    """Read-only overview — metrics, LLMOps, cluster count."""
    st.title("DecisionLENS v2 — Demo")
    render_banner()
    _render_demo_banner()

    scope_label = _selected_scope()
    if not _bind_session_state(scope_label):
        st.error(
            f"Demo report missing for scope **{scope_label}**. "
            f"Run `venv/bin/python pre_warm_demo.py` to regenerate."
        )
        return
    loaded = load_demo_report(scope_label)
    if loaded is None:
        return
    state, report, _trial_cache, src = loaded
    _render_scan_caption(report, src)

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

    llmops = fr.get("llmops_summary") or {}
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

    if meaningful_clusters:
        st.subheader("Top root-cause clusters")
        for c in meaningful_clusters[:3]:
            st.markdown(
                f"{severity_badge(c.get('cluster_severity', 'LOW'))} · "
                f"**{c.get('cluster_id', '?')}** — "
                f"{c.get('root_cause', '—')}  *(see Page 2 for full detail)*"
            )


# ---------------------------------------------------------------------------
# PAGE 2 — Prioritised Issues (filters + clusters + per-issue cards)
# ---------------------------------------------------------------------------

def page_issues() -> None:
    """Prioritised issues — same layout as app_v2 but demo_mode=True."""
    st.title("Prioritised issues")
    render_banner()
    _render_demo_banner()

    scope_label = _selected_scope()
    if not _bind_session_state(scope_label):
        st.error(
            f"Demo report missing for scope **{scope_label}**. "
            f"Run `venv/bin/python pre_warm_demo.py` to regenerate."
        )
        return

    prioritised = get_prioritised_issues()
    clusters = get_clusters()

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

    st.subheader("Root cause clusters")
    if not clusters:
        st.caption("No meaningful clusters surfaced this scan.")
    else:
        for c in clusters:
            render_cluster_expander(c)

    st.subheader("Individual findings")
    filtered = [
        p for p in prioritised
        if p.get("severity") in sev_pick
        and p.get("check_category") in cat_pick
    ]
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
        render_issue_expander(issue, demo_mode=True)

    # Export full report — works against the loaded ScanReport instance.
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
# PAGE 3 — Trial Deep Dive (live MCP calls only on optional resources)
# ---------------------------------------------------------------------------

def page_deep_dive() -> None:
    """Per-trial multi-source breakdown — entirely from the embedded cache."""
    st.title("Trial deep dive")
    render_banner()
    _render_demo_banner()

    scope_label = _selected_scope()
    if not _bind_session_state(scope_label):
        st.error(
            f"Demo report missing for scope **{scope_label}**. "
            f"Run `venv/bin/python pre_warm_demo.py` to regenerate."
        )
        return

    prioritised = get_prioritised_issues()
    flagged_ids = sorted({p["trial_id"] for p in prioritised})
    if not flagged_ids:
        st.warning("No flagged trials in this scan.")
        return

    nct_id = st.selectbox("Trial", flagged_ids)
    if not nct_id:
        return

    trial_cache: dict[str, dict] = st.session_state.get("trial_cache") or {}
    cached: dict = trial_cache.get(nct_id) or {}
    if not cached or "error" in cached:
        st.caption(
            "Trial details not cached for this demo scope. "
            "Available in live mode."
        )
        if cached.get("error"):
            st.caption(f"  (cache error: {cached['error']})")
        return

    # --- Panel 1: AACT trial metadata ------------------------------------
    st.subheader("Trial metadata (AACT)")
    details = cached.get("details") or {}
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
            f"AACT details unavailable: "
            f"{details.get('reason', 'not in cache')}"
        )

    # --- Panel 2: Risk model ---------------------------------------------
    st.subheader("v1 completion risk model")
    risk = cached.get("risk") or {}
    if risk:
        st.metric("Completion Risk Score", f"{risk.get('risk_score', 0.0):.3f}")
        st.caption(
            f"Reliable: {risk.get('risk_reliable')} · "
            f"{risk.get('risk_model_version', '—')}"
        )
        if not risk.get("risk_reliable"):
            st.warning(risk.get("note", "Risk score unreliable."))
    else:
        st.caption("Risk score not cached for this trial.")

    # --- Panel 3: Cross-source context -----------------------------------
    st.subheader("Cross-source context")
    pub_col, fda_col = st.columns(2)

    with pub_col:
        st.markdown("**PubMed**")
        pubs = cached.get("publications") or {}
        if pubs.get("status") == "ok":
            n = pubs.get("total_found", 0)
            st.write(f"Publications found: {n}")
            for pub in pubs.get("publications", [])[:5]:
                title = pub.get("title", "(untitled)")
                pmid = pub.get("pmid", "?")
                st.write(f"- [{pmid}] {title}")
        else:
            st.caption(
                f"PubMed unavailable: {pubs.get('reason', 'not in cache')}"
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
        elig = cached.get("eligibility") or {}
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
                f"{elig.get('reason', 'not in cache')}"
            )


# ---------------------------------------------------------------------------
# PAGE 4 — Audit Log & Evaluation
# ---------------------------------------------------------------------------

def _audit_tab(scope_label: str) -> None:
    """Audit log + LLMOps per-call breakdown + report download."""
    if not _bind_session_state(scope_label):
        st.error(
            f"Demo report missing for scope **{scope_label}**. "
            f"Run `venv/bin/python pre_warm_demo.py` to regenerate."
        )
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
            "No metrics_results.json yet — run "
            "`venv/bin/python -m evaluation.metrics` locally to generate it."
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
    _render_demo_banner()

    scope_label = _selected_scope()
    tab1, tab2 = st.tabs(["Audit Log", "Evaluation"])
    with tab1:
        _audit_tab(scope_label)
    with tab2:
        _evaluation_tab()


# ---------------------------------------------------------------------------
# Router
# ---------------------------------------------------------------------------

PAGES: dict[str, Any] = {
    "1 · Scan Overview":             page_overview,
    "2 · Prioritised Issues":        page_issues,
    "3 · Trial Deep Dive":           page_deep_dive,
    "4 · Audit Log & Evaluation":    page_audit_eval,
}


def main() -> None:
    """Streamlit entry point — render the selected page."""
    with st.sidebar:
        st.markdown("### DecisionLENS v2 — Demo")
        page_choice = st.radio("Navigate", list(PAGES.keys()))
    PAGES[page_choice]()


if __name__ == "__main__":
    main()
