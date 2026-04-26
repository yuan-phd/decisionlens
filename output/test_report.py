"""End-to-end test for output/report.py.

Runs a real agent scan, builds a ScanReport via
``ScanReport.from_agent_state``, saves it to disk, reloads the JSON,
and asserts the roundtrip preserves every required field.

Run with:
    venv/bin/python -m output.test_report
"""

from __future__ import annotations

import json
import logging
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from agents.orchestrator import run_scan  # noqa: E402
from output.report import DATA_SOURCE_NOTE, PrioritisedIssue, ScanReport  # noqa: E402

logging.basicConfig(level=logging.WARNING)


def _hr(char: str = "=", n: int = 72) -> None:
    print(char * n)


REQUIRED_REPORT_FIELDS = [
    "scan_id", "timestamp", "scope", "scope_filters",
    "data_source_note", "trials_scanned", "checks_executed",
    "agent_iterations", "issues", "llmops_summary",
    "prioritised_issues", "root_cause_clusters",
    "pattern_library_updates", "audit_log",
]
REQUIRED_ISSUE_FIELDS = [
    "trial_id", "check_name", "check_category", "finding", "data_points",
    "completion_risk_score", "risk_reliable", "risk_model_version",
    "severity", "explanation", "potential_impact", "suggested_action",
    "confidence", "related_trials", "provenance",
]
REQUIRED_PROVENANCE_FIELDS = [
    "sources_queried", "sources_available", "aact", "pubmed", "openfda",
    "data_accessed_at", "context_enriched", "agent_iteration",
    "llm_call_ids", "risk_model_used", "risk_model_version",
]


def main() -> int:
    """Run the scan, save, reload, assert. Return 0 on full pass."""
    print()
    _hr()
    print("output/report.py — roundtrip test")
    _hr()

    scope = "Phase III oncology trials"
    scope_filters = {
        "therapeutic_area": "oncology",
        "phase": "Phase 3",
        "limit": 30,  # smaller than agent test to keep this run short
    }
    print(f"scope={scope!r}  filters={scope_filters}\n")

    state, report = run_scan(scope, scope_filters)

    print(f"scan_id   : {report.scan_id}")
    print(f"timestamp : {report.timestamp}")
    print(f"issues    : {report.issues}")
    print(f"scanned   : {report.trials_scanned}")
    print(f"checks    : {report.checks_executed}")
    print(f"iterations: {report.agent_iterations}")
    print(f"llm calls : {report.llmops_summary.get('total_llm_calls')}  "
          f"tokens={report.llmops_summary.get('total_tokens')}  "
          f"cost=${report.llmops_summary.get('total_estimated_cost_usd')}")
    print(f"audit_log : {len(report.audit_log)} entry/ies")
    print()

    # --- assertion 1: ScanReport has every required field ---------------
    report_dict = report.to_dict()
    missing_report = [f for f in REQUIRED_REPORT_FIELDS if f not in report_dict]
    report_fields_ok = not missing_report

    # --- assertion 2: data_source_note carries the proxy framing --------
    note_ok = report.data_source_note == DATA_SOURCE_NOTE

    # --- assertion 3: every PrioritisedIssue has every required field ---
    issue_missing = []
    for i, p in enumerate(report_dict["prioritised_issues"]):
        gaps = [f for f in REQUIRED_ISSUE_FIELDS if f not in p]
        if gaps:
            issue_missing.append((i, gaps))
    issue_fields_ok = not issue_missing

    # --- assertion 4: provenance shape complete -------------------------
    provenance_missing = []
    for i, p in enumerate(report_dict["prioritised_issues"]):
        prov = p.get("provenance") or {}
        gaps = [f for f in REQUIRED_PROVENANCE_FIELDS if f not in prov]
        if gaps:
            provenance_missing.append((i, gaps))
    provenance_ok = not provenance_missing

    # --- assertion 5: JSON serialises and round-trips -------------------
    # The saved report was written by run_scan; locate and reload it.
    saved_path = PROJECT_ROOT / "output" / "reports" / f"{report.scan_id}.json"
    json_exists = saved_path.exists()
    reload_ok = False
    reload_equal = False
    if json_exists:
        try:
            loaded = json.loads(saved_path.read_text())
            reload_ok = isinstance(loaded, dict)
            reload_equal = (
                loaded.get("scan_id") == report.scan_id
                and loaded.get("trials_scanned") == report.trials_scanned
                and len(loaded.get("prioritised_issues") or []) == len(report.prioritised_issues)
            )
        except Exception as exc:  # noqa: BLE001
            print(f"  reload failed: {exc}")

    # --- assertion 6: severity buckets in report.issues match counts ----
    counted_severities = {"critical": 0, "high": 0, "medium": 0, "low": 0}
    for p in report.prioritised_issues:
        key = p.severity.lower()
        if key in counted_severities:
            counted_severities[key] += 1
    severity_ok = counted_severities == report.issues

    # --- assertion 7: PrioritisedIssue.from_state_dict works separately -
    factory_ok = True
    try:
        stub = {
            "trial_id": "NCT00000001",
            "check_name": "unit_test",
            "check_category": "temporal",
            "finding": "stub",
            "severity": "LOW",
            "provenance": {"sources_queried": ["aact"]},
            "extra_junk_key_ignored": True,
        }
        parsed = PrioritisedIssue.from_state_dict(stub)
        factory_ok = (
            parsed.trial_id == "NCT00000001"
            and parsed.severity == "LOW"
            and parsed.provenance["sources_queried"] == ["aact"]
            and parsed.confidence == 0.5  # default
        )
    except Exception as exc:  # noqa: BLE001
        print(f"  from_state_dict failed: {exc}")
        factory_ok = False

    # --- results ---------------------------------------------------------
    results = [
        ("ScanReport has all required fields",
         report_fields_ok,
         f"missing: {missing_report}" if missing_report else ""),
        ("data_source_note matches spec",
         note_ok, ""),
        ("every PrioritisedIssue has required fields",
         issue_fields_ok,
         f"gaps: {issue_missing[:3]}" if issue_missing else ""),
        ("every provenance dict has required fields",
         provenance_ok,
         f"gaps: {provenance_missing[:3]}" if provenance_missing else ""),
        ("saved JSON exists on disk",
         json_exists, f"path: {saved_path}"),
        ("saved JSON parses and round-trips",
         reload_ok and reload_equal, ""),
        ("report.issues matches PrioritisedIssue severities",
         severity_ok,
         f"counted={counted_severities} report.issues={report.issues}"
         if not severity_ok else ""),
        ("PrioritisedIssue.from_state_dict handles unknown keys",
         factory_ok, ""),
    ]

    print("-" * 72)
    all_pass = True
    for label, ok, detail in results:
        all_pass &= ok
        tag = "PASS" if ok else "FAIL"
        print(f"  {tag}  {label}" + (f"  ({detail})" if detail else ""))
    _hr()
    print(f"overall: {'PASS' if all_pass else 'FAIL'}  "
          f"report path: {saved_path}")
    _hr()
    return 0 if all_pass else 1


if __name__ == "__main__":
    raise SystemExit(main())
