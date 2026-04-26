"""End-to-end agent smoke test.

Runs a full 5-node scan over ``scope="Phase III oncology trials"`` and
prints the artifacts the user asked to see: planning output, per-
category issue counts, trials investigated, and final severity mix.

Run with:
    venv/bin/python -m agents.test_agent
"""

from __future__ import annotations

import logging
import sys
import time
from collections import Counter
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from agents.orchestrator import run_scan  # noqa: E402

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s %(message)s",
)
log = logging.getLogger("agents.test_agent")


def _hr(char: str = "=", n: int = 72) -> None:
    print(char * n)


def main() -> int:
    """Run one scan and print a human-readable report."""
    scope = "Phase III oncology trials"
    scope_filters = {
        "therapeutic_area": "oncology",
        "phase": "Phase 3",
        "limit": 50,
    }

    print()
    _hr()
    print(f"DecisionLENS v2 agent — scope={scope!r}")
    print(f"scope_filters={scope_filters}")
    _hr()

    wall_started = time.perf_counter()
    state, report = run_scan(scope, scope_filters)
    wall_ms = (time.perf_counter() - wall_started) * 1000.0

    # --- planning ------------------------------------------------------
    print("\n[1] planning")
    _hr("-")
    print(f"  checks planned      : {state.get('plan')}")
    print(f"  checks completed    : {state.get('checks_completed')}")
    for entry in state.get("pattern_library_updates", []):
        if entry.get("source") == "planning":
            print(f"  LLM rationale       : {entry.get('rationale')}")
            break

    # --- scan ----------------------------------------------------------
    issues = state.get("issues_found", [])
    by_cat = Counter(i.get("check_category", "?") for i in issues)
    by_name = Counter(i.get("check_name", "?") for i in issues)
    scan_summary = next(
        (e for e in state.get("pattern_library_updates", [])
         if e.get("source") == "scan_summary"),
        {},
    )
    scan_caps = [
        e for e in state.get("pattern_library_updates", [])
        if e.get("source") == "scan_cap"
    ]
    print("\n[2] scan")
    _hr("-")
    print(f"  issues before cap   : {scan_summary.get('before_cap', '?')}")
    print(f"  issues after cap    : {len(issues)}")
    print(f"  checks truncated    : {scan_summary.get('checks_capped', 0)}")
    print(f"  issues dropped      : {scan_summary.get('issues_dropped', 0)}")
    if scan_caps:
        print("  capped check_names  :")
        for e in scan_caps:
            print(f"    - {e['check_name']:<40s} "
                  f"{e['found']} → {e['kept']}")
    print(f"  distinct trials     : {len({i['trial_id'] for i in issues})}")
    print(f"  issues per category : {dict(by_cat)}")
    top_names = by_name.most_common(8)
    print("  top check_names     :")
    for name, n in top_names:
        print(f"    - {name:<40s} {n}")

    # --- review + context ---------------------------------------------
    investigated = [c.get("trial_id") for c in state.get("context_results", [])]
    print("\n[3] review + context enrichment")
    _hr("-")
    print(f"  iterations used     : {state.get('iteration')}  (cap 3)")
    print(f"  trials investigated : {investigated}")

    # --- prioritisation ------------------------------------------------
    prioritised = state.get("prioritised_issues", [])
    sev_counts = Counter(p.get("severity", "LOW") for p in prioritised)
    print("\n[4] prioritisation")
    _hr("-")
    print(f"  prioritised issues  : {len(prioritised)}")
    print(f"  severity breakdown  : {dict(sev_counts)}")
    # Show up to 5 representative issues across severities
    preview = sorted(
        prioritised,
        key=lambda p: {"CRITICAL": 0, "HIGH": 1, "MEDIUM": 2, "LOW": 3}.get(
            p.get("severity", "LOW"), 4,
        ),
    )[:5]
    for p in preview:
        print(f"\n  • [{p['severity']}] {p['check_name']} (trial={p['trial_id']}, "
              f"risk={p['completion_risk_score']:.2f} "
              f"reliable={p['risk_reliable']})")
        print(f"    finding   : {p['finding']}")
        print(f"    impact    : {p['potential_impact']}")
        print(f"    action    : {p['suggested_action']}")
        print(f"    confidence: {p['confidence']}")
        print(f"    sources   : {p['provenance']['sources_queried']}")

    # --- root-cause clusters ------------------------------------------
    fr = state.get("final_report", {})
    clusters = fr.get("root_cause_clusters") or []
    non_unclustered = [c for c in clusters
                       if c.get("cluster_id") != "RC_UNCLUSTERED"]
    unclustered = [c for c in clusters
                   if c.get("cluster_id") == "RC_UNCLUSTERED"]
    print("\n[5] root cause clusters")
    _hr("-")
    print(f"  total clusters       : {len(clusters)}  "
          f"(meaningful: {len(non_unclustered)})")
    for c in non_unclustered:
        print(f"\n  • {c.get('cluster_id')}  [{c.get('cluster_severity')}]  "
              f"{len(c.get('affected_trials', []))} trial(s)")
        print(f"    root_cause : {c.get('root_cause')}")
        print(f"    pattern    : {c.get('pattern')}")
        print(f"    action     : {c.get('recommended_action')}")
        tids = c.get("affected_trials") or []
        if tids:
            preview = ", ".join(tids[:6])
            more = f" …+{len(tids) - 6}" if len(tids) > 6 else ""
            print(f"    trials     : {preview}{more}")
    if unclustered:
        u = unclustered[0]
        print(f"\n  RC_UNCLUSTERED: {len(u.get('issue_indices', []))} "
              f"issue(s) not grouped")

    # --- LLMOps --------------------------------------------------------
    llmops = fr.get("llmops_summary", {})
    prio_wall = next(
        (e for e in state.get("pattern_library_updates", [])
         if e.get("source") == "prioritisation_wall"),
        {},
    )
    print("\n[6] LLMOps")
    _hr("-")
    print(f"  calls           : {llmops.get('total_llm_calls')}")
    print(f"  tokens          : {llmops.get('total_tokens')}")
    print(f"  latency (sum)   : {llmops.get('total_latency_ms')} ms  "
          "(sum of per-call compute)")
    print(f"  wall (full scan): {wall_ms:.0f} ms")
    if prio_wall:
        print(f"  wall (prio only): {prio_wall['wall_ms']} ms  "
              f"across {prio_wall['batches']} concurrent batch(es)")
    print(f"  cost            : ${llmops.get('total_estimated_cost_usd')}")
    print(f"  model           : "
          f"{state.get('llm_calls', [{}])[0].get('model') if state.get('llm_calls') else 'n/a'}")

    # --- Output quality validation (T2) --------------------------------
    print("\n[7] output quality validation")
    _hr("-")

    checks_passed = 0
    checks_total = 0
    quality_ok = True

    def _assert(label: str, ok: bool, detail: str = "") -> None:
        nonlocal checks_passed, checks_total, quality_ok
        checks_total += 1
        tag = "PASS" if ok else "FAIL"
        if ok:
            checks_passed += 1
        else:
            quality_ok = False
        suffix = f"  ({detail})" if detail else ""
        print(f"  {tag}  {label}{suffix}")

    # 1. Every CRITICAL carries a reliable, risk > 0.6 score.
    critical = [p for p in prioritised if p["severity"] == "CRITICAL"]
    crit_bad = [
        p for p in critical
        if not (
            p["completion_risk_score"] > 0.6
            and p["risk_reliable"] is True
        )
    ]
    _assert(
        "1. every CRITICAL has risk>0.6 AND risk_reliable=True",
        not crit_bad,
        f"{len(critical) - len(crit_bad)}/{len(critical)} CRITICAL issues"
        + (f"; violators: {[p['trial_id'] for p in crit_bad]}"
           if crit_bad else ""),
    )

    # 2. sources_queried populated on every issue.
    empty_sources = [
        p for p in prioritised
        if not (p.get("provenance") or {}).get("sources_queried")
    ]
    _assert(
        "2. sources_queried non-empty on every issue",
        not empty_sources,
        f"{len(prioritised) - len(empty_sources)}/{len(prioritised)} issues"
        + (f"; violators: {[p['trial_id'] for p in empty_sources][:3]}"
           if empty_sources else ""),
    )

    # 3. (trial_id, check_name) pairs are unique.
    pairs = [(p["trial_id"], p["check_name"]) for p in prioritised]
    dupes = [pair for pair in set(pairs) if pairs.count(pair) > 1]
    _assert(
        "3. no duplicate (trial_id, check_name) pairs",
        len(pairs) == len(set(pairs)),
        f"{len(pairs)} pairs, {len(set(pairs))} unique"
        + (f"; dupes: {dupes[:3]}" if dupes else ""),
    )

    # 4. Severity distribution shows >= 2 levels (not collapsed).
    severities = {p["severity"] for p in prioritised}
    _assert(
        "4. severity distribution spans >= 2 levels",
        len(severities) >= 2,
        f"levels present: {sorted(severities)}",
    )

    _hr("-")
    print(f"  quality checks: {checks_passed}/{checks_total} passed")

    # --- ScanReport saved by run_scan() --------------------------------
    print("\n[8] saved report")
    _hr("-")
    print(f"  scan_id   : {report.scan_id}")
    print(f"  timestamp : {report.timestamp}")
    print(f"  path      : output/reports/{report.scan_id}.json")

    _hr()
    return 0 if quality_ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
