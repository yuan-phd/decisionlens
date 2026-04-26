"""Pre-compute demo reports with embedded per-trial metadata.

Runs the agent on each ``DEMO_SCOPES`` entry, then for every trial
that surfaced an issue it pulls the bits Page 3 of the demo viewer
needs (AACT details, eligibility text, PubMed publications, v1 risk
payload). Everything gets stuffed into a single JSON file under
``output/demo/`` so ``app_demo.py`` can serve a fully self-contained
demo without making any live MCP calls.

Run with:
    venv/bin/python pre_warm_demo.py

Total wall time: ~5 min cold (3 scans × ~70 s each + per-trial calls);
warm with API caches it drops to ~3 min.
"""

from __future__ import annotations

import json
import logging
import os
import time
from typing import Any

logging.basicConfig(
    level=logging.WARNING,
    format="%(levelname)s %(name)s: %(message)s",
)
log = logging.getLogger(__name__)

from agents.orchestrator import run_scan  # noqa: E402
from mcp_servers.aact_server import (  # noqa: E402
    tool_get_eligibility_criteria,
    tool_get_trial_details,
)
from mcp_servers.pubmed_server import tool_search_trial_publications  # noqa: E402
from models.risk_scorer import get_completion_risk_score  # noqa: E402

DEMO_OUT_DIR = "output/demo"

DEMO_SCOPES: list[dict[str, Any]] = [
    {
        "name": "Recruiting Phase III Trials",
        "filename": "demo_recruiting_phase3.json",
        "scope": "Recruiting Phase III trials",
        "filters": {
            "phase": "Phase 3",
            "overall_status": "Recruiting",
            "limit": 200,
        },
    },
    {
        "name": "Recruiting Oncology Trials",
        "filename": "demo_oncology_phase3.json",
        "scope": "Recruiting oncology trials",
        "filters": {
            "therapeutic_area": "oncology",
            "overall_status": "Recruiting",
            "limit": 200,
        },
    },
    {
        "name": "Recruiting All Phases",
        "filename": "demo_all_phase3.json",
        "scope": "All recruiting trials",
        "filters": {
            "overall_status": "Recruiting",
            "limit": 200,
        },
    },
]


def _build_trial_cache(prioritised: list[dict]) -> dict[str, dict]:
    """For every distinct trial_id in ``prioritised``, fetch the bits
    Page 3 of the demo viewer needs.

    Each entry is a dict with sub-keys ``details`` / ``eligibility`` /
    ``publications`` / ``risk``. Failures collapse to ``{"error": str}``
    so the demo viewer can still render a "details unavailable" caption
    for that trial.
    """
    cache: dict[str, dict] = {}
    seen: set[str] = set()
    trial_ids = [p.get("trial_id", "") for p in prioritised]
    distinct = [tid for tid in trial_ids if tid and not (tid in seen or seen.add(tid))]
    print(f"  caching metadata for {len(distinct)} distinct trial(s)…")
    started = time.perf_counter()
    for i, tid in enumerate(distinct, 1):
        try:
            details = tool_get_trial_details(tid)
            eligibility = tool_get_eligibility_criteria(tid)
            pubs = tool_search_trial_publications(tid, max_results=5)
            risk = get_completion_risk_score(tid)
            cache[tid] = {
                "details": details,
                "eligibility": eligibility,
                "publications": pubs,
                "risk": risk,
            }
        except Exception as exc:  # noqa: BLE001
            log.exception("trial cache failed for %s: %s", tid, exc)
            cache[tid] = {"error": str(exc)}
        if i % 10 == 0 or i == len(distinct):
            elapsed = time.perf_counter() - started
            print(f"    {i}/{len(distinct)} trials  ({elapsed:.1f}s elapsed)")
    return cache


def main() -> int:
    """Run all 3 demo scans, embed trial caches, write to ``output/demo/``."""
    os.makedirs(DEMO_OUT_DIR, exist_ok=True)
    overall_started = time.perf_counter()

    summary_rows: list[dict[str, Any]] = []

    for cfg in DEMO_SCOPES:
        print()
        print("=" * 72)
        print(f"DEMO SCOPE: {cfg['name']}")
        print(f"  filters: {cfg['filters']}")
        print("=" * 72)

        scan_started = time.perf_counter()
        state, report = run_scan(cfg["scope"], cfg["filters"])
        scan_secs = time.perf_counter() - scan_started

        report_dict = report.to_dict()
        prioritised = report_dict.get("prioritised_issues") or []

        trial_cache = _build_trial_cache(prioritised)
        report_dict["trial_cache"] = trial_cache
        report_dict["demo_scope_name"] = cfg["name"]

        out_path = os.path.join(DEMO_OUT_DIR, cfg["filename"])
        with open(out_path, "w") as f:
            json.dump(report_dict, f, indent=2, default=str)

        # Risk-reliability stats — derived from each cached trial's
        # ``risk`` payload, not the prioritised issues, so duplicates
        # don't bias the percentage.
        reliable = sum(
            1 for v in trial_cache.values()
            if (v.get("risk") or {}).get("risk_reliable")
        )
        cached_n = len(trial_cache)
        reliable_pct = (reliable / cached_n * 100) if cached_n else 0.0

        size_kb = os.path.getsize(out_path) / 1024
        print()
        print(f"  Saved {out_path}")
        print(
            f"  → {len(prioritised)} issue(s) · "
            f"{cached_n} trial(s) cached "
            f"({reliable}/{cached_n} reliable risk = {reliable_pct:.0f}%) · "
            f"{size_kb:.1f} KB · scan {scan_secs:.1f}s"
        )
        summary_rows.append({
            "name": cfg["name"],
            "issues": len(prioritised),
            "trials_cached": cached_n,
            "reliable_count": reliable,
            "reliable_pct": reliable_pct,
        })

    total = time.perf_counter() - overall_started
    print()
    print("=" * 72)
    print("Summary")
    print("=" * 72)
    print(f"{'scope':<32} {'issues':>8} {'trials':>8} "
          f"{'reliable':>10} {'reliable %':>11}")
    print("-" * 72)
    for r in summary_rows:
        print(f"{r['name']:<32} {r['issues']:>8} {r['trials_cached']:>8} "
              f"{r['reliable_count']:>10} {r['reliable_pct']:>10.1f}%")
    total_trials = sum(r["trials_cached"] for r in summary_rows)
    print("-" * 72)
    print(f"  Total trials cached across {len(summary_rows)} scope(s): "
          f"{total_trials}")
    print(f"  Wall time: {total:.1f}s")
    print("=" * 72)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
