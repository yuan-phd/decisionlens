"""Smoke test for the Phase 1 rule engine.

Runs all 5 check functions against the local AACT parquet files and
prints a summary: per-category issue count, severity breakdown, and a
sample finding for each check_name. Designed to be run as a script:

    venv/bin/python -m checks.test_checks
"""

from __future__ import annotations

import logging
import sys
from collections import Counter, defaultdict
from pathlib import Path

# Make project root importable when run as a script.
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from checks.crossfield import check_crossfield_validation  # noqa: E402
from checks.endpoints import check_endpoints_gaps  # noqa: E402
from checks.enrollment import check_enrollment_anomalies  # noqa: E402
from checks.models import Issue  # noqa: E402
from checks.status import check_status_inconsistencies  # noqa: E402
from checks.temporal import check_temporal_consistency  # noqa: E402

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s %(message)s",
)
log = logging.getLogger("checks.test_checks")

CHECKS = [
    ("temporal",   check_temporal_consistency),
    ("enrollment", check_enrollment_anomalies),
    ("status",     check_status_inconsistencies),
    ("endpoint",   check_endpoints_gaps),
    ("crossfield", check_crossfield_validation),
]

LIMIT = 200


def _hr(char: str = "=", n: int = 72) -> None:
    print(char * n)


def _summarise(category: str, issues: list[Issue]) -> None:
    severity_counter: Counter[str] = Counter(i.severity_rule for i in issues)
    by_check: dict[str, list[Issue]] = defaultdict(list)
    for i in issues:
        by_check[i.check_name].append(i)

    _hr("-")
    print(
        f"[{category:>10}] {len(issues):4d} issues  "
        f"HIGH={severity_counter.get('HIGH', 0):3d}  "
        f"MEDIUM={severity_counter.get('MEDIUM', 0):3d}  "
        f"LOW={severity_counter.get('LOW', 0):3d}"
    )
    for name in sorted(by_check):
        sample = by_check[name][0]
        print(f"  - {name:<40s} n={len(by_check[name]):3d}  "
              f"[{sample.severity_rule}] {sample.trial_id}: {sample.finding}")


def main() -> int:
    """Run every check, print per-category summary, and a global total."""
    print()
    _hr("=")
    print(f"DecisionLENS v2 — Phase 1 rule engine smoke test  (limit={LIMIT})")
    _hr("=")

    all_issues: list[Issue] = []
    per_category: list[tuple[str, list[Issue]]] = []
    for category, fn in CHECKS:
        log.info("running check: %s", category)
        issues = fn(limit=LIMIT)
        per_category.append((category, issues))
        all_issues.extend(issues)

    for category, issues in per_category:
        _summarise(category, issues)

    _hr("=")
    severity_total: Counter[str] = Counter(i.severity_rule for i in all_issues)
    category_total: Counter[str] = Counter(i.check_category for i in all_issues)
    print(f"TOTAL issues: {len(all_issues)}")
    print(f"  by severity: HIGH={severity_total.get('HIGH', 0)}  "
          f"MEDIUM={severity_total.get('MEDIUM', 0)}  "
          f"LOW={severity_total.get('LOW', 0)}")
    print(f"  by category: {dict(category_total)}")
    distinct_trials = len({i.trial_id for i in all_issues})
    print(f"  distinct trials with >=1 issue: {distinct_trials}")
    _hr("=")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
