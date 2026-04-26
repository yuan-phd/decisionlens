"""Edge-case tests (T1) for the 5 single-source check functions.

20 assertions covering:

    * unknown trial_id          → returns ``[]``
    * unknown therapeutic_area  → returns a list (graceful degradation)
    * trivial scope (limit=1)   → returns a list, no exception
    * check-specific edges      → returns a list, no exception

Every assertion is wrapped in try/except so a raised exception is a
test failure (not a hard crash). Run with:

    venv/bin/python -m tests.test_checks_edge
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path
from typing import Callable

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from checks.crossfield import check_crossfield_validation  # noqa: E402
from checks.endpoints import check_endpoints_gaps  # noqa: E402
from checks.enrollment import check_enrollment_anomalies  # noqa: E402
from checks.status import check_status_inconsistencies  # noqa: E402
from checks.temporal import check_temporal_consistency  # noqa: E402

# Silence the checks' INFO chatter — we only care about exceptions here.
logging.basicConfig(level=logging.ERROR)


FAKE_NCT = "NCT99999999_DOES_NOT_EXIST"
UNKNOWN_AREA = "xyz_not_a_real_area"

# Real trial_ids from the synthetic AACT data exercised for per-check
# edge scenarios (verified present in data/processed/*.parquet).
NCT_NAN_ACTUAL_ENROLLMENT = "NCT18032585"   # Active, actual_enrollment=NaN
NCT_WITHDRAWN = "NCT74572954"               # Withdrawn
NCT_NO_PRIMARY = "NCT49499059"              # Phase 2, no Primary outcome row
NCT_MULTI_FACILITY = "NCT18032585"          # n_facilities > 0

CHECKS: list[tuple[str, Callable[..., list]]] = [
    ("temporal",   check_temporal_consistency),
    ("enrollment", check_enrollment_anomalies),
    ("status",     check_status_inconsistencies),
    ("endpoints",  check_endpoints_gaps),
    ("crossfield", check_crossfield_validation),
]


Assertion = tuple[str, Callable[[], bool]]


def _is_empty_list(r: object) -> bool:
    return isinstance(r, list) and len(r) == 0


def _is_list(r: object) -> bool:
    return isinstance(r, list)


def _run(check: Callable[..., list], **kwargs) -> object:
    """Invoke a check with kwargs; propagate exceptions to the caller."""
    return check(**kwargs)


def build_assertions() -> list[Assertion]:
    """Construct the 20 edge-case assertions."""
    asserts: list[Assertion] = []

    # Group A — unknown trial_id returns empty list (5)
    for name, fn in CHECKS:
        asserts.append((
            f"A. {name}: unknown trial_id → []",
            (lambda fn=fn: _is_empty_list(_run(fn, trial_id=FAKE_NCT))),
        ))

    # Group B — unknown therapeutic_area returns list, no crash (5)
    for name, fn in CHECKS:
        asserts.append((
            f"B. {name}: unknown therapeutic_area → list (no crash)",
            (lambda fn=fn: _is_list(
                _run(fn, therapeutic_area=UNKNOWN_AREA, limit=5))),
        ))

    # Group C — trivial limit=1 returns list type (5)
    for name, fn in CHECKS:
        asserts.append((
            f"C. {name}: limit=1 → list (no crash)",
            (lambda fn=fn: _is_list(_run(fn, limit=1))),
        ))

    # Group D — check-specific edges (5)
    asserts.append((
        "D. temporal: default limit=20 handles all date values → list",
        (lambda: _is_list(_run(check_temporal_consistency, limit=20))),
    ))
    asserts.append((
        "D. enrollment: trial with NaN actual_enrollment → list",
        (lambda: _is_list(_run(
            check_enrollment_anomalies, trial_id=NCT_NAN_ACTUAL_ENROLLMENT))),
    ))
    asserts.append((
        "D. status: Withdrawn trial → list",
        (lambda: _is_list(_run(
            check_status_inconsistencies, trial_id=NCT_WITHDRAWN))),
    ))
    asserts.append((
        "D. endpoints: trial missing primary outcome → list",
        (lambda: _is_list(_run(
            check_endpoints_gaps, trial_id=NCT_NO_PRIMARY))),
    ))
    asserts.append((
        "D. crossfield: multi-facility trial → list",
        (lambda: _is_list(_run(
            check_crossfield_validation, trial_id=NCT_MULTI_FACILITY))),
    ))

    return asserts


def main() -> int:
    """Execute all assertions, print pass/fail, exit 0 on full pass."""
    asserts = build_assertions()
    print()
    print("=" * 72)
    print(f"T1 — edge-case tests for all 5 check functions "
          f"({len(asserts)} assertions)")
    print("=" * 72)

    n_pass = 0
    n_fail = 0
    for i, (desc, fn) in enumerate(asserts, 1):
        try:
            ok = fn()
            tag = "PASS" if ok else "FAIL"
            if ok:
                n_pass += 1
            else:
                n_fail += 1
            print(f"  {tag}  [{i:>2}/20]  {desc}")
        except Exception as exc:  # noqa: BLE001 — any exception = fail
            n_fail += 1
            print(f"  FAIL  [{i:>2}/20]  {desc}")
            print(f"           EXCEPTION: {type(exc).__name__}: {exc}")

    print("=" * 72)
    print(f"overall: {n_pass}/{len(asserts)} passed, {n_fail} failed")
    print("=" * 72)
    return 0 if n_fail == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
