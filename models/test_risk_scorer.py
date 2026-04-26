"""Smoke test for models/risk_scorer.py.

Pulls 5 real NCT IDs from live Phase 1 findings (rather than
hardcoding) so the test stays reproducible as the rule engine evolves,
then scores them in a single batch call and asserts the scores are
well-formed (in [0, 1] and not all identical).

Run as:
    venv/bin/python -m models.test_risk_scorer
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from checks.crossfield import check_crossfield_validation  # noqa: E402
from checks.endpoints import check_endpoints_gaps  # noqa: E402
from checks.enrollment import check_enrollment_anomalies  # noqa: E402
from checks.status import check_status_inconsistencies  # noqa: E402
from checks.temporal import check_temporal_consistency  # noqa: E402
from models.risk_scorer import (  # noqa: E402
    RISK_MODEL_VERSION,
    escalate_severity,
    get_batch_risk_scores,
)

logging.basicConfig(
    level=logging.WARNING,
    format="%(levelname)s %(name)s %(message)s",
)
log = logging.getLogger("models.test_risk_scorer")

CHECKS = [
    check_temporal_consistency,
    check_enrollment_anomalies,
    check_status_inconsistencies,
    check_endpoints_gaps,
    check_crossfield_validation,
]
SAMPLE_SIZE = 5
SCAN_LIMIT = 50


def collect_phase1_trial_ids(n: int) -> list[str]:
    """Return up to ``n`` distinct trial_ids drawn from real check findings.

    Pulls one trial_id from each check (in order) before looping back —
    so the sample reflects multiple categories rather than concentrating
    on whichever check produces the most findings.
    """
    per_check_ids: list[list[str]] = []
    for fn in CHECKS:
        issues = fn(limit=SCAN_LIMIT)
        seen, uniq = set(), []
        for issue in issues:
            if issue.trial_id not in seen:
                seen.add(issue.trial_id)
                uniq.append(issue.trial_id)
        per_check_ids.append(uniq)

    picked: list[str] = []
    seen: set[str] = set()
    cursor = 0
    while len(picked) < n and any(
        cursor < len(lst) for lst in per_check_ids
    ):
        for lst in per_check_ids:
            if cursor < len(lst) and lst[cursor] not in seen:
                seen.add(lst[cursor])
                picked.append(lst[cursor])
                if len(picked) >= n:
                    break
        cursor += 1
    return picked


def main() -> int:
    """Run the smoke test, print scores, and assert on result quality."""
    print()
    print("=" * 72)
    print(f"models/risk_scorer.py — smoke test  (model={RISK_MODEL_VERSION})")
    print("=" * 72)

    trial_ids = collect_phase1_trial_ids(SAMPLE_SIZE)
    print(f"Sampled {len(trial_ids)} trial_id(s) from Phase 1 findings:")
    for tid in trial_ids:
        print(f"  - {tid}")
    print()

    if len(trial_ids) < SAMPLE_SIZE:
        print(f"FAIL: only collected {len(trial_ids)} trial(s); need {SAMPLE_SIZE}.")
        return 1

    # Include an unknown trial_id so we can verify risk_reliable=False.
    probe_ids = trial_ids + ["NCT00000000"]
    scores = get_batch_risk_scores(probe_ids)

    print(f"{'trial_id':<14}  {'risk_score':>10}  {'reliable':>9}  note")
    print("-" * 72)
    for tid in probe_ids:
        p = scores[tid]
        print(
            f"{tid:<14}  {p['risk_score']:>10.4f}  "
            f"{str(p['risk_reliable']):>9}  {p['note']}"
        )
    print()

    known_payloads = [scores[tid] for tid in trial_ids]
    unknown_payload = scores["NCT00000000"]

    out_of_range = {
        tid: scores[tid]["risk_score"] for tid in probe_ids
        if not (0.0 <= scores[tid]["risk_score"] <= 1.0)
    }
    distinct = len({round(p["risk_score"], 6) for p in known_payloads})

    range_ok = not out_of_range
    variation_ok = distinct >= 2
    reliable_count = sum(p["risk_reliable"] for p in known_payloads)
    # On real AACT ~40% of trials lack the features needed for risk
    # scoring (withdrawn / missing dates / missing designs joins). The
    # scorer correctly returns risk_reliable=False for those. The test
    # only needs to confirm the scorer CAN produce reliable scores —
    # not that every sampled trial is reachable.
    reliable_known = reliable_count >= 1
    unreliable_unknown = unknown_payload["risk_reliable"] is False
    unknown_note_ok = "unavailable" in unknown_payload["note"].lower()

    print(f"in-range [0,1]:       {'PASS' if range_ok else 'FAIL'} "
          f"(violations: {out_of_range or 'none'})")
    print(f"score variation:      {'PASS' if variation_ok else 'FAIL'} "
          f"({distinct} distinct value(s) across {len(known_payloads)} known trial(s))")
    print(f"known → reliable=True:   {'PASS' if reliable_known else 'FAIL'} "
          f"({reliable_count}/{len(known_payloads)})")
    print(f"  reliable: {reliable_count}/{len(known_payloads)} "
          f"(>=1 required, real AACT has ~60% coverage)")
    print(f"unknown → reliable=False:{'PASS' if unreliable_unknown else 'FAIL'} "
          f"(got {unknown_payload['risk_reliable']!r}, note: {unknown_payload['note']!r})")
    print(f"unknown note populated:  {'PASS' if unknown_note_ok else 'FAIL'}")
    print("=" * 72)

    escalation_ok = test_escalate_severity()
    boundary_ok = test_escalate_severity_boundaries()

    all_pass = (
        range_ok and variation_ok and reliable_known
        and unreliable_unknown and unknown_note_ok
        and escalation_ok and boundary_ok
    )
    return 0 if all_pass else 1


def _risk(score: float, reliable: bool = True, note: str = "") -> dict:
    """Build a risk-payload dict for escalation test cases."""
    return {
        "risk_score": score,
        "risk_reliable": reliable,
        "risk_model_version": RISK_MODEL_VERSION,
        "note": note,
    }


def test_escalate_severity() -> bool:
    """Verify escalate_severity() against the CLAUDE.md cases plus the
    unreliable-score guard introduced with the dict-based payload.

    Returns True if every case matches expectation, False otherwise.
    """
    print()
    print("=" * 72)
    print("escalate_severity() — case table")
    print("=" * 72)

    cases: list[tuple[str, dict, str, str]] = [
        # (input_severity, risk_payload, expected_output, label)
        ("HIGH",   _risk(0.70), "CRITICAL",
         "HIGH + 0.70 reliable  (above 0.6 threshold)"),
        ("HIGH",   _risk(0.50), "HIGH",
         "HIGH + 0.50 reliable  (below 0.6 threshold)"),
        ("MEDIUM", _risk(0.75), "HIGH",
         "MEDIUM + 0.75 reliable (above 0.7 threshold)"),
        ("MEDIUM", _risk(0.50), "MEDIUM",
         "MEDIUM + 0.50 reliable (below 0.7 threshold)"),
        ("LOW",    _risk(0.90), "LOW",
         "LOW + 0.90 reliable   (LOW never escalates)"),
        ("HIGH",   _risk(0.95, reliable=False, note="neutral fallback"),
         "HIGH",
         "HIGH + 0.95 UNRELIABLE (must NOT escalate despite high score)"),
    ]

    print(f"{'input':<8} {'score':>6}  {'rel':>5}  "
          f"{'expected':<10} {'actual':<10} {'result'}")
    print("-" * 72)
    all_pass = True
    for sev, risk, expected, label in cases:
        actual = escalate_severity(sev, risk)
        ok = actual == expected
        all_pass &= ok
        print(f"{sev:<8} {risk['risk_score']:>6.2f}  "
              f"{str(risk['risk_reliable']):>5}  "
              f"{expected:<10} {actual:<10} "
              f"{'PASS' if ok else 'FAIL'}   ({label})")
    print("=" * 72)
    print(f"escalation: {'PASS' if all_pass else 'FAIL'}")
    return all_pass


def test_escalate_severity_boundaries() -> bool:
    """T4 — boundary tests probing exact threshold values.

    The escalation thresholds are inclusive: a score that exactly
    equals the threshold escalates, a score one tick below does not.
    """
    print()
    print("=" * 72)
    print("escalate_severity() — boundary tests (T4)")
    print("=" * 72)

    boundary_cases: list[tuple[str, float, str, str]] = [
        ("HIGH",   0.599, "HIGH",     "below CRITICAL threshold"),
        ("HIGH",   0.600, "CRITICAL", "at CRITICAL threshold"),
        ("MEDIUM", 0.700, "HIGH",     "at HIGH threshold"),
        ("MEDIUM", 0.699, "MEDIUM",   "below HIGH threshold"),
    ]

    print(f"{'input':<8} {'score':>6}  "
          f"{'expected':<10} {'actual':<10} {'result'}")
    print("-" * 72)
    all_pass = True
    for severity_rule, score, expected, label in boundary_cases:
        risk = {
            "risk_score": score,
            "risk_reliable": True,
            "risk_model_version": RISK_MODEL_VERSION,
            "note": "",
        }
        actual = escalate_severity(severity_rule, risk)
        ok = actual == expected
        all_pass &= ok
        print(f"{severity_rule:<8} {score:>6.3f}  "
              f"{expected:<10} {actual:<10} "
              f"{'PASS' if ok else 'FAIL'}   ({label})")
    print("=" * 72)
    print(f"boundaries: {'PASS' if all_pass else 'FAIL'}")
    return all_pass


if __name__ == "__main__":
    raise SystemExit(main())
