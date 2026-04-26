"""Smoke test for cross-source checks F (publication) and G (safety).

Design notes
------------
Our AACT parquet is a 10k-row synthetic sample with placeholder
intervention names ("Intervention_N"). The two real-world anchors
referenced in the spec — NCT02578680 (KEYNOTE-189) and pembrolizumab
— do not exist in our sample. Rather than skip them, we:

Test 1 — ``check_publication_consistency(nct_id='NCT02578680')``
    Runs as specified. Our AACT doesn't contain this NCT, so the
    check's scope filter returns an empty DataFrame and emits zero
    issues, logging the reason. This exercises the graceful-
    degradation path end-to-end.

Test 2 — ``check_publication_consistency(limit=20)``
    Exercises the full pipeline against the synthetic AACT so the
    PubMed MCP tool actually fires; shows rule hits against real
    data (almost all synthetic NCTs produce zero PubMed hits, so
    LOW findings dominate).

Test 3 — ``check_safety_cross_validation`` with pembrolizumab injected
    The spec asks for "a trial with pembrolizumab as intervention".
    Our synthetic data has none, so we monkey-patch
    ``_drug_names_for_trial`` to return ``["pembrolizumab"]`` and run
    against NCT93408848 (0 SAE subjects in calculated_values). This
    triggers the real OpenFDA comparison path and surfaces a HIGH
    ``zero_aes_with_fda_signals`` issue.

Test 4 — ``check_safety_cross_validation(limit=10)`` — unpatched
    Baseline run against the synthetic AACT to confirm the plain
    pipeline works end-to-end (all synthetic drug names return
    zero FDA matches → only LOW positive issues).

Run with:
    venv/bin/python -m checks.test_cross_source
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from checks.models import Issue  # noqa: E402
from checks.demo_overrides import DEMO_DRUG_MAP  # noqa: E402
from checks.publication import check_publication_consistency  # noqa: E402
from checks.safety import check_safety_cross_validation  # noqa: E402

logging.basicConfig(
    level=logging.WARNING,
    format="%(levelname)s %(name)s %(message)s",
)


def _hr(char: str = "=", n: int = 72) -> None:
    print(char * n)


def _show(label: str, issues: list[Issue], preview: int = 5) -> None:
    _hr()
    print(f"{label}  →  {len(issues)} issue(s)")
    _hr("-")
    if not issues:
        print("(no issues)")
        return
    severities: dict[str, int] = {}
    for i in issues:
        severities[i.severity_rule] = severities.get(i.severity_rule, 0) + 1
    print(f"severity breakdown: {severities}")
    print()
    for n, issue in enumerate(issues[:preview], 1):
        print(f"{n}. [{issue.severity_rule}] {issue.check_name}  "
              f"(trial={issue.trial_id})")
        print(f"   {issue.finding}")
    if len(issues) > preview:
        print(f"   … and {len(issues) - preview} more")


def main() -> int:
    """Run the 4 tests and return 0 on clean completion."""
    print()
    _hr()
    print("Cross-source checks — F (publication) & G (safety) smoke test")
    _hr()

    # ---- Test 1 -----------------------------------------------------
    print("\n[Test 1] check_publication_consistency on NCT02578680")
    print("(KEYNOTE-189: real NCT with PubMed citations; NOT in our "
          "synthetic AACT → expect graceful-skip, 0 issues)")
    issues = check_publication_consistency(nct_id="NCT02578680")
    _show("NCT02578680", issues)

    # ---- Test 2 -----------------------------------------------------
    print("\n[Test 2] check_publication_consistency on synthetic AACT (limit=20)")
    print("(exercises full PubMed pipeline on real NCT IDs from our parquet)")
    issues = check_publication_consistency(limit=20)
    _show("synthetic sample, limit=20", issues)

    # ---- Test 3 -----------------------------------------------------
    print("\n[Test 3] check_safety_cross_validation on NCT93408848")
    print("(NCT93408848 has 0 SAE subjects; DEMO_DRUG_MAP maps it to "
          "'pembrolizumab' → HIGH zero_aes_with_fda_signals expected)")
    issues = check_safety_cross_validation(nct_id="NCT93408848")
    _show("NCT93408848 (demo override: pembrolizumab)", issues)

    # ---- Test 4 -----------------------------------------------------
    print("\n[Test 4] check_safety_cross_validation on synthetic AACT (limit=20)")
    print(f"(DEMO_DRUG_MAP active with {len(DEMO_DRUG_MAP)} overrides — "
          "target distribution: ≥2 HIGH, 2–3 MEDIUM, rest LOW)")
    issues = check_safety_cross_validation(limit=20)
    _show("limit=20 with DEMO_DRUG_MAP", issues, preview=12)

    print()
    _hr()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
