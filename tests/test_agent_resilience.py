"""T3 — agent resilience tests.

Verifies the 6-node orchestrator degrades gracefully under three LLM
failure modes. Each scenario patches either ``_call_llm`` or
``_call_llm_batch_async`` to raise, runs a full scan, and asserts the
scan completes AND produces a sensible fallback result.

Run with:
    venv/bin/python -m tests.test_agent_resilience
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path
from unittest import mock

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import agents.orchestrator as orch  # noqa: E402

# Each scenario deliberately raises inside a node; suppress the
# log.exception tracebacks that would otherwise dominate test output.
logging.basicConfig(level=logging.CRITICAL)


SCOPE = "Phase III oncology"
FILTERS = {"therapeutic_area": "oncology", "phase": "Phase 3", "limit": 20}


# ---------------------------------------------------------------------------
# Mock helpers
# ---------------------------------------------------------------------------


def _make_selective_raiser(failing_purposes: set[str]):
    """Return a side_effect for _call_llm that raises on specific purposes."""
    original = orch._call_llm

    def side_effect(state, *, purpose, system, user, force_json=True):
        if purpose in failing_purposes:
            raise RuntimeError(f"simulated {purpose} failure")
        return original(
            state, purpose=purpose, system=system, user=user,
            force_json=force_json,
        )

    return side_effect


def _make_async_raiser():
    """AsyncMock suitable for patching ``_call_llm_batch_async``."""

    async def _raise(*_args, **_kwargs):
        raise RuntimeError("simulated batch prioritisation failure")

    return _raise


# ---------------------------------------------------------------------------
# Scenarios
# ---------------------------------------------------------------------------


def _print_result(label: str, ok: bool, detail: str = "") -> None:
    tag = "PASS" if ok else "FAIL"
    suffix = f"  ({detail})" if detail else ""
    print(f"  {tag}  {label}{suffix}")


def scenario_1_planning_fails() -> bool:
    """Planning node LLM raises → scan should fall back to DEFAULT_PLAN."""
    print("\n[Scenario 1] planning LLM raises")
    side = _make_selective_raiser({"planning"})
    try:
        with mock.patch.object(orch, "_call_llm", side_effect=side):
            state, _report = orch.run_scan(
                SCOPE, FILTERS, save_report=False,
            )
        completed = state.get("checks_completed") or []
        plan = state.get("plan") or []

        a1 = True  # scan completed without raising (we got here)
        a2 = len(completed) > 0
        a3 = set(plan) == set(orch.DEFAULT_PLAN)

        _print_result(
            "scan completes without raising", a1, "no exception",
        )
        _print_result(
            "checks_completed is not empty",
            a2, f"completed={completed}",
        )
        _print_result(
            "fallback DEFAULT_PLAN was used",
            a3, f"plan={plan}",
        )
        return a1 and a2 and a3
    except Exception as exc:  # noqa: BLE001
        _print_result(
            "scan completes without raising", False,
            f"EXCEPTION: {type(exc).__name__}: {exc}",
        )
        return False


def scenario_2_batched_prioritisation_fails() -> bool:
    """Batched prioritisation LLM raises → fallback severity per issue."""
    print("\n[Scenario 2] batched prioritisation LLM raises")
    raiser = _make_async_raiser()
    try:
        with mock.patch.object(
            orch, "_call_llm_batch_async", side_effect=raiser,
        ):
            state, _report = orch.run_scan(
                SCOPE, FILTERS, save_report=False,
            )
        prioritised = state.get("prioritised_issues") or []

        a1 = True
        a2 = len(prioritised) > 0

        # Fallback path must produce severity from deterministic code
        # (either _deterministic_low_summary for LOW issues or
        # _fallback_prioritisation for the rest). The tell-tale for the
        # non-LOW fallback is the potential_impact string; for LOW it's
        # the fixed suggested_action. Every prioritised issue should
        # match one of these templates.
        low_action = "Monitor. No immediate action required."
        fallback_impact = "LLM prioritisation unavailable."
        all_deterministic = all(
            p.get("suggested_action") == low_action
            or p.get("potential_impact") == fallback_impact
            for p in prioritised
        )
        # Every "prioritisation" tracked call should be status=error.
        prioritisation_calls = [
            c for c in state.get("llm_calls") or []
            if c.get("purpose") == "prioritisation"
        ]
        prio_all_error = (
            len(prioritisation_calls) > 0
            and all(c.get("status") == "error" for c in prioritisation_calls)
        )

        _print_result(
            "scan completes without raising", a1, "no exception",
        )
        _print_result(
            "prioritised_issues is not empty",
            a2, f"{len(prioritised)} issues",
        )
        _print_result(
            "every issue uses deterministic fallback template",
            all_deterministic,
            "non-LOW→_fallback, LOW→deterministic_low",
        )
        _print_result(
            "every 'prioritisation' LLM call recorded as error",
            prio_all_error,
            f"{len(prioritisation_calls)} calls",
        )
        return a1 and a2 and all_deterministic and prio_all_error
    except Exception as exc:  # noqa: BLE001
        _print_result(
            "scan completes without raising", False,
            f"EXCEPTION: {type(exc).__name__}: {exc}",
        )
        return False


def scenario_3_clustering_fails() -> bool:
    """Clustering LLM raises → clusters empty, prioritisation preserved."""
    print("\n[Scenario 3] clustering LLM raises")
    side = _make_selective_raiser({"clustering"})
    try:
        with mock.patch.object(orch, "_call_llm", side_effect=side):
            state, _report = orch.run_scan(
                SCOPE, FILTERS, save_report=False,
            )
        fr = state.get("final_report") or {}
        clusters = fr.get("root_cause_clusters")
        prioritised = state.get("prioritised_issues") or []

        a1 = True
        a2 = clusters == []
        a3 = len(prioritised) > 0

        _print_result(
            "scan completes without raising", a1, "no exception",
        )
        _print_result(
            "root_cause_clusters is empty list",
            a2, f"clusters={clusters!r}",
        )
        _print_result(
            "prioritised_issues still populated",
            a3, f"{len(prioritised)} issues",
        )
        return a1 and a2 and a3
    except Exception as exc:  # noqa: BLE001
        _print_result(
            "scan completes without raising", False,
            f"EXCEPTION: {type(exc).__name__}: {exc}",
        )
        return False


def main() -> int:
    print()
    print("=" * 72)
    print("T3 — agent resilience under LLM failure")
    print("=" * 72)

    r1 = scenario_1_planning_fails()
    r2 = scenario_2_batched_prioritisation_fails()
    r3 = scenario_3_clustering_fails()

    n_pass = sum([r1, r2, r3])
    print()
    print("=" * 72)
    print(f"overall: {n_pass}/3 scenarios passed")
    print("=" * 72)
    return 0 if n_pass == 3 else 1


if __name__ == "__main__":
    raise SystemExit(main())
