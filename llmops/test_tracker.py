"""Unit test for llmops/tracker.py.

Simulates three LLM calls (planning, review, prioritisation) against a
fresh ``LLMTracker`` and asserts the summary totals and per-call cost
arithmetic. No network — this runs offline.

Run with:
    venv/bin/python -m llmops.test_tracker
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from llmops.tracker import LLMTracker, estimate_cost_usd  # noqa: E402

logging.basicConfig(level=logging.WARNING)


def _hr(char: str = "=", n: int = 72) -> None:
    print(char * n)


def _approx_equal(a: float, b: float, tol: float = 1e-6) -> bool:
    return abs(a - b) <= tol


def main() -> int:
    """Run the 3-call simulation and assert the summary totals."""
    print()
    _hr()
    print("llmops/tracker.py — unit test")
    _hr()

    tracker = LLMTracker()

    # --- Simulate three calls (GPT-4o-mini, $0.15/$0.60 per 1M) -------
    fixtures = [
        {"purpose": "planning",       "pt": 1000, "ct": 200, "lat": 2000.0},
        {"purpose": "review",         "pt": 2000, "ct": 300, "lat": 3000.0},
        {"purpose": "prioritisation", "pt": 3000, "ct": 500, "lat": 5000.0},
    ]
    for fx in fixtures:
        tracker.track_call(
            model="gpt-4o-mini",
            purpose=fx["purpose"],
            prompt_tokens=fx["pt"],
            completion_tokens=fx["ct"],
            latency_ms=fx["lat"],
        )

    summary = tracker.get_summary()

    # --- Expected totals ---------------------------------------------
    exp_prompt = 1000 + 2000 + 3000
    exp_completion = 200 + 300 + 500
    exp_tokens = exp_prompt + exp_completion
    exp_latency = 2000 + 3000 + 5000
    # GPT-4o-mini: $0.15/1M input + $0.60/1M output
    exp_cost = (
        exp_prompt * 0.15 / 1_000_000
        + exp_completion * 0.60 / 1_000_000
    )

    checks = [
        ("total_llm_calls           ",
         summary["total_llm_calls"], 3),
        ("total_prompt_tokens       ",
         summary["total_prompt_tokens"], exp_prompt),
        ("total_completion_tokens   ",
         summary["total_completion_tokens"], exp_completion),
        ("total_tokens              ",
         summary["total_tokens"], exp_tokens),
        ("total_latency_ms          ",
         summary["total_latency_ms"], exp_latency),
        ("calls list length         ",
         len(summary["calls"]), 3),
    ]

    print(f"{'field':<28} {'actual':>14}  {'expected':>14}  {'result'}")
    print("-" * 72)
    all_pass = True
    for label, actual, expected in checks:
        ok = actual == expected
        all_pass &= ok
        print(f"{label} {actual:>14}  {expected:>14}  {'PASS' if ok else 'FAIL'}")

    # Cost tolerance (floating-point) — round 6
    cost_ok = _approx_equal(
        summary["total_estimated_cost_usd"], round(exp_cost, 6),
    )
    all_pass &= cost_ok
    print(
        f"{'total_estimated_cost_usd ':<28} "
        f"{summary['total_estimated_cost_usd']:>14.6f}  "
        f"{round(exp_cost, 6):>14.6f}  {'PASS' if cost_ok else 'FAIL'}"
    )

    # Spot-check one call's cost
    first_call = summary["calls"][0]
    exp_first = estimate_cost_usd("gpt-4o-mini", 1000, 200)
    first_ok = _approx_equal(
        first_call["estimated_cost_usd"], round(exp_first, 6),
    )
    all_pass &= first_ok
    print(
        f"{'first call cost_usd      ':<28} "
        f"{first_call['estimated_cost_usd']:>14.6f}  "
        f"{round(exp_first, 6):>14.6f}  {'PASS' if first_ok else 'FAIL'}"
    )

    # Unknown-model path: cost should fall back to 0
    tracker.track_call(
        model="unknown-model", purpose="debug",
        prompt_tokens=100, completion_tokens=50, latency_ms=10.0,
    )
    unknown_cost = tracker.calls[-1]["estimated_cost_usd"]
    unknown_ok = unknown_cost == 0.0
    all_pass &= unknown_ok
    print(f"{'unknown-model cost=0     ':<28} "
          f"{unknown_cost:>14.6f}  {0.0:>14.6f}  "
          f"{'PASS' if unknown_ok else 'FAIL'}")

    # Reset works
    tracker.reset()
    reset_ok = len(tracker.calls) == 0 and tracker.get_summary()["total_llm_calls"] == 0
    all_pass &= reset_ok
    print(f"{'reset() clears calls     ':<28} "
          f"calls={len(tracker.calls)}  "
          f"{'PASS' if reset_ok else 'FAIL'}")

    _hr()
    print(f"overall: {'PASS' if all_pass else 'FAIL'}")
    _hr()
    return 0 if all_pass else 1


if __name__ == "__main__":
    raise SystemExit(main())
