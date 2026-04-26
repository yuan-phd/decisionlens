"""LLMTracker — single source of truth for LLM call accounting.

Every LLM call from the agent (planning / review / context /
prioritisation) passes through ``LLMTracker.track_call``, which:

* assigns a UUID ``call_id`` for provenance
* computes ``estimated_cost_usd`` from a per-model rate table
* appends a structured record to an internal list
* returns that record so the caller can attach it to agent state

Rate table (USD per 1M tokens) covers the models we actually use. New
models land here when they land in the orchestrator — unknown models
degrade to zero cost and a warning (tracking still works, cost just
reads $0 for those calls).
"""

from __future__ import annotations

import logging
import uuid
from typing import Optional

log = logging.getLogger(__name__)

# (input_rate_per_1M, output_rate_per_1M) in USD.
MODEL_COSTS: dict[str, tuple[float, float]] = {
    "gpt-4o-mini": (0.15, 0.60),
    "gpt-4o":      (2.50, 10.00),
    "gpt-4.1-mini": (0.40, 1.60),
}


def estimate_cost_usd(
    model: str, prompt_tokens: int, completion_tokens: int,
) -> float:
    """Return USD cost for one call using ``MODEL_COSTS``.

    Unknown models log once and cost as $0 so tracking is never blocked.
    """
    rates = MODEL_COSTS.get(model)
    if rates is None:
        log.warning(
            "LLMTracker: no cost rate for model %r; recording $0.", model,
        )
        return 0.0
    input_rate, output_rate = rates
    return (
        prompt_tokens * input_rate / 1_000_000
        + completion_tokens * output_rate / 1_000_000
    )


class LLMTracker:
    """Accumulates LLM call records and exposes a summary view."""

    def __init__(self) -> None:
        self.calls: list[dict] = []

    def track_call(
        self,
        *,
        model: str,
        purpose: str,
        prompt_tokens: int,
        completion_tokens: int,
        latency_ms: float,
        status: str = "ok",
        error: Optional[str] = None,
    ) -> dict:
        """Record one LLM call. Returns the record.

        ``purpose`` is a free-form label but agents use one of:
        ``"planning"``, ``"review"``, ``"context"``, ``"prioritisation"``.
        ``status`` is ``"ok"`` on success or ``"error"`` on failure; error
        records carry zero tokens so they don't inflate cost totals.
        """
        cost = estimate_cost_usd(model, prompt_tokens, completion_tokens)
        record: dict = {
            "call_id": str(uuid.uuid4()),
            "model": model,
            "purpose": purpose,
            "prompt_tokens": int(prompt_tokens),
            "completion_tokens": int(completion_tokens),
            "latency_ms": round(float(latency_ms), 1),
            "estimated_cost_usd": round(cost, 6),
            "status": status,
        }
        if error:
            record["error"] = error
        self.calls.append(record)
        return record

    def get_summary(self) -> dict:
        """Roll up the recorded calls into totals + the raw call list."""
        total_prompt = sum(c["prompt_tokens"] for c in self.calls)
        total_completion = sum(c["completion_tokens"] for c in self.calls)
        total_latency = sum(c["latency_ms"] for c in self.calls)
        total_cost = sum(c["estimated_cost_usd"] for c in self.calls)
        return {
            "total_llm_calls": len(self.calls),
            "total_prompt_tokens": total_prompt,
            "total_completion_tokens": total_completion,
            "total_tokens": total_prompt + total_completion,
            "total_latency_ms": round(total_latency, 1),
            "total_estimated_cost_usd": round(total_cost, 6),
            "calls": list(self.calls),
        }

    def reset(self) -> None:
        """Forget every recorded call (use between scans)."""
        self.calls = []
