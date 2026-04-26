"""Structured JSON output with multi-source provenance.

Two dataclasses:

* ``PrioritisedIssue`` — one row per finding after rule engine +
  v1 risk model + LLM prioritisation. Provenance dict names every
  source consulted so the Page 4 audit tab can render the evidence
  trail.
* ``ScanReport``      — the full scan artifact (metadata + totals +
  all PrioritisedIssues + audit log). Serialises to a single JSON
  file under ``output/reports/<scan_id>.json`` for download from the
  Streamlit dashboard.

The only external coupling is to agent state — ``ScanReport.from_agent_state``
reads ``state["final_report"]``, ``state["prioritised_issues"]``,
``state["pattern_library_updates"]``, and ``state["llm_calls"]``.
"""

from __future__ import annotations

import json
import logging
import uuid
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

log = logging.getLogger(__name__)

PROJECT_ROOT: Path = Path(__file__).resolve().parent.parent
DEFAULT_REPORTS_DIR: Path = PROJECT_ROOT / "output" / "reports"

DATA_SOURCE_NOTE: str = (
    "AACT registry + PubMed + OpenFDA (proxies for EDC/CTMS)"
)


@dataclass
class PrioritisedIssue:
    """One fully-prioritised data-quality finding with multi-source provenance."""

    # From rule engine
    trial_id: str
    check_name: str
    check_category: str
    finding: str
    data_points: dict

    # From v1 risk model
    completion_risk_score: float
    risk_reliable: bool
    risk_model_version: str

    # From LLM prioritisation
    severity: str               # CRITICAL/HIGH/MEDIUM/LOW
    explanation: str
    potential_impact: str
    suggested_action: str
    confidence: float

    related_trials: list[str] = field(default_factory=list)
    provenance: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        """Return a JSON-serialisable dict view."""
        return asdict(self)

    @classmethod
    def from_state_dict(cls, d: dict) -> "PrioritisedIssue":
        """Build from a state dict; unknown keys are dropped."""
        allowed = {f.name for f in cls.__dataclass_fields__.values()}
        kwargs: dict[str, Any] = {k: v for k, v in d.items() if k in allowed}
        # Safe defaults for required fields that may be missing.
        kwargs.setdefault("data_points", {})
        kwargs.setdefault("completion_risk_score", 0.5)
        kwargs.setdefault("risk_reliable", False)
        kwargs.setdefault("risk_model_version", "")
        kwargs.setdefault("severity", "LOW")
        kwargs.setdefault("explanation", "")
        kwargs.setdefault("potential_impact", "")
        kwargs.setdefault("suggested_action", "")
        kwargs.setdefault("confidence", 0.5)
        return cls(**kwargs)


@dataclass
class ScanReport:
    """Full scan artifact — metadata, totals, issues, LLMOps, audit log."""

    scan_id: str
    timestamp: str
    scope: str
    scope_filters: dict
    data_source_note: str = DATA_SOURCE_NOTE
    trials_scanned: int = 0
    checks_executed: list[str] = field(default_factory=list)
    agent_iterations: int = 0
    issues: dict = field(default_factory=dict)            # {critical, high, medium, low}
    llmops_summary: dict = field(default_factory=dict)
    prioritised_issues: list[PrioritisedIssue] = field(default_factory=list)
    root_cause_clusters: list[dict] = field(default_factory=list)
    pattern_library_updates: list[dict] = field(default_factory=list)
    audit_log: list[dict] = field(default_factory=list)

    def to_dict(self) -> dict:
        """Return the full report as a JSON-serialisable dict."""
        return {
            "scan_id": self.scan_id,
            "timestamp": self.timestamp,
            "scope": self.scope,
            "scope_filters": dict(self.scope_filters),
            "data_source_note": self.data_source_note,
            "trials_scanned": self.trials_scanned,
            "checks_executed": list(self.checks_executed),
            "agent_iterations": self.agent_iterations,
            "issues": dict(self.issues),
            "llmops_summary": dict(self.llmops_summary),
            "prioritised_issues": [p.to_dict() for p in self.prioritised_issues],
            "root_cause_clusters": list(self.root_cause_clusters),
            "pattern_library_updates": list(self.pattern_library_updates),
            "audit_log": list(self.audit_log),
        }

    def save(self, path: str | Path = DEFAULT_REPORTS_DIR) -> str:
        """Write the report to ``<path>/<scan_id>.json`` and return the file path."""
        dir_path = Path(path)
        dir_path.mkdir(parents=True, exist_ok=True)
        file_path = dir_path / f"{self.scan_id}.json"
        file_path.write_text(
            json.dumps(self.to_dict(), indent=2, default=str)
        )
        log.info("ScanReport saved: %s", file_path)
        return str(file_path)

    @classmethod
    def from_dict(cls, data: dict) -> "ScanReport":
        """Reconstruct a ScanReport from a previously saved JSON dict.

        Inverse of ``to_dict()`` — used by the read-only demo viewer
        (``app_demo.py``) and by anyone replaying historical scans.
        Unknown top-level keys are ignored; ``prioritised_issues`` is
        rebuilt via ``PrioritisedIssue.from_state_dict`` so future
        schema additions stay backward-compatible.
        """
        prioritised = [
            PrioritisedIssue.from_state_dict(p)
            for p in (data.get("prioritised_issues") or [])
        ]
        return cls(
            scan_id=data.get("scan_id", ""),
            timestamp=data.get("timestamp", ""),
            scope=data.get("scope", ""),
            scope_filters=dict(data.get("scope_filters") or {}),
            data_source_note=data.get("data_source_note", DATA_SOURCE_NOTE),
            trials_scanned=int(data.get("trials_scanned", 0) or 0),
            checks_executed=list(data.get("checks_executed") or []),
            agent_iterations=int(data.get("agent_iterations", 0) or 0),
            issues=dict(data.get("issues") or {}),
            llmops_summary=dict(data.get("llmops_summary") or {}),
            prioritised_issues=prioritised,
            root_cause_clusters=list(data.get("root_cause_clusters") or []),
            pattern_library_updates=list(
                data.get("pattern_library_updates") or []
            ),
            audit_log=list(data.get("audit_log") or []),
        )

    @classmethod
    def from_agent_state(cls, state: dict) -> "ScanReport":
        """Build a ScanReport from a completed agent state dict."""
        fr = state.get("final_report") or {}

        severity_counts = fr.get("severity_counts") or {}
        issues_dict = {
            "critical": int(severity_counts.get("CRITICAL", 0)),
            "high":     int(severity_counts.get("HIGH", 0)),
            "medium":   int(severity_counts.get("MEDIUM", 0)),
            "low":      int(severity_counts.get("LOW", 0)),
        }

        prioritised = [
            PrioritisedIssue.from_state_dict(p)
            for p in (state.get("prioritised_issues") or [])
        ]

        return cls(
            scan_id=uuid.uuid4().hex[:12],
            timestamp=datetime.now(timezone.utc).isoformat(),
            scope=fr.get("scope") or state.get("scope", ""),
            scope_filters=(
                fr.get("scope_filters") or state.get("scope_filters") or {}
            ),
            trials_scanned=int(fr.get("trials_with_issues", 0) or 0),
            checks_executed=list(fr.get("checks_completed") or []),
            agent_iterations=int(fr.get("iterations_used", 0) or 0),
            issues=issues_dict,
            llmops_summary=dict(fr.get("llmops_summary") or {}),
            prioritised_issues=prioritised,
            root_cause_clusters=list(fr.get("root_cause_clusters") or []),
            pattern_library_updates=list(
                state.get("pattern_library_updates") or []
            ),
            audit_log=list(state.get("llm_calls") or []),
        )
