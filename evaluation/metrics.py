"""Severity-classification metrics against the labelled gold set.

Loads ``evaluation/gold_set.json``, runs a scan over the same scope used
to build it (Phase III oncology, limit 50), matches each agent
prediction to its gold entry by ``(trial_id, check_name)``, and computes:

* precision / recall / F1 per severity level
* overall accuracy across matched issues
* Cohen's kappa (chance-corrected agreement)
* match rate — what fraction of the 44 gold entries the scan recovered

Run with:
    venv/bin/python -m evaluation.metrics

Side effects: prints a human-readable report and writes
``evaluation/metrics_results.json``.
"""

from __future__ import annotations

import json
import logging
import sys
from collections import Counter
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from agents.orchestrator import run_scan  # noqa: E402

log = logging.getLogger(__name__)

GOLD_PATH: Path = PROJECT_ROOT / "evaluation" / "gold_set.json"
RESULTS_PATH: Path = PROJECT_ROOT / "evaluation" / "metrics_results.json"

SEVERITIES: list[str] = ["CRITICAL", "HIGH", "MEDIUM", "LOW"]

# Same scope as the gold set was sampled from. Keeping this fixed makes
# the evaluation reproducible — changing it invalidates the gold labels.
SCAN_SCOPE: str = "Phase III Recruiting trials"
SCAN_FILTERS: dict[str, Any] = {
    "phase": "Phase 3",
    "overall_status": "Recruiting",
    "limit": 50,
}

EVALUATION_NOTE = """
This gold set evaluates severity classification
accuracy only. Detection recall evaluation would
require an independently annotated trial dataset
— out of scope for this demo but a natural next
step in production validation. Gold set labelled
by single annotator — in production, 2-3 clinical
data managers with adjudication would be used.
"""


def load_gold_set(path: Path = GOLD_PATH) -> list[dict]:
    """Load the labelled gold set, dropping any entry without a label."""
    raw = json.loads(path.read_text())
    labelled = [e for e in raw if e.get("gold_severity")]
    if len(labelled) != len(raw):
        log.warning(
            "Gold set has %d entries but only %d are labelled; using labelled only.",
            len(raw), len(labelled),
        )
    return labelled


def _safe_div(num: float, den: float) -> float:
    """Return ``num/den``, or 0.0 when the denominator is zero."""
    return float(num) / float(den) if den else 0.0


def per_severity_metrics(
    pairs: list[tuple[str, str]],
) -> dict[str, dict[str, float]]:
    """Compute precision, recall, F1 per severity from (gold, pred) pairs.

    Treats each severity as a one-vs-rest binary problem so the table
    matches what a clinical data manager would expect to see.
    """
    out: dict[str, dict[str, float]] = {}
    for sev in SEVERITIES:
        tp = sum(1 for g, p in pairs if g == sev and p == sev)
        fp = sum(1 for g, p in pairs if g != sev and p == sev)
        fn = sum(1 for g, p in pairs if g == sev and p != sev)
        support = sum(1 for g, _ in pairs if g == sev)
        precision = _safe_div(tp, tp + fp)
        recall = _safe_div(tp, tp + fn)
        f1 = _safe_div(2 * precision * recall, precision + recall)
        out[sev] = {
            "precision": round(precision, 4),
            "recall": round(recall, 4),
            "f1": round(f1, 4),
            "support": support,
            "tp": tp,
            "fp": fp,
            "fn": fn,
        }
    return out


def cohens_kappa(pairs: list[tuple[str, str]]) -> float:
    """Cohen's kappa over (gold, pred) pairs across all severity levels.

    κ = (p_o - p_e) / (1 - p_e), where p_o is observed agreement and
    p_e is the agreement expected by chance given each rater's marginal
    distribution. Returns 0.0 on empty input.
    """
    n = len(pairs)
    if n == 0:
        return 0.0
    p_o = sum(1 for g, p in pairs if g == p) / n
    gold_marg = Counter(g for g, _ in pairs)
    pred_marg = Counter(p for _, p in pairs)
    p_e = sum(
        (gold_marg[s] / n) * (pred_marg[s] / n) for s in SEVERITIES
    )
    if p_e >= 1.0:
        return 1.0 if p_o >= 1.0 else 0.0
    return (p_o - p_e) / (1.0 - p_e)


def match_predictions(
    gold: list[dict], prioritised: list[dict],
) -> tuple[list[tuple[str, str]], list[dict], list[dict]]:
    """Join gold entries to scan predictions on (trial_id, check_name).

    Returns:
        pairs:    (gold_severity, predicted_severity) for every match
        matched:  full row dicts for matched entries (for the JSON report)
        missing:  gold entries with no prediction (for the JSON report)
    """
    pred_index: dict[tuple[str, str], dict] = {
        (p["trial_id"], p["check_name"]): p for p in prioritised
    }
    pairs: list[tuple[str, str]] = []
    matched: list[dict] = []
    missing: list[dict] = []
    for entry in gold:
        key = (entry["trial_id"], entry["check_name"])
        pred = pred_index.get(key)
        if pred is None:
            missing.append({
                "rank": entry.get("rank"),
                "trial_id": entry["trial_id"],
                "check_name": entry["check_name"],
                "gold_severity": entry["gold_severity"],
                "system_severity": "MISSED",
            })
            continue
        pairs.append((entry["gold_severity"], pred["severity"]))
        matched.append({
            "rank": entry.get("rank"),
            "trial_id": entry["trial_id"],
            "check_name": entry["check_name"],
            "gold_severity": entry["gold_severity"],
            "predicted_severity": pred["severity"],
            "agree": entry["gold_severity"] == pred["severity"],
        })
    return pairs, matched, missing


def confusion_matrix(
    pairs: list[tuple[str, str]],
) -> dict[str, dict[str, int]]:
    """Return ``cm[gold][pred]`` for every severity pair seen."""
    cm = {g: {p: 0 for p in SEVERITIES} for g in SEVERITIES}
    for g, p in pairs:
        if g in cm and p in cm[g]:
            cm[g][p] += 1
    return cm


def _print_header(title: str) -> None:
    print()
    print("=" * 72)
    print(title)
    print("=" * 72)


def _print_per_severity(per_sev: dict[str, dict[str, float]]) -> None:
    print(f"{'severity':<10} {'precision':>10} {'recall':>10} "
          f"{'f1':>8} {'support':>9} {'tp':>4} {'fp':>4} {'fn':>4}")
    print("-" * 72)
    for sev in SEVERITIES:
        m = per_sev[sev]
        print(f"{sev:<10} {m['precision']:>10.3f} {m['recall']:>10.3f} "
              f"{m['f1']:>8.3f} {m['support']:>9} "
              f"{m['tp']:>4} {m['fp']:>4} {m['fn']:>4}")


def _print_confusion(cm: dict[str, dict[str, int]]) -> None:
    print(f"{'gold \\ pred':<14}" + "".join(f"{s:>10}" for s in SEVERITIES))
    print("-" * 72)
    for g in SEVERITIES:
        row = "".join(f"{cm[g][p]:>10}" for p in SEVERITIES)
        print(f"{g:<14}{row}")


def evaluate() -> dict[str, Any]:
    """Run the full evaluation pipeline. Returns the results dict."""
    gold = load_gold_set()
    if not gold:
        raise RuntimeError(
            "Gold set has no labelled entries — fill in gold_severity first."
        )

    _print_header(
        f"DecisionLENS v2 — severity classification metrics  "
        f"(gold n={len(gold)})",
    )
    print(f"scope        : {SCAN_SCOPE!r}")
    print(f"scope_filters: {SCAN_FILTERS}")
    print("Running scan… (this re-executes the live agent)")

    state, report = run_scan(SCAN_SCOPE, SCAN_FILTERS, save_report=False)
    prioritised = state.get("prioritised_issues") or []
    print(f"scan produced {len(prioritised)} prioritised issue(s).")

    pairs, matched, missing = match_predictions(gold, prioritised)
    n_matched = len(pairs)
    match_rate = _safe_div(n_matched, len(gold))

    per_sev = per_severity_metrics(pairs)
    accuracy = _safe_div(
        sum(1 for g, p in pairs if g == p), n_matched,
    )
    kappa = cohens_kappa(pairs)
    cm = confusion_matrix(pairs)

    _print_header("Match rate")
    print(f"  gold entries        : {len(gold)}")
    print(f"  matched in scan     : {n_matched}")
    print(f"  match rate          : {match_rate:.1%}")
    if missing:
        print(f"  unmatched gold rows : {len(missing)}")
        for m in missing[:5]:
            print(f"    - rank {m['rank']:>2}  {m['trial_id']:<14} "
                  f"{m['check_name']}")
        if len(missing) > 5:
            print(f"    … +{len(missing) - 5} more")

    _print_header("Per-severity metrics (one-vs-rest)")
    _print_per_severity(per_sev)

    _print_header("Confusion matrix (rows=gold, cols=pred)")
    _print_confusion(cm)

    _print_header("Overall agreement")
    print(f"  accuracy       : {accuracy:.3f}  ({sum(1 for g, p in pairs if g == p)}/{n_matched})")
    print(f"  Cohen's kappa  : {kappa:.3f}")

    if missing:
        _print_header("MISSED gold issues (system did not surface them)")
        print(f"{'rank':>4}  {'trial_id':<14}  {'gold':<9}  check_name")
        print("-" * 72)
        for m in missing:
            print(f"  {m['rank']:>2}  {m['trial_id']:<14}  "
                  f"{m['gold_severity']:<9}  {m['check_name']}")

    _print_header("Evaluation note")
    print(EVALUATION_NOTE.strip())

    results: dict[str, Any] = {
        "scope": SCAN_SCOPE,
        "scope_filters": SCAN_FILTERS,
        "gold_total": len(gold),
        "scan_total": len(prioritised),
        "matched": n_matched,
        "match_rate": round(match_rate, 4),
        "accuracy": round(accuracy, 4),
        "cohens_kappa": round(kappa, 4),
        "per_severity": per_sev,
        "confusion_matrix": cm,
        "matched_pairs": matched,
        "missing_gold": missing,
        "evaluation_note": EVALUATION_NOTE.strip(),
        "scan_id": getattr(report, "scan_id", None),
    }
    RESULTS_PATH.write_text(
        json.dumps(results, indent=2, ensure_ascii=False) + "\n",
    )
    print()
    print(f"Saved metrics to {RESULTS_PATH.relative_to(PROJECT_ROOT)}")
    return results


def main() -> int:
    """CLI entry point — runs ``evaluate()`` and exits 0 on success."""
    logging.basicConfig(
        level=logging.WARNING,
        format="%(levelname)s %(name)s %(message)s",
    )
    evaluate()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
