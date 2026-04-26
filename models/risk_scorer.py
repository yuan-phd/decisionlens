"""v1 completion-risk scorer (XGBoost wrapper).

Loads the pre-trained ``models/xgb_classifier.pkl`` (sklearn ``Pipeline``
wrapping XGBoost — trained on 573,000 real AACT trials, AUC=0.787) and
exposes two helpers used by the agent's scan node:

* ``get_completion_risk_score(trial_id)``      — one trial.
* ``get_batch_risk_scores(trial_ids)``         — many trials, one pass.

A trial's *completion risk score* is ``1 - P(completed)``: higher means
more likely to fail to enrol/complete. Both functions degrade to a
neutral 0.5 (with a logged warning) when a trial cannot be scored.

Module-level caches keep the model file and feature pipeline output in
memory after first use so repeated calls in the same process are cheap.
"""

from __future__ import annotations

import logging
import threading
from pathlib import Path
from typing import Optional

import joblib
import pandas as pd
from dotenv import load_dotenv

from src.data_pipeline import TrialDataPipeline

load_dotenv()

log = logging.getLogger(__name__)

PROJECT_ROOT: Path = Path(__file__).resolve().parent.parent
MODEL_PATH: Path = PROJECT_ROOT / "models" / "xgb_classifier.pkl"
DATA_DIR: Path = PROJECT_ROOT / "data" / "processed"

# Frozen feature list — must match xgb_classifier.feature_names_in_.
# Re-validated against the loaded model on first use.
MODEL_FEATURES: list[str] = [
    "phase_numeric",
    "n_facilities",
    "n_countries",
    "n_eligibility_criteria",
    "geographic_concentration",
    "condition_prevalence_proxy",
    "sponsor_historical_performance",
    "competing_trials_count",
    "enrollment",
    "enrollment_type_is_actual",
    "is_multicountry",
    "sponsor_type",
    "intervention_model",
    "masking",
]

RISK_MODEL_VERSION: str = "xgboost_v2_auc0799_22k_trials_realAACT"
NEUTRAL_SCORE: float = 0.5

# Canonical note for the most common fallback case. Other failure modes
# (model load error, predict exception) carry their own notes so debug
# logs can distinguish root causes.
NEUTRAL_NOTE_UNKNOWN_TRIAL: str = (
    "risk score unavailable for withdrawn/unknown trials"
)


def _risk_payload(
    score: float, reliable: bool, note: str = "",
) -> dict:
    """Build the standard risk-score response dict.

    Every risk-scoring call returns one of these so downstream code
    (escalation, provenance, UI) can branch on ``risk_reliable`` without
    sentinel-value checks against 0.5.
    """
    return {
        "risk_score": float(score),
        "risk_reliable": bool(reliable),
        "risk_model_version": RISK_MODEL_VERSION,
        "note": note,
    }

# Module-level caches. Guarded by a lock so threaded callers don't race
# during the initial (one-shot) load.
_model = None  # noqa: ANN001 — sklearn Pipeline; type kept loose
_features_df: Optional[pd.DataFrame] = None
_lock = threading.Lock()


def _load_model():  # noqa: ANN202 — sklearn Pipeline
    """Lazily load and cache the XGBoost pipeline. Returns ``None`` on failure."""
    global _model
    if _model is not None:
        return _model
    with _lock:
        if _model is not None:
            return _model
        try:
            _model = joblib.load(MODEL_PATH)
            expected = list(getattr(_model, "feature_names_in_", []))
            if expected and expected != MODEL_FEATURES:
                log.warning(
                    "Model feature_names_in_ %s differs from MODEL_FEATURES %s; "
                    "continuing with model's order.", expected, MODEL_FEATURES,
                )
        except Exception as exc:  # noqa: BLE001 — graceful degradation
            log.exception("Failed to load risk model from %s: %s", MODEL_PATH, exc)
            _model = None
    return _model


def _load_features() -> Optional[pd.DataFrame]:
    """Lazily build and cache the feature DataFrame keyed by ``nct_id``.

    The v1 ``TrialDataPipeline`` is whole-dataset by design: it joins all
    AACT tables and filters to interventional, non-withdrawn trials. We
    run it once per process and index the result on ``nct_id`` so
    per-trial lookups are O(1).
    """
    global _features_df
    if _features_df is not None:
        return _features_df
    with _lock:
        if _features_df is not None:
            return _features_df
        try:
            pipeline = TrialDataPipeline()
            tables = pipeline.load_raw_data(DATA_DIR)
            df = pipeline.engineer_features(tables)
            missing = [c for c in MODEL_FEATURES if c not in df.columns]
            if missing:
                log.error(
                    "Feature pipeline missing required columns %s; risk scoring "
                    "will return neutral.", missing,
                )
                return None
            _features_df = df.set_index("nct_id", drop=False)
            log.info(
                "Risk-scorer feature cache built: %d trials, %d features.",
                len(_features_df), len(MODEL_FEATURES),
            )
        except Exception as exc:  # noqa: BLE001 — graceful degradation
            log.exception("Failed to build feature DataFrame: %s", exc)
            _features_df = None
    return _features_df


def _completion_prob_to_risk(prob: float) -> float:
    """Convert P(completed) to completion-risk score, clipped to [0, 1]."""
    risk = 1.0 - float(prob)
    if risk < 0.0:
        return 0.0
    if risk > 1.0:
        return 1.0
    return risk


# Escalation thresholds — see CLAUDE.md "Escalation logic".
HIGH_TO_CRITICAL_THRESHOLD: float = 0.6
MEDIUM_TO_HIGH_THRESHOLD: float = 0.7


def escalate_severity(severity_rule: str, risk: dict) -> str:
    """Apply v1 risk-model escalation to a rule-engine severity.

    ``risk`` is the payload returned by ``get_completion_risk_score`` /
    ``get_batch_risk_scores``. Escalation fires only when
    ``risk["risk_reliable"]`` is True — neutral-fallback scores must not
    trigger upgrades, because 0.5 is a "we don't know" signal, not an
    observed risk level.

    Rules (when reliable). Thresholds are inclusive — a score that
    exactly equals the threshold escalates.
      - HIGH   + risk_score >= 0.6 → CRITICAL
      - MEDIUM + risk_score >= 0.7 → HIGH
      - Otherwise the input severity is returned unchanged.

    LOW never escalates regardless of score. Unknown severity strings
    pass through unchanged so callers can layer their own bookkeeping.
    """
    if not isinstance(risk, dict) or not risk.get("risk_reliable"):
        return severity_rule

    score = float(risk.get("risk_score", NEUTRAL_SCORE))
    if severity_rule == "HIGH" and score >= HIGH_TO_CRITICAL_THRESHOLD:
        return "CRITICAL"
    if severity_rule == "MEDIUM" and score >= MEDIUM_TO_HIGH_THRESHOLD:
        return "HIGH"
    return severity_rule


def get_completion_risk_score(trial_id: str) -> dict:
    """Return a risk-score payload for a single trial.

    Returns a dict with ``risk_score`` (float in [0,1]), ``risk_reliable``
    (bool), ``risk_model_version`` (str), and ``note`` (str — empty on
    success, descriptive on fallback). Fallback cases return
    ``risk_score=0.5`` with ``risk_reliable=False`` and log a warning,
    so callers always get a complete payload and never crash.
    """
    try:
        model = _load_model()
        feats = _load_features()
        if model is None or feats is None:
            log.warning(
                "Risk scoring unavailable for %s (model or features absent); "
                "returning neutral %.2f.", trial_id, NEUTRAL_SCORE,
            )
            return _risk_payload(
                NEUTRAL_SCORE, reliable=False,
                note="risk score unavailable: model or features failed to load",
            )

        if trial_id not in feats.index:
            log.warning(
                "Trial %s not found in feature cache (filtered out by pipeline "
                "or unknown ID); returning neutral %.2f.",
                trial_id, NEUTRAL_SCORE,
            )
            return _risk_payload(
                NEUTRAL_SCORE, reliable=False, note=NEUTRAL_NOTE_UNKNOWN_TRIAL,
            )

        row = feats.loc[[trial_id], MODEL_FEATURES]
        prob_completed = float(model.predict_proba(row)[0, 1])
        return _risk_payload(
            _completion_prob_to_risk(prob_completed), reliable=True,
        )

    except Exception as exc:  # noqa: BLE001 — graceful degradation
        log.warning(
            "Risk scoring failed for %s (%s); returning neutral %.2f.",
            trial_id, exc, NEUTRAL_SCORE,
        )
        return _risk_payload(
            NEUTRAL_SCORE, reliable=False,
            note=f"risk score unavailable: prediction error ({exc})",
        )


def get_batch_risk_scores(trial_ids: list[str]) -> dict[str, dict]:
    """Score many trials in a single pipeline call.

    Returns a dict mapping every input ``trial_id`` to the standard
    risk-score payload (see ``get_completion_risk_score``). Trials that
    cannot be scored individually fall back to the neutral payload
    (``risk_score=0.5``, ``risk_reliable=False``) so the caller always
    gets a complete result. Input order is preserved.
    """
    if not trial_ids:
        return {}

    # Default every requested trial to the unknown-trial neutral payload;
    # overwrite below on success.
    results: dict[str, dict] = {
        tid: _risk_payload(
            NEUTRAL_SCORE, reliable=False, note=NEUTRAL_NOTE_UNKNOWN_TRIAL,
        )
        for tid in trial_ids
    }

    model = _load_model()
    feats = _load_features()
    if model is None or feats is None:
        log.warning(
            "Risk scoring unavailable (model or features absent); returning "
            "neutral for all %d requested trial(s).", len(trial_ids),
        )
        fallback = _risk_payload(
            NEUTRAL_SCORE, reliable=False,
            note="risk score unavailable: model or features failed to load",
        )
        return {tid: dict(fallback) for tid in trial_ids}

    requested = list(dict.fromkeys(trial_ids))  # preserve order, dedupe
    present = [tid for tid in requested if tid in feats.index]
    missing = [tid for tid in requested if tid not in feats.index]
    for tid in missing:
        log.warning(
            "Trial %s not found in feature cache; using neutral %.2f.",
            tid, NEUTRAL_SCORE,
        )

    if not present:
        return results

    try:
        batch = feats.loc[present, MODEL_FEATURES]
        probs_completed = model.predict_proba(batch)[:, 1]
        for tid, p in zip(present, probs_completed):
            results[tid] = _risk_payload(
                _completion_prob_to_risk(p), reliable=True,
            )
    except Exception as exc:  # noqa: BLE001 — graceful degradation
        log.warning(
            "Batch predict failed (%s); falling back to per-trial scoring.", exc,
        )
        for tid in present:
            results[tid] = get_completion_risk_score(tid)

    return results
