"""
genai_utils.py — DecisionLENS generative AI module.

Provides ``EligibilityAnalyzer``, a wrapper around the Groq API (Llama 3.3 70B)
for:

  • ``analyze_criteria()``           — structured enrollment risk analysis
  • ``generate_executive_briefing()`` — 1-page ClinOps executive summary
  • ``compare_criteria()``            — side-by-side protocol comparison

Fallback / demo mode:
  If ``GROQ_API_KEY`` is not set (or ``groq`` is not installed), all
  methods return realistic mock responses so the Streamlit app runs for
  reviewers without a live key.  Set the key in ``.env`` to enable live mode.

Response caching:
  Responses are cached in memory and persisted to disk under
  ``{project_root}/.cache/genai/`` so repeated identical calls during
  development do not consume API quota.

Typical usage::

    from src.genai_utils import EligibilityAnalyzer

    ea = EligibilityAnalyzer()                    # picks up GROQ_API_KEY
    print(ea.is_demo_mode)                        # True if key not set

    result = ea.analyze_criteria(criteria_text)   # -> dict
    briefing = ea.generate_executive_briefing(    # -> str
        prediction_results, competition_data, site_data
    )
    comparison = ea.compare_criteria(proto_a, proto_b)  # -> dict
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
from pathlib import Path
from typing import Optional

log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Optional groq import — graceful degradation if not installed
# ---------------------------------------------------------------------------

try:
    import groq
    _GROQ_AVAILABLE = True
except ImportError:  # pragma: no cover
    _GROQ_AVAILABLE = False

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

#: Default model for all API calls.
MODEL: str = "llama-3.3-70b-versatile"

#: Max output tokens for structured analysis methods (analyze / compare).
MAX_TOKENS_STRUCTURED: int = 2_048

#: Max output tokens for the executive briefing.
MAX_TOKENS_BRIEFING: int = 1_024

#: Disk cache directory (relative to project root).
CACHE_DIR: Path = Path(__file__).parent.parent / ".cache" / "genai"

# ---------------------------------------------------------------------------
# System prompts
# ---------------------------------------------------------------------------

_SYSTEM_CRITERIA: str = """\
You are a clinical trial enrollment optimization expert with 20 years of \
experience in clinical operations at major pharmaceutical companies. Your \
role is to analyze eligibility criteria and identify factors that restrict \
patient enrollment, reducing trial completion rates.

Analyze the eligibility criteria provided and return a structured JSON \
object with exactly four parallel lists (all indexed together):

- risk_factors: specific criteria text snippets that restrict enrollment
- severity_scores: for each risk factor, exactly one of "high", "medium", \
  or "low" based on expected population impact
- simplification_suggestions: plain-English suggestion to broaden each \
  criterion while preserving scientific validity
- estimated_population_impact: qualitative estimate of how much each \
  criterion narrows the eligible pool (e.g., "eliminates ~30% of target \
  patients with prior CNS involvement")

Focus on actionable, clinically grounded insights. Be specific. \
Return only valid JSON — no markdown fences, no prose outside the JSON."""

_SYSTEM_BRIEFING: str = """\
You are a senior clinical operations analyst writing for a VP of Clinical \
Operations at a mid-size biotech. Your briefing must be concise, \
data-driven, and actionable. Write in formal business English.

Format the output in exactly 4 sections with these headings:
1. FORECAST SUMMARY (2–3 sentences)
2. RISK ASSESSMENT (2–3 sentences)
3. COMPETITIVE CONTEXT (2–3 sentences)
4. RECOMMENDED ACTIONS (3–5 bullet points, each beginning with •)

Cite specific numbers from the data provided. Avoid jargon. \
Total length: ~300–400 words."""

_SYSTEM_COMPARE: str = """\
You are a clinical trial protocol expert specialising in patient recruitment \
and protocol optimisation. Compare two eligibility criteria sets and identify \
key differences that affect enrollment velocity and patient access.

Return a structured JSON object with:

- differences: list of objects, each with keys:
    criterion       : the criterion name or topic
    protocol_a      : the Protocol A wording / value
    protocol_b      : the Protocol B wording / value
    more_restrictive: "Protocol A", "Protocol B", or "Neither"
- enrollment_impact: parallel list of strings — for each difference exactly \
  one of "favorable_A", "favorable_B", or "neutral"
- overall_assessment: a 2–3 sentence qualitative summary of which protocol \
  is more enrollment-friendly and why
- recommendations: list of 3–5 actionable changes to improve the less \
  favorable protocol

Return only valid JSON — no markdown fences, no prose outside the JSON."""

# ---------------------------------------------------------------------------
# Mock responses for demo / fallback mode
# ---------------------------------------------------------------------------

_MOCK_CRITERIA: dict = {
    "risk_factors": [
        "Prior lines of therapy ≤ 2",
        "ECOG performance status 0–1 only",
        "No prior exposure to anti-PD-1/PD-L1 agents",
        "eGFR ≥ 60 mL/min/1.73 m²",
        "No brain metastases (active or treated)",
    ],
    "severity_scores": ["high", "high", "medium", "medium", "high"],
    "simplification_suggestions": [
        "Expand to ≤ 3 prior lines to include a broader refractory population.",
        "Allow ECOG PS 0–2 to include patients with moderate functional impairment.",
        "Allow patients with remote prior IO exposure (> 12 months washout).",
        "Reduce eGFR threshold to ≥ 45 mL/min to include mild renal impairment.",
        "Allow treated, stable brain metastases confirmed by MRI.",
    ],
    "estimated_population_impact": [
        "Eliminates ~25% of otherwise eligible patients with ≥ 3 prior lines.",
        "Eliminates ~15% of target patients with PS 2.",
        "Eliminates ~20% of patients who received prior checkpoint inhibitors.",
        "Eliminates ~10% of patients with mild-to-moderate CKD.",
        "Eliminates ~30% of NSCLC/melanoma patients with CNS involvement.",
    ],
}

_MOCK_BRIEFING: str = """\
DECISIONLENS EXECUTIVE BRIEFING  [DEMO MODE — no API key provided]
Clinical Trial Enrollment Decision Support

1. FORECAST SUMMARY
The model predicts an 84% probability (p_completed = 0.84) of meeting the \
450-patient enrollment target within 24 months. Predicted median enrollment \
duration is 18.3 months, with a 90th-percentile estimate of 26 months. \
The primary risk driver is phase-III competitive saturation in NSCLC.

2. RISK ASSESSMENT
Enrollment risk is classified as MEDIUM-HIGH. Five identified eligibility \
criteria restrictions are estimated to reduce the eligible pool by approximately \
55% relative to the total indication population. The "no prior IO exposure" \
criterion is the single largest risk factor and warrants immediate protocol \
committee review before IND submission.

3. COMPETITIVE CONTEXT
47 competing trials in NSCLC are actively recruiting, with a combined target \
enrollment of 38,200 patients. Recruitment saturation in the US is estimated \
at 2.7×, indicating intense site competition. Germany and South Korea represent \
the lowest-saturation, high-capability markets for accelerated site expansion.

4. RECOMMENDED ACTIONS
• Amend criterion 5 (no brain metastases) to allow treated, stable CNS disease — \
  projected to increase eligible pool by ~30%.
• Prioritise site activation in Germany (6 recommended sites) and South Korea \
  (4 recommended sites) to reduce US dependency.
• Negotiate a 60-patient expansion option at the 3 top-performing Dana-Farber \
  and MSK sites, which historically complete 77%+ of their enrolled trials.
• Implement monthly site performance reviews with automatic escalation for sites \
  below 60% of projected enrollment rate at Month 6.
• Re-run eligibility analysis after protocol amendment to confirm risk-score \
  improvement before finalising the IND package."""

_MOCK_COMPARE: dict = {
    "differences": [
        {
            "criterion": "Prior therapy lines",
            "protocol_a": "≤ 2 prior lines",
            "protocol_b": "≤ 3 prior lines",
            "more_restrictive": "Protocol A",
        },
        {
            "criterion": "Performance status",
            "protocol_a": "ECOG 0–1",
            "protocol_b": "ECOG 0–2",
            "more_restrictive": "Protocol A",
        },
        {
            "criterion": "Brain metastases",
            "protocol_a": "No brain metastases allowed",
            "protocol_b": "Treated, stable brain metastases allowed",
            "more_restrictive": "Protocol A",
        },
    ],
    "enrollment_impact": ["favorable_B", "favorable_B", "favorable_B"],
    "overall_assessment": (
        "Protocol B is materially more enrollment-friendly across all three "
        "identified differences. The cumulative population impact of Protocol A's "
        "restrictions is an estimated 40–55% reduction in the eligible pool relative "
        "to Protocol B. Protocol A's conservatism may reflect unresolved safety signals "
        "that warrant scientific justification before any amendment is submitted."
    ),
    "recommendations": [
        "Align Protocol A's prior therapy limit with Protocol B (≤ 3 lines) if "
        "safety data supports it.",
        "Expand ECOG PS allowance in Protocol A to include PS 2 patients with "
        "adequate supportive-care plans.",
        "Amend Protocol A to allow treated, stable brain metastases confirmed by "
        "gadolinium-enhanced MRI.",
        "Conduct a patient-advocacy consultation to assess acceptability of the "
        "more restrictive criteria in the target population.",
        "Model enrollment velocity for both protocols to quantify the timeline "
        "impact of the restriction differences.",
    ],
}


# ---------------------------------------------------------------------------
# EligibilityAnalyzer
# ---------------------------------------------------------------------------


class EligibilityAnalyzer:
    """
    Analyzes clinical trial eligibility criteria using the Groq API.

    **Live mode** (``GROQ_API_KEY`` is set):
      Calls ``llama-3.3-70b-versatile`` via the Groq API.  Structured
      methods (analyze / compare) use JSON mode; the briefing method
      uses streaming.  All three methods return AI-generated analysis.

    **Demo mode** (key absent or ``groq`` not installed):
      Returns realistic pre-written mock responses so the Streamlit app
      works for reviewers who don't have an API key.

    Responses are cached in memory (per-instance) and optionally on disk
    at ``CACHE_DIR`` so repeated identical calls do not consume quota.

    Args:
        api_key: Groq API key. Falls back to ``GROQ_API_KEY`` environment
                 variable if ``None``.
        model: Model ID. Defaults to ``MODEL`` (``llama-3.3-70b-versatile``).
        cache_dir: Directory for disk-based response cache.
                   Defaults to ``{project_root}/.cache/genai/``.
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        model: str = MODEL,
        cache_dir: Path = CACHE_DIR,
    ) -> None:
        self.model     = model
        self.cache_dir = cache_dir
        self._cache: dict[str, object] = {}

        resolved_key    = api_key or os.environ.get("GROQ_API_KEY", "")
        self._demo_mode = not bool(resolved_key) or not _GROQ_AVAILABLE

        if self._demo_mode:
            self._client: Optional[groq.Groq] = None
            if not _GROQ_AVAILABLE:
                log.warning(
                    "groq package not installed — EligibilityAnalyzer in demo "
                    "mode. Install with: pip install groq"
                )
            else:
                log.warning(
                    "GROQ_API_KEY not set — EligibilityAnalyzer in demo mode. "
                    "Add the key to .env to enable live analysis."
                )
        else:
            self._client = groq.Groq(api_key=resolved_key)
            log.info(
                "EligibilityAnalyzer initialised (model=%s, live mode).", model
            )

    # ------------------------------------------------------------------
    # 1. Analyze criteria
    # ------------------------------------------------------------------

    def analyze_criteria(self, criteria_text: str) -> dict:
        """
        Analyze eligibility criteria for enrollment risk factors.

        Sends the criteria text to the Groq API and returns a structured
        assessment of which criteria most restrict enrollment, along with
        simplification suggestions and population-impact estimates.

        Args:
            criteria_text: Raw eligibility criteria (inclusion + exclusion)
                           from a clinical trial protocol.

        Returns:
            dict with four parallel lists (all indexed together):
                - ``risk_factors``               : list[str]
                - ``severity_scores``            : list[str]
                  ("high" | "medium" | "low")
                - ``simplification_suggestions`` : list[str]
                - ``estimated_population_impact``: list[str]

        Note:
            Returns mock data in demo mode — all lists have 5 items.
        """
        if self._demo_mode:
            log.debug("analyze_criteria: demo mode.")
            return _MOCK_CRITERIA.copy()

        cache_key = _make_hash("analyze_criteria", criteria_text)
        cached    = self._get_cache(cache_key)
        if cached is not None:
            return cached  # type: ignore[return-value]

        try:
            response = self._client.chat.completions.create(  # type: ignore[union-attr]
                model=self.model,
                max_tokens=MAX_TOKENS_STRUCTURED,
                response_format={"type": "json_object"},
                messages=[
                    {"role": "system", "content": _SYSTEM_CRITERIA},
                    {
                        "role": "user",
                        "content": (
                            "Analyze the following eligibility criteria and return "
                            "the structured JSON as specified.\n\n"
                            f"ELIGIBILITY CRITERIA:\n{criteria_text}"
                        ),
                    },
                ],
            )
            result_text = response.choices[0].message.content or ""
            result: dict = json.loads(result_text)
            self._set_cache(cache_key, result)
            log.info(
                "analyze_criteria: %d risk factors identified.",
                len(result.get("risk_factors", [])),
            )
            return result

        except Exception as exc:
            log.error(
                "analyze_criteria API call failed: %s — returning mock.", exc
            )
            return _MOCK_CRITERIA.copy()

    # ------------------------------------------------------------------
    # 2. Generate executive briefing
    # ------------------------------------------------------------------

    def generate_executive_briefing(
        self,
        prediction_results: dict,
        competition_data: dict,
        site_data: dict,
    ) -> str:
        """
        Generate a 1-page executive briefing for a ClinOps VP.

        Synthesises enrollment forecast, competitive landscape, and top-site
        data into a concise, structured briefing with four sections.

        Args:
            prediction_results: Output from ``EnrollmentForecaster.predict()``
                                 or any dict with keys such as ``p_completed``
                                 and ``pred_duration_days``.
            competition_data:   Output from
                                 ``CompetitiveAnalyzer.get_landscape()``.
            site_data:          Top-sites DataFrame or dict from
                                 ``InvestigatorAnalyzer.get_top_sites()``.

        Returns:
            Formatted multi-line string briefing (≈ 300–400 words) with
            sections: Forecast Summary, Risk Assessment, Competitive Context,
            Recommended Actions.

        Note:
            Returns a pre-written mock briefing in demo mode.
        """
        if self._demo_mode:
            log.debug("generate_executive_briefing: demo mode.")
            return _MOCK_BRIEFING

        cache_key = _make_hash(
            "briefing",
            json.dumps(prediction_results, default=str, sort_keys=True),
            json.dumps(competition_data,   default=str, sort_keys=True),
        )
        cached = self._get_cache(cache_key)
        if cached is not None:
            return cached  # type: ignore[return-value]

        user_content = (
            "Generate a concise 1-page executive briefing using the data below.\n\n"
            f"ENROLLMENT FORECAST:\n"
            f"{json.dumps(prediction_results, indent=2, default=str)}\n\n"
            f"COMPETITIVE LANDSCAPE:\n"
            f"{json.dumps(competition_data, indent=2, default=str)}\n\n"
            f"TOP INVESTIGATOR SITES:\n"
            f"{json.dumps(site_data, indent=2, default=str)}"
        )

        try:
            # Streaming keeps the connection alive for longer responses.
            stream = self._client.chat.completions.create(  # type: ignore[union-attr]
                model=self.model,
                max_tokens=MAX_TOKENS_BRIEFING,
                messages=[
                    {"role": "system", "content": _SYSTEM_BRIEFING},
                    {"role": "user",   "content": user_content},
                ],
                stream=True,
            )
            text = ""
            for chunk in stream:
                delta = chunk.choices[0].delta.content
                if delta:
                    text += delta

            self._set_cache(cache_key, text)
            log.info(
                "generate_executive_briefing complete (%d chars).", len(text)
            )
            return text

        except Exception as exc:
            log.error(
                "generate_executive_briefing API call failed: %s — returning mock.",
                exc,
            )
            return _MOCK_BRIEFING

    # ------------------------------------------------------------------
    # 3. Compare criteria
    # ------------------------------------------------------------------

    def compare_criteria(self, criteria_a: str, criteria_b: str) -> dict:
        """
        Compare two eligibility criteria sets and highlight enrollment differences.

        Identifies per-criterion differences between two protocols and
        estimates the enrollment impact of each.

        Args:
            criteria_a: Eligibility criteria text for Protocol A.
            criteria_b: Eligibility criteria text for Protocol B.

        Returns:
            dict with keys:
                - ``differences``        : list[dict]
                    Each dict has ``criterion``, ``protocol_a``,
                    ``protocol_b``, ``more_restrictive``.
                - ``enrollment_impact``  : list[str]
                    Parallel to ``differences``.
                    One of "favorable_A" | "favorable_B" | "neutral".
                - ``overall_assessment`` : str — 2–3 sentence summary.
                - ``recommendations``    : list[str] — actionable changes.

        Note:
            Returns mock comparison data in demo mode.
        """
        if self._demo_mode:
            log.debug("compare_criteria: demo mode.")
            return _MOCK_COMPARE.copy()

        cache_key = _make_hash("compare_criteria", criteria_a, criteria_b)
        cached    = self._get_cache(cache_key)
        if cached is not None:
            return cached  # type: ignore[return-value]

        try:
            response = self._client.chat.completions.create(  # type: ignore[union-attr]
                model=self.model,
                max_tokens=MAX_TOKENS_STRUCTURED,
                response_format={"type": "json_object"},
                messages=[
                    {"role": "system", "content": _SYSTEM_COMPARE},
                    {
                        "role": "user",
                        "content": (
                            "Compare these two eligibility criteria sets and return "
                            "the structured JSON as specified.\n\n"
                            f"PROTOCOL A CRITERIA:\n{criteria_a}\n\n"
                            f"PROTOCOL B CRITERIA:\n{criteria_b}"
                        ),
                    },
                ],
            )
            result_text = response.choices[0].message.content or ""
            result: dict = json.loads(result_text)
            self._set_cache(cache_key, result)
            log.info(
                "compare_criteria: %d differences identified.",
                len(result.get("differences", [])),
            )
            return result

        except Exception as exc:
            log.error(
                "compare_criteria API call failed: %s — returning mock.", exc
            )
            return _MOCK_COMPARE.copy()

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def is_demo_mode(self) -> bool:
        """``True`` if the analyzer is running without a live API key."""
        return self._demo_mode

    def __repr__(self) -> str:
        mode = "demo" if self._demo_mode else "live"
        return f"EligibilityAnalyzer(model={self.model!r}, mode={mode!r})"

    # ------------------------------------------------------------------
    # Cache helpers
    # ------------------------------------------------------------------

    def _get_cache(self, key: str) -> Optional[object]:
        """
        Look up ``key`` — memory first, then disk.

        Args:
            key: Cache key (short hex hash).

        Returns:
            Cached value, or ``None`` if not found.
        """
        if key in self._cache:
            log.debug("Cache hit (memory): %s", key)
            return self._cache[key]

        disk_val = _load_disk(self.cache_dir, key)
        if disk_val is not None:
            self._cache[key] = disk_val
            log.debug("Cache hit (disk): %s", key)
        return disk_val

    def _set_cache(self, key: str, value: object) -> None:
        """
        Store ``value`` in memory and on disk.

        Args:
            key:   Cache key (short hex hash).
            value: JSON-serialisable value to store.
        """
        self._cache[key] = value
        _save_disk(self.cache_dir, key, value)


# ---------------------------------------------------------------------------
# Module-level cache helpers (pure functions, no class coupling)
# ---------------------------------------------------------------------------


def _make_hash(*parts: str) -> str:
    """
    Return the first 16 hex chars of the SHA-256 of the joined parts.

    Args:
        *parts: Arbitrary strings to hash together.

    Returns:
        16-character hex string, collision probability negligible for cache use.
    """
    combined = "\n".join(parts)
    return hashlib.sha256(combined.encode("utf-8")).hexdigest()[:16]


def _save_disk(cache_dir: Path, key: str, value: object) -> None:
    """
    Persist ``value`` to ``{cache_dir}/{key}.json`` (best-effort).

    Args:
        cache_dir: Directory in which to write the file.
        key:       Filename stem (hex hash).
        value:     JSON-serialisable value.
    """
    try:
        cache_dir.mkdir(parents=True, exist_ok=True)
        with open(cache_dir / f"{key}.json", "w", encoding="utf-8") as fh:
            json.dump(value, fh, indent=2, ensure_ascii=False)
    except Exception as exc:  # pragma: no cover
        log.debug("Disk cache write failed (non-fatal): %s", exc)


def _load_disk(cache_dir: Path, key: str) -> Optional[object]:
    """
    Load a cached value from ``{cache_dir}/{key}.json`` if it exists.

    Args:
        cache_dir: Directory to search.
        key:       Filename stem (hex hash).

    Returns:
        Parsed JSON value, or ``None`` if the file is absent or unreadable.
    """
    cache_file = cache_dir / f"{key}.json"
    if not cache_file.exists():
        return None
    try:
        with open(cache_file, encoding="utf-8") as fh:
            return json.load(fh)
    except Exception as exc:  # pragma: no cover
        log.debug("Disk cache read failed (non-fatal): %s", exc)
        return None
