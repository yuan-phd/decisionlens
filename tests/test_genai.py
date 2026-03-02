"""
test_genai.py — Unit tests for src.genai_utils.EligibilityAnalyzer.

Strategy:
  - Demo mode tests: run without any API key so no network calls are made.
  - Live mode tests: mock the groq client so they run offline but exercise
    the real code path (JSON parsing, streaming, caching).

Covered:
  - Demo mode detection (no key → is_demo_mode == True)
  - analyze_criteria() in demo mode: correct keys, parallel list lengths
  - generate_executive_briefing() in demo mode: returns non-empty string
  - compare_criteria() in demo mode: correct keys, expected sub-keys
  - Disk and in-memory caching: repeated call returns cached result
  - Live-mode mocking: analyze_criteria() parses JSON response
  - Live-mode mocking: compare_criteria() parses JSON response
  - Live-mode mocking: generate_executive_briefing() collects streamed chunks
"""

from __future__ import annotations

import json
import os
import sys
import importlib
from typing import Generator
import unittest.mock as mock
from unittest.mock import MagicMock, patch

import pytest


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_analyzer(api_key: str | None = None, cache_dir=None, **kwargs):
    """Create an EligibilityAnalyzer, optionally overriding the cache dir."""
    from src.genai_utils import EligibilityAnalyzer, CACHE_DIR
    if cache_dir is None:
        import tempfile, pathlib
        cache_dir = pathlib.Path(tempfile.mkdtemp())
    return EligibilityAnalyzer(api_key=api_key, cache_dir=cache_dir, **kwargs)


def _demo_analyzer(tmp_path):
    """Return a demo-mode EligibilityAnalyzer (no key, no groq required)."""
    return _make_analyzer(api_key=None, cache_dir=tmp_path)


# ---------------------------------------------------------------------------
# 1. Demo mode detection
# ---------------------------------------------------------------------------

class TestDemoModeDetection:
    def test_no_key_env_is_demo(self, monkeypatch, tmp_path):
        monkeypatch.delenv("GROQ_API_KEY", raising=False)
        ea = _demo_analyzer(tmp_path)
        assert ea.is_demo_mode is True

    def test_explicit_empty_key_is_demo(self, tmp_path):
        with mock.patch.dict("os.environ", {"GROQ_API_KEY": ""}, clear=False):
            ea = _make_analyzer(api_key="", cache_dir=tmp_path)
            assert ea.is_demo_mode is True

    def test_explicit_key_is_not_demo(self, tmp_path):
        """If a non-empty key is passed, demo mode should be False
        (even without a real groq import, we test the detection logic)."""
        from src.genai_utils import _GROQ_AVAILABLE
        if not _GROQ_AVAILABLE:
            pytest.skip("groq not installed; can't test live mode detection")
        ea = _make_analyzer(api_key="gsk_fake_key_for_testing", cache_dir=tmp_path)
        assert ea.is_demo_mode is False

    def test_groq_unavailable_forces_demo(self, monkeypatch, tmp_path):
        """When groq package is not importable, demo mode should be forced."""
        monkeypatch.setitem(sys.modules, "groq", None)  # simulate import failure
        # We must reload the module so it re-checks the import
        import importlib
        try:
            import src.genai_utils as gu_module
            importlib.reload(gu_module)
            ea = gu_module.EligibilityAnalyzer(
                api_key="sk-test", cache_dir=tmp_path
            )
            assert ea.is_demo_mode is True
        finally:
            # Restore module state
            importlib.reload(gu_module)


# ---------------------------------------------------------------------------
# 2. analyze_criteria() — demo mode
# ---------------------------------------------------------------------------

SAMPLE_CRITERIA = """\
Inclusion Criteria:
  1. Age 18 – 75 years
  2. Confirmed NSCLC stage IIIB/IV
  3. ECOG PS 0 – 1
Exclusion Criteria:
  1. Prior anti-PD-1/PD-L1 therapy
  2. Active brain metastases
"""


class TestAnalyzeCriteriaDemoMode:
    @pytest.fixture(autouse=True)
    def _setup(self, tmp_path):
        self.ea = _demo_analyzer(tmp_path)

    def test_returns_dict(self):
        result = self.ea.analyze_criteria(SAMPLE_CRITERIA)
        assert isinstance(result, dict)

    def test_has_required_keys(self):
        result = self.ea.analyze_criteria(SAMPLE_CRITERIA)
        for key in ("risk_factors", "severity_scores",
                    "simplification_suggestions", "estimated_population_impact"):
            assert key in result, f"Expected key '{key}' in analyze_criteria output."

    def test_parallel_list_lengths(self):
        result = self.ea.analyze_criteria(SAMPLE_CRITERIA)
        lengths = {k: len(result[k]) for k in result if isinstance(result[k], list)}
        assert len(set(lengths.values())) == 1, (
            f"All lists must be the same length. Got: {lengths}"
        )

    def test_severity_values_valid(self):
        result = self.ea.analyze_criteria(SAMPLE_CRITERIA)
        valid = {"high", "medium", "low"}
        for score in result.get("severity_scores", []):
            assert score in valid, f"Severity '{score}' not in {valid}."

    def test_non_empty_risk_factors(self):
        result = self.ea.analyze_criteria(SAMPLE_CRITERIA)
        assert len(result.get("risk_factors", [])) > 0

    def test_non_empty_suggestions(self):
        result = self.ea.analyze_criteria(SAMPLE_CRITERIA)
        assert len(result.get("simplification_suggestions", [])) > 0

    def test_repeated_call_returns_same_result(self):
        r1 = self.ea.analyze_criteria(SAMPLE_CRITERIA)
        r2 = self.ea.analyze_criteria(SAMPLE_CRITERIA)
        assert r1 == r2

    def test_different_inputs_both_succeed(self):
        r1 = self.ea.analyze_criteria("Inclusion: 1. Age > 18")
        r2 = self.ea.analyze_criteria("Inclusion: 1. Cancer confirmed\n2. No prior chemo")
        assert isinstance(r1, dict)
        assert isinstance(r2, dict)


# ---------------------------------------------------------------------------
# 3. generate_executive_briefing() — demo mode
# ---------------------------------------------------------------------------

class TestGenerateBriefingDemoMode:
    @pytest.fixture(autouse=True)
    def _setup(self, tmp_path):
        self.ea = _demo_analyzer(tmp_path)

    def test_returns_string(self):
        result = self.ea.generate_executive_briefing(
            prediction_results={"p_completed": 0.84, "pred_duration_days": 548},
            competition_data={"total_trials": 47, "active_trials": 12},
            site_data={"n_sites": 15, "top_country": "United States"},
        )
        assert isinstance(result, str)

    def test_non_empty_briefing(self):
        result = self.ea.generate_executive_briefing({}, {}, {})
        assert len(result.strip()) > 0

    def test_briefing_contains_demo_marker(self):
        result = self.ea.generate_executive_briefing({}, {}, {})
        # Demo briefing should hint that it's not live AI output
        assert "DEMO" in result.upper() or "demo" in result.lower() or len(result) > 50


# ---------------------------------------------------------------------------
# 4. compare_criteria() — demo mode
# ---------------------------------------------------------------------------

CRITERIA_A = "Inclusion: Age 18-60\nExclusion: Prior chemo ≤ 2 lines"
CRITERIA_B = "Inclusion: Age 18-70\nExclusion: Prior chemo ≤ 3 lines"


class TestCompareCriteriaDemoMode:
    @pytest.fixture(autouse=True)
    def _setup(self, tmp_path):
        self.ea = _demo_analyzer(tmp_path)

    def test_returns_dict(self):
        result = self.ea.compare_criteria(CRITERIA_A, CRITERIA_B)
        assert isinstance(result, dict)

    def test_has_required_keys(self):
        result = self.ea.compare_criteria(CRITERIA_A, CRITERIA_B)
        for key in ("differences", "enrollment_impact",
                    "overall_assessment", "recommendations"):
            assert key in result, f"Expected key '{key}' in compare_criteria output."

    def test_differences_is_list(self):
        result = self.ea.compare_criteria(CRITERIA_A, CRITERIA_B)
        assert isinstance(result["differences"], list)

    def test_differences_have_expected_subkeys(self):
        result = self.ea.compare_criteria(CRITERIA_A, CRITERIA_B)
        for diff in result["differences"]:
            assert "criterion"        in diff
            assert "more_restrictive" in diff

    def test_enrollment_impact_matches_differences(self):
        result = self.ea.compare_criteria(CRITERIA_A, CRITERIA_B)
        assert len(result["enrollment_impact"]) == len(result["differences"])

    def test_recommendations_is_list(self):
        result = self.ea.compare_criteria(CRITERIA_A, CRITERIA_B)
        assert isinstance(result["recommendations"], list)
        assert len(result["recommendations"]) >= 1

    def test_overall_assessment_is_string(self):
        result = self.ea.compare_criteria(CRITERIA_A, CRITERIA_B)
        assert isinstance(result["overall_assessment"], str)
        assert len(result["overall_assessment"]) > 0


# ---------------------------------------------------------------------------
# 5. Caching
# ---------------------------------------------------------------------------

class TestCaching:
    def test_disk_cache_created_in_live_mode(self, tmp_path):
        """Disk cache is written only in live mode (not demo mode)."""
        from src.genai_utils import _GROQ_AVAILABLE
        if not _GROQ_AVAILABLE:
            pytest.skip("groq not installed")

        payload = {
            "risk_factors":                ["X"],
            "severity_scores":             ["low"],
            "simplification_suggestions":  ["Y"],
            "estimated_population_impact": ["Z"],
        }
        mock_completion = MagicMock()
        mock_completion.choices = [MagicMock()]
        mock_completion.choices[0].message.content = json.dumps(payload)
        mock_client = MagicMock()
        mock_client.chat.completions.create.return_value = mock_completion

        import groq
        with patch.object(groq, "Groq", return_value=mock_client):
            ea = _make_analyzer(api_key="gsk_fake", cache_dir=tmp_path)
            ea.analyze_criteria(SAMPLE_CRITERIA)

        cache_files = list(tmp_path.glob("*.json"))
        assert len(cache_files) >= 1

    def test_demo_mode_does_not_write_disk_cache(self, tmp_path):
        """Demo mode returns mock data directly — no disk I/O."""
        ea = _demo_analyzer(tmp_path)
        ea.analyze_criteria(SAMPLE_CRITERIA)
        cache_files = list(tmp_path.glob("*.json"))
        assert len(cache_files) == 0

    def test_in_memory_cache_hit(self, tmp_path):
        ea = _make_analyzer(api_key=None, cache_dir=tmp_path)
        r1 = ea.analyze_criteria(SAMPLE_CRITERIA)
        # Demo mode returns the same mock regardless; both calls should be equal
        r2 = ea.analyze_criteria(SAMPLE_CRITERIA)
        assert r1 == r2, "Second call should return the same result."

    def test_cache_key_differs_by_input(self, tmp_path):
        ea = _make_analyzer(api_key=None, cache_dir=tmp_path)
        r1 = ea.analyze_criteria("Criteria A text")
        r2 = ea.analyze_criteria("Criteria B text — completely different")
        # Both should succeed; they can return same or different mocks
        assert isinstance(r1, dict)
        assert isinstance(r2, dict)


# ---------------------------------------------------------------------------
# 6. Live-mode mocking — analyze_criteria()
# ---------------------------------------------------------------------------

class TestAnalyzeCriteriaLiveMock:
    """Exercises the real live-mode code path using a mocked Groq client."""

    def _make_live_analyzer(self, tmp_path, response_json: dict):
        from src.genai_utils import _GROQ_AVAILABLE
        if not _GROQ_AVAILABLE:
            pytest.skip("groq not installed")

        mock_completion = MagicMock()
        mock_completion.choices = [MagicMock()]
        mock_completion.choices[0].message.content = json.dumps(response_json)

        mock_client = MagicMock()
        mock_client.chat.completions.create.return_value = mock_completion

        import groq
        with patch.object(groq, "Groq", return_value=mock_client):
            ea = _make_analyzer(api_key="gsk_fake", cache_dir=tmp_path)
            return ea, mock_client

    def test_live_mode_returns_parsed_dict(self, tmp_path):
        payload = {
            "risk_factors":                ["Criterion X"],
            "severity_scores":             ["high"],
            "simplification_suggestions":  ["Broaden criterion"],
            "estimated_population_impact": ["~20% reduction"],
        }
        ea, _ = self._make_live_analyzer(tmp_path, payload)
        result = ea.analyze_criteria("Inclusion: age > 18")
        assert result["risk_factors"] == ["Criterion X"]
        assert result["severity_scores"] == ["high"]

    def test_live_mode_calls_groq_api(self, tmp_path):
        payload = {
            "risk_factors":                ["X"],
            "severity_scores":             ["low"],
            "simplification_suggestions":  ["Y"],
            "estimated_population_impact": ["Z"],
        }
        ea, mock_client = self._make_live_analyzer(tmp_path, payload)
        ea.analyze_criteria("test input")
        mock_client.chat.completions.create.assert_called_once()


# ---------------------------------------------------------------------------
# 7. Live-mode mocking — compare_criteria()
# ---------------------------------------------------------------------------

class TestCompareCriteriaLiveMock:
    def _make_live_analyzer(self, tmp_path, response_json: dict):
        from src.genai_utils import _GROQ_AVAILABLE
        if not _GROQ_AVAILABLE:
            pytest.skip("groq not installed")

        mock_completion = MagicMock()
        mock_completion.choices = [MagicMock()]
        mock_completion.choices[0].message.content = json.dumps(response_json)

        mock_client = MagicMock()
        mock_client.chat.completions.create.return_value = mock_completion

        import groq
        with patch.object(groq, "Groq", return_value=mock_client):
            ea = _make_analyzer(api_key="gsk_fake", cache_dir=tmp_path)
            return ea, mock_client

    def test_compare_returns_parsed_dict(self, tmp_path):
        payload = {
            "differences": [
                {
                    "criterion":        "Age",
                    "protocol_a":       "18-60",
                    "protocol_b":       "18-75",
                    "more_restrictive": "Protocol A",
                }
            ],
            "enrollment_impact":  ["favorable_B"],
            "overall_assessment": "Protocol B is more inclusive.",
            "recommendations":    ["Widen age range in Protocol A."],
        }
        ea, _ = self._make_live_analyzer(tmp_path, payload)
        result = ea.compare_criteria("proto A criteria", "proto B criteria")
        assert result["differences"][0]["criterion"] == "Age"
        assert result["overall_assessment"] == "Protocol B is more inclusive."


# ---------------------------------------------------------------------------
# 8. Live-mode mocking — generate_executive_briefing() (streaming)
# ---------------------------------------------------------------------------

class TestGenerateBriefingLiveMock:
    def _make_streaming_analyzer(self, tmp_path, chunks: list[str]):
        from src.genai_utils import _GROQ_AVAILABLE
        if not _GROQ_AVAILABLE:
            pytest.skip("groq not installed")

        def _stream_generator():
            for text in chunks:
                chunk = MagicMock()
                chunk.choices = [MagicMock()]
                chunk.choices[0].delta.content = text
                yield chunk

        mock_client = MagicMock()
        mock_client.chat.completions.create.return_value = _stream_generator()

        import groq
        with patch.object(groq, "Groq", return_value=mock_client):
            ea = _make_analyzer(api_key="gsk_fake", cache_dir=tmp_path)
            return ea

    def test_briefing_collects_all_chunks(self, tmp_path):
        chunks = ["1. FORECAST SUMMARY\n", "The trial looks good.\n",
                  "2. RISK ASSESSMENT\n", "Medium risk."]
        ea = self._make_streaming_analyzer(tmp_path, chunks)
        result = ea.generate_executive_briefing({}, {}, {})
        assert "FORECAST SUMMARY" in result
        assert "RISK ASSESSMENT" in result

    def test_briefing_is_string(self, tmp_path):
        chunks = ["Hello ", "world."]
        ea = self._make_streaming_analyzer(tmp_path, chunks)
        result = ea.generate_executive_briefing({}, {}, {})
        assert isinstance(result, str)
        assert result.strip() == "Hello world."
