"""
_theme.py — Shared DecisionLENS CSS theme for all Streamlit pages.

Call ``apply_theme()`` once at the top of each page (after set_page_config)
to ensure consistent typography, metric-card styling, and sidebar colour.
"""

from __future__ import annotations

_CSS: str = """
<style>
html, body, [class*="css"] {
    font-family: "Inter", "Segoe UI", Helvetica, Arial, sans-serif;
}
section[data-testid="stSidebar"] {
    background: #f0f4f8;
    border-right: 1px solid #dde3ea;
}
[data-testid="stMetric"] {
    background: #ffffff;
    border: 1px solid #e3e8ef;
    border-radius: 10px;
    padding: 1rem 1.25rem;
    box-shadow: 0 1px 4px rgba(0,0,0,0.05);
}
[data-testid="stMetricLabel"] { font-size: 0.78rem; color: #6b7280; }
[data-testid="stMetricValue"] { font-size: 1.75rem; color: #0077b6; font-weight: 700; }
[data-testid="stMetricDelta"] { font-size: 0.78rem; }
.section-header {
    font-size: 1.05rem; font-weight: 600; color: #1e293b;
    border-bottom: 2px solid #0077b6;
    padding-bottom: 0.3rem; margin: 1.5rem 0 0.75rem 0;
}
hr { border: none; border-top: 1px solid #e3e8ef; margin: 1.25rem 0; }
.risk-high   { color: #d32f2f; font-weight: 700; }
.risk-medium { color: #f9a825; font-weight: 700; }
.risk-low    { color: #388e3c; font-weight: 700; }
</style>
"""


def apply_theme() -> None:
    """
    Inject the shared DecisionLENS CSS into the current Streamlit page.

    Must be called after ``st.set_page_config()``.
    """
    import streamlit as st
    st.markdown(_CSS, unsafe_allow_html=True)
