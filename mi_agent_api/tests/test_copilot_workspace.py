#!/usr/bin/env python3
"""mi_agent_api/tests/test_copilot_workspace.py — the Workspace-promotion
heuristic (a pure classifier over the MI envelope) and the deep-link builder.

Confirmed heuristic:
  unmet             ok is false | chart requested but none produced | truncated
  fully-served-complex   chart produced | >=15 rows | multiple dimensions |
                         multiple metrics | exploratory intent | filtered
  fully-served-simple    everything else
Only complex/unmet carry a Workspace link.
"""

from __future__ import annotations

import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from mi_agent_api import copilot_workspace as cw


def _classify(**kw):
    base = dict(ok=True, question="", spec={}, artifacts=[],
                truncated=False, max_total_rows=0)
    base.update(kw)
    return cw.classify(**base)


# --------------------------------------------------------------------------- #
# unmet
# --------------------------------------------------------------------------- #
def test_failed_answer_is_unmet():
    assert _classify(ok=False) == cw.CLASS_UNMET


def test_chart_requested_but_none_produced_is_unmet():
    assert _classify(question="show me a chart of LTV") == cw.CLASS_UNMET
    assert _classify(question="plot the trend over time") == cw.CLASS_UNMET


def test_truncated_is_unmet():
    assert _classify(truncated=True, max_total_rows=200) == cw.CLASS_UNMET


# --------------------------------------------------------------------------- #
# fully-served-complex
# --------------------------------------------------------------------------- #
def test_chart_produced_is_complex():
    assert _classify(artifacts=[{"type": "chart"}]) == cw.CLASS_COMPLEX


def test_chart_request_satisfied_by_chart_is_complex_not_unmet():
    assert _classify(question="chart of LTV",
                     artifacts=[{"type": "chart"}]) == cw.CLASS_COMPLEX


def test_large_result_threshold_is_15():
    assert _classify(max_total_rows=14) == cw.CLASS_SIMPLE
    assert _classify(max_total_rows=15) == cw.CLASS_COMPLEX


def test_multiple_dimensions_is_complex():
    assert _classify(spec={"dimensions": ["ltv_bucket", "region"]}) == cw.CLASS_COMPLEX
    assert _classify(spec={"hierarchy": ["a", "b"]}) == cw.CLASS_COMPLEX


def test_multiple_metrics_is_complex():
    assert _classify(spec={"metrics": ["a", "b"]}) == cw.CLASS_COMPLEX


def test_exploratory_intent_is_complex():
    assert _classify(spec={"intent": "forecast"}) == cw.CLASS_COMPLEX
    assert _classify(spec={"intent": "cohort"}) == cw.CLASS_COMPLEX
    assert _classify(spec={"top_n": 10}) == cw.CLASS_COMPLEX


def test_filtered_is_complex():
    assert _classify(spec={"filters": {"region": "North"}}) == cw.CLASS_COMPLEX


# --------------------------------------------------------------------------- #
# fully-served-simple
# --------------------------------------------------------------------------- #
def test_plain_lookup_is_simple():
    assert _classify(question="what is the total funded balance?",
                     spec={"intent": "kpi", "metric": "funded_balance",
                           "dimension": None, "dimensions": [], "filters": {}},
                     artifacts=[{"type": "kpi"}],
                     max_total_rows=1) == cw.CLASS_SIMPLE


# --------------------------------------------------------------------------- #
# link emission
# --------------------------------------------------------------------------- #
def test_no_link_without_base_url(monkeypatch):
    monkeypatch.delenv("TRAKT_COPILOT_WORKSPACE_BASE_URL", raising=False)
    assert cw.build_workspace_url(client="c", run="r", question="q",
                                  filters={"x": 1}) is None


def test_link_restores_context(monkeypatch):
    monkeypatch.setenv("TRAKT_COPILOT_WORKSPACE_BASE_URL",
                       "https://trakt.example/ws")
    url = cw.build_workspace_url(client="client_001", run="mi_2025_10",
                                 question="LTV by region",
                                 filters={"region": "North"})
    assert url.startswith("https://trakt.example/ws?")
    assert "client=client_001" in url
    assert "run=mi_2025_10" in url
    assert "q=LTV" in url
    assert "filters=" in url


def test_promote_only_links_complex_and_unmet(monkeypatch):
    monkeypatch.setenv("TRAKT_COPILOT_WORKSPACE_BASE_URL", "https://t.example/ws")
    # simple → no link
    cls, url = cw.promote(ok=True, question="total?", spec={}, artifacts=[],
                          truncated=False, max_total_rows=1, client="c",
                          run=None, filters=None)
    assert cls == cw.CLASS_SIMPLE and url is None
    # unmet → link
    cls, url = cw.promote(ok=False, question="x", spec={}, artifacts=[],
                          truncated=False, max_total_rows=0, client="c",
                          run=None, filters=None)
    assert cls == cw.CLASS_UNMET and url and url.startswith("https://t.example/ws")
