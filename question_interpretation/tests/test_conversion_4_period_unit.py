#!/usr/bin/env python3
"""Stage 3, conversion 4: period_request.requested_unit reads the owner.

This is Stage 5's input. The reading is already correct — it returns the right
grain for every time-series probe, including "by quarter" and "each month",
which `_deterministic_parse` does not recognise as a time axis at all. What has
been missing is carriage, not comprehension.

Scope note: only `requested_unit` and `finer_than` moved. `requested_span` did
NOT, and the reason is recorded in `test_requested_span_is_not_purely_lexical`.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from question_interpretation import lexical  # noqa: E402


def test_period_request_no_longer_declares_the_unit_vocabulary():
    from mi_agent import period_request as PR
    assert not hasattr(PR, "_UNIT_PATTERNS")
    assert not hasattr(PR, "_UNIT_ORDER")


def test_it_delegates_rather_than_reimplementing(monkeypatch):
    from mi_agent import period_request as PR
    monkeypatch.setattr(lexical, "requested_unit", lambda q: "SENTINEL")
    assert PR.requested_unit("funded balance by month") == "SENTINEL"


def test_the_delegation_check_can_fail():
    from mi_agent import period_request as PR
    assert PR.requested_unit("funded balance by month") == "month"


@pytest.mark.parametrize("question,expected", [
    ("funded balance by month", "month"),
    ("funded balance by quarter", "quarter"),
    ("how many loans each month", "month"),
    ("total exposure each quarter", "quarter"),
    ("show the funded balance monthly", "month"),
    ("show the funded balance over time", None),
    ("based on the last few weeks", "week"),
    ("year to date", "year"),
    ("", None),
])
def test_the_rule_is_unchanged(question, expected):
    assert lexical.requested_unit(question) == expected


def test_the_finest_unit_wins():
    """Ordered finest first, so a question naming two units returns the finer.
    A series that cannot express weeks cannot answer a weekly question."""
    assert lexical.requested_unit("weekly balance by quarter") == "week"
    assert lexical.UNIT_PATTERNS[0][0] == "week"


def test_finer_than_orders_the_units():
    assert lexical.finer_than("week", "month") is True
    assert lexical.finer_than("quarter", "month") is False
    assert lexical.finer_than(None, "month") is False


def test_the_grain_is_right_on_every_time_series_probe():
    """The Stage 5 claim, asserted rather than quoted: the reading is correct
    for all 24 probes, and only the carriage is missing."""
    import yaml
    probes = yaml.safe_load(
        (_REPO_ROOT / "clause_splitting_phase1" / "probes" /
         "time_series_probes.yaml").read_text(encoding="utf-8"))["probes"]
    assert len(probes) == 24
    graded = 0
    for probe in probes:
        expected = probe["expect"].get("grouping")
        if not isinstance(expected, list):
            continue
        wanted = [g.split(":", 1)[1] for g in expected if str(g).startswith("time:")]
        if not wanted or wanted[0] == "unspecified":
            continue
        assert lexical.requested_unit(probe["question"]) == wanted[0], probe["id"]
        graded += 1
    assert graded >= 10, graded


def test_requested_span_is_not_purely_lexical_and_stayed_put():
    """Why conversion 4 is partial, recorded rather than left implicit.

    `requested_span` resolves vague recency ("recently", "a few months ago")
    against the governed seasoning configuration — lending_windows.recent_max_months
    — rather than from the wording. That is a config decision wearing a lexical
    coat, and the owner reads question text and nothing else. Moving it would
    have given the owner a config dependency it must not have.
    """
    from mi_agent import period_request as PR
    assert hasattr(PR, "_SPANS")
    assert hasattr(PR, "_VAGUE_RECENCY_RE")
    assert hasattr(PR, "governed_recent_months")
    assert not hasattr(lexical, "requested_span")
