#!/usr/bin/env python3
"""Stage 3, conversion 3: llm_query_parser._metric_slot reads the owner.

This is the commit that satisfies a programme acceptance criterion:

    "The subject-side clause split exists once, not three times."

So the criterion is asserted directly, by searching the production sources for a
second declaration — not merely inferred from the three delegations passing.
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

import pytest

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from question_interpretation import lexical  # noqa: E402

#: The production modules the inventory found declaring this vocabulary.
_CONVERTED = (
    _REPO_ROOT / "mi_agent" / "answer_type.py",
    _REPO_ROOT / "mi_agent" / "llm_query_parser.py",
    _REPO_ROOT / "mi_agent" / "execution_receipt.py",
)


def test_the_parser_no_longer_declares_the_vocabulary():
    from mi_agent import llm_query_parser as P
    assert not hasattr(P, "_FILTER_CLAUSE_OPENERS")
    assert not hasattr(P, "_FILTER_OPENER_RE")


def test_the_split_exists_once_not_three_times():
    """The programme acceptance criterion, asserted rather than inferred.

    A second declaration would be a tuple or set literal containing the
    distinctive opener trio. Searching the sources catches a re-introduction
    that delegation tests would not, because a new copy nothing calls yet still
    passes every behavioural test.
    """
    signature = re.compile(
        r"[\"']where[\"']\s*,\s*[\"']with[\"']\s*,\s*[\"']for[\"']", re.I)
    offenders = [p.name for p in _CONVERTED
                 if signature.search(p.read_text(encoding="utf-8"))]
    assert offenders == [], offenders


def test_that_search_can_fail():
    """Proof the search above is live: it matches the owner's own declaration,
    which is the one place the vocabulary is allowed to appear."""
    signature = re.compile(
        r"[\"']where[\"']\s*,\s*[\"']with[\"']\s*,\s*[\"']for[\"']", re.I)
    owner = (_REPO_ROOT / "question_interpretation" / "lexical.py").read_text(encoding="utf-8")
    assert signature.search(owner)


def test_it_delegates_rather_than_reimplementing(monkeypatch):
    from mi_agent import llm_query_parser as P
    monkeypatch.setattr(lexical, "metric_slot", lambda t: "SENTINEL")
    assert P._metric_slot("balance where LTV above 50%") == "SENTINEL"


def test_the_delegation_check_can_fail():
    from mi_agent import llm_query_parser as P
    assert P._metric_slot("balance where LTV above 50%") == "balance"


def test_metric_slot_does_not_cut_at_a_grouping_clause():
    """The difference from subject_side, and why the owner exposes the two cuts
    separately. The parser splits grouping upstream in _grouping_segments, so
    _metric_slot must NOT repeat it."""
    q = "balance by region where borrower age is over 70"
    assert lexical.metric_slot(q) == "balance by region"
    assert lexical.subject_side(q) == "balance"


@pytest.mark.parametrize("question,expected", [
    ("balance where LTV above 50%", "balance"),
    ("loans with LTV above 50%", "loans"),
    ("regions with the highest LTV", "regions with the highest LTV"),
    ("above 50", "above 50"),
    ("", ""),
])
def test_the_rule_is_unchanged(question, expected):
    assert lexical.metric_slot(question) == expected


def test_the_whole_corpus_is_unchanged():
    """Recomputed against the ORIGINAL rule inline, so the evidence survives the
    removal of the source it came from."""
    from question_interpretation.lexical_decisions import corpus
    openers = ("where", "with", "for", "above", "below", "over", "under",
               "greater than", "less than", "at least", "at most",
               "more than", "fewer than")
    opener_re = re.compile(
        r"\b(" + "|".join(re.escape(t) for t in
                          sorted(openers, key=len, reverse=True)) + r")\b")

    def original(text):
        head = text or ""
        m = opener_re.search(head)
        while m:
            if re.search(r"\d", head[m.end():]):
                return head[:m.start()].strip() or head.strip()
            m = opener_re.search(head, m.end())
        return head.strip()

    cases = corpus()
    # 693, not 690: pop_253/254/255 were added to the calibration bank as the
    # regression test for the point-in-time population hole. The corpus grew;
    # no existing decision moved, which is what the stability half asserts.
    assert len(cases) == 693
    differing = [c["question"] for c in cases
                 if original(c["question"]) != lexical.metric_slot(c["question"])]
    assert differing == [], differing[:5]
