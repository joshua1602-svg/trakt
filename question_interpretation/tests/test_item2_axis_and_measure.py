"""Item 2 — one axis-marker vocabulary; and the drill-through reports the
measure the question named.

TWO DEFECTS, NOT ONE. I proposed B3 and C4 as mirror images of a single
"field word outside the measure position taken as the measure". Diagnosing them
together showed they share a symptom and nothing else:

  C4  "breakdown of balance ACROSS LTV and ticket size"
      `_grouping_segments` split on `\\bby\\b` alone, so the grouping clause was
      never cut; `_detect_metric` (which masks nothing and relies on the caller
      pre-cutting) then read `ltv` from the AXIS clause as the measure.

  B3  "SHOW ME the LTV for LOANS with a balance above £150,000"
      the drill-through branch fires on show/list/display/drill + "loans" + no
      "by", and hard-coded `metric = _balance_metric()`. Nothing to do with the
      measure position: "what is the LTV for loans ..." and "show me the LTV for
      ACCOUNTS ..." both resolved LTV correctly.

`split by`, `broken down by` and `grouped by` passed the by-only consumers
INCIDENTALLY — they contain the word "by" — so only `per` and `across` ever
exposed the gap. Two of four consumers were blind and the declared owner,
`lexical.AXIS_MARKERS`, already held the right list.
"""

from __future__ import annotations

import re

import pytest

from mi_agent import execution_receipt as R
from mi_agent import llm_query_parser as P
from mi_agent.mi_query_validator import load_mi_semantics
from question_interpretation import lexical as L

_REGISTRY = "mi_agent/mi_semantics_field_registry.yaml"


@pytest.fixture(scope="module")
def semantics():
    return load_mi_semantics(_REGISTRY)


# --------------------------------------------------------------------------- #
# One vocabulary, four implementations
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("marker", list(L.AXIS_MARKERS))
def test_every_consumer_knows_every_marker(marker):
    """The measured defect: `_grouping_segments` and `grouping_cut` knew only
    `by`, and the other two knew all six."""
    q = "balance %s region" % marker
    assert P._grouping_segments(q)[1], "the splitter cannot cut on %r" % marker
    assert P._GROUPING_CLAUSE_RE.search(q), "the clause regex misses %r" % marker
    assert L.grouping_cut(q) is not None, "grouping_cut misses %r" % marker
    assert re.search(r"\b(?:" + L.axis_marker_alternation() + r")\s+$",
                     "balance %s " % marker, re.I), (
        "the receipt's suffix test misses %r" % marker)


def test_the_cut_consumers_read_the_owner_rather_than_a_literal():
    """The four consumers that decide WHERE THE GROUPING CLAUSE STARTS build
    their patterns from `lexical.AXIS_MARKERS`, never from a literal.

    Scoped to those four deliberately. A first version asserted that the marker
    words appear nowhere in either module and failed on
    `llm_query_parser._LAYERED_CONDITIONAL`, which contains "split by" beside
    "where", "among", "concentrat" and "breakdown of". That list answers a
    DIFFERENT question — *is this question layered enough to need the LLM?* —
    and shares one word by coincidence.

    By the rule these items earned, share what is one fact and separate what is
    two: routing-to-the-LLM and cutting-the-grouping-clause are two facts, and
    collapsing them because their vocabularies overlap would be the same error
    in the opposite direction. Examined and ruled separate, rather than asserted
    away.
    """
    # Behaviour, not source text. Two earlier versions of this test inspected
    # the source and failed on the COMMENT and then the DOCSTRING describing the
    # by-only split they had just replaced — a test measuring prose, twice. What
    # ownership actually means is that changing the owner changes the consumer,
    # so that is what is asserted.
    import re as _re
    marker = "grouped alongside"
    original = L.AXIS_MARKERS
    try:
        L.AXIS_MARKERS = original + (marker,)
        assert P._grouping_segments("balance %s region" % marker)[1], (
            "_grouping_segments did not pick up a marker added to the owner; "
            "it is not reading lexical.AXIS_MARKERS at call time")
    finally:
        L.AXIS_MARKERS = original

    # `grouping_cut` compiles its pattern at import, so ownership is asserted on
    # the value: every declared marker appears in the compiled alternation.
    for m in L.AXIS_MARKERS:
        assert _re.search(L.AXIS_MARKER_RE, "balance %s region" % m), (
            "AXIS_MARKER_RE does not cover the declared marker %r" % m)


def test_the_alternation_is_longest_first():
    """So `broken down by` is not shadowed by `by`."""
    phrases = L.axis_marker_alternation().split("|")
    lengths = [len(p) for p in phrases]
    assert lengths == sorted(lengths, reverse=True)


# --------------------------------------------------------------------------- #
# THE FULL RANGE, per the rule item 1 earned
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("marker", list(L.AXIS_MARKERS))
def test_a_two_dimension_breakdown_survives_every_marker(marker, semantics):
    """Item 1's lesson applied before it could repeat: a consolidation is
    complete when the CONSUMERS have been exercised across the vocabulary's full
    range, not when the lists agree. Longest (`broken down by`) and shortest
    (`by`) both, in the consumer that actually builds the spec."""
    q = "show me balance %s ltv and ticket size" % marker
    spec = P.parse_user_question(q, semantics)
    spec = spec[0] if isinstance(spec, tuple) else spec
    assert spec.metric == "current_outstanding_balance", (
        "%r resolved the measure to %r" % (marker, spec.metric))
    assert sorted(spec.dimensions) == ["ltv_bucket", "ticket_bucket"], (
        "%r produced dimensions %r" % (marker, spec.dimensions))


# --------------------------------------------------------------------------- #
# The drill-through
# --------------------------------------------------------------------------- #
def test_the_drill_reports_the_measure_the_question_named(semantics):
    spec = P.parse_user_question(
        "Show me the LTV for loans with a balance above £150,000", semantics)
    spec = spec[0] if isinstance(spec, tuple) else spec
    assert spec.metric == "current_loan_to_value"


@pytest.mark.parametrize("question", [
    "Show me loans with LTV above 50%",
    "Show me loans where balance is below £50,000",
    "List loans with an interest rate above 5%",
])
def test_a_genuine_drill_still_reports_the_balance(question, semantics):
    """THE CAN-FAIL. When the field word is the CONDITION rather than the
    measure, the reader asked to see matching loans, not to measure that field.
    `_measure_hits` already discriminates — it masks filter subjects — which is
    why the fix reads it rather than re-deriving the distinction."""
    spec = P.parse_user_question(question, semantics)
    spec = spec[0] if isinstance(spec, tuple) else spec
    assert spec.metric == "current_outstanding_balance"


def test_the_drill_reads_the_measure_owner(semantics):
    """One owner for "which field is the measure" on this branch. `_measure_hits`
    masks grouping regions AND filter subjects; re-deriving either here is how
    the branch came to disagree with the substitution guard about the same
    sentence."""
    hits = [k for _s, _e, k, _a in P._measure_hits(
        "show me the ltv for loans with a balance above 150000", semantics)]
    assert hits == ["current_loan_to_value"]
    assert P._measure_hits("show me loans with ltv above 50%", semantics) == []
