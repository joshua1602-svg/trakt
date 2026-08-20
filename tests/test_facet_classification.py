"""A dimension term is a GROUPING only when the question's words make it one.

The defect this pins
--------------------
"How many joint borrowers are there" contains no grouping clause at all — no
"by", no "per", nothing — and was assigned a grouping facet anyway, then
reported to the reader as a "grouping dimension". A slot filled with no textual
warrant is the same family as the whole-string scans this programme has found
four times over.

It is not a labelling nicety. The two kinds carry different rulings, for a
principled reason:

* A POPULATION that could not be applied changes WHICH ROWS were counted, so
  every number describes something else. "How many joint borrowers" answered
  over the whole book returns 11,035 loans and £1.96bn. That blocks.
* A GROUPING that could not be applied loses the CUT, not the truth of the
  number: balance by region on a book with no region field still returns the
  correct total balance. That is a partial answer, and collapsing it into the
  first would delete VERDICT_PARTIAL outright.
* A RANKING blocks with the population. "Which region has the highest balance"
  has no partial form — without the grouping there is no ranking, and a total
  answers nothing that was asked.

Why this file asserts KINDS and not only outcomes
-------------------------------------------------
Before the fix, the population case produced the RIGHT outcome for the WRONG
reason once groupings were made to block too. A test written against the outcome
would have passed throughout and caught nothing. So every test here asserts the
classification itself, and ``test_the_classifier_can_fail`` forces the
classification back to its defective form and requires the population case to
break — an instrument nobody has seen fail is an untested claim about the
instrument.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

_REPO = Path(__file__).resolve().parents[1]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from mi_agent import execution_receipt as R  # noqa: E402
from mi_agent.mi_query_validator import load_mi_semantics  # noqa: E402


@pytest.fixture(scope="module")
def semantics():
    return load_mi_semantics(
        str(_REPO / "mi_agent" / "mi_semantics_field_registry.yaml"))


def _kinds(question, semantics):
    """{label: kind} for the dimension facets the question produces."""
    dims = R.requested_dimension_terms(question, semantics)
    facets = R.detect_requested_facets(question, semantics,
                                       requested_dimensions=dims)
    return {f.label: f.kind for f in facets
            if f.kind in (R.KIND_GROUPING, R.KIND_POPULATION)}


# --------------------------------------------------------------------------- #
# Classification, asserted directly
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("question,term", [
    ("how many joint borrowers are there", "joint borrower"),
    ("what is the balance for joint borrowers", "joint borrower"),
    ("total lending to single borrowers", "single borrower"),
])
def test_a_term_with_no_grouping_clause_is_a_population(question, term, semantics):
    kinds = _kinds(question, semantics)
    assert kinds.get(term) == R.KIND_POPULATION, (
        f"{term!r} in {question!r} has no grouping clause governing it and must "
        f"not be classified as a breakdown axis; got {kinds}")


@pytest.mark.parametrize("question,term", [
    ("balance by region", "region"),
    ("balance per region", "region"),
    ("show me balance broken down by region", "region"),
])
def test_a_term_a_grouping_clause_governs_is_a_grouping(question, term, semantics):
    assert _kinds(question, semantics).get(term) == R.KIND_GROUPING


def test_a_second_axis_in_a_list_inherits_the_opening_by(semantics):
    """"balance by region and borrower type" — both are axes.

    The scan back from the second term must step over the connector AND over the
    first term's own words to reach the "by" that governs both.
    """
    kinds = _kinds("average LTV by region and borrower type", semantics)
    assert kinds.get("region") == R.KIND_GROUPING
    assert kinds.get("borrower type") == R.KIND_GROUPING


@pytest.mark.parametrize("question", [
    "which region has the largest balance",
    "regions with the most loans",
    "highest average LTV regions",
    "smallest region by balance",
    "top 10 regions by balance",
])
def test_a_superlative_makes_the_dimension_an_axis(question, semantics):
    """A ranking is over the breakdown, so the dimension is an axis even with
    no "by" in the sentence."""
    kinds = _kinds(question, semantics)
    assert R.KIND_POPULATION not in kinds.values(), (
        f"{question!r} ranks over the dimension; it is an axis, not a "
        f"population: {kinds}")


def test_the_axis_superlative_pattern_is_not_the_single_row_one():
    """Two different concepts must not share one module-level name.

    They did, and the later definition silently won — so three cases passed only
    because the OTHER pattern happens to contain "largest"/"highest". The case
    that exposed it is "most", which the other pattern does not carry.
    """
    assert R._AXIS_SUPERLATIVE_RE.search("regions with the most loans")
    assert not R._SUPERLATIVE_RE.search("regions with the most loans")


# --------------------------------------------------------------------------- #
# The rulings each kind carries
# --------------------------------------------------------------------------- #
def _receipt(kind, status=R.UNAVAILABLE):
    facet = R.RequestedFacet(kind=kind, label="borrower type",
                             field_key="borrower_type", status=status,
                             reason="field is unavailable in this dataset")
    return R.ExecutionReceipt(facets=[facet])


def test_an_unapplied_population_blocks():
    verdict, message = R.assess(_receipt(R.KIND_POPULATION))
    assert verdict == R.VERDICT_REFUSE
    assert "not substituted a broader figure" in message


def test_an_unapplied_ranking_blocks():
    verdict, _ = R.assess(_receipt(R.KIND_RANKING))
    assert verdict == R.VERDICT_REFUSE


def test_an_unapplied_grouping_is_a_partial_answer_not_a_refusal():
    """VERDICT_PARTIAL must stay reachable. Collapsing groupings into the
    blocking rule would delete a verdict, not extend a rule."""
    verdict, message = R.assess(_receipt(R.KIND_GROUPING))
    assert verdict == R.VERDICT_PARTIAL, "a lost breakdown is not a lost number"
    assert message


def test_a_partial_answer_leads_with_what_could_not_be_done(semantics):
    """"£2.4bn (region unavailable)" is the footnote pattern three tranches
    have been spent eliminating. Say the loss first."""
    _verdict, message = R.assess(_receipt(R.KIND_GROUPING), semantics=semantics)
    head = message.split(".")[0].lower()
    assert head.startswith("i can't break this down"), message
    assert "borrower type" in message


# --------------------------------------------------------------------------- #
# The instrument must be able to fail
# --------------------------------------------------------------------------- #
def test_the_classifier_can_fail(monkeypatch, semantics):
    """Force the classification back to its defective form and require the
    population assertion to break.

    Without this, every test above would keep passing if the classifier
    regressed to "everything is a grouping" AND the grouping rule were later
    made to block — the exact combination that made the original defect
    invisible.
    """
    monkeypatch.setattr(R, "_governed_by_grouping_clause",
                        lambda *a, **k: True)
    kinds = _kinds("how many joint borrowers are there", semantics)
    assert kinds.get("joint borrower") == R.KIND_GROUPING, (
        "the monkeypatch did not reach the classifier, so this test proves "
        "nothing about it")
    with pytest.raises(AssertionError):
        test_a_term_with_no_grouping_clause_is_a_population(
            "how many joint borrowers are there", "joint borrower", semantics)
