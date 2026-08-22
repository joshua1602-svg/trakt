"""B16a — a narrowing the sentence marked and execution did not apply.

*"What is the balance where account status is active?"* returned **Total Balance
· grouped by Account Status · 2 groups · 11,035 loans**, with the breakdown
certified. The reader asked for the balance of ACTIVE loans and was given the
whole book. B13's class — a wrong NUMBER, not a wrong receipt.

It answered because the categorical selector had **no owner at all**.
`lexical.is_filter_subject` owns the mark for NUMERIC predicates only — both its
patterns need a comparator, a `<>=` symbol or a leading digit — and those are
exactly the cases the parser already resolves.

Every guard below is pinned with the family that earned it. The rule reached 28
questions before the guards, then 13, then 5, and the questions it shed on the
way are Q7.3, Q7.4 and Q8.x on both books: the seasoning and provenance
comparison families, which is `32c263a`'s family.
"""
from __future__ import annotations

from pathlib import Path

import pytest

from mi_agent import execution_receipt as R
from mi_agent import mi_calibration as CAL
from mi_agent.mi_query_validator import load_mi_semantics
from question_interpretation.lexical import selector_mark


_REPO_ROOT = Path(__file__).resolve().parents[2]
_TAPE = (_REPO_ROOT / "demo_platform" / "workspace" / "store" / "processed"
         / "platform" / "alderbridge" / "2026-06-30"
         / "platform_canonical_typed.csv")


@pytest.fixture(scope="module")
def semantics():
    return load_mi_semantics(
        _REPO_ROOT / "mi_agent" / "mi_semantics_field_registry.yaml")


@pytest.fixture(scope="module")
def frame():
    if not _TAPE.exists():
        pytest.skip("calibration book not built")
    return CAL.default_bank_frame(_TAPE)


def _mark(question, token):
    i = question.lower().find(token.lower())
    assert i >= 0, f"{token!r} not in {question!r}"
    return selector_mark(question.lower(), i, i + len(token))


# --------------------------------------------------------------------------- #
# The lexical owner — a decision with none until now
# --------------------------------------------------------------------------- #
def test_the_numeric_owner_does_not_cover_the_categorical_selector():
    """The measurement that made this commit necessary, asserted so it cannot
    quietly stop being true."""
    from question_interpretation.lexical import is_filter_subject, condition_cut

    q = "what is the balance where account status is active"
    i = q.find("account status")
    assert is_filter_subject(q, i, i + len("account status")) is False
    assert condition_cut(q) is None
    # ...while the numeric form it DOES own still works.
    n = "how many loans have ltv above 50%"
    j = n.find("ltv")
    assert is_filter_subject(n, j, j + 3) is True


@pytest.mark.parametrize("question,token", [
    ("what is the balance where account status is active", "active"),
    ("what is the balance for interest roll-up loans", "interest roll-up"),
    ("show loans in the south east", "south east"),
    ("balance by region for joint borrowers", "joint"),
])
def test_the_sentence_marks_a_selector(question, token):
    assert _mark(question, token) is True


@pytest.mark.parametrize("question,token", [
    ("balance by broker", "broker"),
    ("balance split by broker", "broker"),
    ("balance broken down by account status", "account status"),
    ("balance per region", "region"),
])
def test_an_axis_marker_is_not_a_selector(question, token):
    """THE COLLISION. `Broker` is a value of `origination_channel` AND the word
    for the dimension, so "balance by broker" must never read as a narrowing to
    Broker — 78 questions in the corpora."""
    assert _mark(question, token) is False


def test_the_selector_and_axis_vocabularies_are_disjoint():
    """What makes the test above hold, asserted instead of re-checked at runtime.

    A runtime axis check was written and removed: it could never fire, because a
    mention preceded by an axis marker cannot also be preceded by a selector
    opener. Mutating it away changed no outcome and failed no test — which, by
    the D7 standing rule, means the paths must be searched and the result
    stated rather than the branch kept.

    This is that statement. Adding "by" to `SELECTOR_OPENERS` makes the collision
    reachable again, and fails here.
    """
    from question_interpretation.lexical import SELECTOR_OPENERS, AXIS_MARKERS

    assert set(SELECTOR_OPENERS) & set(AXIS_MARKERS) == set()
    # And no axis marker ENDS with a selector opener, which would let the regex
    # match the tail of one ("split by" ending in a hypothetical opener "by").
    for axis in AXIS_MARKERS:
        assert not any(axis.split()[-1] == opener for opener in SELECTOR_OPENERS)


def test_the_window_is_short_enough_that_a_distant_opener_does_not_claim():
    q = "for the pipeline show me the balance and the account status"
    assert _mark(q, "account status") is False


# --------------------------------------------------------------------------- #
# The value allowlist, built from the book
# --------------------------------------------------------------------------- #
def test_a_word_that_names_another_dimension_is_never_a_value(semantics, frame):
    """`Broker` is a value of `origination_channel` and a synonym of
    `broker_channel`. Read from the registry rather than kept as a list here.

    Another DIMENSION, and both qualifiers earn their place. `Redeemed` is a
    value of `account_status` and a synonym of `loan_redemption_flag` — a FLAG,
    so no collision, and excluding it would have dropped a legitimate value.
    "front book" names `seasoning_segment` and is one of its values, which is no
    collision either; that one is left to the seasoning owner.
    """
    values = R.dimension_values(frame, semantics)
    assert "broker" not in values
    assert values.get("redeemed") == "account_status"
    assert values.get("front book") == "seasoning_segment"
    assert values.get("direct") == "origination_channel"


def test_the_geography_stopwords_are_not_reused(semantics, frame):
    """`_GEO_STOPWORDS` contains "active", because a stray "active" must never
    bind a region. `Active` is a legitimate `account_status` value, and importing
    one map's assumptions into another is how the restriction would leak."""
    assert "active" in R._GEO_STOPWORDS
    assert R.dimension_values(frame, semantics).get("active") == "account_status"


def test_a_value_this_book_does_not_carry_is_not_recognised(semantics, frame):
    values = R.dimension_values(frame, semantics)
    assert "aberdeenshire" not in values
    assert "capitalised" not in values


# --------------------------------------------------------------------------- #
# The guards, each with the family that earned it
# --------------------------------------------------------------------------- #
def test_a_comparison_marker_stops_it(semantics, frame):
    """Q7.3, Q7.4 and Q8.x on both books. Without this the rule fires on 13
    questions instead of 5 and refuses the seasoning and provenance comparisons.

    A count of named values cannot replace it: "the front book" versus "our
    seasoned lending" names its second side with a synonym the frame does not
    carry, and "direct" versus "acquired" splits across two fields so each sees
    exactly one value.
    """
    values = R.dimension_values(frame, semantics)
    assert R._detect_lost_narrowing(
        "how has the balance of direct loans moved relative to acquired",
        values) == []
    assert R._detect_lost_narrowing(
        "compare the balance for direct with the rest", values) == []
    # ...and the same sentence without the comparison IS a narrowing.
    assert [f.field_key for f in R._detect_lost_narrowing(
        "what is the balance for direct loans", values)] == ["origination_channel"]


def test_two_values_of_one_field_are_not_a_narrowing(semantics, frame):
    values = R.dimension_values(frame, semantics)
    assert R._detect_lost_narrowing(
        "the balance for active loans and for redeemed loans", values) == []


def test_a_field_another_owner_resolved_is_skipped(semantics, frame):
    """The narrowing owner CONSUMES the parser's answer rather than competing
    with it. Without this the two claimed the same field and the population facet
    vanished — caught by the D2 carriage tests, not by inspection."""
    values = R.dimension_values(frame, semantics)
    q = "what is the balance where account status is active"
    assert [f.field_key for f in R._detect_lost_narrowing(q, values)] == \
        ["account_status"]
    assert R._detect_lost_narrowing(q, values, taken={"account_status"}) == []


def test_geography_is_left_to_its_own_owner(semantics, frame):
    """`_detect_geographic_scope` owns geography, reads a narrower map and
    carries geographic prose. Two owners for one decision is the thing this
    programme exists to remove."""
    facets = R.detect_requested_facets(
        "what is the balance of south east loans", semantics, frame=frame,
        requested_dimensions=R.requested_dimension_terms(
            "what is the balance of south east loans", semantics,
            list(frame.columns)))
    kinds = {(f.kind, f.field_key) for f in facets}
    assert ("geographic_scope", "collateral_geography") in kinds
    assert not [k for k in kinds if k[0] == R.KIND_LOST_NARROWING]


# --------------------------------------------------------------------------- #
# One field is never both narrowed and broken down — B15's shape
# --------------------------------------------------------------------------- #
def test_a_narrowed_field_is_not_also_raised_as_a_breakdown(semantics, frame):
    q = "what is the balance where account status is active"
    facets = R.detect_requested_facets(
        q, semantics, frame=frame,
        requested_dimensions=R.requested_dimension_terms(
            q, semantics, list(frame.columns)))
    by_field = [(f.kind, f.field_key) for f in facets
                if f.field_key == "account_status"]
    assert by_field == [(R.KIND_LOST_NARROWING, "account_status")]


# --------------------------------------------------------------------------- #
# It BLOCKS, and it is PROVABLE
# --------------------------------------------------------------------------- #
def test_a_lost_narrowing_blocks():
    assert R.KIND_LOST_NARROWING in R.NUMBER_OR_SUBJECT_FACETS


def test_a_narrowing_execution_applied_is_stamped_applied():
    """LOST at construction — the first design — would have refused every
    question the parser DOES resolve, which is 32c263a's shape."""
    facet = R.RequestedFacet(kind=R.KIND_LOST_NARROWING, label="Active",
                             field_key="account_status")

    class _Result:
        metadata = {"applied_filter_fields": ["account_status"],
                    "reconciliation": {}}
        data = None

    out = R.reconcile_facets([facet], spec=type("S", (), {"filters": {}})(),
                             query_result=_Result(), semantics={"fields": {}},
                             available_columns={"account_status"})
    assert out[0].status == R.APPLIED


def test_a_routed_narrowing_is_proven_from_the_population_ledger():
    facet = R.RequestedFacet(kind=R.KIND_LOST_NARROWING, label="Active",
                             field_key="account_status")
    envelope = {"metadata": {"populationApplied": {"applied": ["account_status"]}}}
    out = R.reconcile_routed_facets([facet], route="geo_exposure",
                                    semantics={"fields": {}},
                                    available_columns=None, envelope=envelope)
    assert out[0].status == R.APPLIED


def test_a_routed_narrowing_nothing_declares_is_lost():
    facet = R.RequestedFacet(kind=R.KIND_LOST_NARROWING, label="Active",
                             field_key="account_status")
    out = R.reconcile_routed_facets([facet], route="evolution",
                                    semantics={"fields": {}},
                                    available_columns=None, envelope={})
    assert out[0].status == R.LOST


# --------------------------------------------------------------------------- #
# The whole corpus: two questions, both constructed
# --------------------------------------------------------------------------- #
def test_the_rule_fires_on_exactly_the_two_constructed_cases(semantics, frame):
    """A statement about the CORPUS, and the reason the cases are constructed.
    697 questions contain no live instance; if that changes this fails and the
    constructed cases can be replaced by real ones."""
    import sys

    sys.path.insert(0, str(_REPO_ROOT / "due_diligence" / "evidence"
                           / "analytical_intent_v1"))
    from question_interpretation.lexical_decisions import corpus

    columns = list(frame.columns)
    hits = []
    for case in corpus():
        facets = R.detect_requested_facets(
            case["question"], semantics, frame=frame,
            requested_dimensions=R.requested_dimension_terms(
                case["question"], semantics, columns))
        if any(f.kind == R.KIND_LOST_NARROWING for f in facets):
            hits.append(case["id"])
    assert sorted(hits) == ["b16_001", "b16_002"], hits
