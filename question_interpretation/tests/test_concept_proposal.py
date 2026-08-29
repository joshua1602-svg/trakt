#!/usr/bin/env python3
"""The model proposes a concept; the registry binds the field.

What these assert is the CONSTRAINT, not the model. The model is absent from
every test here, because the constraint has to hold whatever it says — that is
the point of putting the binding on the other side of it.

The measured target is the Opus run's two mis-bindings. It bound `lump sum` to
`erm_sub_product_type` and `drawdown` to `account_status`, where the book's own
catalogue claims each for `erm_product_type` and for nothing else. Both must be
unreachable here no matter what a model proposes.
"""
from __future__ import annotations

import pytest

from question_interpretation import concept_proposal as CP
from question_interpretation.concept_proposal import (
    BoundConcept, ProposalFormatError, ProposedConcept, RejectedConcept,
    bind, build_proposal_prompt, parse_proposal_response, vocabulary)

VALUES = {
    "erm_product_type": {"drawdown": "drawdown", "lump_sum": "lump_sum"},
    "ltv_bucket": {"40-50%": "40-50%", "50-60%": "50-60%"},
    "age_bucket": {"75-80": "75-80", "80-85": "80-85"},
    "occupancy_type": {"owner_occupied": "owner_occupied"},
    "origination_channel": {"direct": "direct", "broker": "broker"},
    "source_portfolio_type": {"direct": "direct", "acquired": "acquired"},
}


@pytest.fixture(scope="module")
def semantics():
    from migration_phase0.assurance_semantics import load_assurance_semantics
    return load_assurance_semantics()


@pytest.fixture(scope="module")
def tape_columns():
    """The columns the acceptance tape actually carries. Deliberately WITHOUT
    `erm_sub_product_type`, which the registry declares and the tape does not."""
    return {"erm_product_type", "occupancy_type", "origination_channel",
            "source_portfolio_type", "geographic_region_obligor",
            "broker_channel", "account_status", "current_outstanding_balance",
            "current_loan_to_value", "youngest_borrower_age", "ltv_bucket",
            "age_bucket", "current_interest_rate"}


@pytest.fixture(scope="module")
def vocab(semantics, tape_columns):
    return vocabulary(semantics, available_values=VALUES,
                      available_columns=tape_columns)


def _bind1(term, kind, vocab):
    bound, rejected = bind([ProposedConcept(kind, term)], vocab)
    return (bound or rejected)[0]


# --------------------------------------------------------------------------- #
# The Opus breaks, made unreachable
# --------------------------------------------------------------------------- #
def test_lump_sum_binds_to_the_one_field_that_claims_it(vocab):
    outcome = _bind1("lump sum", CP.KIND_VALUE, vocab)
    assert isinstance(outcome, BoundConcept)
    assert (outcome.field, outcome.value) == ("erm_product_type", "lump_sum")


def test_drawdown_binds_to_the_one_field_that_claims_it(vocab):
    outcome = _bind1("drawdown", CP.KIND_VALUE, vocab)
    assert isinstance(outcome, BoundConcept)
    assert (outcome.field, outcome.value) == ("erm_product_type", "drawdown")


def test_a_field_this_tape_does_not_carry_is_not_proposable(vocab):
    """`erm_sub_product_type` IS a registered dimension, and
    `_explicit_dimensions` binds "erm sub product type" to it WITH the tape's
    columns passed — measured. The vocabulary is book-scoped so the model never
    sees it, and the binder asserts availability rather than trusting the
    owner."""
    assert not vocab.offers(CP.KIND_DIMENSION, "erm sub product type")
    outcome = _bind1("erm sub product type", CP.KIND_DIMENSION, vocab)
    assert isinstance(outcome, RejectedConcept)
    assert outcome.reason == CP.REJECT_UNREGISTERED


def test_a_value_proposal_can_never_reach_a_dimension_field(vocab):
    """THE KIND IS WHAT PROTECTS THE BINDING. A `category_value` proposal is
    asked of the value catalogue and of nothing else, so `lump sum` cannot
    arrive at any dimension field however it is spelled."""
    assert not vocab.offers(CP.KIND_VALUE, "account status")
    assert isinstance(_bind1("account status", CP.KIND_VALUE, vocab),
                      RejectedConcept)


def test_no_raw_field_key_is_offered_as_a_proposable_term(vocab, semantics):
    keys = set(semantics.get("fields") or {})
    offered = {t for terms in vocab.terms.values() for t in terms}
    assert not (offered & keys), sorted(offered & keys)


def test_a_kind_the_vocabulary_does_not_have_is_rejected(vocab):
    outcome = _bind1("erm_product_type", "field_key", vocab)
    assert isinstance(outcome, RejectedConcept)
    assert outcome.reason == CP.REJECT_UNKNOWN_KIND


# --------------------------------------------------------------------------- #
# The threshold kind — what its absence cost, and what its presence must not
# --------------------------------------------------------------------------- #
def _threshold(term, comparator, value):
    return ProposedConcept(CP.KIND_THRESHOLD, term, comparator=comparator,
                           value=value)


def test_a_threshold_binds_to_a_governed_predicate(vocab):
    outcome = (bind([_threshold("ltv", "above", 50)], vocab)[0] or [None])[0]
    assert isinstance(outcome, BoundConcept)
    assert (outcome.field, outcome.operator, outcome.value) == (
        "current_loan_to_value", "gt", 50.0)


def test_the_comparator_vocabulary_is_delegated_not_restated():
    """`lexical.COMPARATOR_PHRASES` is THE list. A second one here would put
    the model's vocabulary and the parser's out of step by exactly the five
    phrases — `bigger than`, `larger than`, `higher than`, `smaller than`,
    `lower than` — whose absence returned the whole book as fact."""
    from question_interpretation import lexical as LEX

    assert dict(CP.comparator_phrases()) == {
        str(p).strip().lower(): op for p, op in LEX.COMPARATOR_PHRASES}
    for phrase in ("bigger than", "larger than", "higher than", "smaller than",
                   "lower than"):
        assert phrase in CP.comparator_phrases()


def test_a_negated_comparator_is_not_read_with_the_negation_dropped(vocab):
    """`no more than 50` is `le 50`, not `gt 50` — the shadowing bug the
    longest-first ordering of the comparator list exists to prevent."""
    outcome = (bind([_threshold("ltv", "no more than", 50)], vocab)[0])[0]
    assert outcome.operator == "le"


def test_a_comparator_this_estate_does_not_read_is_rejected(vocab):
    outcome = (bind([_threshold("ltv", "banana", 50)], vocab)[1])[0]
    assert outcome.reason == CP.REJECT_UNKNOWN_COMPARATOR


def test_a_threshold_missing_a_part_is_rejected_not_completed(vocab):
    for bad in (_threshold("ltv", None, 50), _threshold("ltv", "over", None),
                _threshold("ltv", "over", "quite high")):
        outcome = (bind([bad], vocab)[1])[0]
        assert outcome.reason == CP.REJECT_MALFORMED_THRESHOLD, bad


def test_AN_AXIS_IS_NOT_A_QUANTITY(vocab):
    """`age_bucket` is a band LABEL. "age_bucket over 75" compares a number
    against a name. The threshold vocabulary is the MEASURE list and nothing
    else, because the measure owner is the one that knows which governed fields
    are quantities."""
    for axis in ("age bucket", "ltv bucket", "region", "product type"):
        outcome = (bind([_threshold(axis, "over", 75)], vocab)[1])[0]
        assert isinstance(outcome, RejectedConcept), axis


def test_the_prompt_tells_the_model_not_to_reach_for_buckets(vocab):
    """THE WHOLE POINT OF THE KIND. Measured before it existed: with no way to
    say "borrowers over 75", the model proposed the bucket labels "75-80",
    "80-85", "85+", and the first filled an empty slot — turning a question
    about everyone over 75 into one about a single band, on 8 questions that
    are EXACT today."""
    prompt = build_proposal_prompt("balance for borrowers over 75", vocab)
    assert "threshold" in prompt["system"]
    assert "bucket" in prompt["system"]
    assert "over" in prompt["system"] and "at least" in prompt["system"]


def test_a_threshold_proposal_is_read_back_with_all_three_parts():
    parsed = parse_proposal_response(
        '{"concepts":[{"kind":"threshold","term":"borrower age",'
        '"comparator":"over","value":75,"covers":"borrowers over 75"}]}')
    assert parsed == [ProposedConcept("threshold", "borrower age",
                                      "borrowers over 75", "over", 75)]


def test_the_bucket_values_stay_in_the_vocabulary(vocab):
    """They are GENUINE governed values — `ltv_bucket` really does carry
    `40-50%` and some questions genuinely name a band. The threshold kind
    competes with them; removing them would be deleting real concepts to make a
    measurement look better."""
    assert vocab.offers(CP.KIND_VALUE, "40-50%")


# --------------------------------------------------------------------------- #
# Rejection, never nearest-match
# --------------------------------------------------------------------------- #
def test_an_unregistered_concept_is_rejected_and_named(vocab):
    outcome = _bind1("platinum", CP.KIND_VALUE, vocab)
    assert isinstance(outcome, RejectedConcept)
    assert outcome.reason == CP.REJECT_UNREGISTERED
    assert "platinum" in outcome.detail


def test_an_unregistered_concept_is_not_mapped_to_the_nearest_member(vocab):
    """"lump summ" is one character from a governed value. Nearest-matching it
    would be invisible to every consumer downstream."""
    outcome = _bind1("lump summ", CP.KIND_VALUE, vocab)
    assert isinstance(outcome, RejectedConcept)


def test_a_concept_two_fields_claim_is_rejected_not_resolved(vocab):
    """`direct` is a value of BOTH `origination_channel` and
    `source_portfolio_type`. Preferring one is a coin toss recorded as a fact."""
    assert "direct" in (vocab.ambiguous.get(CP.KIND_VALUE) or ())
    outcome = _bind1("direct", CP.KIND_VALUE, vocab)
    assert isinstance(outcome, RejectedConcept)
    assert outcome.reason == CP.REJECT_AMBIGUOUS


def test_the_binder_refuses_an_ambiguous_value_even_if_the_vocabulary_offers_it(
        semantics, tape_columns):
    """TWO LINES OF DEFENCE, AND THE SECOND ONE IS TESTED. The vocabulary drops
    an ambiguous value, so a mutation that made the binder resolve `direct` by
    preference left every other test in this module green. The binder must
    refuse it on its own, because it is what a caller assembling a vocabulary
    some other way still reaches."""
    catalogue = {"origination_channel": {"direct": "direct"},
                 "source_portfolio_type": {"direct": "direct"}}
    offered = vocabulary(semantics, available_values=catalogue,
                         available_columns=tape_columns)
    forced = CP.ConceptVocabulary(
        terms={**offered.terms, CP.KIND_VALUE: ("direct",)},
        ambiguous={}, cross_kind={}, semantics=semantics,
        available_values=catalogue, available_columns=tape_columns)
    outcome = _bind1("direct", CP.KIND_VALUE, forced)
    assert isinstance(outcome, RejectedConcept)
    assert outcome.reason == CP.REJECT_AMBIGUOUS


def test_the_same_word_binds_cleanly_under_a_different_kind(vocab):
    """The ambiguity is within the VALUE catalogue. As a book, `direct` is one
    thing and binds."""
    outcome = _bind1("direct", CP.KIND_BOOK, vocab)
    assert isinstance(outcome, BoundConcept)
    assert outcome.value == "direct"


def test_cross_kind_collisions_are_reported(vocab):
    """`broker` is a value of `origination_channel` and a synonym for the
    `broker_channel` axis. The kind separates them, so binding is unambiguous —
    but a model proposing the wrong kind gets a wrong binding that looks
    entirely valid, and the estate should be able to count these."""
    assert CP.KIND_VALUE in (vocab.cross_kind.get("broker") or ())
    assert CP.KIND_DIMENSION in (vocab.cross_kind.get("broker") or ())


# --------------------------------------------------------------------------- #
# Normalisation, and the trap it closes
# --------------------------------------------------------------------------- #
def test_case_is_normalised_before_the_owner_is_asked(vocab):
    """`_detect_metric` answers `current_loan_to_value` for "ltv" and None for
    "LTV". Every serving-path caller lowercases first, so the trap has never
    fired there; a term-shaped binder walks straight into it."""
    for spelling in ("ltv", "LTV", "Ltv"):
        outcome = _bind1(spelling, CP.KIND_MEASURE, vocab)
        assert isinstance(outcome, BoundConcept), spelling
        assert outcome.field == "current_loan_to_value"


def test_underscores_and_spaces_are_the_same_term(vocab):
    for spelling in ("lump sum", "lump_sum", "  Lump  Sum "):
        assert isinstance(_bind1(spelling, CP.KIND_VALUE, vocab), BoundConcept)


def test_the_curated_measure_grammar_is_proposable(vocab):
    """`balance` is dropped by `_registry_metric_terms` as an over-generic
    single token and is governed by the curated grammar instead. Leaving it out
    would make the commonest measure in the book unproposable."""
    assert vocab.offers(CP.KIND_MEASURE, "balance")
    outcome = _bind1("balance", CP.KIND_MEASURE, vocab)
    assert isinstance(outcome, BoundConcept)
    assert outcome.field == "current_outstanding_balance"


# --------------------------------------------------------------------------- #
# The prompt, and reading what comes back
# --------------------------------------------------------------------------- #
def test_the_prompt_carries_the_vocabulary_and_forbids_the_near_miss(vocab):
    prompt = build_proposal_prompt("How many lump sum loans?", vocab)
    assert "lump sum" in prompt["system"]
    assert "closest term" in prompt["system"]
    assert "How many lump sum loans?" in prompt["user"]


def test_the_prompt_never_shows_a_governed_field_key(vocab, semantics):
    prompt = build_proposal_prompt("anything", vocab)
    for key in (semantics.get("fields") or {}):
        assert key not in prompt["system"], key


def test_a_wellformed_reply_is_read(vocab):
    proposals = parse_proposal_response(
        '{"concepts":[{"kind":"category_value","term":"lump sum",'
        '"covers":"lump sum lending"}]}')
    assert proposals == [ProposedConcept("category_value", "lump sum",
                                         "lump sum lending")]


def test_a_fenced_reply_is_read(vocab):
    assert parse_proposal_response(
        '```json\n{"concepts":[{"kind":"measure","term":"balance"}]}\n```')


def test_a_reply_that_cannot_be_read_raises_rather_than_guessing():
    """A reply this cannot read is a reply that proposed nothing. Salvaging a
    fragment would put the model back in the business of choosing."""
    for bad in ("not json at all", '{"nope": 1}', '{"concepts": "balance"}',
                '{"concepts":[{"kind":"measure"}]}', '{"concepts":["balance"]}'):
        with pytest.raises(ProposalFormatError):
            parse_proposal_response(bad)


def test_an_empty_proposal_is_valid(vocab):
    assert parse_proposal_response('{"concepts":[]}') == []
    assert bind([], vocab) == ([], [])


# --------------------------------------------------------------------------- #
# The vocabulary itself
# --------------------------------------------------------------------------- #
def test_the_vocabulary_is_scoped_to_the_book_not_the_registry(semantics,
                                                               tape_columns):
    wide = vocabulary(semantics, available_values=VALUES)
    narrow = vocabulary(semantics, available_values=VALUES,
                        available_columns=tape_columns)
    assert narrow.size() < wide.size()
    assert wide.offers(CP.KIND_DIMENSION, "erm sub product type")
    assert not narrow.offers(CP.KIND_DIMENSION, "erm sub product type")


def test_every_offered_term_binds(semantics, tape_columns):
    """A term the model is invited to use and then rejected for using is a
    trap, and the rejection reads as the model's fault. Found by the census:
    `interest rate buckets` is in the registry's synonym map and
    `_explicit_dimensions` does not recognise it; `portfolio type (source)` is
    a registered business name whose parentheses the same owner does not
    match."""
    vocab = vocabulary(semantics, available_values=VALUES,
                       available_columns=tape_columns)
    for kind, terms in vocab.terms.items():
        for term in terms:
            candidate = (ProposedConcept(kind, term, comparator="over", value=0)
                         if kind == CP.KIND_THRESHOLD
                         else ProposedConcept(kind, term))
            bound, rejected = bind([candidate], vocab)
            assert bound, "offered but unbindable: %s / %s -> %s" % (
                kind, term, rejected[0].reason if rejected else "?")


def test_a_term_the_owner_will_not_bind_is_withheld_and_recorded(semantics,
                                                                 tape_columns):
    """Withheld, never silently dropped: the disagreement between the registry's
    synonym map and the question-shaped owner is a finding about the registry."""
    vocab = vocabulary(semantics, available_values=VALUES,
                       available_columns=tape_columns)
    withheld = {t for terms in vocab.withheld.values() for t in terms}
    assert withheld
    for term in withheld:
        assert not any(vocab.offers(k, term) for k in CP.CONCEPT_KINDS)


def test_a_book_with_no_catalogue_offers_no_values(semantics, tape_columns):
    empty = vocabulary(semantics, available_values=None,
                       available_columns=tape_columns)
    assert empty.terms.get(CP.KIND_VALUE) == ()
