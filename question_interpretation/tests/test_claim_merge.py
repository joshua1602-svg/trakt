#!/usr/bin/env python3
"""The merge: what happens to what the model says.

No model appears in this file. Every property here has to hold whatever it
proposes — that is the point of putting the merge on the other side of it.
"""
from __future__ import annotations

import pytest

from question_interpretation import claim_merge as CM
from question_interpretation.claim_merge import (
    MergeFinding, SlotValue, deterministic_slots, merge, merged_contract)
from question_interpretation.completeness import ExecutedContract
from question_interpretation.concept_proposal import (
    BoundConcept, ProposedConcept, RejectedConcept)
from question_interpretation.schema import (
    CHOSEN_BY_A_PERSON, PROV_CALLER_CONTEXT, PROV_DEFAULT, PROV_EXPLICIT_USER,
    PROV_MODEL_INFERRED, SCOPE_PROVENANCES)


def _bound(kind, term, field, value=None):
    return BoundConcept(ProposedConcept(kind, term), field, value, "test")


def _rejected(kind, term, reason, detail=""):
    return RejectedConcept(ProposedConcept(kind, term), reason, detail)


def _slot(slot, key, value, provenance):
    return SlotValue(slot, key, value, provenance)


# --------------------------------------------------------------------------- #
# Rule 1 — the model may fill an empty slot
# --------------------------------------------------------------------------- #
def test_the_model_fills_an_empty_slot_and_it_is_recorded_as_inferred():
    """Q03A: "How many drawdown loans have LTV above 50%?" applied the
    threshold, dropped `drawdown`, and answered over the whole book."""
    existing = (_slot("row_predicates", "current_loan_to_value", 50.0,
                      PROV_EXPLICIT_USER),)
    result = merge(existing,
                   [_bound("category_value", "drawdown", "erm_product_type",
                           "drawdown")])
    filled = result.filled_by_model
    assert [(s.slot, s.key, s.value) for s in filled] == [
        ("row_predicates", "erm_product_type", "drawdown")]
    assert filled[0].provenance == PROV_MODEL_INFERRED
    assert result.findings[0].outcome == CM.FILLED_BY_MODEL


def test_a_predicate_on_another_field_is_a_different_slot():
    """`erm_product_type` and `current_loan_to_value` are two slots, not one
    slot written twice — which is what makes adding `drawdown` beside an
    existing LTV threshold a FILL and not an overwrite."""
    existing = (_slot("row_predicates", "current_loan_to_value", 50.0,
                      PROV_EXPLICIT_USER),)
    result = merge(existing, [_bound("category_value", "drawdown",
                                     "erm_product_type", "drawdown")])
    assert len(result.slots) == 2
    assert not result.conflicts


# --------------------------------------------------------------------------- #
# Rule 2 — the model may not overwrite a filled one
# --------------------------------------------------------------------------- #
def test_a_slot_the_reader_filled_is_never_overwritten():
    existing = (_slot("row_predicates", "erm_product_type", "drawdown",
                      PROV_EXPLICIT_USER),)
    result = merge(existing, [_bound("category_value", "lump sum",
                                     "erm_product_type", "lump_sum")])
    assert [s.value for s in result.slots] == ["drawdown"]
    assert not result.filled_by_model
    assert result.conflicts[0].outcome == CM.DECLINED_PERSON


def test_a_caller_context_slot_is_never_overwritten():
    existing = (_slot("source_scope", None, "direct", PROV_CALLER_CONTEXT),)
    result = merge(existing, [_bound("source_book", "acquired",
                                     "portfolio_lens", "acquired")])
    assert [s.value for s in result.slots] == ["direct"]
    assert result.conflicts[0].outcome == CM.DECLINED_PERSON


def test_A_GOVERNED_DEFAULT_IS_A_FILLED_SLOT():
    """THE DEFINITION THIS MODULE TURNS ON.

    `chat_routing.py:1150` makes "Show me the trend." refuse by testing
    `subject.provenance == PROV_DEFAULT`. If the merge treated that default as
    an empty slot the model would fill it, the provenance would stop being
    `default`, the guard would stop firing, and the question would answer —
    which is exactly how the Opus run walked through it."""
    existing = (_slot("subject", None, "current_outstanding_balance",
                      PROV_DEFAULT),)
    result = merge(existing, [_bound("measure", "ltv", "current_loan_to_value")])
    assert [s.provenance for s in result.slots] == [PROV_DEFAULT]
    assert not result.filled_by_model
    assert result.conflicts[0].outcome == CM.DECLINED_DEFAULT


def test_a_filled_slot_with_no_recorded_provenance_is_declined_and_named():
    """`SubjectClaim.provenance` is None wherever the measure came from the
    question rather than from the governed default. The decline is the same;
    the label says we could not tell which case we were in."""
    existing = (_slot("subject", None, "loan_count", None),)
    result = merge(existing, [_bound("measure", "balance",
                                     "current_outstanding_balance")])
    assert not result.filled_by_model
    assert result.conflicts[0].outcome == CM.DECLINED_UNRECORDED


def test_agreement_is_not_a_conflict_and_changes_nothing():
    existing = (_slot("row_predicates", "erm_product_type", "drawdown",
                      PROV_EXPLICIT_USER),)
    result = merge(existing, [_bound("category_value", "drawdown",
                                     "erm_product_type", "drawdown")])
    assert result.findings[0].outcome == CM.AGREED
    assert not result.conflicts
    assert [s.provenance for s in result.slots] == [PROV_EXPLICIT_USER]


# --------------------------------------------------------------------------- #
# Rule 3 — a disagreement is a finding, not a resolution
# --------------------------------------------------------------------------- #
def test_a_conflict_picks_no_winner_in_either_direction():
    existing = (_slot("subject", None, "current_loan_to_value",
                      PROV_EXPLICIT_USER),)
    result = merge(existing, [_bound("measure", "balance",
                                     "current_outstanding_balance")])
    finding = result.conflicts[0]
    assert finding.deterministic == "current_loan_to_value"
    assert finding.proposed == "current_outstanding_balance"
    assert [s.value for s in result.slots] == ["current_loan_to_value"]


def test_every_conflict_carries_both_sides_and_the_authority():
    existing = (_slot("dimensions", "ltv_bucket", "ltv_bucket",
                      PROV_EXPLICIT_USER),)
    result = merge(existing, [_bound("dimension", "ltv bucket", "ltv_bucket")])
    assert result.findings[0].outcome == CM.AGREED
    result2 = merge((_slot("dataset", None, "funded", PROV_DEFAULT),),
                    [_bound("dataset", "pipeline", "dataset", "pipeline")])
    f = result2.conflicts[0]
    assert (f.deterministic, f.deterministic_provenance, f.proposed) == (
        "funded", PROV_DEFAULT, "pipeline")


# --------------------------------------------------------------------------- #
# Ambiguity must not become silence
# --------------------------------------------------------------------------- #
def test_an_ambiguous_proposal_is_recorded_and_is_not_absence():
    """Q20C's shape: the model dropped `drawdown` entirely and proposed
    nothing. A rejection that produced a silent non-fill would be the same
    object as that, and the failure this split exists to guard against."""
    nothing_proposed = merge((), [], [])
    ambiguous = merge((), [], [_rejected(
        "category_value", "direct",
        "more than one governed field claims this concept", "direct")])
    assert nothing_proposed.findings == ()
    assert len(ambiguous.findings) == 1
    assert ambiguous.findings[0].outcome == CM.AMBIGUOUS
    assert "direct" in str(ambiguous.findings[0].proposed)
    assert nothing_proposed.as_dict() != ambiguous.as_dict()


def test_an_unbindable_proposal_is_recorded_too():
    result = merge((), [], [_rejected("category_value", "platinum",
                                      "not a registered concept", "platinum")])
    assert result.findings[0].outcome == CM.UNBINDABLE
    assert not result.ambiguous


def test_a_proposal_whose_kind_addresses_no_slot_is_recorded():
    result = merge((), [_bound("field_key", "erm_product_type",
                               "erm_product_type")])
    assert result.findings[0].outcome == CM.UNBINDABLE
    assert not result.filled_by_model


# --------------------------------------------------------------------------- #
# Provenance
# --------------------------------------------------------------------------- #
def test_model_inferred_is_in_the_one_provenance_vocabulary():
    """One fact, one vocabulary. A second vocabulary for one fact is the defect
    this programme has removed six times."""
    assert PROV_MODEL_INFERRED in SCOPE_PROVENANCES


def test_model_inferred_is_never_a_person():
    assert PROV_MODEL_INFERRED not in CHOSEN_BY_A_PERSON
    filled = merge((), [_bound("category_value", "drawdown",
                               "erm_product_type", "drawdown")])
    assert not filled.slots[0].chosen_by_a_person


def test_the_claims_that_expose_stated_by_user_say_False_for_it():
    from question_interpretation.schema import (DatasetClaim, FILLED,
                                                SourceScopeClaim)
    assert SourceScopeClaim(state=FILLED, scope="direct",
                            provenance=PROV_MODEL_INFERRED).stated_by_user is False
    assert DatasetClaim(state=FILLED, dataset="pipeline",
                        provenance=PROV_MODEL_INFERRED).stated_by_user is False


# --------------------------------------------------------------------------- #
# Feeding the completeness check
# --------------------------------------------------------------------------- #
def test_the_merged_contract_carries_what_the_model_filled():
    contract = ExecutedContract(filters=("current_loan_to_value",))
    result = merge((_slot("row_predicates", "current_loan_to_value", 50.0,
                          PROV_EXPLICIT_USER),),
                   [_bound("category_value", "drawdown", "erm_product_type",
                           "drawdown")])
    merged = merged_contract(contract, result)
    assert set(merged.filters) == {"current_loan_to_value", "erm_product_type"}


def test_the_merged_contract_does_NOT_carry_what_the_merge_declined():
    """Telling the check a concept survived when the merge refused it would
    make the check certify a loss it exists to catch."""
    contract = ExecutedContract(metric="current_loan_to_value")
    result = merge((_slot("subject", None, "current_loan_to_value",
                          PROV_EXPLICIT_USER),),
                   [_bound("measure", "balance", "current_outstanding_balance")])
    merged = merged_contract(contract, result)
    assert merged.metric == "current_loan_to_value"


def test_a_declined_concept_never_reaches_the_check_as_a_new_field():
    """The stronger form of the test above, and the one that bites: a declined
    proposal on a field the CONTRACT does not carry. Passing it through would
    tell the completeness check that a concept the merge refused had survived,
    and the check would certify the very loss it exists to catch."""
    contract = ExecutedContract(filters=(), dimensions=())
    result = merge((_slot("row_predicates", "erm_product_type", "drawdown",
                          PROV_EXPLICIT_USER),),
                   [_bound("category_value", "lump sum", "erm_product_type",
                           "lump_sum")])
    assert result.conflicts
    merged = merged_contract(contract, result)
    assert merged.filters == (), merged.filters
    assert merged.dimensions == ()


def test_a_merge_that_filled_nothing_leaves_the_contract_alone():
    contract = ExecutedContract(filters=("a",), dimensions=("b",), metric="c")
    merged = merged_contract(contract, merge((), []))
    assert (merged.filters, merged.dimensions, merged.metric) == (
        ("a",), ("b",), "c")


# --------------------------------------------------------------------------- #
# The deterministic side, read from the real contract
# --------------------------------------------------------------------------- #
def test_a_filled_axis_or_predicate_is_read_as_the_readers_own():
    """`DimensionClaim` and `RowPredicateClaim` carry no provenance field and
    do not need one: the parser never raises either by default. If that ever
    stops being true the error is safe in the only direction that matters — a
    defaulted axis is declined more firmly, never filled more freely."""
    from question_interpretation.schema import (DimensionClaim, FILLED,
                                                QuestionInterpretation,
                                                RowPredicateClaim)
    qi = QuestionInterpretation(question="q")
    qi.dimensions = [DimensionClaim(state=FILLED, candidate_concept="ltv_bucket")]
    qi.row_predicates = [RowPredicateClaim(state=FILLED,
                                           field_key="current_loan_to_value",
                                           operator="gt", value=50.0)]
    slots = {s.address: s for s in deterministic_slots(qi)}
    assert slots[("dimensions", "ltv_bucket")].chosen_by_a_person
    assert slots[("row_predicates", "current_loan_to_value")].chosen_by_a_person


def test_an_empty_claim_occupies_no_slot():
    from question_interpretation.schema import (EMPTY, QuestionInterpretation,
                                                SubjectClaim)
    qi = QuestionInterpretation(question="q")
    qi.subject = SubjectClaim(state=EMPTY)
    assert deterministic_slots(qi) == ()
