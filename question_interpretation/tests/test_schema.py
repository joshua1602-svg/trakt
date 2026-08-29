#!/usr/bin/env python3
"""Tests for the QuestionInterpretation schema — Stage 1.

Standing condition 4: no instrument ships without a test proving it can fail.
Five times in this programme an instrument has reproduced the exact defect it
was built to find. So every guarantee below is asserted twice — once that it
holds, and once that the assertion FAILS when the guarantee is removed. A test
that cannot fail is not evidence.

The mutation half is written against a real mutation of the object under test
(a wrong role, a wrong state, an execution word in a linguistic slot), not
against a stubbed double.
"""
from __future__ import annotations

import sys
from dataclasses import fields as dataclass_fields
from pathlib import Path

import pytest

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from question_interpretation.schema import (  # noqa: E402
    AMOUNT, CONFIGURED, COUNT, DIMENSION_ROLES, EMPTY, FILLED, FILTER,
    GROUPING, OPERATION_TYPES, STATED, STATES, UNRESOLVABLE, UNRESOLVED_ROLE,
    DimensionClaim, FilterClaim, OperationClaim, PopulationClaim,
    QuestionInterpretation, Slot, Span, SubjectClaim, TargetClaim, TimeClaim,
)

# --------------------------------------------------------------------------- #
# The central distinction: linguistic claims, not execution claims.
# --------------------------------------------------------------------------- #
#: Words that belong to ExecutionFacet and must never appear as a slot state or
#: a field name on this object.
_EXECUTION_VOCABULARY = {"applied", "unavailable", "unsupported", "rejected",
                         "lost", "status"}


def _all_field_names() -> set:
    names = set()
    for cls in (Slot, OperationClaim, SubjectClaim, DimensionClaim, FilterClaim,
                TargetClaim, PopulationClaim, TimeClaim, QuestionInterpretation):
        names |= {f.name for f in dataclass_fields(cls)}
    return names


def test_no_execution_vocabulary_on_the_object():
    """`applied` and `lost` carry execution evidence a pre-execution object
    cannot have. They must stay in ExecutionFacet."""
    leaked = _EXECUTION_VOCABULARY & (set(STATES) | _all_field_names())
    assert not leaked, leaked


def test_the_execution_vocabulary_check_can_fail():
    """Proof the test above is live: it catches a leak when one is introduced."""
    polluted = _EXECUTION_VOCABULARY & (set(STATES) | _all_field_names() | {"applied"})
    assert polluted == {"applied"}


def test_no_slot_carries_a_resolved_field_key():
    """Resolution belongs to ResolvedConcept. The object may only carry a
    CANDIDATE, and the name must say so."""
    names = _all_field_names()
    assert "field_key" not in names
    assert "alt_keys" not in names
    assert "candidate_concept" in names


def test_the_field_key_check_can_fail():
    assert "field_key" in (_all_field_names() | {"field_key"})


# --------------------------------------------------------------------------- #
# Slot states
# --------------------------------------------------------------------------- #
def test_slot_defaults_to_empty():
    assert Slot().state == EMPTY


def test_an_unknown_state_is_rejected():
    with pytest.raises(ValueError):
        Slot(state="applied")


def test_the_state_check_can_fail():
    """If the guard were removed, `applied` would be accepted. Prove the guard
    is what rejects it, rather than something incidental."""
    assert "applied" not in STATES
    Slot(state=UNRESOLVABLE)  # a legitimate state still constructs


def test_unresolvable_is_a_state_not_an_absence():
    """The whole point: an unresolved fragment is recorded, never dropped."""
    s = Slot(state=UNRESOLVABLE, raw_text="the good book",
             reason="no governed value")
    assert s.state == UNRESOLVABLE and s.raw_text == "the good book"


# --------------------------------------------------------------------------- #
# dimensions[].role — the fix for the KIND_GROUPING conflation
# --------------------------------------------------------------------------- #
def test_role_distinguishes_axis_from_filter():
    """"balance by region for joint borrowers" — both are dimensions, and
    KIND_GROUPING cannot tell them apart. Role must."""
    qi = QuestionInterpretation(question="balance by region for joint borrowers")
    qi.dimensions = [
        DimensionClaim(state=FILLED, raw_text="region", role=GROUPING),
        DimensionClaim(state=FILLED, raw_text="joint borrowers", role=FILTER),
    ]
    assert [d.raw_text for d in qi.dimensions_with_role(GROUPING)] == ["region"]
    assert [d.raw_text for d in qi.dimensions_with_role(FILTER)] == ["joint borrowers"]


def test_the_role_split_can_fail():
    """Collapse both roles to one, as KIND_GROUPING does, and the split stops
    distinguishing them. This is the defect the slot exists to prevent."""
    qi = QuestionInterpretation(question="balance by region for joint borrowers")
    qi.dimensions = [
        DimensionClaim(state=FILLED, raw_text="region", role=GROUPING),
        DimensionClaim(state=FILLED, raw_text="joint borrowers", role=GROUPING),
    ]
    assert len(qi.dimensions_with_role(GROUPING)) == 2
    assert qi.dimensions_with_role(FILTER) == []


def test_an_unknown_role_is_rejected():
    with pytest.raises(ValueError):
        DimensionClaim(state=FILLED, raw_text="region", role="axis")


def test_role_defaults_to_unresolved_not_grouping():
    """Defaulting to `grouping` would silently reproduce KIND_GROUPING."""
    assert DimensionClaim().role == UNRESOLVED_ROLE


# --------------------------------------------------------------------------- #
# target — the slot whose absence narrows the book
# --------------------------------------------------------------------------- #
def test_a_target_is_not_a_filter():
    """"when do we reach £100m" — the value is solved FOR, not narrowed BY.
    With no target slot it lands in filters and cuts the book to loans of
    £100m or more."""
    qi = QuestionInterpretation(question="when do we reach £100m")
    qi.target = TargetClaim(state=FILLED, raw_text="£100m", value="100000000",
                            target_source=STATED)
    assert qi.filters == []
    assert qi.target.state == FILLED


def test_the_target_slot_can_fail():
    """Route the same value into filters instead and the population narrows.
    Demonstrated, so the guarantee is not merely asserted."""
    qi = QuestionInterpretation(question="when do we reach £100m")
    qi.filters = [FilterClaim(state=FILLED, raw_text="£100m",
                              operator="ge", value="100000000")]
    assert qi.target.state == EMPTY
    assert qi.filters and qi.filters[0].operator == "ge"


def test_a_configured_target_is_filled_but_unresolvable_without_a_plan():
    """"are we on target" states no threshold. It is not empty — the question
    asked for one — and it is not filled."""
    t = TargetClaim(state=UNRESOLVABLE, raw_text="on target",
                    target_source=CONFIGURED, reason="no plan configured")
    assert t.state == UNRESOLVABLE and t.value is None


def test_an_unknown_target_source_is_rejected():
    with pytest.raises(ValueError):
        TargetClaim(state=FILLED, raw_text="£100m", target_source="guessed")


def test_target_source_does_not_shadow_the_interpreter_source():
    """Two different meanings cannot share one field."""
    t = TargetClaim(state=FILLED, raw_text="£100m", target_source=STATED,
                    source="llm_query_parser._forecast_target_value")
    assert t.target_source == STATED
    assert t.source == "llm_query_parser._forecast_target_value"


# --------------------------------------------------------------------------- #
# time — grain and window are different claims about the same dimension
# --------------------------------------------------------------------------- #
def test_grain_and_window_are_separate_slots():
    """"balance by month over the last 6 months" states an AXIS and a WINDOW.
    One slot cannot hold both."""
    time = TimeClaim(
        requested_grain=Slot(state=FILLED, raw_text="by month"),
        trend_window=Slot(state=FILLED, raw_text="over the last 6 months"),
        grain="month")
    assert time.requested_grain.state == FILLED
    assert time.trend_window.state == FILLED
    assert time.grain == "month"


def test_the_grain_window_split_can_fail():
    """Collapse them and the question is indistinguishable from one that named
    only a window — which is what the release candidate does today."""
    time = TimeClaim(trend_window=Slot(state=FILLED, raw_text="over the last 6 months"))
    assert time.requested_grain.state == EMPTY
    assert time.grain is None


def test_an_unknown_grain_is_rejected():
    with pytest.raises(ValueError):
        TimeClaim(grain="fortnight")


# --------------------------------------------------------------------------- #
# operation
# --------------------------------------------------------------------------- #
def test_an_unknown_operation_type_is_rejected():
    with pytest.raises(ValueError):
        OperationClaim(state=FILLED, raw_text="how many", type="tally")


def test_count_and_amount_are_distinct_types():
    """The defect this vocabulary exists for: "how many loans have a balance
    above £250k" answered with a balance."""
    assert COUNT in OPERATION_TYPES and AMOUNT in OPERATION_TYPES
    assert COUNT != AMOUNT


# --------------------------------------------------------------------------- #
# residue and unresolvable reporting
# --------------------------------------------------------------------------- #
def test_residue_is_not_the_subject():
    qi = QuestionInterpretation(question="how much is in the good book")
    qi.subject = SubjectClaim(state=FILLED, raw_text="how much",
                              candidate_concept="current_outstanding_balance")
    qi.residue = ["the good book"]
    assert "good book" not in (qi.subject.raw_text or "")
    assert qi.residue == ["the good book"]


def test_unresolvable_slots_reports_every_slot_kind():
    qi = QuestionInterpretation(question="x")
    qi.target = TargetClaim(state=UNRESOLVABLE, raw_text="on target")
    qi.dimensions = [DimensionClaim(state=UNRESOLVABLE, raw_text="stage")]
    qi.filters = [FilterClaim(state=UNRESOLVABLE, raw_text="large loans")]
    qi.population = [PopulationClaim(state=UNRESOLVABLE, raw_text="weaker cohorts")]
    qi.time = TimeClaim(requested_grain=Slot(state=UNRESOLVABLE, raw_text="by week"))
    names = [n for n, _ in qi.unresolvable_slots()]
    assert names == ["target", "time.requested_grain", "dimension", "filter",
                     "population"]


def test_the_unresolvable_report_can_fail():
    """An unresolvable slot that goes unreported is the silent omission the
    state exists to prevent."""
    qi = QuestionInterpretation(question="x")
    qi.filters = [FilterClaim(state=FILLED, raw_text="LTV above 50%")]
    assert qi.unresolvable_slots() == []


# --------------------------------------------------------------------------- #
# spans
# --------------------------------------------------------------------------- #
def test_a_span_locates_its_words_in_the_question():
    q = "balance by region"
    s = Span(11, 17)
    assert s.text_of(q) == "region"


def test_spans_detect_overlap():
    """Precedence between two competing claims is decidable from offsets.
    It is not decidable from a facet, which carries a rendered label only."""
    assert Span(0, 10).overlaps(Span(5, 15))
    assert not Span(0, 5).overlaps(Span(5, 10))


def test_an_invalid_span_is_rejected():
    with pytest.raises(ValueError):
        Span(10, 3)


# --------------------------------------------------------------------------- #
# the object as a whole
# --------------------------------------------------------------------------- #
def test_an_empty_interpretation_is_valid_and_says_nothing():
    """Empty is normal. The object must not imply a claim it does not hold."""
    qi = QuestionInterpretation(question="what is the funded balance?")
    d = qi.as_dict()
    assert d["operation"]["state"] == EMPTY
    assert d["dimensions"] == [] and d["filters"] == []
    assert d["target"]["state"] == EMPTY
    assert qi.unresolvable_slots() == []


def test_as_dict_round_trips_every_slot():
    qi = QuestionInterpretation(question="q")
    keys = set(qi.as_dict())
    # Every addition here is PRE-REGISTERED MOVEMENT, not drift, and this test
    # is the guard that makes the distinction enforceable — a claim added
    # without a decision fails here.
    #
    #   source_scope  PHASE 1A. The source-portfolio lens, carried from
    #                 `mi_agent.portfolio_lens`.
    #   dataset       TARGET-STATE CLOSURE. Which governed tape the answer is
    #                 built from, carried from `mi_agent_api.workspace`. A
    #                 DIFFERENT AXIS from source_scope: a question picks a tape
    #                 and, within it, a portfolio scope. Added because
    #                 `chat_routing._dataset_for` was re-deriving the same
    #                 decision over a wider vocabulary — a duplicate its own
    #                 docstring names "THE SECOND OWNER".
    #   row_predicates
    #                 C6 FILTER BINDING, step 1. The RESOLVED row predicates —
    #                 governed field + operator + value — as
    #                 `llm_query_parser._filter_field_of` bound them and
    #                 `population.material_predicates` normalised them.
    #
    #                 A THIRD channel, and it had to be. `filters` carries what
    #                 the question SAID and deliberately carries no field:
    #                 "identifying that a clause exists is a different job from
    #                 resolving what it binds". `population` names a population
    #                 by INTENT ("the back book") and is deliberately kept apart
    #                 from its resolution. Putting a resolved field on either
    #                 would collapse a distinction each of them exists to hold,
    #                 so the decision was NOT to extend FilterClaim.
    #
    #                 Added because a compositional plan needs `field + operator
    #                 + value` and the projection was writing the field into a
    #                 provenance STRING — source="parser.filters[…]" — where
    #                 nothing downstream could read it without parsing prose or
    #                 re-reading the English.
    assert keys == {"question", "operation", "subject", "dimensions", "filters",
                    "time", "target", "population", "row_predicates",
                    "source_scope", "dataset", "residue", "notes"}


# --------------------------------------------------------------------------- #
# The five corrections earned from the Stage 1 corpus.
#
# Counts below are from the THREE REAL SURFACES only — the calibration bank,
# the ERE golden library and the 44-variation robustness bank, 690 questions.
# The generated harness is excluded: its phrasings are machine-generated from
# registry names, so it is a valid invariant check and no evidence at all about
# client wording, and it must not shape the schema.
# --------------------------------------------------------------------------- #
from question_interpretation.schema import (  # noqa: E402
    BOUND, CLAIM_CONTENTS, FIELD, ROLE_UNATTRIBUTED,
    UNSUPPLIED_TARGET_SOURCES, WORDING,
)


# -- Correction 1: a filter claim says which half of the clause it carries --
def test_a_filter_claim_declares_what_it_carries():
    """Driven by: 76 of 690 questions where ONE clause is read twice, the facet
    layer supplying the wording and the parser the field and bound."""
    facet_half = FilterClaim(state=FILLED, raw_text="LTV over 50",
                             provides=(WORDING,))
    parser_half = FilterClaim(state=FILLED, operator="gt", value="50.0",
                              provides=(FIELD, BOUND))
    assert facet_half.is_half_claim and parser_half.is_half_claim
    assert facet_half.clause_id is None and parser_half.clause_id is None


def test_a_complete_filter_claim_is_not_a_half_claim():
    whole = FilterClaim(state=FILLED, raw_text="LTV over 50", operator="gt",
                        value="50.0", provides=(WORDING, FIELD, BOUND))
    assert not whole.is_half_claim


def test_the_half_claim_check_can_fail():
    """Before the correction a half was inferred from which attributes were
    None — indistinguishable from an interpreter that looked and found
    nothing. Declaring nothing must not read as a complete claim."""
    silent = FilterClaim(state=FILLED, raw_text="LTV over 50", provides=())
    assert not silent.is_half_claim  # declares nothing, so claims nothing


def test_a_filter_claim_declares_nothing_by_default():
    """The DEFAULT must be empty, not complete.

    Added because the mutation check caught a real gap: every other test passed
    `provides` explicitly, so a default of (WORDING, BOUND, FIELD) went
    unnoticed. A claim that has declared nothing must not read as complete —
    that is the whole point of the correction.
    """
    bare = FilterClaim(state=FILLED)
    assert bare.provides == ()
    assert not bare.is_half_claim
    assert bare.as_dict()["provides"] == []


def test_unknown_claim_contents_are_rejected():
    with pytest.raises(ValueError):
        FilterClaim(state=FILLED, provides=("field", "vibes"))


def test_clause_id_is_none_until_an_interpreter_supplies_a_basis():
    """Stage 1 established no interpreter emits anything that makes the join
    sound. An unjoined pair is reported unjoined, never guessed."""
    assert FilterClaim(state=FILLED, provides=(WORDING,)).clause_id is None
    assert set(CLAIM_CONTENTS) == {WORDING, BOUND, FIELD}


# -- Correction 2: span absence is explicit ---------------------------------
def test_a_claim_without_a_span_is_valid_and_says_so():
    """Driven by: 170 dimension and filter claims across 690 questions have no
    recoverable span, because the interpreter emitted a field key or a rendered
    label rather than the words."""
    c = DimensionClaim(state=FILLED, raw_text=None, role=GROUPING,
                       candidate_concept="collateral_geography")
    assert c.has_span is False
    assert c.as_dict()["has_span"] is False


def test_has_span_can_fail():
    c = DimensionClaim(state=FILLED, raw_text="region", role=GROUPING,
                       span=Span(11, 17))
    assert c.has_span is True


# -- Correction 3: `coverage` removed ---------------------------------------
def test_coverage_is_not_an_operation_type():
    """Driven by: produced 0 times in 690 questions, and no interpreter
    supplies it. An unsupplied member invites population by intuition."""
    assert "coverage" not in OPERATION_TYPES


def test_the_coverage_removal_can_fail():
    with pytest.raises(ValueError):
        OperationClaim(state=FILLED, raw_text="what share", type="coverage")


# -- Correction 4: `configured` kept but unsupplied --------------------------
def test_the_configured_target_source_is_recorded_as_unsupplied():
    """Driven by: the wording appears in 0 of 690 questions, and the only thing
    producing it was a regex the projection owned — a reading it invented."""
    assert CONFIGURED in UNSUPPLIED_TARGET_SOURCES
    assert STATED not in UNSUPPLIED_TARGET_SOURCES


def test_a_configured_target_is_still_expressible_when_a_source_appears():
    """Kept in the vocabulary because it is a stated requirement, not because
    the corpus earned it. It must remain constructible."""
    t = TargetClaim(state=UNRESOLVABLE, raw_text="on target",
                    target_source=CONFIGURED, reason="no plan configured")
    assert t.target_source == CONFIGURED


# -- Correction 5: an unresolved role must say why --------------------------
def test_an_unresolved_role_carries_a_reason():
    """Driven by: 55 of 690 questions name a dimension no source assigns a role
    to."""
    c = DimensionClaim(state=FILLED, raw_text="broker", role=UNRESOLVED_ROLE,
                       reason=ROLE_UNATTRIBUTED)
    assert c.role == UNRESOLVED_ROLE and c.reason == ROLE_UNATTRIBUTED


def test_no_conflicted_role_was_invented():
    """Stage 1 proposed splitting `unresolved` into 'no opinion' and 'sources
    disagree'. ZERO of 690 questions show a dimension two sources put in
    different roles, so the second value would have been intuition. It is
    deliberately absent, and this test is what stops it reappearing."""
    assert set(DIMENSION_ROLES) == {GROUPING, FILTER, UNRESOLVED_ROLE}
    assert "conflicted" not in DIMENSION_ROLES
