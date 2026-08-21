"""What Trakt is sure of, and what it merely read.

The reader used to have two outcomes: read nothing (reported as
``unrecognised``, and put in front of the operator) and read something
(applied). It had no third for *read something, but guessing*.

``Interpretation.confidence`` existed and was populated, but every cued match
was graded 1.0 — including the bare topic cue that named a client "needs weekly
pipeline". Confidence was recorded and never earned, so a confidently wrong
read was indistinguishable from a correct one.

The grade now reflects the EVIDENCE:

    CUED       the catalogue's own pattern or closed vocabulary proved the
               value — an LEI, an email, an option
    INFERRED   the cue named the field, but the value's extent is a rule
               rather than a proof: free text runs until something ends it
    SHAPE_ONLY the value was recognised with no cue at all

Only the first is certain, and anything below certainty reaches a human.
"""

from __future__ import annotations

import pytest

from operations_control.occ_agent import extraction as _extraction
from operations_control.occ_agent import planning as _planning
from operations_control.occ_agent.interpretation import DeterministicInterpreter
from operations_control.onboarding.catalogue import catalogue

from .conftest import ACTOR, TENANT_A


def read(text: str):
    return {e.ref.path: e for e in _extraction.read(text, cat=catalogue()).found}


# --------------------------------------------------------------------------- #
# The grades
# --------------------------------------------------------------------------- #

def test_a_value_its_own_pattern_proves_is_certain():
    """An LEI cannot be anything but an LEI."""
    found = read("The LEI is 894500SYNTHETIC00042.")
    assert found["entities.lei"].confidence == _extraction.CUED


def test_a_value_from_a_closed_vocabulary_is_certain():
    """"UK equity release" names the asset class, and only that option."""
    found = read("Onboard Northstar Lending. UK equity release.")
    assert found["portfolios.asset_class"].confidence == _extraction.CUED


def test_free_text_bound_by_a_cue_is_not_certain():
    """The FIELD is certain; where the value ends is a judgement. This is the
    class "the client needs weekly pipeline" fell into."""
    found = read("The client is a mortgage lender called ERM Capital.")
    name = found["client.client_name"]
    assert name.value == "ERM Capital"
    assert name.confidence == _extraction.INFERRED
    assert name.confidence < _planning.CERTAIN


def test_a_value_with_no_cue_at_all_is_least_certain():
    found = read("Onboard Northstar Lending. UK equity release.")
    assert found["client.jurisdiction"].confidence == _extraction.SHAPE_ONLY


def test_the_grades_are_ordered():
    assert (_extraction.SHAPE_ONLY < _extraction.INFERRED
            < _extraction.CUED == _planning.CERTAIN)


# --------------------------------------------------------------------------- #
# What the grade causes
# --------------------------------------------------------------------------- #

def test_an_uncertain_read_is_carried_on_the_interpretation():
    read_back = DeterministicInterpreter(cat=catalogue()).interpret_instruction(
        "The client is a mortgage lender called ERM Capital.")
    assert read_back.confidence.get("client.client_name") == \
        _extraction.INFERRED
    # …and its provenance says Trakt worked it out, not that a human stated it.
    assert read_back.provenance.get("client.client_name") != "human_supplied"


def test_an_uncertain_change_is_material(service, ):
    """Material means a human sees it before it is written."""
    case = service.create_case(tenant=TENANT_A, initiating_user=ACTOR,
                               instruction="Onboard Aldermere Finance. UK "
                                           "equity release.")
    plan = service.plan_instruction(
        case, instruction="The client is a lender called Aldermere Holdings.")
    assert plan.uncertain, "a free-text read must be reported as uncertain"
    assert plan.material


def test_a_certain_change_into_a_blank_is_not_material(service):
    """Grading must not turn every ordinary instruction into a confirmation."""
    case = service.create_case(tenant=TENANT_A, initiating_user=ACTOR,
                               instruction="Onboard Brackenfield. UK equity "
                                           "release.")
    plan = service.plan_instruction(
        case, instruction="They need ESMA Annex 2 reporting.")
    assert not plan.uncertain, [c.to_dict() for c in plan.uncertain]
    assert not plan.material


def test_planning_has_its_own_older_uncertainty_and_it_still_works(service):
    """Not everything below CERTAIN comes from the reader.

    ``plan_changes`` already graded a PROPOSED UPDATE to a matched repeatable
    item at 0.5 — "I think this answer belongs to that entity" is its own
    judgement, made after the reading and independent of it. Both kinds now
    flow through one threshold, and this pins the older one so a change to the
    reader's grades cannot quietly disable it.
    """
    case = service.create_case(tenant=TENANT_A, initiating_user=ACTOR,
                               instruction="Onboard Brackenfield. UK equity "
                                           "release.")
    plan = service.plan_instruction(
        case, instruction="The LEI is 894500SYNTHETIC00042.")
    lei = [c for c in plan.uncertain if c.field == "lei"]
    assert lei, "an LEI proposed onto an existing entity is a judgement"
    assert lei[0].confidence == 0.5
    assert plan.material


def test_the_plan_separates_uncertain_from_understood(service):
    case = service.create_case(tenant=TENANT_A, initiating_user=ACTOR,
                               instruction="Onboard Kestrel Mutual. UK equity "
                                           "release.")
    plan = service.plan_instruction(
        case, instruction="The client is a lender called Kestrel Holdings.")
    disclosure = plan.disclosure()
    assert disclosure["uncertain"], disclosure
    assert any("not certain" in line for line in disclosure["uncertain"])


def test_the_operator_is_told_which_reads_were_guesses(service):
    """An operator scanning a wall of "Recorded:" lines cannot tell which one
    Trakt guessed at — and the guess is the line worth reading."""
    case = service.create_case(tenant=TENANT_A, initiating_user=ACTOR,
                               instruction="Onboard Silverbeck. UK equity "
                                           "release.")
    plan = service.plan_instruction(
        case, instruction="The client is a lender called Silverbeck Group.")
    told = service.describe_plan(case, plan)
    assert "Read, but not certain" in told, told


# --------------------------------------------------------------------------- #
# What the grade must NOT cause
# --------------------------------------------------------------------------- #

def test_uncertainty_does_not_make_an_instruction_incomplete(service):
    """"Read, but unsure" and "could not read at all" are different claims.

    Folding the first into the second would make every ordinary instruction
    look like a failure, and `PartiallyUnderstood` would fire on all of them.
    """
    case = service.create_case(tenant=TENANT_A, initiating_user=ACTOR,
                               instruction="Onboard Thornbury. UK equity "
                                           "release.")
    plan = service.plan_instruction(
        case, instruction="The client is a lender called Thornbury Lending.")
    assert plan.uncertain
    assert plan.complete
    assert not plan.unrecognised


def test_an_opening_instruction_still_applies(service):
    """The operator has just typed it; it is not put back to them."""
    case = service.create_case(
        tenant=TENANT_A, initiating_user=ACTOR,
        instruction="The client is a mortgage lender called ERM Capital. They "
                    "need monthly MI.")
    assert (case.case.answers.get("client") or {}).get("client_name") == \
        "ERM Capital"
