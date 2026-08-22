"""A cadence belongs to a stream, not to every stream.

"They need a weekly pipeline and monthly MI" is two answers. The interpretation
carried ONE ``delivery.cadence`` and the plan ONE ``cadence``, so both source
registrations took whichever was read last — a client was shown a weekly funded
book they had never asked for, and the pack put it in front of them for
confirmation as though Trakt believed it.

This is a domain-model fix rather than a reading one: the model was the wrong
SHAPE. ``streams`` was already a list; cadence was not.
"""

from __future__ import annotations

import pytest

from operations_control.occ_agent.interpretation import (
    DeterministicInterpreter, stream_cadences,
)
from operations_control.onboarding.catalogue import catalogue

from .conftest import ACTOR, TENANT_A

#: The instruction from the production practice case that exposed this.
ERM = ("Onboard ERM Capital Mgt. It is a UK equity release lender. The need "
       "weekly pipeline, monthly MI and Annex 2 regime reporting.")


@pytest.fixture()
def interpreter():
    return DeterministicInterpreter(cat=catalogue())


# --------------------------------------------------------------------------- #
# Reading the pairing
# --------------------------------------------------------------------------- #

def test_a_stream_takes_the_cadence_beside_it():
    paired, blanket = stream_cadences(ERM, catalogue())
    assert paired == {"pipeline": "weekly"}
    # "monthly MI" names no stream, so it answers for everything else.
    assert blanket == "monthly"


def test_two_pairings_in_one_sentence_are_both_read():
    """An order-based rule matched whichever pattern it tried first and
    silently dropped the other."""
    paired, blanket = stream_cadences(
        "They send a funded book monthly and a pipeline weekly.", catalogue())
    assert paired == {"funded": "monthly", "pipeline": "weekly"}
    assert blanket == ""


def test_a_stream_named_after_its_cadence_is_read_too():
    paired, _blanket = stream_cadences("The pipeline is delivered weekly.",
                                       catalogue())
    assert paired == {"pipeline": "weekly"}


def test_a_cadence_with_no_stream_stays_blanket():
    paired, blanket = stream_cadences("They deliver monthly.", catalogue())
    assert paired == {}
    assert blanket == "monthly"


def test_a_comma_separates_two_statements():
    """Without this, "pipeline, monthly" pairs across the comma."""
    paired, blanket = stream_cadences("weekly pipeline, monthly MI",
                                      catalogue())
    assert paired == {"pipeline": "weekly"}
    assert blanket == "monthly"


def test_a_product_never_names_a_stream():
    """"MI" is a product. Inferring the funded book from it would attach a
    cadence to a registration the client never mentioned."""
    paired, blanket = stream_cadences("monthly MI", catalogue())
    assert paired == {}
    assert blanket == "monthly"


def test_the_vocabulary_is_the_catalogues_own():
    """No cadence or dataset word is listed in the reader."""
    import inspect

    from operations_control.occ_agent import interpretation as module
    source = inspect.getsource(module.stream_cadences)
    for word in ("weekly", "monthly", "daily", "funded", "pipeline"):
        assert f'"{word}"' not in source, word


# --------------------------------------------------------------------------- #
# Carrying it through the interpretation
# --------------------------------------------------------------------------- #

def test_the_paired_cadence_does_not_also_become_the_blanket(interpreter):
    read = interpreter.interpret_instruction(ERM)
    assert read.stream_delivery == {"pipeline": {"cadence": "weekly"}}
    assert read.delivery.get("cadence") == "monthly"
    assert "pipeline" in read.streams


def test_an_instruction_with_no_stream_is_unchanged(interpreter):
    read = interpreter.interpret_instruction(
        "Onboard Northstar Lending. UK equity release. Monthly portfolio MI. "
        "Portfolio id direct_101.")
    assert read.stream_delivery == {}
    assert read.delivery.get("cadence") == "monthly"


def test_an_unknown_stream_is_refused(interpreter):
    from operations_control.occ_agent.interpretation import (
        Interpretation, InterpretationError,
    )
    bad = Interpretation(stream_delivery={"invented": {"cadence": "monthly"}})
    with pytest.raises(InterpretationError):
        bad.validate(catalogue())


def test_a_stream_delivery_key_must_be_a_catalogue_field(interpreter):
    from operations_control.occ_agent.interpretation import (
        Interpretation, InterpretationError,
    )
    bad = Interpretation(stream_delivery={"pipeline": {"invented": "x"}})
    with pytest.raises(InterpretationError):
        bad.validate(catalogue())


# --------------------------------------------------------------------------- #
# What actually lands on the case
# --------------------------------------------------------------------------- #

def test_each_registration_keeps_its_own_cadence(service):
    """The defect, end to end: two registrations, two cadences."""
    case = service.create_case(tenant=TENANT_A, initiating_user=ACTOR,
                               instruction=ERM)
    by_dataset = {str(s.get("dataset")): str(s.get("cadence") or "")
                  for s in case.case.items("sources")}
    assert by_dataset.get("pipeline") == "weekly"
    assert by_dataset.get("funded") == "monthly", by_dataset


def test_a_blanket_cadence_still_reaches_every_registration(service):
    case = service.create_case(
        tenant=TENANT_A, initiating_user=ACTOR,
        instruction="Onboard Harbourpoint. UK equity release. They send a "
                    "funded book and a pipeline. Everything monthly.")
    cadences = {str(s.get("cadence") or "")
                for s in case.case.items("sources")}
    assert cadences == {"monthly"}


def test_the_client_is_told_which_book_each_cadence_belongs_to(service):
    """Two rows that disagree are worse than two that repeat, unless the
    client can tell which is which."""
    case = service.create_case(tenant=TENANT_A, initiating_user=ACTOR,
                               instruction=ERM)
    document = service.build_pack(case).document()
    rows = [line for line in document.split("\n")
            if line.startswith("- **Expected cadence**")]
    assert len(rows) == 2
    assert all(" — " in row for row in rows), rows
    assert any("funded" in row and "Monthly" in row for row in rows), rows
    assert any("pipeline" in row and "Weekly" in row for row in rows), rows


def test_the_plan_discloses_both_cadences(service):
    """A human confirming the change sees what it will actually do."""
    case = service.create_case(tenant=TENANT_A, initiating_user=ACTOR,
                               instruction="Onboard Aldermere. UK equity "
                                           "release.")
    plan = service.plan_instruction(
        case,
        instruction="They send a funded book monthly and a pipeline weekly.")
    assert plan.stream_cadence == {"funded": "monthly", "pipeline": "weekly"}
    summary = plan.summary()
    assert "funded" in summary and "pipeline" in summary
