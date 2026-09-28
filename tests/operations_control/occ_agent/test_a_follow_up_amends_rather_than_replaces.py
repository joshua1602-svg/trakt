"""What an operator says in conversation, read as what they meant.

Three defects, one cause: the reader took a sentence apart and threw away the
words that carried its meaning, then wrote what was left as if it were the
whole instruction.

* **"They also need ESMA Annex 2."** The clause splitter cut at "also" and
  discarded it, so the reader saw "need ESMA Annex 2" — not an addition, a
  complete product list. MI was dropped from a client who still needed it.
* **"Remove ESMA Annex 2."** Read as a list of one product, and written as
  the client's products. The exact inverse of the instruction, applied
  silently, and confirmed back as though it were right.
* **A schedule of concentration limits pasted as three sentences.** One
  sentence stored, with the colon still attached to the front of it. The other
  two were not reported as missed either — each was a clause the reader simply
  never connected to the answer it belonged to.

The last one matters most, because prose is the shape a client's own answer
arrives in, and losing two thirds of a governed limit schedule without saying
so is not a reading failure, it is a data-integrity failure.
"""

from __future__ import annotations

import pytest

from operations_control.occ_agent.interpretation import DeterministicInterpreter

from .conftest import ACTOR, TENANT_A

OPENING = ("Onboard Northstar Lending. UK equity release. Monthly portfolio "
           "MI. Portfolio id direct_101.")

LIMITS = ("The concentration tests are: Maximum 10% of the portfolio to any "
          "one postcode district. Maximum LTV 55% at origination. No more "
          "than 5% of loans above GBP 750,000.")


@pytest.fixture()
def opened(service):
    return service.create_case(tenant=TENANT_A, initiating_user=ACTOR,
                               instruction=OPENING)


def products(case) -> list:
    return list((case.case.answers.get("reporting") or {}).get("products")
                or [])


# --------------------------------------------------------------------------- #
# Adding a product keeps the ones already chosen
# --------------------------------------------------------------------------- #

def test_the_opening_instruction_selects_mi(opened):
    """The starting point every case below is amended from."""
    assert products(opened) == ["mi"]


def test_also_adds_a_product_rather_than_replacing_the_list(service, opened):
    turn = service.instruct(opened, text="They also need ESMA Annex 2.",
                            actor=ACTOR, confirm=True)
    assert products(turn.case) == ["mi", "esma_annex2"], \
        "'also' replaced the client's products instead of adding to them"


def test_adding_twice_keeps_everything_added(service, opened):
    case = opened
    for product in ("ESMA Annex 2", "investor reporting"):
        case = service.instruct(case, text=f"They also need {product}.",
                                actor=ACTOR, confirm=True).case
    assert products(case) == ["mi", "esma_annex2", "investor_reporting"]


def test_in_addition_adds_too(service, opened):
    turn = service.instruct(opened, text="In addition they need Annex 2.",
                            actor=ACTOR, confirm=True)
    assert products(turn.case) == ["mi", "esma_annex2"]


# --------------------------------------------------------------------------- #
# Removing a product removes it
# --------------------------------------------------------------------------- #

def test_remove_takes_a_product_away(service, opened):
    case = service.instruct(opened, text="They also need ESMA Annex 2.",
                            actor=ACTOR, confirm=True).case
    assert products(case) == ["mi", "esma_annex2"]

    turn = service.instruct(case, text="Remove ESMA Annex 2.",
                            actor=ACTOR, confirm=True)
    assert products(turn.case) == ["mi"], \
        "'remove' set the products TO the one being removed"


def test_drop_takes_a_product_away(service, opened):
    case = service.instruct(opened, text="They also need investor reporting.",
                            actor=ACTOR, confirm=True).case
    turn = service.instruct(case, text="Drop investor reporting.",
                            actor=ACTOR, confirm=True)
    assert products(turn.case) == ["mi"]


def test_removing_what_was_never_selected_changes_nothing(service, opened):
    turn = service.instruct(opened, text="Remove static pools.",
                            actor=ACTOR, confirm=True)
    assert products(turn.case) == ["mi"]


def test_one_sentence_states_two_products_and_withdraws_a_third(service,
                                                                opened):
    """A statement and an exception in one breath."""
    turn = service.instruct(
        opened,
        text="They need MI and investor reporting but not static pools.",
        actor=ACTOR, confirm=True)
    assert products(turn.case) == ["investor_reporting", "mi"]


def test_an_opening_list_still_replaces(service, opened):
    """Nothing here turns a plain statement into an amendment."""
    turn = service.instruct(opened, text="They need investor reporting.",
                            actor=ACTOR, confirm=True)
    assert products(turn.case) == ["investor_reporting"]


def test_adding_several_products_at_once_adds_all_of_them(service, opened):
    turn = service.instruct(
        opened, text="They also need ESMA Annex 2 and investor reporting.",
        actor=ACTOR, confirm=True)
    assert set(products(turn.case)) == {"mi", "esma_annex2",
                                        "investor_reporting"}


def test_a_removal_among_several_values_of_one_field_is_not_guessed(service,
                                                                    opened):
    """The reader knows where the FIRST of several values sits, not the rest,
    so it cannot tell which of them a removal word reached. "Originator, not
    the reporting entity" must not withdraw the originator as well — so
    several values stay a statement, exactly as they were."""
    turn = service.instruct(
        opened,
        text="ERE Funding Limited is the originator, not the reporting entity.",
        actor=ACTOR, confirm=True)
    roles = [set(e.get("roles") or [])
             for e in turn.case.case.items("entities")]
    assert any("originator" in r for r in roles), \
        "the originator was withdrawn by a word that was not about it"


# --------------------------------------------------------------------------- #
# Prose survives the reading
# --------------------------------------------------------------------------- #

def test_every_sentence_of_a_pasted_limit_schedule_is_kept():
    """The one that lost two of three limits without reporting either."""
    read = DeterministicInterpreter().interpret_instruction(LIMITS)
    stored = read.steps["risk_limits"]["concentration_tests"]

    assert "postcode district" in stored
    assert "Maximum LTV 55%" in stored, "the second limit was dropped"
    assert "5% of loans above GBP 750,000" in stored, \
        "the third limit was dropped"


def test_the_join_is_not_part_of_the_answer():
    read = DeterministicInterpreter().interpret_instruction(LIMITS)
    stored = read.steps["risk_limits"]["concentration_tests"]
    assert not stored.startswith(":"), "the colon was stored as text"
    assert stored.startswith("Maximum 10%")


def test_a_limit_that_names_a_currency_is_not_read_as_the_reporting_currency():
    """"GBP 750,000" inside a limit is a limit, not an answer to another
    question — and treating it as one is what cut the schedule short."""
    read = DeterministicInterpreter().interpret_instruction(LIMITS)
    assert "client" not in read.steps


def test_prose_stops_where_another_question_starts():
    """Prose runs on, but not over a sentence that names something else."""
    read = DeterministicInterpreter().interpret_instruction(
        "The concentration tests are: Maximum LTV 55% at origination. "
        "The reporting contact is Jane Doe.")
    assert read.steps["risk_limits"]["concentration_tests"] == \
        "Maximum LTV 55% at origination"
    assert read.steps["contacts"]["reporting_contact_name"] == "Jane Doe"


def test_a_comma_does_not_end_a_prose_answer():
    read = DeterministicInterpreter().interpret_instruction(
        "The concentration tests are no more than 10% in any one postcode "
        "district, and no more than 5% above GBP 750,000.")
    stored = read.steps["risk_limits"]["concentration_tests"]
    assert "5% above GBP 750,000" in stored, "the answer stopped at the comma"


def test_nothing_read_into_prose_is_reported_as_missed():
    read = DeterministicInterpreter().interpret_instruction(LIMITS)
    assert read.unrecognised == []
