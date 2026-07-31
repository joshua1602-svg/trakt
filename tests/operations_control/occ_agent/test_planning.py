"""Planning: what an instruction would do, and what it may never do silently.

Two properties this module exists to prove.

**A repeatable item is never silently replaced.** The bug this replaced renamed
a client's first portfolio when an operator said "they also have direct_102",
and moved its delivery registration with it. Sequentially adding two portfolios
is therefore tested directly, from both directions.

**Nothing is applied that was not understood.** A plan reports four
populations — understood, proposed, questions, unrecognised — and a plan that
carries either of the last two is refused unless the human confirms the
disclosed remainder.
"""

from __future__ import annotations

import pytest

from operations_control.occ_agent import planning as _planning
from operations_control.occ_agent.service import PartiallyUnderstood

from .conftest import ACTOR, TENANT_A

OPENING = ("Onboard Northstar Lending. UK equity release. Monthly portfolio "
           "MI. Portfolio id direct_101.")


@pytest.fixture()
def opened(service):
    return service.create_case(tenant=TENANT_A, initiating_user=ACTOR,
                               instruction=OPENING)


# --------------------------------------------------------------------------- #
# A second portfolio is a second portfolio
# --------------------------------------------------------------------------- #

def test_a_second_portfolio_is_added_not_merged_over_the_first(service, opened):
    """The regression. Two identifiers means two books."""
    turn = service.instruct(
        opened, text="They also have portfolio id direct_102, equity release.",
        actor=ACTOR, confirm=True)
    portfolios = turn.case.case.answers["portfolios"]
    ids = [p["portfolio_id"] for p in portfolios]
    assert ids == ["direct_101", "direct_102"], \
        "the second portfolio replaced the first"


def test_adding_two_portfolios_in_sequence_keeps_all_three(service, opened):
    case = opened
    for identifier in ("direct_102", "direct_103"):
        turn = service.instruct(
            case, text=f"They also have portfolio id {identifier}, equity "
                       f"release.", actor=ACTOR, confirm=True)
        case = turn.case
    ids = [p["portfolio_id"] for p in case.case.answers["portfolios"]]
    assert ids == ["direct_101", "direct_102", "direct_103"]


def test_the_first_portfolios_delivery_survives_a_second_being_added(service,
                                                                     opened):
    """A book's delivery registration must not move with a rename that never
    happened."""
    before = {s.get("portfolio_id") for s in opened.case.items("sources")}
    assert "direct_101" in before
    turn = service.instruct(
        opened, text="They also have portfolio id direct_102, equity release.",
        actor=ACTOR, confirm=True)
    after = {s.get("portfolio_id") for s in turn.case.case.items("sources")}
    assert "direct_101" in after, "the first book's delivery was moved"
    assert "direct_102" in after, "the second book got no delivery"


def test_naming_the_same_identifier_updates_that_book(service, opened):
    turn = service.instruct(
        opened, text="Portfolio id direct_101 is an acquired book.",
        actor=ACTOR, confirm=True)
    portfolios = turn.case.case.answers["portfolios"]
    assert len(portfolios) == 1
    assert portfolios[0]["portfolio_type"] == "acquired"


def test_a_change_with_no_identifier_and_several_books_asks_which(service,
                                                                  opened):
    """With more than one candidate the agent asks rather than picking."""
    case = service.instruct(
        opened, text="They also have portfolio id direct_102, equity release.",
        actor=ACTOR, confirm=True).case
    plan = service.plan_instruction(case, instruction="The book was acquired.")
    assert plan.questions, "the agent picked a book instead of asking"
    assert plan.complete is False
    assert not plan.steps.get("portfolios"), "something was planned anyway"


def test_a_change_with_no_identifier_and_one_book_is_only_proposed(service,
                                                                   opened):
    plan = service.plan_instruction(opened,
                                    instruction="The book was acquired.")
    changes = [c for c in plan.understood if c.field == "portfolio_type"]
    assert changes
    assert changes[0].confidence == 0.5, "an assumed target was taken as certain"
    assert plan.material is True


def test_a_rename_is_never_produced_by_a_new_identifier():
    """The rule, at the unit level, with no service in the way."""
    from operations_control.onboarding.case import OnboardingCase
    from operations_control.onboarding.catalogue import catalogue

    cat = catalogue()
    case = OnboardingCase(case_id="ONB-2026-0001")
    case.answers["portfolios"] = [{"portfolio_id": "direct_101",
                                   "asset_class": "equity_release"}]
    plan = _planning.plan_changes(
        case, cat, steps={"portfolios": {"portfolios": [
            {"portfolio_id": "direct_102"}]}})
    merged = plan.steps["portfolios"]["portfolios"]
    assert [p["portfolio_id"] for p in merged] == ["direct_101", "direct_102"]
    assert all(c.action == _planning.ADD for c in plan.understood)


# --------------------------------------------------------------------------- #
# Nothing is applied that was not understood
# --------------------------------------------------------------------------- #

def test_a_partly_read_instruction_applies_nothing(service, opened):
    plan = service.plan_instruction(
        opened, instruction="The LEI is 894500SYNTHETIC00042. Chase them for "
                            "the rest and sort out the thing we discussed.")
    assert plan.complete is False
    assert plan.unrecognised
    with pytest.raises(PartiallyUnderstood) as excinfo:
        service.apply_plan(opened, plan=plan, actor=ACTOR)
    disclosure = excinfo.value.disclosure
    assert disclosure["understood"], "nothing was reported as understood"
    assert disclosure["unrecognised"], "the missed clause was not reported"
    # And nothing reached the case.
    reloaded = service.load(TENANT_A, opened.case_ref)
    assert not (reloaded.case.items("entities") or [{}])[0].get("lei")


def test_the_disclosed_remainder_can_be_confirmed(service, opened):
    plan = service.plan_instruction(
        opened, instruction="The LEI is 894500SYNTHETIC00042. Sort out the "
                            "thing we discussed.")
    case = service.apply_plan(opened, plan=plan, actor=ACTOR, confirm=True)
    assert case.case.items("entities")[0]["lei"] == "894500SYNTHETIC00042"


def test_a_turn_reports_all_four_populations(service, opened):
    turn = service.instruct(
        opened, text="The LEI is 894500SYNTHETIC00042 and sort out the thing "
                     "we discussed.", actor=ACTOR)
    assert turn.applied is False
    assert turn.proposal is not None
    disclosure = turn.proposal["disclosure"]
    for key in ("understood", "proposed", "questions", "unrecognised"):
        assert key in disclosure, key
    assert "could not read" in turn.reply.lower()


def test_a_fully_read_instruction_needs_no_confirmation_to_be_planned(service,
                                                                      opened):
    plan = service.plan_instruction(
        opened, instruction="The reporting contact is dana@northstar.example.")
    assert plan.complete is True
    assert plan.unrecognised == []
    service.apply_plan(opened, plan=plan, actor=ACTOR)
    reloaded = service.load(TENANT_A, opened.case_ref)
    assert reloaded.case.answers["contacts"]["reporting_contact_email"] == \
        "dana@northstar.example"


def test_planning_writes_nothing(service, opened):
    before = opened.case.answers.get("client", {}).get("jurisdiction")
    service.plan_instruction(opened,
                             instruction="The jurisdiction is the Netherlands.")
    reloaded = service.load(TENANT_A, opened.case_ref)
    assert reloaded.case.answers["client"].get("jurisdiction") == before


# --------------------------------------------------------------------------- #
# Provenance
# --------------------------------------------------------------------------- #

def test_every_applied_answer_records_where_it_came_from(service, opened):
    """Two maps, and the categorised one uses the governed vocabulary.

    ``provenance`` is Client Onboarding's own free-text map, which its screens
    render as written; ``provenance_class`` is the category a control can act
    on. Both are populated, and only the second is constrained.
    """
    from operations_control.occ_agent.interpretation import PROVENANCE_SOURCES
    assert opened.case.provenance, "nothing recorded its provenance"
    assert opened.case.provenance_class, "nothing recorded a category"
    for path, source in opened.case.provenance_class.items():
        assert source in PROVENANCE_SOURCES, f"{path} -> {source}"
        assert opened.case.provenance.get(path), \
            f"{path} has a category but no sentence"


def test_an_answer_the_client_gave_is_marked_as_the_clients(service, opened):
    agent_case = service.request_client_information(opened, actor=ACTOR)
    request = agent_case.case.outstanding_requests[0]
    agent_case = service.record_client_response(
        agent_case, request_id=request.request_id, actor=ACTOR,
        answers={"contacts": {"reporting_contact_name": "Dana Fox"}})
    assert agent_case.case.provenance_class[
        "contacts.reporting_contact_name"] == "client_supplied"


def test_a_value_the_agent_could_only_propose_is_marked_human_approved(service,
                                                                       opened):
    plan = service.plan_instruction(opened,
                                    instruction="The book was acquired.")
    case = service.apply_plan(opened, plan=plan, actor=ACTOR, confirm=True)
    assert case.case.provenance_class["portfolios[0].portfolio_type"] == \
        "human_approved"
