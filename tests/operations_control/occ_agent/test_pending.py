"""What "what is still pending?" has to answer.

An operator asks that question when they cannot see the way forward. The
answer that used to come back was the readiness table — the criteria for the
END of a rehearsal — which at the start of a case is thirteen unmet criteria
whose remedies restate their own labels ("Onboarding approved: work the
onboarding through to approval"). Every word of it true, and none of it usable.

These tests pin the two things the answer must do instead: name who the case is
waiting on, and name the next thing this operator can actually do.
"""

from __future__ import annotations

import pytest

from operations_control.occ_agent import states as _states

from .conftest import ACTOR, TENANT_A

OPENING = ("Onboard Northstar Lending. UK equity release. Monthly portfolio "
           "MI. Portfolio id direct_101.")


@pytest.fixture()
def opened(service):
    return service.create_case(tenant=TENANT_A, initiating_user=ACTOR,
                               instruction=OPENING)


@pytest.fixture()
def issued(service, opened):
    """A case whose pack has actually gone out — the state the defect was in."""
    approved = service.approve_pack(service.draft_pack(opened, actor=ACTOR),
                                    actor=ACTOR)
    return service.send_pack(approved, actor=ACTOR,
                             to=["reporting@northstar.example"])


# --------------------------------------------------------------------------- #
# The defect
# --------------------------------------------------------------------------- #

def test_what_is_pending_does_not_answer_with_the_readiness_table(service,
                                                                  issued):
    reply = service.answer(issued, "What is still pending")
    assert "Work the onboarding through to approval" not in reply
    assert "Generate the orchestration plan" not in reply


def test_what_is_pending_names_what_the_client_still_has_to_tell_us(service,
                                                                   issued):
    reply = service.answer(issued, "What is still pending")
    assert "Legal Entity Identifier" in reply


def test_what_is_pending_names_who_it_is_waiting_on(service, issued):
    reply = service.answer(issued, "What is still pending")
    assert "waiting on Northstar Lending" in reply


def test_what_is_pending_names_something_the_operator_can_do(service, issued):
    reply = service.answer(issued, "What is still pending")
    assert "What you can do now:" in reply
    assert "provide a file" in reply


def test_a_case_whose_pack_has_not_gone_out_is_not_waiting_on_the_client(
        service, opened):
    p = service.pending(opened)
    assert p["issued"] is False
    assert p["waiting_on"] == "you"
    assert "waiting on Northstar Lending" not in \
        service.pending_sentences(opened)


def test_the_readiness_table_is_still_reachable_by_name(service, issued):
    reply = service.answer(issued, "What is outstanding for readiness?")
    assert "Work the onboarding through to approval" in reply


# --------------------------------------------------------------------------- #
# The shape of the answer
# --------------------------------------------------------------------------- #

@pytest.mark.parametrize("question", [
    "What is still pending",
    "What is left to do?",
    "Why is this stuck?",
    "What should I do next?",
    "What are we waiting for?",
])
def test_every_way_of_asking_reaches_the_same_answer(service, issued,
                                                     question):
    assert "What you can do now:" in service.answer(issued, question)


def test_pending_reports_the_open_information_requests(service, issued):
    asked = service.request_client_information(issued, actor=ACTOR)
    p = service.pending(asked)
    assert p["requests"], "an open request was not reported"
    assert "cannot be submitted for approval" in \
        service.pending_sentences(asked)


def test_pending_never_invents_a_next_step_it_cannot_offer(service, issued):
    p = service.pending(issued)
    # Every action named is one the case would actually accept right now.
    for action in p["actions"]:
        assert _states.action_allowed(issued.run.state, action) \
            or action in (_states.ACTION_REQUEST_INFORMATION,
                          _states.ACTION_RECORD_RESPONSE,
                          _states.ACTION_SUBMIT_FOR_APPROVAL,
                          _states.ACTION_APPROVE_ONBOARDING)


def test_pending_does_not_offer_cancelling_as_the_way_forward(service, issued):
    assert _states.ACTION_CANCEL not in service.pending(issued)["actions"]


def test_available_actions_will_not_offer_approval_the_case_would_refuse(
        service, issued):
    """The submit button is not offered while information is outstanding.

    Client Onboarding refuses ``submit_for_approval`` until its own readiness
    is clean, so naming it here would send an operator to a refusal — which is
    exactly the experience of a workflow that looks stuck.
    """
    assert not service.onboarding_readiness(issued)["ready"]
    actions = service.available_actions(issued)
    assert _states.ACTION_SUBMIT_FOR_APPROVAL not in actions
    assert _states.ACTION_APPROVE_ONBOARDING not in actions


def test_available_actions_offers_submission_once_the_case_is_clean(service):
    from operations_control.occ_agent.scenarios import run_scenario
    ran = run_scenario(service, "scenario_a_clean", tenant=TENANT_A,
                       actor=ACTOR)
    # A case worked through to approval has nothing left to submit.
    assert service.onboarding_readiness(ran.case)["ready"]


def test_the_answer_survives_a_case_with_no_client_name(service):
    bare = service.create_case(tenant=TENANT_A, initiating_user=ACTOR)
    reply = service.pending_sentences(bare)
    assert reply
    assert "the client" in reply or "What you can do now:" in reply


# --------------------------------------------------------------------------- #
# The list must not say the same thing twice
# --------------------------------------------------------------------------- #

def test_the_answer_does_not_list_the_client_questions_twice(service, issued):
    """Every unanswered client field is BOTH a checklist row and a blocking
    problem. Reporting both in full is what buried the two items that were
    actually different among seven that were not."""
    reply = service.pending_sentences(issued)
    assert reply.count("Legal Entity Identifier") == 1


def test_what_is_pending_surfaces_what_the_operator_must_supply(service,
                                                               issued):
    p = service.pending(issued)
    assert p["yours"], "nothing was attributed to the operator"
    # None of it duplicates a question already put to the client.
    labels = [row["label"] for row in p["client"]]
    for message in p["yours"]:
        assert not any(label.split(" — ")[0] in message for label in labels)


def test_the_residue_is_named_as_the_operators_own(service, issued):
    assert "Needs you, not the client" in service.pending_sentences(issued)


def test_the_answer_says_the_operator_need_not_wait_for_the_email(service,
                                                                  issued):
    assert "do not have to wait" in service.pending_sentences(issued)
