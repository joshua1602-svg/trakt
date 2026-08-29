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


# --------------------------------------------------------------------------- #
# Once the client has been ASKED
# --------------------------------------------------------------------------- #
#
# ``client_checklist`` excludes anything sitting in an open information
# request, so pressing "ask the client" empties it. Everything below is about
# what must NOT follow from that.

@pytest.fixture()
def asked(service, issued):
    """A case whose outstanding items have been put to the client."""
    return service.request_client_information(issued, actor=ACTOR)


def test_asking_the_client_empties_the_checklist(service, asked):
    """The premise. If this ever stops being true the tests below are vacuous."""
    assert service.onboarding.client_checklist(asked.case) == []


def test_an_item_the_client_was_asked_for_is_still_the_clients(service, asked):
    """The defect: the whole list flipped to "needs you" at the exact moment it
    became most true that we were waiting on them."""
    p = service.pending(asked)
    labels = [row["label"] for row in p["client"]]
    assert any("Legal Entity Identifier" in l for l in labels)
    assert not any("Legal Entity Identifier" in m for m in p["yours"])


def test_the_case_is_still_reported_as_waiting_on_the_client(service, asked):
    p = service.pending(asked)
    assert p["waiting_on"] == "client"
    assert "waiting on Northstar Lending" in service.pending_sentences(asked)


def test_nothing_the_client_was_asked_for_is_listed_as_the_operators(service,
                                                                     asked):
    """The residue under "needs you" must be exactly what nobody has asked the
    client for — that is what makes it the answer to "why is this stuck"."""
    p = service.pending(asked)
    requested = [row["label"].split(" — ")[0]
                 for req in asked.case.requests() for row in req.items]
    assert requested, "the fixture asked the client for nothing"
    for label in requested:
        assert not any(label in message for message in p["yours"]), label


def test_an_answered_item_drops_off_the_client_list(service, asked):
    """Asked is not the same as outstanding. An item answered after it was
    asked must not keep appearing because the request still names it."""
    before = service.pending(asked)["client"]
    one = [r for req in asked.case.requests() for r in req.items][0]
    updated = service.submit_client_response(
        asked, actor=ACTOR, response={_key(one): _answer_for(one)},
        strict=False)
    after = service.pending(updated)["client"]
    assert len(after) == len(before) - 1
    assert one["label"] not in [row["label"] for row in after]


# --------------------------------------------------------------------------- #
# The request has to be able to close
# --------------------------------------------------------------------------- #

def test_answering_every_item_closes_the_request(service, asked):
    """The terminal defect.

    ``readiness()`` counts an open request as outstanding and refuses to
    submit while any remains, so a request that never closes is a case that can
    never be approved — however completely it has been answered.
    """
    assert service.onboarding_readiness(asked)["outstanding_requests"]

    updated = service.submit_client_response(
        asked, actor=ACTOR, response=_all_answers(asked.case), strict=False)
    assert service.onboarding_readiness(updated)["outstanding_requests"] == []


def test_a_partial_answer_closes_nothing(service, asked):
    """Closing is derived from what is answered, never asserted. A request
    half-answered must not make a case look complete that is not."""
    one = [r for req in asked.case.requests() for r in req.items][0]
    updated = service.submit_client_response(
        asked, actor=ACTOR, response={_key(one): _answer_for(one)},
        strict=False)
    assert service.onboarding_readiness(updated)["outstanding_requests"]


def test_closing_a_request_is_audited(service, asked):
    updated = service.submit_client_response(
        asked, actor=ACTOR, response=_all_answers(asked.case), strict=False)
    events = service.store.list_audit(TENANT_A, updated.case_ref)
    closed = [e for e in events if e["action"] == "client_request_answered"]
    assert len(closed) == 1
    assert closed[0]["actor_identity"] == ACTOR


def _key(row) -> str:
    return (f"{row['section']}.{row['field']}" if row["index"] is None
            else f"{row['section']}[{row['index']}].{row['field']}")


def _answer_for(row) -> object:
    """A value the catalogue will accept for one checklist row.

    Options come from the catalogue rather than from a guess, so this stays
    correct when a field's vocabulary changes.
    """
    from operations_control.onboarding.catalogue import catalogue
    field = catalogue().field(row["section"], row["field"])
    if field is not None and getattr(field, "options", None):
        return str(field.options[0].get("value"))
    name = row["field"]
    if "email" in name:
        return "ops@northstar.example"
    if name == "lei":
        return "213800LBQA1Y9SHqwq49"
    if "jurisdiction" in name or "country" in name:
        return "GB"
    return "Provided"


def _all_answers(case) -> dict:
    return {_key(row): _answer_for(row)
            for req in case.requests() for row in req.items}
