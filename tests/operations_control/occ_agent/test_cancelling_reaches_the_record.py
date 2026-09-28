"""Cancelling from the chat box, all the way to the withdrawn case.

``tests/test_cancelling_says_why.py`` pins how the sentence is READ. This pins
what the sentence DOES — because a reason that reaches the payload and stops
there is the same defect wearing a different coat, and the payload was never
the thing anyone reads six months later.

The chat path called ``cancel(agent_case, actor=actor)`` with no reason at all,
so whatever the operator typed, the record said "The practice case was
cancelled." The governed dialog refuses a blank reason; this path collected one
and discarded it. Asserting on the withdrawn case is the only assertion that
would have caught that.
"""

from __future__ import annotations

import pytest

from operations_control.occ_agent import states as _states

from .conftest import ACTOR, TENANT_A

OPENING = ("Onboard Northstar Lending. UK equity release. Monthly portfolio "
           "MI. Portfolio id direct_101.")

#: A real handover sentence: the instruction, then why.
SAID = ("Cancel this practice case. Superseded by a fresh onboarding so that "
        "elapsed time measures the client engagement.")


@pytest.fixture()
def opened(service):
    return service.create_case(tenant=TENANT_A, initiating_user=ACTOR,
                               instruction=OPENING)


def test_it_is_proposed_before_it_is_done(service, opened):
    """Abandoning a case is material, so one sentence never does it."""
    result = service.instruct(opened, text=SAID, actor=ACTOR)
    # Coming back as a proposal at all IS the gate: a material change is never
    # applied on the turn that asked for it.
    assert result.applied is False
    assert result.proposal is not None
    assert result.proposal["action"] == _states.ACTION_CANCEL
    assert result.proposal["material"] is True
    # The reason has to survive being parked as a proposal and picked up again,
    # so it is asserted here as well as on the withdrawn case.
    assert result.proposal["payload"]["reason"] == SAID
    assert service.load(TENANT_A, opened.case_ref).case.status != "withdrawn"


def test_the_operators_words_are_the_withdrawal_reason(service, opened):
    service.instruct(opened, text=SAID, actor=ACTOR, confirm=True)

    reloaded = service.load(TENANT_A, opened.case_ref)
    assert reloaded.case.status == "withdrawn"
    assert reloaded.case.withdrawal_reason == SAID
    assert reloaded.case.withdrawn_by == ACTOR
    assert reloaded.case.withdrawn_at


def test_the_run_is_cancelled_too(service, opened):
    """One instruction ends both: the practice run and the case under it."""
    service.instruct(opened, text=SAID, actor=ACTOR, confirm=True)
    assert service.load(TENANT_A, opened.case_ref).run.state == \
        _states.CANCELLED


def test_the_audit_records_what_was_said(service, opened):
    """Not "cancelled by the operator" — the sentence itself."""
    service.instruct(opened, text=SAID, actor=ACTOR, confirm=True)
    entries = [e for e in service.store.list_audit(TENANT_A, opened.case_ref)
               if e.get("action") == "practice_case_cancelled"]
    assert entries, "cancelling is audited"
    assert entries[-1].get("decision_basis") == SAID
    assert service.store.verify_audit_chain(TENANT_A, opened.case_ref)


def test_saying_only_the_instruction_still_reads_sensibly(service, opened):
    """No rationale given is not a crash, and not an empty reason.

    An operator who types the bare instruction gets their own words on the
    record rather than a synthesised sentence — which is still an improvement,
    because "Cancel this case." is at least verifiably what they said.
    """
    service.instruct(opened, text="Cancel this case.", actor=ACTOR,
                     confirm=True)
    reloaded = service.load(TENANT_A, opened.case_ref)
    assert reloaded.case.status == "withdrawn"
    assert reloaded.case.withdrawal_reason.strip()
