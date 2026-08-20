"""The adapter behind the real OCC Agent, with the real approval gate.

These drive :class:`operations_control.occ_agent.service.OccAgentService`
itself — not a stand-in — over file-backed storage in a tmp directory, with the
Graph transport injected. What is being asserted is that registering a real
mail adapter changed the *receipt* and nothing else:

* the pack workflow still refuses to send anything a human has not approved;
* the audit chain still records the issue, now with the real message id;
* the practice container is still the only place anything was written.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from apps.blob_trigger_app.storage import Storage
from operations_control.engine import OpsError
from operations_control.occ_agent import communication as _comms
from operations_control.occ_agent import states as _states
from operations_control.occ_agent.service import OccAgentService

from trakt_mail.adapter import MailDeliveryRefused, outbound_adapter

from .conftest import MAILBOX, FakeGraph, mail_env

SYNTHETIC_CONTAINER = "operations-control-synthetic"
LIVE_CONTAINER = "operations-control"
TENANT = "client_a"
ACTOR = "Alice"
ME = "joshua@example.com"

OPENING = ("Onboard Northstar Lending. UK equity release. Monthly portfolio "
           "MI. Portfolio id direct_101.")


@pytest.fixture()
def env(tmp_path, monkeypatch):
    monkeypatch.setenv("TRAKT_STORAGE_BACKEND", "file")
    monkeypatch.setenv("TRAKT_LOCAL_BLOB_ROOT", str(tmp_path / "blob"))
    monkeypatch.setenv("TRAKT_OCC_AGENT_SANDBOX_ROOT", str(tmp_path / "sandbox"))
    monkeypatch.setenv("TRAKT_OCC_AGENT_SYNTHETIC_CONTAINER",
                       SYNTHETIC_CONTAINER)
    monkeypatch.setenv("OCC_AGENT_SYNTHETIC_ENABLED", "true")
    # Deliberately absent, as everywhere else in the OCC Agent suite.
    for name in ("AZURE_STORAGE_CONNECTION_STRING", "AZURE_CLIENT_ID",
                 "AZURE_CLIENT_SECRET", "AZURE_TENANT_ID"):
        monkeypatch.delenv(name, raising=False)
    return {"tmp": tmp_path, "blob_root": tmp_path / "blob",
            "sandbox": tmp_path / "sandbox"}


@pytest.fixture()
def storage(env) -> Storage:
    return Storage(env["blob_root"])


@pytest.fixture()
def mailed(env, storage, graph):
    """The service the deployment builds, with Graph mail registered."""
    adapter = outbound_adapter(storage, env=mail_env(), transport=graph)
    assert adapter is not None
    return OccAgentService(storage, container=SYNTHETIC_CONTAINER,
                           sandbox=env["sandbox"], communication=adapter)


@pytest.fixture()
def opened(mailed):
    return mailed.create_case(tenant=TENANT, initiating_user=ACTOR,
                              instruction=OPENING)


# --------------------------------------------------------------------------- #
# The human approval gate is unchanged
# --------------------------------------------------------------------------- #

def test_nothing_can_be_sent_before_a_pack_is_drafted(mailed, opened, graph):
    with pytest.raises(OpsError):
        mailed.send_pack(opened, actor=ACTOR, to=[ME])
    assert graph.send_count == 0


def test_nothing_can_be_sent_before_a_human_approves_it(mailed, opened, graph):
    """The whole point of the seam: an adapter that CAN send still cannot be
    reached from a state a human has not approved."""
    drafted = mailed.draft_pack(opened, actor=ACTOR)
    assert drafted.run.pack_status == _comms.HUMAN_REVIEW_REQUIRED
    with pytest.raises(OpsError):
        mailed.send_pack(drafted, actor=ACTOR, to=[ME])
    assert graph.send_count == 0


def test_an_approved_pack_is_sent_once_and_cannot_be_sent_again(mailed, opened,
                                                                graph):
    approved = mailed.approve_pack(mailed.draft_pack(opened, actor=ACTOR),
                                   actor=ACTOR, reason="read it, it is right")
    assert approved.run.pack_status == _comms.APPROVED_TO_SEND

    sent = mailed.send_pack(approved, actor=ACTOR, to=[ME])
    assert graph.send_count == 1
    assert sent.run.pack_status == _comms.SENT
    assert sent.run.state == _states.PACK_SENT

    # SENT → SENT is not a legal move, so a repeated call cannot mail twice.
    with pytest.raises(OpsError):
        mailed.send_pack(sent, actor=ACTOR, to=[ME])
    assert graph.send_count == 1


# --------------------------------------------------------------------------- #
# The receipt, and the audit trail
# --------------------------------------------------------------------------- #

def test_the_receipt_on_the_run_reports_a_real_send(mailed, opened, graph):
    sent = mailed.send_pack(
        mailed.approve_pack(mailed.draft_pack(opened, actor=ACTOR),
                            actor=ACTOR),
        actor=ACTOR, to=[ME])
    receipt = sent.run.pack_receipt
    assert receipt["adapter"] == "graph_mail"
    assert receipt["sent"] is True
    assert receipt["to"] == [ME]
    assert MAILBOX in receipt["statement"]
    assert "<0001@traktinfra.io>" in receipt["statement"]
    # And the artefacts it names are the ones that were attached.
    attached = {a["name"] for a in
                graph.sent_payloads()[0]["message"]["attachments"]}
    assert attached == {a["name"] for a in receipt["artefacts"]}


def test_the_audit_chain_carries_the_message_identifier(mailed, opened, graph):
    sent = mailed.send_pack(
        mailed.approve_pack(mailed.draft_pack(opened, actor=ACTOR),
                            actor=ACTOR),
        actor=ACTOR, to=[ME])
    events = mailed.store.list_audit(TENANT, sent.case_ref)
    issued = [e for e in events if e["action"] == "onboarding_pack_issued"]
    assert len(issued) == 1
    event = issued[0]
    assert event["detail"]["sent"] is True
    assert event["detail"]["adapter"] == "graph_mail"
    # output_reference is the receipt id, which is the x-trakt-receipt-id
    # header on the message itself — the link from audit to mailbox.
    assert event["output_reference"] == sent.run.pack_receipt["receipt_id"]
    assert "<0001@traktinfra.io>" in event["decision_basis"]
    assert mailed.store.verify_audit_chain(TENANT, sent.case_ref) is True


def test_a_send_that_graph_refuses_leaves_the_pack_approved_and_resendable(
        env, storage, mailed, opened):
    """A failed send must not consume the approval."""
    mailed.communication = outbound_adapter(
        storage, env=mail_env(), transport=FakeGraph(send_error=503))

    approved = mailed.approve_pack(mailed.draft_pack(opened, actor=ACTOR),
                                   actor=ACTOR)
    with pytest.raises(MailDeliveryRefused):
        mailed.send_pack(approved, actor=ACTOR, to=[ME])
    assert approved.run.pack_status == _comms.APPROVED_TO_SEND

    # The same approval now sends, once Graph is available again — no second
    # human decision, and no re-draft.
    recovered = FakeGraph()
    mailed.communication = outbound_adapter(storage, env=mail_env(),
                                            transport=recovered)
    reloaded = mailed.load(TENANT, approved.case_ref)
    assert reloaded.run.pack_status == _comms.APPROVED_TO_SEND
    assert mailed.send_pack(reloaded, actor=ACTOR, to=[ME]).run.pack_status \
        == _comms.SENT
    assert recovered.send_count == 1


# --------------------------------------------------------------------------- #
# The practice boundary still holds
# --------------------------------------------------------------------------- #

def test_registering_a_mail_adapter_writes_nothing_to_the_live_container(
        mailed, opened, env):
    mailed.send_pack(
        mailed.approve_pack(mailed.draft_pack(opened, actor=ACTOR),
                            actor=ACTOR),
        actor=ACTOR, to=[ME])
    live = Path(env["blob_root"]) / LIVE_CONTAINER
    assert not [p for p in live.rglob("*") if p.is_file()] \
        if live.exists() else True


def test_an_unconfigured_deployment_keeps_the_record_only_adapter(env, storage):
    """The fallback path: no mail settings, so the OCC states plainly that
    nothing was sent — exactly as it did before this package existed."""
    assert outbound_adapter(storage, env={}) is None
    service = OccAgentService(storage, container=SYNTHETIC_CONTAINER,
                              sandbox=env["sandbox"],
                              communication=outbound_adapter(storage, env={}))
    case = service.create_case(tenant=TENANT, initiating_user=ACTOR,
                               instruction=OPENING)
    sent = service.send_pack(
        service.approve_pack(service.draft_pack(case, actor=ACTOR),
                             actor=ACTOR),
        actor=ACTOR, to=[ME])
    assert sent.run.pack_receipt["sent"] is False
    assert "did not send" in sent.run.pack_receipt["statement"]
