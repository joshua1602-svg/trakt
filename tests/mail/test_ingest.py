"""A client's reply, taken into a real case through the real governed methods.

These drive :class:`operations_control.occ_agent.service.OccAgentService`
itself over file-backed storage, send a pack through the injected Graph
transport, then put a reply in the same fake mailbox and ingest it.

The properties under test are the ones that make this safe to run against a
live mailbox:

* a file the client attached is registered by the SAME method an operator's
  upload uses, so it inherits the same state check and the same audit;
* the client's WORDS are recorded and never applied — no answer on the case
  changes because an email arrived;
* the same reply cannot be taken in twice;
* a message that cannot be tied to the case by evidence is refused, and no
  parameter exists to override that.
"""

from __future__ import annotations

import base64

import pytest

from apps.blob_trigger_app.storage import Storage
from operations_control.occ_agent import states as _states
from operations_control.occ_agent.service import OccAgentService

from trakt_mail import config as _config
from trakt_mail import ingest as _ingest
from trakt_mail.adapter import outbound_adapter

from .conftest import FakeGraph, inbound_env, mail_env

SYNTHETIC_CONTAINER = "operations-control-synthetic"
TENANT = "client_a"
ACTOR = "Alice"
CLIENT_CONTACT = "ops@erm.example"

OPENING = ("Onboard ERM Capital Management. UK equity release. Monthly "
           "portfolio MI. Portfolio id direct_101.")


@pytest.fixture()
def env(tmp_path, monkeypatch):
    monkeypatch.setenv("TRAKT_STORAGE_BACKEND", "file")
    monkeypatch.setenv("TRAKT_LOCAL_BLOB_ROOT", str(tmp_path / "blob"))
    monkeypatch.setenv("TRAKT_OCC_AGENT_SANDBOX_ROOT", str(tmp_path / "sandbox"))
    monkeypatch.setenv("TRAKT_OCC_AGENT_SYNTHETIC_CONTAINER",
                       SYNTHETIC_CONTAINER)
    monkeypatch.setenv("OCC_AGENT_SYNTHETIC_ENABLED", "true")
    for name in ("AZURE_STORAGE_CONNECTION_STRING", "AZURE_CLIENT_ID",
                 "AZURE_CLIENT_SECRET", "AZURE_TENANT_ID"):
        monkeypatch.delenv(name, raising=False)
    return {"blob_root": tmp_path / "blob", "sandbox": tmp_path / "sandbox"}


@pytest.fixture()
def graph() -> FakeGraph:
    return FakeGraph()


@pytest.fixture()
def service(env, graph):
    storage = Storage(env["blob_root"])
    adapter = outbound_adapter(storage, env=mail_env(), transport=graph)
    assert adapter is not None
    return OccAgentService(storage, container=SYNTHETIC_CONTAINER,
                           sandbox=env["sandbox"], communication=adapter)


@pytest.fixture()
def issued(service, graph):
    """A case whose pack has actually been sent through the fake Graph."""
    opened = service.create_case(tenant=TENANT, initiating_user=ACTOR,
                                 instruction=OPENING)
    approved = service.approve_pack(service.draft_pack(opened, actor=ACTOR),
                                    actor=ACTOR, reason="read it")
    return service.send_pack(approved, actor=ACTOR, to=[CLIENT_CONTACT])


@pytest.fixture()
def config():
    return _config.load_inbound(inbound_env())


def reply(case_ref: str, *, attachments: bool = False, **overrides):
    """The client's reply, as Graph would return it from the inbox."""
    receipt_conversation = "AAQkAGI2-conversation-id"
    payload = {
        "id": "AAMkAGI2-reply-1",
        "internetMessageId": "<reply-1@erm.example>",
        "conversationId": receipt_conversation,
        "subject": f"RE: Onboarding information request ({case_ref})",
        "from": {"emailAddress": {"name": "Dana Okafor",
                                  "address": CLIENT_CONTACT}},
        "receivedDateTime": "2026-08-21T08:14:00Z",
        "hasAttachments": attachments,
        "isRead": False,
        "uniqueBody": {"contentType": "text",
                       "content": "Our LEI is 5493001KJTIIGC8Y1R12."},
        "internetMessageHeaders": [
            {"name": "In-Reply-To", "value": "<0001@traktinfra.io>"}],
    }
    payload.update(overrides)
    return payload


TAPE = {
    "@odata.type": "#microsoft.graph.fileAttachment",
    "name": "erm_funded_book.csv",
    "contentType": "text/csv",
    "size": 24,
    "contentBytes": base64.b64encode(
        b"loan_id,balance\nL1,1000\n").decode("ascii"),
}


def waiting_for(service, graph, config, case_ref):
    found = _ingest.waiting(service, tenant=TENANT, config=config,
                            transport=graph)
    matched = found.for_case(case_ref)
    assert matched, f"no reply was matched to {case_ref}: {found.to_dict()}"
    return found, matched[0]


# --------------------------------------------------------------------------- #
# Finding the case
# --------------------------------------------------------------------------- #

def test_a_reply_is_matched_to_the_case_whose_pack_it_answers(service, issued,
                                                              graph, config):
    graph.inbox = [reply(issued.case_ref)]
    _found, chosen = waiting_for(service, graph, config, issued.case_ref)
    assert chosen.correlation.matched
    assert "conversation" in chosen.correlation.bases


def test_only_a_case_that_actually_issued_a_pack_can_be_matched(service, graph,
                                                                config):
    """A case Trakt has never written to cannot be the subject of a reply, and
    is not offered to the correlator at all."""
    service.create_case(tenant=TENANT, initiating_user=ACTOR,
                        instruction=OPENING)
    assert _ingest.targets_for(service, TENANT) == []


def test_looking_at_the_mailbox_writes_nothing_and_marks_nothing(service,
                                                                 issued, graph,
                                                                 config):
    graph.inbox = [reply(issued.case_ref)]
    before = service.store.load(TENANT, issued.case_ref).to_dict()
    _ingest.waiting(service, tenant=TENANT, config=config, transport=graph)
    after = service.store.load(TENANT, issued.case_ref).to_dict()
    assert before == after
    assert graph.marked_read == []


def test_reading_is_reported_as_off_rather_than_failing(service, issued, graph):
    found = _ingest.waiting(service, tenant=TENANT, env={}, transport=graph)
    assert found.messages == []
    assert "not enabled" in found.note


# --------------------------------------------------------------------------- #
# Taking one in
# --------------------------------------------------------------------------- #

def test_an_attached_file_is_registered_through_the_governed_method(
        service, issued, graph, config):
    graph.inbox = [reply(issued.case_ref, attachments=True)]
    graph.attachments = {"AAMkAGI2-reply-1": [TAPE]}
    _found, chosen = waiting_for(service, graph, config, issued.case_ref)

    updated, outcome = _ingest.ingest(service, issued, chosen, actor=ACTOR,
                                      config=config, transport=graph)
    assert outcome.registered == ["erm_funded_book.csv"]
    names = [a["source_file"] for a in updated.run.received_artefacts]
    assert "erm_funded_book.csv" in names
    # Registered into the practice sandbox, exactly as an upload would be.
    assert all(a["execution_status"] != "live" for a in
               updated.run.received_artefacts)


def test_the_clients_words_are_recorded_and_not_applied(service, issued, graph,
                                                        config):
    """The line this module must not cross.

    A sentence in an email is not an approved answer. The reply is recorded so
    an operator can read it; the case's answers are untouched until a human
    instructs the change through the existing path.
    """
    before = dict(service.onboarding.load_case(issued.case_ref).answers)
    graph.inbox = [reply(issued.case_ref)]
    _found, chosen = waiting_for(service, graph, config, issued.case_ref)

    updated, outcome = _ingest.ingest(service, issued, chosen, actor=ACTOR,
                                      config=config, transport=graph)
    assert outcome.recorded_text is True
    after = service.onboarding.load_case(issued.case_ref).answers
    assert after == before, "an email changed the case without a human"

    client_turns = [m for m in updated.run.messages if m["role"] == "client"]
    assert len(client_turns) == 1
    assert "5493001KJTIIGC8Y1R12" in client_turns[0]["text"]
    assert client_turns[0]["author"] == "Dana Okafor"


def test_the_statement_says_the_reply_was_not_applied(service, issued, graph,
                                                      config):
    graph.inbox = [reply(issued.case_ref)]
    _found, chosen = waiting_for(service, graph, config, issued.case_ref)
    _updated, outcome = _ingest.ingest(service, issued, chosen, actor=ACTOR,
                                       config=config, transport=graph)
    assert "has NOT been applied" in outcome.statement


def test_taking_a_reply_in_is_audited_with_the_evidence_it_was_matched_on(
        service, issued, graph, config):
    graph.inbox = [reply(issued.case_ref)]
    _found, chosen = waiting_for(service, graph, config, issued.case_ref)
    _ingest.ingest(service, issued, chosen, actor=ACTOR, config=config,
                   transport=graph)
    events = service.store.list_audit(TENANT, issued.case_ref)
    ingested = [e for e in events if e["action"] == "client_reply_ingested"]
    assert len(ingested) == 1
    assert ingested[0]["actor_identity"] == ACTOR
    assert "conversation" in ingested[0]["decision_basis"]


def test_the_same_reply_cannot_be_taken_in_twice(service, issued, graph,
                                                 config):
    """Read state in the mailbox is a hint, not the memory: a person opening
    Outlook marks mail read too. The run's own record is what decides."""
    graph.inbox = [reply(issued.case_ref, attachments=True)]
    graph.attachments = {"AAMkAGI2-reply-1": [TAPE]}
    _found, chosen = waiting_for(service, graph, config, issued.case_ref)

    updated, first = _ingest.ingest(service, issued, chosen, actor=ACTOR,
                                    config=config, transport=graph)
    assert first.registered == ["erm_funded_book.csv"]

    graph.inbox = [reply(issued.case_ref, attachments=True, isRead=True)]
    again = _ingest.waiting(service, tenant=TENANT, config=config,
                            transport=graph)
    assert again.messages[0].already_ingested is True

    updated, second = _ingest.ingest(service, updated, again.messages[0],
                                     actor=ACTOR, config=config,
                                     transport=graph)
    assert second.already is True
    assert second.registered == []
    files = [a["source_file"] for a in updated.run.received_artefacts]
    assert files.count("erm_funded_book.csv") == 1


def test_a_reply_is_marked_read_so_it_is_not_offered_again(service, issued,
                                                           graph, config):
    graph.inbox = [reply(issued.case_ref)]
    _found, chosen = waiting_for(service, graph, config, issued.case_ref)
    _ingest.ingest(service, issued, chosen, actor=ACTOR, config=config,
                   transport=graph)
    assert graph.marked_read == ["AAMkAGI2-reply-1"]


# --------------------------------------------------------------------------- #
# What it refuses
# --------------------------------------------------------------------------- #

def test_a_message_that_cannot_be_tied_to_a_case_is_refused(service, issued,
                                                            graph, config):
    graph.inbox = [reply(issued.case_ref, conversationId="other",
                         subject="Some files", internetMessageHeaders=[])]
    found = _ingest.waiting(service, tenant=TENANT, config=config,
                            transport=graph)
    assert found.for_case(issued.case_ref) == []
    with pytest.raises(_ingest.MailIngestError) as excinfo:
        _ingest.ingest(service, issued, found.messages[0], actor=ACTOR,
                       config=config, transport=graph)
    assert "could not be tied" in str(excinfo.value)


def test_a_reply_belonging_to_another_case_is_refused(service, issued, graph,
                                                      config):
    other = service.create_case(tenant=TENANT, initiating_user=ACTOR,
                                instruction=OPENING)
    other = service.send_pack(
        service.approve_pack(service.draft_pack(other, actor=ACTOR),
                             actor=ACTOR), actor=ACTOR, to=[CLIENT_CONTACT])
    graph.inbox = [reply(issued.case_ref)]
    found = _ingest.waiting(service, tenant=TENANT, config=config,
                            transport=graph)
    chosen = found.messages[0]
    with pytest.raises(_ingest.MailIngestError) as excinfo:
        _ingest.ingest(service, other, chosen, actor=ACTOR, config=config,
                       transport=graph)
    assert "belongs to" in str(excinfo.value)


def test_ingest_takes_no_override_that_would_force_an_unmatched_message():
    """There is deliberately no ``force``. If evidence cannot establish which
    case a message belongs to, the fix is a person looking at the mailbox."""
    import inspect
    parameters = set(inspect.signature(_ingest.ingest).parameters)
    assert "force" not in parameters
    assert "case_ref" not in parameters


def test_a_signature_image_is_not_registered_as_a_client_data_file(
        service, issued, graph, config):
    graph.inbox = [reply(issued.case_ref, attachments=True)]
    graph.attachments = {"AAMkAGI2-reply-1": [
        TAPE,
        {"@odata.type": "#microsoft.graph.fileAttachment",
         "name": "logo.png", "contentType": "image/png", "size": 4,
         "isInline": True, "contentBytes": base64.b64encode(b"\x89PNG").decode()},
    ]}
    _found, chosen = waiting_for(service, graph, config, issued.case_ref)
    _updated, outcome = _ingest.ingest(service, issued, chosen, actor=ACTOR,
                                       config=config, transport=graph)
    assert outcome.registered == ["erm_funded_book.csv"]
    assert [s["name"] for s in outcome.skipped] == ["logo.png"]


def test_an_oversize_attachment_is_reported_not_silently_dropped(
        service, issued, graph):
    config = _config.load_inbound(inbound_env(
        TRAKT_MAIL_INBOUND_MAX_ATTACHMENT_BYTES="10"))
    graph.inbox = [reply(issued.case_ref, attachments=True)]
    graph.attachments = {"AAMkAGI2-reply-1": [TAPE]}
    _found, chosen = waiting_for(service, graph, config, issued.case_ref)
    _updated, outcome = _ingest.ingest(service, issued, chosen, actor=ACTOR,
                                       config=config, transport=graph)
    assert outcome.registered == []
    assert "over the limit" in outcome.skipped[0]["reason"]
    assert "erm_funded_book.csv" in outcome.statement


def test_registering_still_obeys_the_runs_own_state(service, issued, graph,
                                                    config):
    """The governed method is not bypassed. A run in a state that does not
    permit registering an artefact refuses a mail attachment exactly as it
    would refuse an operator's upload."""
    graph.inbox = [reply(issued.case_ref, attachments=True)]
    graph.attachments = {"AAMkAGI2-reply-1": [TAPE]}
    _found, chosen = waiting_for(service, graph, config, issued.case_ref)

    issued.run.state = _states.SYNTHETIC_ONBOARDING_RUNNING
    service.store.save(issued.run)
    with pytest.raises(Exception) as excinfo:
        _ingest.ingest(service, issued, chosen, actor=ACTOR, config=config,
                       transport=graph)
    assert "register" in str(excinfo.value).lower()
