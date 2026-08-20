"""The adapter: what it refuses, what it attaches, and what the receipt says.

Every refusal here is asserted to have sent NOTHING. A half-send — a covering
note without the pack, or a message to an address that should not have received
one — is the failure mode this layer exists to prevent, and "it raised" is not
the same claim as "no mail left the building".
"""

from __future__ import annotations

import base64

import pytest

from trakt_mail.adapter import (
    ADAPTER_NAME, GraphMailAdapter, MAX_ATTACHMENTS, MAX_RECIPIENTS,
    MailDeliveryRefused, from_env, outbound_adapter,
)
from trakt_mail.config import MailNotConfigured
from trakt_mail.graph_client import RECEIPT_HEADER

from .conftest import CLIENT_SECRET, MAILBOX, FakeGraph, mail_env

PACK = [{"name": "onboarding_pack.md", "ref": "blob://synthetic/pack.md"},
        {"name": "covering_email.txt", "ref": "blob://synthetic/email.txt"}]

PACK_MARKDOWN = b"""# Onboarding \xe2\x80\x94 Northstar Lending

Reference: ONB-2026-0007

There are 2 questions for you, 1 of them required.

## About your business

- [required] **Legal entity name**
      As registered at Companies House.
- [optional] **Trading name**

## Files to send

- **Primary loan tape** - required
"""

STORED = {
    "blob://synthetic/pack.md": PACK_MARKDOWN,
    "blob://synthetic/email.txt": b"To: client\nSubject: Trakt onboarding\n",
}


def adapter(graph: FakeGraph, *, resolver=None, **env) -> GraphMailAdapter:
    return from_env(resolver=resolver if resolver is not None else STORED.get,
                    transport=graph, env=mail_env(**env))


def deliver(built: GraphMailAdapter, **kwargs):
    payload = {"to": ["client@example.com"], "subject": "Trakt onboarding",
               "body": "Please complete the attached pack.",
               "artefacts": PACK, "content_hash": "abc123", "actor": "Alice"}
    payload.update(kwargs)
    return built.deliver(**payload)


# --------------------------------------------------------------------------- #
# The happy path, and the receipt
# --------------------------------------------------------------------------- #

def test_an_approved_pack_is_sent_and_the_receipt_names_the_message(graph):
    receipt = deliver(adapter(graph))
    assert receipt.adapter == ADAPTER_NAME
    assert receipt.sent is True
    assert receipt.to == ["client@example.com"]
    assert receipt.content_hash == "abc123"
    # The identifiers an auditor follows from the audit chain to the mailbox.
    assert "<0001@traktinfra.io>" in receipt.statement
    assert "AAMkAGI2-graph-message-id" in receipt.statement
    assert MAILBOX in receipt.statement
    assert "Alice" in receipt.statement
    assert receipt.receipt_id in receipt.statement
    assert "Trakt_Onboarding_Checklist_Northstar_Lending.pdf" in \
        receipt.statement


def test_the_receipt_id_is_the_header_on_the_message_that_was_sent(graph):
    """The audit record and the message in the mailbox name each other."""
    receipt = deliver(adapter(graph))
    headers = graph.sent_payloads()[0]["message"]["internetMessageHeaders"]
    assert {"name": RECEIPT_HEADER, "value": receipt.receipt_id} in headers


def test_the_client_receives_one_pdf_and_no_markdown(graph):
    """The two defects, asserted: no covering_email.txt, no .md."""
    deliver(adapter(graph))
    attachments = graph.sent_payloads()[0]["message"]["attachments"]
    assert [a["name"] for a in attachments] == [
        "Trakt_Onboarding_Checklist_Northstar_Lending.pdf"]
    assert attachments[0]["@odata.type"] == "#microsoft.graph.fileAttachment"
    assert attachments[0]["contentType"] == "application/pdf"
    assert base64.b64decode(attachments[0]["contentBytes"]).startswith(b"%PDF-")


def test_the_covering_message_is_in_the_body_not_an_attachment(graph):
    deliver(adapter(graph), body="Please complete the attached pack.")
    message = graph.sent_payloads()[0]["message"]
    assert message["body"]["contentType"] == "HTML"
    assert "Please complete the attached pack." in message["body"]["content"]
    names = [a["name"] for a in message["attachments"]]
    assert "covering_email.txt" not in names
    assert "onboarding_pack.md" not in names


def test_the_subject_and_the_approved_words_are_sent_verbatim(graph):
    """The adapter reformats; it does not rewrite. Every approved word
    survives into the body it renders."""
    deliver(adapter(graph), subject="EXACTLY THIS",
            body="First approved line.\n\nSecond approved line.")
    message = graph.sent_payloads()[0]["message"]
    assert message["subject"] == "EXACTLY THIS"
    assert "First approved line." in message["body"]["content"]
    assert "Second approved line." in message["body"]["content"]


def test_the_body_is_html_with_no_remote_resources(graph):
    """A body that fetches anything renders as a broken box for most clients
    and trips image blocking for the rest."""
    deliver(adapter(graph))
    content = graph.sent_payloads()[0]["message"]["body"]["content"]
    assert "<table" in content
    for forbidden in ("<script", "<img", "http://", "https://", "@import"):
        assert forbidden not in content, forbidden


def test_a_communication_with_no_pack_sends_the_body_alone(graph):
    """A follow-up need not carry a checklist."""
    receipt = deliver(adapter(graph), artefacts=[])
    message = graph.sent_payloads()[0]["message"]
    assert "attachments" not in message
    assert receipt.sent is True
    assert "No attachment." in receipt.statement


def test_an_unconfirmed_send_is_reported_as_unconfirmed_not_as_confirmed(graph):
    graph.read_error = 403
    receipt = deliver(adapter(graph))
    assert receipt.sent is True                 # Graph accepted it; it has gone
    assert "could not be read back" in receipt.statement
    assert receipt.receipt_id in receipt.statement


# --------------------------------------------------------------------------- #
# The refusals — each one sends nothing
# --------------------------------------------------------------------------- #

def test_a_recipient_that_is_not_an_address_is_refused(graph):
    with pytest.raises(MailDeliveryRefused):
        deliver(adapter(graph), to=["not-an-address"])
    assert graph.send_count == 0


def test_no_recipient_is_refused(graph):
    with pytest.raises(MailDeliveryRefused):
        deliver(adapter(graph), to=["", "   "])
    assert graph.send_count == 0


def test_a_recipient_outside_the_allow_list_is_refused(graph):
    built = adapter(graph, TRAKT_MAIL_RECIPIENT_ALLOWLIST="me@example.com")
    with pytest.raises(MailDeliveryRefused) as caught:
        deliver(built, to=["a-real-client@example.com"])
    assert caught.value.http_status == 403
    assert graph.send_count == 0
    # …and the permitted address still goes.
    assert deliver(built, to=["me@example.com"]).sent is True


def test_more_recipients_than_a_client_pack_can_have_is_refused(graph):
    many = [f"c{n}@example.com" for n in range(MAX_RECIPIENTS + 1)]
    with pytest.raises(MailDeliveryRefused):
        deliver(adapter(graph), to=many)
    assert graph.send_count == 0


def test_duplicate_recipients_are_collapsed_not_sent_twice(graph):
    receipt = deliver(adapter(graph),
                      to=["client@example.com", "client@example.com"])
    assert receipt.to == ["client@example.com"]
    assert len(graph.sent_payloads()[0]["message"]["toRecipients"]) == 1


def test_an_unreadable_approved_document_refuses_the_whole_send(graph):
    """Better an operator sees an error than a client sees half a pack."""
    def broken(_ref):
        raise OSError("blob unavailable")

    with pytest.raises(MailDeliveryRefused) as caught:
        deliver(adapter(graph, resolver=broken))
    assert "onboarding_pack.md" in caught.value.message
    assert graph.send_count == 0


def test_an_empty_approved_document_refuses_the_send(graph):
    with pytest.raises(MailDeliveryRefused):
        deliver(adapter(graph, resolver=lambda _ref: b""))
    assert graph.send_count == 0


def test_documents_too_large_for_one_message_refuse_the_send(graph):
    """Measured on the base64 payload, which is what Graph's limit counts."""
    huge = [{"name": "tape.csv", "ref": "blob://synthetic/tape.csv"}]
    with pytest.raises(MailDeliveryRefused):
        deliver(adapter(graph, resolver=lambda _ref: b"x" * (3 * 1024 * 1024)),
                artefacts=huge)
    assert graph.send_count == 0


def test_an_artefact_that_is_not_a_pack_is_attached_unchanged(graph):
    """Only the two known presentation artefacts are special-cased."""
    extra = PACK + [{"name": "sample.csv", "ref": "blob://synthetic/s.csv"}]
    store = dict(STORED, **{"blob://synthetic/s.csv": b"id,balance\n1,100\n"})
    deliver(adapter(graph, resolver=store.get), artefacts=extra)
    attachments = graph.sent_payloads()[0]["message"]["attachments"]
    assert [a["name"] for a in attachments] == [
        "Trakt_Onboarding_Checklist_Northstar_Lending.pdf", "sample.csv"]
    assert base64.b64decode(attachments[1]["contentBytes"]) == \
        b"id,balance\n1,100\n"


def test_more_documents_than_a_pack_can_carry_is_refused(graph):
    many = [{"name": f"f{n}.md", "ref": f"blob://synthetic/{n}"}
            for n in range(MAX_ATTACHMENTS + 1)]
    with pytest.raises(MailDeliveryRefused):
        deliver(adapter(graph, resolver=lambda _ref: b"data"), artefacts=many)
    assert graph.send_count == 0


def test_an_artefact_with_no_reference_is_refused(graph):
    with pytest.raises(MailDeliveryRefused):
        deliver(adapter(graph), artefacts=[{"name": "pack.md", "ref": ""}])
    assert graph.send_count == 0


def test_a_graph_failure_leaves_nothing_claimed_as_sent(graph):
    graph.send_error = 500
    with pytest.raises(MailDeliveryRefused) as caught:
        deliver(adapter(graph))
    assert "still approved" in caught.value.message
    assert CLIENT_SECRET not in caught.value.message


# --------------------------------------------------------------------------- #
# Construction is fail-closed
# --------------------------------------------------------------------------- #

def test_an_unconfigured_deployment_gets_no_adapter_at_all():
    """`None` is the fail-closed answer: the caller keeps RecordOnlyAdapter,
    which reports honestly that nothing was sent."""
    assert outbound_adapter(None, env={}) is None
    assert outbound_adapter(None, env=mail_env(TRAKT_MAIL_MAILBOX="")) is None
    assert outbound_adapter(
        None, env=mail_env(TRAKT_MAIL_OUTBOUND_ENABLED="false")) is None


def test_from_env_refuses_loudly_when_asked_directly():
    with pytest.raises(MailNotConfigured):
        from_env(env={})


def test_a_configured_deployment_gets_an_adapter_wired_to_storage():
    class FakeStorage:
        def read_bytes(self, uri):
            return b"bytes for " + uri.encode()

    built = outbound_adapter(FakeStorage(), env=mail_env())
    assert built is not None
    assert built.name == ADAPTER_NAME
    assert built.resolver("blob://x") == b"bytes for blob://x"
