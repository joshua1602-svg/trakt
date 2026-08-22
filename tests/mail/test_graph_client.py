"""The Graph transport: what goes over the wire, and what comes back.

The transport is injected rather than mocked at the client boundary, so these
assert on the actual request the deployment would issue — the token endpoint,
the scope, the mailbox in the path, the bearer header, the message body and the
correlation header that makes the send auditable afterwards.
"""

from __future__ import annotations

import json
import urllib.request

import pytest

from trakt_mail.graph_client import (
    GraphMailClient, GraphMailError, RECEIPT_HEADER, _assert_graph_url,
)

from .conftest import CLIENT_ID, CLIENT_SECRET, MAILBOX, TENANT_ID, FakeGraph


def client(graph: FakeGraph, **kwargs) -> GraphMailClient:
    kwargs.setdefault("confirm_attempts", 1)
    kwargs.setdefault("confirm_delay", 0.0)
    kwargs.setdefault("sleep", lambda _seconds: None)
    return GraphMailClient(tenant_id=TENANT_ID, client_id=CLIENT_ID,
                           client_secret=CLIENT_SECRET, mailbox=MAILBOX,
                           transport=graph, **kwargs)


# --------------------------------------------------------------------------- #
# Authentication
# --------------------------------------------------------------------------- #

def test_the_token_is_a_client_credentials_grant_scoped_to_graph(graph):
    client(graph).send(to=["a@example.com"], subject="s", body="b",
                       receipt_id="com_1")
    token_request = graph.requests[0]
    assert token_request.full_url == (
        f"https://login.microsoftonline.com/{TENANT_ID}/oauth2/v2.0/token")
    form = token_request.data.decode("utf-8")
    assert "grant_type=client_credentials" in form
    assert "scope=https%3A%2F%2Fgraph.microsoft.com%2F.default" in form


def test_the_token_is_reused_across_sends(graph):
    graph_client = client(graph)
    graph_client.send(to=["a@example.com"], subject="s", body="b",
                      receipt_id="com_1")
    graph_client.send(to=["a@example.com"], subject="s", body="b",
                      receipt_id="com_2")
    assert graph.token_calls == 1
    assert graph.send_count == 2


def test_an_unconfigured_client_refuses_without_calling_anything(graph):
    bare = GraphMailClient(tenant_id=None, client_id=None, client_secret=None,
                           mailbox=None, transport=graph)
    assert bare.configured is False
    with pytest.raises(GraphMailError) as caught:
        bare.send(to=["a@example.com"], subject="s", body="b",
                  receipt_id="com_1")
    assert caught.value.retryable is False
    assert graph.requests == []


def test_a_rejected_secret_is_not_retryable(graph):
    graph.token_status = 401
    with pytest.raises(GraphMailError) as caught:
        client(graph).send(to=["a@example.com"], subject="s", body="b",
                           receipt_id="com_1")
    assert caught.value.retryable is False


# --------------------------------------------------------------------------- #
# The send
# --------------------------------------------------------------------------- #

def test_the_send_addresses_the_configured_mailbox_and_carries_the_receipt(graph):
    result = client(graph).send(to=["one@example.com", "two@example.com"],
                                subject="Trakt onboarding — Northstar",
                                body="the covering note",
                                receipt_id="com_abc123")
    send = [r for r in graph.requests if r.full_url.endswith("/sendMail")][0]
    assert send.full_url == (
        f"https://graph.microsoft.com/v1.0/users/{MAILBOX}/sendMail")
    assert send.get_header("Authorization") == "Bearer gr@ph-token"

    payload = graph.sent_payloads()[0]
    message = payload["message"]
    assert payload["saveToSentItems"] is True
    assert message["subject"] == "Trakt onboarding — Northstar"
    assert message["body"] == {"contentType": "Text",
                               "content": "the covering note"}
    assert [r["emailAddress"]["address"] for r in message["toRecipients"]] == [
        "one@example.com", "two@example.com"]
    assert {"name": RECEIPT_HEADER, "value": "com_abc123"} in \
        message["internetMessageHeaders"]
    assert result.status == 202


def test_a_send_with_no_recipient_is_refused_before_the_call(graph):
    with pytest.raises(GraphMailError):
        client(graph).send(to=[], subject="s", body="b", receipt_id="com_1")
    assert graph.send_count == 0


def test_throttling_is_retryable_and_a_forbidden_mailbox_is_not(graph):
    graph.send_error = 429
    with pytest.raises(GraphMailError) as throttled:
        client(graph).send(to=["a@example.com"], subject="s", body="b",
                           receipt_id="com_1")
    assert throttled.value.retryable is True

    graph.send_error = 403          # e.g. outside the Application Access Policy
    with pytest.raises(GraphMailError) as forbidden:
        client(graph).send(to=["a@example.com"], subject="s", body="b",
                           receipt_id="com_2")
    assert forbidden.value.retryable is False
    assert forbidden.value.status == 403


# --------------------------------------------------------------------------- #
# The read-back — the message identifier
# --------------------------------------------------------------------------- #

def test_the_message_is_read_back_from_sent_items(graph):
    result = client(graph).send(to=["a@example.com"], subject="s", body="b",
                                receipt_id="com_confirmed")
    assert result.confirmed is True
    assert result.internet_message_id == "<0001@traktinfra.io>"
    assert result.graph_message_id == "AAMkAGI2-graph-message-id"
    assert result.conversation_id == "AAQkAGI2-conversation-id"


def test_a_message_with_a_different_receipt_is_not_claimed(graph):
    """Two sends in the same window must not be confused for one another."""
    graph.echo_sent = False
    graph.sent_items = [{
        "id": "someone-elses",
        "internetMessageId": "<9999@traktinfra.io>",
        "subject": "s",
        "internetMessageHeaders": [{"name": RECEIPT_HEADER,
                                    "value": "com_a_different_send"}],
    }]
    result = client(graph).send(to=["a@example.com"], subject="s", body="b",
                                receipt_id="com_mine")
    assert result.confirmed is False
    assert result.graph_message_id == ""


def test_a_mailbox_that_cannot_be_read_still_reports_the_send(graph):
    """Mail.Read absent, or not covered by the access policy. The message has
    gone; reporting it as a failure would be worse than reporting it unconfirmed."""
    graph.read_error = 403
    result = client(graph).send(to=["a@example.com"], subject="s", body="b",
                                receipt_id="com_1")
    assert result.status == 202
    assert result.confirmed is False
    assert "could not read Sent Items" in result.confirmation_note


def test_confirmation_retries_before_giving_up(graph):
    """Sent Items indexes with a lag, so one empty read is not an answer."""
    graph.echo_sent = False
    slept = []

    class LateGraph(FakeGraph):
        def __call__(self, request, timeout):
            if ("/mailFolders/sentitems/messages" in request.full_url
                    and len(slept) >= 1 and not self.sent_items):
                self.sent_items = [{
                    "id": "late-id",
                    "internetMessageId": "<late@traktinfra.io>",
                    "conversationId": "conv",
                    "subject": "s",
                    "internetMessageHeaders": [
                        {"name": RECEIPT_HEADER, "value": "com_late"}],
                }]
            return super().__call__(request, timeout)

    late = LateGraph(echo_sent=False)
    result = client(late, confirm_attempts=3,
                    sleep=lambda seconds: slept.append(seconds)).send(
        to=["a@example.com"], subject="s", body="b", receipt_id="com_late")
    assert slept, "the second attempt must wait for the mailbox to catch up"
    assert result.confirmed is True
    assert result.internet_message_id == "<late@traktinfra.io>"


# --------------------------------------------------------------------------- #
# Where a bearer token may be sent
# --------------------------------------------------------------------------- #

@pytest.mark.parametrize("url", [
    "http://graph.microsoft.com/v1.0/users/x/sendMail",   # not TLS
    "https://graph.microsoft.com.evil.test/v1.0/x",       # look-alike host
    "https://example.com/v1.0/users/x/sendMail",
])
def test_a_credential_is_never_sent_to_a_non_graph_endpoint(url):
    with pytest.raises(GraphMailError):
        _assert_graph_url(url)


def test_the_real_graph_endpoint_is_accepted():
    _assert_graph_url("https://graph.microsoft.com/v1.0/users/x/sendMail")
