"""Fixtures for the outbound Graph mail suite.

Nothing here reaches a network and nothing reads a real credential. The Graph
transport is injected, so the client's own request construction — the URL, the
scope, the bearer header, the message body, the correlation header — is what is
asserted on, rather than a mock of the client.
"""

from __future__ import annotations

import json
import sys
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any, Dict, List, Optional

import pytest

_REPO = Path(__file__).resolve().parents[2]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

TENANT_ID = "22222222-2222-2222-2222-222222222222"
CLIENT_ID = "33333333-3333-3333-3333-333333333333"
CLIENT_SECRET = "a-secret-that-must-never-be-logged"
MAILBOX = "admin@traktinfra.io"

TOKEN_BODY = json.dumps({"access_token": "gr@ph-token",
                         "expires_in": 3600}).encode("utf-8")


def mail_env(**overrides: str) -> Dict[str, str]:
    """A complete, enabled outbound-mail environment, with overrides."""
    env = {
        "TRAKT_MAIL_OUTBOUND_ENABLED": "true",
        "TRAKT_MAIL_TENANT_ID": TENANT_ID,
        "TRAKT_MAIL_CLIENT_ID": CLIENT_ID,
        "TRAKT_MAIL_CLIENT_SECRET": CLIENT_SECRET,
        "TRAKT_MAIL_MAILBOX": MAILBOX,
        # Confirmation is exercised explicitly where it matters; the default of
        # one attempt with no delay keeps every other test instant.
        "TRAKT_MAIL_CONFIRM_ATTEMPTS": "1",
        "TRAKT_MAIL_CONFIRM_DELAY_SECONDS": "0.01",
    }
    env.update(overrides)
    return env


class FakeGraph:
    """A scripted Microsoft Graph.

    Records every request so a test can assert on what would actually have gone
    over the wire, and answers by URL shape rather than by call order, so a test
    that adds a confirmation read does not have to re-script the token call.
    """

    def __init__(self, *, sent_items: Optional[List[Dict[str, Any]]] = None,
                 token_status: int = 200,
                 send_status: int = 202,
                 send_error: Optional[int] = None,
                 read_error: Optional[int] = None,
                 echo_sent: bool = True,
                 inbox: Optional[List[Dict[str, Any]]] = None,
                 attachments: Optional[Dict[str, List[Dict[str, Any]]]] = None,
                 inbox_error: Optional[int] = None):
        self.requests: List[urllib.request.Request] = []
        self.bodies: List[Any] = []
        self.sent_items = list(sent_items or [])
        #: What the reader finds in the folder it is pointed at.
        self.inbox = list(inbox or [])
        #: Keyed by the message id the inbox entry declares.
        self.attachments = dict(attachments or {})
        self.inbox_error = inbox_error
        #: Every message the reader marked read, in order.
        self.marked_read: List[str] = []
        self.token_status = token_status
        self.send_status = send_status
        self.send_error = send_error
        self.read_error = read_error
        #: When true, a successful send appends a matching Sent Items entry, so
        #: the confirmation read finds what a real mailbox would.
        self.echo_sent = echo_sent
        self.token_calls = 0

    # -- the transport ----------------------------------------------------- #
    def __call__(self, request: urllib.request.Request,
                 timeout: float) -> "tuple[int, bytes]":
        self.requests.append(request)
        body = None
        if request.data:
            try:
                body = json.loads(request.data.decode("utf-8"))
            except Exception:  # noqa: BLE001 — the token call is form-encoded
                body = request.data.decode("utf-8")
        self.bodies.append(body)
        url = request.full_url

        if "login.microsoftonline.com" in url:
            self.token_calls += 1
            if self.token_status != 200:
                raise urllib.error.HTTPError(url, self.token_status, "no", {},
                                             None)
            return 200, TOKEN_BODY

        if url.endswith("/sendMail"):
            if self.send_error:
                raise urllib.error.HTTPError(url, self.send_error, "no", {},
                                             None)
            if self.echo_sent:
                self.sent_items.insert(0, _sent_entry(body))
            return self.send_status, b""

        if "/attachments" in url:
            message_id = url.split("/messages/")[-1].split("/")[0]
            return 200, json.dumps(
                {"value": self.attachments.get(message_id, [])}
            ).encode("utf-8")

        if "/mailFolders/sentitems/messages" in url:
            if self.read_error:
                raise urllib.error.HTTPError(url, self.read_error, "no", {},
                                             None)
            return 200, json.dumps({"value": self.sent_items}).encode("utf-8")

        if "/mailFolders/" in url and "/messages" in url:
            if self.inbox_error:
                raise urllib.error.HTTPError(url, self.inbox_error, "no", {},
                                             None)
            return 200, json.dumps({"value": self.inbox}).encode("utf-8")

        if "/messages/" in url:
            if request.get_method() == "PATCH":
                self.marked_read.append(url.split("/messages/")[-1])
                return 200, b"{}"
            if self.read_error:
                raise urllib.error.HTTPError(url, self.read_error, "no", {},
                                             None)
            return 200, json.dumps({"value": self.sent_items}).encode("utf-8")

        raise AssertionError(f"unexpected Graph call: {url}")

    # -- assertions helpers ------------------------------------------------ #
    def sent_payloads(self) -> List[Dict[str, Any]]:
        return [b for r, b in zip(self.requests, self.bodies)
                if r.full_url.endswith("/sendMail")]

    @property
    def send_count(self) -> int:
        return len(self.sent_payloads())


def _sent_entry(send_body: Any) -> Dict[str, Any]:
    """The Sent Items record a real mailbox would hold after this send."""
    message = (send_body or {}).get("message") or {}
    return {
        "id": "AAMkAGI2-graph-message-id",
        "internetMessageId": "<0001@traktinfra.io>",
        "conversationId": "AAQkAGI2-conversation-id",
        "subject": message.get("subject"),
        "sentDateTime": "2026-08-19T09:00:00Z",
        "internetMessageHeaders": list(
            message.get("internetMessageHeaders") or []),
    }


#: A complete, enabled INBOUND environment. Separate from ``mail_env`` because
#: the two switches are separate, and a test that turns one on must not get the
#: other for free.
def inbound_env(**overrides: str) -> Dict[str, str]:
    env = {
        "TRAKT_MAIL_INBOUND_ENABLED": "true",
        "TRAKT_MAIL_TENANT_ID": TENANT_ID,
        "TRAKT_MAIL_CLIENT_ID": CLIENT_ID,
        "TRAKT_MAIL_CLIENT_SECRET": CLIENT_SECRET,
        "TRAKT_MAIL_MAILBOX": "onboarding@traktinfra.io",
    }
    env.update(overrides)
    return env


@pytest.fixture()
def graph() -> FakeGraph:
    return FakeGraph()
