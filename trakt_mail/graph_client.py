"""trakt_mail.graph_client — a Microsoft Graph mail transport over ``urllib``.

Deliberately a direct REST client rather than the Graph SDK, for the same
reason :mod:`trakt_notifications.teams_client` is: the SDK would pull a large
dependency tree into a service whose only job here is two documented HTTP
calls, and this way ``transport`` stays injectable so every path — success,
throttling, a revoked secret, a mailbox the application may not touch — is
testable without a network.

The two calls:

  1. a client-credentials token from
     ``https://login.microsoftonline.com/{tenant}/oauth2/v2.0/token``, scoped
     to ``https://graph.microsoft.com/.default``;
  2. ``POST /users/{mailbox}/sendMail``.

Why a third call exists
-----------------------
``sendMail`` answers **202 Accepted with an empty body**. It returns no message
identifier, so a receipt built from it alone could only say "Graph accepted
it", which is not evidence that anything is in the mailbox. The message
therefore carries a unique ``x-trakt-receipt-id`` internet header, and after a
successful send Sent Items is read back looking for it. That yields the real
``internetMessageId``, the Graph message id and the conversation id — the
identifiers an auditor can follow from the audit chain to the message itself.

The read-back is **best effort and never fails the send**. The mail has gone;
a slow Sent Items index, or a deployment holding ``Mail.Send`` without
``Mail.Read``, must not turn a delivered message into a reported failure. When
it does not resolve, :class:`SentMessage` says ``confirmed = False`` and the
correlation id is still on the message, so it remains findable by hand.

Failures are classified rather than raised uniformly. A 429 or 5xx is worth
another attempt; a 403 from an Application Access Policy that does not cover
this mailbox is not, and retrying it would hide a real configuration error
behind a growing attempt count.
"""

from __future__ import annotations

import json
import logging
import time
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional

logger = logging.getLogger("trakt_mail.graph")

LOGIN_HOST = "login.microsoftonline.com"
GRAPH_HOST = "graph.microsoft.com"
GRAPH_ROOT = f"https://{GRAPH_HOST}/v1.0"
GRAPH_SCOPE = f"https://{GRAPH_HOST}/.default"

#: The header that ties an audit record to the message in the mailbox. Custom
#: internet headers must begin with ``x-``; Graph rejects anything else.
RECEIPT_HEADER = "x-trakt-receipt-id"

#: How many Sent Items entries are examined per confirmation attempt. The
#: message just sent is the newest, so a small window is sufficient and keeps
#: the response small.
CONFIRM_WINDOW = 15

Transport = Callable[[urllib.request.Request, float], "tuple[int, bytes]"]


class GraphMailError(Exception):
    """A Graph call failed. ``retryable`` decides whether trying again helps."""

    def __init__(self, message: str, *, retryable: bool,
                 status: Optional[int] = None):
        super().__init__(message)
        self.retryable = retryable
        self.status = status


@dataclass
class SentMessage:
    """What is known about one sent message.

    ``confirmed`` is the honest answer to "did we find it in the mailbox?".
    ``sent`` is not a field here because reaching this object at all means
    Graph accepted the message; the caller records that.
    """

    receipt_id: str
    status: int
    internet_message_id: str = ""
    graph_message_id: str = ""
    conversation_id: str = ""
    confirmed: bool = False
    confirmation_note: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {"receipt_id": self.receipt_id, "status": self.status,
                "internet_message_id": self.internet_message_id,
                "graph_message_id": self.graph_message_id,
                "conversation_id": self.conversation_id,
                "confirmed": self.confirmed,
                "confirmation_note": self.confirmation_note}


def _default_transport(request: urllib.request.Request,
                       timeout: float) -> "tuple[int, bytes]":
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return response.status, response.read()


def _retryable_status(status: int) -> bool:
    """429 and 5xx are the service saying "later"; 4xx is it saying "no"."""
    return status == 429 or 500 <= status < 600


def _assert_graph_url(url: str) -> None:
    """Pin the host a bearer token may be sent to.

    Every URL in this module is built from constants, so this can only fail if
    someone later assembles one from configuration. That is exactly when it
    should fail — the same reasoning as ``ALLOWED_SERVICE_HOSTS`` in the Teams
    client.
    """
    parsed = urllib.parse.urlparse(url)
    if parsed.scheme != "https" or (parsed.hostname or "").lower() != GRAPH_HOST:
        raise GraphMailError(
            "refusing to send a Graph credential to a non-Graph endpoint",
            retryable=False)


class GraphMailClient:
    """Send mail as one configured mailbox, and read back what was sent."""

    def __init__(self, *, tenant_id: Optional[str], client_id: Optional[str],
                 client_secret: Optional[str], mailbox: Optional[str],
                 transport: Optional[Transport] = None,
                 timeout: float = 20.0,
                 confirm_attempts: int = 3,
                 confirm_delay: float = 2.0,
                 sleep: Optional[Callable[[float], None]] = None):
        self._tenant_id = tenant_id
        self._client_id = client_id
        self._client_secret = client_secret
        self.mailbox = mailbox or ""
        self._transport = transport or _default_transport
        self._timeout = timeout
        self._confirm_attempts = max(int(confirm_attempts), 0)
        self._confirm_delay = max(float(confirm_delay), 0.0)
        self._sleep = sleep or time.sleep
        self._token: Optional[str] = None
        self._token_expiry: float = 0.0

    @property
    def configured(self) -> bool:
        return bool(self._tenant_id and self._client_id
                    and self._client_secret and self.mailbox)

    # -- authentication ---------------------------------------------------- #
    def _access_token(self) -> str:
        """A cached client-credentials token for the Trakt OCC Mail identity.

        Cached with a safety margin rather than to the exact expiry: a token
        that expired between the check and the request would surface as a
        spurious 401 and burn an attempt.
        """
        if not self.configured:
            raise GraphMailError(
                "outbound mail is not configured for this deployment",
                retryable=False)
        now = time.time()
        if self._token and now < self._token_expiry:
            return self._token

        url = (f"https://{LOGIN_HOST}/"
               f"{urllib.parse.quote(str(self._tenant_id), safe='')}"
               f"/oauth2/v2.0/token")
        payload = urllib.parse.urlencode({
            "grant_type": "client_credentials",
            "client_id": self._client_id,
            "client_secret": self._client_secret,
            "scope": GRAPH_SCOPE,
        }).encode("utf-8")
        request = urllib.request.Request(
            url, data=payload, method="POST",
            headers={"Content-Type": "application/x-www-form-urlencoded"})
        try:
            status, body = self._transport(request, self._timeout)
        except urllib.error.HTTPError as exc:
            # A rejected secret is not transient. Retrying only delays the
            # administrator finding out the credential is wrong or expired.
            raise GraphMailError(f"Graph token request failed with {exc.code}",
                                 retryable=_retryable_status(exc.code),
                                 status=exc.code) from None
        except Exception as exc:  # noqa: BLE001 - network class
            raise GraphMailError(f"Graph token request failed: {exc}",
                                 retryable=True) from None

        if status != 200:
            raise GraphMailError(f"Graph token request returned {status}",
                                 retryable=_retryable_status(status),
                                 status=status)
        try:
            data = json.loads(body.decode("utf-8"))
        except Exception:  # noqa: BLE001
            raise GraphMailError("Graph token response was not readable",
                                 retryable=False) from None
        token = data.get("access_token")
        if not token:
            raise GraphMailError("Graph token response carried no access token",
                                 retryable=False)
        self._token = token
        self._token_expiry = now + max(int(data.get("expires_in") or 3600) - 60, 0)
        return token

    # -- requests ---------------------------------------------------------- #
    def _call(self, method: str, url: str, *,
              body: Optional[Dict[str, Any]] = None
              ) -> "tuple[int, Dict[str, Any]]":
        _assert_graph_url(url)
        token = self._access_token()
        data = json.dumps(body).encode("utf-8") if body is not None else None
        headers = {"Authorization": f"Bearer {token}",
                   "Accept": "application/json"}
        if data is not None:
            headers["Content-Type"] = "application/json"
        request = urllib.request.Request(url, data=data, method=method,
                                         headers=headers)
        try:
            status, raw = self._transport(request, self._timeout)
        except urllib.error.HTTPError as exc:
            if exc.code == 401:
                # The cached token may have been revoked; drop it so a retry
                # fetches a fresh one rather than replaying a dead credential.
                self._token = None
                self._token_expiry = 0.0
            raise GraphMailError(f"Graph rejected the request with {exc.code}",
                                 retryable=_retryable_status(exc.code)
                                 or exc.code == 401,
                                 status=exc.code) from None
        except Exception as exc:  # noqa: BLE001 - network class
            raise GraphMailError(f"the Graph request failed: {exc}",
                                 retryable=True) from None

        if status >= 400:
            raise GraphMailError(f"Graph returned {status}",
                                 retryable=_retryable_status(status),
                                 status=status)
        if not raw:
            return status, {}
        try:
            parsed = json.loads(raw.decode("utf-8"))
        except Exception:  # noqa: BLE001 — an unreadable success body is not a
            return status, {}      # failure; the call succeeded.
        return status, parsed if isinstance(parsed, dict) else {}

    def _mailbox_path(self, *segments: str) -> str:
        parts = [urllib.parse.quote(self.mailbox, safe="@.")]
        parts.extend(urllib.parse.quote(str(s), safe="") for s in segments)
        return f"{GRAPH_ROOT}/users/" + "/".join(parts)

    # -- send -------------------------------------------------------------- #
    def send(self, *, to: List[str], subject: str, body: str,
             receipt_id: str,
             attachments: Optional[List[Dict[str, Any]]] = None,
             body_type: str = "Text") -> SentMessage:
        """Send one message as the configured mailbox.

        ``receipt_id`` is carried as the ``x-trakt-receipt-id`` internet header
        so the audit record and the message in the mailbox name each other.
        """
        if not to:
            raise GraphMailError("a message must have at least one recipient",
                                 retryable=False)
        message: Dict[str, Any] = {
            "subject": subject,
            "body": {"contentType": body_type, "content": body},
            "toRecipients": [{"emailAddress": {"address": address}}
                             for address in to],
            "internetMessageHeaders": [
                {"name": RECEIPT_HEADER, "value": receipt_id},
            ],
        }
        if attachments:
            message["attachments"] = list(attachments)

        status, _ = self._call(
            "POST", self._mailbox_path("sendMail"),
            # saveToSentItems is what makes the message auditable in the
            # mailbox afterwards, and what the confirmation below reads.
            body={"message": message, "saveToSentItems": True})

        sent = SentMessage(receipt_id=receipt_id, status=status)
        self._confirm(sent, subject)
        return sent

    # -- confirmation (best effort, never fatal) --------------------------- #
    def _confirm(self, sent: SentMessage, subject: str) -> None:
        """Find the message just sent in Sent Items and record its identifiers.

        Every failure mode here is swallowed into ``confirmation_note``: the
        message has already gone, and reporting a delivered message as failed
        would be worse than reporting it as unconfirmed.
        """
        if self._confirm_attempts <= 0:
            sent.confirmation_note = "confirmation disabled by configuration"
            return
        last = "not found in Sent Items"
        for attempt in range(self._confirm_attempts):
            if attempt:
                self._sleep(self._confirm_delay)
            try:
                found = self._find_in_sent_items(sent.receipt_id, subject)
            except GraphMailError as exc:
                # Most likely Mail.Read is absent or not covered by the
                # Application Access Policy. Worth stating exactly once.
                last = f"could not read Sent Items ({exc})"
                logger.info("mail: %s", last)
                break
            except Exception as exc:  # noqa: BLE001 — confirmation is optional
                last = f"could not read Sent Items ({exc})"
                logger.info("mail: %s", last)
                break
            if found is not None:
                sent.graph_message_id = str(found.get("id") or "")
                sent.internet_message_id = str(
                    found.get("internetMessageId") or "")
                sent.conversation_id = str(found.get("conversationId") or "")
                sent.confirmed = True
                sent.confirmation_note = "found in Sent Items"
                return
        sent.confirmation_note = last

    def _find_in_sent_items(self, receipt_id: str,
                            subject: str) -> Optional[Dict[str, Any]]:
        query = urllib.parse.urlencode({
            "$top": str(CONFIRM_WINDOW),
            "$orderby": "sentDateTime desc",
            "$select": ("id,internetMessageId,conversationId,subject,"
                        "sentDateTime,internetMessageHeaders"),
        })
        url = f"{self._mailbox_path('mailFolders', 'sentitems', 'messages')}?{query}"
        _status, payload = self._call("GET", url)
        for item in payload.get("value") or []:
            if not isinstance(item, dict):
                continue
            if _carries_receipt(item, receipt_id):
                return item
            # A deployment whose $select does not return headers on the
            # collection still resolves, by reading the one candidate whose
            # subject matches rather than every message in the window.
            if ("internetMessageHeaders" not in item
                    and str(item.get("subject") or "") == subject):
                detail = self._read_message(str(item.get("id") or ""))
                if detail and _carries_receipt(detail, receipt_id):
                    return {**item, **detail}
        return None

    def _read_message(self, message_id: str) -> Optional[Dict[str, Any]]:
        if not message_id:
            return None
        query = urllib.parse.urlencode({
            "$select": "id,internetMessageId,conversationId,internetMessageHeaders",
        })
        url = f"{self._mailbox_path('messages', message_id)}?{query}"
        _status, payload = self._call("GET", url)
        return payload or None


def _carries_receipt(message: Dict[str, Any], receipt_id: str) -> bool:
    for header in message.get("internetMessageHeaders") or []:
        if not isinstance(header, dict):
            continue
        if (str(header.get("name") or "").lower() == RECEIPT_HEADER
                and str(header.get("value") or "") == receipt_id):
            return True
    return False


def from_config(config: Any, *,
                transport: Optional[Transport] = None) -> GraphMailClient:
    """A client built from a :class:`trakt_mail.config.MailConfig`."""
    return GraphMailClient(
        tenant_id=config.tenant_id, client_id=config.client_id,
        client_secret=config.client_secret, mailbox=config.mailbox,
        transport=transport, timeout=config.timeout,
        confirm_attempts=config.confirm_attempts,
        confirm_delay=config.confirm_delay)
