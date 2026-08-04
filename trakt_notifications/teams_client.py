#!/usr/bin/env python3
"""Bot Framework transport for a proactive personal message.

Deliberately a direct REST client over ``urllib`` rather than the Bot Framework
SDK. The SDK would pull a large dependency tree into an Azure Function whose
only job here is to POST one activity to a known conversation, and the proactive
flow is two documented HTTP calls:

  1. a client-credentials token from the Bot Framework login endpoint, scoped to
     ``https://api.botframework.com/.default``;
  2. ``POST {serviceUrl}/v3/conversations/{conversationId}/activities`` carrying
     the Adaptive Card attachment.

``transport`` is injectable so the whole delivery path is testable without a
network, and so a deployment that must route through a proxy can supply one.

Failures are classified rather than raised uniformly, because the outbox needs
to know the difference: a 429 or a 5xx is worth retrying, a 403 from a user who
uninstalled the app is not, and retrying the latter forever would hide a real
operator action behind a permanently growing attempt count.
"""

from __future__ import annotations

import json
import logging
import time
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass
from typing import Any, Callable, Dict, Optional

logger = logging.getLogger("trakt_notifications.teams")

LOGIN_URL = ("https://login.microsoftonline.com/botframework.com/oauth2/v2.0/"
             "token")
BOT_SCOPE = "https://api.botframework.com/.default"

#: Service URLs a proactive send may target. Bot Framework supplies the service
#: URL on the inbound activity; pinning the acceptable hosts stops a tampered
#: or stale conversation reference redirecting a card — and the credentials
#: used to send it — to an arbitrary endpoint.
ALLOWED_SERVICE_HOSTS = (
    ".botframework.com",
    ".azure.com",
    ".azure.us",
)


class TeamsSendError(Exception):
    """A send failed. ``retryable`` decides what the outbox does next."""

    def __init__(self, message: str, *, retryable: bool,
                 status: Optional[int] = None):
        super().__init__(message)
        self.retryable = retryable
        self.status = status


@dataclass
class SendResult:
    """What came back from a successful send."""

    teams_message_id: Optional[str]
    status: int


def _default_transport(request: urllib.request.Request,
                       timeout: float) -> "tuple[int, bytes]":
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return response.status, response.read()


Transport = Callable[[urllib.request.Request, float], "tuple[int, bytes]"]


def _is_allowed_service_url(service_url: str) -> bool:
    try:
        parsed = urllib.parse.urlparse(service_url)
    except ValueError:
        return False
    if parsed.scheme != "https" or not parsed.hostname:
        return False
    host = parsed.hostname.lower()
    return any(host == suffix.lstrip(".") or host.endswith(suffix)
               for suffix in ALLOWED_SERVICE_HOSTS)


def _retryable_status(status: int) -> bool:
    """429 and 5xx are the transport's way of saying "later"."""
    return status == 429 or 500 <= status < 600


class TeamsClient:
    """Proactive sends to a captured Teams conversation."""

    def __init__(self, *, app_id: Optional[str], app_password: Optional[str],
                 transport: Optional[Transport] = None,
                 timeout: float = 15.0):
        self._app_id = app_id
        self._app_password = app_password
        self._transport = transport or _default_transport
        self._timeout = timeout
        self._token: Optional[str] = None
        self._token_expiry: float = 0.0

    @property
    def configured(self) -> bool:
        return bool(self._app_id and self._app_password)

    # -- authentication ---------------------------------------------------- #
    def _access_token(self) -> str:
        """A cached client-credentials token for the bot identity.

        Cached with a safety margin rather than to the exact expiry: a token
        that expires between the check and the request would surface as a
        spurious 401 and burn a retry attempt.
        """
        if not self.configured:
            raise TeamsSendError(
                "the Teams bot identity is not configured for this deployment",
                retryable=False)
        now = time.time()
        if self._token and now < self._token_expiry:
            return self._token

        payload = urllib.parse.urlencode({
            "grant_type": "client_credentials",
            "client_id": self._app_id,
            "client_secret": self._app_password,
            "scope": BOT_SCOPE,
        }).encode("utf-8")
        request = urllib.request.Request(
            LOGIN_URL, data=payload, method="POST",
            headers={"Content-Type": "application/x-www-form-urlencoded"})
        try:
            status, body = self._transport(request, self._timeout)
        except urllib.error.HTTPError as exc:
            # A bad secret is not a transient condition; retrying it just
            # delays the operator finding out.
            raise TeamsSendError(
                f"bot token request failed with {exc.code}",
                retryable=_retryable_status(exc.code),
                status=exc.code) from None
        except Exception as exc:  # noqa: BLE001 - network class
            raise TeamsSendError(f"bot token request failed: {exc}",
                                 retryable=True) from None

        if status != 200:
            raise TeamsSendError(f"bot token request returned {status}",
                                 retryable=_retryable_status(status),
                                 status=status)
        data = json.loads(body.decode("utf-8"))
        token = data.get("access_token")
        if not token:
            raise TeamsSendError("bot token response carried no access token",
                                 retryable=False)
        # 60s margin, and never negative.
        self._token = token
        self._token_expiry = now + max(int(data.get("expires_in") or 3600) - 60, 0)
        return token

    # -- send -------------------------------------------------------------- #
    def send_card(self, *, service_url: str, conversation_id: str,
                  attachment: Dict[str, Any],
                  summary: str) -> SendResult:
        """POST one Adaptive Card activity to an existing personal conversation."""
        if not service_url or not conversation_id:
            raise TeamsSendError(
                "no Teams conversation reference is available for this recipient",
                retryable=False)
        if not _is_allowed_service_url(service_url):
            raise TeamsSendError(
                "the recorded Teams service URL is not an accepted Bot "
                "Framework endpoint", retryable=False)

        token = self._access_token()
        url = (f"{service_url.rstrip('/')}/v3/conversations/"
               f"{urllib.parse.quote(conversation_id, safe='')}/activities")
        activity = {
            "type": "message",
            # ``summary`` drives the toast and activity-feed preview, where the
            # card itself is not rendered.
            "text": summary,
            "summary": summary,
            "attachments": [attachment],
        }
        request = urllib.request.Request(
            url, data=json.dumps(activity).encode("utf-8"), method="POST",
            headers={"Content-Type": "application/json",
                     "Authorization": f"Bearer {token}"})
        try:
            status, body = self._transport(request, self._timeout)
        except urllib.error.HTTPError as exc:
            if exc.code == 401:
                # The cached token may simply have been revoked; drop it so the
                # next attempt fetches a fresh one.
                self._token = None
                self._token_expiry = 0.0
            raise TeamsSendError(
                f"Teams rejected the message with {exc.code}",
                retryable=_retryable_status(exc.code) or exc.code == 401,
                status=exc.code) from None
        except Exception as exc:  # noqa: BLE001 - network class
            raise TeamsSendError(f"Teams send failed: {exc}",
                                 retryable=True) from None

        if status >= 400:
            raise TeamsSendError(f"Teams send returned {status}",
                                 retryable=_retryable_status(status),
                                 status=status)
        message_id = None
        try:
            message_id = (json.loads(body.decode("utf-8")) or {}).get("id")
        except Exception:  # noqa: BLE001 - an unparseable success body is not
            pass          # a failure; the message went out.
        return SendResult(teams_message_id=message_id, status=status)


def from_config(transport: Optional[Transport] = None) -> TeamsClient:
    """A client built from the deployment's bot credentials."""
    from .config import bot_credentials
    creds = bot_credentials()
    return TeamsClient(app_id=creds["app_id"],
                       app_password=creds["app_password"],
                       transport=transport)
