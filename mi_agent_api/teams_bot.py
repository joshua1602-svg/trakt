#!/usr/bin/env python3
"""teams_bot.py — the Bot Framework messaging endpoint.

Its only job is registration. Teams delivers an activity here when the Trakt app
is installed for a user, when they message the bot, and when they remove it;
this endpoint validates that the activity really came from Bot Framework and
records (or clears) the conversation reference that a proactive send needs. It
answers no questions and returns no portfolio data — that is what the existing
Copilot actions and the React workspace are for, and both already authenticate
the caller.

**Why the inbound token is validated rather than trusted.** The conversation
reference and the service URL captured here are used later to send a card
carrying portfolio information, using the bot's own credentials. An unvalidated
endpoint would let anyone POST a conversation reference of their choosing and
have Trakt deliver a client's portfolio position to it. The token is therefore
checked the way :mod:`mi_agent_api.copilot_auth` checks a Copilot token —
signature against the issuer's JWKS, issuer, audience, expiry — and the module
fails closed when it is not configured.

**Registration does not authorise.** A captured recipient has no portfolio
contexts and notifications disabled. An operator maps them to permitted
contexts separately. Installing the app can therefore never, by itself, start a
feed of portfolio data to whoever installed it.

Configuration::

  ``TRAKT_TEAMS_BOT_ENABLED``    "1"/"true" to mount the router at all.
  ``TRAKT_TEAMS_BOT_APP_ID``     the bot's Entra app (client) id — the expected
                                 token audience.
  ``TRAKT_TEAMS_BOT_AUTH_MODE``  ``botframework`` (default, fail closed) |
                                 ``disabled`` (local dev / tests ONLY).
  ``TRAKT_TEAMS_TRAKT_TENANT``   the Trakt tenant this deployment serves.
"""

from __future__ import annotations

import logging
import os
from typing import Any, Dict, Optional

from fastapi import APIRouter, HTTPException, Request, status

logger = logging.getLogger("mi_agent_api.teams_bot")

router = APIRouter(prefix="/v1/teams", tags=["teams-bot"])

ENABLED_ENV = "TRAKT_TEAMS_BOT_ENABLED"
APP_ID_ENV = "TRAKT_TEAMS_BOT_APP_ID"
AUTH_MODE_ENV = "TRAKT_TEAMS_BOT_AUTH_MODE"
TENANT_ENV = "TRAKT_TEAMS_TRAKT_TENANT"

#: Bot Framework's OpenID metadata for channel-issued tokens.
OPENID_METADATA = ("https://login.botframework.com/v1/.well-known/"
                   "openidconfiguration")
BOT_ISSUER = "https://api.botframework.com"

_ON = ("1", "true", "on", "yes", "enabled")

try:  # PyJWT is already a deploy-time dependency for the Copilot routes.
    import jwt as _jwt  # noqa: N816
    from jwt import PyJWKClient as _PyJWKClient
except Exception:  # noqa: BLE001 - absence makes the default mode fail closed
    _jwt = None
    _PyJWKClient = None

_JWKS_CLIENT: Optional[Any] = None


def enabled() -> bool:
    """Whether the bot endpoint is mounted. Off unless explicitly switched on."""
    return (os.environ.get(ENABLED_ENV) or "").strip().lower() in _ON


def _auth_mode() -> str:
    return (os.environ.get(AUTH_MODE_ENV) or "botframework").strip().lower()


def _app_id() -> Optional[str]:
    return (os.environ.get(APP_ID_ENV) or "").strip() or None


def _trakt_tenant() -> Optional[str]:
    return (os.environ.get(TENANT_ENV) or "").strip() or None


def _jwks_client():
    global _JWKS_CLIENT  # noqa: PLW0603 - PyJWKClient caches signing keys
    if _JWKS_CLIENT is None:
        if _PyJWKClient is None:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="bot authentication is unavailable")
        import json
        import urllib.request
        with urllib.request.urlopen(OPENID_METADATA, timeout=10) as response:
            metadata = json.loads(response.read().decode("utf-8"))
        _JWKS_CLIENT = _PyJWKClient(metadata["jwks_uri"])
    return _JWKS_CLIENT


def validate_activity_token(authorization: Optional[str]) -> Dict[str, Any]:
    """Validate the inbound Bot Framework token, or refuse.

    Returns the decoded claims. In ``disabled`` mode — local development and
    the test suite, which set it explicitly — returns a synthetic marker and
    logs a warning, so an accidentally unset variable can never be the reason
    the endpoint runs open.
    """
    if _auth_mode() == "disabled":
        logger.warning("teams bot: token validation DISABLED by configuration")
        return {"synthetic": True}

    app_id = _app_id()
    if _jwt is None or not app_id:
        # Fail closed: an unconfigured deployment refuses every activity rather
        # than accepting one it cannot verify.
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="bot authentication is not configured")

    if not authorization or not authorization.lower().startswith("bearer "):
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED,
                            detail="a bearer token is required")
    token = authorization.split(" ", 1)[1].strip()
    try:
        key = _jwks_client().get_signing_key_from_jwt(token).key
        return _jwt.decode(token, key, algorithms=["RS256"],
                           audience=app_id, issuer=BOT_ISSUER)
    except HTTPException:
        raise
    except Exception as exc:  # noqa: BLE001 - never echo token contents
        logger.warning("teams bot: token rejected (%s)", type(exc).__name__)
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED,
                            detail="the activity token could not be validated"
                            ) from None


# --------------------------------------------------------------------------- #
# Activity handling
# --------------------------------------------------------------------------- #
def conversation_reference(activity: Dict[str, Any]) -> Dict[str, Any]:
    """The Bot Framework conversation reference for a proactive send.

    Built from the activity's own fields in the shape the SDK would produce, so
    a later migration to the SDK can consume a stored reference unchanged.
    """
    return {
        "activityId": activity.get("id"),
        "user": activity.get("from") or {},
        "bot": activity.get("recipient") or {},
        "conversation": activity.get("conversation") or {},
        "channelId": activity.get("channelId"),
        "serviceUrl": activity.get("serviceUrl"),
        "locale": activity.get("locale"),
    }


def _microsoft_tenant(activity: Dict[str, Any]) -> Optional[str]:
    """The Entra tenant, from the channel data the service populates."""
    channel = activity.get("channelData") or {}
    tenant = channel.get("tenant") or {}
    return tenant.get("id") or (activity.get("conversation") or {}).get("tenantId")


def _entra_object_id(activity: Dict[str, Any]) -> Optional[str]:
    """The user's stable object id.

    ``aadObjectId`` is the directory identity and is what an operator maps.
    The channel-scoped ``from.id`` is a fallback for a channel that does not
    supply one; a display name is never used, because it is user-controlled in
    most directories.
    """
    sender = activity.get("from") or {}
    return sender.get("aadObjectId") or sender.get("id")


def handle_activity(activity: Dict[str, Any], *,
                    tenant_id: Optional[str] = None) -> Dict[str, Any]:
    """Register, update or clear a recipient from one inbound activity.

    Pure with respect to HTTP so the whole registration path is testable
    without a server. Returns an operator-facing summary only — never the
    conversation reference it stored.
    """
    from apps.blob_trigger_app.storage import open_storage
    from trakt_notifications.recipients import RecipientStore, recipient_id

    trakt_tenant = tenant_id or _trakt_tenant()
    if not trakt_tenant:
        # Without a governed tenant mapping there is nothing to attach the
        # recipient to, and guessing one would be the cross-tenant defect.
        return {"registered": False,
                "reason": "no Trakt tenant is configured for this deployment"}

    microsoft_tenant = _microsoft_tenant(activity)
    object_id = _entra_object_id(activity)
    if not microsoft_tenant or not object_id:
        return {"registered": False,
                "reason": "the activity carried no usable directory identity"}

    store = RecipientStore(open_storage())
    rid = recipient_id(microsoft_tenant, object_id)
    activity_type = str(activity.get("type") or "")

    # Removal — the app was uninstalled. Disable delivery, keep the record.
    if activity_type == "installationUpdate" \
            and str(activity.get("action") or "") in ("remove", "remove-upgrade"):
        store.mark_uninstalled(trakt_tenant, rid)
        return {"registered": False, "recipient_id": rid,
                "reason": "the Trakt app was removed for this user"}
    removed_members = (activity.get("membersRemoved") or [])
    if activity_type == "conversationUpdate" and any(
            (m or {}).get("id") == ((activity.get("recipient") or {}).get("id"))
            for m in removed_members):
        store.mark_uninstalled(trakt_tenant, rid)
        return {"registered": False, "recipient_id": rid,
                "reason": "the Trakt app was removed for this user"}

    conversation = activity.get("conversation") or {}
    conversation_id = conversation.get("id")
    service_url = activity.get("serviceUrl")
    if not conversation_id or not service_url:
        return {"registered": False,
                "reason": "the activity carried no addressable conversation"}

    # v1 delivers to personal chats only. A channel or group-chat activity is
    # acknowledged and ignored rather than stored as a destination.
    if str(conversation.get("conversationType") or "personal") != "personal":
        return {"registered": False, "recipient_id": rid,
                "reason": "only personal (1:1) chats are supported"}

    sender = activity.get("from") or {}
    recipient = store.capture_conversation(
        tenant_id=trakt_tenant,
        microsoft_tenant_id=microsoft_tenant,
        entra_object_id=object_id,
        conversation_id=conversation_id,
        service_url=service_url,
        conversation_reference=conversation_reference(activity),
        display_name=sender.get("name"),
        user_principal_name=sender.get("userPrincipalName"))

    return {
        "registered": True,
        "recipient_id": recipient.recipient_id,
        # Stated in the response so an operator inspecting the registration can
        # see that capture alone has not authorised anything.
        "notifications_enabled": recipient.notifications_enabled,
        "portfolio_contexts": list(recipient.portfolio_contexts),
        "next_step": ("an operator must map this recipient to permitted "
                      "portfolio contexts before any message is delivered"),
    }


@router.post("/bot/messages")
async def messages(request: Request) -> Dict[str, Any]:
    """The Bot Framework messaging endpoint.

    Always returns 200 for a validated activity, including one it chose not to
    act on: Bot Framework retries non-2xx responses, and retrying an activity
    that was correctly ignored achieves nothing but noise.
    """
    if not enabled():
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND,
                            detail="the Teams bot endpoint is not enabled")
    validate_activity_token(request.headers.get("authorization"))
    try:
        activity = await request.json()
    except Exception:  # noqa: BLE001
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST,
                            detail="the activity body could not be read"
                            ) from None
    if not isinstance(activity, dict):
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST,
                            detail="the activity body was not an object")

    result = handle_activity(activity)
    logger.info("teams bot: activity=%s registered=%s reason=%s",
                activity.get("type"), result.get("registered"),
                result.get("reason", ""))
    return result
