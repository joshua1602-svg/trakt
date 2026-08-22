"""trakt_mail.config — the deployment's Graph mail identity, or nothing at all.

Fail-closed is the whole design. There is no default mailbox, no default
tenant, and no way to obtain a partially configured client: :func:`load`
either returns a complete :class:`MailConfig` or raises
:class:`MailNotConfigured` naming the settings that are absent. A deployment
that has not been configured therefore keeps the record-only adapter and says
plainly that nothing was sent — the same honest failure the OCC already has —
rather than half-sending.

Two switches, not one. ``TRAKT_MAIL_OUTBOUND_ENABLED`` is an operator kill
switch independent of the credentials, so sending can be stopped without
removing the identity or redeploying; it follows the same opt-in-token
convention as ``TRAKT_TEAMS_NOTIFICATIONS``
(:mod:`trakt_notifications.config`).

Credentials are read from the environment and never logged. :meth:`MailConfig
.redacted` is what diagnostics print: the tenant and client identifiers are
opaque identifiers rather than secrets, and the secret itself is reported only
as present or absent.
"""

from __future__ import annotations

import os
import re
from dataclasses import dataclass
from typing import Dict, List, Optional

#: Master kill switch. Only an explicit opt-in token enables outbound mail, so
#: an unset or misspelled value leaves the deployment sending nothing.
ENABLED_ENV = "TRAKT_MAIL_OUTBOUND_ENABLED"
_ON = ("1", "true", "on", "yes", "enabled")

TENANT_ENV = "TRAKT_MAIL_TENANT_ID"
CLIENT_ID_ENV = "TRAKT_MAIL_CLIENT_ID"
CLIENT_SECRET_ENV = "TRAKT_MAIL_CLIENT_SECRET"
MAILBOX_ENV = "TRAKT_MAIL_MAILBOX"

#: Optional. Comma-separated addresses (or ``@domain`` suffixes) that this
#: deployment may send to. EMPTY MEANS NO RESTRICTION — the OCC's own approval
#: gate is the control that always applies. Set it during the first controlled
#: onboarding so a mistyped or stale contact address cannot reach a real
#: client while the capability is being proven.
ALLOWLIST_ENV = "TRAKT_MAIL_RECIPIENT_ALLOWLIST"

TIMEOUT_ENV = "TRAKT_MAIL_TIMEOUT_SECONDS"
CONFIRM_ATTEMPTS_ENV = "TRAKT_MAIL_CONFIRM_ATTEMPTS"
CONFIRM_DELAY_ENV = "TRAKT_MAIL_CONFIRM_DELAY_SECONDS"

DEFAULT_TIMEOUT = 20.0
#: How many times Sent Items is re-read looking for the message just sent.
#: ``sendMail`` returns 202 with no body, so the message identifier can only
#: come from the mailbox, and the mailbox needs a moment. Three short attempts
#: is enough in practice and bounded enough to sit inside an API request.
DEFAULT_CONFIRM_ATTEMPTS = 3
DEFAULT_CONFIRM_DELAY = 2.0

#: Good enough to catch a mistyped setting; deliberately not an RFC 5322
#: parser, which would reject addresses Exchange accepts.
ADDRESS_RE = re.compile(r"^[^@\s,;<>]+@[^@\s,;<>]+\.[^@\s,;<>]+$")


class MailNotConfigured(Exception):
    """Outbound mail is off, or its identity is incomplete.

    ``missing`` names the SETTINGS, never their values, so this exception is
    safe to log and safe to show an administrator.
    """

    def __init__(self, missing: List[str], *, disabled: bool = False):
        self.missing = list(missing)
        self.disabled = disabled
        if disabled:
            detail = f"{ENABLED_ENV} is not set to an enabling value"
        else:
            detail = "missing settings: " + ", ".join(self.missing)
        super().__init__(f"outbound mail is not configured ({detail})")


@dataclass(frozen=True)
class MailConfig:
    """A complete, validated Graph mail identity. There is no partial one."""

    tenant_id: str
    client_id: str
    client_secret: str
    mailbox: str
    recipient_allowlist: tuple = ()
    timeout: float = DEFAULT_TIMEOUT
    confirm_attempts: int = DEFAULT_CONFIRM_ATTEMPTS
    confirm_delay: float = DEFAULT_CONFIRM_DELAY

    def permits(self, address: str) -> bool:
        """Whether this deployment may send to ``address``.

        An empty allow-list permits everything: the restriction is an extra
        control for a controlled rollout, not the primary one.
        """
        if not self.recipient_allowlist:
            return True
        candidate = (address or "").strip().lower()
        for entry in self.recipient_allowlist:
            if entry.startswith("@"):
                if candidate.endswith(entry):
                    return True
            elif candidate == entry:
                return True
        return False

    def redacted(self) -> Dict[str, object]:
        """What diagnostics may print. The secret is never one of them."""
        return {
            "mailbox": self.mailbox,
            "tenant_id": self.tenant_id,
            "client_id": self.client_id,
            "client_secret": "set" if self.client_secret else "absent",
            "recipient_allowlist": len(self.recipient_allowlist),
            "timeout": self.timeout,
        }


def outbound_enabled(env: Optional[Dict[str, str]] = None) -> bool:
    """The kill switch alone. Fails closed on anything unrecognised."""
    source = env if env is not None else os.environ
    return str(source.get(ENABLED_ENV, "")).strip().lower() in _ON


def _clean(source: Dict[str, str], name: str) -> str:
    return str(source.get(name, "") or "").strip()


def _number(source: Dict[str, str], name: str, default: float) -> float:
    raw = _clean(source, name)
    if not raw:
        return default
    try:
        value = float(raw)
    except ValueError:
        return default
    return value if value > 0 else default


def _allowlist(raw: str) -> tuple:
    return tuple(sorted({
        entry.strip().lower() for entry in raw.split(",") if entry.strip()
    }))


def load(env: Optional[Dict[str, str]] = None) -> MailConfig:
    """The deployment's mail identity, or a refusal naming what is absent.

    Read per call rather than cached at import: an administrator who fixes a
    setting and restarts one worker should not have to redeploy, and the OCC
    API already reads its operator configuration the same way.
    """
    source = env if env is not None else os.environ
    if not outbound_enabled(source):
        raise MailNotConfigured([ENABLED_ENV], disabled=True)

    values = {name: _clean(source, name) for name in
              (TENANT_ENV, CLIENT_ID_ENV, CLIENT_SECRET_ENV, MAILBOX_ENV)}
    missing = [name for name, value in values.items() if not value]
    if missing:
        raise MailNotConfigured(sorted(missing))

    mailbox = values[MAILBOX_ENV]
    if not ADDRESS_RE.match(mailbox):
        # A mailbox that is not an address would be sent to Graph as a user id
        # and fail there with a far less obvious message.
        raise MailNotConfigured([f"{MAILBOX_ENV} (not an email address)"])

    return MailConfig(
        tenant_id=values[TENANT_ENV],
        client_id=values[CLIENT_ID_ENV],
        client_secret=values[CLIENT_SECRET_ENV],
        mailbox=mailbox,
        recipient_allowlist=_allowlist(_clean(source, ALLOWLIST_ENV)),
        timeout=_number(source, TIMEOUT_ENV, DEFAULT_TIMEOUT),
        confirm_attempts=int(_number(source, CONFIRM_ATTEMPTS_ENV,
                                     DEFAULT_CONFIRM_ATTEMPTS)),
        confirm_delay=_number(source, CONFIRM_DELAY_ENV, DEFAULT_CONFIRM_DELAY),
    )
