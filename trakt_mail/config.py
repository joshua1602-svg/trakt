"""trakt_mail.config — the deployment's Graph mail identity, or nothing at all.

Fail-closed is the whole design. There is no default mailbox, no default
tenant, and no way to obtain a partially configured client: :func:`load`
either returns a complete :class:`MailConfig` or raises
:class:`MailNotConfigured` naming the settings that are absent. A deployment
that has not been configured therefore keeps the record-only adapter and says
plainly that nothing was sent — the same honest failure the OCC already has —
rather than half-sending.

Switches, not one. ``TRAKT_MAIL_OUTBOUND_ENABLED`` is an operator kill
switch independent of the credentials, so sending can be stopped without
removing the identity or redeploying; it follows the same opt-in-token
convention as ``TRAKT_TEAMS_NOTIFICATIONS``
(:mod:`trakt_notifications.config`). ``TRAKT_MAIL_INBOUND_ENABLED`` is the
separate switch for READING the mailbox, and the separation is the point:
sending a pack and reading a client's reply are different capabilities with
different blast radii, and a deployment must be able to hold one without the
other. Turning inbound on is also what makes ``Mail.Read`` meaningful, so it is
the switch an administrator can point at when asked what this application can
see.

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

#: Master kill switch for READING the mailbox. Independent of the outbound
#: switch, and off unless explicitly enabled.
INBOUND_ENABLED_ENV = "TRAKT_MAIL_INBOUND_ENABLED"

#: The mail folder replies are read from. Defaults to the inbox. Naming a
#: dedicated folder is how a deployment that must share a mailbox narrows what
#: the reader looks at — a process control, not a technical one, because the
#: Exchange Application Access Policy scopes the application to a whole
#: mailbox and cannot scope it to a folder.
INBOUND_FOLDER_ENV = "TRAKT_MAIL_INBOUND_FOLDER"
DEFAULT_INBOUND_FOLDER = "inbox"

#: How many messages one read examines. A bound, not a target: the reader is
#: called from an API request and must not be able to pull an entire mailbox
#: into one response.
INBOUND_MAX_ENV = "TRAKT_MAIL_INBOUND_MAX"
DEFAULT_INBOUND_MAX = 25

#: Optional. Comma-separated addresses (or ``@domain`` suffixes) whose mail may
#: be ingested. EMPTY MEANS NO RESTRICTION, as with the recipient list — the
#: correlation evidence is the control that always applies, and a message that
#: cannot be tied to a case is never ingested whoever it came from.
SENDER_ALLOWLIST_ENV = "TRAKT_MAIL_SENDER_ALLOWLIST"

#: Largest single attachment the reader will download, in bytes. A client
#: sending a loan tape is expected; a client sending a video is a mistake, and
#: pulling it into an API worker's memory is a worse one.
INBOUND_MAX_ATTACHMENT_ENV = "TRAKT_MAIL_INBOUND_MAX_ATTACHMENT_BYTES"
DEFAULT_INBOUND_MAX_ATTACHMENT = 25 * 1024 * 1024

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
    inbound_folder: str = DEFAULT_INBOUND_FOLDER
    inbound_max: int = DEFAULT_INBOUND_MAX
    sender_allowlist: tuple = ()
    inbound_max_attachment: int = DEFAULT_INBOUND_MAX_ATTACHMENT

    def permits(self, address: str) -> bool:
        """Whether this deployment may send to ``address``.

        An empty allow-list permits everything: the restriction is an extra
        control for a controlled rollout, not the primary one.
        """
        return _permitted(address, self.recipient_allowlist)

    def permits_sender(self, address: str) -> bool:
        """Whether mail from ``address`` may be ingested.

        Separate from :meth:`permits`, and deliberately so: who a deployment
        may WRITE to and whose mail it may READ are different questions, and
        collapsing them would let an outbound allow-list silently decide what
        the reader ingests.
        """
        return _permitted(address, self.sender_allowlist)

    def redacted(self) -> Dict[str, object]:
        """What diagnostics may print. The secret is never one of them."""
        return {
            "mailbox": self.mailbox,
            "tenant_id": self.tenant_id,
            "client_id": self.client_id,
            "client_secret": "set" if self.client_secret else "absent",
            "recipient_allowlist": len(self.recipient_allowlist),
            "sender_allowlist": len(self.sender_allowlist),
            "inbound_folder": self.inbound_folder,
            "timeout": self.timeout,
        }


def _permitted(address: str, allowlist: tuple) -> bool:
    """Whether ``address`` matches an allow-list entry. Empty permits all."""
    if not allowlist:
        return True
    candidate = (address or "").strip().lower()
    for entry in allowlist:
        if entry.startswith("@"):
            if candidate.endswith(entry):
                return True
        elif candidate == entry:
            return True
    return False


def outbound_enabled(env: Optional[Dict[str, str]] = None) -> bool:
    """The sending kill switch alone. Fails closed on anything unrecognised."""
    source = env if env is not None else os.environ
    return str(source.get(ENABLED_ENV, "")).strip().lower() in _ON


def inbound_enabled(env: Optional[Dict[str, str]] = None) -> bool:
    """The reading kill switch alone. Fails closed on anything unrecognised."""
    source = env if env is not None else os.environ
    return str(source.get(INBOUND_ENABLED_ENV, "")).strip().lower() in _ON


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


def _identity(source: Dict[str, str]) -> MailConfig:
    """The validated identity, whichever switch asked for it.

    Both directions use ONE identity — the same tenant, application and
    mailbox — so this is written once. What differs between them is only which
    switch has to be on, which is what :func:`load` and :func:`load_inbound`
    decide before they get here.
    """
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

    folder = _clean(source, INBOUND_FOLDER_ENV) or DEFAULT_INBOUND_FOLDER

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
        inbound_folder=folder,
        inbound_max=int(_number(source, INBOUND_MAX_ENV, DEFAULT_INBOUND_MAX)),
        sender_allowlist=_allowlist(_clean(source, SENDER_ALLOWLIST_ENV)),
        inbound_max_attachment=int(_number(source, INBOUND_MAX_ATTACHMENT_ENV,
                                           DEFAULT_INBOUND_MAX_ATTACHMENT)),
    )


def load(env: Optional[Dict[str, str]] = None) -> MailConfig:
    """The identity for SENDING, or a refusal naming what is absent.

    Read per call rather than cached at import: an administrator who fixes a
    setting and restarts one worker should not have to redeploy, and the OCC
    API already reads its operator configuration the same way.
    """
    source = env if env is not None else os.environ
    if not outbound_enabled(source):
        raise MailNotConfigured([ENABLED_ENV], disabled=True)
    return _identity(source)


def load_inbound(env: Optional[Dict[str, str]] = None) -> MailConfig:
    """The identity for READING, or a refusal naming what is absent.

    Deliberately not ``load()`` with a flag. A deployment that sends but does
    not read, or reads but does not send, is a legitimate and probably common
    configuration, and each has to fail closed on its own switch without the
    other's state affecting it.
    """
    source = env if env is not None else os.environ
    if not inbound_enabled(source):
        raise MailNotConfigured([INBOUND_ENABLED_ENV], disabled=True)
    return _identity(source)
