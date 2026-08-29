"""trakt_mail.adapter — the OCC's communication seam, implemented over Graph.

:class:`operations_control.occ_agent.communication.CommunicationAdapter` is a
Protocol the OCC Agent already calls; :class:`GraphMailAdapter` is an
implementation of it and nothing more. The pack workflow above it is untouched:
``OccAgentService.send_pack`` still refuses in any state but
``APPROVED_TO_SEND``, a human still has to have approved the content, and the
recipient, subject and body still come from the governed pack. Registering this
adapter changes exactly one thing — the receipt now reports a real send.

What this adapter will not do
-----------------------------
It composes nothing. There is no template, no rewriting and no model in this
module: ``to``, ``subject`` and ``body`` arrive already approved and are sent
verbatim. That is what keeps "the agent cannot send arbitrary mail" a property
of the code rather than a policy statement — the only way to reach
:meth:`deliver` is through the approval gate, and the only thing it can send is
what was approved.

Four refusals, all before anything leaves
-----------------------------------------
* no recipients, or an address that is not an address;
* a recipient outside ``TRAKT_MAIL_RECIPIENT_ALLOWLIST`` when one is set;
* more recipients than a client pack plausibly has;
* an approved artefact whose bytes cannot be attached.

The last is deliberate and is the one worth explaining. The pack document is
the substance of the communication; sending the covering note without it would
be a half-delivery that looks like a whole one to everybody except the client.
So a resolution failure refuses the send, leaving the pack in
``APPROVED_TO_SEND`` where it can simply be sent again once the cause is fixed.
"""

from __future__ import annotations

import base64
import datetime as _dt
import logging
import mimetypes
import re
from typing import Any, Callable, Dict, List, Optional

from operations_control.engine import OpsError
from operations_control.occ_agent.communication import Receipt

from . import presentation as _presentation
from .config import ADDRESS_RE, MailConfig, MailNotConfigured, load
from .graph_client import GraphMailClient, GraphMailError, Transport, from_config

logger = logging.getLogger("trakt_mail.adapter")

ADAPTER_NAME = "graph_mail"

#: Graph's ``sendMail`` request is bounded at 4 MB including the base64
#: expansion of every attachment, so the raw budget is deliberately well under
#: it. A client pack is a few tens of kilobytes; anything approaching this is a
#: mistake worth refusing rather than truncating.
MAX_TOTAL_ATTACHMENT_BYTES = 3 * 1024 * 1024
MAX_ATTACHMENTS = 10

#: A client pack goes to a client's named contacts. A larger list is a sign of
#: a mis-populated case, not of a legitimate mailing.
MAX_RECIPIENTS = 10

#: Resolves the durable reference recorded on the run (a ``blob://`` URI) to
#: the bytes that were approved. Injected so this module performs no storage
#: I/O of its own and can be tested without one.
Resolver = Callable[[str], bytes]


class MailDeliveryRefused(OpsError):
    """The pack was not sent, and the pack workflow has not advanced.

    An :class:`OpsError` so the API answers with the same operator-safe
    envelope as every other refusal, and so the state machine is left where a
    retry is legal.
    """

    def __init__(self, message: str, *, http_status: int = 502):
        super().__init__("OCC_AGENT_MAIL_NOT_SENT", message,
                         http_status=http_status)


def _attachment(name: str, data: bytes) -> Dict[str, Any]:
    content_type = mimetypes.guess_type(name)[0] or "application/octet-stream"
    return {
        "@odata.type": "#microsoft.graph.fileAttachment",
        "name": name,
        "contentType": content_type,
        "contentBytes": base64.b64encode(data).decode("ascii"),
    }


class GraphMailAdapter:
    """Issue an approved OCC communication as mail from the OCC mailbox."""

    name = ADAPTER_NAME

    def __init__(self, client: GraphMailClient, *,
                 config: MailConfig,
                 resolver: Optional[Resolver] = None):
        self.client = client
        self.config = config
        self.resolver = resolver

    # ------------------------------------------------------------------ #
    def deliver(self, *, to: List[str], subject: str, body: str,
                artefacts: List[Dict[str, str]], content_hash: str,
                actor: str) -> Receipt:
        """Send, then report what is actually true about the send."""
        recipients = self._recipients(to)

        # The receipt id is minted BEFORE anything is rendered, so it can be
        # stamped on every page of the PDF, printed in the email footer AND
        # carried as the message's internet header. One approved communication,
        # one identifier, visible in all three places.
        receipt = Receipt(adapter=self.name, sent=False, to=recipients,
                          subject=subject, artefacts=list(artefacts),
                          content_hash=content_hash)

        pack_markdown, passthrough = self._resolve(artefacts)
        try:
            presented = _presentation.build(
                body, pack_markdown, receipt_id=receipt.receipt_id,
                content_hash=content_hash, generated_at=_generated_at(),
                fallback_case_ref=_case_ref_in(subject))
        except _presentation.PresentationError as exc:
            logger.warning("mail: could not render the client pack (%s)", exc)
            raise MailDeliveryRefused(
                f"The approved pack could not be prepared for the client "
                f"({exc}), so nothing was sent. The pack is still approved.",
                http_status=500) from None

        attachments = list(passthrough)
        if presented.pdf:
            attachments.insert(0, _attachment(presented.pdf_name,
                                              presented.pdf))
        attached = [a["name"] for a in attachments]
        self._assert_within_budget(attachments)

        try:
            result = self.client.send(to=recipients, subject=subject,
                                      body=presented.html_body,
                                      body_type="HTML",
                                      receipt_id=receipt.receipt_id,
                                      attachments=attachments)
        except GraphMailError as exc:
            logger.warning("mail: send refused for %s (retryable=%s status=%s)",
                           self.config.mailbox, exc.retryable, exc.status)
            raise MailDeliveryRefused(
                "The approved pack was not sent: Microsoft Graph refused the "
                f"message ({exc}). The pack is still approved and can be sent "
                "again once this is resolved.",
                http_status=502 if exc.retryable else 409) from None

        receipt.sent = True
        # Structured, not only narrated. These three used to appear solely
        # inside the statement prose, which made them readable by a person and
        # useless to the reply reader that has to match a client's answer back
        # to this case.
        receipt.internet_message_id = result.internet_message_id
        receipt.conversation_id = result.conversation_id
        receipt.graph_message_id = result.graph_message_id
        receipt.statement = _statement(self.config.mailbox, result, attached,
                                       actor)
        logger.info(
            "mail: sent from %s recipients=%d attachments=%d receipt=%s "
            "internet_message_id=%s confirmed=%s",
            self.config.mailbox, len(recipients), len(attachments),
            receipt.receipt_id, result.internet_message_id or "-",
            result.confirmed)
        return receipt

    # ------------------------------------------------------------------ #
    def _recipients(self, to: List[str]) -> List[str]:
        cleaned: List[str] = []
        for address in to or []:
            candidate = str(address or "").strip()
            if not candidate:
                continue
            if not ADDRESS_RE.match(candidate):
                raise MailDeliveryRefused(
                    f"{candidate!r} is not an email address, so nothing was "
                    "sent. Correct the contact on the case and approve again.",
                    http_status=400)
            if not self.config.permits(candidate):
                raise MailDeliveryRefused(
                    f"This deployment is restricted to an approved list of "
                    f"recipients and {candidate!r} is not on it. Nothing was "
                    "sent.", http_status=403)
            if candidate not in cleaned:
                cleaned.append(candidate)
        if not cleaned:
            raise MailDeliveryRefused(
                "There is no recipient address on this pack, so nothing was "
                "sent.", http_status=409)
        if len(cleaned) > MAX_RECIPIENTS:
            raise MailDeliveryRefused(
                f"A client pack may go to at most {MAX_RECIPIENTS} recipients; "
                f"this one names {len(cleaned)}. Nothing was sent.",
                http_status=409)
        return cleaned

    def _resolve(self, artefacts: List[Dict[str, str]]
                 ) -> "tuple[str, List[Dict[str, Any]]]":
        """Read the approved artefacts, and sort them into what they are for.

        Two are special, and both were client-facing defects:

        * ``covering_email.txt`` holds the covering message, which is already
          the email body. Attaching it sent the client the same words twice.
          It is read and discarded here, not sent.
        * ``onboarding_pack.md`` is the pack in its SOURCE format. It stays in
          the case store as the auditable original; what the client receives is
          the PDF rendered from it.

        Anything else is attached as it stands, so a later artefact does not
        need this method changed to reach a client.
        """
        declared = [a for a in (artefacts or []) if isinstance(a, dict)]
        if not declared:
            return "", []
        if self.resolver is None:
            raise MailDeliveryRefused(
                "This pack has approved documents but the mail adapter has no "
                "way to read them, so nothing was sent.", http_status=500)
        if len(declared) > MAX_ATTACHMENTS:
            raise MailDeliveryRefused(
                f"A pack may carry at most {MAX_ATTACHMENTS} documents; this "
                f"one has {len(declared)}. Nothing was sent.", http_status=409)

        pack_markdown = ""
        passthrough: List[Dict[str, Any]] = []
        for artefact in declared:
            name = str(artefact.get("name") or "").strip()
            ref = str(artefact.get("ref") or "").strip()
            if not name or not ref or any(ord(c) < 32 for c in ref):
                raise MailDeliveryRefused(
                    "An approved document on this pack has no usable "
                    "reference, so nothing was sent.", http_status=409)
            try:
                data = self.resolver(ref)
            except Exception as exc:  # noqa: BLE001 — storage class
                logger.warning("mail: could not read approved artefact %s (%s)",
                               name, type(exc).__name__)
                raise MailDeliveryRefused(
                    f"The approved document {name!r} could not be read, so "
                    "nothing was sent. The pack is still approved.",
                    http_status=502) from None
            if not data:
                raise MailDeliveryRefused(
                    f"The approved document {name!r} is empty, so nothing was "
                    "sent.", http_status=409)

            if name == _presentation.COVERING_TEXT:
                continue                    # it is the body, not a document
            if name.lower().endswith(".md"):
                pack_markdown = data.decode("utf-8", errors="replace")
                continue                    # rendered to PDF, not attached
            passthrough.append(_attachment(name, data))
        return pack_markdown, passthrough

    def _assert_within_budget(self, attachments: List[Dict[str, Any]]) -> None:
        """Refuse a message Graph would reject for size, before sending it.

        Measured on the base64 payload rather than the source bytes, because
        that is what actually travels and what the 4 MB request limit counts.
        """
        total = sum(len(a.get("contentBytes") or "") for a in attachments)
        if total > MAX_TOTAL_ATTACHMENT_BYTES:
            raise MailDeliveryRefused(
                "The approved documents are too large to send as one "
                "message, so nothing was sent.", http_status=409)


def _generated_at() -> str:
    """The date printed on the checklist. A date, not a timestamp: a client
    reading it does not need the second it was rendered."""
    return _dt.datetime.now(_dt.timezone.utc).strftime("%d %B %Y")


_CASE_REF_IN_SUBJECT = re.compile(r"\(([A-Z]{2,8}-\d{4}-\d{4,8})\)")


def _case_ref_in(subject: str) -> str:
    """The case reference from the approved subject line.

    Only a fallback: the pack document states its own reference, and that is
    what is used when it is present.
    """
    found = _CASE_REF_IN_SUBJECT.search(subject or "")
    return found.group(1) if found else ""


def _statement(mailbox: str, result: Any, attached: List[str],
               actor: str) -> str:
    """The operator-facing account of the send. Never optimistic.

    It states the mailbox it left from, who approved it, what went with it, and
    the identifiers that make the message findable — including when the
    read-back did not resolve, which is reported rather than glossed.
    """
    parts = [f"Sent from {mailbox} via Microsoft Graph"]
    if actor:
        parts.append(f"on the approval of {actor}")
    parts.append(f"(receipt {result.receipt_id})")
    line = " ".join(parts) + "."
    if attached:
        line += f" Attached: {', '.join(attached)}."
    else:
        line += " No attachment."
    if result.confirmed:
        line += (f" Confirmed in Sent Items as internet message id "
                 f"{result.internet_message_id or '(none reported)'}")
        if result.graph_message_id:
            line += f", Graph message id {result.graph_message_id}"
        line += "."
    else:
        line += (" Microsoft Graph accepted the message "
                 f"(HTTP {result.status}) but it could not be read back from "
                 f"Sent Items ({result.confirmation_note}); the message "
                 f"carries header x-trakt-receipt-id: {result.receipt_id}, "
                 "which identifies it in the mailbox.")
    return line


# --------------------------------------------------------------------------- #
# Construction
# --------------------------------------------------------------------------- #
def from_env(*, resolver: Optional[Resolver] = None,
             transport: Optional[Transport] = None,
             env: Optional[Dict[str, str]] = None) -> GraphMailAdapter:
    """Build the adapter from the environment, or raise :class:`MailNotConfigured`."""
    config = load(env)
    return GraphMailAdapter(from_config(config, transport=transport),
                            config=config, resolver=resolver)


def outbound_adapter(storage: Any = None, *,
                     env: Optional[Dict[str, str]] = None,
                     transport: Optional[Transport] = None
                     ) -> Optional[GraphMailAdapter]:
    """The adapter this deployment should use, or ``None``.

    ``None`` is the fail-closed answer, and the caller's fallback is the OCC's
    own ``RecordOnlyAdapter`` — which reports honestly that nothing was sent.
    A misconfiguration therefore degrades to the previous behaviour instead of
    to a broken send path or a failed service start.

    ``storage`` supplies the resolver: approved artefacts are read back from
    the durable reference the run itself recorded, so what is attached is what
    was approved. ``transport`` exists so the whole construction path — not a
    stand-in for it — can be exercised without a network.
    """
    try:
        resolver = storage.read_bytes if storage is not None else None
        adapter = from_env(resolver=resolver, env=env, transport=transport)
    except MailNotConfigured as exc:
        logger.info("outbound mail is off: %s", exc)
        return None
    except Exception:  # noqa: BLE001 — must never stop the API starting
        logger.exception("outbound mail could not be configured")
        return None
    logger.info("outbound mail enabled: %s", adapter.config.redacted())
    return adapter
