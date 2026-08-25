"""trakt_mail.inbound — reading a client's reply, and knowing whose it is.

The outbound half of this package answers "did it leave Trakt?". This half
answers the two questions that follow: **what came back**, and **which case is
it about**.

The second question is the hard one, and getting it wrong is worse than not
answering it. A mailbox receives mail from many people about many things; a
client contact may appear on more than one onboarding; a person forwards a
thread to a colleague who replies from a different address. Guessing produces a
client's answers written onto the wrong case, which is a data incident rather
than a bug. So this module correlates on **evidence the mail system carries**,
grades it, and returns "I do not know" rather than a best guess:

* ``conversation`` — the reply is in the same Graph conversation as the pack
  Trakt sent. Set by Exchange, not by the client, and the strongest signal
  there is;
* ``in_reply_to`` — the reply's ``In-Reply-To`` or ``References`` header names
  the ``internetMessageId`` of the message Trakt sent;
* ``case_ref`` — the case reference Trakt put in the subject line survives in
  the ``Re:`` the client sends back;
* ``sender`` — the address is a contact on the case. **Never sufficient on its
  own**: one contact can be on several onboardings, and a shared mailbox is one
  address for a whole firm.

A message with at least one of the first three is matched. A message with only
the fourth is reported as *unmatched, with candidates*, which is a question for
an operator rather than an answer from the agent.

Nothing here writes anything, moves anything or deletes anything. It reads a
folder, parses what it finds and reports it. What is done with the result is
:mod:`trakt_mail.ingest`'s decision, and every write it makes goes through the
OCC's own governed methods.
"""

from __future__ import annotations

import base64
import binascii
import html.parser
import logging
import re
from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Optional, Sequence

from .config import MailConfig
from .graph_client import GraphMailClient, GraphMailError

logger = logging.getLogger("trakt_mail.inbound")

#: How a case reference appears in a subject line. The pack's own subject
#: writes it in parentheses; a mail client's ``Re:`` and ``FW:`` prefixes leave
#: it intact, which is what makes it usable as evidence at all.
CASE_REF_RE = re.compile(r"\b([A-Z]{2,8}-\d{4}-\d{4,8})\b")

#: ``In-Reply-To`` and ``References`` carry angle-bracketed message ids. More
#: than one appears in ``References``; all of them are candidates.
MESSAGE_ID_RE = re.compile(r"<[^<>\s]+>")

#: Where a mail client starts quoting the message being replied to. Graph's
#: ``uniqueBody`` normally removes the quoted history for us; these are the
#: fallback for a client whose reply Graph does not split, and for the plain
#: ``body`` when ``uniqueBody`` is absent.
_QUOTE_MARKERS = (
    re.compile(r"^\s*-{2,}\s*original message\s*-{2,}\s*$", re.I),
    re.compile(r"^\s*_{5,}\s*$"),
    re.compile(r"^\s*from:\s.+$", re.I),
    re.compile(r"^\s*on .{4,80}\bwrote:\s*$", re.I),
    re.compile(r"^\s*sent from my \w+", re.I),
    re.compile(r"^\s*>"),
)

#: Correlation bases, strongest first. The first three are evidence from the
#: mail system; the fourth is a coincidence of address.
BASIS_CONVERSATION = "conversation"
BASIS_IN_REPLY_TO = "in_reply_to"
BASIS_CASE_REF = "case_ref"
BASIS_SENDER = "sender"

#: A match needs at least one of these. ``BASIS_SENDER`` is deliberately absent.
SUFFICIENT_BASES = (BASIS_CONVERSATION, BASIS_IN_REPLY_TO, BASIS_CASE_REF)


# --------------------------------------------------------------------------- #
# What arrived
# --------------------------------------------------------------------------- #

@dataclass
class InboundAttachment:
    """One file a client attached.

    ``data`` is the decoded bytes, or empty when the attachment was over the
    configured bound. ``oversize`` says which of those two an empty ``data``
    means, because "the client sent nothing" and "the client sent something too
    big" need different things said to them.
    """

    name: str
    content_type: str = ""
    size: int = 0
    data: bytes = b""
    inline: bool = False
    oversize: bool = False

    def to_dict(self) -> Dict[str, Any]:
        """Everything except the bytes. This is what a status route may show."""
        return {"name": self.name, "content_type": self.content_type,
                "size": self.size, "inline": self.inline,
                "oversize": self.oversize, "readable": bool(self.data)}


@dataclass
class InboundMessage:
    """One message in the OCC mailbox, in the shape this package works in."""

    graph_id: str = ""
    internet_message_id: str = ""
    conversation_id: str = ""
    subject: str = ""
    sender: str = ""
    sender_name: str = ""
    received_at: str = ""
    body_text: str = ""
    in_reply_to: List[str] = field(default_factory=list)
    references: List[str] = field(default_factory=list)
    has_attachments: bool = False
    is_read: bool = False
    attachments: List[InboundAttachment] = field(default_factory=list)

    @property
    def case_refs(self) -> List[str]:
        """Case references named in the subject line, in order."""
        seen: List[str] = []
        for ref in CASE_REF_RE.findall(self.subject or ""):
            if ref not in seen:
                seen.append(ref)
        return seen

    @property
    def answered_message_ids(self) -> List[str]:
        """Every message id this one claims to be a reply to."""
        seen: List[str] = []
        for value in list(self.in_reply_to) + list(self.references):
            if value and value not in seen:
                seen.append(value)
        return seen

    def to_dict(self) -> Dict[str, Any]:
        return {"graph_id": self.graph_id,
                "internet_message_id": self.internet_message_id,
                "conversation_id": self.conversation_id,
                "subject": self.subject,
                "sender": self.sender,
                "sender_name": self.sender_name,
                "received_at": self.received_at,
                "body_text": self.body_text,
                "has_attachments": self.has_attachments,
                "attachments": [a.to_dict() for a in self.attachments]}


# --------------------------------------------------------------------------- #
# Whose is it
# --------------------------------------------------------------------------- #

@dataclass(frozen=True)
class CaseTarget:
    """What one open case looks like to the correlator.

    Built by the caller from records the OCC already holds — the pack receipt
    and the case's contact addresses — so this module reads no store and knows
    nothing about how a case is persisted.
    """

    case_ref: str
    tenant: str = ""
    conversation_id: str = ""
    internet_message_id: str = ""
    contacts: tuple = ()

    def knows(self, address: str) -> bool:
        candidate = (address or "").strip().lower()
        return bool(candidate) and candidate in {
            str(c).strip().lower() for c in self.contacts}


@dataclass
class Correlation:
    """Which case a message belongs to, and on what evidence.

    ``matched`` is False whenever the evidence is only an address. The
    ``candidates`` are still reported, because "this looks like it might be one
    of these two cases" is a useful thing to put in front of an operator and a
    dangerous thing to act on.
    """

    case_ref: str = ""
    tenant: str = ""
    bases: List[str] = field(default_factory=list)
    candidates: List[str] = field(default_factory=list)
    note: str = ""

    @property
    def matched(self) -> bool:
        return bool(self.case_ref) and any(b in SUFFICIENT_BASES
                                           for b in self.bases)

    def to_dict(self) -> Dict[str, Any]:
        return {"case_ref": self.case_ref, "tenant": self.tenant,
                "bases": list(self.bases), "candidates": list(self.candidates),
                "matched": self.matched, "note": self.note}


def correlate(message: InboundMessage,
              targets: Sequence[CaseTarget]) -> Correlation:
    """Which case this message answers, on the strongest evidence available.

    Evidence is gathered per target and the strongest wins; ties are refused
    rather than broken, because two cases with equal claim to one reply is
    exactly the situation where picking one silently does the damage.
    """
    scored: List["tuple[int, CaseTarget, List[str]]"] = []
    answered = set(message.answered_message_ids)
    refs = set(message.case_refs)

    for target in targets:
        bases: List[str] = []
        if (target.conversation_id
                and target.conversation_id == message.conversation_id):
            bases.append(BASIS_CONVERSATION)
        if target.internet_message_id and target.internet_message_id in answered:
            bases.append(BASIS_IN_REPLY_TO)
        if target.case_ref and target.case_ref in refs:
            bases.append(BASIS_CASE_REF)
        if target.knows(message.sender):
            bases.append(BASIS_SENDER)
        if bases:
            scored.append((_strength(bases), target, bases))

    if not scored:
        return Correlation(note="Nothing in this message ties it to a case.")

    best = max(s for s, _t, _b in scored)
    leaders = [(t, b) for s, t, b in scored if s == best]

    if len(leaders) > 1:
        return Correlation(
            candidates=sorted({t.case_ref for t, _b in leaders}),
            note=("More than one case has an equal claim to this message, so "
                  "none was chosen."))

    target, bases = leaders[0]
    if not any(b in SUFFICIENT_BASES for b in bases):
        return Correlation(
            candidates=[target.case_ref],
            note=("The sender is a contact on this case, but nothing else in "
                  "the message ties it there. An address alone is not enough "
                  "to record a client's answer against a case."))
    return Correlation(case_ref=target.case_ref, tenant=target.tenant,
                       bases=bases,
                       note="Matched on " + ", ".join(bases) + ".")


def _strength(bases: Sequence[str]) -> int:
    """A conversation beats a reply header beats a subject beats an address."""
    weights = {BASIS_CONVERSATION: 8, BASIS_IN_REPLY_TO: 4,
               BASIS_CASE_REF: 2, BASIS_SENDER: 1}
    return sum(weights.get(b, 0) for b in bases)


# --------------------------------------------------------------------------- #
# Reading the mailbox
# --------------------------------------------------------------------------- #

def read_folder(client: GraphMailClient, config: MailConfig, *,
                unread_only: bool = True,
                with_attachments: bool = True) -> List[InboundMessage]:
    """Everything currently in the configured folder, parsed.

    Attachments are fetched per message rather than in the listing, because
    Graph does not return their bytes on a collection and a listing that
    silently reported ``has_attachments`` with no attachments would be worse
    than one extra call per message that has any.

    A message from an address outside ``TRAKT_MAIL_SENDER_ALLOWLIST`` is
    dropped here, before its attachments are downloaded — the allow-list is
    meant to stop this deployment reading that mail at all, so honouring it
    after pulling the payload down would be honouring it in name only.
    """
    listed = client.list_messages(folder=config.inbound_folder,
                                  top=config.inbound_max,
                                  unread_only=unread_only)
    out: List[InboundMessage] = []
    for payload in listed:
        message = parse_message(payload)
        if not config.permits_sender(message.sender):
            logger.info("mail: ignoring a message from an address outside the "
                        "sender allow-list")
            continue
        if with_attachments and message.has_attachments:
            message.attachments = _attachments_for(client, config, message)
        out.append(message)
    return out


def _attachments_for(client: GraphMailClient, config: MailConfig,
                     message: InboundMessage) -> List[InboundAttachment]:
    """One message's attachments, or none — never a failed read.

    A message whose attachments cannot be fetched is still worth reporting: the
    operator can see that a client replied, and that Trakt could not read what
    they sent, which is a far more actionable thing to be told than nothing.
    """
    try:
        raw = client.message_attachments(
            message.graph_id, max_bytes=config.inbound_max_attachment)
    except GraphMailError as exc:
        logger.info("mail: could not read attachments on a message (%s)", exc)
        return []
    out: List[InboundAttachment] = []
    for item in raw:
        data = b""
        content = item.get("content") or ""
        if content:
            try:
                data = base64.b64decode(content, validate=True)
            except (binascii.Error, ValueError):
                logger.info("mail: an attachment's content was not readable")
                data = b""
        out.append(InboundAttachment(
            name=str(item.get("name") or ""),
            content_type=str(item.get("content_type") or ""),
            size=int(item.get("size") or 0),
            data=data,
            inline=bool(item.get("inline")),
            oversize=bool(item.get("oversize"))))
    return out


def parse_message(payload: Dict[str, Any]) -> InboundMessage:
    """One Graph message, in this package's shape.

    Tolerant by design: a field Graph did not return becomes an empty value
    rather than an exception, because a mailbox is not a contract and one
    unusual message must not stop the rest being read.
    """
    headers = _headers(payload)
    unique = _content_of(payload.get("uniqueBody"))
    whole = _content_of(payload.get("body"))
    # uniqueBody is Graph's own answer to "what did this person actually
    # write?". Preferred, and the local quote stripping is the fallback for
    # when it is absent or empty.
    text = unique or strip_quoted(whole) or str(payload.get("bodyPreview") or "")

    return InboundMessage(
        graph_id=str(payload.get("id") or ""),
        internet_message_id=str(payload.get("internetMessageId") or ""),
        conversation_id=str(payload.get("conversationId") or ""),
        subject=str(payload.get("subject") or ""),
        sender=_address_of(payload.get("from") or payload.get("sender")),
        sender_name=_name_of(payload.get("from") or payload.get("sender")),
        received_at=str(payload.get("receivedDateTime") or ""),
        body_text=text.strip(),
        in_reply_to=_message_ids(headers.get("in-reply-to", "")),
        references=_message_ids(headers.get("references", "")),
        has_attachments=bool(payload.get("hasAttachments")),
        is_read=bool(payload.get("isRead")))


def _headers(payload: Dict[str, Any]) -> Dict[str, str]:
    out: Dict[str, str] = {}
    for header in payload.get("internetMessageHeaders") or []:
        if not isinstance(header, dict):
            continue
        name = str(header.get("name") or "").strip().lower()
        if not name:
            continue
        value = str(header.get("value") or "")
        out[name] = f"{out[name]} {value}" if name in out else value
    return out


def _message_ids(raw: str) -> List[str]:
    return MESSAGE_ID_RE.findall(raw or "")


def _address_of(node: Any) -> str:
    if not isinstance(node, dict):
        return ""
    inner = node.get("emailAddress")
    if isinstance(inner, dict):
        return str(inner.get("address") or "").strip()
    return str(node.get("address") or "").strip()


def _name_of(node: Any) -> str:
    if not isinstance(node, dict):
        return ""
    inner = node.get("emailAddress")
    if isinstance(inner, dict):
        return str(inner.get("name") or "").strip()
    return ""


def _content_of(node: Any) -> str:
    """A Graph body, as text, whichever content type it declares."""
    if not isinstance(node, dict):
        return ""
    content = str(node.get("content") or "")
    if not content:
        return ""
    if str(node.get("contentType") or "").lower() == "html":
        return html_to_text(content)
    return content


# --------------------------------------------------------------------------- #
# HTML and quoted history
# --------------------------------------------------------------------------- #

class _TextExtractor(html.parser.HTMLParser):
    """The visible text of an HTML mail body.

    Deliberately not a renderer and deliberately not a sanitiser: nothing
    extracted here is ever put back into a page. ``script`` and ``style``
    contents are dropped because they are not words a person wrote, and block
    elements become line breaks because a client's answers are usually a list
    and running them together loses which answer went with which question.
    """

    _SKIP = {"script", "style", "head", "title"}
    _BREAK = {"br", "p", "div", "tr", "li", "h1", "h2", "h3", "h4", "h5", "h6",
              "table", "blockquote", "ul", "ol"}

    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.parts: List[str] = []
        self._skipping = 0

    def handle_starttag(self, tag: str, attrs: Any) -> None:
        if tag in self._SKIP:
            self._skipping += 1
        elif tag in self._BREAK:
            self.parts.append("\n")

    def handle_endtag(self, tag: str) -> None:
        if tag in self._SKIP and self._skipping:
            self._skipping -= 1
        elif tag in self._BREAK:
            self.parts.append("\n")

    def handle_data(self, data: str) -> None:
        if not self._skipping and data:
            self.parts.append(data)

    def text(self) -> str:
        joined = "".join(self.parts)
        joined = joined.replace("\xa0", " ")   # mail HTML is full of nbsp
        lines = [re.sub(r"[ \t]+", " ", line).strip()
                 for line in joined.splitlines()]
        return "\n".join(line for line in _drop_runs(lines))


def _drop_runs(lines: Iterable[str]) -> Iterable[str]:
    """Collapse runs of blank lines to one — mail HTML is full of them."""
    blank = False
    for line in lines:
        if line:
            blank = False
            yield line
        elif not blank:
            blank = True
            yield ""


def html_to_text(source: str) -> str:
    parser = _TextExtractor()
    try:
        parser.feed(source)
        parser.close()
    except Exception:  # noqa: BLE001 — malformed mail HTML is normal
        logger.info("mail: an HTML body could not be fully parsed")
    return parser.text().strip()


def strip_quoted(text: str) -> str:
    """The part of a reply the sender actually wrote.

    Only used when Graph's ``uniqueBody`` is unavailable. Conservative: it
    stops at the first quote marker and keeps everything before it, so a false
    positive costs the tail of a message rather than mixing Trakt's own words
    back in as though the client had written them.
    """
    kept: List[str] = []
    for line in (text or "").splitlines():
        if any(marker.match(line) for marker in _QUOTE_MARKERS):
            break
        kept.append(line)
    return "\n".join(kept).strip()


def targets_from(records: Iterable[Dict[str, Any]]) -> List[CaseTarget]:
    """Build correlation targets from plain dictionaries.

    The shape is the caller's to produce; this exists so the mapping from a
    record to a target lives beside the correlator that consumes it rather than
    being written out at each call site.
    """
    out: List[CaseTarget] = []
    for record in records or []:
        if not isinstance(record, dict):
            continue
        case_ref = str(record.get("case_ref") or "").strip()
        if not case_ref:
            continue
        out.append(CaseTarget(
            case_ref=case_ref,
            tenant=str(record.get("tenant") or ""),
            conversation_id=str(record.get("conversation_id") or ""),
            internet_message_id=str(record.get("internet_message_id") or ""),
            contacts=tuple(str(c) for c in (record.get("contacts") or []) if c)))
    return out


def client_for(config: MailConfig, *, transport: Optional[Any] = None
               ) -> GraphMailClient:
    """A reading client for this configuration.

    Constructed through the same factory the sending path uses, so there is one
    authentication implementation rather than two that could drift.
    """
    from .graph_client import from_config
    return from_config(config, transport=transport)
