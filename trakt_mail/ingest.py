"""trakt_mail.ingest — a client's reply, brought into the OCC's own workflow.

:mod:`trakt_mail.inbound` reads the mailbox and works out whose reply a message
is. This module is what happens next, and its whole design is about what it
deliberately does **not** do.

It does not write a client's answers onto a case. A reply saying "our LEI is
549300…" is a sentence in an email, not an approved answer, and the OCC's
governance says a case changes when a human decides it does. So the text is
recorded — attributed, timestamped, audited, and put in front of the operator
in the same conversation they already work in — and the operator applies it
through the existing instruction path, where the interpreter shows its reading
and asks for confirmation. Nothing about the approval gates changes because a
message arrived by mail rather than by being typed.

What it DOES do is the part that is unambiguous: a file a client attached is a
file, and registering one is the same governed act as an operator uploading it.
So attachments go through ``OccAgentService.register_synthetic_artefact`` —
the same method, the same state check, the same sandbox, the same audit. If the
run's state does not permit registering an artefact, this fails exactly as an
operator's upload would.

Two more properties worth stating, because both are load-bearing:

* **it is pull, not push.** Nothing here runs on a timer. An operator asks
  whether anything has arrived, and this answers. A background poller writing
  to cases unattended is a different risk profile and is not what this is;
* **it is idempotent on the run's record.** The same message ingested twice
  does not produce two copies of a client's file or two conversation entries.
  Read state in the mailbox is a hint, not the memory — a person opening the
  mailbox in Outlook marks mail read too — so the run's own record of what it
  has ingested is what decides.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence

from operations_control.contracts import now_iso
from operations_control.occ_agent.run import Message

from . import inbound as _inbound
from .config import MailConfig, MailNotConfigured, load_inbound
from .graph_client import GraphMailError
from .inbound import CaseTarget, Correlation, InboundMessage

logger = logging.getLogger("trakt_mail.ingest")

#: Recorded on the run for every message taken in, so the same reply is never
#: ingested twice and an auditor can see what came from where.
INGESTED_KEY = "ingested_mail"

#: The conversation role a client's own words are recorded under. Distinct from
#: "operator" and "agent" so no surface can present a client's sentence as
#: something Trakt said or an operator instructed.
ROLE_CLIENT = "client"

#: What an inline image in a signature block is. Registering these as client
#: data files would fill the sandbox with logos.
_SKIP_ATTACHMENT_TYPES = ("image/gif", "image/png", "image/jpeg", "image/bmp")


class MailIngestError(Exception):
    """Reading the mailbox failed. Never raised for "nothing has arrived"."""


# --------------------------------------------------------------------------- #
# What is waiting
# --------------------------------------------------------------------------- #

@dataclass
class WaitingMessage:
    """One message, and what is known about which case it belongs to."""

    message: InboundMessage
    correlation: Correlation
    already_ingested: bool = False

    def to_dict(self) -> Dict[str, Any]:
        return {**self.message.to_dict(),
                "correlation": self.correlation.to_dict(),
                "already_ingested": self.already_ingested}


@dataclass
class Waiting:
    """Everything currently in the folder, sorted into what can be acted on."""

    messages: List[WaitingMessage] = field(default_factory=list)
    mailbox: str = ""
    folder: str = ""
    note: str = ""

    def for_case(self, case_ref: str) -> List[WaitingMessage]:
        return [w for w in self.messages
                if w.correlation.case_ref == case_ref and w.correlation.matched]

    def unmatched(self) -> List[WaitingMessage]:
        return [w for w in self.messages if not w.correlation.matched]

    def to_dict(self) -> Dict[str, Any]:
        return {"mailbox": self.mailbox, "folder": self.folder,
                "note": self.note,
                "messages": [w.to_dict() for w in self.messages],
                "matched": len([w for w in self.messages
                                if w.correlation.matched]),
                "unmatched": len(self.unmatched())}


def _issued_runs(service: Any, tenant: str) -> List[Any]:
    """Every run in ``tenant`` whose pack has actually been sent.

    A case whose pack has not gone out cannot have a reply, so it is not a
    candidate — which keeps the set small and, more importantly, means a
    client's message can never be matched to a case they have never been
    written to. Loaded once per read: the targets and the record of what has
    already been ingested both come from these same runs, and loading them
    twice inside one API request is a cost with nothing to show for it.
    """
    out: List[Any] = []
    for row in service.store.list_runs(tenant):
        if str(row.get("pack_status") or "") != "SENT":
            continue
        case_ref = str(row.get("case_ref") or "")
        if not case_ref:
            continue
        try:
            out.append(service.store.load(tenant, case_ref))
        except Exception:  # noqa: BLE001 — one unreadable run must not stop the
            logger.info("mail: could not read run %s", case_ref)
    return out              # rest being offered as targets


def _target_for(run: Any, tenant: str) -> CaseTarget:
    receipt = run.pack_receipt or {}
    return CaseTarget(
        case_ref=run.case_ref,
        tenant=tenant,
        conversation_id=str(receipt.get("conversation_id") or ""),
        internet_message_id=str(receipt.get("internet_message_id") or ""),
        # The addresses the pack actually went to, from the receipt — not the
        # contacts currently on the case, which an operator may have edited
        # since. Evidence is what was true at the time.
        contacts=tuple(str(a) for a in (receipt.get("to") or []) if a))


def targets_for(service: Any, tenant: str) -> List[CaseTarget]:
    """Correlation targets for every case in ``tenant`` that issued a pack."""
    return [_target_for(run, tenant) for run in _issued_runs(service, tenant)]


def waiting(service: Any, *, tenant: str,
            config: Optional[MailConfig] = None,
            transport: Optional[Any] = None,
            env: Optional[Dict[str, str]] = None) -> Waiting:
    """What is in the mailbox, correlated against this tenant's sent packs.

    Read-only in both systems: it neither writes to a case nor marks anything
    read. An operator can look as often as they like.
    """
    try:
        config = config or load_inbound(env)
    except MailNotConfigured as exc:
        return Waiting(note=f"Reading the mailbox is not enabled ({exc}).")

    client = _inbound.client_for(config, transport=transport)
    try:
        messages = _inbound.read_folder(client, config)
    except GraphMailError as exc:
        raise MailIngestError(
            f"The mailbox could not be read ({exc}). Nothing was ingested."
        ) from None

    runs = _issued_runs(service, tenant)
    targets = [_target_for(run, tenant) for run in runs]
    seen = _ingested_ids(runs)
    out = Waiting(mailbox=config.mailbox, folder=config.inbound_folder)
    for message in messages:
        correlation = _inbound.correlate(message, targets)
        out.messages.append(WaitingMessage(
            message=message, correlation=correlation,
            already_ingested=message.graph_id in seen))
    if not out.messages:
        out.note = "Nothing new has arrived in this mailbox."
    return out


def _ingested_ids(runs: Sequence[Any]) -> set:
    """Every mailbox message id these cases have already taken in."""
    seen: set = set()
    for run in runs:
        for entry in (run.control_results or []):
            if entry.get("kind") == INGESTED_KEY and entry.get("graph_id"):
                seen.add(str(entry["graph_id"]))
    return seen


# --------------------------------------------------------------------------- #
# Taking one in
# --------------------------------------------------------------------------- #

@dataclass
class Ingested:
    """What one ingest actually did. Never optimistic."""

    case_ref: str = ""
    graph_id: str = ""
    registered: List[str] = field(default_factory=list)
    skipped: List[Dict[str, str]] = field(default_factory=list)
    recorded_text: bool = False
    already: bool = False
    statement: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {"case_ref": self.case_ref, "graph_id": self.graph_id,
                "registered": list(self.registered),
                "skipped": list(self.skipped),
                "recorded_text": self.recorded_text,
                "already": self.already, "statement": self.statement}


def ingest(service: Any, agent_case: Any, waiting_message: WaitingMessage, *,
           actor: str, config: Optional[MailConfig] = None,
           transport: Optional[Any] = None,
           mark_read: bool = True) -> "tuple[Any, Ingested]":
    """Take one correlated reply into the case. Files are registered; words are
    recorded and left for the operator to apply.

    Refuses an uncorrelated message outright. There is no ``force`` and no
    operator override at this level on purpose: if a message cannot be tied to
    a case by evidence, the honest fix is for a person to look at the mailbox,
    not for the agent to be told to believe something it could not establish.
    """
    message = waiting_message.message
    correlation = waiting_message.correlation
    run = agent_case.run
    result = Ingested(case_ref=agent_case.case_ref, graph_id=message.graph_id)

    if not correlation.matched:
        raise MailIngestError(
            "This message could not be tied to this case by anything the mail "
            "system carries, so it was not ingested. " + (correlation.note
                                                          or ""))
    if correlation.case_ref != agent_case.case_ref:
        raise MailIngestError(
            f"This message belongs to {correlation.case_ref}, not "
            f"{agent_case.case_ref}. Nothing was ingested.")

    if _already(run, message.graph_id):
        result.already = True
        result.statement = "This reply has already been taken in."
        return agent_case, result

    for attachment in message.attachments:
        reason = _skip_reason(attachment)
        if reason:
            result.skipped.append({"name": attachment.name, "reason": reason})
            continue
        agent_case = service.register_synthetic_artefact(
            agent_case, filename=attachment.name, data=attachment.data,
            actor=actor, declared_type="")
        result.registered.append(attachment.name)

    run = agent_case.run
    if message.body_text.strip():
        run.messages.append(Message(
            role=ROLE_CLIENT, text=message.body_text[:4000],
            at=message.received_at or now_iso(),
            author=message.sender_name or message.sender).to_dict())
        result.recorded_text = True

    result.statement = _statement(message, result)
    run.control_results.append({
        "kind": INGESTED_KEY,
        "graph_id": message.graph_id,
        "internet_message_id": message.internet_message_id,
        "conversation_id": message.conversation_id,
        "sender": message.sender,
        "received_at": message.received_at,
        "subject": message.subject,
        "bases": list(correlation.bases),
        "registered": list(result.registered),
        "skipped": [s["name"] for s in result.skipped],
    })
    service.store.save(run)
    service.audit_mail_ingest(run, actor=actor, message=message,
                              correlation=correlation, result=result)

    if mark_read and message.graph_id:
        try:
            cfg = config or load_inbound()
            _inbound.client_for(cfg, transport=transport).mark_read(
                message.graph_id)
        except (MailNotConfigured, GraphMailError) as exc:
            # The reply is already in the case. Failing to tidy the mailbox
            # afterwards is untidy, not wrong, and must not undo the ingest.
            logger.info("mail: could not mark a message read (%s)", exc)
    return agent_case, result


def _already(run: Any, graph_id: str) -> bool:
    if not graph_id:
        return False
    return any(entry.get("kind") == INGESTED_KEY
               and entry.get("graph_id") == graph_id
               for entry in (run.control_results or []))


def _skip_reason(attachment: Any) -> str:
    """Why an attachment is not a client data file, or "" if it is one."""
    if not attachment.name:
        return "it has no filename"
    if attachment.oversize:
        return (f"it is {attachment.size} bytes, over the limit this "
                "deployment will read")
    if not attachment.data:
        return "its content could not be read"
    if attachment.inline:
        return "it is embedded in the message, not sent as a document"
    if attachment.content_type.lower() in _SKIP_ATTACHMENT_TYPES:
        return "it is an image, most likely part of a signature"
    return ""


def _statement(message: InboundMessage, result: Ingested) -> str:
    """The operator-facing account of one ingest."""
    parts = [f"Reply from {message.sender or 'an unnamed sender'}"]
    if message.received_at:
        parts.append(f"received {message.received_at}")
    line = " ".join(parts) + "."
    if result.registered:
        line += f" Registered: {', '.join(result.registered)}."
    if result.skipped:
        line += " Not registered: " + "; ".join(
            f"{s['name']} ({s['reason']})" for s in result.skipped) + "."
    if not result.registered and not result.skipped:
        line += " No attachments."
    if result.recorded_text:
        line += (" The client's message was recorded on the case. It has NOT "
                 "been applied to any answer — read it and instruct the agent "
                 "if it should change the case.")
    return line
