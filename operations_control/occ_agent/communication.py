"""operations_control.occ_agent.communication — issuing the pack to a client.

Sending something to a client is the first thing this feature does that leaves
Trakt, so it is the first thing that needs a governed workflow rather than a
button:

    DRAFTED → HUMAN_REVIEW_REQUIRED → APPROVED_TO_SEND → SENT

Every step is attributed and audited, and the approval is a person's, not the
agent's. The agent drafts; a human reads it; a human approves it; only then can
it be issued.

Delivery itself is behind an adapter, because *how* something reaches a client
is infrastructure and the workflow above is not. Two implementations:

* :class:`RecordOnlyAdapter` — the default, and the only one in this pass. It
  records what was approved, to whom, and the exact rendered artefacts, and it
  **says plainly that nothing was sent**. It never claims otherwise, and the
  receipt it returns carries ``sent`` as False.
* a mail adapter, when one exists. The workflow above does not change when it
  arrives; only :meth:`CommunicationAdapter.deliver` does.

The distinction matters: an operator who approved a pack for sending and sees
"recorded, not sent" knows to send it themselves. An operator told "sent" when
nothing was would find out from the client.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional, Protocol

from ..contracts import new_id, now_iso

# -- the four states of a pack ---------------------------------------------- #
DRAFTED = "DRAFTED"
HUMAN_REVIEW_REQUIRED = "HUMAN_REVIEW_REQUIRED"
APPROVED_TO_SEND = "APPROVED_TO_SEND"
SENT = "SENT"

PACK_WORKFLOW = (DRAFTED, HUMAN_REVIEW_REQUIRED, APPROVED_TO_SEND, SENT)

#: What may follow what. Redrafting after review or after issue is legitimate;
#: skipping the review or the approval is not.
PACK_TRANSITIONS: Dict[str, tuple] = {
    "": (DRAFTED,),
    DRAFTED: (HUMAN_REVIEW_REQUIRED, DRAFTED),
    HUMAN_REVIEW_REQUIRED: (APPROVED_TO_SEND, DRAFTED),
    APPROVED_TO_SEND: (SENT, DRAFTED),
    SENT: (DRAFTED,),
}


class PackWorkflowError(Exception):
    """An illegal move in the pack's own workflow."""

    def __init__(self, from_state: str, to_state: str):
        self.code = "OCC_AGENT_PACK_WORKFLOW"
        self.message = (f"A pack cannot go from "
                        f"{from_state or 'nothing'} to {to_state}.")
        self.http_status = 409
        super().__init__(self.message)


def assert_pack_transition(from_state: str, to_state: str) -> None:
    if to_state not in PACK_TRANSITIONS.get(from_state or "", ()):
        raise PackWorkflowError(from_state, to_state)


@dataclass
class Receipt:
    """What happened when a pack was issued.

    ``sent`` is the honest answer to "did this leave Trakt?". A record-only
    adapter always returns False, and every surface reads that rather than
    assuming.
    """

    adapter: str
    sent: bool
    at: str = field(default_factory=now_iso)
    receipt_id: str = field(default_factory=lambda: new_id("com"))
    to: List[str] = field(default_factory=list)
    subject: str = ""
    #: Where the rendered artefacts were stored, so the operator can retrieve
    #: exactly what was approved.
    artefacts: List[Dict[str, str]] = field(default_factory=list)
    content_hash: str = ""
    #: Operator-facing, and never optimistic.
    statement: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class CommunicationAdapter(Protocol):
    """The seam a real mail integration would implement."""

    name: str

    def deliver(self, *, to: List[str], subject: str, body: str,
                artefacts: List[Dict[str, str]], content_hash: str,
                actor: str) -> Receipt: ...


class RecordOnlyAdapter:
    """Records approved delivery intent. Sends nothing, and says so.

    The rendered document and email are stored by the caller before this is
    reached, so what an operator sends by hand is byte-identical to what was
    approved.
    """

    name = "record_only"

    #: Stated on every receipt. Deliberately not "queued" or "pending" — the
    #: pack is not going anywhere on its own.
    STATEMENT = ("Recorded as issued. Trakt did not send it: no email "
                 "integration is enabled in this environment. Send the "
                 "approved pack and covering email from the record.")

    def deliver(self, *, to: List[str], subject: str, body: str,
                artefacts: List[Dict[str, str]], content_hash: str,
                actor: str) -> Receipt:
        del body, actor                    # recorded by the caller, not here
        return Receipt(adapter=self.name, sent=False, to=list(to),
                       subject=subject, artefacts=list(artefacts),
                       content_hash=content_hash, statement=self.STATEMENT)


def default_adapter(policy: Optional[Any] = None) -> CommunicationAdapter:
    """The adapter this environment uses.

    Record-only unless a policy permits external email, which no synthetic
    policy does and no live policy grants in this pass. Written as a lookup so
    a real adapter is a registration rather than a rewrite.
    """
    if policy is not None and getattr(policy, "allow_external_email", False):
        raise NotImplementedError(
            "No mail adapter is registered. Register one here rather than "
            "sending from a call site.")
    return RecordOnlyAdapter()
