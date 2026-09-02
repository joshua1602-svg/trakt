"""Autonomous enrichment: it may improve the briefing; it may never withhold it.

THE INVARIANT
-------------
    The deterministic briefing is guaranteed. The autonomous layer is additive.

Everything in this module exists to make that true under every failure mode the
red-team found and several it did not: a blocked review, a degraded one, a model
error, a timeout, an exhausted API balance, a malformed payload, an exception
anywhere in the controller. In all of them the caller gets its deterministic
batch back, unchanged, and a record of what went wrong.

WHY THIS IS A SEPARATE MODULE AND NOT A BRANCH IN ``generate``
--------------------------------------------------------------
``generate.build`` produces the guaranteed briefing. If enrichment lived inside
it, every future edit to enrichment would be an edit to the thing that must not
fail. Here the dependency runs one way — enrichment imports the batch, the batch
knows nothing about enrichment — so there is no path by which a change to the
autonomous layer can alter what the deterministic layer produces.

The one rule that makes this hold in practice is
:func:`enrich`'s bare ``except``. Catching everything is normally a smell. It is
correct here for the same reason ``trigger.on_publication_approved`` does it: the
alternative is that an unforeseeable failure in the optional half suppresses the
mandatory half, and no exception is worth that.

WHAT THE READER SEES, AND WHAT THEY NEVER SEE
---------------------------------------------
They see the deterministic facts, then between zero and four autonomous
observations under a heading that says whose judgement they are. They never see
a dropped finding, an error, a stack trace, a note that enrichment was
attempted, or a gap where enrichment would have been. A briefing that apologises
for its optional half is worse than one that simply does not have it.

The operator sees all of it, in ``batch.provenance['enrichment']``.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional

from .contract import MessageItem, NotificationBatch, NotificationMessage
from .contract import MESSAGE_PORTFOLIO_UPDATE

logger = logging.getLogger("trakt_notifications.enrichment")

#: How many autonomous observations a card may carry. The commercial brief asks
#: for "approximately 2-4": four is the ceiling, and there is deliberately no
#: floor — a quiet period that yields one good observation should carry one, and
#: padding to a target is how a briefing becomes a tool dump.
MAX_INSIGHTS = 4

#: What the reader sees above them. Names the author, because a governed number
#: and an agent's judgement are different kinds of claim and the card should not
#: blur them.
ENRICHMENT_HEADING = "Management observations"

#: Outcome of the enrichment attempt, recorded for the operator.
ENRICHED = "enriched"            # observations were added
NOTHING_TO_ADD = "nothing_to_add"  # the review published, but had no findings
BLOCKED = "blocked"              # the gate withheld the review
FAILED = "failed"                # the autonomous layer errored
NOT_ATTEMPTED = "not_attempted"  # no reviewer was configured


@dataclass
class EnrichmentRecord:
    """What the autonomous layer contributed, and what it could not.

    Stored on the batch so an operator can answer "did enrichment succeed,
    degrade or block, and what did it drop" without re-running anything.
    """

    status: str = NOT_ATTEMPTED
    gate_status: str = ""
    added: int = 0
    #: Findings the gate refused, with the figure that failed. Never rendered.
    dropped: List[Dict[str, Any]] = field(default_factory=list)
    #: Findings the agent produced beyond MAX_INSIGHTS.
    withheld: int = 0
    unsupported_claims: List[Dict[str, Any]] = field(default_factory=list)
    error: Optional[str] = None
    steps: Optional[int] = None
    tool_calls: Optional[int] = None
    out_of_mandate_calls: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "status": self.status, "gate_status": self.gate_status,
            "added": self.added, "withheld": self.withheld,
            "dropped": list(self.dropped),
            "unsupported_claims": list(self.unsupported_claims),
            "error": self.error, "steps": self.steps,
            "tool_calls": self.tool_calls,
            "out_of_mandate_calls": list(self.out_of_mandate_calls),
        }

    @property
    def delivered_enrichment(self) -> bool:
        return self.status == ENRICHED and self.added > 0


def _update_message(batch: NotificationBatch) -> Optional[NotificationMessage]:
    return next((m for m in batch.messages
                 if m.message_type == MESSAGE_PORTFOLIO_UPDATE), None)


def _observation_text(finding: Dict[str, Any]) -> str:
    """One finding as one line a manager reads.

    ``observation`` carries the governed measurement and ``why_it_matters`` the
    agent's judgement; both are kept because a number without a reason is a
    statistic and a reason without a number is an opinion. They are joined into
    a single line rather than nested, because a Teams card is read at a glance
    and a two-level bullet is not.
    """
    observation = str(finding.get("observation") or "").strip()
    why = str(finding.get("why_it_matters") or "").strip()
    if observation and why:
        return f"{observation} {why}"
    return observation or why or str(finding.get("title") or "").strip()


def attach(batch: NotificationBatch, card: Optional[Dict[str, Any]],
           record: EnrichmentRecord) -> NotificationBatch:
    """Add the card's observations to the batch's Portfolio Update.

    Mutates and returns the SAME batch. Nothing here can remove a deterministic
    item, change a deterministic figure, or alter the message's severity — the
    autonomous layer appends and does nothing else.
    """
    message = _update_message(batch)
    if message is None or not card:
        return batch

    findings = list(card.get("findings") or ())
    if len(findings) > MAX_INSIGHTS:
        record.withheld += len(findings) - MAX_INSIGHTS
        findings = findings[:MAX_INSIGHTS]

    added = 0
    for finding in findings:
        text = _observation_text(finding)
        if not text:
            continue
        message.items.append(MessageItem(
            text=text, metric="autonomous_observation",
            insight_id=str(finding.get("title") or "")[:120] or None))
        added += 1

    if added:
        record.added = added
        record.status = ENRICHED
        message.provenance = {
            **(message.provenance or {}),
            "enrichment": (f"{added} observation(s) selected by the Portfolio "
                           "Review Agent from governed MI; every figure was "
                           "verified against a governed tool result before "
                           "publication"),
        }
    elif record.status == NOT_ATTEMPTED:
        record.status = NOTHING_TO_ADD
    return batch


def enrich(batch: NotificationBatch, *,
           reviewer: Optional[Callable[[], Any]] = None) -> NotificationBatch:
    """Attempt enrichment. Return the batch whatever happens.

    ``reviewer`` is a zero-argument callable returning a
    :class:`portfolio_review.controller.ReviewOutcome`. It is injected rather
    than constructed here so the fallback paths are testable without a model,
    and so this module holds no opinion about how a review is run.

    There is no failure mode in which this returns ``None``, raises, or hands
    back a batch with fewer deterministic items than it was given.
    """
    record = EnrichmentRecord()
    try:
        if reviewer is None:
            record.status = NOT_ATTEMPTED
            return _record(batch, record)

        outcome = reviewer()
        record.gate_status = str(getattr(outcome, "gate_status", "") or "")
        record.steps = getattr(outcome, "steps", None)
        efficiency = getattr(outcome, "efficiency", None) or {}
        record.tool_calls = efficiency.get("total_calls")
        record.out_of_mandate_calls = [
            str(c.get("tool")) for c in
            (getattr(outcome, "out_of_mandate_calls", None) or ())]
        record.dropped = [
            {"title": (d.get("finding") or {}).get("title"),
             "reason": d.get("reason")}
            for d in (getattr(outcome, "dropped_findings", None) or ())]
        record.unsupported_claims = list(
            getattr(outcome, "unsupported_claims", None) or ())

        card = getattr(outcome, "card", None)
        if not card:
            # BLOCKED, or the review never completed. Either way the reader
            # gets the deterministic briefing and is told nothing about it.
            record.status = BLOCKED
            logger.info(
                "enrichment: withheld for %s/%s (%s) — deterministic briefing "
                "delivered unchanged", batch.tenant_id, batch.portfolio_id,
                record.gate_status or "no card")
            return _record(batch, record)

        attach(batch, card, record)
        return _record(batch, record)

    except Exception as exc:  # noqa: BLE001 - see the module docstring
        # A model error, a timeout, an exhausted balance, a parse failure, a
        # controller bug. None of them may cost the reader their briefing.
        logger.exception("enrichment: failed for %s/%s; delivering the "
                         "deterministic briefing", batch.tenant_id,
                         batch.portfolio_id)
        record.status = FAILED
        record.error = f"{type(exc).__name__}: {exc}"
        return _record(batch, record)


def _record(batch: NotificationBatch, record: EnrichmentRecord
            ) -> NotificationBatch:
    """Attach the operator's audit record and hand the batch back."""
    batch.provenance = {**(batch.provenance or {}),
                        "enrichment": record.to_dict()}
    return batch


def record_of(batch: NotificationBatch) -> Dict[str, Any]:
    """The enrichment record for one batch, for operators and tests."""
    return dict((batch.provenance or {}).get("enrichment") or {})
