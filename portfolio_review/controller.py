"""The Portfolio Review controller — pin the period, run the loop, keep the evidence.

Deliberately thin. The loop is ``readiness_agent.run_assessment``, the door is
``GovernedSession`` and the tools are ``trakt_tools``; a second agent framework
beside those would duplicate the one property that matters — that the model has
no data to compute over — and duplicating a safety property is how one copy
quietly loses it.

WHAT THIS ADDS
--------------
* **The period the review is about.** Resolved from the governed snapshot
  discovery every other funded and pipeline surface reads, and stated in the
  opening message, so the model reviews the period that was approved rather than
  whatever "latest" meant when it happened to call a tool.
* **The evidence record.** Every finding is returned alongside the governed tool
  transcript that produced it, so a narrative can be traced back to the calls
  behind it without re-running anything.

WHAT THIS DELIBERATELY DOES NOT ADD
-----------------------------------
A first call, a metric list, an ordering, or any nudge toward a dimension. The
deterministic materiality layer already runs a fixed set of checks every period
and states what it suppressed; this layer exists for what that set cannot
anticipate, and giving it a checklist would leave the estate with two.

It also runs no model itself. ``client`` is injected, so the controller — the
period resolution, the evidence assembly, the refusal paths — is exercised in
tests against a scripted model without an API key.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional

from readiness_agent.agent import DEFAULT_MAX_STEPS, DEFAULT_MODEL
from readiness_agent.session import GovernedSession

from . import brief, mandate, numeric_gate
from .objective import (
    MONTHLY_FUNDED, SUBMIT_REVIEW, SYSTEM_PROMPT, WEEKLY_PIPELINE, objective_for,
)
from .session import MIScopedSession

logger = logging.getLogger("portfolio_review.controller")

PERIOD_WEEKLY_PIPELINE = WEEKLY_PIPELINE
PERIOD_MONTHLY_FUNDED = MONTHLY_FUNDED

#: A period review is a narrower question than a readiness assessment, so it is
#: given a smaller ceiling. The ceiling is the ONLY control on the loop: it
#: bounds what a confused run costs, and it steers nothing.
DEFAULT_REVIEW_STEPS = 24


@dataclass
class ReviewOutcome:
    """One completed review, and everything observable about how it got there.

    ``review`` is what the model submitted. ``card`` is what a reader may
    actually be shown — the review after the numeric gate has refused anything
    unsupported and the word budget has selected what fits. They are separate
    fields on purpose: a reviewer investigating a bad card needs to see what the
    model said *and* what the gate did to it, and collapsing them would hide
    exactly the step that matters.
    """

    period: str
    resource: str
    review: Optional[Dict[str, Any]] = None
    stopped_reason: str = ""
    steps: int = 0
    #: The governed tool calls, in order, with a digest of each result. This is
    #: the evidence: a finding that cites no call in here is unsupported.
    transcript: List[Dict[str, Any]] = field(default_factory=list)
    efficiency: Dict[str, Any] = field(default_factory=dict)
    usage: Dict[str, int] = field(default_factory=dict)
    period_context: Dict[str, Any] = field(default_factory=dict)
    unavailable: Optional[str] = None
    #: What the publication gate decided. PUBLISHABLE, DEGRADED or BLOCKED.
    gate_status: str = ""
    gate_reasons: List[str] = field(default_factory=list)
    #: §16: every published figure and the governed field behind it.
    claim_ledger: List[Dict[str, Any]] = field(default_factory=list)
    unsupported_claims: List[Dict[str, Any]] = field(default_factory=list)
    dropped_findings: List[Dict[str, Any]] = field(default_factory=list)
    #: §15: every tool the model reached for, and whether the mandate allowed it.
    tool_call_audit: List[Dict[str, Any]] = field(default_factory=list)
    out_of_mandate_calls: List[Dict[str, Any]] = field(default_factory=list)
    #: The Teams-facing selection.
    card: Optional[Dict[str, Any]] = None

    @property
    def available(self) -> bool:
        return self.card is not None

    @property
    def submitted(self) -> bool:
        """Did the model finish, regardless of what the gate then did?"""
        return self.review is not None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "period": self.period, "resource": self.resource,
            "available": self.available, "unavailable": self.unavailable,
            "review": self.review, "card": self.card,
            "stopped_reason": self.stopped_reason,
            "steps": self.steps, "period_context": dict(self.period_context),
            "transcript": list(self.transcript),
            "efficiency": dict(self.efficiency), "usage": dict(self.usage),
            "gate_status": self.gate_status,
            "gate_reasons": list(self.gate_reasons),
            "claim_ledger": list(self.claim_ledger),
            "unsupported_claims": list(self.unsupported_claims),
            "dropped_findings": list(self.dropped_findings),
            "tool_call_audit": list(self.tool_call_audit),
            "out_of_mandate_calls": list(self.out_of_mandate_calls),
        }

    def evidence_for(self, finding: Dict[str, Any]) -> List[Dict[str, Any]]:
        """The governed calls one finding rests on.

        Matched on the tool names the model cited. A finding citing a tool it
        never called returns nothing, which is exactly what a reviewer needs to
        see — the alternative, quietly attaching the whole transcript, would
        make every finding look equally well evidenced.
        """
        cited = {str(t) for t in (finding.get("evidence_tools") or ())}
        return [call for call in self.transcript if call.get("tool") in cited]


# --------------------------------------------------------------------------- #
# Period resolution
# --------------------------------------------------------------------------- #
def resolve_period(period: str, *, client_id: str,
                   output_root: Any = None,
                   pipeline_root: Any = None,
                   to_run_id: Optional[str] = None) -> Dict[str, Any]:
    """The two dates this review compares, from the governed discovery.

    Stated to the model rather than left to it. A review told only "review the
    latest period" would resolve its own period on every call, and two tools
    disagreeing about which week is current is exactly the defect the shared
    resolvers exist to prevent.
    """
    if period == MONTHLY_FUNDED:
        from mi_agent_api import snapshots as snap

        if not output_root:
            return {"available": False,
                    "reason": "no governed funded output root is configured"}
        discovered = snap.discover_snapshots(output_root)
        runs = next((p["runs"] for p in discovered.get("portfolios", [])
                     if p.get("client_id") == client_id), [])
        if to_run_id:
            index = next((i for i, r in enumerate(runs)
                          if r["run_id"] == to_run_id), None)
            runs = runs[:index + 1] if index is not None else []
        if len(runs) < 2:
            return {"available": False,
                    "reason": ("at least two governed funded reporting periods "
                               "are needed to review a movement"),
                    "periods_available": len(runs)}
        current, prior = runs[-1], runs[-2]
        return {
            "available": True, "period": MONTHLY_FUNDED,
            "current_run_id": current.get("run_id"),
            "prior_run_id": prior.get("run_id"),
            "current_reporting_date": current.get("reporting_date"),
            "prior_reporting_date": prior.get("reporting_date"),
        }

    from mi_agent_api import pipeline_contract as pipeline_mod

    if not pipeline_root:
        return {"available": False,
                "reason": "no governed weekly pipeline root is configured"}
    inventory = pipeline_mod.weekly_extract_inventory(pipeline_root, client_id)
    extracts = inventory.get("extracts") or []
    if len(extracts) < 2:
        return {"available": False,
                "reason": ("at least two governed weekly pipeline extracts are "
                           "needed to review a movement"),
                "periods_available": len(extracts)}
    current, prior = extracts[-1], extracts[-2]
    return {
        "available": True, "period": WEEKLY_PIPELINE,
        "current_reporting_date": current.get("pipeline_extract_date"),
        "prior_reporting_date": prior.get("pipeline_extract_date"),
    }


def _context_note(context: Dict[str, Any]) -> str:
    """What the model is told about the period. Dates and nothing else.

    No figure appears here on purpose: a number in the opening message is a
    number the model did not obtain from a tool call, and the rule that every
    figure it states came from Trakt has to be true of its INPUT as well as its
    output.
    """
    lines = [
        "This review covers one governed reporting period.",
        f"  current period ends {context.get('current_reporting_date')}",
        f"  compared against {context.get('prior_reporting_date')}",
    ]
    if context.get("current_run_id"):
        lines.append(f"  governed run: {context['current_run_id']}")
    lines.append(
        "Ask Trakt for every figure. Nothing about the position is stated here.")
    return "\n".join(lines)


# --------------------------------------------------------------------------- #
# The review
# --------------------------------------------------------------------------- #
def run_review(session: GovernedSession, *,
               period: str = MONTHLY_FUNDED,
               client_id: Optional[str] = None,
               output_root: Any = None,
               pipeline_root: Any = None,
               to_run_id: Optional[str] = None,
               period_context: Optional[Dict[str, Any]] = None,
               model: str = DEFAULT_MODEL,
               max_steps: int = DEFAULT_REVIEW_STEPS,
               client: Any = None,
               on_step: Optional[Callable[[int, str], None]] = None,
               ) -> ReviewOutcome:
    """Run one autonomous period review.

    Returns a :class:`ReviewOutcome` whether or not the model produced a review:
    a period with no comparable prior is a governed answer, and raising on it
    would make "nothing to compare" indistinguishable from a broken agent.
    """
    from readiness_agent.agent import run_assessment

    objective = objective_for(period)
    scoped = (session if isinstance(session, MIScopedSession)
              else MIScopedSession(session))
    resolved = period_context or resolve_period(
        period, client_id=client_id or scoped.resource.split("/")[0],
        output_root=output_root, pipeline_root=pipeline_root,
        to_run_id=to_run_id)

    if not resolved.get("available"):
        return ReviewOutcome(
            period=period, resource=scoped.resource,
            unavailable=resolved.get("reason") or "the period could not be resolved",
            stopped_reason="period unavailable", period_context=dict(resolved))

    run = run_assessment(
        scoped, objective=objective, context_note=_context_note(resolved),
        model=model, max_steps=max_steps, client=client, on_step=on_step,
        system_prompt=SYSTEM_PROMPT, submit_tool=SUBMIT_REVIEW,
        tool_schemas=mandate.tool_schemas())

    # The gate runs on every completed review, including one that looks clean.
    # A control that only engages when something already looks wrong is not a
    # control; it is a second opinion.
    gated = numeric_gate.apply(run.assessment, scoped.index)
    card = brief.render(gated.review) if gated.publishable else None

    unavailable = None
    if not run.assessment:
        unavailable = f"the review did not complete: {run.stopped_reason}"
    elif card is None:
        unavailable = ("the review stated figures no governed result contains, "
                       "so it was withheld: " + "; ".join(gated.reasons))

    return ReviewOutcome(
        period=period, resource=scoped.resource,
        review=run.assessment, stopped_reason=run.stopped_reason,
        steps=run.steps, transcript=run.transcript, efficiency=run.efficiency,
        usage=run.usage, period_context=dict(resolved),
        gate_status=gated.status, gate_reasons=list(gated.reasons),
        claim_ledger=gated.ledger(),
        unsupported_claims=[{"stated": c.stated + (c.unit or ""),
                             "in": c.field_path, "excerpt": c.excerpt}
                            for c in gated.unsupported],
        dropped_findings=list(gated.dropped_findings),
        tool_call_audit=scoped.tool_call_audit(),
        out_of_mandate_calls=scoped.out_of_mandate_calls(),
        card=card.to_dict() if card else None,
        unavailable=unavailable)
