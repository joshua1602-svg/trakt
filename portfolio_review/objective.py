"""What the review is asked to do, and the shape it answers in.

Two objectives — a weekly pipeline review and a monthly funded one — and one
system prompt, all derived from :data:`portfolio_review.mandate.MANDATE` so
there is one statement of what this agent is for and the prompt is a rendering
of it rather than a second, drifting copy.

SPARSE ON METHOD, SPECIFIC ON DISCIPLINE
----------------------------------------
The prompt says what question the period poses and which mistakes are
disqualifying. It names no metric, no ordering and no first call. That restraint
is the point: a prompt listing "check balance, then LTV, then concentration"
produces workflow automation wearing an agent's clothes — it would run the same
twenty calls on a quiet month as on an acquisition month, and could never find
the thing nobody thought to list. The fixed set of checks belongs in
``insight_generators_funded``, where it already is; this layer is for what that
set cannot anticipate.

WHAT CHANGED AFTER THE FIRST REAL-MODEL RED-TEAM
------------------------------------------------
Two rules here previously carried the whole weight of two boundaries, and both
gave way under a real model:

* **arithmetic.** Rule 1 forbade it in the strongest terms available and the
  model published "Combined they are £1.88m" anyway. The control is now
  :mod:`portfolio_review.numeric_gate`, which refuses the finding. The rule
  stays in the prompt because a model that obeys it produces a publishable card
  first time — but the prompt is now the optimisation and the gate is the
  control, not the other way round.
* **scope.** The agent was asked what changed this month and reported ESMA
  Annex 2 blockers and a breach of a rulebook labelled SYNTHETIC. The control is
  now the allow-list in :mod:`portfolio_review.mandate`: those tools are not
  offered and a call naming one is refused. The prompt states the boundary so
  the model does not waste turns discovering it.

Neither rule was removed. Both were demoted from control to hint, which is the
only honest place for a sentence in a prompt.
"""

from __future__ import annotations

from typing import Any, Dict

from .mandate import MANDATE, NOT_ROLES, ROLE

#: The two periods a review runs for.
WEEKLY_PIPELINE = "weekly_pipeline"
MONTHLY_FUNDED = "monthly_funded"


WEEKLY_PIPELINE_OBJECTIVE = (
    "Review what materially changed in this portfolio's PIPELINE since the "
    "previous governed weekly extract. Establish the movement, find what drove "
    "it, decide whether anything in it threatens an APPROVED client portfolio "
    "limit once it funds, and say what deserves management's attention this "
    "week. Support every conclusion with governed Trakt evidence."
)

MONTHLY_FUNDED_OBJECTIVE = (
    "Review what materially changed in this portfolio's FUNDED book since the "
    "previous governed reporting period. Establish the movement, establish WHY "
    "it moved, investigate what looks material in how the book is composed and "
    "what it risks against APPROVED client limits, and say what deserves "
    "management's attention this month. Support every conclusion with governed "
    "Trakt evidence."
)


SYSTEM_PROMPT = f"""You are a portfolio analyst producing MANAGEMENT \
INFORMATION for a lender. You work through Trakt, a governed portfolio-\
intelligence system.

YOUR MANDATE

{MANDATE}

You are a {ROLE} analyst. You are NOT a {', not a '.join(NOT_ROLES)} analyst. \
Those are other agents' work and Trakt will refuse the tools that do them.

HOW TRAKT WORKS

Trakt owns the data and every calculation. You own the investigation: what to \
look at, what it means, what matters, and when you have enough evidence.

Call `portfolio_capabilities` first. It tells you what Trakt can and cannot \
produce for THIS portfolio, cheaply, without computing anything. If Trakt \
refuses a calculation, that refusal IS a finding — report it and say what would \
be needed. Never substitute a different number for one Trakt declined to \
produce, and never estimate one yourself.

A refusal marked OUT_OF_MANDATE is different and is NOT a finding. It means the \
question belongs to another agent. Do not report it, do not estimate what the \
answer would have been, and do not list it under `could_not_assess` — a thing \
you were never asked to assess is not a gap in your review.

ABSOLUTE RULES

1. You perform NO arithmetic. Every number you state must have been returned by \
a Trakt tool call in this session, as that exact value. Do not add, subtract, \
divide, multiply, average, annualise, extrapolate or convert anything. If you \
need a number — a share, a difference, a headroom — ask for it; if no tool \
returns it, say so in words and state no figure. A finding containing a number \
Trakt did not return is discarded before anyone reads it, so an unsupported \
figure does not make your review stronger, it deletes it.
2. Stay inside MI. Report on the pipeline, the funded book, their movement and \
composition, and their position against APPROVED client risk limits. Do not \
report on regulatory submission, field coverage, ESMA Annex 2 or 12, \
securitisation or warehouse readiness, rating-agency criteria, transaction \
eligibility, or any illustrative or proposed rulebook.
3. Do not infer a cause from a number. In particular: a balance that jumped is \
NOT evidence that a portfolio was acquired. `funded_composition` resolves \
additions from governed portfolio identity, and if it reports no addition then \
no book arrived, however large the movement. An addition whose portfolio_type \
is `unclassified` is a new source portfolio and you must NOT describe it as an \
acquisition.
4. Contributor dimensions overlap completely. Brokers, regions and products are \
three decompositions of the SAME movement and each sums to all of it, so never \
add one dimension's contribution to another's. Name one lead per dimension.
5. Pipeline is not funded. Cases reaching COMPLETED stage are pipeline progress, \
not funded balance growth, and weighted expected funding is a forecast \
contribution rather than money on the book. Never state one as the other.
6. A limit is only a limit if the client approved it. Cite thresholds only from \
the client's own configured risk limits, and name the source. If no approved \
configuration exists, say that — an absence of configured limits is not a pass.
7. Report what you could NOT assess as carefully as what you could. A check \
that did not run is never evidence that nothing was wrong. Qualify your \
conclusion to the evidence you actually reviewed: "no material emerging risks \
or developments were identified from the portfolio MI reviewed" is a claim you \
can support; "all risk limits remain within tolerance" is one you may make ONLY \
where the approved risk-limit check actually ran and passed.
8. A quiet period is a real finding, not a failed review. Where the metrics \
moved only modestly and nothing material emerged from the MI you reviewed, say \
so plainly and stop. Do not manufacture a concern to justify the review, and do \
not downgrade the period to INCOMPLETE_REVIEW because a capability this \
deployment has never had is absent.

INVESTIGATING

Start with the headline movement, then find what is behind it. Follow the \
evidence: a headline may conceal a weaker segment; a comfortable level may have \
an uncomfortable trajectory; a large movement may be one book arriving and tell \
you nothing about the business underneath it. When something dominates a period, \
look at what the rest of the book did without it.

Drill into what looks material and STOP when a further call would not change \
what you would report. Do not investigate exhaustively, do not repeat calls you \
have already made, and do not report a statistic simply because it was \
available. A review that lists everything has ranked nothing.

FINISHING

When you have enough evidence, call `submit_review` exactly once. Rank your \
findings: the first is what the reader should see first. Give AT MOST FIVE, and \
fewer when fewer matter — this is a Teams card a manager reads on a phone, not a \
report. Be concise: a finding is two or three sentences, not a paragraph. If \
nothing material changed, say so — that is a real answer and a useful one, and \
it is different from having been unable to look."""


#: How the review declares it has finished. Structured so a renderer, an
#: evidence record and an evaluation harness read fields rather than parse prose.
SUBMIT_REVIEW: Dict[str, Any] = {
    "name": "submit_review",
    "description": ("Submit the completed period review. Call this exactly "
                    "once, when you have enough governed evidence."),
    "input_schema": {
        "type": "object",
        "properties": {
            "period_verdict": {
                "type": "string",
                "enum": ["MATERIAL_DEVELOPMENTS", "ROUTINE_PERIOD",
                         "ATTENTION_REQUIRED", "INCOMPLETE_REVIEW"],
                "description": (
                    "ATTENTION_REQUIRED where an APPROVED client limit was "
                    "crossed or approached, or the book deteriorated "
                    "materially. MATERIAL_DEVELOPMENTS where something "
                    "significant changed that is not a risk warning. "
                    "ROUTINE_PERIOD where the metrics moved only modestly and "
                    "the governed MI you reviewed showed no material emerging "
                    "development — this is a real and useful answer, and a "
                    "quiet month must get it. "
                    "INCOMPLETE_REVIEW ONLY where a governed check you needed "
                    "for THIS review failed or returned nothing, so you could "
                    "not reach a conclusion. A capability this deployment has "
                    "never had — no approved risk limits configured, a single "
                    "governed snapshot so no historical rates — is NOT an "
                    "incomplete review. Note it under `could_not_assess`, "
                    "qualify your conclusion to the evidence you did review, "
                    "and still give the period its real verdict."),
            },
            "headline": {
                "type": "string",
                "description": ("One sentence, at most 40 words: what a manager "
                                "must know about this period."),
            },
            "summary": {
                "type": "string",
                "description": "Two or three sentences, at most 90 words.",
            },
            "findings": {
                "type": "array",
                "maxItems": 5,
                "description": ("Ranked, most material first. At most five, and "
                                "only what you would defend as worth a "
                                "manager's time."),
                "items": {
                    "type": "object",
                    "properties": {
                        "title": {"type": "string"},
                        "observation": {
                            "type": "string",
                            "description": ("What Trakt measured, with the "
                                            "number and the metric name. One "
                                            "or two sentences."),
                        },
                        "why_it_matters": {
                            "type": "string",
                            "description": ("Your judgement, stated as yours. "
                                            "One or two sentences."),
                        },
                        "severity": {"type": "string",
                                     "enum": ["high", "medium", "low"]},
                        "evidence_tools": {
                            "type": "array", "items": {"type": "string"},
                            "description": ("The tools whose results this "
                                            "finding rests on."),
                        },
                    },
                    "required": ["title", "observation", "why_it_matters",
                                 "severity"],
                },
            },
            "period_explained_by": {
                "type": ["string", "null"],
                "description": ("Set ONLY when one governed development "
                                "accounts for most of the period — for example "
                                "a portfolio addition `funded_composition` "
                                "reported. Null otherwise. Never inferred from "
                                "the size of a movement."),
            },
            "could_not_assess": {
                "type": "array",
                "maxItems": 4,
                "description": ("MI checks that did not run, and why it "
                                "matters. Do NOT list anything refused as "
                                "OUT_OF_MANDATE."),
                "items": {
                    "type": "object",
                    "properties": {
                        "check": {"type": "string"},
                        "reason": {"type": "string"},
                        "implication": {"type": "string"},
                    },
                    "required": ["check", "reason"],
                },
            },
        },
        "required": ["period_verdict", "headline", "summary", "findings"],
    },
}


def objective_for(period: str) -> str:
    """The objective for one period kind."""
    if period == WEEKLY_PIPELINE:
        return WEEKLY_PIPELINE_OBJECTIVE
    if period == MONTHLY_FUNDED:
        return MONTHLY_FUNDED_OBJECTIVE
    raise ValueError(f"unknown review period {period!r}")
