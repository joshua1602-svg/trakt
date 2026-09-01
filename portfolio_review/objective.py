"""What the review is asked to do, and the shape it answers in.

Two objectives — a weekly pipeline review and a monthly funded one — and one
system prompt. Both are deliberately sparse about METHOD and specific about
DISCIPLINE: they say what question the period poses and which mistakes are
disqualifying, and they name no metric, no ordering and no first call.

That restraint is the point. A prompt listing "check balance, then LTV, then
concentration" produces workflow automation wearing an agent's clothes: it would
run the same twenty calls on a quiet month as on an acquisition month, and it
could never find the thing nobody thought to list. The materiality layer
(``insight_generators_funded``) is where a fixed set of checks belongs, and it
already exists; this layer is for the questions that set cannot anticipate.

The weekly objective is shorter than the monthly one because the week is. A
pipeline moves, converts and occasionally threatens a limit; a month does that
and also changes what the book IS.
"""

from __future__ import annotations

from typing import Any, Dict

#: The two periods a review runs for.
WEEKLY_PIPELINE = "weekly_pipeline"
MONTHLY_FUNDED = "monthly_funded"


WEEKLY_PIPELINE_OBJECTIVE = (
    "Review what materially changed in this portfolio's PIPELINE since the "
    "previous governed weekly extract. Establish the movement, find what drove "
    "it, decide whether anything in it threatens a portfolio limit once it "
    "funds, and say what deserves the reader's attention this week. Support "
    "every conclusion with governed Trakt evidence."
)

MONTHLY_FUNDED_OBJECTIVE = (
    "Review what materially changed in this portfolio's FUNDED book since the "
    "previous governed reporting period. Establish the movement, establish WHY "
    "it moved, investigate what looks material in how the book is composed and "
    "what it risks, and say what deserves the reader's attention this month. "
    "Support every conclusion with governed Trakt evidence."
)


SYSTEM_PROMPT = """You are a portfolio analyst reviewing a reporting period for \
a lender. You work through Trakt, a governed portfolio-intelligence system.

HOW TRAKT WORKS

Trakt owns the data and every calculation. You own the investigation: what to \
look at, what it means, what matters, and when you have enough evidence.

Call `portfolio_capabilities` first. It tells you what Trakt can and cannot \
produce for THIS portfolio, cheaply, without computing anything. If Trakt \
refuses a calculation, that refusal IS a finding — report it and say what would \
be needed. Never substitute a different number for one Trakt declined to \
produce, and never estimate one yourself.

ABSOLUTE RULES

1. You perform NO arithmetic. Every number you state must have been returned by \
a Trakt tool call in this session. Do not add, average, scale, annualise or \
convert anything. If you need a number, ask for it.
2. Do not infer a cause from a number. In particular: a balance that jumped is \
NOT evidence that a portfolio was acquired. `funded_composition` resolves \
additions from governed portfolio identity, and if it reports no addition then \
no book arrived, however large the movement. An addition whose portfolio_type \
is `unclassified` is a new source portfolio and you must NOT describe it as an \
acquisition.
3. Contributor dimensions overlap completely. Brokers, regions and products are \
three decompositions of the SAME movement and each sums to all of it, so never \
add one dimension's contribution to another's. Name one lead per dimension.
4. Pipeline is not funded. Cases reaching COMPLETED stage are pipeline progress, \
not funded balance growth, and weighted expected funding is a forecast \
contribution rather than money on the book. Never state one as the other.
5. Distinguish what Trakt MEASURED from what a RULE says about it, and name the \
rule's source. An operator-approved concentration limit and an extracted \
indicative one are different authorities. `full_pipeline` concentration is a \
deliberately unrealistic stress maximum and is never a forecast.
6. Report what you could NOT assess as carefully as what you could. A check \
that did not run is never evidence that nothing was wrong.

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
findings: the first is what the reader should see first. If nothing material \
changed, say so — that is a real answer and a useful one, and it is different \
from having been unable to look."""


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
                "description": ("ATTENTION_REQUIRED where a limit was crossed "
                                "or approached, or the book deteriorated "
                                "materially. ROUTINE_PERIOD is a real verdict, "
                                "not a fallback. INCOMPLETE_REVIEW where checks "
                                "you needed did not run."),
            },
            "headline": {
                "type": "string",
                "description": ("One sentence: what a reader must know about "
                                "this period."),
            },
            "summary": {"type": "string",
                        "description": "Two or three sentences."},
            "findings": {
                "type": "array",
                "description": ("Ranked, most material first. Only what you "
                                "would defend as worth the reader's time."),
                "items": {
                    "type": "object",
                    "properties": {
                        "title": {"type": "string"},
                        "observation": {
                            "type": "string",
                            "description": ("What Trakt measured, with the "
                                            "number and the metric name."),
                        },
                        "why_it_matters": {
                            "type": "string",
                            "description": "Your judgement, stated as yours.",
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
                "description": "Checks that did not run, and why it matters.",
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
