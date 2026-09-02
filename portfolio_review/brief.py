"""What actually reaches Teams: a ranked selection, inside a word budget.

THE PROBLEM THIS SOLVES
-----------------------
The autonomous reviews the red-team collected ran 744–1,011 words. The
deterministic card runs about 40. Nobody reads a thousand words in a Teams
notification, and a briefing nobody reads is worth less than a short one that is
merely adequate — the excess is not thoroughness, it is a failure to rank.

SELECTION, NOT TRUNCATION
-------------------------
Nothing here cuts a sentence short. It drops whole findings from the bottom of
the model's own ranking until the card fits, and says how many it dropped. A
half-sentence is worse than a missing paragraph: the reader cannot tell what was
lost, and a severed clause can invert a claim.

WHY THE BUDGET IS ENFORCED HERE AND NOT ASKED FOR IN THE PROMPT
---------------------------------------------------------------
The prompt does ask. It also asked for no arithmetic, and that held until it
didn't. Anything that must be true of what a lender receives belongs somewhere
deterministic, and "somewhere deterministic" is this module and
:mod:`portfolio_review.numeric_gate`. The prompt's job is to make the model
*want* to be brief so that selection rarely has to bite; the budget's job is to
make brevity true when it doesn't.

THE GAPS ARE NAMED, NOT ARGUED
------------------------------
``could_not_assess`` carried roughly a fifth of the words in the red-team runs,
most of it careful prose about implications. That reasoning is worth keeping —
it goes to the audit record — but the card names the checks and stops. A reader
who wants to know why a check could not run asks; a reader skimming a
notification needs to know only that something did not run.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

#: The card's ceilings. Findings first because a card is a ranking.
MAX_FINDINGS = 5
MAX_GAPS = 4
MAX_HEADLINE_WORDS = 40
MAX_SUMMARY_WORDS = 90

#: Total words across everything a reader sees. Chosen against the measured
#: extremes — ~40 for the deterministic card, 744–1,011 for the ungated agent —
#: to sit far enough above the former to carry the agent's added value and far
#: enough below the latter to be read on a phone.
MAX_CARD_WORDS = 260


def _words(text: Optional[str]) -> int:
    return len((text or "").split())


def _finding_words(finding: Dict[str, Any]) -> int:
    return sum(_words(finding.get(k))
               for k in ("title", "observation", "why_it_matters"))


@dataclass
class Card:
    """The Teams-facing review, and an honest account of what was left out."""

    period_verdict: str
    headline: str
    summary: str
    findings: List[Dict[str, Any]] = field(default_factory=list)
    gaps: List[str] = field(default_factory=list)
    period_explained_by: Optional[str] = None
    #: Findings the budget dropped, kept so the audit record is complete even
    #: though the card is not.
    withheld: List[Dict[str, Any]] = field(default_factory=list)
    notes: List[str] = field(default_factory=list)

    @property
    def word_count(self) -> int:
        return (_words(self.headline) + _words(self.summary)
                + sum(_finding_words(f) for f in self.findings)
                + sum(_words(g) for g in self.gaps))

    def to_dict(self) -> Dict[str, Any]:
        return {
            "period_verdict": self.period_verdict, "headline": self.headline,
            "summary": self.summary, "findings": list(self.findings),
            "could_not_assess": list(self.gaps),
            "period_explained_by": self.period_explained_by,
            "withheld_findings": list(self.withheld), "notes": list(self.notes),
            "word_count": self.word_count,
        }


def _over_long(label: str, text: str, limit: int) -> Optional[str]:
    return (f"{label} is {_words(text)} words against a {limit}-word limit"
            if _words(text) > limit else None)


def render(review: Dict[str, Any]) -> Card:
    """Select the card from a gated review.

    The review must already have passed :func:`portfolio_review.numeric_gate.apply`.
    This makes no judgement about whether a figure is supported — it assumes
    that was settled — and only about how much of a supported review fits.
    """
    findings = list(review.get("findings") or ())
    gaps = [str(g.get("check") or "").strip()
            for g in (review.get("could_not_assess") or ())]
    gaps = [g for g in gaps if g][:MAX_GAPS]

    card = Card(
        period_verdict=str(review.get("period_verdict") or ""),
        headline=str(review.get("headline") or "").strip(),
        summary=str(review.get("summary") or "").strip(),
        period_explained_by=review.get("period_explained_by"),
        gaps=gaps)

    #: Over-length prose is reported rather than cut: the model wrote a headline
    #: that does not fit, and a reviewer should see that it did.
    for note in (_over_long("headline", card.headline, MAX_HEADLINE_WORDS),
                 _over_long("summary", card.summary, MAX_SUMMARY_WORDS)):
        if note:
            card.notes.append(note)

    if len(findings) > MAX_FINDINGS:
        card.withheld.extend(findings[MAX_FINDINGS:])
        card.notes.append(
            f"kept the top {MAX_FINDINGS} of {len(findings)} findings")
        findings = findings[:MAX_FINDINGS]
    card.findings = findings

    # Drop from the bottom of the model's own ranking until it fits. One
    # finding always survives: a card with a headline and nothing under it is
    # not a briefing, and dropping the last finding to satisfy a budget the
    # SUMMARY blew would punish the wrong part of the review.
    trimmed = False
    while card.word_count > MAX_CARD_WORDS and len(card.findings) > 1:
        card.withheld.append(card.findings.pop())
        trimmed = True
    if trimmed:
        card.notes.append(f"trimmed to {card.word_count} words against a "
                          f"{MAX_CARD_WORDS}-word budget")

    # Selection cannot fix prose it does not control. Say so rather than let
    # the budget look like a guarantee it is not.
    if card.word_count > MAX_CARD_WORDS:
        card.notes.append(
            f"OVER BUDGET at {card.word_count} words: the headline and summary "
            f"alone are {_words(card.headline) + _words(card.summary)} words, "
            f"which selection cannot reduce")
    return card


def over_budget(card: Card) -> bool:
    """Did the card fail its own budget? Only possible via headline/summary."""
    return card.word_count > MAX_CARD_WORDS
