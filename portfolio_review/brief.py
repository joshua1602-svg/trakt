"""What actually reaches Teams: a ranked selection, judged on quality not length.

THE PROBLEM THIS SOLVES
-----------------------
The autonomous reviews the red-team collected ran 744–1,011 words. The
deterministic card runs about 40. Nobody reads a thousand words in a Teams
notification, and the excess was not thoroughness — it was a failure to rank.

LENGTH IS A SYMPTOM, NOT THE DEFECT
-----------------------------------
An earlier revision made 260 words a hard pass/fail line. That was the wrong
test and it is gone. The commercial requirement is that the message reads like a
concise management briefing rather than an essay or a tool dump, and a word
count answers neither question: a padded 200-word card fails it and a dense
300-word one passes. Worse, optimising for the count costs exactly the context
that makes a briefing worth sending.

So this module now does two separable things. ``render`` **selects** — a card is
a ranking, and the eleventh finding is not worth a manager's time whatever its
length. ``quality_flags`` **judges**, on the failures that actually make a
briefing bad: repetition, methodology exposition, raw tool output, duplicate
findings, hedging stacked until the claim disappears. Going over the soft guide
is a note on the card, never a failure.

SELECTION, NOT TRUNCATION
-------------------------
Nothing here cuts a sentence short. It drops whole findings from the bottom of
the model's own ranking, and says how many it dropped. A half-sentence is worse
than a missing paragraph: the reader cannot tell what was lost, and a severed
clause can invert a claim.

WHY SELECTION IS ENFORCED HERE AND NOT ASKED FOR IN THE PROMPT
---------------------------------------------------------------
The prompt does ask. It also asked for no arithmetic, and that held until it
didn't. Anything that must be true of what a lender receives belongs somewhere
deterministic, and "somewhere deterministic" is this module and
:mod:`portfolio_review.numeric_gate`.

THE GAPS ARE NAMED, NOT ARGUED
------------------------------
``could_not_assess`` carried roughly a fifth of the words in the red-team runs,
most of it careful prose about implications. That reasoning is worth keeping —
it goes to the audit record — but the card names the checks and stops. A reader
who wants to know why a check could not run asks; a reader skimming a
notification needs to know only that something did not run.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

#: The card's ceilings. Findings first because a card is a ranking.
MAX_FINDINGS = 5
MAX_GAPS = 4
MAX_HEADLINE_WORDS = 40
MAX_SUMMARY_WORDS = 90

#: A soft ceiling, deliberately not a pass/fail threshold.
#:
#: An earlier revision treated 260 words as a hard limit and a card over it as a
#: defect. That is the wrong test: the commercial requirement is that the
#: message reads like a management briefing rather than an essay or a tool dump,
#: and a genuinely useful 300-word briefing is better than a thin 200-word one.
#: Optimising for the count would cost exactly the context that makes the
#: message worth sending.
#:
#: So selection still trims — a card is a ranking, and the eleventh finding is
#: not worth a manager's time whatever its length — but going over is reported
#: as a note, never as a failure. :func:`quality_flags` carries the tests that
#: DO indicate a bad briefing.
SOFT_CARD_WORDS = 320


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

    # Drop from the bottom of the model's own ranking until it fits the soft
    # ceiling. One finding always survives: a card with a headline and nothing
    # under it is not a briefing, and dropping the last finding to satisfy a
    # length the SUMMARY blew would punish the wrong part of the review.
    trimmed = False
    while card.word_count > SOFT_CARD_WORDS and len(card.findings) > 1:
        card.withheld.append(card.findings.pop())
        trimmed = True
    if trimmed:
        card.notes.append(f"trimmed to {card.word_count} words against a "
                          f"{SOFT_CARD_WORDS}-word guide")
    elif card.word_count > SOFT_CARD_WORDS:
        card.notes.append(
            f"{card.word_count} words, above the {SOFT_CARD_WORDS}-word guide — "
            f"a note, not a defect")
    return card


#: Phrasing that belongs in a methodology appendix, not a management briefing.
_METHODOLOGY = re.compile(
    r"\b(methodolog\w+|is calculated as|is computed|we (?:then )?(?:ran|called)|"
    r"the tool returned|per the formula|by construction|as defined in|"
    r"weighted by balance across)\b", re.I)

#: Raw machine output leaking into prose a person reads.
_RAW_OUTPUT = re.compile(r"[\{\[]\s*['\"]|=>|\bNone\b|\bnull\b|_id\b|\bdict\(")

#: Hedging stacked until the claim disappears.
_QUALIFIERS = re.compile(
    r"\b(may|might|could|possibly|potentially|appears? to|seems? to|"
    r"suggests?|arguably|it is unclear|cannot be (?:fully )?determined)\b", re.I)


def quality_flags(card: Card) -> List[str]:
    """What actually makes a briefing bad. Length is not on this list.

    The commercial test is whether the message reads like a concise management
    briefing rather than an essay or a tool dump, and a word count answers
    neither question — a padded 200-word card fails and a dense 300-word one
    passes. These are the failures worth naming, each observable in the text.
    """
    flags: List[str] = []
    body = [card.headline, card.summary] + [
        f"{f.get('title','')} {f.get('observation','')} {f.get('why_it_matters','')}"
        for f in card.findings]
    joined = " ".join(t for t in body if t)

    if len(card.findings) > MAX_FINDINGS:
        flags.append(f"{len(card.findings)} findings, above the "
                     f"{MAX_FINDINGS} a card should carry")

    titles = [str(f.get("title") or "").strip().lower() for f in card.findings]
    if len(titles) != len(set(titles)):
        flags.append("duplicate finding titles")

    # Two findings whose observations share most of their vocabulary are one
    # finding written twice — the commonest way a card gets long without
    # getting more useful.
    observations = [set(str(f.get("observation") or "").lower().split())
                    for f in card.findings]
    for i, a in enumerate(observations):
        for b in observations[i + 1:]:
            if a and b and len(a & b) / max(len(a), len(b)) > 0.6:
                flags.append("two findings substantially repeat each other")
                break
        else:
            continue
        break

    if _METHODOLOGY.search(joined):
        flags.append("explains method rather than reporting a result")
    if _RAW_OUTPUT.search(joined):
        flags.append("carries raw tool output")

    hedges = len(_QUALIFIERS.findall(joined))
    if hedges > 2 + len(card.findings):
        flags.append(f"{hedges} hedging phrases — the claims are qualified away")
    return flags
