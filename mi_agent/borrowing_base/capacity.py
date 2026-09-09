"""THE BORROWING-BASE CAPABILITY'S OWN READING OF "HOW MUCH CAN WE BORROW".

A lender who asks *how much collateral value we are able to borrow against*,
or *how much we can draw against the facility*, is asking the borrowing-base
capability a question — the amount the governed borrowing base makes
available — not asking the funded book for a valuation total. The registry's
`current_valuation_amount` carries "collateral value" as a synonym, and the
generic measure binder had no way to know that here the words belong to a
different owner; so the question was answered as a £-valuation with ok=true.

This module is that owner's claim, stated ONCE as a general rule: a BORROWING
VERB governing a COLLATERAL or FACILITY noun through *against / on / from*.
It reads no question-specific alias and no acceptance question; the parser
asks it (through `mi_agent.capability_ownership`) before any measure binds,
and the borrowing-base route asks it to recognise the same question.
"""
from __future__ import annotations

import re
from typing import List, Optional, Tuple

CAPABILITY = "borrowing_base"

#: What the claim is about — the concept the capability answers with.
CONCEPT_BORROWING_CAPACITY = "borrowing_capacity"      # → measure `borrowing_base`
CONCEPT_FACILITY_HEADROOM = "facility_headroom"        # → measure `borrowing_base_headroom`

#: THE BORROWING VERBS ONLY. "lend" and "fund" were here and are gone:
#: measured on the 843-question corpus, "compare the front BOOK with our older
#: LENDING FROM a risk perspective" was claimed as a borrowing-base question,
#: because a gerund naming ORIGINATION sat between a collateral noun and a
#: preposition. A capability that claims a span it does not own is exactly the
#: defect this module exists to remove, so its vocabulary stays narrow.
_VERB = r"(?:borrow|draw(?:\s*down)?|drawdown|advance)\w*"
_PREP = r"(?:against|on|off|from|out of)"
_COLLATERAL = (r"(?:collateral(?:\s+value|\s+valuation|\s+pool)?|security|"
               r"assets?|book|portfolio|loans?|mortgages?|eligible\s+\w+)")
_FACILITY = r"(?:facility|facilities|line|warehouse|commitment)"
_NOUN = rf"(?P<noun>{_COLLATERAL}|{_FACILITY})"
_DET = r"(?:the|our|this|that|its|their)\s+"

#: What may stand between the noun and the borrowing verb: the modal/relative
#: scaffolding of "collateral value ARE WE ABLE TO borrow against". A
#: comparison, a second clause or a list is NOT scaffolding — the noun and the
#: verb are then not in one clause, and the claim would be a guess.
_SCAFFOLD = (r"(?:\s+(?:are|is|can|could|may|might|do|does|did|we|i|you|they|it|"
             r"the|our|able|allowed|permitted|eligible|available|left|room|"
             r"still|actually|really|to|be|been|get|then)\b){1,8}"
             r"|\s+")

# noun ... verb prep   — "collateral value are we able to borrow against"
_NOUN_FIRST = re.compile(
    rf"\b{_NOUN}\b(?:{_SCAFFOLD})\s*\b(?P<verb>{_VERB})\s+{_PREP}\b", re.I)
# verb prep [det] noun — "draw against the facility", "borrow on our book"
_VERB_FIRST = re.compile(
    rf"\b(?P<verb>{_VERB})\s+{_PREP}\s+(?:{_DET})?{_NOUN}\b", re.I)
_FACILITY_RE = re.compile(rf"^{_FACILITY}$", re.I)


def claims(question: Optional[str]) -> List[Tuple[str, str, int, int, str]]:
    """``[(capability, concept, start, end, evidence), ...]`` — every span of
    ``question`` this capability claims. Offsets index ``question`` as given."""
    text = str(question or "")
    out: List[Tuple[str, str, int, int, str]] = []
    seen: set = set()
    for pattern in (_NOUN_FIRST, _VERB_FIRST):
        for m in pattern.finditer(text):
            span = (m.start(), m.end())
            if span in seen:
                continue
            seen.add(span)
            noun = m.group("noun").strip().lower()
            concept = (CONCEPT_FACILITY_HEADROOM if _FACILITY_RE.match(noun.split()[0])
                       else CONCEPT_BORROWING_CAPACITY)
            out.append((CAPABILITY, concept, span[0], span[1], m.group(0)))
    return out


def measure_for(concept: str) -> str:
    """The governed borrowing-base measure a claimed concept is answered with."""
    return ("borrowing_base_headroom" if concept == CONCEPT_FACILITY_HEADROOM
            else "borrowing_base")


__all__ = ["CAPABILITY", "CONCEPT_BORROWING_CAPACITY", "CONCEPT_FACILITY_HEADROOM",
           "claims", "measure_for"]
