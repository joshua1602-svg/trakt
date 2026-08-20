"""The TYPE of an answer, and the type a question asks for.

Why this exists
---------------
The calibration bank pinned an answer's MEASURE (``expected_metric``) and its
SHAPE (``expected_artifact_type``). It pinned nothing for a COUNT, because a
count carries no measure — ``expected_metric`` is legitimately null. That left a
hole a defect walked through: "how many loans have a balance above £250k"
answered with a BALANCE and passed, because there was no field to disagree with.

Thirty-one of the bank's cases are counts. This module gives them, and every
other case, a declared answer TYPE that is independent of the measure, so the
class of defect where a count question returns currency is caught by the bank
rather than by an audit that happens to be run.

One classifier, used by both the bank's evaluator and the type-conformance
sweep. Two would be two things to keep in step.
"""

from __future__ import annotations

import re
from typing import Any, Mapping, Optional

#: The governed answer types.
CURRENCY = "currency"
COUNT = "count"
RATE = "rate"
AGE = "age"
DATE = "date"
#: A response that legitimately carries several measures at once — a portfolio
#: summary is a count AND a balance AND a weighted average.
MIXED = "mixed"
#: No data answer is expected: a refusal or a clarification.
NONE = "none"
#: The question names no measure, so the governed default applies and any
#: governed measure is acceptable. "breakdown by region" is the shape.
ANY = "any"

TYPES = (CURRENCY, COUNT, RATE, AGE, DATE, MIXED, NONE, ANY)

#: The question vocabulary, deliberately small and explicit, most specific
#: first. A question matching none of these names no measure.
_ASK_PATTERNS = (
    (DATE,     r"\bwhen\b|\bhow long until\b|\bby when\b|\bwhat date\b"),
    (COUNT,    r"\bhow many\b|\bnumber of\b|\bloan count\b|\bcase count\b|\bcount\b"),
    (RATE,     r"\bltv\b|\bloan to value\b|\binterest rate\b|\bwhat rate\b|"
               r"\bpercentage\b|\bproportion\b|\bshare of\b|\bwhat %|\bcoupon\b"),
    (AGE,      r"\bborrower age\b|\bage of\b|\bhow old\b"),
    (CURRENCY, r"\bbalance\b|\bexposure\b|\baum\b|\bamount\b|\bvalue of\b|"
               r"\bhow much\b|\bloan size\b|£"),
)
#: A milestone verb followed by an amount: the amount is the target the
#: question asks WHEN we reach, never the measure it asks for.
_MILESTONE_RE = re.compile(
    r"\b(reach|reaches|reached|cross|crosses|hit|hits|get to|gets to|"
    r"achieve|achieves|exceed|exceeds|grow to|grows to)\b[^.?]{0,40}?[£$]?\d")

#: A whole-book digest names no single measure.
_SUMMARY_RE = re.compile(r"\b(portfolio|book)\s+(summary|overview)\b|"
                         r"\bkey metrics\b|\bsummarise\b|\bsummarize\b|"
                         r"\boverview\b|\bsnapshot\b")


#: Clause openers that introduce a CONDITION. The same rule the parser applies
#: in ``llm_query_parser._metric_slot``, and for the same reason: a measure word
#: inside a condition names the field being filtered ON. The first version of
#: this classifier scanned the whole question and typed "balance by region where
#: borrower age is over 70" as an AGE question — the very defect the sweep was
#: built to find, reproduced in the instrument looking for it.
_CONDITION_OPENERS = ("where", "with", "for", "above", "below", "over", "under",
                      "greater than", "less than", "at least", "at most",
                      "more than", "fewer than")
_CONDITION_RE = re.compile(
    r"\b(" + "|".join(re.escape(t) for t in
                      sorted(_CONDITION_OPENERS, key=len, reverse=True)) + r")\b")


def subject_side(question: str) -> str:
    """The span of a question that may name the measure being reported.

    Truncates at the first condition clause, and only where a numeric bound
    follows it — so "loans with LTV above 50%" is cut while "regions with the
    highest LTV" is not.
    """
    head = (question or "")
    m = _CONDITION_RE.search(head)
    while m:
        if re.search(r"\d", head[m.end():]):
            return head[: m.start()].strip() or head.strip()
        m = _CONDITION_RE.search(head, m.end())
    return head.strip()


def asked(question: str) -> str:
    """The answer type a QUESTION asks for, from its wording alone."""
    text = (question or "").lower()
    if _SUMMARY_RE.search(text):
        return MIXED
    subject = subject_side(text)
    hits = [kind for kind, pattern in _ASK_PATTERNS if re.search(pattern, subject)]
    # A MILESTONE names its money as the target, not as the answer: "when do we
    # expect the funded book to reach GBP 100m?" wants a date, and the amount is
    # the input to it. Without this the currency cue fired on the threshold and
    # typed four date questions as asking for both.
    if DATE in hits and CURRENCY in hits and _MILESTONE_RE.search(subject):
        hits = [h for h in hits if h != CURRENCY]
    # A question that asks for a value AND a timing — "what's the value of
    # outstanding offers, and when do we expect them to fund?" — is not a date
    # question that happens to mention money. Both halves must be answered, so
    # neither type alone is the right expectation.
    if DATE in hits and len(hits) > 1:
        return MIXED
    if hits:
        return hits[0]
    # Nothing in the subject named a measure. A COUNT phrasing anywhere still
    # counts: "loans where LTV is over 50 — how many?" is a count question.
    for kind, pattern in _ASK_PATTERNS:
        if kind == COUNT and re.search(pattern, text):
            return kind
    return ANY


def of_measure(metric: Optional[str], aggregation: Optional[str] = None,
               semantics: Optional[Mapping[str, Any]] = None) -> str:
    """The answer type a RESULT has, from the measure and aggregation it used.

    Registry-declared format first; the measure's name only when the registry
    does not carry it — the pipeline measures live in the pipeline contract, not
    in the funded field registry, and typing them from the name is better than
    reporting every pipeline answer as untyped.
    """
    agg = (aggregation or "").lower()
    if agg in ("count", "count_distinct") and not metric:
        return COUNT
    if agg == "share":
        return RATE
    if not metric:
        return ANY
    fields = (semantics or {}).get("fields") or (semantics or {})
    entry = fields.get(metric) or {}
    fmt = str(entry.get("format") or "").lower()
    unit = str(entry.get("unit") or "").lower()
    if fmt in ("currency", "money") or unit in ("gbp", "currency"):
        return CURRENCY
    if fmt in ("percent", "percentage", "rate"):
        return RATE
    name = str(metric).lower()
    if "age" in name:
        return AGE
    if any(t in name for t in ("amount", "balance", "value", "exposure")):
        return CURRENCY
    if any(t in name for t in ("rate", "ltv", "yield", "conversion", "share")):
        return RATE
    if "count" in name:
        return COUNT
    return ANY


#: Which observed types satisfy each declared type. Deliberately not the
#: identity relation: a share question legitimately reports the balance basis
#: alongside the percentage, and a whole-book digest carries several measures.
_SATISFIES = {
    CURRENCY: {CURRENCY},
    COUNT:    {COUNT},
    RATE:     {RATE, CURRENCY},
    AGE:      {AGE},
    DATE:     {DATE, ANY, CURRENCY, COUNT},
    MIXED:    {MIXED, CURRENCY, COUNT, RATE, AGE, ANY},
    ANY:      set(TYPES),
    NONE:     set(TYPES),
}


def satisfies(expected: Optional[str], observed: str) -> bool:
    """Does an observed answer type satisfy the declared expectation?"""
    if not expected:
        return True
    return observed in _SATISFIES.get(expected, {expected})
