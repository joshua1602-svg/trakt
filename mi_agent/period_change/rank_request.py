#!/usr/bin/env python3
"""mi_agent/period_change/rank_request.py — what the user asked to rank.

Detection only, kept out of both the ranking function (which stays pure) and
the P0 guard (which stays an independent control rather than a parser).

A P1C question resolves to exactly four things:

    ONE measure      -> the ranking basis
    ONE dimension    -> which governed distribution to rank
    ONE direction    -> increase | decrease | either
    optional top N

Anything it cannot resolve confidently returns ``None`` so the caller keeps its
existing behaviour. Detection is deliberately narrow: a false positive turns a
governed period-change narrative into a ranking of the wrong thing.

The measure the user names DETERMINES the basis — the convention is stated here
rather than decided silently:

    "added the most balance", "grew the most"  -> absolute balance movement
    "grew fastest", "in percentage terms"      -> percentage balance movement
    "most loans", "biggest increase in cases"  -> absolute loan-count movement
    "increased its share the most"             -> share-of-balance movement

"grew the most" with no measure named is ABSOLUTE BALANCE movement. That is the
documented product convention: the book is measured in balance, and a bare
"grew" about a lending book means money, not proportion. It is stated in the
execution receipt on every answer, so the reader can see which basis ran.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Optional

from . import ranking as rk

# --------------------------------------------------------------------------- #
# Direction
# --------------------------------------------------------------------------- #
_INCREASE_RE = re.compile(
    r"\b(?:grew|grow|grown|growth|increas(?:e|ed|ing)|ris(?:e|en|ing)|"
    r"gain(?:ed|s)?|expand(?:ed|ing)?|added|up)\b", re.I)
_DECREASE_RE = re.compile(
    r"\b(?:declin(?:e|es|ed|ing)|f[ae]ll(?:s|en|ing)?|drop(?:s|ped|ping)?|"
    r"decreas(?:e|es|ed|ing)|shr(?:ank|unk|inking)|reduc(?:e|es|ed|tion|tions)|"
    r"lost|down)\b", re.I)
#: A superlative or explicit rank instruction — without one this is a narrative
#: period-change question, not a ranked one.
_RANK_RE = re.compile(
    r"\b(?:most|largest|biggest|greatest|highest|fastest|smallest|lowest|"
    r"top|rank|order)\b", re.I)
#: How many results the reader asked for. Deliberately three narrow shapes, so a
#: number that means something else — "the 3 months to June", "over 85" — is
#: never mistaken for a Top N:
#:
#:     "top 3", "largest 5"            a rank word immediately before a digit
#:     "three largest", "five biggest" a word-number immediately before a rank word
#:     "which five segments grew..."   "which" + a word-number opening the question
_TOP_N_RE = re.compile(
    r"\b(?:top|first|best|largest|biggest)\s+(?P<digits>\d+)\b|"
    r"\b(?P<word>three|four|five|ten)\s+(?:largest|biggest|greatest|smallest)\b|"
    r"\bwhich\s+(?P<which_word>three|four|five|ten)\b|"
    r"\bwhich\s+(?P<which_digits>\d+)\b", re.I)
_WORD_NUMBERS = {"three": 3, "four": 4, "five": 5, "ten": 10}

# --------------------------------------------------------------------------- #
# Basis — the measure decides
# --------------------------------------------------------------------------- #
_SHARE_RE = re.compile(r"\bshare\b|\bproportion\b|\bcomposition\b|\bmix\b", re.I)
_PERCENT_RE = re.compile(
    r"\bfastest\b|\bin percentage terms\b|\bpercentage\b|\bpercent\b|\b%\b|"
    r"\brelative growth\b", re.I)
_COUNT_RE = re.compile(
    r"\b(?:loan|loans|case|cases|account|accounts|number of)\b", re.I)
_BALANCE_RE = re.compile(
    r"\bbalance\b|\bexposure\b|\bbook\b|\bvalue\b|£", re.I)


@dataclass(frozen=True)
class RankRequest:
    """A resolved P1C ranking instruction."""

    basis: str
    direction: str
    top_n: Optional[int]
    #: The dimension term the user named ("region", "ltv band"), lowercased.
    dimension_term: str

    @property
    def basis_label(self) -> str:
        return rk.BASIS_LABELS.get(self.basis, self.basis)


def has_rank_language(question: str) -> bool:
    """True when the question contains a superlative or rank instruction.

    The cheap first gate: a caller uses it to decide whether ranking is even in
    play before doing the more expensive dimension resolution.
    """
    return bool(question and _RANK_RE.search(question))


def _detect_top_n(question: str) -> Optional[int]:
    match = _TOP_N_RE.search(question or "")
    if not match:
        return None
    digits = match.group("digits") or match.group("which_digits")
    if digits:
        try:
            value = int(digits)
        except ValueError:
            return None
        return value if value > 0 else None
    word = match.group("word") or match.group("which_word") or ""
    return _WORD_NUMBERS.get(word.lower())


def _detect_basis(question: str) -> str:
    q = question or ""
    if _SHARE_RE.search(q):
        return rk.BASIS_BALANCE_SHARE
    if _PERCENT_RE.search(q):
        return rk.BASIS_BALANCE_PERCENT
    # A count basis needs the count noun to be what GREW, not just present:
    # "which region grew the most" mentions no measure and means balance.
    if _COUNT_RE.search(q) and not _BALANCE_RE.search(q):
        return rk.BASIS_COUNT_ABSOLUTE
    return rk.BASIS_BALANCE_ABSOLUTE


def detect_rank_request(question: str, dimension_term: Optional[str]
                        ) -> Optional[RankRequest]:
    """The ranking instruction in ``question``, or None if it is not a ranked
    question. ``dimension_term`` is the dimension the caller already resolved —
    ranking without a dimension to rank is not a P1C question."""
    if not question or not dimension_term:
        return None
    if not _RANK_RE.search(question):
        return None

    decrease = bool(_DECREASE_RE.search(question))
    increase = bool(_INCREASE_RE.search(question))
    if decrease and not increase:
        direction = rk.DIRECTION_DECREASE
    elif increase and not decrease:
        direction = rk.DIRECTION_INCREASE
    elif re.search(r"\bmovement|moved|chang(?:e|ed|es)\b", question, re.I):
        direction = rk.DIRECTION_ANY
    elif increase and decrease:
        # Both named ("biggest increases and decreases") — rank by magnitude
        # rather than guessing which the reader meant.
        direction = rk.DIRECTION_ANY
    else:
        # "rank regions by growth" / "top 3 regions": a bare rank instruction
        # with no verb is an increase ranking.
        direction = rk.DIRECTION_INCREASE

    return RankRequest(basis=_detect_basis(question), direction=direction,
                       top_n=_detect_top_n(question),
                       dimension_term=str(dimension_term).lower())
