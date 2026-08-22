#!/usr/bin/env python3
"""question_interpretation/mi_phrasing_bank.py

A WIDER PHRASING BANK for the eight time-series shapes.

Why
---
Reachability measured against the declared 29 phrasings is a FLOOR, not a
ceiling: "no declared wording reaches this" is a weaker statement than "no
wording reaches this", and every phrasing that turns out reachable moves work
out of a build.

These are written as a LENDER WOULD TYPE THEM — colloquial, plural, elliptical,
often with no verb from the existing vocabulary — and deliberately NOT derived
from `time_series_surface.SHAPES`. They were written before they were run, and
none was adjusted after seeing its result: a bank tuned against its own outcome
measures nothing.

Each entry reuses its shape's declared `want` fragments, so a phrasing here is
rated by exactly the rule the declared bank is rated by.

    python -m question_interpretation.mi_capability_recontent --bank widened
"""
from __future__ import annotations

from typing import Dict, List, Tuple

#: shape id -> additional phrasings. Four per shape, 32 in total.
WIDENED: Dict[str, Tuple[str, ...]] = {
    "T1": (
        "what has the book done over the last few periods",
        "give me the balance trend",
        "how is the loan book tracking month to month",
        "outstanding balances by period",
    ),
    "T2": (
        "how have the big loans moved over time",
        "balance trend for high LTV lending",
        "how is the front book tracking over the periods",
        "trend for loans above 150k",
    ),
    "T3": (
        "how are the regions trending",
        "give me balances per region each month",
        "regional balance trend",
        "how has each region moved over the periods",
    ),
    "T4": (
        "how are the regions trending for the big loans",
        "regional trend for the front book",
        "balances per region each month for loans above 150k",
        "how has each LTV band moved for the acquired book",
    ),
    "T5": (
        "balances by region and LTV band each month",
        "how have regions and ticket sizes moved over time",
        "monthly balance split by region and band",
        "give me region by LTV band over the periods",
    ),
    "T6": (
        "what changed by region since last time",
        "region movement between the last two periods",
        "how much did each region move last month",
        "movement by LTV band period on period",
    ),
    "T7": (
        "which regions are growing fastest",
        "top growing regions",
        "who grew most over the periods",
        "rank the LTV bands by movement",
    ),
    "T8": (
        "direct and acquired balances side by side",
        "how do the front and back books compare over the periods",
        "acquired against direct, balance trend",
        "front book and back book movement",
    ),
}


def widened_phrasings() -> List[Tuple[str, str, str, List[str]]]:
    """(shape id, shape name, question, want fragments), reusing SHAPES' wants."""
    from question_interpretation.time_series_surface import SHAPES
    wants = {sid: (name, list(want)) for sid, name, _qs, want in SHAPES}
    out = []
    for sid, questions in WIDENED.items():
        name, want = wants[sid]
        for q in questions:
            out.append((sid, name, q, want))
    return out


def combined_phrasings() -> List[Tuple[str, str, str, List[str]]]:
    """The declared 29 plus these 32 — 61 in total."""
    from question_interpretation.time_series_surface import SHAPES
    declared = [(sid, name, q, list(want))
                for sid, name, qs, want in SHAPES for q in qs]
    return declared + widened_phrasings()
