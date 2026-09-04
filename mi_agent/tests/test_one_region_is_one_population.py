#!/usr/bin/env python3
"""A region filter selects EVERY row of that region, or it refuses.

THE DEFECT, measured against the live ERE/2026-06-30 categories. The funded tape
carries the direct and acquired books' own spellings side by side, so "balance by
region" returns 24 groups for about a dozen regions:

    YORKSHIRE AND HUMBERSIDE   49 loans
    Yorkshire and humberside   11 loans
    Yorkshire & Humberside      5 loans

Filtering is case-insensitive, so LONDON/London and SOUTH WEST/South West are
already one population — proven below, because an assumption is not evidence.
`&` is not case. Asked for "Yorkshire & Humberside" the executor returned FIVE of
the sixty-five loans and presented it as the answer: 7.7% of the balance, no
warning, no refusal. That is a wrong number, which is worse than a refusal.

WHO OWNS THIS. `engine.region_taxonomy` already resolves every one of those
spellings to one canonical region — it cleans `&` to `and`, folds punctuation and
separators, and carries the client's approved synonym table. It is the governed
answer to "are these the same region". `mi_agent.region_resolution` was a second,
weaker implementation of the same question: `strip().lower()`, which cannot see
through an ampersand. Two owners of one concept, disagreeing.

So the taxonomy is consulted rather than re-implemented, and the executor asks
the domain resolver on EVERY categorical filter rather than only when the exact
match reached nothing — a partial exact match is precisely the case that must not
short-circuit it.

No geography is named here. A tape with no taxonomy configured resolves exactly
as it did before.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from mi_agent import region_resolution as RR
from mi_agent.mi_query_executor import execute_mi_query
from mi_agent.mi_query_spec import MIQuerySpec
from mi_agent.mi_query_validator import load_mi_semantics

_FIELD = "collateral_geography"
_BALANCE = "current_outstanding_balance"

#: The live categories, with their real loan counts and balances.
LIVE = [("LONDON", 61, 18670647.0), ("London", 5, 1790083.0),
        ("SOUTH WEST", 117, 21320506.0), ("South West", 24, 3277539.22),
        ("SCOTLAND", 42, 6083538.0), ("Scotland", 10, 819567.84),
        ("YORKSHIRE AND HUMBERSIDE", 49, 7212476.0),
        ("Yorkshire and humberside", 11, 1364275.69),
        ("Yorkshire & Humberside", 5, 366941.0)]

#: The oracle, computed from LIVE rather than from a previous run.
TRUTH = {"london": 66, "south west": 141, "scotland": 52, "yorkshire": 65}


def _semantics():
    return load_mi_semantics(_REPO_ROOT / "mi_agent" /
                             "mi_semantics_field_registry.yaml")


def _book():
    rows = []
    for name, n, bal in LIVE:
        rows += [{_FIELD: name, _BALANCE: bal / n} for _ in range(n)]
    return pd.DataFrame(rows)


def _rows_for(term, frame=None, field=_FIELD):
    spec = MIQuerySpec(intent="summary", metric=_BALANCE, aggregation="sum",
                       filters={field: term})
    result = execute_mi_query(spec, _book() if frame is None else frame,
                              _semantics())
    recon = (result.metadata or {}).get("reconciliation") or {}
    return recon.get("records_after_filters")


# ------------------------------------------------- what already worked, pinned #
def test_case_alone_was_already_one_population():
    """Proven, not assumed — these three were never the defect."""
    assert _rows_for("London") == TRUTH["london"]
    assert _rows_for("South West") == TRUTH["south west"]
    assert _rows_for("Scotland") == TRUTH["scotland"]


# ------------------------------------------------------------ the partial gap #
def test_an_ampersand_spelling_selects_the_whole_region():
    """5 of 65 before this. A number that is 7.7% of the truth."""
    assert _rows_for("Yorkshire & Humberside") == TRUTH["yorkshire"]


def test_every_spelling_of_one_region_selects_the_same_population():
    got = {t: _rows_for(t) for t in
           ("Yorkshire & Humberside", "Yorkshire and humberside",
            "YORKSHIRE AND HUMBERSIDE")}
    assert set(got.values()) == {TRUTH["yorkshire"]}, got


def test_the_governed_canonical_name_reaches_the_book_that_spells_it_otherwise():
    """"Yorkshire and The Humber" is the taxonomy's canonical name; the tape
    writes "YORKSHIRE AND HUMBERSIDE". They are one region."""
    assert _rows_for("Yorkshire and The Humber") == TRUTH["yorkshire"]


# --------------------------------------------------- and it must not OVER-widen #
def test_a_region_filter_never_reaches_another_region():
    """The whole book is 324 rows. Widening to the governed equivalents must
    never spill into a different region."""
    for term, expected in (("London", TRUTH["london"]),
                           ("Scotland", TRUTH["scotland"]),
                           ("South West", TRUTH["south west"])):
        assert _rows_for(term) == expected, term


def test_a_region_the_book_does_not_hold_still_reaches_nothing():
    """An empty result is a real answer — "no exposure here" — and the workflow
    turns it into a controlled refusal. Widening must not invent a population."""
    assert _rows_for("Northern Ireland") == 0


def test_the_populations_partition_the_book():
    total = sum(n for _, n, _ in LIVE)
    covered = (_rows_for("London") + _rows_for("South West")
               + _rows_for("Scotland") + _rows_for("Yorkshire & Humberside"))
    assert covered == total


# --------------------------------------------------------------- the resolver #
def test_the_resolver_delegates_to_the_governed_taxonomy():
    present = [name for name, _, _ in LIVE]
    got = set(RR.resolve("Yorkshire & Humberside", present))
    assert got == {"YORKSHIRE AND HUMBERSIDE", "Yorkshire and humberside",
                   "Yorkshire & Humberside"}, got


def test_the_resolver_still_answers_nothing_for_an_absent_region():
    assert RR.resolve("Northern Ireland", [n for n, _, _ in LIVE]) == []


def test_a_book_with_no_region_values_resolves_nothing():
    assert RR.resolve("London", []) == []
