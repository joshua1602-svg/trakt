#!/usr/bin/env python3
"""tests/test_dimension_selection.py

Which dimensions earn a panel — the shared rule both surfaces read.

A breakdown with one meaningful category is not insight. "Broker / channel:
Direct 100%" spends a panel restating a fact the reader already had, and on a
four-panel matrix it displaces a dimension that would have said something.

These pin the rule itself. It lives in ``mi_agent_api.presentation`` — the
existing owner of shared display semantics — so React and the deck cannot
disagree about which cut is worth showing.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from mi_agent_api import presentation as P  # noqa: E402


def bars(*pairs):
    return [{"label": label, "balance": float(value)} for label, value in pairs]


# --------------------------------------------------------------------------- #
# Informativeness.
# --------------------------------------------------------------------------- #

def test_a_single_category_is_not_a_distribution():
    """Catches: "Pipeline amount by broker / channel — Direct, £7.8MM"."""
    shape = P.dispersion(bars(("Direct", 7_800_000)))
    assert shape["informative"] is False
    assert "one category" in shape["reason"]


def test_a_sliver_second_category_is_not_a_distribution_either():
    """Catches: one full-width bar and a hairline, drawn as a comparison.

    Two categories is not enough on its own — the second has to carry weight.
    """
    assert P.is_informative(bars(("A", 999.0), ("B", 1.0))) is False


def test_a_real_spread_is_informative():
    assert P.is_informative(bars(("A", 40), ("B", 35), ("C", 25))) is True


def test_a_dominant_but_not_degenerate_split_still_earns_its_panel():
    """A book genuinely concentrated in one band is a finding worth showing —
    the rule suppresses the uninformative, not the uncomfortable."""
    assert P.is_informative(bars(("A", 80), ("B", 12), ("C", 8))) is True


def test_an_empty_or_zero_distribution_is_never_informative():
    assert P.is_informative([]) is False
    assert P.is_informative(bars(("A", 0), ("B", 0))) is False


# --------------------------------------------------------------------------- #
# Selection.
# --------------------------------------------------------------------------- #

def _candidates():
    return [
        {"key": "broker", "label": "By broker / channel",
         "bars": bars(("Direct", 100))},
        {"key": "ltv", "label": "By LTV band",
         "bars": bars(("20-30%", 30), ("30-40%", 25), ("40-50%", 25), ("50-60%", 20))},
        {"key": "region", "label": "By region",
         "bars": bars(("London", 40), ("Wales", 35), ("Scotland", 25))},
        {"key": "ticket", "label": "By ticket size",
         "bars": bars(("100-150k", 50), ("150-200k", 30), ("200-300k", 20))},
    ]


def test_an_uninformative_dimension_is_never_selected():
    """Catches: broker/channel = 100% Direct taking one of four panel slots."""
    out = P.select_dimensions(_candidates(), want=3,
                              preferred=("ltv", "ticket", "age", "region"))
    keys = [e["key"] for e in out["selected"]]
    assert "broker" not in keys, keys
    assert len(keys) == 3


def test_the_next_best_dimension_takes_the_freed_slot():
    """The panel is not left empty because the default lost its place."""
    out = P.select_dimensions(_candidates(), want=3,
                              preferred=("broker", "ltv", "ticket", "region"))
    assert [e["key"] for e in out["selected"]] == ["ltv", "ticket", "region"]


def test_selection_is_deterministic():
    first = [e["key"] for e in
             P.select_dimensions(_candidates(), want=2,
                                 preferred=("ltv", "region"))["selected"]]
    for _ in range(5):
        again = [e["key"] for e in
                 P.select_dimensions(list(reversed(_candidates())), want=2,
                                     preferred=("ltv", "region"))["selected"]]
        assert again == first, (again, first)


def test_the_preferred_order_is_honoured_where_it_is_informative():
    out = P.select_dimensions(_candidates(), want=2,
                              preferred=("region", "ltv", "ticket"))
    assert [e["key"] for e in out["selected"]] == ["region", "ltv"]


def test_nothing_is_dropped_silently():
    """Every candidate that does not make the page carries the reason it lost —
    which is what the methodology ledger prints."""
    out = P.select_dimensions(_candidates(), want=2,
                              preferred=("ltv", "ticket", "region"))
    assert len(out["selected"]) == 2
    rejected = {r["key"]: r["reason"] for r in out["rejected"]}
    assert set(rejected) == {"broker", "region"}
    assert all(rejected.values()), rejected
    assert "one category" in rejected["broker"]
    assert "more informative" in rejected["region"]


def test_asking_for_more_than_exists_returns_what_exists():
    out = P.select_dimensions(_candidates(), want=10, preferred=())
    assert len(out["selected"]) == 3      # broker is not informative
    assert [r["key"] for r in out["rejected"]] == ["broker"]
