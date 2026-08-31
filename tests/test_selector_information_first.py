#!/usr/bin/env python3
"""tests/test_selector_information_first.py

Information content decides; preference only breaks ties.

THE DEFECT. The selector introduced in the composition-elevation sprint made
presentation preference the FIRST sort key, so a preferred dimension beat a
more informative one whatever the data said. On a representative book carrying
every governed field:

    dimension      categories   top share   selected
    age                 7          14.5%       no
    region              7          14.5%       no
    ltv                 7          24.8%       yes
    vintage             5          20.4%       no
    ticket              5          56.8%       yes   <- 57% in one band
    rate                3          39.9%       no
    borrower_type       2          66.4%       no
    product             2          74.3%       no

Ticket size was drawn with 57% of the book in a single band while origination
vintage, spread across five, was not — purely because "ticket" appeared
earlier in a list. And the methodology ledger then reported that more
informative dimensions had been available, which those same numbers
contradict.

These probes are the ones the brief names, A through H.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from mi_agent_api import presentation as P  # noqa: E402

#: The governed preference order the deck passes for funded stratifications.
PREFERRED = ("ltv", "ticket", "age", "region", "rate", "product",
             "borrower_type", "broker")


def bars(*pairs):
    return [{"label": str(label), "balance": float(value)}
            for label, value in pairs]


def even(prefix, n, total=100.0):
    return bars(*[(f"{prefix}{i}", total / n) for i in range(n)])


#: The representative book from the capability audit, reproduced exactly.
def audit_book():
    return [
        {"key": "age", "label": "By borrower age", "bars": even("a", 7)},
        {"key": "region", "label": "By region", "bars": even("r", 7)},
        {"key": "ltv", "label": "By LTV band",
         "bars": bars(("20-30%", 24.8), ("30-40%", 14), ("40-50%", 13),
                      ("50-60%", 13), ("60-70%", 12), ("70-80%", 12),
                      ("80-90%", 11.2))},
        {"key": "vintage", "label": "By origination vintage",
         "bars": bars(("2022", 20.4), ("2023", 20), ("2024", 20),
                      ("2025", 20), ("2026", 19.6))},
        {"key": "ticket", "label": "By ticket size",
         "bars": bars(("50-100k", 56.8), ("100-150k", 15), ("150-200k", 12),
                      ("200-300k", 9), ("300-500k", 7.2))},
        {"key": "rate", "label": "By rate band",
         "bars": bars(("5-6%", 39.9), ("6-7%", 31), ("7-8%", 29.1))},
        {"key": "borrower_type", "label": "By borrower type",
         "bars": bars(("Joint", 66.4), ("Single", 33.6))},
        {"key": "product", "label": "By product",
         "bars": bars(("Lifetime Mortgage", 74.3),
                      ("Retirement Interest Only", 25.7))},
        {"key": "broker", "label": "By broker / channel",
         "bars": bars(("Direct", 100.0))},
    ]


def select(candidates, want=4, preferred=PREFERRED):
    return P.select_dimensions(candidates, want=want, preferred=preferred)


def keys(out):
    return [e["key"] for e in out["selected"]]


def rejected(out):
    return {r["key"]: r for r in out["rejected"]}


# --------------------------------------------------------------------------- #
# A. ONE CATEGORY.
# --------------------------------------------------------------------------- #

def test_A_a_single_category_dimension_is_suppressed():
    out = select(audit_book())
    assert "broker" not in keys(out)
    assert rejected(out)["broker"]["reasonCode"] == P.REASON_ONE_CATEGORY


# --------------------------------------------------------------------------- #
# H. THE CRITICAL REGRESSION — a lower-preference dimension that is clearly
#    more informative than ticket must win.
# --------------------------------------------------------------------------- #

def test_H_vintage_beats_ticket_on_the_audit_book():
    """The exact case the audit proved wrong."""
    out = select(audit_book())
    assert "vintage" in keys(out), keys(out)
    assert "ticket" not in keys(out), keys(out)


def test_H_the_scores_explain_the_outcome():
    """Auditable, not opaque: the numbers are recomputable from the bars."""
    book = {e["key"]: e for e in audit_book()}
    vintage = P.dispersion(book["vintage"]["bars"])
    ticket = P.dispersion(book["ticket"]["bars"])
    assert vintage["effectiveCategories"] == ticket["effectiveCategories"] == 5
    # Same granularity, so evenness alone separates them — which is the point.
    assert vintage["granularity"] == ticket["granularity"]
    assert vintage["evenness"] > ticket["evenness"]
    assert vintage["score"] > ticket["score"]


def test_H_preference_cannot_rescue_a_weaker_dimension():
    """Even named FIRST, ticket does not displace a better-spread dimension."""
    out = select(audit_book(), want=4,
                 preferred=("ticket", "product", "broker", "ltv"))
    assert "ticket" not in keys(out), keys(out)


# --------------------------------------------------------------------------- #
# B-F. The named dimensions can earn a place when the data supports it.
# --------------------------------------------------------------------------- #

def test_B_borrower_type_can_be_selected():
    """A binary split is not automatically ineligible: single versus joint can
    be the more informative cut on a book whose other dimensions are thin."""
    out = select([
        {"key": "borrower_type", "label": "By borrower type",
         "bars": bars(("Joint", 52), ("Single", 48))},
        {"key": "ltv", "label": "By LTV band",
         "bars": bars(("20-30%", 92), ("30-40%", 8))},
        {"key": "ticket", "label": "By ticket size",
         "bars": bars(("50-100k", 90), ("100-150k", 10))},
    ], want=1)
    assert keys(out) == ["borrower_type"], keys(out)


def test_C_vintage_can_be_selected():
    out = select(audit_book())
    assert "vintage" in keys(out)


def test_D_rate_can_be_selected():
    out = select([e for e in audit_book()
                  if e["key"] in ("rate", "ticket", "product", "broker")],
                 want=2)
    assert "rate" in keys(out), keys(out)


def test_E_product_can_be_selected_on_a_multi_product_book():
    out = select([
        {"key": "product", "label": "By product",
         "bars": bars(("Lifetime Mortgage", 34), ("Retirement Interest Only", 33),
                      ("Drawdown", 33))},
        {"key": "broker", "label": "By broker / channel",
         "bars": bars(("Direct", 100))},
    ], want=2)
    assert keys(out) == ["product"], keys(out)


def test_F_a_single_product_book_suppresses_product():
    out = select([
        {"key": "product", "label": "By product",
         "bars": bars(("Lifetime Mortgage", 100))},
        {"key": "ltv", "label": "By LTV band", "bars": even("l", 5)},
    ], want=2)
    assert keys(out) == ["ltv"]
    assert rejected(out)["product"]["reasonCode"] == P.REASON_ONE_CATEGORY


# --------------------------------------------------------------------------- #
# Preference as a tie-breaker, and only there.
# --------------------------------------------------------------------------- #

def test_preference_decides_a_genuine_tie():
    """Two identical distributions are indistinguishable to a reader, and THAT
    is where the governed economic relevance of the cut may decide."""
    tie = [{"key": "region", "label": "By region", "bars": even("r", 6)},
           {"key": "age", "label": "By borrower age", "bars": even("a", 6)}]
    assert P.dispersion(tie[0]["bars"])["score"] == \
        P.dispersion(tie[1]["bars"])["score"]
    assert keys(select(tie, want=1, preferred=("age", "region"))) == ["age"]
    assert keys(select(tie, want=1, preferred=("region", "age"))) == ["region"]


def test_preference_does_not_decide_a_real_difference():
    near = [{"key": "region", "label": "By region", "bars": even("r", 7)},
            {"key": "age", "label": "By borrower age",
             "bars": bars(("a0", 70), ("a1", 20), ("a2", 10))}]
    assert keys(select(near, want=1, preferred=("age", "region"))) == ["region"]


def test_selection_is_deterministic_and_order_independent():
    book = audit_book()
    first = keys(select(book))
    for _ in range(5):
        assert keys(select(list(reversed(book)))) == first


# --------------------------------------------------------------------------- #
# The ledger must be derivable from the selector's own inputs.
# --------------------------------------------------------------------------- #

def test_every_rejection_carries_a_truthful_code_and_its_numbers():
    out = select(audit_book())
    for row in out["rejected"]:
        assert row["reasonCode"] in P.REASON_WORDING or \
            row["reasonCode"] == P.REASON_LOWER_RANKED
        assert row["reason"]
        assert row["score"] is not None
        assert row["effectiveCategories"] is not None


def test_a_lower_ranked_dimension_is_not_called_uninformative():
    """Catches the false ledger entry: ticket was told it lost to "more
    informative dimensions" while carrying 57% in one band, which the numbers
    contradicted. A ranking statement is now made as one, with both scores."""
    out = select(audit_book())
    ticket = rejected(out)["ticket"]
    assert ticket["reasonCode"] == P.REASON_LOWER_RANKED
    assert "scored higher" in ticket["reason"]
    assert f"{ticket['score']:.2f}" in ticket["reason"]


def test_the_ledger_matches_what_was_actually_selected():
    out = select(audit_book(), want=3)
    assert len(out["selected"]) == 3
    assert len(out["rejected"]) == len(audit_book()) - 3
    assert not (set(keys(out)) & {r["key"] for r in out["rejected"]})
