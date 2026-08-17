#!/usr/bin/env python3
"""tests/test_p1c_ranked_movement.py — P1C ranked period-over-period movement.

The ranking function is PURE and ranks only figures the governed period-change
engine already produced, so these tests reconcile against pandas computed
directly from two snapshot frames — never against a previous MI Agent run.

Tie-breaking convention (inherited from ``distribution_change``'s own top-shift
lists): rank value first, then category name ascending. Pandas row order must
never decide business semantics.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import pytest

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from mi_agent.period_change import ranking as rk
from mi_agent.period_change.models import CategoryShift

_BALANCE = "current_outstanding_balance"
_REGION = "collateral_geography"


# --------------------------------------------------------------------------- #
# Two snapshots, and the truth derived from them independently
# --------------------------------------------------------------------------- #
def _snapshots():
    """Opening and closing frames with a KNOWN movement per region."""
    opening = pd.DataFrame({
        _REGION: (["London"] * 4 + ["South East"] * 3 + ["Wales"] * 2 + ["Scotland"] * 2),
        _BALANCE: [100.0, 100.0, 100.0, 100.0,      # London  400
                   100.0, 100.0, 100.0,             # SE      300
                   100.0, 100.0,                    # Wales   200
                   100.0, 100.0],                   # Scot    200
    })
    closing = pd.DataFrame({
        _REGION: (["London"] * 5 + ["South East"] * 3 + ["Wales"] * 1 + ["Scotland"] * 2),
        _BALANCE: [100.0, 100.0, 100.0, 100.0, 100.0,   # London  500  (+100, +25%)
                   120.0, 120.0, 120.0,                 # SE      360  (+60,  +20%)
                   100.0,                                # Wales   100  (-100, -50%)
                   100.0, 100.0],                        # Scot    200  ( 0,     0%)
    })
    return opening, closing


def _truth(opening, closing):
    """Movement per region, computed independently with pandas."""
    o = opening.groupby(_REGION)[_BALANCE].sum()
    c = closing.groupby(_REGION)[_BALANCE].sum()
    absolute = (c - o).sort_values(ascending=False)
    percent = ((c - o) / o * 100).sort_values(ascending=False)
    return o, c, absolute, percent


def _shifts_from(opening, closing):
    """CategoryShift rows in the shape the governed engine emits."""
    o = opening.groupby(_REGION)[_BALANCE].sum()
    c = closing.groupby(_REGION)[_BALANCE].sum()
    o_n = opening.groupby(_REGION).size()
    c_n = closing.groupby(_REGION).size()
    o_total, c_total = float(o.sum()), float(c.sum())
    out = []
    for category in sorted(set(o.index) | set(c.index)):
        sb, eb = float(o.get(category, 0.0)), float(c.get(category, 0.0))
        out.append(CategoryShift(
            category=category,
            start_count=int(o_n.get(category, 0)), end_count=int(c_n.get(category, 0)),
            start_count_share=float(o_n.get(category, 0)) / len(opening),
            end_count_share=float(c_n.get(category, 0)) / len(closing),
            count_share_movement=(float(c_n.get(category, 0)) / len(closing)
                                  - float(o_n.get(category, 0)) / len(opening)),
            start_balance=sb, end_balance=eb,
            start_balance_share=sb / o_total, end_balance_share=eb / c_total,
            balance_share_movement=(eb / c_total) - (sb / o_total),
            presence="both"))
    return out


@pytest.fixture(scope="module")
def fixture():
    opening, closing = _snapshots()
    return opening, closing, _shifts_from(opening, closing)


# --------------------------------------------------------------------------- #
# Truth reconciliation
# --------------------------------------------------------------------------- #
def test_largest_absolute_increase_matches_pandas(fixture):
    opening, closing, shifts = fixture
    _, _, absolute, _ = _truth(opening, closing)

    result = rk.rank_movement(shifts, basis=rk.BASIS_BALANCE_ABSOLUTE,
                              direction=rk.DIRECTION_INCREASE)
    assert result.ok
    assert result.rows[0].category == absolute.index[0] == "London"
    assert result.rows[0].absolute_movement == pytest.approx(float(absolute.iloc[0]))
    assert result.rows[0].start_value == pytest.approx(400.0)
    assert result.rows[0].end_value == pytest.approx(500.0)
    # Only genuinely increasing categories appear, in pandas' own order.
    assert [r.category for r in result.rows] == [
        c for c in absolute.index if absolute[c] > 0]


def test_largest_percentage_increase_can_differ_from_absolute(fixture):
    """The basis must change the ANSWER, or it is not a real basis. London adds
    the most balance; London also grows fastest here — so assert the numbers,
    not just the order."""
    opening, closing, shifts = fixture
    _, _, _, percent = _truth(opening, closing)

    result = rk.rank_movement(shifts, basis=rk.BASIS_BALANCE_PERCENT,
                              direction=rk.DIRECTION_INCREASE)
    assert result.ok
    for row in result.rows:
        assert row.percent_movement == pytest.approx(float(percent[row.category]))
    assert result.rows[0].percent_movement == pytest.approx(25.0)


def test_largest_decrease_matches_pandas(fixture):
    opening, closing, shifts = fixture
    _, _, absolute, _ = _truth(opening, closing)

    result = rk.rank_movement(shifts, basis=rk.BASIS_BALANCE_ABSOLUTE,
                              direction=rk.DIRECTION_DECREASE)
    assert result.ok
    assert result.rows[0].category == "Wales"
    assert result.rows[0].absolute_movement == pytest.approx(
        float(absolute["Wales"])) == pytest.approx(-100.0)
    assert all(r.absolute_movement < 0 for r in result.rows)


def test_top_n_truncates_after_ranking(fixture):
    _, _, shifts = fixture
    full = rk.rank_movement(shifts, basis=rk.BASIS_BALANCE_ABSOLUTE,
                            direction=rk.DIRECTION_INCREASE)
    top1 = rk.rank_movement(shifts, basis=rk.BASIS_BALANCE_ABSOLUTE,
                            direction=rk.DIRECTION_INCREASE, top_n=1)
    assert top1.ok and len(top1.rows) == 1 and top1.top_n == 1
    assert top1.rows[0].category == full.rows[0].category


def test_count_basis_ranks_on_loan_count_not_balance(fixture):
    """Measure substitution guard at the ranking layer: asking for loan-count
    movement must not silently rank on balance."""
    opening, closing, shifts = fixture
    result = rk.rank_movement(shifts, basis=rk.BASIS_COUNT_ABSOLUTE,
                              direction=rk.DIRECTION_INCREASE)
    assert result.ok
    expected = (closing.groupby(_REGION).size()
                - opening.groupby(_REGION).size()).sort_values(ascending=False)
    assert result.rows[0].category == expected.index[0]
    assert result.rows[0].absolute_movement == pytest.approx(float(expected.iloc[0]))


def test_share_basis_ranks_on_composition_movement(fixture):
    _, _, shifts = fixture
    result = rk.rank_movement(shifts, basis=rk.BASIS_BALANCE_SHARE,
                              direction=rk.DIRECTION_INCREASE)
    assert result.ok
    by_category = {s.category: s for s in shifts}
    for row in result.rows:
        assert row.absolute_movement == pytest.approx(
            by_category[row.category].balance_share_movement)


# --------------------------------------------------------------------------- #
# Deterministic behaviour at the edges
# --------------------------------------------------------------------------- #
def test_ties_break_on_category_name_ascending():
    """Two identical movements must always order the same way — never by input
    order, which pandas would otherwise decide."""
    shifts = [
        CategoryShift(category="Zed", start_count=1, end_count=2,
                      start_count_share=0.5, end_count_share=0.5,
                      count_share_movement=0.0,
                      start_balance=100.0, end_balance=150.0),
        CategoryShift(category="Alpha", start_count=1, end_count=2,
                      start_count_share=0.5, end_count_share=0.5,
                      count_share_movement=0.0,
                      start_balance=100.0, end_balance=150.0),
    ]
    forward = rk.rank_movement(shifts, basis=rk.BASIS_BALANCE_ABSOLUTE)
    reversed_ = rk.rank_movement(list(reversed(shifts)),
                                 basis=rk.BASIS_BALANCE_ABSOLUTE)
    assert [r.category for r in forward.rows] == ["Alpha", "Zed"]
    assert [r.category for r in forward.rows] == [r.category for r in reversed_.rows]


def test_a_category_with_no_opening_balance_is_excluded_from_percentage_ranking():
    """Infinite growth is not a fact. The category is excluded WITH a reason,
    never ranked as zero — which would bury a new category mid-table."""
    shifts = [
        CategoryShift(category="New", start_count=0, end_count=5,
                      start_count_share=0.0, end_count_share=0.5,
                      count_share_movement=0.5,
                      start_balance=0.0, end_balance=500.0, presence="end_only"),
        CategoryShift(category="Old", start_count=5, end_count=6,
                      start_count_share=1.0, end_count_share=0.5,
                      count_share_movement=-0.5,
                      start_balance=100.0, end_balance=200.0),
    ]
    result = rk.rank_movement(shifts, basis=rk.BASIS_BALANCE_PERCENT)
    assert result.ok
    assert [r.category for r in result.rows] == ["Old"]
    assert ("New", rk._NO_OPENING_BASE) in result.excluded
    # On an ABSOLUTE basis the same category ranks normally.
    absolute = rk.rank_movement(shifts, basis=rk.BASIS_BALANCE_ABSOLUTE)
    assert absolute.rows[0].category == "New"


def test_all_categories_declining_refuses_an_increase_ranking():
    shifts = [CategoryShift(category=c, start_count=2, end_count=1,
                            start_count_share=0.5, end_count_share=0.5,
                            count_share_movement=0.0,
                            start_balance=200.0, end_balance=100.0)
              for c in ("A", "B")]
    result = rk.rank_movement(shifts, basis=rk.BASIS_BALANCE_ABSOLUTE,
                              direction=rk.DIRECTION_INCREASE)
    assert result.ok is False
    assert "no category increased" in result.reason


def test_all_categories_increasing_refuses_a_decrease_ranking():
    shifts = [CategoryShift(category=c, start_count=1, end_count=2,
                            start_count_share=0.5, end_count_share=0.5,
                            count_share_movement=0.0,
                            start_balance=100.0, end_balance=200.0)
              for c in ("A", "B")]
    result = rk.rank_movement(shifts, basis=rk.BASIS_BALANCE_ABSOLUTE,
                              direction=rk.DIRECTION_DECREASE)
    assert result.ok is False
    assert "no category decreased" in result.reason


def test_a_basis_absent_from_both_snapshots_is_excluded_not_zeroed():
    shifts = [CategoryShift(category="A", start_count=1, end_count=1,
                            start_count_share=0.5, end_count_share=0.5,
                            count_share_movement=0.0,
                            start_balance=None, end_balance=None)]
    result = rk.rank_movement(shifts, basis=rk.BASIS_BALANCE_ABSOLUTE)
    assert result.ok is False
    assert result.excluded and "not reported in both snapshots" in result.excluded[0][1]


@pytest.mark.parametrize("bad", [0, -1, -5])
def test_invalid_top_n_refuses(bad, fixture):
    _, _, shifts = fixture
    result = rk.rank_movement(shifts, basis=rk.BASIS_BALANCE_ABSOLUTE, top_n=bad)
    assert result.ok is False and "invalid top-N" in result.reason


def test_an_unsupported_basis_refuses():
    result = rk.rank_movement([], basis="vibes")
    assert result.ok is False and "unsupported ranking basis" in result.reason


def test_an_unsupported_direction_refuses(fixture):
    _, _, shifts = fixture
    result = rk.rank_movement(shifts, basis=rk.BASIS_BALANCE_ABSOLUTE,
                              direction="sideways")
    assert result.ok is False and "unsupported direction" in result.reason


def test_no_categories_at_all_refuses():
    result = rk.rank_movement([], basis=rk.BASIS_BALANCE_ABSOLUTE)
    assert result.ok is False


def test_the_basis_label_is_stated_for_the_receipt(fixture):
    _, _, shifts = fixture
    result = rk.rank_movement(shifts, basis=rk.BASIS_BALANCE_ABSOLUTE)
    assert result.basis_label == "absolute balance movement"
    assert result.to_dict()["basisLabel"] == "absolute balance movement"


def test_ranking_never_mutates_the_governed_input(fixture):
    """The engine's output is evidence. Ranking must not alter it."""
    _, _, shifts = fixture
    before = [s.to_dict() for s in shifts]
    rk.rank_movement(shifts, basis=rk.BASIS_BALANCE_ABSOLUTE)
    rk.rank_movement(shifts, basis=rk.BASIS_BALANCE_PERCENT,
                     direction=rk.DIRECTION_DECREASE, top_n=2)
    assert [s.to_dict() for s in shifts] == before


# =========================================================================== #
# Ranking-intent detection — narrow by design
# =========================================================================== #
from mi_agent.period_change import rank_request as rr  # noqa: E402

DETECTION = [
    ("Which region grew the most last month?", rk.BASIS_BALANCE_ABSOLUTE,
     rk.DIRECTION_INCREASE, None),
    ("Which region added the most balance month-on-month?",
     rk.BASIS_BALANCE_ABSOLUTE, rk.DIRECTION_INCREASE, None),
    ("Which region declined the most last month?", rk.BASIS_BALANCE_ABSOLUTE,
     rk.DIRECTION_DECREASE, None),
    ("Where did balance fall most?", rk.BASIS_BALANCE_ABSOLUTE,
     rk.DIRECTION_DECREASE, None),
    ("Show me the three largest declines.", rk.BASIS_BALANCE_ABSOLUTE,
     rk.DIRECTION_DECREASE, 3),
    ("What were the top 3 regional increases?", rk.BASIS_BALANCE_ABSOLUTE,
     rk.DIRECTION_INCREASE, 3),
    ("Which five segments grew the most?", rk.BASIS_BALANCE_ABSOLUTE,
     rk.DIRECTION_INCREASE, 5),
    ("Which region grew fastest in percentage terms?", rk.BASIS_BALANCE_PERCENT,
     rk.DIRECTION_INCREASE, None),
    ("Which region increased its share of the portfolio the most?",
     rk.BASIS_BALANCE_SHARE, rk.DIRECTION_INCREASE, None),
    ("Which region added the most loans month-on-month?",
     rk.BASIS_COUNT_ABSOLUTE, rk.DIRECTION_INCREASE, None),
    ("Rank regions by growth since last month.", rk.BASIS_BALANCE_ABSOLUTE,
     rk.DIRECTION_INCREASE, None),
]


@pytest.mark.parametrize("question,basis,direction,top_n", DETECTION,
                         ids=[q[:46] for q, _, _, _ in DETECTION])
def test_rank_request_detection(question, basis, direction, top_n):
    request = rr.detect_rank_request(question, "region")
    assert request is not None, f"{question!r} was not recognised as ranked"
    assert request.basis == basis
    assert request.direction == direction
    assert request.top_n == top_n


@pytest.mark.parametrize("question", [
    "What changed since last month?",
    "How much has the funded book grown?",
    "What moved month on month?",
    "What drove the change in funded balance?",
])
def test_a_narrative_period_change_question_is_not_a_ranking(question):
    """No superlative, no rank instruction — these must keep the existing
    period-change narrative rather than being turned into a ranking."""
    assert rr.detect_rank_request(question, "region") is None


def test_ranking_without_a_dimension_is_not_a_p1c_question():
    assert rr.detect_rank_request("Which grew the most?", None) is None


def test_the_bare_growth_convention_is_absolute_balance():
    """Stated convention: "grew the most" with no measure named ranks on
    absolute balance movement, and the receipt says so."""
    request = rr.detect_rank_request("Which region grew the most?", "region")
    assert request.basis == rk.BASIS_BALANCE_ABSOLUTE
    assert request.basis_label == "absolute balance movement"


# =========================================================================== #
# The unknown bucket, and what is left out
# =========================================================================== #
def test_the_unknown_bucket_cannot_win_a_ranking_of_categories(fixture):
    """"Unknown" is a real row in a governed distribution and can genuinely move
    the most — but it is missing data, not a region. It is excluded WITH ITS
    REASON, so nothing is hidden."""
    _, _, base = fixture
    shifts = list(base) + [CategoryShift(
        category="Unknown", start_count=1, end_count=1,
        start_count_share=0.0, end_count_share=0.0, count_share_movement=0.0,
        start_balance=10.0, end_balance=9_999.0,
        start_balance_share=0.0, end_balance_share=0.0,
        balance_share_movement=0.0, presence="both")]

    ranked = rk.rank_movement(shifts, basis=rk.BASIS_BALANCE_ABSOLUTE)
    assert ranked.ok
    assert "Unknown" not in [r.category for r in ranked.rows]
    assert ("Unknown", rk._NOT_A_CATEGORY) in ranked.excluded

    # …and a caller that genuinely wants it ranked can ask for that explicitly.
    everything = rk.rank_movement(shifts, basis=rk.BASIS_BALANCE_ABSOLUTE,
                                  exclude_categories=())
    assert everything.rows[0].category == "Unknown"


def test_categories_moving_the_other_way_are_counted(fixture):
    """A ranking of increases drops the decliners. How many were dropped is
    reported, so "these three grew" can never read as "these are all of them"."""
    _, _, shifts = fixture
    ranked = rk.rank_movement(shifts, basis=rk.BASIS_BALANCE_ABSOLUTE,
                              direction=rk.DIRECTION_INCREASE)
    assert ranked.direction_excluded == len(shifts) - len(ranked.rows)
    assert ranked.direction_excluded > 0
    assert ranked.to_dict()["directionExcluded"] == ranked.direction_excluded


# =========================================================================== #
# The P0 boundary — a declared ranking is verified, never taken on trust
# =========================================================================== #
from mi_agent import execution_receipt as rc  # noqa: E402

_SEMANTICS = {"fields": {"collateral_geography": {"canonical_field": "collateral_geography"},
                         "ltv_bucket": {"canonical_field": "ltv_bucket"}}}


def _ranking_facet(field_key="collateral_geography", label="ranking by region"):
    return rc.RequestedFacet(kind=rc.KIND_RANKING, label=label, field_key=field_key)


def _envelope_declaring(field, display="Collateral Geography", **extra):
    declared = {"applied": True, "canonicalField": field, "displayName": display,
                "basis": "balance_absolute",
                "basisLabel": "absolute balance movement",
                "direction": "increase", "topN": None, "categoriesAnalysed": 12,
                "openingPeriod": "2026-05-31", "closingPeriod": "2026-06-30"}
    declared.update(extra)
    return {"metadata": {"rankedMovement": declared}}


def test_a_ranking_the_route_proves_it_performed_is_applied():
    facets = rc.reconcile_routed_facets(
        [_ranking_facet()], route="period_change_analysis", semantics=_SEMANTICS,
        envelope=_envelope_declaring("collateral_geography"))
    assert facets[0].status == rc.APPLIED


def test_a_ranking_of_a_DIFFERENT_dimension_is_lost_not_accepted():
    """The route ranked LTV band; the question asked about region. Accepting the
    route's claim would be the silent substitution P0 exists to prevent."""
    facets = rc.reconcile_routed_facets(
        [_ranking_facet()], route="period_change_analysis", semantics=_SEMANTICS,
        envelope=_envelope_declaring("ltv_bucket", display="LTV Bucket"))
    assert facets[0].status == rc.LOST
    assert "LTV Bucket" in facets[0].reason


@pytest.mark.parametrize("declared", [
    None, {}, {"applied": False, "reason": "dimension_not_resolved"},
    {"applied": True},                       # a claim with no field is not evidence
    {"canonicalField": "collateral_geography"},   # a field with no claim likewise
])
def test_a_ranking_claim_without_evidence_is_not_accepted(declared):
    envelope = {"metadata": {"rankedMovement": declared}}
    assert rc.ranking_evidence(envelope) == {}
    facets = rc.reconcile_routed_facets(
        [_ranking_facet()], route="period_change_analysis", semantics=_SEMANTICS,
        envelope=envelope)
    assert facets[0].status == rc.LOST


def test_an_unrouted_answer_is_unaffected_by_the_new_evidence_path():
    """No declaration at all: the pre-P1C behaviour is unchanged."""
    facets = rc.reconcile_routed_facets(
        [_ranking_facet()], route="period_change_analysis", semantics=_SEMANTICS)
    assert facets[0].status == rc.LOST


# =========================================================================== #
# The receipt — the basis, direction, Top N and both dates
# =========================================================================== #
def test_the_receipt_states_how_the_ranking_was_ordered():
    receipt = rc.build_routed_receipt(
        route="period_change_analysis",
        envelope=_envelope_declaring("collateral_geography"),
        facets=[])
    line = receipt.render()
    assert "ranked by Collateral Geography" in line
    assert "absolute balance movement, largest increases first" in line
    assert "2026-05-31 → 2026-06-30" in line
    assert receipt.to_dict()["ranking"].startswith("absolute balance movement")


def test_the_receipt_states_a_top_n_and_the_universe_it_came_from():
    receipt = rc.build_routed_receipt(
        route="period_change_analysis",
        envelope=_envelope_declaring("collateral_geography", topN=3),
        facets=[])
    assert "top 3 of 12" in receipt.render()


def test_a_decrease_ranking_says_so():
    receipt = rc.build_routed_receipt(
        route="period_change_analysis",
        envelope=_envelope_declaring("collateral_geography", direction="decrease"),
        facets=[])
    assert "largest decreases first" in receipt.render()


# =========================================================================== #
# Intent resolution in the route — narrative vs dimension
# =========================================================================== #
from mi_agent_api import period_change_route as pcr  # noqa: E402


@pytest.mark.parametrize("question,subject", [
    ("Which region grew the most last month?", "region"),
    ("Which five segments grew the most?", "segments"),
    ("What were the top 3 regional increases since last month?", "regional"),
    ("What were the largest movements since last month?", "movements"),
    ("Rank the regions by growth", "regions"),
])
def test_the_rank_subject_is_the_noun_not_the_auxiliary(question, subject):
    assert pcr._rank_subject(question) == subject


@pytest.mark.parametrize("question", [
    "What were the largest movements since last month?",
    "What were the biggest changes month on month?",
    "What were the largest increases since last month?",
])
def test_a_superlative_about_the_narrative_is_not_a_dimension_ranking(question):
    """"Largest movements" is the governed narrative. Refusing it for lacking a
    dimension would break an answer that already works."""
    outcome = pcr.resolve_rank_intent(question)
    assert outcome.requested is False
    assert outcome.refusal is None
