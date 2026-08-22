"""A routed grouping is APPLIED only on evidence — never because it cannot be disproved.

The worst defect found in this programme. `reconcile_routed_facets` ended its
grouping branch with:

    else:
        # The route grouped by something it declared; without a result
        # frame we cannot disprove it, and refusing on an unprovable
        # facet would disable working governed analytics.
        facet.status, facet.reason = APPLIED, ""

That is the exact inverse of the bar every other guard here holds.
`reconcile_population`: "a facet is APPLIED only when the route reports having
applied that field. A route that reports nothing leaves every population facet
LOST."

And the premise was false. `evolution` publishes a result frame; its rows carry
`period` and `value` and nothing else. "balance by month by region" returned the
whole-book series — byte-identical values, no region column — and the receipt
vouched for a regional breakdown that was never computed. A receipt that misses
an error is a gap; one that affirms it is worse than no receipt.
"""
from __future__ import annotations

import pytest

from mi_agent import execution_receipt as R


def _grouping(field="collateral_geography", alts=()):
    return R.RequestedFacet(kind=R.KIND_GROUPING, label="region",
                            field_key=field, alt_keys=alts)


def _run(facet, envelope, route="evolution"):
    return R.reconcile_routed_facets([facet], route=route,
                                     semantics={"fields": {}},
                                     available_columns=None,
                                     envelope=envelope)[0]


def _frame(*rows):
    return {"artifacts": [{"rows": list(rows)}]}


# --------------------------------------------------------------------------- #
# The defect
# --------------------------------------------------------------------------- #
def test_a_time_only_frame_does_not_prove_a_breakdown():
    """The reported defect, exactly: `evolution` rows are period and value."""
    out = _run(_grouping(), _frame({"period": "2026-04", "value": 1.0},
                                   {"period": "2026-05", "value": 2.0}))
    assert out.status == R.LOST
    assert "neither narrowed to nor broken down by region" in out.reason


def test_an_empty_envelope_does_not_prove_a_breakdown():
    """A route that reports nothing leaves the facet LOST — the population bar."""
    assert _run(_grouping(), {}).status == R.LOST


# --------------------------------------------------------------------------- #
# Can-fail: it must not refuse a route that genuinely grouped
# --------------------------------------------------------------------------- #
def test_a_frame_naming_the_field_proves_it():
    out = _run(_grouping(), _frame({"collateral_geography": "London", "value": 1.0}))
    assert out.status == R.APPLIED


def test_a_frame_naming_an_alt_key_proves_it():
    out = _run(_grouping(alts=("geographic_region_obligor",)),
               _frame({"geographic_region_obligor": "TLI43", "value": 1.0}))
    assert out.status == R.APPLIED


def test_a_route_that_labels_its_axis_for_display_is_not_refused():
    """geo_exposure publishes `area`/`code`, never `collateral_geography`.

    This used to pass on the residual rung — the frame proved THAT a breakdown
    happened and not WHICH one, so the facet stayed applied. Measured in D7, that
    rung was carrying every live certification on the routed path, eight of them
    false. It is gone.

    This still passes, and now for a reason: `geo_exposure` DECLARES its axis in
    `ROUTE_DECLARED_AXES`. Same outcome, evidence instead of an inability to
    disprove — and `test_d7_grouping_evidence.py` checks that declaration against
    a real envelope rather than against the comment beside it.
    """
    out = _run(_grouping(), _frame({"area": "Westminster", "code": "TLI43",
                                    "balance": 1.0, "count": 2, "share": 0.1}),
               route="geo_exposure")
    assert out.status == R.APPLIED


# --------------------------------------------------------------------------- #
# The axis reader
# --------------------------------------------------------------------------- #
def test_measures_and_the_time_axis_are_not_axes():
    assert R.answer_axis_keys(_frame({"period": "2026-04", "value": 1.0})) == set()
    assert R.answer_axis_keys(_frame(
        {"month": "2026-04", "base": 1.0, "upside": 2.0, "downside": 0.5})) == set()
    assert R.answer_axis_keys(_frame(
        {"current_outstanding_balance_sum": 1.0, "loan_count": 2,
         "concentration_pct": 0.1})) == set()


def test_a_categorical_key_is_an_axis():
    assert "area" in R.answer_axis_keys(_frame({"area": "X", "balance": 1.0}))
    assert "category" in R.answer_axis_keys(_frame({"category": "X", "exposure": 1.0}))


def test_a_ranked_declaration_still_wins():
    """The pre-existing proven path is untouched."""
    envelope = {"metadata": {"rankedMovement": {"applied": True,
                                                "canonicalField": "collateral_geography"}}}
    assert _run(_grouping(), envelope).status == R.APPLIED
