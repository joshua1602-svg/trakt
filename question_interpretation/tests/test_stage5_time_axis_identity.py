"""Change 2 — a time axis distinguishable from a dimension axis.

The inventory found the facet layer raising NO facet for the time axis in any of
24 time-series probes: where `grouping_dimension` fired it was for the non-time
axis. The reason is structural rather than an oversight — `KIND_GROUPING` works
by matching a registry `field_key` against group keys, result columns and spec
filters, and a time grain is none of those. There is no `week` column. The layer
had no way to say what it had seen.

A grain is a `KIND_GRANULARITY`: the level an answer is REPORTED at. The same
reconciler branch adjudicates a spatial grain and a temporal one, because they
are the same kind of claim about an answer.

NOT YET WIRED. Nothing calls `time_axis_disclosure` until change 1 supplies the
reading from the facet layer. These tests exercise the identity directly.
"""
from __future__ import annotations

import pytest

from mi_agent import execution_receipt as R


def _reconcile_routed(facet, route="evolution"):
    return R.reconcile_routed_facets([facet], route=route,
                                     semantics={"fields": {}},
                                     available_columns=None, envelope={})[0]


# --------------------------------------------------------------------------- #
# A time grain is not a dimension, at every stage that reads one
# --------------------------------------------------------------------------- #
def test_a_time_grain_carries_no_field_key():
    facet = R.time_axis_disclosure("week", "evolution")
    assert facet.field_key is None
    assert facet.satisfied_by() == ()


def test_a_time_grain_is_not_a_grouping_kind():
    """The distinction the whole change exists for."""
    assert R.time_axis_disclosure("week", "evolution").kind == R.KIND_GRANULARITY
    assert R.KIND_GRANULARITY != R.KIND_GROUPING


def test_the_role_split_does_not_touch_a_time_grain():
    """Stage 4's split reads KIND_GROUPING and a field_key. A grain has neither,
    so it must pass through untouched rather than being read as a dimension no
    source gave a role to — which is what would happen if it were a grouping."""
    class _Spec:
        filters, dimensions, dimension = {}, [], None

    facet = R.time_axis_disclosure("week", "evolution")
    out = R._split_named_dimension_roles([facet], _Spec())
    assert [f.kind for f in out] == [R.KIND_GRANULARITY]


def test_a_dimension_axis_and_a_time_axis_survive_the_same_receipt():
    """Both in one sentence — "balance by month by region" — must stay apart."""
    time_axis = R.time_axis_disclosure("month", "evolution")
    dimension = R.RequestedFacet(kind=R.KIND_GROUPING, label="region",
                                 field_key="collateral_geography")
    kinds = sorted(f.kind for f in (time_axis, dimension))
    assert kinds == sorted([R.KIND_GRANULARITY, R.KIND_GROUPING])


# --------------------------------------------------------------------------- #
# Which routes publish a series, and what they publish
# --------------------------------------------------------------------------- #
def test_a_route_that_publishes_no_series_raises_no_grain():
    """A point-in-time KPI is never told it failed to honour a grain nobody
    could have expected it to."""
    assert R.time_axis_disclosure("week", None) is None
    assert R.time_axis_disclosure("week", "risk_limits") is None


def test_every_series_route_publishes_months():
    """All of them read the governed month-end snapshots. If one ever publishes
    a finer series this fails, which is the point: the grain is a declared
    property of the capability, not a guess from the prose."""
    for route in ("evolution", "period_movement", "temporal_compare",
                  "cohort_progression", "forecast_extrapolation"):
        assert R.route_time_grain(route) == "month"


# --------------------------------------------------------------------------- #
# Adjudication, through the branch change 5 built
# --------------------------------------------------------------------------- #
def test_a_grain_the_series_expresses_is_applied():
    out = _reconcile_routed(R.time_axis_disclosure("month", "evolution"))
    assert out.status == R.APPLIED


@pytest.mark.parametrize("unit", ["week", "quarter"])
def test_a_grain_the_series_does_not_express_is_unsupported_and_blocks(unit):
    """Both directions. A finer request cannot be expressed at all; a coarser one
    could be in principle, and this route does not aggregate — it returns
    months either way, so either way a different level was measured."""
    out = _reconcile_routed(R.time_axis_disclosure(unit, "evolution"))
    assert out.status == R.UNSUPPORTED
    assert out.reason == "this answer is reported at month level, not by %s" % unit
    verdict, message = R.assess(R.ExecutionReceipt(facets=[out]))
    assert verdict == R.VERDICT_REFUSE
    assert unit in message
