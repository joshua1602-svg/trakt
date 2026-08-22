"""D7 (B12) — a grouping is certified only where execution DECLARED it.

The routed evidence ladder's last rung stamped APPLIED on any facet it could not
disprove. Measured across 593 corpus questions it was not a residue: NINETEEN of
nineteen live certifications stood on it, because routes label their columns for
a reader and never by canonical field, so the rung above it fired zero times.
Eight of the nineteen were false.

These tests pin the replacement — declarations, and nothing else — and they pin
each declaration against what the route actually publishes rather than against a
comment beside it.
"""
from __future__ import annotations

import os
from pathlib import Path

import pytest

from mi_agent import execution_receipt as R


_REPO_ROOT = Path(__file__).resolve().parents[2]
_TAPE = (_REPO_ROOT / "demo_platform" / "workspace" / "store" / "processed"
         / "platform" / "alderbridge" / "2026-06-30"
         / "platform_canonical_typed.csv")


@pytest.fixture(scope="module")
def ask():
    if not _TAPE.exists():
        pytest.skip("calibration book not built")
    from demo_platform import config as cfg

    os.environ.update(cfg.mi_env(period_role="current"))
    os.environ["MI_AGENT_LLM_PARSER"] = "off"
    os.environ["MI_AGENT_LLM_ENABLED"] = "0"
    import logging

    logging.disable(logging.WARNING)
    from mi_agent_api.mi_service import MiQueryRequest, execute_governed_mi_query
    from trakt_core.context import ExecutionContext

    ctx = ExecutionContext.for_internal(cfg.CLIENT_ID)
    return lambda q: (execute_governed_mi_query(
        MiQueryRequest(question=q), ctx).result or {})


def _facet(field="collateral_geography", alts=(), kind=None):
    return R.RequestedFacet(kind=kind or R.KIND_GROUPING, label="region",
                            field_key=field, alt_keys=tuple(alts))


# --------------------------------------------------------------------------- #
# The bar itself
# --------------------------------------------------------------------------- #
def test_a_declaration_naming_the_field_proves_it():
    assert R.grouping_proven(_facet(), {"collateral_geography"}, {})


def test_no_declaration_proves_nothing():
    """The bar `reconcile_population` holds, now held here. A route that reports
    nothing gets no certification — which is the whole of D7."""
    assert not R.grouping_proven(_facet(), set(), {})
    assert not R.grouping_proven(_facet(), {"amortisation_type"}, {})


def test_every_key_the_facet_resolves_to_counts():
    """A generic term resolves to different concrete fields depending on the
    book, and execution may legitimately name any of them."""
    assert R.grouping_proven(
        _facet(alts=("geographic_region_obligor",)),
        {"geographic_region_obligor"}, {})


def test_a_canonical_field_counts_as_the_key_it_belongs_to(): 
    assert R.grouping_proven(
        _facet(field="region_alias"), {"collateral_geography"},
        {"region_alias": {"canonical_field": "collateral_geography"}})


# --------------------------------------------------------------------------- #
# Where declarations come from
# --------------------------------------------------------------------------- #
def test_the_concentration_workflow_declaration_is_read_not_guessed():
    """It was on the envelope before D7 existed and nothing read it. The reader
    guessed from a display column called `category` instead, and certified
    breakdowns by dimensions the route had not used."""
    envelope = {"workflow": {"dimension_results": [
        {"field": "amortisation_type", "display_name": "Amortisation Type"},
        {"field": "collateral_geography", "display_name": "Collateral Geography"},
    ]}}
    assert R.declared_group_fields(envelope) == {
        "amortisation_type", "collateral_geography"}


def test_two_populations_of_one_field_are_a_breakdown():
    """"How does the front book compare with our older lending" declares two
    governed populations of `seasoning_segment`. That comparison IS the
    breakdown, with two groups."""
    envelope = {"metadata": {"analyticalComposition": True}, "analytical": {}}
    plan = {"narrowedTo": [
        {"field": "seasoning_segment", "value": "Front Book", "rows": 1177},
        {"field": "seasoning_segment", "value": "Back Book", "rows": 9858}]}
    assert R._two_or_more_populations(plan) == {"seasoning_segment"}


def test_one_population_is_a_filter_and_not_a_breakdown():
    """The threshold is the point. Narrowing to a SINGLE population is what
    `row_population` records; counting it here would certify a breakdown of an
    answer that reports one group."""
    plan = {"narrowedTo": [
        {"field": "seasoning_segment", "value": "Front Book", "rows": 1177}]}
    assert R._two_or_more_populations(plan) == set()


def test_a_route_absent_from_the_registry_declares_nothing():
    assert R.declared_group_fields({}, "a_route_that_does_not_exist") == set()


# --------------------------------------------------------------------------- #
# The registry, checked against real envelopes rather than against its comment
# --------------------------------------------------------------------------- #
_REGISTRY_PROBES = {
    "geo_exposure": ("Show NNEG exposure by region.",
                     ("area", "code")),
}


@pytest.mark.parametrize("route", sorted(R.ROUTE_DECLARED_AXES))
def test_declared_axes_match_what_the_route_publishes(ask, route):
    """A comment stating an invariant is not evidence it holds.

    Every entry in `ROUTE_DECLARED_AXES` is a claim that the route ALWAYS cuts
    its answer by those fields. This drives the route and requires the claim to
    be true of a real answer: the route must publish a table, and that table must
    be cut by an axis — otherwise the registry is certifying a breakdown that is
    not there, which is the defect it was added to close.
    """
    question, expected_axis_keys = _REGISTRY_PROBES[route]
    res = ask(question)
    assert (res.get("metadata") or {}).get("route") == route
    axes = R.answer_axis_keys(res)
    assert axes, f"{route} declares an axis and publishes no table cut by one"
    assert set(expected_axis_keys) <= axes, (
        f"{route} publishes {sorted(axes)}, not {sorted(expected_axis_keys)} — "
        "the declaration no longer describes the answer")


def test_risk_limits_declares_only_the_fields_its_executed_tests_cover(ask):
    """Not a fixed registry entry: which limits exist comes from the governing
    document, so the route derives the declaration per answer — and only from
    tests that actually COMPUTED. A limit reported `unavailable` measured
    nothing and must certify nothing."""
    from mi_agent_api import risk_limits as risk_mod

    res = ask("Are any regional limits breached?")
    assert (res.get("metadata") or {}).get("route") == "risk_limits"
    declared = R.declared_group_fields(res, "risk_limits")
    assert "collateral_geography" in declared, (
        "the Schedule 8 tests are written per region and the route no longer "
        "says so")
    assert "account_status" not in declared, (
        "no limit test covers account status; declaring it would restore the "
        "false certification D7 closed")
    # And the derivation itself: an unavailable test contributes nothing.
    assert risk_mod.tested_fields(
        [{"category": "age_limit", "status": "unavailable"}]) == []
    assert risk_mod.tested_fields(
        [{"category": "age_limit", "status": "green"}]) == ["youngest_borrower_age"]


# --------------------------------------------------------------------------- #
# The eight false certifications, by name
# --------------------------------------------------------------------------- #
def test_a_breakdown_by_the_wrong_dimension_is_not_certified(ask):
    """`concentration_analysis` returns seven concentration tables and none of
    them is by borrower age bucket. The receipt used to say the age-bucket
    breakdown was applied."""
    res = ask("Show NNEG exposure by borrower age bucket.")
    assert (res.get("metadata") or {}).get("route") == "concentration_analysis"
    facets = [(f.get("kind"), f.get("field"), f.get("status"))
              for f in ((res.get("executionSummary") or {}).get("facets") or [])]
    assert ("grouping_dimension", "age_bucket", "applied") not in facets
    assert ("grouping_dimension", "age_bucket", "lost") in facets
    assert res.get("ok") is False


def test_a_breakdown_the_route_really_made_is_still_certified(ask):
    """The can-fail. Eleven of the nineteen claims were TRUE, and a change that
    simply stopped certifying would satisfy every test above and destroy them."""
    res = ask("Show NNEG exposure by region.")
    facets = [(f.get("kind"), f.get("field"), f.get("status"))
              for f in ((res.get("executionSummary") or {}).get("facets") or [])]
    assert ("grouping_dimension", "collateral_geography", "applied") in facets
    assert res.get("ok") is True
