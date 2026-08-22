"""D8 — one raiser and one stamper for "was this narrowing actually applied".

The census classed D8 **agree-by-maintenance**: three evidence sources, each read
in a different place, each scoped to a path the others do not run on, so they
could not contradict one another — by construction of the paths rather than by
design. The maintenance had already lapsed in two places, and neither was the
contradiction the census anticipated:

  B17   RAISED, NEVER APPLIED. `material_predicates` was computed from
        `parsed.spec.filters` before `try_route` merged the caller's filters into
        that same spec, so every drill-through on a routed question refused.

  NEW   APPLIED, NEVER RAISED. "What is the balance of the front book?" with a
        drill to South East answered 278 loans — down from 1,177, both narrowings
        run — and the receipt recorded one. B16a's mirror.

And a third, inside this programme's own work: B16a needed the same three sources
for `KIND_LOST_NARROWING` and got them by COPYING the branches.
"""
from __future__ import annotations

import os
from pathlib import Path

import pytest

from mi_agent import execution_receipt as R
from mi_agent.mi_query_validator import load_mi_semantics


_REPO_ROOT = Path(__file__).resolve().parents[2]
_TAPE = (_REPO_ROOT / "demo_platform" / "workspace" / "store" / "processed"
         / "platform" / "alderbridge" / "2026-06-30"
         / "platform_canonical_typed.csv")


@pytest.fixture(scope="module")
def semantics():
    return load_mi_semantics(
        _REPO_ROOT / "mi_agent" / "mi_semantics_field_registry.yaml")


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
    return lambda q, f=None: (execute_governed_mi_query(
        MiQueryRequest(question=q, filters=f), ctx).result or {})


def _facets(envelope):
    return [(f.get("kind"), f.get("field"), f.get("status"))
            for f in ((envelope.get("executionSummary") or {}).get("facets") or [])]


def _pop(field="collateral_geography"):
    return R.RequestedFacet(kind=R.KIND_POPULATION, field_key=field,
                            label="the population %s = South East" % field)


# --------------------------------------------------------------------------- #
# The one stamper — three sources, one function
# --------------------------------------------------------------------------- #
def test_the_executors_applied_fields_prove_it():
    assert R.population_applied(_pop(), applied_fields=["collateral_geography"])


def test_the_routes_ledger_proves_it():
    assert R.population_applied(
        _pop(), ledger={"applied": ["collateral_geography = South East"]})


def test_the_analytical_plans_declaration_proves_it():
    """The plan's declaration, through the accessor that fits the facet's LABEL.

    The two are NOT interchangeable and the caller chooses. A POPULATION facet
    carries "the population X = South East" and goes through `envelope`;
    a GEOGRAPHIC SCOPE facet carries the bare value and goes through
    `analytical`, matching on the value alone because the field a facet resolves
    to and the field a plan narrowed on can be two names for one geography.

    Passing a population facet through the second silently fails to match, which
    is how this test was first written.
    """
    envelope = {"metadata": {"analyticalPlan": {}},
                "analyticalFindings": [
                    {"predicates": {"collateral_geography": "South East"},
                     "rows": 2420}]}
    assert R.population_applied(_pop(), envelope=envelope) is \
        R._analytical_population_satisfies(envelope, _pop())

    scope = R.RequestedFacet(kind=R.KIND_GEOGRAPHIC_SCOPE, label="South East",
                             field_key="collateral_geography")
    plan = {"narrowedTo": [{"field": "geographic_region_obligor",
                            "value": "South East", "rows": 2420}]}
    assert R.population_applied(scope, analytical=plan)
    # A row count is required: a predicate that selected nothing scoped nothing.
    empty = {"narrowedTo": [{"field": "geographic_region_obligor",
                             "value": "South East", "rows": 0}]}
    assert not R.population_applied(scope, analytical=empty)


def test_nothing_proves_nothing():
    """The bar `reconcile_population` has always held, unchanged: spec presence,
    route identity and answer wording are not evidence."""
    assert not R.population_applied(_pop())
    assert not R.population_applied(_pop(), ledger={"applied": []})
    assert not R.population_applied(
        _pop(), ledger={"applied": ["account_status = Active"]})


def test_every_key_the_facet_resolves_to_counts(semantics):
    facet = R.RequestedFacet(kind=R.KIND_POPULATION, field_key="region_alias",
                             label="the population region_alias = South East")
    assert R.population_applied(
        facet, applied_fields=["collateral_geography"],
        fields={"region_alias": {"canonical_field": "collateral_geography"}})


# --------------------------------------------------------------------------- #
# The one raiser — and why it is the CALLER'S filters and not the whole spec
# --------------------------------------------------------------------------- #
def test_a_drill_through_is_raised_as_a_population(semantics):
    facets = R.drill_population_facets(
        {"collateral_geography": "South East"}, semantics)
    assert [(f.kind, f.field_key) for f in facets] == \
        [(R.KIND_POPULATION, "collateral_geography")]


def test_no_drill_raises_nothing(semantics):
    assert R.drill_population_facets(None, semantics) == []
    assert R.drill_population_facets({}, semantics) == []


def test_the_whole_spec_is_deliberately_not_raised(semantics):
    """Measured before choosing: raising every `spec.filters` entry on the
    point-in-time path would add a population facet to **81 corpus questions**,
    all of them numeric bounds — "LTV above 50%" — which `KIND_THRESHOLD`
    already represents. Two facets for one predicate is the duplicate-claim
    defect, not a fix for the missing one.

    The caller's filters are precisely the narrowings no reader of the QUESTION
    can see, because they were never spoken.
    """
    import inspect

    source = inspect.getsource(R.drill_population_facets)
    assert "extra_filters" in source
    assert "spec" not in source.split('"""')[2], (
        "the raiser must read the caller's filters, not the spec")


# --------------------------------------------------------------------------- #
# The two defects, end to end
# --------------------------------------------------------------------------- #
def test_a_drill_through_on_a_routed_question_is_applied(ask):
    """B17. Every drill-through on a routed question used to refuse."""
    out = ask("Show geographic exposure by ITL3 area.",
              {"collateral_geography": "South East"})
    assert out.get("ok") is True
    assert ("row_population", "collateral_geography", "applied") in _facets(out)


def test_a_drill_through_that_ran_is_recorded(ask):
    """The mirror. Both narrowings ran; the receipt recorded one."""
    out = ask("What is the balance of the front book?",
              {"collateral_geography": "South East"})
    populations = [f for f in _facets(out) if f[0] == "row_population"]
    assert len(populations) == 2, populations
    assert all(status == "applied" for _k, _f, status in populations)
    assert "278 loans" in (out.get("answer") or ""), (
        "the answer must still be the narrowed one; only the receipt changed")


def test_a_route_that_reports_no_narrowing_still_refuses(ask):
    """THE CAN-FAIL. `forecast_extrapolation` answers from a replayed history
    model rather than a resolved frame, so it reports no evidence — and no
    evidence means no certification, whatever the spec carries.

    Its first draft was written against `geo_exposure`, which DOES resolve its
    frame through the governed population seam and narrows on any material
    predicate. The case fired, the evidence was checked, and the premise was
    what was wrong.
    """
    out = ask("What is the completion run rate?",
              {"collateral_geography": "South East"})
    assert out.get("ok") is False
    assert ("row_population", "collateral_geography", "lost") in _facets(out)


def test_no_duplicate_population_reaches_a_receipt(ask):
    """`7c46f81`'s ten-minute defect, guarded on the path this commit adds a
    raiser to: the seasoning owner and the drill ledger can name the same
    governed population."""
    for question, drill in (
            ("What is the balance of the front book?",
             {"seasoning_segment": "Front Book"}),
            ("What is the balance of the front book?",
             {"collateral_geography": "South East"}),
            ("What is the balance by region?",
             {"collateral_geography": "South East"})):
        out = ask(question, drill)
        claims = [(f[0], f[1]) for f in _facets(out)]
        assert len(claims) == len(set(claims)), (question, drill, claims)
