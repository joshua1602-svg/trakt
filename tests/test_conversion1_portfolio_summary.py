"""Conversion 1 — `portfolio_summary` executes as a plan.

The first shipped route whose execution is a composition over derived
primitives, fed from the interpretation contract.

What these pin is not the economics — those were proven in Phase 0 and re-proven
on the governed population path in Phase 1G. They pin the properties a
conversion can silently lose:

    the plan cannot read the question
    the population comes from governed ids, never a raw type column
    the contract's provenance decides precedence, not a second resolver
    a contract that cannot be built costs a fall-through, not the answer
"""
from __future__ import annotations

import inspect
import os
import sys
import warnings
from pathlib import Path

import pytest

_REPO = Path(__file__).resolve().parents[1]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))


@pytest.fixture(scope="module")
def book():
    warnings.simplefilter("ignore")
    os.environ.setdefault("TRAKT_RUNTIME_MODE", "development")
    from demo_platform import config as cfg
    before = dict(os.environ)
    os.environ.update(cfg.mi_env(period_role="current"))
    os.environ["MI_AGENT_LLM_PARSER"] = "off"
    os.environ["MI_AGENT_LLM_ENABLED"] = "0"
    try:
        yield cfg.CLIENT_ID
    finally:
        os.environ.clear()
        os.environ.update(before)


def _ask(question, default=None):
    from mi_agent_api.mi_service import (MiQueryRequest,
                                         execute_governed_mi_query)
    from demo_platform import config as cfg
    from trakt_core.context import ExecutionContext
    ctx = ExecutionContext.for_internal(cfg.CLIENT_ID)
    return execute_governed_mi_query(
        MiQueryRequest(question=question,
                       source_portfolio_lens=default), ctx).result or {}


class TestThePlanCannotReadTheQuestion:
    """A4's first case is an ABORT: the contract owns a decision and the plan
    rereads the question anyway. This is the structural guard against it."""

    def test_build_plan_takes_no_question_parameter(self):
        from mi_agent_api import portfolio_summary_plan as plan_mod
        params = set(inspect.signature(plan_mod.build_plan).parameters)
        assert not params & {"question", "q", "text", "raw", "raw_question"}

    def test_the_plan_module_never_imports_a_question_resolver(self):
        """It may not reach the owners that read raw text, even indirectly.

        Checked over IMPORTS AND CALLS from the parsed AST, not over substrings:
        the step kind is the string `"source_portfolio_lens"`, and a substring
        search flags it as the resolver it is named after. A guard that fires on
        its own vocabulary teaches people to ignore it.
        """
        import ast
        tree = ast.parse((_REPO / "mi_agent_api/portfolio_summary_plan.py")
                         .read_text(encoding="utf-8"))
        reached = set()
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                reached.update(a.name for a in node.names)
            elif isinstance(node, ast.ImportFrom):
                reached.add(node.module or "")
                reached.update(a.name for a in node.names)
            elif isinstance(node, ast.Call):
                func = node.func
                reached.add(func.id if isinstance(func, ast.Name)
                            else getattr(func, "attr", ""))
        for forbidden in ("resolve_lens", "resolve_lens_with_default",
                          "portfolio_lens", "mentions_portfolio",
                          "names_governed_portfolio", "_dataset_for",
                          "requested_span", "lens_from_selection"):
            assert forbidden not in reached, forbidden


class TestTheRoutedPathReallyBuildsAContract:
    """THE TEST THAT WAS MISSING, and its absence cost a phase.

    Phase 1G wired a construction site onto the routed path and asserted it with
    a lambda through `RouteRequest`. That passed. The real provider raised on
    every question — `Index or []` is a ValueError, not a falsy fallback — and
    the try/except around it returned None silently, so the site never
    constructed anything. The first payload-equivalence run then reported 0
    differences across 54 pairs while the compositional path was taken zero
    times.

    A provider is not wired because a unit test says it is. It is wired when a
    real question against a real frame produces a real contract.
    """

    def test_a_routed_question_produces_a_contract(self, book):
        from mi_agent_api import chat_routing as routing
        seen = []
        original = routing._summary_population

        def _spy(question, source_lens, interpretation, **kw):
            seen.append(interpretation)
            return original(question, source_lens, interpretation, **kw)

        routing._summary_population = _spy
        try:
            _ask("Summarise the acquired book")
        finally:
            routing._summary_population = original
        assert seen, "the route never reached the summary population"
        assert seen[0] is not None, (
            "the routed path handed the handler no interpretation — the "
            "provider raised and was swallowed")
        assert seen[0].source_scope.state == "filled"
        assert seen[0].source_scope.portfolio_ids == ("alp_acquired",)


class TestThePopulationIsGoverned:
    def test_a_category_selects_governed_ids_not_a_type_column(self, book):
        from mi_agent_api import portfolio_context as ctx_mod
        from mi_agent_api import portfolio_summary_plan as plan_mod
        from mi_agent.mi_query_validator import load_mi_semantics
        from mi_agent_api.datasets import semantics_path
        from question_interpretation import projection

        qi = projection.project("Summarise the acquired book",
                                semantics=load_mi_semantics(semantics_path()),
                                registry=ctx_mod.build_registry())
        plan = plan_mod.build_plan(qi, region_column="collateral_geography",
                                   has_portfolio_column=True)
        filters = plan_mod.lens_filters(plan)
        assert filters == {"source_portfolio_id": ["alp_acquired"]}
        assert "source_portfolio_type" not in (filters or {})

    def test_the_whole_book_carries_no_filter(self, book):
        from mi_agent_api import portfolio_context as ctx_mod
        from mi_agent_api import portfolio_summary_plan as plan_mod
        from mi_agent.mi_query_validator import load_mi_semantics
        from mi_agent_api.datasets import semantics_path
        from question_interpretation import projection

        qi = projection.project("Please provide a portfolio summary",
                                semantics=load_mi_semantics(semantics_path()),
                                registry=ctx_mod.build_registry())
        plan = plan_mod.build_plan(qi, region_column="collateral_geography",
                                   has_portfolio_column=True)
        assert plan_mod.lens_filters(plan) is None


class TestUnresolvableDoesNotBlockButDoesNotWiden:
    """EMPTY and UNRESOLVABLE are different, and getting it backwards costs
    route identity.

    EMPTY is "nobody looked" and cannot be planned from. UNRESOLVABLE is "the
    owner looked and found a name this book does not hold" — a REFUSAL whose
    single route-independent owner is the facet layer. Blocking it in the plan
    put a second refusal owner in the path and, measured, moved 23 payload and
    receipt fields because the route deferred to the point-in-time path.
    """

    @pytest.mark.parametrize("question", [
        "Summarise the acquired_001 book",
        "Summarise the Highgate Mortgages Book",
    ])
    def test_an_unresolvable_scope_still_refuses_from_this_route(
            self, book, question):
        result = _ask(question)
        assert (result.get("metadata") or {}).get("route") == "portfolio_summary"
        assert result.get("ok") is False
        assert result.get("controlledRefusal") is True
        assert "11,035" not in (result.get("answer") or "")

    def test_an_empty_claim_blocks_the_plan(self):
        """The case that MUST block: nothing looked, so nothing can be planned,
        and reading it as Total would widen a population the question may have
        narrowed."""
        from mi_agent_api import portfolio_summary_plan as plan_mod
        from question_interpretation.schema import QuestionInterpretation

        qi = QuestionInterpretation(question="anything")   # source_scope EMPTY
        plan = plan_mod.build_plan(qi, region_column=None,
                                   has_portfolio_column=False)
        assert plan.blocked
        assert not plan.executable


class TestTheContractDecidesPrecedence:
    @pytest.mark.parametrize("question,default,loans", [
        ("Please provide a portfolio summary", "acquired", "3,909"),
        ("Please provide a portfolio summary", None, "11,035"),
        ("Summarise the acquired book", "direct", "3,909"),
        ("Summarise the funded book", "acquired", "11,035"),
    ])
    def test_precedence_end_to_end(self, book, question, default, loans):
        assert loans in (_ask(question, default).get("answer") or "")


class TestTheFallThroughIsBehaviourNotScaffolding:
    def test_no_contract_still_answers(self, book):
        """A route that lost its answer to a contract failure would be worse
        than the route before it. The legacy population path stands when no
        interpretation can be built."""
        from mi_agent_api import chat_routing as routing
        summary, label, narrowed = routing._summary_population(
            "Summarise the acquired book", None, None,
            output_root=os.environ["MI_AGENT_ONBOARDING_OUTPUT_ROOT"],
            client_id=os.environ["MI_AGENT_CLIENT_ID"], run_id=None)
        assert summary.get("available") is True
        assert label == "Acquired" and narrowed is True

    def test_both_paths_agree_on_the_same_question(self, book):
        from mi_agent_api import chat_routing as routing
        from mi_agent_api import portfolio_context as ctx_mod
        from mi_agent.mi_query_validator import load_mi_semantics
        from mi_agent_api.datasets import semantics_path
        from question_interpretation import projection

        kw = dict(output_root=os.environ["MI_AGENT_ONBOARDING_OUTPUT_ROOT"],
                  client_id=os.environ["MI_AGENT_CLIENT_ID"], run_id=None)
        qi = projection.project("Summarise the acquired book",
                                semantics=load_mi_semantics(semantics_path()),
                                registry=ctx_mod.build_registry())
        legacy, l_label, l_narrow = routing._summary_population(
            "Summarise the acquired book", None, None, **kw)
        planned, p_label, p_narrow = routing._summary_population(
            "Summarise the acquired book", None, qi, **kw)
        assert (l_label, l_narrow) == (p_label, p_narrow)
        for key in ("available", "period", "reportingDate", "periodCount",
                    "regionColumn", "metrics", "topRegions", "cohorts",
                    "cohortBalances"):
            assert legacy[key] == planned[key], key
