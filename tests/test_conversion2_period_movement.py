"""Conversion 2 — `period_movement` executes as a plan.

The second converted route, and the experiment in whether Conversion 1's
overrun was one-off infrastructure.

What these pin is what a conversion can silently lose. The economics are proven
by the envelope snapshot (36 pairs, 0 differences); these hold the properties
that a passing economic check would not notice:

    the plan cannot read the question
    BOTH semantic inputs come from the contract — scope AND window
    the window's wording stays with the module that owns it
    the shared plan layer is shared, not copied
"""
from __future__ import annotations

import ast
import inspect
import os
import sys
import textwrap
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
    from demo_platform import config as cfg
    from mi_agent_api.mi_service import (MiQueryRequest,
                                         execute_governed_mi_query)
    from trakt_core.context import ExecutionContext
    ctx = ExecutionContext.for_internal(cfg.CLIENT_ID)
    return execute_governed_mi_query(
        MiQueryRequest(question=question,
                       source_portfolio_lens=default), ctx).result or {}


def _project(question, **kw):
    from mi_agent.mi_query_validator import load_mi_semantics
    from mi_agent_api import portfolio_context as ctx_mod
    from mi_agent_api.datasets import semantics_path
    from question_interpretation import projection
    return projection.project(question, semantics=load_mi_semantics(semantics_path()),
                              registry=ctx_mod.build_registry(), **kw)


class TestThePlanCannotReadTheQuestion:
    def test_the_movement_plan_takes_no_question_parameter(self):
        from mi_agent_api import analytical_plan as plan_mod
        params = set(inspect.signature(plan_mod.build_period_movement_plan).parameters)
        assert not params & {"question", "q", "text", "raw", "raw_question"}

    def test_the_route_calls_no_question_resolver(self):
        """Over the AST, not the source text: the docstrings name the resolvers
        in order to say they are no longer called."""
        from mi_agent_api import chat_routing as routing
        tree = ast.parse(textwrap.dedent(
            inspect.getsource(routing._route_period_movement)))
        called = set()
        for n in ast.walk(tree):
            if isinstance(n, ast.Call):
                called.add(getattr(n.func, "id", "")
                           or getattr(n.func, "attr", ""))
        assert "_resolve_lens" not in called
        assert "requested_span" not in called


class TestBothSemanticInputsComeFromTheContract:
    def test_the_window_comes_from_the_contract(self, book):
        from mi_agent_api import analytical_plan as plan_mod
        for question, periods in (
                ("What has changed since last month over the last 3 months", 3),
                ("What has changed month on month this year?", 12),
                ("What has changed since last month?", 1)):
            qi = _project(question)
            plan = plan_mod.build_period_movement_plan(
                qi, region_column="collateral_geography", has_portfolio_column=True)
            assert plan_mod.span_periods(plan) == periods, question

    def test_a_question_naming_no_span_uses_the_routes_own_default(self, book):
        """"since last period" names no countable span. The route has always
        compared one period in that case, and the plan carries that default
        rather than inventing one."""
        from mi_agent_api import analytical_plan as plan_mod
        qi = _project("How has the book changed since last period?")
        assert qi.time.window_periods is None
        plan = plan_mod.build_period_movement_plan(
            qi, region_column=None, has_portfolio_column=False)
        assert plan_mod.span_periods(plan) == plan_mod.DEFAULT_SPAN_PERIODS == 1

    def test_the_population_comes_from_governed_ids(self, book):
        from mi_agent_api import analytical_plan as plan_mod
        qi = _project("What has changed since last month in the acquired book?")
        plan = plan_mod.build_period_movement_plan(
            qi, region_column="collateral_geography", has_portfolio_column=True)
        assert plan_mod.lens_filters(plan) == {
            "source_portfolio_id": ["alp_acquired"]}

    def test_the_contract_and_the_owner_agree_on_the_window(self, book):
        """The contract must carry what `requested_span` reads, or the plan
        cannot be built from it."""
        from mi_agent import period_request
        for question in ("What has changed since last month over the last 3 months",
                         "What has changed month on month this year?",
                         "What has changed since last month?"):
            span = period_request.requested_span(question)
            qi = _project(question)
            assert qi.time.window_periods == (span.periods if span else None), question


class TestTheSpanWordingStaysWithItsOwner:
    def test_span_from_claim_takes_the_label_from_the_claim(self, book):
        """An earlier cut guessed the label — "this year" for 12 periods — which
        would have made the route a second author of wording `period_request`
        owns. Every field now comes from the claim."""
        from mi_agent import period_request
        qi = _project("What has changed month on month this year?")
        span = period_request.span_from_claim(qi.time)
        assert span.periods == 12
        assert span.label == qi.time.trend_window.raw_text

    def test_no_window_gives_no_span(self, book):
        from mi_agent import period_request
        qi = _project("What has changed since last month?")
        # One period is the route's default, not a named window.
        assert period_request.span_from_claim(qi.time).periods == 1
        qi2 = _project("How has the book changed since last period?")
        assert period_request.span_from_claim(qi2.time) is None


class TestThePlanLayerIsSharedNotCopied:
    def test_every_route_plan_reads_the_one_population_step(self):
        """Two copies of the EMPTY / UNRESOLVABLE / FILLED decision would drift,
        and Conversion 1 measured what getting it backwards costs: 23 payload
        and receipt fields moved.

        This used to name the two plan builders that existed when it was
        written. Conversion 3 added a third, and a test that lists the current
        builders is a changelog rather than a guard. The invariant is asserted
        instead: there is exactly ONE definition of the population step, and
        EVERY plan builder reaches it. A fourth conversion that quietly copies
        the decision fails here without anyone remembering to update a list.
        """
        from mi_agent_api import analytical_plan as plan_mod
        tree = ast.parse(Path(plan_mod.__file__).read_text(encoding="utf-8"))
        defs = [f.name for f in ast.walk(tree)
                if isinstance(f, ast.FunctionDef) and f.name == "_population_step"]
        assert defs == ["_population_step"], defs

        builders = {f.name for f in ast.walk(tree)
                    if isinstance(f, ast.FunctionDef)
                    and f.name.startswith("build_") and f.name.endswith("plan")}
        callers = {f.name for f in ast.walk(tree)
                   if isinstance(f, ast.FunctionDef)
                   and any(isinstance(n, ast.Call)
                           and getattr(n.func, "id", "") == "_population_step"
                           for n in ast.walk(f))}
        assert builders, "no plan builders found — the guard is not looking at "\
                         "the plan module"

        # A ROUTE THAT DOES NOT NARROW MUST SAY SO, AND THE CLAIM IS CHECKED.
        #
        # Conversion 5 added the first plan builder for a route with no
        # source-portfolio narrowing. The first version of this rule let such a
        # builder opt out by writing the literal string `"whole_dataset"`, and
        # an independent audit showed a builder that SHOULD narrow could claim
        # it and pass. A magic string is not a governed fact.
        #
        # So there are now exactly TWO governed constructors for the population
        # step, and every builder must reach one of them:
        #
        #   _population_step       the route narrows; the scope states decide
        #   _whole_dataset_step    the route does not narrow, AND it proves it
        #                          against `Recogniser.lens_aware`, the single
        #                          place the platform declares which routes
        #                          narrow
        #
        # A builder can no longer exempt itself by assertion.
        whole = {f.name for f in ast.walk(tree)
                 if isinstance(f, ast.FunctionDef)
                 and any(isinstance(n, ast.Call)
                         and getattr(n.func, "id", "") == "_whole_dataset_step"
                         for n in ast.walk(f))}
        assert builders <= (callers | whole), sorted(builders - callers - whole)

        # And the checked constructor may not be handed a literal that bypasses
        # the registry: it takes a ROUTE NAME, and the step it returns is
        # BLOCKED for a route the registry declares lens-aware.
        from mi_agent_api import analytical_plan as plan_mod
        assert plan_mod._whole_dataset_step("geo_exposure").blocked, (
            "a route declared lens_aware must not obtain a whole-dataset step")
        assert plan_mod._whole_dataset_step("temporal_compare").blocked is None
        assert plan_mod._whole_dataset_step("not_a_route").blocked, (
            "an unprovable claim must block rather than pass")

    def test_there_is_exactly_one_plan_module(self):
        modules = sorted(p.name for p in (_REPO / "mi_agent_api").glob("*plan*.py"))
        assert modules == ["analytical_plan.py"], modules

    def test_compare_reuses_the_existing_implementation(self):
        """A2's fourth threshold is a NEW implementation of a primitive. The
        movement's deltas already exist in `movement_summary`; the plan calls
        them rather than re-deriving them."""
        from mi_agent_api import analytical_plan as plan_mod
        src = inspect.getsource(plan_mod.period_movement)
        assert "summary_mod.period_movement(" in src


class TestTheAnswersAreUnchanged:
    @pytest.mark.parametrize("question,default,fragment", [
        ("What has changed since last month?", None, "£18.1m"),
        ("What has changed since last month in the acquired book?", None, "£6.6m"),
        ("What has changed since last month in the direct book?", None, "£11.5m"),
        ("What has changed since last month?", "acquired", "£6.6m"),
    ])
    def test_movement_figures(self, book, question, default, fragment):
        result = _ask(question, default)
        assert (result.get("metadata") or {}).get("route") == "period_movement"
        assert fragment in (result.get("answer") or "")

    def test_an_unresolvable_scope_still_refuses_from_this_route(self, book):
        result = _ask("What has changed since last month in the acquired_001 book")
        assert (result.get("metadata") or {}).get("route") == "period_movement"
        assert result.get("controlledRefusal") is True

    def test_no_contract_means_the_route_defers(self, book):
        from mi_agent_api import chat_routing as routing
        assert routing._route_period_movement(
            "What has changed since last month?", None, {},
            client_id="c", run_id=None, output_root="blob://x",
            portfolio_id=None, as_of=None) is None
