"""Conversion 3 — `geo_exposure` — the properties that must not silently come undone.

Each class is a defect this programme has already paid for once. Every
structural guard reads the AST rather than the source text: three guards in
earlier phases passed by matching a docstring that *denied* the thing they were
looking for, and a guard that reads prose is not reading code.
"""
from __future__ import annotations

import ast
import inspect
import os
import warnings
from pathlib import Path

import pytest

_REPO = Path(__file__).resolve().parent.parent


def _tree(path: str) -> ast.Module:
    return ast.parse((_REPO / path).read_text(encoding="utf-8"))


def _fn(tree: ast.Module, name: str) -> ast.FunctionDef:
    return next(f for f in ast.walk(tree)
                if isinstance(f, ast.FunctionDef) and f.name == name)


def _calls(node: ast.AST) -> set:
    return {getattr(c.func, "id", None) or getattr(c.func, "attr", None)
            for c in ast.walk(node) if isinstance(c, ast.Call)}


class TestThePlanCannotReadTheQuestion:
    def test_the_builder_takes_no_question(self):
        from mi_agent_api import analytical_plan as plan_mod
        params = list(inspect.signature(
            plan_mod.build_geo_exposure_plan).parameters)
        assert not any("question" in p for p in params), params

    def test_the_executor_takes_no_question(self):
        from mi_agent_api import analytical_plan as plan_mod
        params = list(inspect.signature(plan_mod.geo_exposure).parameters)
        assert not any("question" in p for p in params), params

    def test_the_route_no_longer_resolves_a_lens_from_the_question(self):
        """The switch, checked over the AST.

        `_route_geo`'s docstring and comments still MENTION `_resolve_lens` and
        `_apply_lens_filter` — to say they are deliberately unreachable. A
        substring check reads those sentences as the calls they deny.
        """
        geo = _fn(_tree("mi_agent_api/chat_routing.py"), "_route_geo")
        calls = _calls(geo)
        assert "_resolve_lens" not in calls
        assert "_apply_lens_filter" not in calls
        assert "exposure_by_itl3" not in calls


class TestTheOneSemanticInputComesFromTheContract:
    def test_the_plan_reads_the_source_scope_and_nothing_else(self):
        from mi_agent_api import analytical_plan as plan_mod
        builder = _fn(_tree("mi_agent_api/analytical_plan.py"),
                      "build_geo_exposure_plan")
        read = {n.args[1].value for n in ast.walk(builder)
                if isinstance(n, ast.Call)
                and getattr(n.func, "id", "") == "getattr"
                and len(n.args) >= 2 and isinstance(n.args[1], ast.Constant)}
        assert read == {"source_scope"}, read

    def test_the_route_defers_without_a_contract(self):
        """One population owner, or none. A fall-back would leave the lens
        resolver reachable exactly when the contract failed."""
        from mi_agent_api import chat_routing as routing
        out = routing._route_geo(
            "What is the geographic exposure?", {}, client_id="c", run_id=None,
            frame_resolver=lambda *a, **k: None, portfolio_id=None, as_of=None)
        assert out is None


class TestThisRouteHasOneNarrowingOwner:
    def test_the_plan_narrows_through_its_own_governed_filters(self):
        from mi_agent_api import analytical_plan as plan_mod
        scope_frame = _fn(_tree("mi_agent_api/analytical_plan.py"), "scope_frame")
        calls = _calls(scope_frame)
        assert "lens_filters" in calls
        assert "_scope_frame_lens" in calls

    def test_apply_lens_filter_is_kept_for_the_routes_that_still_own_it(self):
        """NOT retired globally. `period_change_route` still calls it, and
        Conversion 3 is not a consolidation exercise."""
        tree = _tree("mi_agent_api/chat_routing.py")
        assert any(isinstance(f, ast.FunctionDef)
                   and f.name == "_apply_lens_filter" for f in ast.walk(tree))
        other = _tree("mi_agent_api/period_change_route.py")
        assert "_apply_lens_filter" in {
            getattr(c.func, "attr", None)
            for c in ast.walk(other) if isinstance(c, ast.Call)}


class TestThePlanLayerIsSharedNotCopied:
    def test_scope_frame_is_in_the_shared_module(self):
        from mi_agent_api import analytical_plan as plan_mod
        assert callable(plan_mod.scope_frame)
        assert Path(plan_mod.__file__).name == "analytical_plan.py"

    def test_the_geo_plan_reuses_the_shared_population_step(self):
        builder = _fn(_tree("mi_agent_api/analytical_plan.py"),
                      "build_geo_exposure_plan")
        assert "_population_step" in _calls(builder)

    def test_no_new_primitive_was_declared(self):
        from mi_agent_api import analytical_plan as plan_mod
        ids = {v for k, v in vars(plan_mod).items()
               if k.isupper() and isinstance(v, str) and v.islower()
               and k in {"STACK_PERIODS", "SELECT_POPULATION", "RESOLVE_MEASURE",
                         "GROUP", "RANK", "COMPARE", "PROJECT"}}
        assert ids <= {"stack_periods", "select_population", "resolve_measure",
                       "group", "rank", "compare", "project"}, ids


class TestAPointInTimeAnswerDoesNotClaimAPeriod:
    def test_the_plan_declares_no_period_step(self):
        """Geographic concentration is answered at ONE date. A `stack_periods`
        step would declare a governance property this answer does not have."""
        from mi_agent_api import analytical_plan as plan_mod

        class _Scope:
            state = "filled"
            base_population = "funded"
            portfolio_ids = ()
            provenance = "default"
            portfolio_label = None
            raw_text = None

        class _QI:
            source_scope = _Scope()

        plan = plan_mod.build_geo_exposure_plan(_QI())
        primitives = [s.primitive for s in plan.steps]
        assert plan_mod.STACK_PERIODS not in primitives, primitives
        assert plan_mod.SELECT_POPULATION in primitives
        assert plan.declares_grouped_by == ("itl3_code",)


@pytest.fixture(scope="module")
def live():
    warnings.simplefilter("ignore")
    os.environ.setdefault("TRAKT_RUNTIME_MODE", "development")
    from demo_platform import config as cfg
    before = dict(os.environ)
    os.environ.update(cfg.mi_env(period_role="current"))
    os.environ["MI_AGENT_LLM_PARSER"] = "off"
    os.environ["MI_AGENT_LLM_ENABLED"] = "0"
    from mi_agent_api.mi_service import MiQueryRequest, execute_governed_mi_query
    from trakt_core.context import ExecutionContext
    ctx = ExecutionContext.for_internal(cfg.CLIENT_ID)

    def ask(question, scope=None):
        return execute_governed_mi_query(
            MiQueryRequest(question=question,
                           source_portfolio_lens=scope), ctx).result or {}
    try:
        yield ask
    finally:
        os.environ.clear()
        os.environ.update(before)


class TestTheAnswersAreUnchanged:
    def test_the_whole_book_is_unchanged(self, live):
        e = live("What is the geographic exposure?")
        assert (e.get("metadata") or {}).get("route") == "geo_exposure"
        assert e.get("ok") is True
        assert "£83.4m" in e["answer"] and "4.2% of the book" in e["answer"]

    @pytest.mark.parametrize("question,expected", [
        ("Geographic exposure for the acquired book", ("£26.3m", "Acquired")),
        ("Geographic exposure for the direct book", ("£57.1m", "Direct")),
    ])
    def test_a_named_scope_narrows_and_says_so(self, live, question, expected):
        e = live(question)
        assert e.get("ok") is True
        for token in expected:
            assert token in e["answer"], e["answer"]

    def test_the_caller_scope_narrows_the_same_way_the_question_does(self, live):
        by_question = live("Geographic exposure for the acquired book")["answer"]
        by_caller = live("What is the geographic exposure?", "acquired")["answer"]
        assert "£26.3m" in by_question and "£26.3m" in by_caller

    def test_an_unheld_name_still_refuses_rather_than_widening(self, live):
        e = live("Geographic exposure for the Highgate Mortgages Book")
        assert e.get("ok") is False
        assert "Highgate Mortgages Book" in (e.get("answer") or "")
        assert "£83.4m" not in (e.get("answer") or "")
