"""Conversion 4 — `funded_bridge` — and the `dimensions` axis, bridged.

The first NEW contract axis connected to the plan layer since Conversion 2
bridged `time`. Conversions 1–3 all drew on axes already carried, so the
properties below are as much about the BRIDGE being generic as about this route.

Structural guards read the AST, never the source text: `_route_bridge`'s
comments still name `resolve_lens_with_default` to say it is deliberately
unreachable, and a substring guard reads that sentence as the call it denies.
"""
from __future__ import annotations

import ast
import inspect
import os
import warnings
from pathlib import Path

import pytest

_REPO = Path(__file__).resolve().parent.parent


def _tree(rel):
    return ast.parse((_REPO / rel).read_text(encoding="utf-8"))


def _fn(tree, name):
    return next(f for f in ast.walk(tree)
                if isinstance(f, ast.FunctionDef) and f.name == name)


def _calls(node):
    return {getattr(c.func, "id", None) or getattr(c.func, "attr", None)
            for c in ast.walk(node) if isinstance(c, ast.Call)}


class _Dim:
    def __init__(self, concept, role):
        self.candidate_concept, self.role = concept, role


class _Slot:
    def __init__(self, state="empty", raw_text=None):
        self.state, self.raw_text = state, raw_text


class _Time:
    def __init__(self, comparison_period=None):
        self.comparison_period = comparison_period or _Slot()


class _QI:
    def __init__(self, dimensions=(), time=None):
        self.dimensions, self.time = list(dimensions), time or _Time()


# --------------------------------------------------------------------------- #
# The dimensions bridge is GENERIC and obeys the role
# --------------------------------------------------------------------------- #
class TestTheDimensionsBridge:
    def test_it_returns_only_grouping_role_concepts(self):
        from mi_agent_api import analytical_plan as plan
        qi = _QI([_Dim("collateral_geography", "grouping"),
                  _Dim("borrower_type", "filter"),
                  _Dim("ltv_bucket", "unresolved")])
        assert plan.grouping_concepts(qi) == ("collateral_geography",)

    def test_a_filter_is_never_promoted_to_an_axis(self):
        """The collapse the role split exists to prevent."""
        from mi_agent_api import analytical_plan as plan
        qi = _QI([_Dim("borrower_type", "filter")])
        assert plan.grouping_concepts(qi) == ()

    def test_an_unresolved_role_is_never_promoted(self):
        from mi_agent_api import analytical_plan as plan
        assert plan.grouping_concepts(_QI([_Dim("region", "unresolved")])) == ()

    def test_it_preserves_contract_order_and_deduplicates(self):
        from mi_agent_api import analytical_plan as plan
        qi = _QI([_Dim("a", "grouping"), _Dim("b", "grouping"), _Dim("a", "grouping")])
        assert plan.grouping_concepts(qi) == ("a", "b")

    def test_it_knows_nothing_about_funded_bridge(self):
        """A shared bridge: reusable by any later route needing a grouping."""
        from mi_agent_api import analytical_plan as plan
        src = inspect.getsource(plan.grouping_concepts)
        assert "bridge_query" not in src and "funded_bridge" not in src
        assert "question" not in inspect.signature(plan.grouping_concepts).parameters


class TestTheComparisonPeriodIsNotANewAxis:
    def test_a_filled_slot_is_read(self):
        from mi_agent_api import analytical_plan as plan
        qi = _QI(time=_Time(_Slot("filled", "March")))
        assert plan.comparison_period(qi) == "March"

    def test_an_empty_slot_answers_none(self):
        from mi_agent_api import analytical_plan as plan
        assert plan.comparison_period(_QI()) is None

    def test_it_reads_the_existing_time_claim(self):
        """`time` was bridged by Conversion 2; this reads another FIELD on it,
        so no new semantic axis is created."""
        from mi_agent_api import analytical_plan as plan
        src = inspect.getsource(plan.comparison_period)
        assert "comparison_period" in src
        assert "question" not in inspect.signature(plan.comparison_period).parameters


# --------------------------------------------------------------------------- #
# The plan cannot read the question, and the route defers without a contract
# --------------------------------------------------------------------------- #
class TestThePlanCannotReadTheQuestion:
    def test_the_builder_takes_no_question(self):
        from mi_agent_api import analytical_plan as plan
        params = list(inspect.signature(plan.build_funded_bridge_plan).parameters)
        assert not any("question" in p for p in params), params

    def test_the_route_no_longer_owns_any_semantics(self):
        """Checked over the AST — the comments still NAME these to say they are
        deliberately unreachable."""
        fn = _fn(_tree("mi_agent_api/chat_routing.py"), "_route_bridge")
        calls = _calls(fn)
        assert "resolve_lens_with_default" not in calls
        assert "lens_from_selection" not in calls
        assert "_apply_lens_filter" not in calls

    def test_the_registry_helper_no_longer_reads_the_spec(self):
        """`_bridge_dimension` keeps registry resolution and loses the
        semantics: it takes a governed concept, not the spec."""
        from mi_agent_api import chat_routing as routing
        params = list(inspect.signature(routing._bridge_dimension).parameters)
        assert params[0] == "concept"

    def test_the_route_defers_without_a_contract(self):
        from mi_agent_api import chat_routing as routing
        out = routing._route_bridge(
            "funded balance bridge", None, {}, client_id="c", run_id=None,
            output_root="blob://x", portfolio_id=None, as_of=None, semantics={})
        assert out is None


class TestTheOldPopulationOwnerIsUnreachableButNotRetired:
    def test_other_routes_keep_resolve_lens_with_default(self):
        """NOT retired globally — `_route_cohort_progression` and
        `mi_agent_workflow` still depend on it."""
        tree = _tree("mi_agent_api/chat_routing.py")
        callers = {f.name for f in ast.walk(tree)
                   if isinstance(f, ast.FunctionDef)
                   and "resolve_lens_with_default" in _calls(f)}
        assert callers, "the helper lost every owner — this conversion over-reached"
        assert "_route_bridge" not in callers


class TestThePlanLayerIsSharedNotCopied:
    def test_the_bridge_plan_reuses_the_shared_population_step(self):
        builder = _fn(_tree("mi_agent_api/analytical_plan.py"),
                      "build_funded_bridge_plan")
        assert "_population_step" in _calls(builder)

    def test_no_new_primitive_was_declared(self):
        from mi_agent_api import analytical_plan as plan
        ids = {v for k, v in vars(plan).items()
               if k.isupper() and isinstance(v, str) and v.islower()
               and k in {"STACK_PERIODS", "SELECT_POPULATION", "RESOLVE_MEASURE",
                         "GROUP", "RANK", "COMPARE", "PROJECT"}}
        assert ids <= {"stack_periods", "select_population", "resolve_measure",
                       "group", "rank", "compare", "project"}, ids

    def test_a_plan_with_no_governed_dimension_is_blocked_not_defaulted(self):
        """The route's fallback is its own convention; a step must not claim the
        contract asked for it."""
        from mi_agent_api import analytical_plan as plan
        p = plan.build_funded_bridge_plan(_QI(), dimension_key=None,
                                          dimension_label="")
        assert p.blocked and not p.executable


# --------------------------------------------------------------------------- #
# Behaviour is unchanged
# --------------------------------------------------------------------------- #
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

    def ask(q, scope=None):
        return execute_governed_mi_query(
            MiQueryRequest(question=q, source_portfolio_lens=scope), ctx).result or {}
    try:
        yield ask
    finally:
        os.environ.clear()
        os.environ.update(before)


class TestTheAnswersAreUnchanged:
    def test_a_grouped_bridge_delivers_the_same_figures(self, live):
        e = live("Funded balance bridge by region")
        assert e["ok"] is True
        assert (e["metadata"]["groupedBy"]) == ["collateral_geography"]
        assert "£1.93bn" in e["answer"] and "£1.96bn" in e["answer"]

    def test_a_scoped_bridge_is_unchanged(self, live):
        e = live("Funded balance bridge for the acquired book")
        assert e["ok"] is True and "£568.3m" in e["answer"]

    def test_a_missing_dimension_still_refuses(self, live):
        """Defect B's gate, carried through the conversion."""
        e = live("Bridge the funded balance by product")
        assert e["ok"] is False
        assert (e.get("metadata") or {}).get("groupedBy") is None
        assert "£0" not in (e.get("answer") or "")

    def test_an_unheld_portfolio_name_still_refuses(self, live):
        e = live("Funded balance bridge for the Highgate Mortgages Book")
        assert e["ok"] is False
        assert "Highgate Mortgages Book" in (e.get("answer") or "")

    def test_a_governed_portfolio_name_still_refuses_as_before(self, live):
        """S3 refuses before and after. The CONVERSION corrects the population
        this case resolves to (the old path was registry-blind and widened to
        Total), but the guard refuses it either way, so nothing client-visible
        moves."""
        e = live("Funded balance bridge for the ALP Origination Book")
        assert e["ok"] is False
