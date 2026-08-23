"""Phase 1F — why `portfolio_summary` still cannot be converted.

Phase 0 proved the economics: 9 cases, 3 scopes, 0 differences, no bespoke
exception and no new primitive. Phase 1B stopped on a contract property; Phase
1C stopped on a production defect that Phase 1E has since closed. This module
pins what remains, so the next attempt starts from measurements rather than
from a re-derivation.

THE RULE THESE PROTECT: a compositional plan may read the interpretation
contract and nothing else. It may not reread the question (abort condition A4,
first case) and it may not be handed the lens from outside — Phase 0 recorded
that as `lensFiltersSuppliedExternally` and said plainly that identical
economics on an externally-supplied lens "prove the composition; they do not
prove the plan could be built".

Tests that assert MEASURED behaviour pass. Tests that assert the property the
conversion NEEDS and the contract does not yet have are `xfail(strict=True)`,
so they announce themselves the moment someone closes the gap.
"""
from __future__ import annotations

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


def _claim(question, book):
    from mi_agent.mi_query_validator import load_mi_semantics
    from mi_agent_api import portfolio_context as ctx_mod
    from mi_agent_api.datasets import semantics_path
    from question_interpretation import projection
    return projection.project(
        question, semantics=load_mi_semantics(semantics_path()),
        registry=ctx_mod.build_registry()).source_scope


def _shipped_scope(question, default):
    from mi_agent_api import chat_routing as routing
    return routing._resolve_lens(question, default).name


class TestTheRouteSurfaceIsWhatWeThinkItIs:
    """Ownership is asked of the recogniser, never assumed from a list.

    Phase 0's rule, and it earned itself: "Summarise the front book" reads like
    a portfolio summary and is NOT claimed by this route, so comparing on it
    would have manufactured an equivalence that means nothing.
    """

    @pytest.mark.parametrize("question", [
        "Please provide a portfolio summary",
        "Summarise the acquired book",
        "Summarise the direct book",
        "portfolio summary across all portfolios",
        "Summarise the ALP Origination Book",
        "Summarise the acquired_001 book",
        "Summarise the funded book",
    ])
    def test_the_recogniser_claims_these(self, question):
        from mi_agent_api import chat_routing as routing
        assert routing._is_portfolio_summary(question) is True

    @pytest.mark.parametrize("question", [
        "Summarise the front book",                        # a seasoning population
        "What is the portfolio position for the direct book?",
        "Summarise the portfolio by region",               # a stratification
        "For the London book, give me balance, number of loans, "
        "weighted-average LTV and average borrower age.",  # a governed value
    ])
    def test_the_recogniser_does_not_claim_these(self, question):
        from mi_agent_api import chat_routing as routing
        assert routing._is_portfolio_summary(question) is False


class TestThePhase1BBlockerIsClosed:
    """CLOSED IN PHASE 1G. This class pinned the blocker; it now pins the fix.

    The contract carried WHICH scope was resolved and not WHETHER the question
    named one, and the second fact is what decides precedence against a
    caller-supplied default. `SourceScopeClaim.provenance` now carries it.
    """

    def test_a_silent_question_defers_to_the_caller_default(self, book):
        """`source_portfolio_lens` is a live field on MiQueryRequest, populated
        from the workspace dropdown. A surface measured only at None is
        measured on one third of its inputs."""
        assert _shipped_scope("Please provide a portfolio summary", None) == "total"
        assert _shipped_scope("Please provide a portfolio summary", "acquired") == "acquired"

    def test_an_explicit_whole_book_question_overrides_the_caller_default(self, book):
        assert _shipped_scope("portfolio summary across all portfolios", "acquired") == "total"

    def test_the_two_readings_resolve_to_the_same_scope(self, book):
        """Both still resolve to `total` — the scope was never the problem."""
        silent = _claim("Please provide a portfolio summary", book)
        explicit = _claim("portfolio summary across all portfolios", book)
        assert silent.scope == explicit.scope == "total"
        assert silent.state == explicit.state == "filled"
        # ... and production answers them differently under the same default.
        assert _shipped_scope("Please provide a portfolio summary", "acquired") == "acquired"
        assert _shipped_scope("portfolio summary across all portfolios", "acquired") == "total"

    def test_the_contract_can_decide_precedence(self, book):
        """The Phase 1F stop condition, closed. Same scope, different claim."""
        silent = _claim("Please provide a portfolio summary", book)
        explicit = _claim("portfolio summary across all portfolios", book)
        assert silent.as_dict() != explicit.as_dict()
        assert silent.provenance == "default"
        assert explicit.provenance == "explicit_user"
        assert silent.stated_by_user is False
        assert explicit.stated_by_user is True


class TestTheTotalScopeVocabularyIsEven:
    """LEVELLED IN PHASE 1G, by decision and with authorisation.

    Phase 1F reported this as a finding it would not fix: two explicit
    whole-book phrasings had opposite precedence, because "across all
    portfolios" was in `_TOTAL_TERMS` and "the funded book" was not. Phase 1G §1
    makes the business meaning authoritative — Funded Book is the complete
    funded population — so the phrase is now explicit and wins, like its synonym.

    The client-visible effect is measured by name in
    `migration_phase0/funded_book_precedence_change.py`.
    """

    def test_across_all_portfolios_overrides_the_dropdown(self, book):
        assert _shipped_scope("portfolio summary across all portfolios", "acquired") == "total"

    def test_the_funded_book_now_does_too(self, book):
        assert _shipped_scope("Summarise the funded book", "acquired") == "total"
        assert _claim("Summarise the funded book", book).provenance == "explicit_user"

    def test_a_funded_MEASURE_is_still_not_a_scope(self, book):
        """The constraint that keeps the vocabulary honest. Bare "funded" names
        a MEASURE — "funded balance", "funded amount" — and reading a measure as
        a scope is the silent mutation the lens vocabulary exists to prevent.
        Only the noun phrases that name the BOOK were added."""
        assert _shipped_scope("What is the funded balance?", "acquired") == "acquired"
        assert _shipped_scope("Show funded balance by region", "acquired") == "acquired"


class TestPhase1EPopulationSemanticsHold:
    """§6, measured end to end at the shipped default (no dropdown selection)."""

    @staticmethod
    def _ask(question):
        from mi_agent_api.mi_service import (MiQueryRequest,
                                             execute_governed_mi_query)
        from trakt_core.context import ExecutionContext
        from demo_platform import config as cfg
        ctx = ExecutionContext.for_internal(cfg.CLIENT_ID)
        return execute_governed_mi_query(
            MiQueryRequest(question=question), ctx).result or {}

    def test_funded_book_is_every_funded_asset(self, book):
        assert "11,035" in (self._ask("Summarise the funded book").get("answer") or "")

    def test_direct_book_is_the_direct_portfolios(self, book):
        assert "7,126" in (self._ask("Summarise the direct book").get("answer") or "")

    def test_acquired_book_is_the_acquired_portfolios(self, book):
        assert "3,909" in (self._ask("Summarise the acquired book").get("answer") or "")

    def test_a_named_portfolio_is_only_that_portfolio(self, book):
        answer = self._ask("Summarise the ALP Origination Book").get("answer") or ""
        assert "7,126" in answer and "11,035" not in answer

    def test_an_unknown_explicit_scope_refuses(self, book):
        result = self._ask("Summarise the acquired_001 book")
        assert result.get("controlledRefusal") is True
        assert "11,035" not in (result.get("answer") or "")

    def test_a_valid_portfolio_with_no_funded_rows_states_it(self, book):
        result = self._ask("Summarise the spv1_sponsored portfolio")
        answer = result.get("answer") or ""
        assert "no funded loans in spv1_sponsored" in answer
        assert "11,035" not in answer

    def test_the_back_book_label_collision_still_refuses(self, book):
        """§7 — the safe refusal must not disappear. A conversion that turns
        this into an answer has silently dropped one of the two interpretations
        rather than arbitrating between them."""
        result = self._ask("Summarise the ALP Acquired Back Book")
        assert result.get("controlledRefusal") is True


class TestThePlanMustResolveThroughTheRegistry:
    """§8 — positive governed population evidence, not a raw type filter.

    The shadow plan rebuilds its lens with `lens_from_selection(scope)`, which
    yields `{'source_portfolio_type': 'acquired'}`. Production resolves
    `{'source_portfolio_id': ['alp_acquired']}` through the registry. On the
    shipped book they select identical rows — one portfolio per type — which is
    exactly why no economic check on this book can catch the difference.

    Phase 1C measured the divergence on a two-portfolio fixture: governed 300.00
    against raw 1200.00. A conversion must carry the governed path.
    """

    def test_the_shadow_plan_rebuilds_a_raw_type_filter(self, book):
        from mi_agent import portfolio_lens as lens_mod
        rebuilt = lens_mod.lens_from_selection("acquired")
        assert rebuilt.filters == {"source_portfolio_type": "acquired"}

    def test_production_resolves_a_governed_id_list(self, book):
        from mi_agent_api import portfolio_context as ctx_mod
        scope = ctx_mod.resolve_context("acquired", discover_pipeline=False).scope
        assert dict(scope.filters) == {"source_portfolio_id": ["alp_acquired"]}

    @pytest.mark.xfail(strict=True, reason=(
        "The shadow plan's `lens_for` resolves the raw type filter, not the "
        "governed id list production uses. Invisible on the shipped book; "
        "Phase 1C measured 300.00 vs 1200.00 on a two-portfolio fixture."))
    def test_the_plan_and_production_select_the_same_way(self, book):
        from mi_agent import portfolio_lens as lens_mod
        from mi_agent_api import portfolio_context as ctx_mod
        rebuilt = lens_mod.lens_from_selection("acquired")
        governed = ctx_mod.resolve_context("acquired", discover_pipeline=False).scope
        assert set(rebuilt.filters) == set(governed.filters)


class TestTheRoutedPathBuildsAnInterpretation:
    """WIRED IN PHASE 1G §9. This class pinned the gap; it now pins the wiring.

    Phase 1F found that a routed question never built a `QuestionInterpretation`
    — the single construction site was on the point-in-time path, which routing
    bypasses. `try_route` now supplies one through `RouteRequest`, built with the
    governed registry and the caller's workspace selection, so a converted
    handler has a contract to plan from.

    NOTHING READS IT YET: it is carried, exactly as the workflow's has been.
    """

    def test_the_routed_path_now_has_a_construction_site(self):
        """Searched by IMPORT of the projection entry points, not by call name:
        the workflow aliases `from_parts` to `_qi_build`, so a call-name search
        misses the one site that matters and matches the projection module's
        own internal call instead."""
        import re
        pattern = re.compile(
            r"from\s+question_interpretation(?:\.projection)?\s+import[^\n]*"
            r"(from_parts|project)\b")
        hits = []
        for path in _REPO.rglob("*.py"):
            rel = path.relative_to(_REPO).as_posix()
            if ("/tests/" in rel
                    or rel.startswith(("tests/", "migration_phase0/",
                                       "compositional_plan_scoping/",
                                       "question_interpretation/"))):
                continue
            if pattern.search(path.read_text(encoding="utf-8", errors="ignore")):
                hits.append(rel)
        assert sorted(hits) == ["mi_agent/mi_agent_workflow.py",
                                "mi_agent_api/chat_routing.py"], hits

    def test_the_routed_site_passes_the_registry_and_the_caller_scope(self):
        """The two inputs that make the claim usable: the governed identity, and
        the provenance that decides precedence. The point-in-time site still
        passes neither, which is a known and separate gap."""
        text = (_REPO / "mi_agent_api/chat_routing.py").read_text(encoding="utf-8")
        call = text[text.index("return _qi_build("):]
        call = call[:call.index("\n\n")]
        assert "registry=" in call and "caller_scope=" in call

    def test_a_route_request_carries_the_contract(self):
        """The handoff itself: one object, resolved lazily, memoised per
        request — the same shape `history_model` uses, so adding it changed no
        handler signature."""
        from mi_agent_api.recogniser_registry import RouteRequest
        assert hasattr(RouteRequest, "resolve_interpretation")
        marker = object()
        request = RouteRequest(
            question="q", spec=None, spec_dict={}, semantics={}, view="funded",
            client_id="c", run_id=None, portfolio_id=None,
            interpretation_provider=lambda: marker)
        assert request.resolve_interpretation() is marker

    def test_a_provider_that_raises_does_not_cost_the_answer(self):
        """A plan that cannot be built must refuse on the contract's own terms,
        not lose the request to an exception."""
        from mi_agent_api.recogniser_registry import RouteRequest

        def _boom():
            raise RuntimeError("no contract")

        request = RouteRequest(
            question="q", spec=None, spec_dict={}, semantics={}, view="funded",
            client_id="c", run_id=None, portfolio_id=None,
            interpretation_provider=_boom)
        assert request.resolve_interpretation() is None
