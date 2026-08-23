"""Phase 1D — do React and MI mean the same thing by "portfolio"?

React's selector is built from ``portfolio_context.context_index()`` — the
governed registry (frontend/mi-agent-ui/src/state/useWorkspace.ts:300, "derived
from the governed hierarchy ... exactly one source of portfolio truth"). So the
two share an identity MODEL.

They do not share an identity VOCABULARY, and these tests pin where it breaks.
They assert MEASURED behaviour, not desired behaviour: each one that starts
failing means someone changed the thing it describes, which is the point.
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


def _ids_for_question(question):
    """The governed portfolio ids MI ends up selecting for a question."""
    from mi_agent import portfolio_lens as lens_mod
    from mi_agent_api import portfolio_context as ctx_mod
    lens = lens_mod.resolve_lens(question)
    scope = ctx_mod.resolve_context(lens_mod.context_id(lens),
                                    discover_pipeline=False).scope
    return tuple(scope.portfolio_ids)


def _ids_for_context(context_id, client_id=None):
    from mi_agent_api import portfolio_context as ctx_mod
    return tuple(ctx_mod.resolve_context(context_id,
                                         discover_pipeline=False).scope.portfolio_ids)


class TestReactAndMiShareTheModel:
    def test_react_builds_its_selector_from_the_governed_registry(self, book):
        """Not a duplicate table — the same context index MI resolves against."""
        from mi_agent_api import portfolio_context as ctx_mod
        index = ctx_mod.context_index(client_id=book)
        assert index["available"] is True
        ids = {c["context_id"] for c in index["contexts"]}
        assert {"total", "direct", "acquired"} <= ids
        for ctx in index["contexts"]:
            assert ctx["label"], ctx


class TestCategoryLabelsResolve:
    """The two type categories work, from several phrasings."""

    @pytest.mark.parametrize("question", [
        "Summarise the Direct book", "Summarise the direct book",
        "Direct portfolio summary"])
    def test_direct_resolves(self, question, book):
        assert _ids_for_question(question) == _ids_for_context("direct")

    @pytest.mark.parametrize("question", [
        "Summarise the Acquired book", "Summarise the acquired book",
        "Acquired portfolio summary"])
    def test_acquired_resolves(self, question, book):
        assert _ids_for_question(question) == _ids_for_context("acquired")


class TestNamedPortfolioLabelsDoNotResolve:
    """THE GAP. A label React renders, typed by the client, is not understood —
    and every failure widens SILENTLY to the whole book."""

    @pytest.mark.parametrize("question", [
        "Summarise the ALP Origination Book",
        "portfolio summary for ALP Origination Book",
        "Summarise the spv1_sponsored portfolio",
        "portfolio summary for spv1_sponsored",
    ])
    def test_a_react_label_widens_to_the_whole_book(self, question, book):
        selected = _ids_for_question(question)
        assert selected == _ids_for_context("total"), selected
        assert len(selected) > 1

    def test_a_governed_portfolio_id_is_not_recognised_in_text(self, book):
        """The ids this client actually has are invisible to the lens layer."""
        from mi_agent import portfolio_lens as lens_mod
        for pid in ("alp_acquired", "alp_origination", "spv1_sponsored"):
            lens = lens_mod.resolve_lens(f"Summarise the {pid} book")
            assert lens.name == "total", (pid, lens.name)

    def test_the_storage_convention_IS_recognised(self):
        """And the shape MI does recognise is the storage/seed convention —
        `direct_NNN` / `acquired_NNN`, which this client does not use.

            _COHORT_ID_RE = r"\\b((?:direct|acquired)_\\d+)\\b"
        """
        from mi_agent import portfolio_lens as lens_mod
        for pid in ("acquired_001", "direct_001"):
            lens = lens_mod.resolve_lens(f"Summarise the {pid} book")
            assert lens.name == "cohort" and lens.cohort_id == pid


class TestMultipleAcquiredBooks:
    """§5 — with two acquired books, a named-portfolio question answers for both."""

    @staticmethod
    def _registry():
        from trakt_core import portfolio as portfolio_mod
        return portfolio_mod.build_registry([
            {"source_portfolio_id": "alp_acquired",
             "source_portfolio_type": "acquired",
             "source_portfolio_label": "ALP Acquired Back Book"},
            {"source_portfolio_id": "nbs_acquired",
             "source_portfolio_type": "acquired",
             "source_portfolio_label": "NBS Acquired Book"},
            {"source_portfolio_id": "alp_origination",
             "source_portfolio_type": "direct",
             "source_portfolio_label": "ALP Origination Book"},
        ], client_id="two_acquired")

    def test_the_acquired_category_selects_every_acquired_book(self):
        from trakt_core import portfolio as portfolio_mod
        scope = portfolio_mod.resolve_scope(self._registry(), "acquired")
        assert set(scope.portfolio_ids) == {"alp_acquired", "nbs_acquired"}

    def test_a_named_acquired_book_answers_for_both(self):
        """THE COINCIDENCE, broken.

        On the shipped book "Summarise the ALP Acquired Back Book" looks right —
        it returns the acquired population, because there is exactly one acquired
        book. It resolves to the TYPE, not the portfolio. Add a second acquired
        book and the same question answers for a portfolio the client did not
        name.
        """
        from mi_agent import portfolio_lens as lens_mod
        from trakt_core import portfolio as portfolio_mod

        registry = self._registry()
        lens = lens_mod.resolve_lens("Summarise the ALP Acquired Back Book")
        got = portfolio_mod.resolve_scope(registry, lens_mod.context_id(lens))
        wanted = portfolio_mod.resolve_scope(registry, "alp_acquired")

        assert lens.name == "acquired"                       # the TYPE
        assert set(got.portfolio_ids) == {"alp_acquired", "nbs_acquired"}
        assert set(wanted.portfolio_ids) == {"alp_acquired"}
        assert set(got.portfolio_ids) != set(wanted.portfolio_ids)


class TestVintageIsNotPortfolioIdentity:
    """§4/§10 — vintage is a cohort/time filter WITHIN a population, and the
    year only survives on the progression path.

    Measured, not desired. `_cohort_vintage` is reachable only from
    `_cohort_progression_recognizer`, which requires a PROGRESSION MARKER first
    (`evolve|progress|season|over time|trend|...`). A point-in-time vintage
    question therefore never sets `spec.cohort_vintage`.
    """

    @staticmethod
    def _spec(question):
        from mi_agent.llm_query_parser import parse_with_repair
        from mi_agent.mi_query_validator import load_mi_semantics
        from mi_agent_api.data_source import semantics_path
        spec, _meta = parse_with_repair(question, load_mi_semantics(semantics_path()),
                                        llm_enabled=False)
        return spec

    def test_vintage_does_not_change_the_portfolio_scope(self, book):
        """A vintage narrows within whatever population the scope selected."""
        from mi_agent import portfolio_lens as lens_mod
        assert lens_mod.resolve_lens(
            "Show the 2023 vintage of the acquired book").name == "acquired"
        assert lens_mod.resolve_lens("Summarise the 2023 vintage").name == "total"

    def test_the_year_survives_only_with_a_progression_marker(self, book):
        with_marker = self._spec("How has the 2023 vintage evolved over time?")
        without = self._spec("Summarise the 2023 vintage")
        assert with_marker.cohort_vintage == "2023"
        assert with_marker.cohort_progression is True
        # Same year, same word "vintage", no progression marker -> dropped.
        assert without.cohort_vintage is None
        assert without.cohort_progression is False

    def test_a_point_in_time_vintage_year_is_dropped_without_disclosure(self, book):
        """The client asks for ONE vintage and is shown ALL of them.

        The answer groups by vintage across every cohort the book holds, and the
        only facet raised is the grouping DIMENSION. The requested YEAR — a
        narrowing — is not represented and nothing says it was dropped.
        """
        from mi_agent_api.mi_service import MiQueryRequest, execute_governed_mi_query
        from trakt_core.context import ExecutionContext

        ctx = ExecutionContext.for_internal(os.environ["MI_AGENT_CLIENT_ID"])
        result = execute_governed_mi_query(
            MiQueryRequest(question="Summarise the 2023 vintage"), ctx).result or {}
        summary = result.get("executionSummary") or {}
        facets = summary.get("facets") or []

        assert result.get("ok") is True
        assert "group(s)" in (result.get("answer") or "")
        assert any(f.get("kind") == "grouping_dimension"
                   and f.get("label") == "vintage" for f in facets)
        # Nothing anywhere mentions the year that was asked for and dropped.
        assert not any("2023" in str(f.get("label")) for f in facets), facets
        assert not any("2023" in str(w) for w in (result.get("warnings") or []))

    def test_the_progression_path_refuses_honestly(self, book):
        """By contrast, where the vintage IS carried and cannot be applied, the
        product says so — which is the behaviour the other path lacks."""
        from mi_agent_api.mi_service import MiQueryRequest, execute_governed_mi_query
        from trakt_core.context import ExecutionContext

        ctx = ExecutionContext.for_internal(os.environ["MI_AGENT_CLIENT_ID"])
        result = execute_governed_mi_query(
            MiQueryRequest(question="How has the 2023 vintage evolved over time?"),
            ctx).result or {}
        assert result.get("ok") is False
        assert "vintage" in (result.get("answer") or "").lower()
