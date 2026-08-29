"""Phase 1C — what production ACTUALLY does with source scope.

These tests do not assert what the product SHOULD do. They pin what it does, so
the compositional path can be measured against measured behaviour rather than
against a description of it — and so that a change to any of it is visible.

Two groups:

``TestProductionPrecedence``   §3 Cases A-D: how the question and a caller
                               default interact, executed through the real
                               resolvers.
``TestGovernedVsRawResolution`` §6/§7: the registry decides group membership,
                               NOT the data's type column — with a focused
                               fixture that makes the difference observable,
                               because the shipped book cannot (it has exactly
                               one portfolio per type, so both paths coincide).
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


def _effective(question, default_selection):
    """The lens production ends up using, through the real precedence rule."""
    from mi_agent import portfolio_lens as lens_mod
    default = (lens_mod.lens_from_selection(default_selection)
               if default_selection is not None else None)
    return lens_mod.resolve_lens_with_default(question, default)


class TestProductionPrecedence:
    """§3 — measured, so the contract can be judged against behaviour."""

    def test_case_a_no_explicit_scope_the_caller_default_wins(self, book):
        """A question silent about source scope yields to the dropdown."""
        lens = _effective("Please provide a portfolio summary", "acquired")
        assert lens.name == "acquired"

    def test_case_b_explicit_scope_beats_the_caller_default(self, book):
        """A question that names a scope overrides the dropdown."""
        lens = _effective("Summarise the direct book", "acquired")
        assert lens.name == "direct"

    def test_case_b2_an_explicit_whole_book_reading_also_beats_the_default(self, book):
        """THE PHASE 1B BLOCKER, pinned.

        These questions MENTION a portfolio and resolve to `total`, so the
        question wins and the dropdown is ignored. A consumer that sees only the
        resolved scope (`total`) cannot tell this from Case A, where `total`
        means "silent" and the dropdown MUST win.
        """
        for question in ("portfolio summary across all portfolios",
                         "summarise the portfolio excluding the acquired book"):
            assert _effective(question, "acquired").name == "total", question
            # ... while a genuinely silent question with the same default does not.
            assert _effective("Please provide a portfolio summary",
                              "acquired").name == "acquired"

    def test_case_c_neither_gives_the_unrestricted_population(self, book):
        lens = _effective("Please provide a portfolio summary", None)
        assert lens.name == "total"
        assert lens.filters == {}

    def test_case_d_an_unresolvable_scope_widens_to_the_whole_book(self, book):
        """§3 Case D asked us to show production does NOT silently widen.

        IT DOES. Pinned here as measured behaviour, not endorsed: the governed
        registry falls back to Total for a portfolio id it does not hold, and
        records `fell_back_to_total`. See
        docs/mi_phase1c_report.md — this is why Phase 1C stopped.
        """
        from mi_agent_api import portfolio_context as ctx_mod

        scope = ctx_mod.resolve_context("acquired_001", discover_pipeline=False).scope
        assert scope.fell_back_to_total is True
        assert scope.requested_context_id == "acquired_001"
        assert scope.filters == {}          # no narrowing at all
        assert len(scope.portfolio_ids) > 1  # the whole book

    def test_case_d_the_widening_is_now_refused_rather_than_printed(self, book):
        """CLOSED IN PHASE 1E. This test used to pin the opposite.

        What it recorded in Phase 1C, verbatim from the run:

            ok=True, "At 30 June 2026 the portfolio (acquired_001) holds
            11,035 loans with a funded balance of GBP1.96bn", facets [],
            warnings [], portfolioScope.fell_back_to_total True

        — the requested scope's NAME against the WHOLE BOOK's figures, with the
        one object that knew (the scope) never reaching the reader. `acquired_001`
        is a storage folder name (Phase 1D), so it names no governed portfolio at
        all, and the answer presented it as though it did.

        1E resolves the lens against the governed registry, which can now say "I
        do not hold this" instead of selecting nothing and falling back. The
        request is raised as a LOST narrowing and honour-or-clarify refuses.

        The `fell_back_to_total` assertion above is DELIBERATELY KEPT in
        `test_case_d_an_unresolvable_scope_widens_to_the_whole_book`: the
        governed contract still widens, and still only discloses. What changed
        is that the answer no longer prints the widening as a result.
        """
        from mi_agent_api.mi_service import MiQueryRequest, execute_governed_mi_query
        from trakt_core.context import ExecutionContext

        ctx = ExecutionContext.for_internal(os.environ["MI_AGENT_CLIENT_ID"])
        result = execute_governed_mi_query(
            MiQueryRequest(question="Summarise the acquired_001 book"), ctx).result or {}
        answer = result.get("answer") or ""
        assert result.get("ok") is False
        assert result.get("controlledRefusal") is True
        # The wording that asked is quoted back, so the reader can correct it.
        assert "acquired_001" in answer
        # And the whole book's figure appears nowhere.
        assert "11,035" not in answer
        assert "1.96bn" not in answer


class TestGovernedVsRawResolution:
    """§6/§7 — the registry decides membership, not the data's type column."""

    def test_on_the_shipped_book_both_paths_coincide(self, book):
        """Why a discrimination fixture is needed at all.

        One portfolio per type here, so `type == 'acquired'` and
        `id in ['alp_acquired']` select the same rows. Phase 1A's equivalence
        was numerically right for exactly this reason and proved nothing about
        the semantics.
        """
        from mi_agent_api import evolution as evolution_mod

        frames = evolution_mod.funded_frames(
            os.environ["MI_AGENT_ONBOARDING_OUTPUT_ROOT"], book, None)
        df = frames[-1]["df"]
        raw = evolution_mod._scope_frame_lens(df, {"source_portfolio_type": "acquired"})
        governed = evolution_mod._scope_frame_lens(
            df, {"source_portfolio_id": ["alp_acquired"]})
        assert len(raw) == len(governed)

    def test_the_registry_not_the_data_decides_group_membership(self):
        """THE DISCRIMINATION TEST.

        Two books both carrying ``source_portfolio_type='acquired'`` IN THE
        DATA, but only one typed ``acquired`` in the governed registry — the
        other is untyped, exactly as ``spv1_sponsored`` is on the shipped book.

        Governed resolution selects the ONE registry member. A raw
        ``source_portfolio_type`` filter selects BOTH. A compositional path that
        filtered on the raw lens would answer for a population the registry does
        not place in the group, and no economic check on the shipped book could
        catch it.
        """
        import pandas as pd
        from mi_agent_api import evolution as evolution_mod
        from trakt_core import portfolio as portfolio_mod

        records = [
            {"source_portfolio_id": "acq_one", "source_portfolio_type": "acquired",
             "source_portfolio_label": "Acquired One"},
            # In the data as `acquired`; UNTYPED in the governed registry.
            {"source_portfolio_id": "acq_two", "source_portfolio_label": "Acquired Two"},
        ]
        registry = portfolio_mod.build_registry(records, client_id="fixture")
        scope = portfolio_mod.resolve_scope(registry, "acquired")

        assert scope.portfolio_ids == ("acq_one",), scope.portfolio_ids
        assert scope.fell_back_to_total is False

        frame = pd.DataFrame({
            "source_portfolio_id": ["acq_one", "acq_one", "acq_two"],
            "source_portfolio_type": ["acquired", "acquired", "acquired"],
            "current_outstanding_balance": [100.0, 200.0, 900.0],
        })
        governed = evolution_mod._scope_frame_lens(frame, scope.filters)
        raw = evolution_mod._scope_frame_lens(
            frame, {"source_portfolio_type": "acquired"})

        assert float(governed["current_outstanding_balance"].sum()) == 300.0
        assert float(raw["current_outstanding_balance"].sum()) == 1200.0
        assert len(governed) == 2 and len(raw) == 3, (
            "the raw path would answer for a book the registry does not place "
            "in the group")
