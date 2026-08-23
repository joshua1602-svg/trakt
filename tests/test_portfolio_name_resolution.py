"""Phase 1E — does MI resolve the portfolio NAMES the client can see?

Phase 1D established that React and MI share an identity MODEL (the governed
registry) and not an identity VOCABULARY: MI's text side recognised the STORAGE
convention (`acquired_001`) and nothing else, so every name React renders —
"ALP Origination Book", "ALP Acquired Back Book" — resolved to the whole book.

These tests pin the vocabulary after 1E. The business rule they encode, as
confirmed for this phase:

    Funded Book = ALL funded assets, across Direct AND Acquired (and any other
                  governed funded category). It is NOT a synonym for Direct.
    Direct Book = every governed funded portfolio classified `direct`
    Acquired    = every governed funded portfolio classified `acquired`
    named book  = exactly that one governed portfolio
    unknown name= clarified, never widened to Funded/Total

The live book carries one portfolio per category, so a category collapsing onto
a single portfolio is INVISIBLE there. The fixture registry below has two
acquired portfolios for exactly that reason.
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


#: Two acquired + one direct. "Acquired Book" answering with ONE id is a
#: category collapsing onto a portfolio, and only a registry like this shows it.
FIXTURE_RECORDS = (
    {"source_portfolio_id": "alp_origination", "source_portfolio_type": "direct",
     "source_portfolio_label": "ALP Origination Book"},
    {"source_portfolio_id": "alp_acquired", "source_portfolio_type": "acquired",
     "source_portfolio_label": "ALP Acquired Back Book"},
    {"source_portfolio_id": "nbs_acquired", "source_portfolio_type": "acquired",
     "source_portfolio_label": "NBS Acquired Book"},
)


@pytest.fixture(scope="module")
def registry():
    warnings.simplefilter("ignore")
    os.environ.setdefault("TRAKT_RUNTIME_MODE", "development")
    from trakt_core import portfolio as portfolio_mod
    return portfolio_mod.build_registry(FIXTURE_RECORDS, client_id="phase1e")


def _resolve(registry, question):
    """``(lens name, governed ids, fell_back_to_total)`` for a question."""
    from mi_agent import portfolio_lens as lens_mod
    from trakt_core import portfolio as portfolio_mod
    lens = lens_mod.resolve_lens(question, registry=registry)
    scope = portfolio_mod.resolve_scope(registry, lens_mod.context_id(lens))
    return lens.name, tuple(scope.portfolio_ids), bool(scope.fell_back_to_total)


ALL_IDS = ("alp_origination", "alp_acquired", "nbs_acquired")


class TestNamedPortfoliosResolve:
    """A name the client can SEE is a name MI can READ."""

    @pytest.mark.parametrize("question,expected", [
        ("Summarise the ALP Origination Book", "alp_origination"),
        ("Summarise the ALP Acquired Back Book", "alp_acquired"),
        ("Summarise the NBS Acquired Book", "nbs_acquired"),
    ])
    def test_react_display_label_resolves_to_its_governed_id(
            self, registry, question, expected):
        name, ids, _ = _resolve(registry, question)
        assert name == "cohort"
        assert ids == (expected,)

    @pytest.mark.parametrize("question,expected", [
        ("Summarise the alp_origination book", "alp_origination"),
        ("Summarise the nbs_acquired book", "nbs_acquired"),
    ])
    def test_governed_id_resolves_to_itself(self, registry, question, expected):
        name, ids, _ = _resolve(registry, question)
        assert name == "cohort"
        assert ids == (expected,)

    def test_a_named_portfolio_does_not_broaden_to_its_category(self, registry):
        """The 1D failure: "ALP Acquired Back Book" answering for BOTH acquired
        books is a wrong number, not a loose label."""
        _, ids, _ = _resolve(registry, "Summarise the ALP Acquired Back Book")
        assert ids == ("alp_acquired",)
        assert "nbs_acquired" not in ids


class TestCategoriesStayCategories:
    """A category is every portfolio in it — never one of them."""

    def test_acquired_book_is_every_acquired_portfolio(self, registry):
        name, ids, _ = _resolve(registry, "Summarise the acquired book")
        assert name == "acquired"
        assert set(ids) == {"alp_acquired", "nbs_acquired"}

    def test_direct_book_is_every_direct_portfolio(self, registry):
        name, ids, _ = _resolve(registry, "Summarise the direct book")
        assert name == "direct"
        assert set(ids) == {"alp_origination"}

    def test_funded_book_is_direct_and_acquired_together(self, registry):
        """Business-confirmed for 1E: Funded is the COMPLETE funded population.
        Resolving it to Direct alone would understate this book by 2 of 3
        portfolios."""
        _, ids, _ = _resolve(registry, "Summarise the funded book")
        assert set(ids) == set(ALL_IDS)

    def test_no_scope_named_is_the_whole_funded_book(self, registry):
        name, ids, _ = _resolve(registry, "Please provide a portfolio summary")
        assert name == "total"
        assert set(ids) == set(ALL_IDS)


class TestUnknownNamesAreNotWidened:
    """Naming a book MI cannot find is a question, not a licence."""

    @pytest.mark.parametrize("question,requested", [
        ("Summarise the Highgate Mortgages Book", "Highgate Mortgages Book"),
        ("Summarise the acquired_001 book", "acquired_001"),
    ])
    def test_an_unknown_name_resolves_to_unresolved_not_total(
            self, registry, question, requested):
        from mi_agent import portfolio_lens as lens_mod
        lens = lens_mod.resolve_lens(question, registry=registry)
        assert lens.name == lens_mod.LENS_UNRESOLVED
        assert lens.label == requested
        assert not lens.filters

    @pytest.mark.parametrize("question", [
        "Summarise the Highgate Mortgages Book",
        "Summarise the acquired_001 book",
    ])
    def test_the_governed_contract_discloses_the_widening_it_does_not_prevent(
            self, registry, question):
        """STATED RESIDUAL, pinned so it cannot change silently.

        `resolve_scope` still falls back to every id for an unresolved context.
        What stops the ANSWER widening is the facet layer downstream
        (`unresolved_scope_facets`), asserted end-to-end below. The contract's
        contribution is `fell_back_to_total`, and this test pins that the flag
        is actually raised — a fallback that did NOT set it would leave the
        downstream refusal as the only line of defence, undetectably.
        """
        _, ids, fell_back = _resolve(registry, question)
        assert set(ids) == set(ALL_IDS)
        assert fell_back is True


class TestOrdinaryQuestionsAreUntouched:
    """The unknown-name detector must not fire on ordinary lending English."""

    @pytest.mark.parametrize("question", [
        "Summarise the portfolio",
        "Please provide a portfolio summary",
        "Summarise the acquired book",
        "Summarise the Acquired Book",
        "Summarise the Acquired Back Book",
        "Summarise the direct book",
        "Summarise the funded book",
        "Show funded balance by region",
        "What is the average LTV of the loan book?",
        "How does the direct book compare with the acquired book?",
    ])
    def test_no_unresolved_lens_for_an_ordinary_question(self, registry, question):
        from mi_agent import portfolio_lens as lens_mod
        lens = lens_mod.resolve_lens(question, registry=registry)
        assert lens.name != lens_mod.LENS_UNRESOLVED, lens.label


class TestNoRegistryIsExactlyThePreviousBehaviour:
    """The registry parameter ADDS resolution; it never removes any.

    Every caller that has not been given a registry must behave precisely as it
    did before 1E, or this is a migration that changed answers by omission.
    """

    @pytest.mark.parametrize("question", [
        "Summarise the ALP Origination Book",
        "Summarise the Highgate Mortgages Book",
    ])
    def test_without_a_registry_a_name_still_resolves_to_total(self, question):
        from mi_agent import portfolio_lens as lens_mod
        assert lens_mod.resolve_lens(question).name == lens_mod.LENS_TOTAL

    def test_without_a_registry_a_storage_shaped_id_is_still_a_cohort_lens(self):
        """The pre-1E behaviour for `acquired_001`, unchanged: `_COHORT_ID_RE`
        matches it, so the lens is `cohort` and it is `resolve_scope` that then
        widens to Total. Only a registry can say the id is not held — which is
        why the UNRESOLVED verdict needs one and this call does not get it."""
        from mi_agent import portfolio_lens as lens_mod
        lens = lens_mod.resolve_lens("Summarise the acquired_001 book")
        assert lens.name == lens_mod.LENS_COHORT
        assert lens.filters == {lens_mod.SOURCE_ID_FIELD: "acquired_001"}


class TestVintageIsStillADifferentAxis:
    """§10 — resolving a portfolio NAME must not disturb the vintage axis.

    Phase 1D established that vintage narrows WITHIN whatever population the
    scope selected: "the 2023 vintage of the acquired book" is a seasoning
    filter applied to a source lens, and neither implies the other. 1E changes
    what the source lens recognises, so the independence is re-proved here
    against a NAMED portfolio, which 1D could not do because no name resolved.
    """

    def test_a_vintage_does_not_change_a_named_portfolio_scope(self, registry):
        _, ids, _ = _resolve(registry,
                             "Show the 2023 vintage of the NBS Acquired Book")
        assert ids == ("nbs_acquired",)

    def test_a_named_portfolio_does_not_swallow_the_vintage_year(self, registry):
        """The scope resolution must leave the vintage there to be read. A
        portfolio name that consumed "2023" would silently drop the narrowing
        the question actually asked for."""
        from mi_agent import portfolio_lens as lens_mod
        question = "Show the 2023 vintage of the NBS Acquired Book"
        spans = lens_mod.lens_phrase_spans(question)
        year_at = question.index("2023")
        assert all(not (start <= year_at < end) for start, end in spans)

    def test_a_bare_vintage_still_scopes_to_the_whole_book(self, registry):
        name, ids, _ = _resolve(registry, "Summarise the 2023 vintage")
        assert name == "total"
        assert set(ids) == set(ALL_IDS)


class TestKnownDefectALabelThatCollidesWithSeasoningVocabulary:
    """MEASURED AND NOT FIXED IN 1E — reported in docs/mi_phase1e_report.md.

    `ALP Acquired Back Book` is a governed display label. The lens layer now
    resolves it correctly to `alp_acquired` (asserted above). The POPULATION
    parser, reading the same sentence with a different vocabulary, separately
    reads the substring "Back Book" as `seasoning_segment = Back Book`, and
    `portfolio_summary` cannot apply a population filter — so the answer is a
    controlled REFUSAL.

    Fail-closed, so it is not a wrong number and not a widening; it is a
    governed portfolio that cannot be asked about by its own registered name.
    Fixing it means the span-masking layer (`mask_scope_phrases`) knowing the
    registry, which is a change to every parser call site and is not 1E's.

    Asked by its governed ID the same portfolio answers correctly, which is
    what makes the collision the label's and not the portfolio's.
    """

    def test_the_lens_layer_resolves_the_colliding_label_correctly(self, registry):
        _, ids, _ = _resolve(registry, "Summarise the ALP Acquired Back Book")
        assert ids == ("alp_acquired",)

    @pytest.mark.xfail(strict=True, reason=(
        "KNOWN DEFECT, pre-existing and unchanged by 1E: the population parser "
        "reads 'Back Book' inside the governed label as a seasoning segment, so "
        "the answer refuses. Fix requires registry-aware span masking."))
    def test_the_colliding_label_can_be_asked_about_end_to_end(self):
        import os
        import warnings
        warnings.simplefilter("ignore")
        os.environ.setdefault("TRAKT_RUNTIME_MODE", "development")
        from demo_platform import config as cfg
        before = dict(os.environ)
        os.environ.update(cfg.mi_env(period_role="current"))
        os.environ["MI_AGENT_LLM_PARSER"] = "off"
        os.environ["MI_AGENT_LLM_ENABLED"] = "0"
        try:
            from mi_agent_api.mi_service import (MiQueryRequest,
                                                 execute_governed_mi_query)
            from trakt_core.context import ExecutionContext
            ctx = ExecutionContext.for_internal(cfg.CLIENT_ID)
            result = execute_governed_mi_query(
                MiQueryRequest(question="Summarise the ALP Acquired Back Book"),
                ctx).result or {}
        finally:
            os.environ.clear()
            os.environ.update(before)
        assert result.get("ok") is True
        assert "3,909" in (result.get("answer") or "")


@pytest.fixture(scope="module")
def live_book():
    """The shipped governed book, for the end-to-end assertions below."""
    import os
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


def _ask(client_id, question):
    from mi_agent_api.mi_service import (MiQueryRequest,
                                         execute_governed_mi_query)
    from trakt_core.context import ExecutionContext
    ctx = ExecutionContext.for_internal(client_id)
    return execute_governed_mi_query(MiQueryRequest(question=question),
                                     ctx).result or {}


class TestTheRefusalIsRouteIndependent:
    """Phase 0 recorded this as a governance prerequisite: a receipt proof that
    holds only on the routed path is not a proof.

    Measured with the routed guard alone in place, before `_guard_unresolved_scope`
    was applied at both sites:

        "What is the funded balance by region for the Highgate Mortgages Book?"
        -> ok=True, "Total Balance, grouped by Region, 12 groups, 11,035 loans"

    The whole book, under the name of a book this platform has never onboarded,
    because the question fell through to the point-in-time path. Which route
    happens to claim a question is not a fact about whether its scope resolved.
    """

    @pytest.mark.parametrize("question,requested", [
        # point-in-time path (route is None)
        ("What is the funded balance by region for the Highgate Mortgages Book?",
         "Highgate Mortgages Book"),
        ("What is the funded balance of the acquired_001 book?", "acquired_001"),
        # routed paths
        ("Summarise the Highgate Mortgages Book", "Highgate Mortgages Book"),
        ("How has the Highgate Mortgages Book evolved over time?",
         "Highgate Mortgages Book"),
    ])
    def test_an_unheld_name_refuses_whichever_path_answers(
            self, live_book, question, requested):
        result = _ask(live_book, question)
        answer = result.get("answer") or ""
        assert result.get("ok") is False, answer
        assert result.get("controlledRefusal") is True, answer
        # The wording that asked is quoted back, so the reader can correct it.
        assert requested in answer
        # And no whole-book figure is substituted.
        assert "11,035" not in answer
        assert "1.96bn" not in answer

    def test_a_held_name_is_not_refused_by_this_guard(self, live_book):
        """The guard must fire on names the registry does NOT hold, and only
        those. A guard that refused every named portfolio would be safe and
        useless."""
        result = _ask(live_book, "Summarise the ALP Origination Book")
        assert result.get("ok") is True
        assert "7,126" in (result.get("answer") or "")

    def test_a_question_naming_no_portfolio_is_untouched(self, live_book):
        result = _ask(live_book, "Please provide a portfolio summary")
        assert result.get("ok") is True
        assert "11,035" in (result.get("answer") or "")

    @pytest.mark.parametrize("question", [
        # A governed VALUE of collateral_geography, not a portfolio. Both of
        # these are answered questions in this estate's own golden bank.
        "For the London book, give me balance, number of loans, "
        "weighted-average LTV and average borrower age.",
        "What is the funded balance of the London book?",
    ])
    def test_a_value_this_book_carries_is_a_population_not_a_book_name(
            self, live_book, question):
        """The lens layer has no vocabulary for what values the tape carries, so
        it reads "London Book" as a book name it cannot find. The guard has that
        vocabulary and must not refuse on it.

        Measured before the guard consulted `dimension_values`: eight tests
        across `test_p1e_golden_bank`, `test_p1e_measure_safety` and
        `test_p1e_multi_measure` failed, every one of them on "the London book".
        A false refusal on a question the system answers correctly is a worse
        failure than the widening this facet exists to stop.
        """
        result = _ask(live_book, question)
        answer = result.get("answer") or ""
        assert "not a governed portfolio" not in answer, answer
        assert result.get("controlledRefusal") is not True, answer
