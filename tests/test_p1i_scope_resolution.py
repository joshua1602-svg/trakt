#!/usr/bin/env python3
"""tests/test_p1i_scope_resolution.py — governed portfolio SCOPE resolution.

A phrase that names the POPULATION BEING REPORTED ON is a scope reference. It
is not a row predicate, not a grouping axis, and not a place. That distinction
was being lost four different ways, each producing a confident wrong answer or
a spurious refusal:

    "the entire portfolio"   -> collateral_geography = "Entire"
    "the current portfolio"  -> collateral_geography = "Current"
    "the acquired portfolio" -> grouped by acquired_portfolio_id
    "the funded portfolio"   -> filter funded_status = "Funded"   (LLM path)

All four are one defect: a scope phrase consumed by a resolver looking for
something else. The cure is to CLAIM THE SPAN FIRST so the invalid filter is
never created — never to create one and delete it, which would silently widen
the population it had narrowed.

Every figure here is computed with pandas from the canonical fixture. The MI
scope resolver is never its own oracle.

Run: python -m pytest tests/test_p1i_scope_resolution.py
"""
from __future__ import annotations

import logging
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd
import pytest

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from demo_platform import config as cfg  # noqa: E402
from mi_agent import portfolio_lens as plens  # noqa: E402

BALANCE = "current_outstanding_balance"
LTV = "current_loan_to_value"
AGE = "youngest_borrower_age"
DIRECT_ID = "alp_origination"
ACQUIRED_ID = "alp_acquired"


@pytest.fixture(scope="module")
def governed_env():
    root = cfg.local_blob_root() / "processed" / "platform" / cfg.CLIENT_ID
    if not root.exists():
        pytest.skip("the demonstration platform has not been built — run "
                    "`python -m demo_platform.run_demo --generate --orchestrate`")
    before = dict(os.environ)
    os.environ.update(cfg.mi_env(period_role="current"))
    os.environ["MI_AGENT_DATA_CACHE_TTL"] = "0"
    os.environ["MI_AGENT_LLM_PARSER"] = "off"
    os.environ.pop("ANTHROPIC_API_KEY", None)
    logging.disable(logging.WARNING)
    from mi_agent_api import data_source

    data_source.reset_cache()
    yield
    logging.disable(logging.NOTSET)
    os.environ.clear()
    os.environ.update(before)
    data_source.reset_cache()


@pytest.fixture(scope="module")
def book(governed_env) -> pd.DataFrame:
    from mi_agent_api.data_source import get_dataframe

    return get_dataframe()


@pytest.fixture(scope="module")
def ask(governed_env):
    from trakt_core.context import ExecutionContext
    from mi_agent_api.mi_service import MiQueryRequest, execute_governed_mi_query

    ctx = ExecutionContext.for_internal(cfg.CLIENT_ID)
    cache: Dict[Any, Dict[str, Any]] = {}

    def _ask(question: str, selection: Optional[Any] = None) -> Dict[str, Any]:
        key = (question, tuple(selection) if isinstance(selection, list) else selection)
        if key not in cache:
            req = MiQueryRequest(question=question)
            if selection is not None:
                req.source_portfolio_lens = selection
            cache[key] = (execute_governed_mi_query(req, ctx).result or {})
        return cache[key]

    return _ask


# --------------------------------------------------------------------------- #
# Independent truth
# --------------------------------------------------------------------------- #
def wavg(frame: pd.DataFrame, value: str) -> float:
    valid = frame[[value, BALANCE]].dropna()
    return float((valid[value] * valid[BALANCE]).sum() / valid[BALANCE].sum())


def population(book: pd.DataFrame, ids: Optional[List[str]]) -> pd.DataFrame:
    return book if ids is None else book[book["source_portfolio_id"].isin(ids)]


def scope_ids(envelope) -> List[str]:
    return list((envelope.get("portfolioScope") or {}).get("portfolio_ids") or [])


def coverage(envelope) -> Dict[str, Any]:
    return envelope.get("portfolioCoverage") or {}


def spec_filters(envelope) -> Dict[str, Any]:
    return (envelope.get("spec") or {}).get("filters") or {}


def answer(envelope) -> str:
    return envelope.get("answer") or envelope.get("error") or ""


def kpis(envelope) -> Dict[str, float]:
    for artifact in envelope.get("artifacts") or []:
        if artifact.get("type") == "kpi":
            return {k.get("field"): k.get("rawValue")
                    for k in (artifact.get("kpis") or [])}
    return {}


# =========================================================================== #
# The scope-phrase layer itself
# =========================================================================== #
@pytest.mark.parametrize("phrase", [
    "for the funded portfolio", "in the funded book", "of the whole book",
    "for the entire portfolio", "in the current portfolio",
    "for the acquired portfolio", "of the direct book", "the total portfolio",
])
def test_a_governed_scope_phrase_is_claimed(phrase):
    assert plens.scope_phrase_spans(f"give me balance {phrase} please")


@pytest.mark.parametrize("text", [
    "what is the current ltv?",                       # a MEASURE, not a book
    "show the portfolio as at the current reporting date.",   # TIME, not a book
    "what is the balance in london?",                 # a PLACE
    "show balance by region",                         # a DIMENSION
    "what is the average borrower age?",
])
def test_ordinary_language_is_not_claimed_as_scope(text):
    """A bare qualifier is ordinary English. It becomes a scope reference only
    when it qualifies a book noun — which is what keeps "current LTV" and
    "current reporting date" out of this vocabulary."""
    assert plens.scope_phrase_spans(text) == ()


def test_masking_preserves_offsets():
    """Blanking rather than deleting keeps every other offset valid, so a
    detector reading the remainder sees the sentence it expects."""
    text = "give me balance for the entire portfolio by region"
    masked = plens.mask_scope_phrases(text)
    assert len(masked) == len(text)
    assert "region" in masked and "balance" in masked
    assert "entire portfolio" not in masked


def test_only_a_book_noun_makes_it_a_selected_scope():
    assert plens.names_selected_scope("what is the ltv in the current portfolio?")
    assert not plens.names_selected_scope("what is the current ltv?")


# =========================================================================== #
# A. Funded scope — the whole tape IS the funded book
# =========================================================================== #
@pytest.mark.parametrize("question", [
    "What is the average borrower age in the funded portfolio?",
    "How many loans are in the funded book?",
    "What is the total exposure in the funded portfolio?",
])
def test_funded_scope_creates_no_row_predicate(ask, book, question):
    envelope = ask(question)
    assert envelope["ok"] is True, envelope.get("error")
    filters = spec_filters(envelope)
    assert "funded_status" not in filters
    assert filters.get("collateral_geography") != "Funded"
    assert len(scope_ids(envelope)) >= 2, "the funded book was narrowed"


def test_funded_scope_keeps_the_active_selection(ask):
    """Product ruling: "funded book" names the active dataset, not a scope
    override. With one book selected the answer stays on that book."""
    envelope = ask("How many loans are in the funded book?", ACQUIRED_ID)
    assert envelope["ok"] is True, envelope.get("error")
    assert scope_ids(envelope) == [ACQUIRED_ID]
    assert "funded_status" not in spec_filters(envelope)


# =========================================================================== #
# B. Whole book / entire portfolio — the entire AuM
# =========================================================================== #
@pytest.mark.parametrize("question", [
    "What is the balance of the whole book?",
    "Give me balance, loan count and WA LTV for the entire portfolio.",
])
def test_whole_book_is_the_entire_aum(ask, book, question):
    envelope = ask(question)
    assert envelope["ok"] is True, envelope.get("error")
    assert spec_filters(envelope).get("collateral_geography") not in ("Entire", "Whole")
    assert len(scope_ids(envelope)) >= 2


def test_the_entire_portfolio_reconciles_to_every_book(ask, book):
    envelope = ask("Give me balance, loan count and WA LTV for the entire portfolio.")
    delivered = kpis(envelope)
    assert delivered[f"{BALANCE}_sum"] == pytest.approx(float(book[BALANCE].sum()), rel=1e-9)
    assert delivered["loan_count"] == len(book)
    assert delivered[f"{LTV}_weighted_avg"] == pytest.approx(wavg(book, LTV), rel=1e-9)


# =========================================================================== #
# B2. Sponsored book — a governed scope phrase meaning the FULL AuM
# =========================================================================== #
# In Trakt, "the sponsored book" is the sponsor/client's full AuM across every
# directly originated and acquired portfolio — equivalent to the entire book,
# NOT the direct book and NOT the spv-sponsored cohort.
@pytest.mark.parametrize("question", [
    "What is the balance of the sponsored book?",
    "What is the balance of the sponsored portfolio?",
    "What is the sponsored AUM?",
])
def test_sponsored_book_is_the_full_aum(ask, book, question):
    envelope = ask(question)
    assert envelope["ok"] is True, envelope.get("error")
    # every governed book, not just the direct one
    assert len(scope_ids(envelope)) >= 2
    # and NEVER a source-type predicate narrowing to direct
    assert "source_portfolio_type" not in spec_filters(envelope)


def test_sponsored_book_reconciles_to_every_book(ask, book):
    envelope = ask("Give me balance, loan count and WA LTV for the sponsored book.")
    delivered = kpis(envelope)
    assert delivered[f"{BALANCE}_sum"] == pytest.approx(float(book[BALANCE].sum()), rel=1e-9)
    assert delivered["loan_count"] == len(book)
    assert delivered[f"{LTV}_weighted_avg"] == pytest.approx(wavg(book, LTV), rel=1e-9)


def test_sponsored_book_overrides_a_narrower_selection(ask, book):
    """"The sponsored book" is an EXPLICIT widening to full AuM, so it must
    override an active narrower UI selection rather than defer to it — the
    opposite of "the current book"."""
    envelope = ask("What is the balance of the sponsored book?", ACQUIRED_ID)
    assert envelope["ok"] is True, envelope.get("error")
    assert len(scope_ids(envelope)) >= 2      # widened past the acquired selection
    assert "source_portfolio_type" not in spec_filters(envelope)
    assert kpis(envelope)[f"{BALANCE}_sum"] == pytest.approx(
        float(book[BALANCE].sum()), rel=1e-9)


def test_a_model_emitted_type_predicate_is_refused_under_full_aum(governed_env):
    """The LLM path misread: the model emits source_portfolio_type=direct for
    "the sponsored book". The governed intent is the full-AuM widening, so the
    narrower predicate is refused at parse-time normalisation — never executed
    and then dropped."""
    from mi_agent_api.data_source import get_dataframe, semantics_path
    from mi_agent.mi_query_validator import load_mi_semantics
    from mi_agent.mi_query_spec import MIQuerySpec
    import mi_agent.llm_query_parser as parser

    parser._SEMANTICS_FOR_SCOPE = load_mi_semantics(semantics_path())
    cols = list(get_dataframe().columns)
    for question in ("What is the balance of the sponsored book?",
                     "What is the balance of the whole book?"):
        spec = MIQuerySpec.from_dict({
            "metric": BALANCE, "filters": {"source_portfolio_type": "direct"}})
        rejected = parser.reject_scope_role_filters(spec, question, cols)
        assert rejected and "source_portfolio_type" in rejected[0]
        assert spec.filters == {}


def test_the_sponsor_cohort_id_is_still_addressable(governed_env):
    """The whole-client PHRASE is full AuM, but a genuine spv-sponsored cohort
    stays reachable by its portfolio id — the phrase change does not swallow the
    id path."""
    from mi_agent.portfolio_lens import resolve_lens, names_total_scope
    # An explicit portfolio id is not the full-AuM phrase.
    assert names_total_scope("balance for spv1_sponsored") is False


# =========================================================================== #
# C. Current / selected portfolio
# =========================================================================== #
def test_current_portfolio_is_the_selected_book(ask, book):
    envelope = ask("What is the WA LTV in the current portfolio?", ACQUIRED_ID)
    assert envelope["ok"] is True, envelope.get("error")
    assert scope_ids(envelope) == [ACQUIRED_ID]
    assert spec_filters(envelope).get("collateral_geography") != "Current"


def test_current_book_is_the_selected_book(ask):
    envelope = ask("What is the balance of the current book?", DIRECT_ID)
    assert envelope["ok"] is True, envelope.get("error")
    assert scope_ids(envelope) == [DIRECT_ID]


def test_a_multi_selection_is_exactly_the_selected_books(ask, book):
    """Product ruling: "the current portfolio" with several books selected means
    exactly those books. Resolving to their provenance type would widen the
    answer to books the caller did not choose."""
    envelope = ask("What is the WA LTV in the current portfolio?",
                   [ACQUIRED_ID, DIRECT_ID])
    assert envelope["ok"] is True, envelope.get("error")
    assert sorted(scope_ids(envelope)) == sorted([ACQUIRED_ID, DIRECT_ID])


def test_a_multi_selection_never_falls_back_to_total(governed_env, book):
    """The defect: a list selection fell through to Total, discarding a
    deliberate narrowing — a silent WIDENING, the worst direction to fail."""
    from mi_agent_api.data_source import get_dataframe
    from mi_agent.portfolio_scope import registry_for_frame
    from trakt_core.portfolio import resolve_scope

    registry = registry_for_frame(get_dataframe())
    scope = resolve_scope(registry, [ACQUIRED_ID, DIRECT_ID])
    assert scope.context_kind == "selection"
    assert sorted(scope.portfolio_ids) == sorted([ACQUIRED_ID, DIRECT_ID])
    assert scope.fell_back_to_total is False


def test_an_unrecognised_selection_still_widens_and_says_so(governed_env):
    """Unchanged behaviour: an unknown context widens to Total and discloses it,
    so a stale UI selection cannot mislabel a result."""
    from mi_agent_api.data_source import get_dataframe
    from mi_agent.portfolio_scope import registry_for_frame
    from trakt_core.portfolio import resolve_scope

    registry = registry_for_frame(get_dataframe())
    scope = resolve_scope(registry, ["not_a_book", "also_not"])
    assert scope.is_total and scope.fell_back_to_total is True


# =========================================================================== #
# D / E. Direct and acquired books, from governed metadata
# =========================================================================== #
@pytest.mark.parametrize("question", [
    "What is the balance of the direct book?",
    "Give me balance, loan count and LTV for the direct book.",
    "What is borrower age in the direct book?",
])
def test_the_direct_book_is_the_governed_direct_portfolios(ask, question):
    envelope = ask(question)
    assert envelope["ok"] is True, envelope.get("error")
    assert scope_ids(envelope) == [DIRECT_ID]


@pytest.mark.parametrize("question", [
    "What is the balance of the acquired book?",
    "What is the average borrower age in the acquired book?",
    "Give me balance, count and WA LTV for the acquired portfolio.",
])
def test_the_acquired_book_is_the_governed_acquired_portfolios(ask, question):
    envelope = ask(question)
    assert envelope["ok"] is True, envelope.get("error")
    assert scope_ids(envelope) == [ACQUIRED_ID]


def test_membership_comes_from_metadata_not_from_id_spelling(governed_env):
    """"direct" is whatever the registry currently types direct. A book whose
    id does not start with "direct" still belongs, and a renamed book moves
    without a code change."""
    from trakt_core.portfolio import (PortfolioRecord, PortfolioRegistry,
                                      resolve_scope)

    registry = PortfolioRegistry(portfolios=(
        PortfolioRecord("wibble_01", portfolio_type="direct"),
        PortfolioRecord("wobble_02", portfolio_type="direct"),
        PortfolioRecord("thing_03", portfolio_type="acquired"),
    ))
    assert list(resolve_scope(registry, "direct").portfolio_ids) == [
        "wibble_01", "wobble_02"]
    assert list(resolve_scope(registry, "acquired").portfolio_ids) == ["thing_03"]


def test_a_book_aggregates_every_member(governed_env):
    """D/E of the brief: two direct books aggregate. The fixture carries one of
    each, so the aggregation rule is proven on a registry that has two."""
    from trakt_core.portfolio import (PortfolioRecord, PortfolioRegistry,
                                      resolve_scope)

    registry = PortfolioRegistry(portfolios=(
        PortfolioRecord("direct_0001", portfolio_type="direct"),
        PortfolioRecord("direct_0002", portfolio_type="direct"),
        PortfolioRecord("acquired_001", portfolio_type="acquired"),
        PortfolioRecord("acquired_002", portfolio_type="acquired"),
    ))
    assert list(resolve_scope(registry, "direct").portfolio_ids) == [
        "direct_0001", "direct_0002"]
    assert list(resolve_scope(registry, "acquired").portfolio_ids) == [
        "acquired_001", "acquired_002"]


@pytest.mark.parametrize("question,ids,column,expected", [
    ("What is the balance of the direct book?", [DIRECT_ID], BALANCE, "sum"),
    ("What is the balance of the acquired book?", [ACQUIRED_ID], BALANCE, "sum"),
    ("What is the average borrower age in the acquired book?", [ACQUIRED_ID],
     AGE, "mean"),
    ("What is the average borrower age in the direct book?", [DIRECT_ID],
     AGE, "mean"),
])
def test_scope_calculations_reconcile_independently(ask, book, question, ids,
                                                    column, expected):
    """The DELIVERED figure against pandas over exactly the governed books.

    Read from the KPI artifact rather than the prose: the artifact is what the
    user receives, so if the number is not there it was not answered — and a
    scope defect shows up as the right arithmetic over the wrong population.
    """
    envelope = ask(question)
    assert envelope["ok"] is True, envelope.get("error")
    assert scope_ids(envelope) == ids
    frame = population(book, ids)
    truth = (float(frame[column].sum()) if expected == "sum"
             else float(frame[column].mean()))
    delivered = kpis(envelope)
    key = next((k for k in delivered if k.startswith(column)), None)
    assert key, f"no KPI for {column}: {sorted(delivered)}"
    assert delivered[key] == pytest.approx(truth, rel=1e-9)


# =========================================================================== #
# Comparison must keep working
# =========================================================================== #
def test_direct_versus_acquired_still_compares(ask):
    envelope = ask("Compare the direct and acquired books on LTV.")
    assert envelope["ok"] is True, envelope.get("error")
    text = answer(envelope)
    assert "Direct" in text and "Acquired" in text
    assert "Loan To Value" in text


def test_scope_does_not_become_a_cohort_synonym(ask):
    """P1G identity, preserved: sourcing and seasoning stay different concepts.
    "direct book" must not become a synonym for "new origination"."""
    envelope = ask("Is the credit quality of new origination better or worse "
                   "than the back book?")
    assert envelope["ok"] is False
    assert "how long the loans have been on the book" in answer(envelope).lower()


# =========================================================================== #
# Negative controls — scope words must not suppress legitimate meaning
# =========================================================================== #
def test_a_real_geography_survives(ask, book):
    envelope = ask("What is the balance in London?")
    assert envelope["ok"] is True, envelope.get("error")
    assert spec_filters(envelope).get("collateral_geography") == "London"


def test_a_scope_phrase_and_a_place_can_coexist(ask, book):
    envelope = ask("What is the balance in London for the entire portfolio?")
    assert envelope["ok"] is True, envelope.get("error")
    assert spec_filters(envelope).get("collateral_geography") == "London"


def test_a_genuine_predicate_on_an_absent_field_still_refuses_by_name(governed_env):
    """The role rejection needs BOTH an absent field AND a scope phrase naming
    it. "Show funded versus unfunded applications" is a real status question:
    it keeps its predicate and refuses by name rather than being dropped."""
    from mi_agent_api.data_source import get_dataframe, semantics_path
    from mi_agent.mi_query_validator import load_mi_semantics
    from mi_agent.mi_query_spec import MIQuerySpec
    import mi_agent.llm_query_parser as parser

    parser._SEMANTICS_FOR_SCOPE = load_mi_semantics(semantics_path())
    columns = list(get_dataframe().columns)
    spec = MIQuerySpec.from_dict({"metric": BALANCE,
                                  "filters": {"funded_status": "Funded"}})
    rejected = parser.reject_scope_role_filters(
        spec, "Show funded versus unfunded applications.", columns)
    assert rejected == []
    assert spec.filters == {"funded_status": "Funded"}


def test_a_real_column_is_never_rejected(governed_env):
    from mi_agent_api.data_source import get_dataframe, semantics_path
    from mi_agent.mi_query_validator import load_mi_semantics
    from mi_agent.mi_query_spec import MIQuerySpec
    import mi_agent.llm_query_parser as parser

    parser._SEMANTICS_FOR_SCOPE = load_mi_semantics(semantics_path())
    spec = MIQuerySpec.from_dict({
        "metric": BALANCE, "filters": {"collateral_geography": "London"}})
    rejected = parser.reject_scope_role_filters(
        spec, "balance in the funded portfolio for London",
        list(get_dataframe().columns))
    assert rejected == []
    assert spec.filters == {"collateral_geography": "London"}


def test_the_scope_role_filter_is_rejected_when_both_conditions_hold(governed_env):
    from mi_agent_api.data_source import get_dataframe, semantics_path
    from mi_agent.mi_query_validator import load_mi_semantics
    from mi_agent.mi_query_spec import MIQuerySpec
    import mi_agent.llm_query_parser as parser

    parser._SEMANTICS_FOR_SCOPE = load_mi_semantics(semantics_path())
    spec = MIQuerySpec.from_dict({
        "metric": AGE, "aggregation": "avg",
        "filters": {"funded_status": "Funded"}})
    rejected = parser.reject_scope_role_filters(
        spec, "What is the average borrower age in the funded portfolio?",
        list(get_dataframe().columns))
    assert rejected and "funded_status" in rejected[0]
    assert spec.filters == {}


def test_the_governed_scope_field_is_not_a_row_predicate(governed_env):
    """"The direct book" names the scope. The governed lens resolves it to the
    registry's explicit portfolio ids; a ``source_portfolio_type`` predicate
    alongside that is a second, coarser scope expressed as a filter."""
    from mi_agent_api.data_source import get_dataframe, semantics_path
    from mi_agent.mi_query_validator import load_mi_semantics
    from mi_agent.mi_query_spec import MIQuerySpec
    import mi_agent.llm_query_parser as parser

    parser._SEMANTICS_FOR_SCOPE = load_mi_semantics(semantics_path())
    columns = list(get_dataframe().columns)
    for question, ptype in (("What is the total balance of the direct book?",
                             "direct"),
                            ("What is the total balance of the acquired book?",
                             "acquired")):
        spec = MIQuerySpec.from_dict({
            "metric": BALANCE,
            "filters": {"source_portfolio_type": ptype}})
        rejected = parser.reject_scope_role_filters(spec, question, columns)
        assert rejected and "source_portfolio_type" in rejected[0]
        assert spec.filters == {}


def test_the_governed_scope_id_is_never_rejected(governed_env):
    """The id is the FINER grain, so dropping it could widen. It stays."""
    from mi_agent_api.data_source import get_dataframe, semantics_path
    from mi_agent.mi_query_validator import load_mi_semantics
    from mi_agent.mi_query_spec import MIQuerySpec
    import mi_agent.llm_query_parser as parser

    parser._SEMANTICS_FOR_SCOPE = load_mi_semantics(semantics_path())
    spec = MIQuerySpec.from_dict({
        "metric": BALANCE, "filters": {"source_portfolio_id": [DIRECT_ID]}})
    rejected = parser.reject_scope_role_filters(
        spec, "What is the total balance of the direct book?",
        list(get_dataframe().columns))
    assert rejected == []
    assert spec.filters == {"source_portfolio_id": [DIRECT_ID]}


def test_a_disagreeing_scope_predicate_is_not_silently_dropped(governed_env):
    """Rejection is for EXACT redundancy only. A predicate that contradicts the
    lens the same question resolves to is a conflict, not a duplicate, and must
    not be resolved by quietly deleting one side of it."""
    from mi_agent_api.data_source import get_dataframe, semantics_path
    from mi_agent.mi_query_validator import load_mi_semantics
    from mi_agent.mi_query_spec import MIQuerySpec
    import mi_agent.llm_query_parser as parser

    parser._SEMANTICS_FOR_SCOPE = load_mi_semantics(semantics_path())
    spec = MIQuerySpec.from_dict({
        "metric": BALANCE, "filters": {"source_portfolio_type": "acquired"}})
    rejected = parser.reject_scope_role_filters(
        spec, "What is the total balance of the direct book?",
        list(get_dataframe().columns))
    assert rejected == []
    assert spec.filters == {"source_portfolio_type": "acquired"}


def test_a_scope_predicate_survives_without_a_governed_scope_phrase(governed_env):
    """No scope phrase in the question means no governed scope to defer to, so
    there is nothing that could replace the predicate. It stays."""
    from mi_agent_api.data_source import get_dataframe, semantics_path
    from mi_agent.mi_query_validator import load_mi_semantics
    from mi_agent.mi_query_spec import MIQuerySpec
    import mi_agent.llm_query_parser as parser

    parser._SEMANTICS_FOR_SCOPE = load_mi_semantics(semantics_path())
    spec = MIQuerySpec.from_dict({
        "metric": BALANCE, "filters": {"source_portfolio_type": "direct"}})
    rejected = parser.reject_scope_role_filters(
        spec, "What is the total balance for directly originated lending?",
        list(get_dataframe().columns))
    assert rejected == []
    assert spec.filters == {"source_portfolio_type": "direct"}


def test_rejecting_the_type_predicate_does_not_change_the_population(ask):
    """The governed id filter that replaces it comes from the identical phrase,
    so the answer must be byte-identical to the pre-rejection one."""
    from mi_agent.portfolio_lens import SOURCE_TYPE_FIELD

    for question, ids in (("What is the total balance of the direct book?",
                           [DIRECT_ID]),
                          ("What is the total balance of the acquired book?",
                           [ACQUIRED_ID])):
        envelope = ask(question)
        assert SOURCE_TYPE_FIELD not in spec_filters(envelope)
        assert coverage(envelope).get("portfolios_used") == ids


# =========================================================================== #
# Execution-scope evidence
# =========================================================================== #
def test_the_envelope_states_requested_resolved_and_executed_scope(ask):
    envelope = ask("What is the balance of the acquired book?")
    cov = coverage(envelope)
    assert (cov.get("scope_requested") or {}).get("context_id") == "acquired"
    assert cov.get("portfolios_in_scope") == [ACQUIRED_ID]
    assert cov.get("portfolios_used") == [ACQUIRED_ID]


def test_scope_is_visible_in_the_receipt(ask):
    envelope = ask("What is the balance of the acquired book?")
    receipt = (envelope.get("executionSummary") or {}).get("receipt") or ""
    assert ACQUIRED_ID in receipt
