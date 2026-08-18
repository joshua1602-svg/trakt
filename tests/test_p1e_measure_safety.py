#!/usr/bin/env python3
"""tests/test_p1e_measure_safety.py — what P1E must REFUSE to do.

Breadth is only worth having if it cannot be used to produce a confident wrong
answer. P1E lets one question carry several governed measures; each test here
is a way that could go wrong, and the assertion is that it does not.

The governing rule, unchanged from P0:

    A number the user asked for is either CALCULATED or NAMED AS NOT
    CALCULATED. It is never quietly absent, and it is never replaced.

Run: python -m pytest tests/test_p1e_measure_safety.py
"""
from __future__ import annotations

import logging
import os
import sys
from pathlib import Path
from typing import Any, Dict, List

import pytest

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from demo_platform import config as cfg  # noqa: E402
from mi_agent import execution_receipt as er  # noqa: E402
from mi_agent.mi_query_spec import MAX_MEASURES, MIQuerySpec  # noqa: E402

BALANCE = "current_outstanding_balance"
LTV = "current_loan_to_value"
VALUATION = "current_valuation_amount"


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
def ask(governed_env):
    from trakt_core.context import ExecutionContext
    from mi_agent_api.mi_service import MiQueryRequest, execute_governed_mi_query

    ctx = ExecutionContext.for_internal(cfg.CLIENT_ID)
    cache: Dict[str, Dict[str, Any]] = {}

    def _ask(question: str) -> Dict[str, Any]:
        if question not in cache:
            cache[question] = (execute_governed_mi_query(
                MiQueryRequest(question=question), ctx).result or {})
        return cache[question]

    return _ask


@pytest.fixture(scope="module")
def semantics(governed_env):
    from mi_agent_api.data_source import semantics_path
    from mi_agent.mi_query_validator import load_mi_semantics

    return load_mi_semantics(semantics_path())


@pytest.fixture(scope="module")
def columns(governed_env):
    from mi_agent_api.data_source import get_dataframe

    return list(get_dataframe().columns)


def executed(envelope: Dict[str, Any]) -> List[Dict[str, Any]]:
    return list((envelope.get("queryTrace") or {}).get("executed_measures") or [])


def receipt(envelope: Dict[str, Any]) -> str:
    return (envelope.get("executionSummary") or {}).get("receipt") or ""


def answer(envelope: Dict[str, Any]) -> str:
    return envelope.get("answer") or envelope.get("error") or ""


# =========================================================================== #
# 1. A measure the dataset does not carry
# =========================================================================== #
def test_an_unavailable_measure_refuses_the_whole_question(ask):
    """Three of four is the failure mode, not the mitigation.

    The book has no credit score. Returning balance, count and LTV — and saying
    nothing about the fourth figure — reads as a complete answer, so the whole
    question refuses and names the measure it could not calculate.
    """
    envelope = ask("Give me balance, loan count, weighted-average LTV and "
                   "average borrower credit score.")
    assert envelope["ok"] is False
    assert "credit score" in answer(envelope).lower()
    assert "£" not in answer(envelope), "a figure was offered despite refusing"


def test_an_unavailable_measure_is_no_more_permitted_than_an_unknown_one(ask):
    """Consistency: a measure the agent UNDERSTANDS but cannot calculate and a
    measure it does not understand are treated the same way. Being more relaxed
    about words it failed to parse would invert the safety ordering."""
    unavailable = ask("Give me balance, loan count and average credit score.")
    unrecognised = ask("What are the total balance, number of loans, weighted "
                       "average LTV and weighted average rate?")
    assert unavailable["ok"] is False and unrecognised["ok"] is False


# =========================================================================== #
# 2. A measure the parser dropped
# =========================================================================== #
@pytest.mark.parametrize("question,dropped", [
    ("What are the total balance, number of loans, weighted average LTV and "
     "weighted average rate?", "weighted average rate"),
    ("Balance, loans, WA LTV and WA coupon please.", "loans"),
])
def test_a_dropped_measure_slot_is_named_not_silently_omitted(ask, question,
                                                              dropped):
    """The defect this guard exists for.

    Each question names four measures; the vocabulary understands three. Before
    the guard the agent answered the three it understood and said nothing about
    the fourth — a confident, incomplete answer with no indication anything was
    missing.
    """
    envelope = ask(question)
    assert envelope["ok"] is False, "a 3-of-4 answer was returned"
    assert dropped in answer(envelope).lower()


def test_the_guard_is_structural_not_a_vocabulary_list(semantics, columns):
    """It reports slots it cannot interpret, which is why it also covers
    phrasings nobody has taught it yet."""
    from mi_agent.llm_query_parser import unresolved_measure_slots

    slots = unresolved_measure_slots(
        "give me balance, loan count and the flurble index", semantics, columns)
    assert slots == ("the flurble index",)


@pytest.mark.parametrize("question", [
    "Give me the current funded balance, loan count, weighted-average LTV and "
    "weighted-average interest rate.",
    "For the London book, give me balance, number of loans, weighted-average "
    "LTV and average borrower age.",
    "Show me balance, loan count and weighted-average LTV by region.",
    "For borrowers over 85, what are the total balance, loan count, "
    "weighted-average LTV and average interest rate?",
    "What is the total balance?",
    "Show me balance by region",
    "For loans with LTV above 60%, give me balance and loan count.",
])
def test_the_guard_does_not_fire_on_a_question_it_understands(ask, question):
    """Scope control. A guard that refuses good questions is not a safety
    feature — it is an outage."""
    envelope = ask(question)
    assert envelope["ok"] is True, envelope.get("error")


# =========================================================================== #
# 3. A different scope per measure
# =========================================================================== #
def test_two_populations_in_one_question_are_refused(ask):
    """P1E composes measures over ONE population. "balance in London and the
    loan count in the South East" is two questions, and answering it from
    either population alone would attach the wrong scope to one of the
    figures."""
    envelope = ask("Give me the balance in London and the loan count in the "
                   "South East.")
    assert envelope["ok"] is False
    assert "london" in answer(envelope).lower()


def test_one_scope_applies_to_every_measure_in_the_set(ask):
    """The positive half of the same rule: a single scope is applied to all of
    them, and the receipt states it once."""
    envelope = ask("For the London book, give me balance, number of loans, "
                   "weighted-average LTV and average borrower age.")
    assert envelope["ok"] is True, envelope.get("error")
    assert len(executed(envelope)) == 4
    assert receipt(envelope).count("London") == 1


# =========================================================================== #
# 4. The aggregation is governed, not chosen by the question
# =========================================================================== #
def test_an_aggregation_the_registry_forbids_is_refused_not_downgraded(
        semantics, columns):
    """Reporting a simple mean where a weighted average is governed — or the
    reverse — is a DIFFERENT NUMBER, not a rounding difference. The measure is
    dropped with a reason rather than quietly recalculated another way."""
    from mi_agent.mi_query_executor import resolve_measures

    spec = MIQuerySpec.from_dict({"measures": [
        {"field": BALANCE, "aggregation": "sum"},
        {"field": VALUATION, "aggregation": "weighted_avg"},
    ]})
    resolved, unavailable = resolve_measures(spec, semantics, columns)
    assert [m.key for m in resolved] == [BALANCE]
    assert any("weighted_avg" in note for note in unavailable)


def test_the_house_weighting_is_applied_without_being_asked(ask):
    """"weighted-average LTV" is balance-weighted because the registry says so,
    not because the question named a weight field."""
    envelope = ask("Give me balance and weighted-average LTV.")
    ltv = next(m for m in executed(envelope) if m["canonical_field"] == LTV)
    assert ltv["aggregation"] == "weighted_avg"
    assert ltv["weight_field"] == BALANCE


def test_an_explicitly_simple_average_is_labelled_as_one(ask):
    """A governed alternative the registry allows is honoured — and the receipt
    says which one ran, so "average" and "weighted average" can never be read
    as each other."""
    envelope = ask("Give me the simple average LTV and the loan count.")
    assert envelope["ok"] is True, envelope.get("error")
    ltv = next(m for m in executed(envelope) if m["canonical_field"] == LTV)
    assert ltv["aggregation"] == "avg"
    assert "Average Current LTV" in receipt(envelope)
    assert "Weighted-average Current LTV" not in receipt(envelope)


@pytest.mark.xfail(reason="KNOWN LIMITATION, pre-dates P1E and is not scoped to "
                          "it: 'median' is not in the aggregation-qualifier "
                          "vocabulary, so a median request resolves to the "
                          "registry default. The receipt still names what ran "
                          "('Total Balance'), so the substitution is disclosed "
                          "rather than hidden — but the request is not honoured.",
                   strict=True)
def test_a_median_request_is_honoured(ask):
    envelope = ask("Give me the median balance and the loan count.")
    balance = next(m for m in executed(envelope)
                   if m["canonical_field"] == BALANCE)
    assert balance["aggregation"] == "median"


# =========================================================================== #
# 5. Duplicates
# =========================================================================== #
def test_the_same_measure_said_twice_is_reported_once(ask):
    """"balance and total balance" is one governed measure. Two identical
    figures side by side would read as a discrepancy that does not exist."""
    envelope = ask("Give me balance, total balance, loan count and number of "
                   "loans.")
    assert envelope["ok"] is True, envelope.get("error")
    fields = [m["field"] for m in executed(envelope)]
    assert len(fields) == len(set(fields)) == 2


# =========================================================================== #
# 6. More measures than the contract carries
# =========================================================================== #
def test_the_cap_is_a_documented_number_not_an_accident():
    assert MAX_MEASURES == 4


def test_beyond_the_cap_the_question_refuses_rather_than_truncating(ask):
    """Six measures asked for, none returned. Answering the first four would be
    a silent truncation — and the user has no way to know which four."""
    envelope = ask("Give me balance, loan count, weighted-average LTV, "
                   "weighted-average interest rate, average borrower age and "
                   "average property value.")
    assert envelope["ok"] is False
    assert executed(envelope) == []
    assert "more than one measure" in answer(envelope)


def test_a_set_at_the_cap_is_answered(ask):
    """The boundary is inclusive: four is supported, five is not."""
    envelope = ask("Give me the current funded balance, loan count, "
                   "weighted-average LTV and weighted-average interest rate.")
    assert envelope["ok"] is True, envelope.get("error")
    assert len(executed(envelope)) == MAX_MEASURES


# =========================================================================== #
# 7. The P0 boundary is not weakened by any of the above
# =========================================================================== #
def test_the_unresolved_measure_facet_refuses_it_never_merely_discloses():
    """A missing NUMBER refuses. Only facets that change an answer's SHAPE — a
    grouping — may stand as an explicit partial."""
    assert er.KIND_UNRESOLVED_MEASURE in er.NUMBER_OR_SUBJECT_FACETS
    assert er.KIND_UNRESOLVED_MEASURE not in er.SHAPE_FACETS


def test_a_relationship_question_is_not_claimed_as_a_measure_set(semantics,
                                                                 columns):
    """"ltv vs interest rate" names two governed measures but asks how they
    RELATE, which only a loan-level plot expresses. A measure set would answer
    a different question with two confident numbers."""
    from mi_agent.llm_query_parser import _deterministic_parse

    spec, _ = _deterministic_parse("ltv vs interest rate", semantics, columns)
    assert spec.chart_type == "scatter"
    assert not spec.measures or len(spec.measures) <= 1


def test_a_filter_subject_is_not_read_as_a_second_measure(semantics, columns):
    """"balance below 75% LTV" measures ONE thing and filters on another."""
    from mi_agent.llm_query_parser import detect_measure_set

    assert detect_measure_set("how much balance is below 75% ltv",
                              semantics, columns) == []


def test_a_grouping_axis_is_not_read_as_a_measure(semantics, columns):
    """"balance by region and age bucket" is one measure across two axes."""
    from mi_agent.llm_query_parser import detect_measure_set

    assert detect_measure_set("show balance by region and age bucket",
                              semantics, columns) == []


# =========================================================================== #
# 8. The model's role: it may NAME governed concepts, never calculate
# =========================================================================== #
def test_the_prompt_documents_the_measure_array_so_the_model_can_express_a_set():
    """The original defect on the LLM path was not misunderstanding — the model
    said in its own explanation that it understood all four measures. It had
    nowhere in the contract to PUT them, so it returned one."""
    from mi_agent.llm_query_parser import _SYSTEM_INSTRUCTIONS

    assert "'measures'" in _SYSTEM_INSTRUCTIONS
    assert "loan_count" in _SYSTEM_INSTRUCTIONS
    # And the two things it must NOT do with the array it has been given.
    assert "Do NOT pick one measure and drop the rest" in _SYSTEM_INSTRUCTIONS
    assert "do NOT invent a measure" in _SYSTEM_INSTRUCTIONS


def test_the_house_weighting_convention_overrides_the_models_choice():
    """A bare "average" on a PERCENT measure means the balance-weighted average
    in MI. The model does not know that, and a simple mean is a DIFFERENT
    NUMBER — so the governed aggregation replaces the model's."""
    from mi_agent.llm_query_parser import reconcile_measure_aggregations

    llm = MIQuerySpec.from_dict({"measures": [
        {"field": BALANCE, "aggregation": "sum"},
        {"field": LTV, "aggregation": "avg"},          # the model's choice
    ]})
    deterministic = MIQuerySpec.from_dict({"measures": [
        {"field": BALANCE, "aggregation": "sum"},
        {"field": LTV, "aggregation": "weighted_avg"},  # the house convention
    ]})
    adjusted = reconcile_measure_aggregations(llm, deterministic)
    assert adjusted == [f"measure_aggregation:{LTV}"]
    assert llm.measures[1]["aggregation"] == "weighted_avg"


def test_reconciliation_can_only_move_an_aggregation_never_a_measure():
    """The narrow mandate: it may not add, drop or re-point a measure. A model
    that recognised something the deterministic parser did not keeps it."""
    from mi_agent.llm_query_parser import reconcile_measure_aggregations

    llm = MIQuerySpec.from_dict({"measures": [
        {"field": LTV, "aggregation": "avg"},
        {"field": "some_field_only_the_model_saw", "aggregation": "avg"},
    ]})
    deterministic = MIQuerySpec.from_dict({"measures": [
        {"field": LTV, "aggregation": "weighted_avg"},
        {"field": BALANCE, "aggregation": "sum"},
    ]})
    reconcile_measure_aggregations(llm, deterministic)
    assert [m["field"] for m in llm.measures] == [
        LTV, "some_field_only_the_model_saw"]
    assert llm.measures[1]["aggregation"] == "avg"


def test_a_measure_the_model_invents_cannot_reach_a_number(semantics, columns):
    """The model may name governed concepts. A field that is not in the
    registry is reported, not calculated — no value is fabricated for it."""
    from mi_agent.mi_query_executor import resolve_measures

    spec = MIQuerySpec.from_dict({"measures": [
        {"field": BALANCE, "aggregation": "sum"},
        {"field": "esg_alignment_score", "aggregation": "avg"},
    ]})
    resolved, unavailable = resolve_measures(spec, semantics, columns)
    assert [m.key for m in resolved] == [BALANCE]
    assert any("esg_alignment_score" in note for note in unavailable)


# =========================================================================== #
# 9. A measure a ROUTED capability cannot express
# =========================================================================== #
def test_a_comparison_compares_every_measure_including_the_loan_count(ask):
    """A loan count used to vanish from a comparison.

    It was excluded because it is not an arithmetic measure over a value — and
    that was right — but the exclusion was silent, so a four-measure question
    came back with three figures and nothing to show the fourth was missing.

    Loan cardinality is now a governed Business Semantics Registry measure,
    declared on the canonical identifier with a ``count`` aggregation, so it
    goes through the same comparability rules as everything else.
    """
    envelope = ask("Compare the direct and acquired books on balance, loan "
                   "count, weighted-average LTV and average borrower age.")
    assert envelope["ok"] is True, envelope.get("error")
    text = answer(envelope)
    for measure in ("Current Outstanding Balance", "Loan Count",
                    "Current Loan To Value", "Youngest Borrower Age"):
        assert measure in text
    assert "Not compared" not in text


def test_the_compared_loan_count_reconciles_to_the_book(ask, governed_env):
    """The figure, against pandas — and rendered as a whole number, because a
    cardinality with a decimal place reads as an estimate."""
    from mi_agent_api.data_source import get_dataframe

    book = get_dataframe()
    expected = {"alp_origination": int((book["source_portfolio_id"]
                                        == "alp_origination").sum()),
                "alp_acquired": int((book["source_portfolio_id"]
                                     == "alp_acquired").sum())}
    envelope = ask("Compare the direct and acquired books on balance, loan "
                   "count, weighted-average LTV and average borrower age.")
    row = next(r for artifact in envelope["artifacts"]
               if artifact.get("type") == "table"
               for r in (artifact.get("rows") or [])
               if r.get("metric") == "Loan Count")
    assert row["a"] == f"{expected['alp_origination']:,}"
    assert row["b"] == f"{expected['alp_acquired']:,}"
    assert row["difference"] == (
        f"{expected['alp_origination'] - expected['alp_acquired']:,}")


def test_the_loan_count_measure_is_declared_neutral(governed_env):
    """More loans is neither better nor worse — it is scale. The registry says
    so explicitly, which is what stops a comparison reading a direction into a
    difference in book size."""
    from mi_workflows import semantics as bsr_module

    entry = bsr_module.load_business_semantics().get("loan_count")
    assert entry is not None, "loan cardinality is not governed by the BSR"
    assert entry.default_aggregation == "count"
    assert entry.directionality == "neutral"
    assert entry.analytical_role == "measure"


def test_the_loan_count_is_derived_so_no_identifier_enters_the_registry():
    """The first attempt keyed the count on ``loan_identifier`` and counted
    there. An identifier must never enter the Business Semantics Registry — it
    is not an analytical field, and admitting one invites it to be ranked,
    summed or published as evidence. Cardinality is a property of the
    POPULATION, and declaring it as one is what keeps the identifier out."""
    from mi_workflows import semantics as bsr_module

    registry = bsr_module.load_business_semantics()
    assert registry.get("loan_identifier") is None
    assert registry.get("loan_count").derived is True


def test_the_count_is_scoped_to_comparison_not_to_period_change(governed_env):
    """Period-change over an identifier is not a concept, and naming the
    identifier field in a period-change audit trips the no-loan-level-data
    guard for no analytical gain. The curation is scoped to the workflow that
    actually compares it."""
    from mi_workflows import semantics as bsr_module

    entry = bsr_module.load_business_semantics().get("loan_count")
    assert "portfolio_comparison" in entry.workflow_tags
    assert "period_change" not in entry.workflow_tags


# =========================================================================== #
# 10. Parser consistency — the two paths must not disagree about one sentence
# =========================================================================== #
def test_a_measure_set_the_model_returned_as_one_metric_is_carried(semantics,
                                                                   columns):
    """The original P1E defect on the LLM path.

    The model understood the question — it said so in its own explanation — and
    returned one metric because the contract had one slot. It has the array
    now, but if it still answers with a single metric the two parsers disagree
    about the same sentence, and the LLM path refuses a question the
    deterministic path answers.
    """
    from mi_agent.llm_query_parser import (_deterministic_parse,
                                           carry_specialist_intent)

    question = ("give me the current funded balance, loan count, "
                "weighted-average ltv and weighted-average interest rate.")
    deterministic, _ = _deterministic_parse(question, semantics, columns)
    model = MIQuerySpec.from_dict({"intent": "summary", "metric": BALANCE,
                                   "aggregation": "sum"})
    carried = carry_specialist_intent(model, deterministic)
    assert "measures" in carried
    assert [m["field"] for m in model.measures] == [
        m["field"] for m in deterministic.measures]


def test_carrying_a_measure_set_can_never_re_point_the_question(semantics,
                                                                columns):
    """Narrow by design: if the model named a measure the deterministic set does
    not contain, the model read the question differently and its reading
    stands. Overwriting it would answer a question neither parser was asked."""
    from mi_agent.llm_query_parser import (_deterministic_parse,
                                           carry_specialist_intent)

    deterministic, _ = _deterministic_parse(
        "give me balance, loan count and weighted-average ltv.",
        semantics, columns)
    model = MIQuerySpec.from_dict({"intent": "summary", "metric": VALUATION,
                                   "aggregation": "avg"})
    assert "measures" not in carry_specialist_intent(model, deterministic)
    assert [m["field"] for m in model.measures] == [VALUATION]


def test_a_set_the_model_expressed_itself_is_left_alone(semantics, columns):
    """It may only FILL a gap, never overrule a model that did the job."""
    from mi_agent.llm_query_parser import (_deterministic_parse,
                                           carry_specialist_intent)

    deterministic, _ = _deterministic_parse(
        "give me balance, loan count and weighted-average ltv.",
        semantics, columns)
    model = MIQuerySpec.from_dict({"measures": [
        {"field": BALANCE, "aggregation": "sum"},
        {"field": LTV, "aggregation": "weighted_avg"}]})
    assert "measures" not in carry_specialist_intent(model, deterministic)
    assert len(model.measures) == 2


# =========================================================================== #
# 11. A scope word is not a place
# =========================================================================== #
@pytest.mark.parametrize("question,expected_population", [
    # Sourcing cohort — the portfolio lens already resolves this one.
    ("For the acquired book, what are balance, loan count and "
     "weighted-average LTV?", 3909),
    # Funding state. This one was refused for an EMPTY population, because
    # "the funded book" resolved to a geography called "Funded" which matches
    # no row. It pre-dates P1E and broke the landing-page demo pack build.
    ("How many loans are in the funded book?", 11035),
])
def test_a_scope_word_is_never_resolved_as_a_geography(ask, question,
                                                       expected_population):
    """A place filter is only ever a place. Inventing a geography value the
    column does not contain empties the population and refuses a good
    question — the failure mode is a refusal rather than a wrong number, but
    it is still an outage."""
    envelope = ask(question)
    assert envelope["ok"] is True, envelope.get("error")
    assert f"{expected_population:,} loans" in receipt(envelope)


def test_the_non_place_guard_lists_scope_words_not_real_regions(governed_env):
    """Scope control: the guard must not shadow an actual region in the book."""
    from mi_agent.llm_query_parser import _NON_PLACE_TERMS
    from mi_agent_api.data_source import get_dataframe

    regions = {str(r).lower() for r in
               get_dataframe()["collateral_geography"].dropna().unique()}
    shadowed = {w for r in regions for w in r.split() if w in _NON_PLACE_TERMS}
    assert not shadowed, f"the guard shadows real region words: {shadowed}"


# =========================================================================== #
# 12. A proportion is not an amount
# =========================================================================== #
def test_a_share_question_answered_with_absolute_figures_refuses(ask):
    """Found by the genuine-LLM acceptance pass, not by reading the code.

    "What proportion of the book is eligible for a 75% LTV securitisation?"
    came back as ``Balance: £1.96bn · Loans: 11,007`` — two absolute figures
    and no proportion anywhere in it. P0 carried facets for scope, thresholds,
    ranking, grouping, contribution and measure sets, but none for a SHARE, so
    the lost intent was invisible and the guard returned ``ok``.

    A share needs two populations: the filtered numerator and the whole-book
    denominator. The numerator alone answers a different question, and leaves
    the reader to do arithmetic against a denominator the answer never states.
    """
    envelope = ask("What proportion of the book is eligible for a 75% LTV "
                   "securitisation?")
    assert envelope["ok"] is False
    assert "proportion" in answer(envelope).lower()


def test_a_governed_share_still_answers_and_states_its_denominator(ask):
    """Scope control. The guard must catch the LOST share, not share questions.

    The receipt names the basis and both populations, so the proportion is
    auditable from the answer alone rather than implied.
    """
    envelope = ask("What proportion of the book is above 60% LTV?")
    assert envelope["ok"] is True, envelope.get("error")
    text = receipt(envelope)
    assert "Share of" in text
    assert "of 11,035" in text, "the whole-book denominator is not stated"


def test_a_routed_capability_may_satisfy_a_share_by_stating_one(ask):
    """The concentration route answers "how much of the book is concentrated in
    the top 10 postcodes" with "£83.4m (4.2% of the book)" — a proportion,
    stated in its own terms, from a capability that never builds a spec.

    Accepted on EVIDENCE: the route's own answer must actually contain a
    percentage. Accepting the route by name alone would let any listing answer
    silently discharge a share request.
    """
    envelope = ask("How much of the book is concentrated in the top 10 "
                   "postcodes?")
    assert envelope["ok"] is True, envelope.get("error")
    assert "%" in answer(envelope)


def test_the_share_facet_refuses_rather_than_disclosing():
    """A missing proportion is a missing NUMBER, so it fails closed."""
    assert er.KIND_SHARE in er.NUMBER_OR_SUBJECT_FACETS
    assert er.KIND_SHARE not in er.SHAPE_FACETS


def test_the_share_detector_is_the_parsers_own(governed_env):
    """One detector, so the guard and the parser can never disagree about which
    questions are share questions."""
    from mi_agent.llm_query_parser import _SHARE_RE

    for question in ("what proportion of the book is above 60% ltv?",
                     "what share of the book is in arrears?",
                     "how much of the book is concentrated?",
                     "what percentage of loans are over 85?"):
        assert er._asks_for_a_share(question)
        assert _SHARE_RE.search(question)
    for question in ("what is the total balance?",
                     "show me balance by region",
                     "give me balance, loan count and weighted-average ltv."):
        assert not er._asks_for_a_share(question)
