"""tests/test_time_axis_vocabulary.py — root cause 1, and the three limits on it.

The widening itself is small. What needs pinning is what it must NOT do:

1. it must not make a SEASONING window or a DIMENSION into a time axis;
2. it must not let a question that names NO MEASURE become answerable — the line
   path defaults a missing metric to Total Balance, and inheriting that default
   is the substitution the no-silent-substitution rule prevents;
3. it must not narrow anything that produced a line before.

And one structural requirement: the axis question has ONE owner. Three readers
were asking it — `lexical.time_axis_request`, `llm_query_parser.is_line` and
`chat_routing._EVOLUTION_MARKERS`. With only two of the three widened, "balance
by period" became a line, missed the evolution route, and was answered by the
generic executor as 13 VINTAGE YEARS instead of 3 reporting periods. That is
asserted directly below, because it is the failure this work nearly shipped.
"""
from __future__ import annotations

import pytest

from question_interpretation.lexical import time_axis_request


# --------------------------------------------------------------------------- #
# 1. THE WIDENING — every term has a phrasing from the measured banks
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("question,evidence", [
    ("balance by period", "outstanding balances by period (widened T1)"),
    ("balance per period", "per-period lender voice"),
    ("balance over the periods", "how is the front book tracking over the periods"),
    ("balance by quarter", "grain the owner already held; carriage only"),
    ("balance each month", "give me balances per region each month (widened T3)"),
    ("balance every month", "sibling of each month"),
    ("balance between periods", "Which LTV band moved most between periods? (T7)"),
    ("balance month on month", "month-on-month change in balance by region (T6)"),
    ("balance month to month", "how is the loan book tracking month to month (T1)"),
    ("balance period on period", "period on period movement by LTV band (T6)"),
    ("balance year on year", "sibling of month on month"),
])
def test_a_lender_time_phrase_names_a_time_axis(question, evidence):
    assert time_axis_request(question) is not None, evidence


# --------------------------------------------------------------------------- #
# 2. WHAT IT MUST NOT SWALLOW — seasoning by name, and spans
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("question", [
    # SEASONING WINDOWS are resolved by their own reader and are not time axes.
    "balance for the front book",
    "balance for the back book",
    "how has the front book performed",
    # DIMENSIONS are groupings, not axes.
    "balance by region",
    "balance by LTV band",
    "balance by ticket size",
    # A SPAN names two dates, not a series.
    "balance between January and March",
    # A unit inside a FILTER is not an axis — the adjacency rule.
    "balance by region for loans under 12 months old",
])
def test_these_are_not_time_axes(question):
    assert time_axis_request(question) is None


# --------------------------------------------------------------------------- #
# 3. ONE OWNER — the three readers must agree
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("question", [
    "balance by period", "balance per period", "balance over the periods",
    "balance each month", "balance between periods", "balance by quarter",
])
def test_the_route_recogniser_reads_the_owner(question):
    """`_is_evolution` must accept every axis the owner names, given a line spec.

    Without this the question becomes a line, misses the evolution route and is
    answered by the generic executor over `vintage_year` — a cohort label, not a
    point on a time axis.
    """
    from mi_agent_api.chat_routing import _is_evolution

    class _Spec:
        chart_type = "line"
        x = "origination_date"

    assert _is_evolution(question, _Spec()) is True


def test_the_route_recogniser_still_refuses_a_vintage_axis():
    from mi_agent_api.chat_routing import _is_evolution

    class _Vintage:
        chart_type = "line"
        x = "vintage_year"

    assert _is_evolution("balance by period", _Vintage()) is False


def test_a_non_line_spec_is_never_an_evolution():
    from mi_agent_api.chat_routing import _is_evolution

    class _Bar:
        chart_type = "bar"
        x = None

    assert _is_evolution("balance by period", _Bar()) is False


# --------------------------------------------------------------------------- #
# 4. THE NO-MEASURE GUARD — root 2 is not touched by root 1
# --------------------------------------------------------------------------- #
@pytest.fixture(scope="module")
def semantics():
    from pathlib import Path
    from mi_agent.mi_query_validator import load_mi_semantics
    return load_mi_semantics(str(
        Path(__file__).resolve().parent.parent / "mi_agent"
        / "mi_semantics_field_registry.yaml"))


def _spec(question, semantics):
    from mi_agent import llm_query_parser as P
    spec, _meta = P.parse_with_repair(question, semantics, llm_enabled=False)
    return spec


def test_a_newly_carried_axis_does_not_answer_a_question_naming_no_measure(semantics):
    """"how is the loan book tracking month to month" names NO measure. The
    widened axis must not pull it into the line path, which would default the
    metric to Total Balance — a guess for a question that named none."""
    spec = _spec("how is the loan book tracking month to month", semantics)
    assert spec.chart_type != "line", \
        "a metric-less question must not become a line on the strength of the axis alone"


def test_the_same_question_with_a_measure_named_does_become_a_series(semantics):
    """The control: the axis carries as soon as a measure is present."""
    spec = _spec("how is the balance tracking month to month", semantics)
    assert spec.chart_type == "line"


def test_the_legacy_vocabulary_is_not_narrowed_by_the_guard(semantics):
    """"over time" produced a line before this change even with no resolvable
    metric, and must still. Narrowing it broke
    test_pipeline_by_stage_over_time_e2e."""
    spec = _spec("Show pipeline by stage over time.", semantics)
    assert spec.chart_type == "line"
