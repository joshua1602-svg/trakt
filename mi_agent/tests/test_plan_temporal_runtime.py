#!/usr/bin/env python3
"""Slice 2 offline tests: the perimeter, the resolution, and the arithmetic.

No live model calls. The plans are compiled from CandidateIntent payloads by the
REAL deterministic compiler, so what is tested is the production compiler's
output rather than a hand-written plan the product would never emit.

The figures are checked against `portfolio_truth_oracle`, which imports nothing
from the product.
"""
from __future__ import annotations

from pathlib import Path

import pytest

from mi_agent import plan_runtime_adapter as adapter
from mi_agent import plan_temporal_runtime as temporal
from mi_agent.interpretation_v2.compiler import DeterministicCompiler
from mi_agent.interpretation_v2.intent import parse_candidate_intent
from mi_agent.mi_query_validator import load_mi_semantics
from mi_agent.states.selectors import SnapshotSelector
from mi_agent.tests import portfolio_truth_oracle as truth
from mi_agent.tests import temporal_snapshot_fixture as fixture

_REPO_ROOT = Path(__file__).resolve().parents[2]
_REGISTRY = _REPO_ROOT / "mi_agent" / "mi_semantics_field_registry.yaml"

BASE_INTENT = {
    "schema_version": "candidate_intent/1.0",
    "capability": "generic_analysis",
    "population": {"base": "funded", "lens": "all", "seasoning": "any"},
    "measures": [{"concept": "current_outstanding_balance", "statistic": "sum"}],
    "dimensions": [], "filters": [],
    "geography": {"requested": False},
    "comparison": {"kind": "none"},
}


def intent(**overrides):
    payload = dict(BASE_INTENT)
    payload.update(overrides)
    return payload


@pytest.fixture(scope="module")
def semantics():
    return load_mi_semantics(str(_REGISTRY))


@pytest.fixture(scope="module")
def history():
    return fixture.default_history()


@pytest.fixture(scope="module")
def store(tmp_path_factory, history):
    return fixture.build_store(tmp_path_factory.mktemp("snaps"), history)


@pytest.fixture(scope="module")
def compiler():
    return DeterministicCompiler()


def plan_for(compiler, payload):
    result = compiler.compile(parse_candidate_intent(payload))
    assert result.is_plan, f"expected a plan, got {result.outcome} {result.codes()}"
    return result.plan.to_dict()


def run(compiler, store, semantics, payload):
    return temporal.execute_temporal_plan(
        plan_for(compiler, payload), store=store, client_id=fixture.CLIENT_ID,
        semantics=semantics, route=fixture.ROUTE)


# --------------------------------------------------------------------------- #
# the perimeter
# --------------------------------------------------------------------------- #

def test_a_current_period_plan_belongs_to_slice_one(compiler):
    """The two perimeters are disjoint, so no plan can be claimed by both."""
    plan = plan_for(compiler, intent(operation="point_in_time",
                                     time={"form": "current"}))
    assert adapter.check_eligibility(plan)[0] is True
    eligible, reason, _ = temporal.check_temporal_eligibility(plan)
    assert (eligible, reason) == (False, temporal.PERIOD_NOT_TEMPORAL)


def test_a_temporal_plan_is_still_refused_by_slice_one(compiler):
    """Slice 1's answer to a historical period is unchanged by slice 2."""
    series = plan_for(compiler, intent(
        operation="series",
        time={"form": "series", "grain": "monthly", "periods_back": 6}))
    assert adapter.check_eligibility(series)[:2] == (
        False, adapter.OPERATION_NOT_GENERIC)
    historical = plan_for(compiler, intent(
        operation="point_in_time",
        time={"form": "explicit_period", "labels": ["April"]}))
    assert adapter.check_eligibility(historical)[:2] == (
        False, adapter.PERIOD_NOT_CURRENT)


@pytest.mark.parametrize("payload,expected", [
    # a specialist capability owns its own arithmetic
    (dict(capability="borrowing_base", operation="series",
          measures=[{"concept": "borrowing_base"}],
          time={"form": "series", "grain": "monthly", "periods_back": 6}),
     adapter.CAPABILITY_NOT_GENERIC),
    # a movement is an attribution, not an evaluation per snapshot
    (dict(operation="movement",
          time={"form": "relative_pair", "grain": "monthly", "periods_back": 1}),
     temporal.OPERATION_NOT_TEMPORAL),
    # a series over a period PAIR states two windows at once
    (dict(operation="series",
          time={"form": "relative_pair", "grain": "monthly", "periods_back": 1}),
     temporal.OPERATION_PERIOD_MISMATCH),
    # a comparison over an open span states no pair to compare
    (dict(operation="compare",
          time={"form": "series", "grain": "monthly", "periods_back": 6}),
     temporal.OPERATION_PERIOD_MISMATCH),
    # three axes
    (dict(operation="series",
          dimensions=["ltv_bucket", "erm_product_type", "broker_channel"],
          time={"form": "series", "grain": "monthly", "periods_back": 6}),
     adapter.TOO_MANY_DIMENSIONS),
    # several measures in one request
    (dict(operation="series",
          measures=[{"concept": "current_outstanding_balance", "statistic": "sum"},
                    {"concept": "loan", "statistic": "count"}],
          time={"form": "series", "grain": "monthly", "periods_back": 6}),
     adapter.NOT_SINGLE_OUTPUT),
    # an explicit Direct/Acquired lens is another owner's
    (dict(operation="series",
          population={"base": "funded", "lens": "acquired", "seasoning": "any"},
          time={"form": "series", "grain": "monthly", "periods_back": 6}),
     adapter.EXPLICIT_LENS),
    # a governed geography axis is the geography owner's
    (dict(operation="series",
          geography={"requested": True, "basis": "collateral",
                     "level": "reporting", "group_by": True},
          time={"form": "series", "grain": "monthly", "periods_back": 3}),
     adapter.GEOGRAPHY_REQUESTED),
])
def test_the_perimeter_refuses_what_it_cannot_carry(compiler, payload, expected):
    eligible, reason, _ = temporal.check_temporal_eligibility(
        plan_for(compiler, intent(**payload)))
    assert (eligible, reason) == (False, expected)


def test_a_pipeline_population_is_not_a_funded_snapshot(compiler):
    """Slice 2 selects funded snapshots; another base is not narrowed into one."""
    plan = plan_for(compiler, intent(
        operation="series",
        population={"base": "pipeline", "lens": "all", "seasoning": "any"},
        measures=[{"concept": "loan", "statistic": "count"}],
        time={"form": "series", "grain": "monthly", "periods_back": 6}))
    assert temporal.check_temporal_eligibility(plan)[:2] == (
        False, temporal.POPULATION_NOT_FUNDED)


# --------------------------------------------------------------------------- #
# resolution
# --------------------------------------------------------------------------- #

def test_last_n_selects_the_latest_n_and_no_others(compiler, store, history):
    plan = plan_for(compiler, intent(
        operation="series",
        time={"form": "series", "grain": "monthly", "periods_back": 6,
              "labels": ["the last six months"]}))
    resolution = temporal.resolve_temporal(plan, store,
                                           client_id=fixture.CLIENT_ID,
                                           route=fixture.ROUTE)
    assert resolution.ok and resolution.shape == temporal.SHAPE_SERIES
    assert resolution.basis == "count"
    assert resolution.selector.mode == "last_n"
    assert list(resolution.reporting_dates) == [d for d, _ in history[-6:]]


def test_a_whole_series_label_takes_every_governed_period(compiler, store,
                                                          history):
    plan = plan_for(compiler, intent(
        operation="series",
        time={"form": "series", "grain": "monthly", "labels": ["each month"]}))
    resolution = temporal.resolve_temporal(plan, store,
                                           client_id=fixture.CLIENT_ID,
                                           route=fixture.ROUTE)
    assert resolution.basis == "whole_series"
    assert list(resolution.reporting_dates) == [d for d, _ in history]


def test_since_an_anchor_runs_from_that_period_to_the_latest(compiler, store,
                                                            history):
    plan = plan_for(compiler, intent(
        operation="series",
        time={"form": "range", "grain": "monthly", "labels": ["since March"]}))
    resolution = temporal.resolve_temporal(plan, store,
                                           client_id=fixture.CLIENT_ID,
                                           route=fixture.ROUTE)
    assert resolution.basis == "anchor"
    assert list(resolution.reporting_dates) == [
        d for d, _ in history if d >= "2026-03-31"]


def test_an_explicit_period_is_one_snapshot(compiler, store):
    plan = plan_for(compiler, intent(
        operation="point_in_time",
        time={"form": "explicit_period", "grain": "monthly",
              "labels": ["April"]}))
    resolution = temporal.resolve_temporal(plan, store,
                                           client_id=fixture.CLIENT_ID,
                                           route=fixture.ROUTE)
    assert resolution.shape == temporal.SHAPE_POINT
    assert list(resolution.reporting_dates) == ["2026-04-30"]


def test_the_previous_reporting_period_is_one_period_not_a_pair(compiler, store,
                                                                history):
    """The singular form names one period; only `compare` puts two side by side."""
    point = temporal.resolve_temporal(
        plan_for(compiler, intent(operation="point_in_time",
                                  time={"form": "previous_reporting_period",
                                        "grain": "monthly",
                                        "labels": ["last month"]})),
        store, client_id=fixture.CLIENT_ID, route=fixture.ROUTE)
    assert point.shape == temporal.SHAPE_POINT
    assert list(point.reporting_dates) == [history[-2][0]]

    pair = temporal.resolve_temporal(
        plan_for(compiler, intent(operation="compare",
                                  time={"form": "previous_reporting_period",
                                        "grain": "monthly",
                                        "labels": ["last month"]})),
        store, client_id=fixture.CLIENT_ID, route=fixture.ROUTE)
    assert pair.shape == temporal.SHAPE_COMPARISON
    assert list(pair.reporting_dates) == [history[-2][0], history[-1][0]]


def test_a_relative_pair_two_back_reaches_two_back(compiler, store, history):
    resolution = temporal.resolve_temporal(
        plan_for(compiler, intent(operation="compare",
                                  time={"form": "relative_pair",
                                        "grain": "monthly", "periods_back": 2,
                                        "labels": ["two months ago"]})),
        store, client_id=fixture.CLIENT_ID, route=fixture.ROUTE)
    assert list(resolution.reporting_dates) == [history[-3][0], history[-1][0]]


def test_every_date_on_a_selector_came_from_the_catalogue(compiler, store,
                                                          history):
    """The model authors no binding, and neither does this module.

    A selector may only ever carry a reporting date the catalogue itself
    returned. A computed one — "three months before today" — would be a date the
    book may not have, which is how a fabricated snapshot gets requested.
    """
    known = {d for d, _ in history}
    payloads = [
        intent(operation="series",
               time={"form": "range", "grain": "monthly",
                     "labels": ["since March"]}),
        intent(operation="point_in_time",
               time={"form": "explicit_period", "grain": "monthly",
                     "labels": ["April"]}),
        intent(operation="series",
               time={"form": "series", "grain": "monthly", "periods_back": 4}),
        intent(operation="compare",
               time={"form": "relative_pair", "grain": "monthly",
                     "periods_back": 1}),
    ]
    for payload in payloads:
        resolution = temporal.resolve_temporal(
            plan_for(compiler, payload), store, client_id=fixture.CLIENT_ID,
            route=fixture.ROUTE)
        assert resolution.ok, resolution.detail
        selector = resolution.selector
        for value in (selector.reporting_date, selector.start_date,
                      selector.end_date, selector.baseline_date,
                      selector.current_date):
            assert value is None or str(value) in known, value


# --------------------------------------------------------------------------- #
# negative and safety controls
# --------------------------------------------------------------------------- #

def test_a_period_the_book_does_not_reach_is_not_shortened(compiler, store):
    outcome = run(compiler, store, None, intent(
        operation="series",
        time={"form": "series", "grain": "monthly", "periods_back": 12,
              "labels": ["the last twelve months"]}))
    assert outcome.reason == temporal.PERIOD_NOT_AVAILABLE
    assert outcome.clarifiable
    assert outcome.points == ()


def test_a_cadence_the_catalogue_does_not_carry_is_refused(compiler, store):
    outcome = run(compiler, store, None, intent(
        operation="series",
        time={"form": "series", "grain": "weekly", "periods_back": 4,
              "labels": ["the last four weeks"]}))
    assert outcome.reason == temporal.UNSUPPORTED_CADENCE


def test_a_catalogue_that_declares_no_cadence_cannot_honour_a_grain(
        compiler, tmp_path, history):
    silent = fixture.build_store(tmp_path / "nocadence", history, cadence=None)
    outcome = temporal.execute_temporal_plan(
        plan_for(compiler, intent(operation="series",
                                  time={"form": "series", "grain": "monthly",
                                        "periods_back": 3})),
        store=silent, client_id=fixture.CLIENT_ID, semantics=None,
        route=fixture.ROUTE)
    assert outcome.reason == temporal.UNSUPPORTED_CADENCE


def test_a_bare_cadence_is_a_stated_span(compiler, store, history):
    """`{form: series, grain: monthly, labels: [], periods_back: null}`.

    The exact shape the live run measured Opus emitting for all four "each
    month" questions. The compiler accepts it and emits a plan; this asserts the
    resolver now agrees with the compiler instead of refusing what it authorised.
    """
    plan = plan_for(compiler, intent(
        operation="series",
        time={"form": "series", "grain": "monthly", "labels": [],
              "periods_back": None}))
    resolution = temporal.resolve_temporal(plan, store,
                                           client_id=fixture.CLIENT_ID,
                                           route=fixture.ROUTE)
    assert resolution.ok, resolution.detail
    assert resolution.basis == "cadence"
    assert list(resolution.reporting_dates) == [d for d, _ in history]


def test_a_bare_cadence_the_catalogue_does_not_keep_still_fails_closed(
        compiler, tmp_path, history):
    """The branch reads the CATALOGUE's cadence, not the plan's word for it.

    A weekly series against a monthly book must not come back monthly. The
    cadence guard catches it first; this asserts the new branch cannot rescue it
    afterwards.
    """
    monthly = fixture.build_store(tmp_path / "monthly", history)
    outcome = temporal.execute_temporal_plan(
        plan_for(compiler, intent(operation="series",
                                  time={"form": "series", "grain": "weekly",
                                        "labels": [], "periods_back": None})),
        store=monthly, client_id=fixture.CLIENT_ID, semantics=None,
        route=fixture.ROUTE)
    assert outcome.reason == temporal.UNSUPPORTED_CADENCE


def test_a_catalogue_declaring_no_cadence_does_not_get_the_bare_span(
        compiler, tmp_path, history):
    """A book that records no rhythm cannot have one read back out of it."""
    silent = fixture.build_store(tmp_path / "silent", history, cadence=None)
    resolution = temporal.resolve_temporal(
        plan_for(compiler, intent(operation="series",
                                  time={"form": "series", "grain": "monthly",
                                        "labels": [], "periods_back": None})),
        silent, client_id=fixture.CLIENT_ID, route=fixture.ROUTE)
    assert not resolution.ok
    assert resolution.reason == temporal.UNSUPPORTED_CADENCE


def test_a_bare_range_is_still_an_incomplete_request(compiler, store):
    """`range` states bounds. One with neither bound and no label is not a span.

    Only `series` carries the "every period at this rhythm" reading; widening
    `range` the same way would answer an incomplete request instead of asking
    about it.
    """
    resolution = temporal.resolve_temporal(
        plan_for(compiler, intent(operation="series",
                                  time={"form": "range", "grain": "monthly",
                                        "labels": [], "periods_back": None})),
        store, client_id=fixture.CLIENT_ID, route=fixture.ROUTE)
    assert not resolution.ok
    assert resolution.reason == temporal.PERIOD_LABEL_UNRESOLVED


def test_an_unreadable_label_still_clarifies_even_with_a_grain(compiler, store):
    """The bare-cadence branch requires NO label, not merely no usable one.

    "the last few months" names a narrower window than the whole series. Reading
    the grain instead would answer a question the reader did not ask, which is
    the substitution the whole contract exists to stop.
    """
    resolution = temporal.resolve_temporal(
        plan_for(compiler, intent(
            operation="series",
            time={"form": "series", "grain": "monthly",
                  "labels": ["the last few months"]})),
        store, client_id=fixture.CLIENT_ID, route=fixture.ROUTE)
    assert not resolution.ok
    assert resolution.reason == temporal.PERIOD_LABEL_UNRESOLVED


def test_a_vague_recency_clarifies_rather_than_choosing_a_window(compiler,
                                                                 store):
    outcome = run(compiler, store, None, intent(
        operation="series",
        time={"form": "series", "grain": "monthly",
              "labels": ["the last few months"]}))
    assert outcome.reason == temporal.PERIOD_LABEL_UNRESOLVED
    assert outcome.clarifiable


def test_a_month_the_book_carries_twice_is_ambiguous(compiler, tmp_path):
    long_book = fixture.build_store(tmp_path / "long",
                                    fixture.repeated_month_history())
    outcome = temporal.execute_temporal_plan(
        plan_for(compiler, intent(operation="series",
                                  time={"form": "range", "grain": "monthly",
                                        "labels": ["since March"]})),
        store=long_book, client_id=fixture.CLIENT_ID, semantics=None,
        route=fixture.ROUTE)
    assert outcome.reason == temporal.PERIOD_LABEL_AMBIGUOUS


def test_a_month_the_book_does_not_carry_is_not_substituted(compiler, store):
    outcome = run(compiler, store, None, intent(
        operation="point_in_time",
        time={"form": "explicit_period", "grain": "monthly",
              "labels": ["August"]}))
    assert outcome.reason == temporal.PERIOD_NOT_AVAILABLE
    assert outcome.points == ()


def test_an_empty_catalogue_answers_nothing(compiler, tmp_path):
    empty = fixture.build_store(tmp_path / "empty", [])
    outcome = temporal.execute_temporal_plan(
        plan_for(compiler, intent(operation="series",
                                  time={"form": "series", "grain": "monthly",
                                        "periods_back": 3})),
        store=empty, client_id=fixture.CLIENT_ID, semantics=None,
        route=fixture.ROUTE)
    assert outcome.reason == temporal.NO_SNAPSHOTS


def test_a_year_alone_is_not_a_period_anchor():
    """"2025" names twelve reporting periods; picking one would be a guess."""
    assert temporal.parse_anchor("2025") is None
    assert temporal.parse_anchor("since 2025") is None
    assert temporal.parse_anchor("March 2026").year == 2026


# --------------------------------------------------------------------------- #
# execution against the independent oracle
# --------------------------------------------------------------------------- #

def test_a_series_reconciles_period_by_period(compiler, store, semantics,
                                              history):
    outcome = run(compiler, store, semantics, intent(
        operation="series",
        time={"form": "series", "grain": "monthly", "periods_back": 6,
              "labels": ["the last six months"]}))
    assert outcome.executed and outcome.reconciled
    expected = [(d, truth.total(f, truth.BALANCE)) for d, f in history[-6:]]
    assert [p.reporting_date for p in outcome.points] == [d for d, _ in expected]
    for point, (_, figure) in zip(outcome.points, expected):
        assert point.value == pytest.approx(figure, abs=0.01)


def test_a_loan_count_series_reconciles(compiler, store, semantics, history):
    outcome = run(compiler, store, semantics, intent(
        operation="series", measures=[{"concept": "loan", "statistic": "count"}],
        time={"form": "series", "grain": "monthly", "labels": ["each month"]}))
    assert outcome.executed
    assert [p.value for p in outcome.points] == [
        float(truth.row_count(f)) for _, f in history]


def test_a_filter_is_applied_on_every_snapshot_and_the_receipt_says_so(
        compiler, store, semantics, history):
    outcome = run(compiler, store, semantics, intent(
        operation="series",
        measures=[{"concept": "current_loan_to_value",
                   "statistic": "weighted_average",
                   "weight": "current_outstanding_balance"}],
        filters=[{"concept": "erm_product_type", "comparator": "eq",
                  "value": "drawdown"}],
        time={"form": "series", "grain": "monthly", "periods_back": 3,
              "labels": ["the last three months"]}))
    assert outcome.executed and outcome.reconciled
    predicates = [("erm_product_type", "eq", "Drawdown")]
    expected = [truth.weighted_average(f, truth.LTV, truth.BALANCE, predicates)
                for _, f in history[-3:]]
    assert [p.value for p in outcome.points] == pytest.approx(expected)
    for point in outcome.points:
        fields = {entry["field"] for entry in point.receipt["applied_predicates"]}
        assert "erm_product_type" in fields


def test_a_grouped_series_reconciles_cell_by_cell(compiler, store, semantics,
                                                  history):
    outcome = run(compiler, store, semantics, intent(
        operation="series", dimensions=["ltv_bucket"],
        time={"form": "series", "grain": "monthly", "periods_back": 3,
              "labels": ["the last three months"]}))
    assert outcome.executed
    for point, (_, frame) in zip(outcome.points, history[-3:]):
        expected = truth.grouped(frame, ["ltv_bucket"], column=truth.BALANCE,
                                 how="sum")
        produced = {(cell["ltv_bucket"],): cell["value"] for cell in point.cells}
        assert set(produced) == set(expected)
        for key, figure in expected.items():
            assert produced[key] == pytest.approx(figure, abs=0.01)


def test_a_two_dimension_grid_reconciles_cell_by_cell(compiler, store,
                                                      semantics, history):
    outcome = run(compiler, store, semantics, intent(
        operation="series", dimensions=["ltv_bucket", "erm_product_type"],
        measures=[{"concept": "loan", "statistic": "count"}],
        time={"form": "series", "grain": "monthly", "periods_back": 2,
              "labels": ["the last two months"]}))
    assert outcome.executed
    for point, (_, frame) in zip(outcome.points, history[-2:]):
        expected = truth.grouped(frame, ["ltv_bucket", "erm_product_type"],
                                 how="count")
        produced = {(cell["ltv_bucket"], cell["erm_product_type"]): cell["value"]
                    for cell in point.cells}
        assert produced == pytest.approx(expected)


def test_a_period_comparison_computes_its_own_change(compiler, store, semantics,
                                                     history):
    outcome = run(compiler, store, semantics, intent(
        operation="compare",
        time={"form": "relative_pair", "grain": "monthly", "periods_back": 1,
              "labels": ["this month versus last month"]}))
    assert outcome.executed and outcome.shape == temporal.SHAPE_COMPARISON
    baseline = truth.total(history[-2][1], truth.BALANCE)
    current = truth.total(history[-1][1], truth.BALANCE)
    assert outcome.comparison["baseline_value"] == pytest.approx(baseline,
                                                                abs=0.01)
    assert outcome.comparison["current_value"] == pytest.approx(current,
                                                               abs=0.01)
    assert outcome.comparison["absolute_change"] == pytest.approx(
        current - baseline, abs=0.01)
    assert outcome.comparison["percent_change"] == pytest.approx(
        (current - baseline) / baseline * 100.0)


def test_an_explicit_period_answers_from_that_period_only(compiler, store,
                                                          semantics, history):
    outcome = run(compiler, store, semantics, intent(
        operation="point_in_time",
        time={"form": "explicit_period", "grain": "monthly",
              "labels": ["April"]}))
    assert outcome.executed and len(outcome.points) == 1
    april = dict(history)["2026-04-30"]
    assert outcome.points[0].value == pytest.approx(truth.total(april,
                                                                truth.BALANCE),
                                                    abs=0.01)


def test_a_population_empty_in_every_period_is_not_served(compiler, store,
                                                          semantics):
    outcome = run(compiler, store, semantics, intent(
        operation="series",
        filters=[{"concept": "youngest_borrower_age", "comparator": "gt",
                  "value": 200}],
        time={"form": "series", "grain": "monthly", "periods_back": 3,
              "labels": ["the last three months"]}))
    assert outcome.reason == temporal.EMPTY_ACROSS_EVERY_SNAPSHOT
    assert not outcome.reconciled


def test_one_spec_runs_against_every_snapshot(compiler, store, semantics):
    """The same authorised work on each period, bound once and not re-derived."""
    plan = plan_for(compiler, intent(
        operation="series", dimensions=["ltv_bucket"],
        filters=[{"concept": "current_loan_to_value", "comparator": "gt",
                  "value": 50}],
        time={"form": "series", "grain": "monthly", "periods_back": 3,
              "labels": ["the last three months"]}))
    outcome = temporal.execute_temporal_plan(
        plan, store=store, client_id=fixture.CLIENT_ID, semantics=semantics,
        route=fixture.ROUTE)
    assert outcome.executed
    assert outcome.spec.to_dict() == adapter.spec_for_plan(plan).to_dict()
    for point in outcome.points:
        assert point.receipt["group_field_keys"] == ["ltv_bucket"]
        assert {e["field"] for e in point.receipt["applied_predicates"]} == {
            "current_loan_to_value"}


# --------------------------------------------------------------------------- #
# governance properties of the module itself
# --------------------------------------------------------------------------- #

def test_the_module_cannot_read_a_question_or_a_date_column():
    """A structural property, asserted rather than promised.

    Read off the parsed module the way slice 1's own adapter test reads its
    one, so the assertion is about the NAMES the code reaches rather than about
    the prose explaining that it does not.

    No `question`/`sentence`/`parsed` name exists, so no function can take one.
    Nothing imports `re`, so nothing can pattern-match a sentence. No row-level
    date column and no frame reader appears, so no code path can simulate a
    missing snapshot by filtering the rows of a present one.

    `text` is NOT in the forbidden set here, and slice 1's adapter has it there
    for a reason that does not hold in this module: the adapter reads no string
    at all, while this one reads exactly one — the plan's own `period.labels`,
    which the compiler authored and which `intent._DATE_LIKE` already forbids
    from carrying a date or a snapshot id. Banning the word rather than the
    behaviour would be satisfied by a rename and would prove nothing.
    """
    import ast

    source = Path(temporal.__file__).read_text(encoding="utf-8")
    tree = ast.parse(source)
    names = {n.id for n in ast.walk(tree) if isinstance(n, ast.Name)}
    names |= {n.attr for n in ast.walk(tree) if isinstance(n, ast.Attribute)}
    names |= {a.arg for n in ast.walk(tree)
              if isinstance(n, ast.arguments) for a in n.args}
    names |= {a.arg for n in ast.walk(tree)
              if isinstance(n, ast.arguments) for a in n.kwonlyargs}
    for forbidden in ("question", "sentence", "parsed", "raw_question"):
        assert forbidden not in names, f"the temporal runtime reaches {forbidden!r}"

    imported = {n.module for n in ast.walk(tree)
                if isinstance(n, ast.ImportFrom) and n.module}
    imported |= {a.name for n in ast.walk(tree) if isinstance(n, ast.Import)
                 for a in n.names}
    assert "re" not in imported

    for forbidden in ("llm_query_parser", "parsed_question",
                      "question_interpretation", "opus_interpreter",
                      "chat_routing", "period_request", "recognition",
                      "reporting_date_column", "as_at_date",
                      "origination_date", "completion_date", "to_datetime",
                      "read_csv"):
        assert forbidden not in source, f"the temporal runtime reaches {forbidden}"


def test_the_selector_gained_one_mode_and_kept_the_others():
    for mode in ("latest", "as_of", "range", "compare", "last_n"):
        assert hasattr(SnapshotSelector, mode)


def test_last_n_never_returns_a_short_series(store):
    """Fewer periods than asked for is a refusal, not a shorter answer."""
    from snapshot.model import SnapshotNotFoundError
    assert len(store.resolve_last_n(fixture.CLIENT_ID, 3,
                                    route=fixture.ROUTE)) == 3
    with pytest.raises(SnapshotNotFoundError):
        store.resolve_last_n(fixture.CLIENT_ID, 99, route=fixture.ROUTE)
    with pytest.raises(SnapshotNotFoundError):
        store.resolve_last_n(fixture.CLIENT_ID, 0, route=fixture.ROUTE)
