"""The funded book's remaining gaps from the 2026-09-29 bank, closed at their
owners rather than around them.

  "by region" (7 questions)   the plan's geography binding — ONE field the
                              compiler resolved through the governed contract —
                              is carried to the executor as an axis (or, for
                              "in London", a predicate), and the coverage owner
                              proves it ran. It used to be refused wholesale.
  median / largest / smallest the executor has always computed them; the plan's
                              statistic had no entry in the adapter's map.
  "borrower structure"        a legacy second name for single vs joint; the
                              registry already said "prefer borrower_type". The
                              model is now shown one concept per meaning.
  a field the book lacks      named as the book's (`FIELD_NOT_IN_BOOK`), from
                              the executor's own validator — never a crash, and
                              never answered from a neighbouring field.

Every figure here is compared with the same prepared frame the executor read.
"""
from __future__ import annotations

import glob

import pandas as pd
import pytest

from mi_agent import plan_runtime_adapter as adapter
from mi_agent.interpretation_v2.compiler import CompilerContext, DeterministicCompiler
from mi_agent.interpretation_v2.intent import parse_candidate_intent
from tests.interpretation_v2.test_specialist_runtime_pipeline import _Scripted

_BALANCE = [{"concept": "current_outstanding_balance"}]
_REGION = {"requested": True, "group_by": True, "level": "reporting"}


@pytest.fixture(scope="module")
def semantics():
    from mi_agent.mi_query_validator import load_mi_semantics
    from mi_agent_api.data_source import semantics_path
    return load_mi_semantics(semantics_path())


@pytest.fixture(scope="module")
def book():
    """The fixture's prepared funded book, as the funded route resolves it."""
    import os
    from pathlib import Path
    from mi_agent_api import datasets as ds
    root = Path(__file__).resolve().parents[2]
    old = os.getcwd()
    os.chdir(root)
    try:
        frame, err = ds._resolve_query_frame("funded", "client_001/mi_2025_11")
    finally:
        os.chdir(old)
    assert err is None and frame is not None
    return frame


@pytest.fixture(scope="module")
def harmonised(book):
    """The same book with its reporting region stamped by the funded book's
    own preparation step — what the platform path does for every production
    book, choosing the source column from the book's geography basis (this
    book's obligor column holds ITL3 codes; its collateral names resolve)."""
    from mi_agent_api import funded_prep
    frame = book.copy()
    assert funded_prep._apply_region_taxonomy(frame).get("applied")
    assert frame["canonical_region_reporting"].notna().any()
    return frame


def _intent(**over):
    payload = {"schema_version": "candidate_intent/1.0",
               "capability": "generic_analysis", "operation": "breakdown",
               "population": {"base": "funded"}, "measures": _BALANCE,
               "time": {"form": "current"}}
    payload.update(over)
    return payload


def _plan(**over):
    result = DeterministicCompiler(CompilerContext()).compile(
        parse_candidate_intent(_intent(**over)))
    assert result.plan is not None, [(r.code, r.detail) for r in result.reasons]
    return result.plan.to_dict()


def _served(intent, monkeypatch, frame, semantics):
    from mi_agent import plan_serving_canary as canary
    from mi_agent import plan_shadow_evidence as evidence
    from mi_agent import plan_shadow_wiring as wiring
    from mi_agent_api.mi_service import _governed_plan_coverage

    monkeypatch.setenv("MI_AGENT_PLAN_SERVE", "canary")
    monkeypatch.setenv("MI_AGENT_PLAN_SERVE_PRINCIPALS", "canary-principal")
    wiring.set_interpreter_factory(lambda: _Scripted(intent))
    written = []
    monkeypatch.setattr(evidence, "write", lambda body: written.append(body))

    class Principal:
        actor_id = "canary-principal"

    try:
        payload = canary.serve(
            question="q", context=Principal(), client_id="client_001",
            run_id="mi_2025_11", legacy_result={"ok": True}, frame=frame,
            semantics=semantics, view="funded",
            portfolio_id="client_001/mi_2025_11",
            render_portfolio_id="client_001/mi_2025_11", as_of=None)
    finally:
        wiring.set_interpreter_factory(None)
    record = written[-1] if written else {}
    coverage = _governed_plan_coverage(payload) if payload else None
    return payload, record, coverage


# --------------------------------------------------------------------------- #
# by region
# --------------------------------------------------------------------------- #

def test_balance_by_region_is_the_books_reporting_regions(monkeypatch, harmonised,
                                                         semantics):
    payload, record, coverage = _served(_intent(geography=_REGION), monkeypatch,
                                        harmonised, semantics)
    assert payload is not None, record.get("execution")
    executed = payload["metadata"]["governedPlan"]["executed"]
    assert "canonical_region_reporting" in executed["group_field_keys"]
    assert coverage["unaccounted"] == []
    assert any(e["kind"] == "governed_plan:geography" for e in coverage["concepts"])
    cells = {c["canonical_region_reporting"]: c["value"]
             for c in record["execution"]["grouped_cells"]}
    expected = harmonised.groupby("canonical_region_reporting")[
        "current_outstanding_balance"].sum()
    assert cells == pytest.approx({k: float(v) for k, v in expected.items()})


@pytest.mark.parametrize("measure", [
    [{"concept": "loan", "statistic": "count"}],
    [{"concept": "current_loan_to_value", "statistic": "weighted_average"}]])
def test_other_measures_by_region_are_served_too(measure, monkeypatch, harmonised,
                                                 semantics):
    payload, record, coverage = _served(
        _intent(geography=_REGION, measures=measure), monkeypatch, harmonised,
        semantics)
    assert payload is not None, record.get("execution")
    assert coverage["unaccounted"] == []


def test_a_region_restriction_is_a_proved_predicate(monkeypatch, harmonised,
                                                   semantics):
    region = str(harmonised["canonical_region_reporting"].dropna().iloc[0])
    intent = _intent(operation="point_in_time",
                     geography={"requested": True, "group_by": False,
                                "level": "reporting", "values": [region]})
    payload, record, coverage = _served(intent, monkeypatch, harmonised, semantics)
    assert payload is not None, record.get("execution")
    assert coverage["unaccounted"] == []
    expected = harmonised.loc[harmonised["canonical_region_reporting"] == region,
                              "current_outstanding_balance"].sum()
    assert record["execution"]["value"] == pytest.approx(float(expected))


def test_a_book_without_the_reporting_region_is_named_not_crashed(
        monkeypatch, book, semantics):
    assert "canonical_region_reporting" not in book.columns
    payload, record, _ = _served(_intent(geography=_REGION), monkeypatch, book,
                                 semantics)
    assert payload is None
    assert record["execution"]["fields_not_in_book"] == ["canonical_region_reporting"]
    assert record["disposition"] != "EXECUTION_ERROR"


def test_the_temporal_path_still_refuses_geography():
    plan = _plan(geography=_REGION)
    assert adapter.check_structure(plan)[1] == adapter.GEOGRAPHY_REQUESTED
    assert adapter.check_eligibility(plan) == (True, "", "")


def test_region_and_two_dimensions_is_too_many_axes():
    plan = _plan(geography=_REGION, dimensions=["ltv_bucket", "product_type"])
    assert adapter.check_eligibility(plan)[1] == adapter.TOO_MANY_DIMENSIONS


# --------------------------------------------------------------------------- #
# median / largest / smallest
# --------------------------------------------------------------------------- #

@pytest.mark.parametrize("statistic, reduce, words", [
    ("median", pd.Series.median, "Median Balance: £"),
    ("max", pd.Series.max, "Maximum Balance: £"),
    ("min", pd.Series.min, "Minimum Balance: £")])
def test_median_largest_smallest_are_the_executors(statistic, reduce, words,
                                                  monkeypatch, book, semantics):
    intent = _intent(operation="point_in_time",
                     measures=[{"concept": "current_outstanding_balance",
                                "statistic": statistic}])
    payload, record, coverage = _served(intent, monkeypatch, book, semantics)
    assert payload is not None, record.get("execution")
    assert coverage["unaccounted"] == []
    assert record["execution"]["value"] == pytest.approx(
        float(reduce(pd.to_numeric(book["current_outstanding_balance"]))))
    # The answer names the statistic and states money as money.
    assert payload["answer"].startswith(words), payload["answer"]


# --------------------------------------------------------------------------- #
# one concept per meaning; a field the book lacks
# --------------------------------------------------------------------------- #

def test_borrower_structure_is_the_governed_borrower_type():
    from mi_agent.interpretation_v2.vocabulary import load_governed_vocabulary
    vocabulary = load_governed_vocabulary()
    assert "borrower_structure" not in vocabulary.concepts
    for word in ("borrower_structure", "sole_or_joint", "single_vs_joint"):
        assert vocabulary.resolve(word).concept_id == "borrower_type"
    plan = _plan(dimensions=["borrower_structure"])
    assert plan["outputs"][0]["dimensions"][0]["canonical_field"] == "borrower_type"


def test_a_field_the_book_lacks_is_named_as_the_books(monkeypatch, book, semantics):
    assert "borrower_type" not in book.columns
    payload, record, _ = _served(_intent(dimensions=["borrower_type"]),
                                 monkeypatch, book, semantics)
    assert payload is None
    assert record["execution"]["fields_not_in_book"] == ["borrower_type"]


def test_the_book_carrying_it_is_answered(monkeypatch, book, semantics):
    frame = book.copy()
    frame["borrower_type"] = ["joint" if i % 3 else "single" for i in range(len(frame))]
    payload, record, coverage = _served(_intent(dimensions=["borrower_type"]),
                                        monkeypatch, frame, semantics)
    assert payload is not None, record.get("execution")
    assert coverage["unaccounted"] == []


def test_occupancy_type_is_answered_where_the_book_carries_it(monkeypatch, book,
                                                              semantics):
    payload, record, _ = _served(_intent(dimensions=["occupancy_type"]),
                                 monkeypatch, book, semantics)
    assert payload is not None, record.get("execution")
    payload, record, _ = _served(_intent(dimensions=["occupancy_type"]), monkeypatch,
                                 book.drop(columns=["occupancy_type"]), semantics)
    assert payload is None
    assert record["execution"]["fields_not_in_book"] == ["occupancy_type"]
