"""SELECT_POPULATION(kind=row_predicates) — the filter-plan prerequisite.

The C6 filter question was never "can Trakt filter". It was: can a compositional
plan express a narrowing WITHOUT any route reading English again? These tests
assert the two halves of that:

  BUILT FROM THE CONTRACT   the population step comes from `RowPredicateClaim`
                            and from nothing else — not `spec.filters`, not the
                            question text, not a provenance string.
  EXECUTED BY THE ONE OWNER the predicates go to `apply_population`, which since
                            the parity work runs `governed_predicate_mask`, the
                            same owner `_apply_filters` uses. That is why the
                            substitution is row-for-row identical rather than
                            merely similar.

The two `select_population` modes stay structurally apart. Overloading
`lens_filters` with value predicates would put portfolio IDENTITY back into the
value channel, which is the P1I-A ruling in reverse.
"""
from __future__ import annotations

import os
from pathlib import Path

import pytest

from mi_agent_api import analytical_plan as plan_mod
from mi_agent_api.analytical_plan import (KIND_ROW_PREDICATES,
                                          KIND_SOURCE_PORTFOLIO_LENS,
                                          SELECT_POPULATION, Plan)
from question_interpretation.schema import FILLED, RowPredicateClaim

_REPO_ROOT = Path(__file__).resolve().parents[2]


class _Contract:
    """Only what the plan builder is allowed to see."""

    def __init__(self, *claims):
        self.row_predicates = list(claims)


def _claim(field, op, value):
    return RowPredicateClaim(state=FILLED, field_key=field, operator=op, value=value)


# --------------------------------------------------------------------------- #
# The step is built from the contract, and only from the contract
# --------------------------------------------------------------------------- #
def test_the_population_step_is_built_from_row_predicate_claims():
    step = plan_mod.row_predicate_step(
        _Contract(_claim("current_loan_to_value", "gt", 50.0)))
    assert step.primitive == SELECT_POPULATION
    assert step.inputs["kind"] == KIND_ROW_PREDICATES
    assert step.inputs["predicates"] == [
        {"field": "current_loan_to_value", "op": "gt", "value": 50.0}]
    assert not step.blocked


def test_a_question_that_narrows_nothing_plans_no_narrowing():
    """None, not a blocked step: planning no narrowing is the ordinary case."""
    assert plan_mod.row_predicate_step(_Contract()) is None


def test_a_claim_with_no_resolved_field_is_not_planned():
    """An unresolved claim must not become a predicate on a field named None."""
    assert plan_mod.row_predicate_step(_Contract(_claim(None, "gt", 50.0))) is None


def test_the_reader_returns_governed_predicates_not_dicts():
    """The one thing every caller must NOT do is re-interpret these, so they
    arrive as the executor's own type."""
    from mi_agent.population import Predicate
    step = plan_mod.row_predicate_step(_Contract(_claim("youngest_borrower_age", "gt", 75)))
    assert plan_mod.row_predicates(step) == [Predicate("youngest_borrower_age", "gt", 75)]


def test_two_clauses_plan_two_predicates_in_order():
    step = plan_mod.row_predicate_step(
        _Contract(_claim("youngest_borrower_age", "gt", 75),
                  _claim("current_loan_to_value", "gt", 50.0)))
    assert [p["field"] for p in step.inputs["predicates"]] == [
        "youngest_borrower_age", "current_loan_to_value"]


# --------------------------------------------------------------------------- #
# The two modes stay structurally separate
# --------------------------------------------------------------------------- #
def test_the_lens_reader_ignores_a_row_predicate_step():
    """`lens_filters` must never see a value predicate as a portfolio id."""
    step = plan_mod.row_predicate_step(_Contract(_claim("borrower_type", "eq", "Joint")))
    assert plan_mod.lens_filters(Plan((step,), ())) is None


def test_the_lens_label_reader_ignores_a_row_predicate_step():
    """It used to take the FIRST select_population step. Once a plan can carry
    two, "the first one" is no longer the lens."""
    step = plan_mod.row_predicate_step(_Contract(_claim("borrower_type", "eq", "Joint")))
    assert plan_mod.lens_label(Plan((step,), ())) == "Total"


def test_the_row_predicate_reader_ignores_a_lens_step():
    lens = plan_mod.Step(SELECT_POPULATION,
                         {"kind": KIND_SOURCE_PORTFOLIO_LENS,
                          "portfolio_ids": ["nbs"], "label": "Northbridge"},
                         because="scope")
    assert plan_mod.row_predicates(Plan((lens,), ())) == []


def test_a_blocked_step_yields_no_predicates():
    """A plan with a blocked step is a refusal, never an answer with the step
    omitted — so its predicates must not leak out as if they had run."""
    step = plan_mod.row_predicate_step(_Contract(_claim("current_loan_to_value", "gt", 50.0)))
    blocked = plan_mod.Step(step.primitive, step.inputs, because=step.because,
                            blocked="no contract field")
    assert plan_mod.row_predicates(Plan((blocked,), ())) == []


# --------------------------------------------------------------------------- #
# End to end: the delivered filtered evolution case
# --------------------------------------------------------------------------- #
_TAPE = (_REPO_ROOT / "demo_platform" / "workspace" / "store" / "processed"
         / "platform" / "alderbridge" / "2026-06-30"
         / "platform_canonical_typed.csv")

#: The delivered series, to the penny. Established before the filter wiring and
#: required to survive it unchanged.
LTV_ABOVE_50 = (432425355.79, 450969362.11, 472527483.38)


@pytest.fixture(scope="module")
def ask():
    if not _TAPE.exists():
        pytest.skip("calibration book not built")
    import logging

    from demo_platform import config as cfg

    os.environ.update(cfg.mi_env(period_role="current"))
    os.environ["MI_AGENT_LLM_PARSER"] = "off"
    os.environ["MI_AGENT_LLM_ENABLED"] = "0"
    logging.disable(logging.WARNING)
    from mi_agent_api.mi_service import MiQueryRequest, execute_governed_mi_query
    from trakt_core.context import ExecutionContext

    ctx = ExecutionContext.for_internal(cfg.CLIENT_ID)
    return lambda q: (execute_governed_mi_query(
        MiQueryRequest(question=q), ctx).result or {})


def _series(envelope):
    seen, out = set(), []
    for artifact in envelope.get("artifacts") or []:
        for row in artifact.get("rows") or []:
            key = row.get("period") or row.get("week")
            if key and key not in seen:
                seen.add(key)
                out.append(round(float(row.get("value")), 2))
    return out


def test_the_delivered_filtered_series_is_unchanged_to_the_penny(ask):
    envelope = ask("balance trend where LTV above 50%")
    assert (envelope.get("metadata") or {}).get("route") == "evolution"
    assert _series(envelope) == [round(v, 2) for v in LTV_ABOVE_50]


def test_the_filtered_series_declares_the_field_the_plan_applied(ask):
    """The ledger is EXECUTION evidence. It now names the field the plan
    selected, not the key the spec happened to carry."""
    ledger = ((ask("balance trend where LTV above 50%").get("metadata") or {})
              .get("populationApplied") or {})
    assert ledger["applied"] == [
        "current_loan_to_value (applied within each period)"]
    assert ledger["unavailable"] == []
    assert ledger["rowsAfter"] == 1889


def test_the_filtered_series_is_genuinely_narrower_than_the_unfiltered_one(ask):
    """Non-vacuity: a filter that changed nothing would make every assertion
    above pass against a whole-book series."""
    filtered = _series(ask("balance trend where LTV above 50%"))
    whole = _series(ask("balance trend"))
    assert filtered and whole and len(filtered) == len(whole)
    assert all(f < w for f, w in zip(filtered, whole))


@pytest.mark.parametrize("question", [
    "balance trend",
    "loan count trend",
    "pipeline trend",
    "show the funnel by week",
])
def test_an_unfiltered_evolution_question_is_untouched(ask, question):
    """The new population step must not reach a question that narrows nothing."""
    envelope = ask(question)
    assert envelope.get("answer")
    assert not ((envelope.get("metadata") or {}).get("populationApplied") or {}
                ).get("unavailable")


# --------------------------------------------------------------------------- #
# The route no longer interprets a filter
# --------------------------------------------------------------------------- #
def test_the_per_period_narrower_cannot_see_the_spec():
    """An architectural guard, not a style rule: a function that never receives
    the spec cannot re-derive a filter's meaning from it, whatever a later edit
    does inside the body.

    Read with AST, not with `in`. The first cut of this guard used a substring
    check and failed against its own docstring, which explains what it replaced
    — the same way an earlier estate guard flagged a label map and then
    `ast.parse`. A mention is not a call.
    """
    import ast
    import inspect
    import textwrap

    from mi_agent_api import chat_routing

    signature = inspect.signature(chat_routing._filtered_funded_evo)
    assert "spec" not in signature.parameters
    assert "predicates" in signature.parameters

    tree = ast.parse(textwrap.dedent(
        inspect.getsource(chat_routing._filtered_funded_evo)))
    called = {node.func.id for node in ast.walk(tree)
              if isinstance(node, ast.Call) and isinstance(node.func, ast.Name)}
    assert "_apply_filters" not in called, "the per-period narrower calls the spec filterer"
    read = {f"{node.value.id}.{node.attr}" for node in ast.walk(tree)
            if isinstance(node, ast.Attribute) and isinstance(node.value, ast.Name)}
    assert "spec.filters" not in read, "the per-period narrower reads spec.filters"
    assert not any(name.startswith("spec.") for name in read), (
        f"the per-period narrower still reaches into the spec: {sorted(read)}")


def test_the_filter_prose_is_described_from_the_governed_predicate():
    from mi_agent.population import Predicate
    from mi_agent_api import chat_routing

    assert chat_routing._filter_summary(
        [Predicate("current_loan_to_value", "gt", 50.0)]) == \
        "current_loan_to_value gt 50.0"
    assert chat_routing._filter_summary([]) == ""


def test_an_unappliable_population_fails_closed_rather_than_widening():
    """The route defers to the controlled point-in-time path, which refuses —
    it must never answer a whole-book trend labelled as a narrowed one."""
    import pandas as pd
    import pytest as _pytest

    from mi_agent.mi_query_executor import MIQueryExecutionError
    from mi_agent.population import Predicate
    from mi_agent_api import chat_routing

    frames = [{"df": pd.DataFrame({"current_outstanding_balance": [1.0, 2.0]}),
               "reporting_date": "2026-06-30"}]
    original = chat_routing.evolution_mod.funded_frames
    chat_routing.evolution_mod.funded_frames = lambda *a, **k: frames
    try:
        with _pytest.raises(MIQueryExecutionError):
            chat_routing._filtered_funded_evo(
                None, "c", None, [Predicate("borrower_type", "eq", "Joint")],
                {}, "funded_balance")
    finally:
        chat_routing.evolution_mod.funded_frames = original
