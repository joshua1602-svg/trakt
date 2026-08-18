#!/usr/bin/env python3
"""tests/test_p1l_population_propagation.py — the governed ROW POPULATION.

P1K proved the guard was structurally blind to a dropped population. A question
names a population in one of two channels:

  * the PORTFOLIO SCOPE channel  — "the acquired book"  (portfolio_lens)
  * the ROW PREDICATE channel    — "the back book"      (spec.filters)

Twelve of thirteen specialist routes read the first and ignored the second, so
"which region grew the most last month in the back book?" answered across the
WHOLE book, returned ok=True, and shipped a spec still claiming the filter. The
figure was plausible — the back book is 89% of this portfolio — and nothing in
the answer or the receipt revealed the loss.

The invariant this file defends:

    requested population -> applied -> execution evidence -> P0 facet -> receipt

with exactly four permitted outcomes (APPLIED / UNAVAILABLE / UNSUPPORTED /
REJECTED) and no fifth state in which the whole book is quietly substituted.

Every figure is recomputed with pandas from the fixture. The agent is never its
own oracle.

Run: python -m pytest tests/test_p1l_population_propagation.py
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
from mi_agent import execution_receipt as er  # noqa: E402
from mi_agent import population as pop  # noqa: E402

BALANCE = "current_outstanding_balance"
LTV = "current_loan_to_value"
AGE = "youngest_borrower_age"
SEG = "seasoning_segment"


@pytest.fixture(scope="module")
def governed_env():
    root = cfg.local_blob_root() / "processed" / "platform" / cfg.CLIENT_ID
    if not root.exists():
        pytest.skip("the demonstration platform has not been built")
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


def answer(envelope) -> str:
    return envelope.get("answer") or envelope.get("error") or ""


def population_applied(envelope) -> Dict[str, Any]:
    return (envelope.get("metadata") or {}).get("populationApplied") or {}


def population_facets(envelope) -> List[Dict[str, Any]]:
    guard = envelope.get("semanticGuard") or {}
    return [f for f in (guard.get("facets") or [])
            if f.get("kind") == er.KIND_POPULATION]


def wavg(frame: pd.DataFrame, value: str = LTV) -> float:
    valid = frame[[value, BALANCE]].dropna()
    return float((valid[value] * valid[BALANCE]).sum() / valid[BALANCE].sum())


# =========================================================================== #
# The population model itself
# =========================================================================== #
def test_scope_keys_are_not_row_predicates():
    """P1I-A governs the portfolio scope as SCOPE. Collapsing it into a row
    predicate to make one common mechanism would regress that ruling."""
    predicates = pop.material_predicates(
        {"source_portfolio_id": ["alp_acquired"], SEG: "Back Book"})
    assert [p.field for p in predicates] == [SEG]


def test_a_reporting_basis_key_is_not_a_population():
    """Selecting a snapshot does not change WHICH loans are included."""
    assert pop.material_predicates({"reporting_date": "2026-06-30"}) == []


def test_an_unclassified_predicate_counts_as_population():
    """The ledger fails SAFE: a key nobody has classified is population-defining
    until someone rules otherwise, rather than assumed harmless."""
    predicates = pop.material_predicates({"some_new_governed_field": "X"})
    assert [p.field for p in predicates] == ["some_new_governed_field"]


def test_apply_population_narrows_and_evidences(book):
    predicates = pop.material_predicates({SEG: "Back Book"})
    narrowed, evidence = pop.apply_population(book, predicates)
    assert evidence.rows_before == len(book)
    assert evidence.rows_after == int((book[SEG] == "Back Book").sum())
    assert evidence.applied and not evidence.unavailable
    assert len(narrowed) == evidence.rows_after


def test_an_absent_column_is_unavailable_never_skipped(book):
    """A predicate the frame cannot express must be reported, so the caller
    refuses. Silently ignoring it is the P1K defect."""
    predicates = pop.material_predicates({"no_such_governed_field": "X"})
    _narrowed, evidence = pop.apply_population(book, predicates)
    assert evidence.unavailable and not evidence.applied


def test_population_loss_is_number_changing_not_a_shape():
    """Dropping the population changes which rows were counted, so it changes
    every number. It can never be disclosed as a mere partial."""
    assert er.KIND_POPULATION in er.NUMBER_OR_SUBJECT_FACETS


def test_spec_presence_is_not_execution_evidence():
    """The heart of P1K: the spec still carried the filter while the route
    ignored it. A facet with no evidence must be LOST, not APPLIED."""
    facets = er.population_facets({"filters": {SEG: "Back Book"}})
    er.reconcile_population(facets, None)
    assert facets and facets[0].status == er.LOST


def test_evidence_marks_the_facet_applied():
    facets = er.population_facets({"filters": {SEG: "Back Book"}})
    er.reconcile_population(
        facets, {"applied": [f"{SEG} = Back Book"], "unavailable": []})
    assert facets[0].status == er.APPLIED


# =========================================================================== #
# The five P1K silent semantic errors — each must be CORRECT or REFUSED
# =========================================================================== #
P1K_QUESTIONS = [
    "Which region grew the most last month in the back book?",
    "How much of the back book is in the top 10 postcodes?",
    "What is the largest single-loan exposure in the back book and what share "
    "of the back book is it?",
    "Where is the back book most concentrated?",
    "Am I close to breaching any concentration limits in the back book?",
]


@pytest.mark.parametrize("question", P1K_QUESTIONS)
def test_a_population_question_is_answered_or_refused_never_widened(ask, question):
    """The P1L invariant, end to end. Either the route honoured the population —
    and says so with evidence — or it refused. There is no third outcome."""
    envelope = ask(question)
    if envelope.get("ok"):
        evidence = population_applied(envelope)
        assert evidence.get("applied"), (
            f"answered without proving the population: {answer(envelope)[:160]!r}")
        assert evidence.get("rowsAfter") is not None
        assert evidence["rowsAfter"] < evidence["rowsBefore"], (
            "the population was claimed but narrowed nothing")
    else:
        assert "not substituted a broader figure" in answer(envelope) or \
               "could not be applied" in answer(envelope)


@pytest.mark.parametrize("question", P1K_QUESTIONS)
def test_no_whole_book_substitute(ask, book, question):
    """The specific harm: a whole-book figure presented for a narrowed question."""
    envelope = ask(question)
    if not envelope.get("ok"):
        return
    evidence = population_applied(envelope)
    assert evidence.get("rowsBefore") == len(book)
    assert evidence.get("rowsAfter") == int((book[SEG] == "Back Book").sum())


# =========================================================================== #
# Independent truth for the answers that now succeed
# =========================================================================== #
def test_back_book_geographic_concentration_reconciles(ask, book):
    envelope = ask("How much of the back book is in the top 10 postcodes?")
    if not envelope.get("ok"):
        pytest.skip("route refuses this composition; covered above")
    back = book[book[SEG] == "Back Book"]
    top = (back.groupby("geographic_region_collateral_itl3")[BALANCE].sum()
           .sort_values(ascending=False))
    # The governed answer states the largest concentration in £m.
    assert f"{top.iloc[0] / 1e6:.1f}m" in answer(envelope)


def test_back_book_largest_loan_share_uses_the_back_book_denominator(ask, book):
    """§11 denominator invariant. The share must be over the EXECUTED
    population — largest back-book loan / total BACK-BOOK exposure — not over
    the whole book, which is what P1K returned."""
    envelope = ask("What is the largest single-loan exposure in the back book "
                   "and what share of the back book is it?")
    if not envelope.get("ok"):
        pytest.skip("route refuses this composition; covered above")
    back = book[book[SEG] == "Back Book"]
    correct = back[BALANCE].max() / back[BALANCE].sum() * 100
    whole = book[BALANCE].max() / book[BALANCE].sum() * 100
    assert f"{correct:.3f}%" in answer(envelope) or f"{correct:.2f}%" in answer(envelope)
    assert f"{whole:.3f}%" not in answer(envelope), "whole-book denominator used"


# =========================================================================== #
# Generalised beyond the back book (§10) — other population predicates
# =========================================================================== #
@pytest.mark.parametrize("question", [
    "Where is exposure concentrated for borrowers over 85?",
    "Which region has the most balance among loans below 75% LTV?",
    "Where is exposure concentrated in the back book?",
])
def test_other_population_predicates_are_also_proven(ask, question):
    """P1L must not be a back-book patch: age, LTV and seasoning predicates all
    travel the same ledger."""
    envelope = ask(question)
    if envelope.get("ok"):
        assert population_applied(envelope).get("applied")
    else:
        assert "could not be applied" in answer(envelope) or \
               "not substituted" in answer(envelope)


def test_a_question_with_no_population_is_untouched(ask, book):
    """Scope control: the ledger must not disturb an ordinary whole-book
    question, or it would refuse the entire product."""
    envelope = ask("Where is the portfolio most concentrated?")
    assert envelope["ok"] is True, envelope.get("error")
    assert not population_facets(envelope)


# =========================================================================== #
# The scope channel must keep working (P1I-A / P1J-1 preserved)
# =========================================================================== #
def test_provenance_still_travels_the_scope_channel(ask):
    """"the acquired book" is SCOPE, applied by the lens, and must still narrow
    a routed answer — P1L must not have moved it into the predicate channel."""
    envelope = ask("Which region grew the most last month in the acquired book?")
    assert envelope["ok"] is True, envelope.get("error")
    scope = (envelope.get("portfolioScope") or {}).get("portfolio_ids") or []
    assert scope == ["alp_acquired"]


def test_entire_aum_remains_explicitly_widening(ask, book):
    """Sponsored book = ENTIRE_AUM (P1I-A). A widening scope is not a population
    restriction and must not be refused by the ledger."""
    envelope = ask("For the sponsored book, give me balance and loan count.")
    assert envelope["ok"] is True, envelope.get("error")
    assert not population_facets(envelope)


def test_seasoning_phrases_do_not_become_places(ask):
    """P1K finding C: P1I-A masks governed SCOPE phrases from the place
    resolver; P1J-1's seasoning vocabulary was never added, so "the front book"
    produced a filter on a collateral geography called "Front" that matched no
    rows."""
    from mi_agent.seasoning import mask_segment_phrases

    masked = mask_segment_phrases("how many acquired loans are in the front book?")
    assert "front book" not in masked.lower()
    assert len(masked) == len("how many acquired loans are in the front book?")

    envelope = ask("How many acquired loans are in the front book?")
    assert (envelope.get("spec") or {}).get("filters", {}).get(
        "collateral_geography") is None
