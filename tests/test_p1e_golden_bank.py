#!/usr/bin/env python3
"""tests/test_p1e_golden_bank.py — the focused P1E question bank.

The five curated CFO questions live in ``test_p1e_multi_measure.py``. This is
the breadth bank behind them: the same governed capability asked twenty-six
ways — different phrasings of the same measure set, different sized sets,
scoped, grouped, and both at once.

EVERY expected number is computed here with pandas against the canonical
fixture. The MI Agent's own aggregation functions are never the oracle, so a
defect in the executor cannot make its own bank pass.

Two properties are asserted for every question in the bank:

  1. every governed measure the question named was CALCULATED, and
  2. every measure was calculated over the SAME population — the one the
     receipt names.

Run: python -m pytest tests/test_p1e_golden_bank.py
"""
from __future__ import annotations

import logging
import os
import sys
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

import pandas as pd
import pytest

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from demo_platform import config as cfg  # noqa: E402

BALANCE = "current_outstanding_balance"
LTV = "current_loan_to_value"
RATE = "current_interest_rate"
AGE = "youngest_borrower_age"
REGION = "collateral_geography"
SOURCE = "source_portfolio_id"


# =========================================================================== #
# Environment
# =========================================================================== #
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
    cache: Dict[str, Dict[str, Any]] = {}

    def _ask(question: str) -> Dict[str, Any]:
        if question not in cache:
            cache[question] = (execute_governed_mi_query(
                MiQueryRequest(question=question), ctx).result or {})
        return cache[question]

    return _ask


# =========================================================================== #
# Independent truth — pandas, not the agent
# =========================================================================== #
def wavg(frame: pd.DataFrame, value: str) -> float:
    valid = frame[[value, BALANCE]].dropna()
    return float((valid[value] * valid[BALANCE]).sum() / valid[BALANCE].sum())


#: ``executed column -> how to compute it independently``
TRUTH: Dict[str, Callable[[pd.DataFrame], float]] = {
    f"{BALANCE}_sum": lambda f: float(f[BALANCE].sum()),
    "loan_count": lambda f: float(len(f)),
    f"{LTV}_weighted_avg": lambda f: wavg(f, LTV),
    f"{RATE}_weighted_avg": lambda f: wavg(f, RATE),
    f"{AGE}_avg": lambda f: float(f[AGE].mean()),
    f"{AGE}_weighted_avg": lambda f: wavg(f, AGE),
    "current_valuation_amount_avg": lambda f: float(f["current_valuation_amount"].mean()),
}


def kpis(envelope: Dict[str, Any]) -> Dict[str, float]:
    for artifact in envelope.get("artifacts") or []:
        if artifact.get("type") == "kpi":
            return {k.get("field"): k.get("rawValue")
                    for k in (artifact.get("kpis") or [])}
    return {}


def table_rows(envelope: Dict[str, Any]) -> List[Dict[str, Any]]:
    for artifact in envelope.get("artifacts") or []:
        if artifact.get("type") == "table":
            return list(artifact.get("rows") or [])
    return []


def executed(envelope: Dict[str, Any]) -> List[Dict[str, Any]]:
    return list((envelope.get("queryTrace") or {}).get("executed_measures") or [])


def receipt(envelope: Dict[str, Any]) -> str:
    return (envelope.get("executionSummary") or {}).get("receipt") or ""


# =========================================================================== #
# The bank
# =========================================================================== #
#: ``(id, question, expected measure columns, population)``. The population is a
#: predicate over the canonical frame — the SAME population every measure in the
#: answer must have been calculated over.
WHOLE_BOOK: Optional[Callable[[pd.DataFrame], pd.Series]] = None
LONDON = lambda f: f[REGION] == "London"                       # noqa: E731
SOUTH_EAST = lambda f: f[REGION] == "South East"               # noqa: E731
OVER_85 = lambda f: f[AGE] > 85                                # noqa: E731
OVER_80 = lambda f: f[AGE] > 80                                # noqa: E731
LTV_OVER_60 = lambda f: f[LTV] > 60                            # noqa: E731
ACQUIRED = lambda f: f[SOURCE] == "alp_acquired"               # noqa: E731

B = f"{BALANCE}_sum"
N = "loan_count"
L = f"{LTV}_weighted_avg"
R = f"{RATE}_weighted_avg"
A = f"{AGE}_avg"

BANK = [
    # --- the same four measures, said four ways --------------------------- #
    ("P1E-01", "Give me the current funded balance, loan count, weighted-average "
     "LTV and weighted-average interest rate.", [B, N, L, R], WHOLE_BOOK),
    ("P1E-02", "Show me total exposure, loan count, weighted-average loan to "
     "value and weighted-average interest rate.", [B, N, L, R], WHOLE_BOOK),
    ("P1E-03", "What is the outstanding balance, the number of loans, the "
     "weighted average LTV and the weighted average interest rate?",
     [B, N, L, R], WHOLE_BOOK),
    ("P1E-04", "Total balance, how many loans, weighted-average LTV and "
     "weighted-average coupon.", [B, N, L, R], WHOLE_BOOK),
    # --- two measures ------------------------------------------------------ #
    ("P1E-05", "What is the total balance and the loan count?", [B, N], WHOLE_BOOK),
    ("P1E-06", "Give me balance and weighted-average LTV.", [B, L], WHOLE_BOOK),
    ("P1E-07", "How many loans are there and what is the weighted average "
     "interest rate?", [N, R], WHOLE_BOOK),
    ("P1E-08", "Total balance and average borrower age please.", [B, A], WHOLE_BOOK),
    ("P1E-09", "Weighted-average LTV and weighted-average interest rate.",
     [L, R], WHOLE_BOOK),
    # --- three measures ---------------------------------------------------- #
    ("P1E-10", "Balance, loan count and weighted-average LTV.", [B, N, L], WHOLE_BOOK),
    ("P1E-11", "Give me loan count, weighted-average LTV and average borrower age.",
     [N, L, A], WHOLE_BOOK),
    ("P1E-12", "What is the total balance, weighted average LTV and weighted "
     "average interest rate?", [B, L, R], WHOLE_BOOK),
    ("P1E-13", "Give me balance, loan count and average property value.",
     [B, N, "current_valuation_amount_avg"], WHOLE_BOOK),
    # --- scoped by geography ----------------------------------------------- #
    ("P1E-14", "For the London book, give me balance, number of loans, "
     "weighted-average LTV and average borrower age.", [B, N, L, A], LONDON),
    ("P1E-15", "In the South East, what is the total balance, loan count and "
     "weighted-average LTV?", [B, N, L], SOUTH_EAST),
    ("P1E-16", "For London, give me total balance and weighted-average interest "
     "rate.", [B, R], LONDON),
    # --- scoped by threshold ----------------------------------------------- #
    ("P1E-17", "For borrowers over 85, what are the total balance, loan count, "
     "weighted-average LTV and average interest rate?", [B, N, L, R], OVER_85),
    ("P1E-18", "For loans with LTV above 60%, give me balance, loan count and "
     "weighted-average interest rate.", [B, N, R], LTV_OVER_60),
    ("P1E-19", "For borrowers over 80, what is the balance and the loan count?",
     [B, N], OVER_80),
    # --- scoped by sourcing cohort ----------------------------------------- #
    ("P1E-20", "For the acquired book, what are balance, loan count and "
     "weighted-average LTV?", [B, N, L], ACQUIRED),
]

#: Grouped questions: one table, every measure a column, one row per group.
GROUPED_BANK = [
    ("P1E-21", "Show me balance, loan count and weighted-average LTV by region.",
     [B, N, L], REGION, WHOLE_BOOK),
    ("P1E-22", "By region, give me total balance, number of loans and weighted "
     "average interest rate.", [B, N, R], REGION, WHOLE_BOOK),
    ("P1E-23", "Show balance and loan count by originator.", [B, N],
     "originator_name", WHOLE_BOOK),
    ("P1E-24", "For borrowers over 80, show balance and loan count by region.",
     [B, N], REGION, OVER_80),
    ("P1E-25", "By region, show me weighted-average LTV and weighted-average "
     "interest rate.", [L, R], REGION, WHOLE_BOOK),
    ("P1E-26", "For London, show balance, loan count and weighted-average LTV "
     "by originator.", [B, N, L], "originator_name", LONDON),
]


#: The executor buckets rows with a missing grouping value under an explicit
#: label rather than dropping them, so the group totals still reconcile to the
#: portfolio. The oracle must model that too — silently dropping two loans here
#: would make a CORRECT answer look wrong, and would hide a real regression if
#: the product ever started dropping them for real.
from mi_agent.mi_query_executor import MISSING_BUCKET_LABEL  # noqa: E402


def population(book: pd.DataFrame, predicate) -> pd.DataFrame:
    return book if predicate is None else book[predicate(book)]


def grouped(frame: pd.DataFrame, dimension: str) -> Dict[str, pd.DataFrame]:
    """``{group label: rows}``, missing values bucketed as the executor does."""
    labels = frame[dimension].fillna(MISSING_BUCKET_LABEL)
    return {str(label): frame[labels == label] for label in labels.unique()}


# --------------------------------------------------------------------------- #
# Ungrouped: every measure calculated, and calculated correctly
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("case_id,question,columns,predicate", BANK,
                         ids=[c[0] for c in BANK])
def test_every_named_measure_is_calculated(ask, case_id, question, columns,
                                           predicate):
    envelope = ask(question)
    assert envelope["ok"] is True, f"{case_id}: {envelope.get('error')}"
    delivered = kpis(envelope)
    missing = [c for c in columns if c not in delivered]
    assert not missing, (f"{case_id} asked for {len(columns)} measures and "
                         f"{len(missing)} were not delivered: {missing} "
                         f"(delivered {sorted(delivered)})")


@pytest.mark.parametrize("case_id,question,columns,predicate", BANK,
                         ids=[c[0] for c in BANK])
def test_every_figure_reconciles_to_pandas(ask, book, case_id, question,
                                           columns, predicate):
    envelope = ask(question)
    assert envelope["ok"] is True, f"{case_id}: {envelope.get('error')}"
    frame = population(book, predicate)
    delivered = kpis(envelope)
    for column in columns:
        assert column in TRUTH, f"{case_id}: no independent truth for {column}"
        assert delivered[column] == pytest.approx(TRUTH[column](frame), rel=1e-9), (
            f"{case_id}: {column} disagrees with pandas")


@pytest.mark.parametrize("case_id,question,columns,predicate", BANK,
                         ids=[c[0] for c in BANK])
def test_one_population_serves_every_measure(ask, book, case_id, question,
                                             columns, predicate):
    """The composition rule: measures share a population, they do not each
    carry their own. Proven by the loan count — if any measure had been
    calculated over a different frame, the count could not match all of them."""
    envelope = ask(question)
    assert envelope["ok"] is True, f"{case_id}: {envelope.get('error')}"
    expected = len(population(book, predicate))
    assert f"{expected:,} loans" in receipt(envelope), (
        f"{case_id}: the receipt does not name the {expected:,}-loan population")


# --------------------------------------------------------------------------- #
# Grouped: ONE table carrying every measure, not one table per measure
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("case_id,question,columns,dimension,predicate",
                         GROUPED_BANK, ids=[c[0] for c in GROUPED_BANK])
def test_a_grouped_measure_set_is_one_table(ask, book, case_id, question,
                                            columns, dimension, predicate):
    envelope = ask(question)
    assert envelope["ok"] is True, f"{case_id}: {envelope.get('error')}"
    rows = table_rows(envelope)
    assert rows, f"{case_id}: no table was returned"
    groups = grouped(population(book, predicate), dimension)
    assert len(rows) == len(groups), (
        f"{case_id}: {len(rows)} rows for {len(groups)} groups — a measure set "
        f"must not split the table")
    for column in columns:
        assert column in rows[0], (f"{case_id}: {column} is missing from the "
                                   f"table (columns {sorted(rows[0])})")


@pytest.mark.parametrize("case_id,question,columns,dimension,predicate",
                         GROUPED_BANK, ids=[c[0] for c in GROUPED_BANK])
def test_every_group_reconciles_to_pandas(ask, book, case_id, question,
                                          columns, dimension, predicate):
    envelope = ask(question)
    assert envelope["ok"] is True, f"{case_id}: {envelope.get('error')}"
    groups = grouped(population(book, predicate), dimension)
    for row in table_rows(envelope):
        group = str(row.get(dimension))
        assert group in groups, f"{case_id}: unexpected group {group!r}"
        subset = groups[group]
        for column in columns:
            assert row[column] == pytest.approx(TRUTH[column](subset), rel=1e-9), (
                f"{case_id}: {column} for {group!r} disagrees with pandas")


# --------------------------------------------------------------------------- #
# The bank as a whole
# --------------------------------------------------------------------------- #
def test_the_bank_covers_two_three_and_four_measure_requests():
    sizes = {len(c[2]) for c in BANK} | {len(c[2]) for c in GROUPED_BANK}
    assert {2, 3, 4} <= sizes, f"the bank does not exercise every set size: {sizes}"


def test_the_bank_is_answered_deterministically(ask):
    """No question in the bank depends on an LLM being reachable: the parser
    is switched off in this module's environment and every question answers."""
    for case in BANK + [(c[0], c[1]) for c in GROUPED_BANK]:
        envelope = ask(case[1])
        assert (envelope.get("metadata") or {}).get("parserMode") == "deterministic"
