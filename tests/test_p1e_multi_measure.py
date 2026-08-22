#!/usr/bin/env python3
"""tests/test_p1e_multi_measure.py — P1E multi-measure composition.

A CFO asks ONE analytical question containing several governed measures:

    "For the London book, give me balance, number of loans, weighted-average
     LTV and average borrower age."

That is one request over one population, not four questions and not an excuse to
return one number. Before P1E the contract had a single ``metric`` slot, so the
deterministic parser kept whichever measure it matched first and the LLM — which
understood the request perfectly, saying so in its explanation — had nowhere to
put the set and returned none.

Every expected figure here is computed with pandas from the canonical fixture.
The agent's own aggregation functions are never the oracle.
"""
from __future__ import annotations

import logging
import os
import sys
from pathlib import Path
from typing import Any, Dict, List

import pandas as pd
import pytest

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from demo_platform import config as cfg  # noqa: E402
from mi_agent.mi_query_spec import MAX_MEASURES, MIQuerySpec, normalise_measures  # noqa: E402

BALANCE = "current_outstanding_balance"
LTV = "current_loan_to_value"
RATE = "current_interest_rate"
AGE = "youngest_borrower_age"
REGION = "collateral_geography"
COHORT = "source_portfolio_type"


# =========================================================================== #
# Environment + independent truth
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


def weighted_average(frame: pd.DataFrame, value: str) -> float:
    """The governed weighting: balance-weighted, per the semantic registry."""
    valid = frame[[value, BALANCE]].dropna()
    return float((valid[value] * valid[BALANCE]).sum() / valid[BALANCE].sum())


def profile(frame: pd.DataFrame) -> Dict[str, float]:
    """Independent truth for one population."""
    return {"balance": float(frame[BALANCE].sum()), "loans": int(len(frame)),
            "wa_ltv": weighted_average(frame, LTV),
            "wa_rate": weighted_average(frame, RATE),
            "avg_age": float(frame[AGE].mean())}


def _summary_row(envelope: Dict[str, Any]) -> Dict[str, Any]:
    """Every KPI the answer actually published, keyed by its executed column.

    The KPI artifact is the delivered surface: if a measure is not here, the
    user did not receive it, whatever the spec asked for.
    """
    for artifact in envelope.get("artifacts") or []:
        if artifact.get("type") == "kpi":
            return {kpi.get("field"): kpi.get("rawValue")
                    for kpi in (artifact.get("kpis") or [])}
    return {}


def _executed(envelope: Dict[str, Any]) -> List[Dict[str, Any]]:
    return list((envelope.get("queryTrace") or {}).get("executed_measures") or [])


def _receipt(envelope: Dict[str, Any]) -> str:
    return (envelope.get("executionSummary") or {}).get("receipt") or ""


# =========================================================================== #
# The semantic contract — backward compatible by NORMALISATION
# =========================================================================== #
def test_a_single_metric_normalises_to_a_one_element_measure_set():
    """The compatibility guarantee: one contract downstream, not two engines."""
    spec = MIQuerySpec.from_dict({"metric": BALANCE, "aggregation": "sum"})
    assert spec.measures == [{"field": BALANCE, "aggregation": "sum"}]
    assert spec.metric == BALANCE and spec.aggregation == "sum"


def test_a_spec_with_no_metric_carries_no_measures():
    assert MIQuerySpec.from_dict({"intent": "summary"}).measures == []


@pytest.mark.parametrize("raw,expected", [
    ([{"field": "a", "aggregation": "sum"}], [{"field": "a", "aggregation": "sum"}]),
    (["a", "b"], [{"field": "a"}, {"field": "b"}]),
    ({"field": "a"}, [{"field": "a"}]),
    ([{"metric": "a", "agg": "avg"}], [{"field": "a", "aggregation": "avg"}]),
])
def test_every_recognised_measure_shape_is_folded(raw, expected):
    assert normalise_measures(raw)[0] == expected


def test_an_unrecognised_measure_is_reported_never_dropped():
    measures, notes = normalise_measures([123, {"aggregation": "sum"}])
    assert measures == []
    assert len(notes) == 2 and all(notes)


def test_duplicate_measures_are_folded_not_double_reported():
    """"balance and total balance" is one governed measure."""
    measures, _ = normalise_measures(
        [{"field": BALANCE, "aggregation": "sum"}, {"field": BALANCE}])
    assert measures == [{"field": BALANCE, "aggregation": "sum"}]


def test_the_measure_set_is_visible_to_validation():
    """Every measure is a field reference, so validation sees the whole request
    rather than only its first measure."""
    spec = MIQuerySpec.from_dict({"measures": [{"field": BALANCE}, {"field": LTV},
                                               {"field": "loan_count"}]})
    referenced = spec.referenced_fields()
    assert BALANCE in referenced and LTV in referenced
    assert "loan_count" not in referenced   # a governed count, not a field


# =========================================================================== #
# The five curated CFO questions
# =========================================================================== #
CFO_01 = ("Give me the current funded balance, loan count, weighted-average LTV "
          "and weighted-average interest rate.")
CFO_02 = ("Compare the direct and acquired books on balance, loan count, "
          "weighted-average LTV and average borrower age.")
CFO_03 = ("For the London book, give me balance, number of loans, "
          "weighted-average LTV and average borrower age.")
CFO_04 = ("For borrowers over 85, what are the total balance, loan count, "
          "weighted-average LTV and average interest rate?")
CFO_05 = "Show me balance, loan count and weighted-average LTV by region."


def test_cfo_01_portfolio_snapshot(ask, book):
    envelope = ask(CFO_01)
    assert envelope["ok"] is True, envelope.get("error")
    truth = profile(book)
    row = _summary_row(envelope)
    assert row["loan_count"] == truth["loans"]
    assert row[f"{BALANCE}_sum"] == pytest.approx(truth["balance"], rel=1e-9)
    assert row[f"{LTV}_weighted_avg"] == pytest.approx(truth["wa_ltv"], rel=1e-9)
    assert row[f"{RATE}_weighted_avg"] == pytest.approx(truth["wa_rate"], rel=1e-9)
    receipt = _receipt(envelope)
    for phrase in ("Balance", "Loans", "Weighted-average Current LTV",
                   "Weighted-average Interest Rate"):
        assert phrase in receipt


def test_cfo_02_direct_versus_acquired(ask, book):
    envelope = ask(CFO_02)
    assert envelope["ok"] is True, envelope.get("error")
    comparisons = {c["field"]: c for c in
                   envelope["workflow"]["metric_comparisons"]}
    direct = profile(book[book[COHORT] == "direct"])
    acquired = profile(book[book[COHORT] == "acquired"])

    assert comparisons[BALANCE]["portfolio_a"]["value"] == pytest.approx(
        direct["balance"], rel=1e-9)
    assert comparisons[BALANCE]["portfolio_b"]["value"] == pytest.approx(
        acquired["balance"], rel=1e-9)
    assert comparisons[LTV]["portfolio_a"]["value"] == pytest.approx(
        direct["wa_ltv"], rel=1e-9)
    assert comparisons[AGE]["portfolio_a"]["value"] == pytest.approx(
        direct["avg_age"], rel=1e-9)
    assert comparisons[AGE]["portfolio_b"]["value"] == pytest.approx(
        acquired["avg_age"], rel=1e-9)
    # Row counts are the loan-count measure on this route.
    sides = envelope["workflow"]["portfolio_results"]
    assert sides[0]["row_count"] == direct["loans"]
    assert sides[1]["row_count"] == acquired["loans"]


def test_cfo_03_london_profile(ask, book):
    envelope = ask(CFO_03)
    assert envelope["ok"] is True, envelope.get("error")
    truth = profile(book[book[REGION] == "London"])
    row = _summary_row(envelope)
    assert row["loan_count"] == truth["loans"] == 1380
    assert row[f"{BALANCE}_sum"] == pytest.approx(truth["balance"], rel=1e-9)
    assert row[f"{LTV}_weighted_avg"] == pytest.approx(truth["wa_ltv"], rel=1e-9)
    assert row[f"{AGE}_avg"] == pytest.approx(truth["avg_age"], rel=1e-9)
    assert "London" in _receipt(envelope)


def test_cfo_04_older_borrower_exposure(ask, book):
    envelope = ask(CFO_04)
    assert envelope["ok"] is True, envelope.get("error")
    truth = profile(book[book[AGE] > 85])
    row = _summary_row(envelope)
    assert row["loan_count"] == truth["loans"] == 86
    assert row[f"{BALANCE}_sum"] == pytest.approx(truth["balance"], rel=1e-9)
    assert row[f"{LTV}_weighted_avg"] == pytest.approx(truth["wa_ltv"], rel=1e-9)
    assert row[f"{RATE}_weighted_avg"] == pytest.approx(truth["wa_rate"], rel=1e-9)
    # The house convention, not >=85 (which would be 136 loans).
    assert "> 85" in _receipt(envelope)


def test_cfo_05_regional_management_view(ask, book):
    envelope = ask(CFO_05)
    assert envelope["ok"] is True, envelope.get("error")
    table = next(a for a in envelope["artifacts"] if a["type"] == "table")
    rows = {r[REGION]: r for r in table["rows"]}

    frame = book.copy()
    region = frame[REGION].astype("string").str.strip()
    frame[REGION] = region.where(
        ~(region.isna() | region.eq("")
          | region.str.lower().isin(["nan", "none", "nat"])), "Unknown / Missing")
    for name, group in frame.groupby(REGION):
        row = rows[name]
        assert row["loan_count"] == len(group)
        assert row[f"{BALANCE}_sum"] == pytest.approx(float(group[BALANCE].sum()),
                                                      rel=1e-9)
        assert row[f"{LTV}_weighted_avg"] == pytest.approx(
            weighted_average(group, LTV), rel=1e-9)


def test_the_grouped_view_is_one_table_not_one_per_measure(ask):
    """CFO-readable MI: three measures, one table."""
    envelope = ask(CFO_05)
    tables = [a for a in envelope["artifacts"] if a["type"] == "table"]
    assert len(tables) == 1
    keys = {c.get("key") for c in tables[0]["columns"]}
    assert {REGION, "loan_count", f"{BALANCE}_sum",
            f"{LTV}_weighted_avg"} <= keys


# =========================================================================== #
# SCOPE IDENTITY — every measure on the SAME population
# =========================================================================== #
@pytest.mark.parametrize("question,predicate", [
    (CFO_03, lambda df: df[df[REGION] == "London"]),
    (CFO_04, lambda df: df[df[AGE] > 85]),
])
def test_every_measure_uses_the_same_filtered_population(ask, book, question,
                                                         predicate):
    """The failure this prevents: balance(London) beside LTV(whole book) — four
    numbers that do not describe the same thing."""
    envelope = ask(question)
    filtered = predicate(book)
    whole = profile(book)
    truth = profile(filtered)
    row = _summary_row(envelope)

    assert row["loan_count"] == truth["loans"]
    assert row[f"{BALANCE}_sum"] == pytest.approx(truth["balance"], rel=1e-9)
    # …and demonstrably NOT the whole-book figure for any measure.
    assert row[f"{BALANCE}_sum"] != pytest.approx(whole["balance"], rel=1e-6)
    assert row[f"{LTV}_weighted_avg"] != pytest.approx(whole["wa_ltv"], rel=1e-6)
    assert row["loan_count"] != whole["loans"]


def test_the_executed_population_is_the_receipt_population(ask, book):
    """Scope identity, stated: the receipt's population is the one every measure
    was computed over."""
    envelope = ask(CFO_04)
    assert "86 loans" in _receipt(envelope)
    assert _summary_row(envelope)["loan_count"] == 86
